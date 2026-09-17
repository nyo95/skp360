#!/usr/bin/env python3
"""Phase 0K Step 10 — Full 360 Panorama Feasibility Test.

For each of 6 cubemap faces (from Phase 0C):
  1. Load Phase 0C RGB (1024x1024) + depth EXR
  2. Apply E_PERCENTILE_CLAMPED depth mapping + RC1 inversion + resize to 512x512
  3. Run ComfyUI ControlNet SD1.5 (Stage 1) — same params as Phase 0I rerun
  4. Run FLUX (Stage 2) with lighting-constrained prompt
  5. Stitch all 6 Stage 2 outputs into 2048x1024 ERP using cubemap-to-ERP projection

Outputs:
  output/phase0k/panorama/faces/{face}/{stage1,stage2}.png
  output/phase0k/panorama/final_erp.png
  output/phase0k/panorama/qc_report.json
  output/phase0k/phase0k_panorama_report.json

STOP CONDITION: Script stops after writing final_erp.png and qc_report.json.
Do NOT automatically start multi-face generation again.
"""
from __future__ import annotations

import json, time, uuid, base64, http.client, shutil, io
import struct, zlib, math
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

ROOT     = Path(__file__).resolve().parents[1]
PHASE0K  = ROOT / "output" / "phase0k" / "panorama"
PHASE0C  = ROOT / "output" / "phase0c"
LIGHTS   = ROOT / "output" / "phase0k" / "lights" / "lights.json"

CONFIG_JSON = (
    Path.home() / "AppData" / "Roaming" / "SketchUp" / "SketchUp 2021" /
    "SketchUp" / "Plugins" / "rad_ai360_visualizer" / "config.json"
)

COMFYUI_DIR = ROOT / ".local" / "ComfyUI"
COMFYUI_ENV = ROOT / ".local" / "comfyui_env"
COMFYUI_URL = "http://127.0.0.1:8188"
SD15_MODEL  = "v1-5-pruned-emaonly.safetensors"
CN_MODEL    = "control_v11f1p_sd15_depth.safetensors"
HF_CACHE    = Path("D:/hf_cache")

DENOISE  = 0.40
CN_STR   = 0.8
SEED     = 42
STEPS    = 20
CFG      = 7.0
FACE_PX  = 512
ERP_W, ERP_H = 2048, 1024

MODEL_CF = "@cf/black-forest-labs/flux-2-klein-4b"
GUIDANCE = 3.5
REF_LIMIT = 511

FACE_NAMES = ["front", "right", "back", "left", "top", "bottom"]

# Face vectors from phase0c_report.json
FACE_VECTORS = {
    "front":  {"forward": np.array([1,0,0]),  "up": np.array([0,0,1]),   "right": np.array([0,-1,0])},
    "right":  {"forward": np.array([0,-1,0]), "up": np.array([0,0,1]),   "right": np.array([-1,0,0])},
    "back":   {"forward": np.array([-1,0,0]), "up": np.array([0,0,1]),   "right": np.array([0,1,0])},
    "left":   {"forward": np.array([0,1,0]),  "up": np.array([0,0,1]),   "right": np.array([1,0,0])},
    "top":    {"forward": np.array([0,0,1]),  "up": np.array([-1,0,0]),  "right": np.array([0,-1,0])},
    "bottom": {"forward": np.array([0,0,-1]), "up": np.array([1,0,0]),   "right": np.array([0,-1,0])},
}

# Camera basis for ERP projection (from scene.json)
CAM_FWD   = np.array([1.0, 0.0, 0.0])
CAM_UP    = np.array([0.0, 0.0, 1.0])
CAM_RIGHT = np.array([0.0, -1.0, 0.0])

FLUX_PROMPT_BASE = (
    "Enhance this existing architectural interior rendering into a "
    "high-end photorealistic interior photograph. "
    "Preserve the exact room layout, all furniture positions, all wall openings, "
    "cabinetry, and ceiling height precisely as shown. "
    "Upgrade materials: warm walnut wood grain on TV unit and shelving, "
    "polished concrete floor with subtle reflections, "
    "linen-textured sofa, painted plaster walls. "
    "Lighting: warm directional daylight from the left with soft indirect fill; "
    "physically plausible illumination and soft shadows. "
    "Camera: fixed — do not alter the perspective, framing, or field of view. "
    "Style: contemporary residential interior photography, wide-angle architectural lens."
)


# ── Credentials ────────────────────────────────────────────────────────────────

def load_credentials():
    data = json.loads(CONFIG_JSON.read_text())
    cf = data["cloudflare"]
    return cf["account_id"], cf["api_token"]


# ── Depth helpers ──────────────────────────────────────────────────────────────

def load_exr_depth(exr_path: Path) -> np.ndarray:
    import OpenEXR, Imath
    exr = OpenEXR.InputFile(str(exr_path))
    dw  = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    raw = exr.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(h, w)


def depth_to_png_inverted(depth_arr: np.ndarray, size: int) -> np.ndarray:
    p5  = np.percentile(depth_arr[np.isfinite(depth_arr)], 5)
    p95 = np.percentile(depth_arr[np.isfinite(depth_arr)], 95)
    mapped = np.clip((depth_arr - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)
    mapped = (mapped * 255).astype(np.uint8)
    inverted = 255 - mapped                          # RC1: near=white
    img = Image.fromarray(inverted, mode="L")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


# ── ComfyUI helpers ────────────────────────────────────────────────────────────

def is_comfyui_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def start_comfyui():
    import subprocess
    python = str(COMFYUI_ENV / "Scripts" / "python.exe")
    main   = str(COMFYUI_DIR / "main.py")
    log    = str(PHASE0K / "comfyui.log")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [python, main, "--port", "8188", "--lowvram", "--cpu-vae"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        cwd=str(COMFYUI_DIR),
    )
    for _ in range(60):
        time.sleep(2)
        if is_comfyui_running():
            print("    ComfyUI ready.")
            return proc
    raise RuntimeError("ComfyUI did not start within 120s")


def ensure_comfyui():
    if is_comfyui_running():
        print("    ComfyUI already running.")
        return None
    print("    Starting ComfyUI...")
    return start_comfyui()


def copy_to_comfyui_input(src: Path, name: str) -> str:
    dst_dir = COMFYUI_DIR / "input"
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / name
    shutil.copy2(src, dst)
    return name


def build_workflow(rgb_name: str, depth_name: str) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": SD15_MODEL}},
        "2": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CN_MODEL}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1],
                         "text": ("photorealistic interior photo, sharp, natural light, "
                                  "architectural photography, 8k")}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1],
                         "text": "blurry, dark, oversaturated, cartoon, 3d render, flat"}},
        "5": {"class_type": "LoadImage",
              "inputs": {"image": rgb_name, "upload": "image"}},
        "6": {"class_type": "LoadImage",
              "inputs": {"image": depth_name, "upload": "image"}},
        "7": {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["3", 0],
                         "control_net": ["2", 0],
                         "image": ["6", 0],
                         "strength": CN_STR}},
        "8": {"class_type": "VAEEncode",
              "inputs": {"vae": ["1", 2], "pixels": ["5", 0]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0],
                         "positive": ["7", 0], "negative": ["4", 0],
                         "latent_image": ["8", 0],
                         "seed": SEED, "steps": STEPS, "cfg": CFG,
                         "sampler_name": "euler_ancestral",
                         "scheduler": "karras",
                         "denoise": DENOISE}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"vae": ["1", 2], "samples": ["9", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "phase0k_"}},
    }


def submit_workflow(workflow: dict) -> str:
    import urllib.request
    body = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())["prompt_id"]


def wait_for_output(prompt_id: str, timeout: int = 300) -> Path:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        hist = json.loads(resp.read())
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"ComfyUI error for {prompt_id}")
        outputs = entry.get("outputs", {})
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                img_path = COMFYUI_DIR / "output" / img["filename"]
                if img_path.exists():
                    return img_path
    raise TimeoutError(f"ComfyUI did not complete {prompt_id} in {timeout}s")


# ── FLUX helper ────────────────────────────────────────────────────────────────

def load_lighting_prompt() -> str:
    if LIGHTS.exists():
        constraint = json.loads(LIGHTS.read_text()).get("prompt_constraint", "")
        return FLUX_PROMPT_BASE + ("\n\n" + constraint if constraint else "")
    return FLUX_PROMPT_BASE


def call_flux(account_id, api_token, ref_bytes, out_path, width, height, prompt):
    boundary = "----phase0k-" + uuid.uuid4().hex
    crlf = b"\r\n"

    def part(name, value):
        return (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode()

    body  = part("prompt", prompt)
    body += part("width",  str(width))
    body += part("height", str(height))
    body += part("guidance", str(GUIDANCE))
    body += (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="input_image_0"; filename="ref.png"\r\n'
             f"Content-Type: image/png\r\n\r\n").encode() + ref_bytes + crlf
    body += f"--{boundary}--\r\n".encode()

    url = f"/client/v4/accounts/{account_id}/ai/run/{MODEL_CF}"
    conn = http.client.HTTPSConnection("api.cloudflare.com", timeout=300)
    t0 = time.time()
    conn.request("POST", url, body=body, headers={
        "Authorization": f"Bearer {api_token}",
        "Content-Type":  f"multipart/form-data; boundary={boundary}",
    })
    resp = conn.getresponse()
    elapsed = time.time() - t0
    raw = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"FLUX HTTP {resp.status}: {raw[:200]}")
    payload = json.loads(raw)
    img_b64 = (payload.get("result", {}).get("image") or
               payload.get("images", [None])[0])
    if not img_b64:
        raise RuntimeError(f"No image in FLUX response: {list(payload.keys())}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(img_b64))
    return elapsed


def img_to_ref_bytes(img_path: Path) -> bytes:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > REF_LIMIT:
        s = REF_LIMIT / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── ERP stitcher ───────────────────────────────────────────────────────────────

def build_erp(face_images: dict[str, np.ndarray], w: int, h: int) -> np.ndarray:
    """Stitch 6 face images (512x512 RGB float32 [0,1]) into a W×H ERP."""
    erp = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    phi = (xs / w - 0.5) * 2 * np.pi       # longitude [-π, π]
    theta = (0.5 - ys / h) * np.pi         # latitude  [-π/2, π/2]

    # Ray directions in world space using camera basis
    cos_t = np.cos(theta)
    dx = cos_t * np.cos(phi)                        # along CAM_FWD
    dy = cos_t * np.sin(phi)                        # along CAM_RIGHT
    dz = np.sin(theta)                              # along CAM_UP

    # World direction = dx*[1,0,0] + dy*[0,-1,0] + dz*[0,0,1]
    d_world = np.stack([dx, -dy, dz], axis=-1)      # (H, W, 3)

    for face_name, face_img in face_images.items():
        vecs = FACE_VECTORS[face_name]
        F = vecs["forward"]
        R = vecs["right"]
        U = vecs["up"]

        dot_f = np.dot(d_world, F)                  # (H, W)
        mask  = dot_f > 1e-4
        dot_r = np.dot(d_world, R)
        dot_u = np.dot(d_world, U)

        # Face UV
        s = np.where(mask, dot_r / np.where(mask, dot_f, 1.0), 0.0)
        t = np.where(mask, dot_u / np.where(mask, dot_f, 1.0), 0.0)
        face_u = np.clip((s + 1) / 2, 0.0, 1.0)
        face_v = np.clip(1 - (t + 1) / 2, 0.0, 1.0)

        fh, fw = face_img.shape[:2]
        px = np.clip((face_u * fw).astype(np.int32), 0, fw - 1)
        py = np.clip((face_v * fh).astype(np.int32), 0, fh - 1)

        # Confidence weight — penalize pixels near face edges
        conf = np.minimum(
            np.minimum(face_u, 1 - face_u),
            np.minimum(face_v, 1 - face_v)
        )
        w_px = mask.astype(np.float32) * np.maximum(conf, 0.0)

        sampled = face_img[py, px]              # (H, W, 3)
        erp    += sampled * w_px[..., None]
        weight += w_px

    valid = weight > 0
    erp[valid] /= weight[valid, None]
    erp[~valid] = 0.0
    return (np.clip(erp, 0.0, 1.0) * 255).astype(np.uint8)


# ── QC ─────────────────────────────────────────────────────────────────────────

def qc_erp(erp_arr: np.ndarray, face_stage2: dict) -> dict:
    h, w = erp_arr.shape[:2]
    gray = cv2.cvtColor(erp_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Seam check: max column-to-column delta at x=0 boundary
    left_col  = gray[:, 0].astype(float)
    right_col = gray[:, -1].astype(float)
    seam_delta = float(np.mean(np.abs(left_col - right_col)))

    # Coverage: nonzero pixels
    coverage = float(np.mean(erp_arr.any(axis=-1)))

    # Pole uniformity: top/bottom 5% rows should be uniform
    pole_rows = int(h * 0.05)
    top_std    = float(gray[:pole_rows].std())
    bottom_std = float(gray[-pole_rows:].std())

    # Cross-face continuity: measure variance at face boundaries
    # Faces tile at x = w//4 boundaries (roughly) for front/right/back/left
    boundary_xs = [w // 4, w // 2, 3 * w // 4]
    boundary_deltas = []
    for bx in boundary_xs:
        col_l = gray[:, bx - 1].astype(float)
        col_r = gray[:, bx + 1].astype(float)
        boundary_deltas.append(float(np.mean(np.abs(col_l - col_r))))
    cross_face_mean_delta = float(np.mean(boundary_deltas))

    return {
        "coverage_fraction": round(coverage, 4),
        "seam_left_right_delta": round(seam_delta, 2),
        "pole_top_std": round(top_std, 2),
        "pole_bottom_std": round(bottom_std, 2),
        "cross_face_boundary_delta_mean": round(cross_face_mean_delta, 2),
        "seam_pass": seam_delta < 20.0,
        "coverage_pass": coverage > 0.99,
        "pole_pass": top_std < 30.0 and bottom_std < 30.0,
        "cross_face_pass": cross_face_mean_delta < 25.0,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 0K Step 10 — Full 360 Panorama Feasibility ===")
    PHASE0K.mkdir(parents=True, exist_ok=True)

    account_id, api_token = load_credentials()
    flux_prompt = load_lighting_prompt()
    print(f"  Flux prompt: {len(flux_prompt)} chars")

    # Ensure ComfyUI running
    proc = ensure_comfyui()

    face_stage2_imgs = {}
    face_log = {}

    for face in FACE_NAMES:
        print(f"\n  --- Face: {face} ---")
        face_dir = PHASE0K / "faces" / face
        face_dir.mkdir(parents=True, exist_ok=True)

        # Load Phase 0C RGB
        rgb_src = PHASE0C / "rgb" / f"{face}.png"
        if not rgb_src.exists():
            raise FileNotFoundError(f"Phase 0C RGB missing: {rgb_src}")
        rgb_img = Image.open(rgb_src).convert("RGB").resize((FACE_PX, FACE_PX), Image.LANCZOS)
        rgb_face = face_dir / "rgb_input.png"
        rgb_img.save(rgb_face)

        # Load Phase 0C depth EXR, convert + invert
        depth_src = PHASE0C / "depth" / f"{face}.exr"
        if not depth_src.exists():
            raise FileNotFoundError(f"Phase 0C depth EXR missing: {depth_src}")
        depth_arr = load_exr_depth(depth_src)
        depth_inv = depth_to_png_inverted(depth_arr, FACE_PX)
        depth_face = face_dir / "depth_inverted.png"
        Image.fromarray(depth_inv, mode="L").save(depth_face)

        # Stage 1 — ComfyUI ControlNet
        stage1_out = face_dir / "stage1.png"
        if stage1_out.exists():
            print(f"    Stage 1 cache hit: {stage1_out}")
        else:
            rgb_input_name   = copy_to_comfyui_input(rgb_face,   f"phase0k_{face}_rgb.png")
            depth_input_name = copy_to_comfyui_input(depth_face, f"phase0k_{face}_depth.png")
            wf = build_workflow(rgb_input_name, depth_input_name)
            t0 = time.time()
            pid = submit_workflow(wf)
            print(f"    Stage 1 submitted: {pid}")
            result = wait_for_output(pid)
            shutil.copy2(result, stage1_out)
            elapsed = round(time.time() - t0, 1)
            print(f"    Stage 1 done: {elapsed}s → {stage1_out}")
            face_log.setdefault(face, {})["stage1_s"] = elapsed

        # Stage 2 — FLUX
        stage2_out = face_dir / "stage2.png"
        if stage2_out.exists():
            print(f"    Stage 2 cache hit: {stage2_out}")
        else:
            ref_bytes = img_to_ref_bytes(stage1_out)
            t0 = time.time()
            elapsed = call_flux(account_id, api_token, ref_bytes,
                                stage2_out, FACE_PX, FACE_PX, flux_prompt)
            print(f"    Stage 2 done: {elapsed:.1f}s → {stage2_out}")
            face_log.setdefault(face, {})["stage2_s"] = round(elapsed, 1)

        # Load for stitcher
        arr = np.array(Image.open(stage2_out).convert("RGB")).astype(np.float32) / 255.0
        face_stage2_imgs[face] = arr
        print(f"    Loaded for stitch: {face} ({arr.shape})")

    # Stop ComfyUI if we started it
    if proc:
        proc.terminate()
        print("\n  ComfyUI stopped.")

    # ERP stitch
    print("\n  Stitching ERP...")
    erp_arr = build_erp(face_stage2_imgs, ERP_W, ERP_H)
    erp_path = PHASE0K / "final_erp.png"
    Image.fromarray(erp_arr).save(erp_path)
    print(f"  ERP saved: {erp_path} ({ERP_W}x{ERP_H})")

    # QC
    qc = qc_erp(erp_arr, face_stage2_imgs)
    qc_path = PHASE0K / "qc_report.json"
    qc_path.write_text(json.dumps(qc, indent=2))
    print(f"  QC: {qc}")

    # Report
    report = {
        "phase": "0K",
        "step": 10,
        "title": "Full 360 Panorama Feasibility Test",
        "date": "2026-09-17",
        "inputs": {"phase0c_rgb": str(PHASE0C / "rgb"),
                   "phase0c_depth": str(PHASE0C / "depth")},
        "stage1": {"model": SD15_MODEL, "controlnet": CN_MODEL,
                   "denoise": DENOISE, "cn_strength": CN_STR,
                   "steps": STEPS, "cfg": CFG, "seed": SEED,
                   "resolution": f"{FACE_PX}x{FACE_PX}"},
        "stage2": {"model": MODEL_CF, "guidance": GUIDANCE,
                   "ref_limit_px": REF_LIMIT},
        "faces_processed": list(face_stage2_imgs.keys()),
        "face_timings": face_log,
        "erp": {"path": str(erp_path.relative_to(ROOT)),
                "resolution": f"{ERP_W}x{ERP_H}"},
        "qc": qc,
        "stop_condition": (
            "STOP. Phase 0K panorama feasibility complete. "
            "Do NOT automatically start multi-face generation again. "
            "Do not call this 'lighting solved'."
        ),
    }
    report_path = ROOT / "output" / "phase0k" / "phase0k_panorama_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report: {report_path}")
    print("\n  *** STOP — Phase 0K complete. Review final_erp.png before any further work. ***")


if __name__ == "__main__":
    main()
