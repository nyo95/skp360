#!/usr/bin/env python3
"""RAD AI360 Pipeline Worker.

Entry point for the packaged exe:
    ai360_pipeline.exe --job job.json

Hybrid pipeline (job["mode"] == "hybrid"):
  Stage 0 — Blender headless  : RGB PNG + depth HDR per face (6x)
  Stage 1 — ComfyUI ControlNet: SD1.5 + depth conditioning per face (6x)
  Stage 2 — Cloudflare FLUX   : photorealism pass per face (6x)
  Stage 3 — Stitch            : cubemap → equirectangular ERP

Legacy pipeline (mode == "generate" | "stitch"):
  Unchanged RGB-only FLUX + stitch.
"""
from __future__ import annotations

import argparse, base64, json, os, shutil, subprocess, sys, time, uuid
import urllib.error, urllib.request
from pathlib import Path

import cv2
import numpy as np

FACES = ("front", "right", "back", "left", "top", "bottom")

# When frozen by PyInstaller, __file__ is the .exe; bundled data lives in sys._MEIPASS.
# When run as a plain script, both resolve to the same directory.
_EXE_DIR    = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", str(Path(__file__).parent)))

CONFIG_PATH = _EXE_DIR / "config.json"
PLUGIN_DIR  = _EXE_DIR


def _bundled(filename: str) -> Path:
    """Return path to a file bundled with the exe (or next to the script)."""
    p = _BUNDLE_DIR / filename
    if p.exists():
        return p
    return _EXE_DIR / filename

SD15_MODEL  = "v1-5-pruned-emaonly.safetensors"
CN_MODEL    = "control_v11f1p_sd15_depth.safetensors"
COMFYUI_URL = "http://127.0.0.1:8188"
_COMFYUI_SEARCH = [
    Path("D:/Projects/skpto360/.local/ComfyUI"),
    Path.home() / "ComfyUI",
    Path("C:/ComfyUI"),
]

CF_MODEL    = "@cf/black-forest-labs/flux-2-klein-4b"
GUIDANCE    = 3.5
REF_LIMIT   = 511
DENOISE     = 0.40
CN_STRENGTH = 0.8
SEED        = 42
STEPS       = 20
CFG         = 7.0

FLUX_PROMPT = (
    "Enhance this architectural interior rendering into a high-end photorealistic "
    "interior photograph. Preserve exact room layout, furniture positions, wall "
    "openings, cabinetry and ceiling height precisely as shown. "
    "Upgrade materials: warm wood grain, polished concrete floor with reflections, "
    "linen-textured sofa, painted plaster walls. "
    "Lighting: warm directional daylight with soft indirect fill, physically plausible. "
    "Camera: fixed — do not alter perspective, framing, or field of view. "
    "Style: contemporary residential interior photography, wide-angle lens. "
    "LIGHTING CONSTRAINTS: Do not add ceiling lights, downlights, pendants, "
    "track lights, wall lights, sconces or luminaires not visible in the input."
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Credentials ────────────────────────────────────────────────────────────────

def load_credentials(job: dict | None = None) -> tuple[str, str]:
    config_file = Path(job["config_path"]) if job and job.get("config_path") else _bundled("config.json")
    account_id  = os.environ.get("CF_ACCOUNT_ID")
    api_token   = os.environ.get("CF_API_TOKEN")
    if config_file.is_file():
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8-sig"))
            cf  = cfg.get("cloudflare", {})
            account_id = account_id or cf.get("account_id")
            api_token  = api_token  or cf.get("api_token")
        except Exception as e:
            raise RuntimeError(f"Invalid config: {e}")
    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare credentials missing. "
            "Add account_id/api_token to config.json or set CF_ACCOUNT_ID/CF_API_TOKEN.")
    return account_id, api_token


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Blender render (RGB + depth)
# ══════════════════════════════════════════════════════════════════════════════

def run_blender_render(job: dict) -> dict[str, dict[str, Path]]:
    render_dir = Path(job["output_dir"]) / "blender_render"
    meta_path  = render_dir / "render_meta.json"

    # Cache hit: all 12 files exist
    if meta_path.exists():
        meta  = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        paths = {
            face: {"rgb":   Path(v["rgb"]),
                   "depth": Path(v["depth"])}
            for face, v in meta["faces"].items()
        }
        if all(p.exists() for fv in paths.values() for p in fv.values()):
            log("  [Stage 0] Blender render cache hit — skipping.")
            return paths

    blender_exe   = job["blender_exe"]
    render_script = str(_bundled("blender_render_worker.py"))
    job_path      = Path(job["_job_path"])

    log("  [Stage 0] Blender headless render (RGB + depth)...")
    cmd = [blender_exe, "--background", "--factory-startup",
           "--python", render_script, "--", "--job", str(job_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = result.stdout + result.stderr
    log(out[-4000:] if len(out) > 4000 else out)
    if result.returncode != 0:
        raise RuntimeError(f"Blender render failed (exit {result.returncode})")

    meta  = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    return {
        face: {"rgb":   Path(v["rgb"]),
               "depth": Path(v["depth"])}
        for face, v in meta["faces"].items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — ComfyUI ControlNet
# ══════════════════════════════════════════════════════════════════════════════

def find_comfyui() -> Path | None:
    env_path = os.environ.get("RAD_AI360_COMFYUI")
    if env_path and (Path(env_path) / "main.py").exists():
        return Path(env_path)
    for p in _COMFYUI_SEARCH:
        if (p / "main.py").exists():
            return p
    return None


def find_comfyui_python(cdir: Path) -> str:
    for candidate in [
        cdir.parent / "comfyui_env" / "Scripts" / "python.exe",
        cdir / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]:
        if Path(candidate).exists():
            return str(candidate)
    return sys.executable


def is_comfyui_running() -> bool:
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def start_comfyui(cdir: Path, log_path: Path) -> subprocess.Popen:
    python = find_comfyui_python(cdir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [python, str(cdir / "main.py"), "--port", "8188", "--lowvram", "--cpu-vae"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        cwd=str(cdir),
    )
    for _ in range(60):
        time.sleep(2)
        if is_comfyui_running():
            log("    ComfyUI ready.")
            return proc
    raise RuntimeError("ComfyUI did not start within 120s")


def depth_hdr_to_png_inverted(hdr_path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read HDR: {hdr_path}")
    arr    = img[:, :, 2].astype(np.float32)    # BGR→R channel = depth
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if len(finite) == 0:
        log(f"    Warning: {hdr_path.name} has no geometry — using neutral depth.")
        return np.full((size, size), 128, dtype=np.uint8)
    p5, p95  = np.percentile(finite, 5), np.percentile(finite, 95)
    mapped   = np.clip((arr - p5) / max(p95 - p5, 1e-8), 0.0, 1.0)
    inverted = 255 - (mapped * 255).astype(np.uint8)   # near=white
    return cv2.resize(inverted, (size, size), interpolation=cv2.INTER_LANCZOS4)


def copy_to_comfyui_input(cdir: Path, src: Path, name: str) -> str:
    dst = cdir / "input" / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(src, dst)
    return name


def build_cn_workflow(rgb_name: str, depth_name: str) -> dict:
    return {
        "1":  {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": SD15_MODEL}},
        "2":  {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": CN_MODEL}},
        "3":  {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["1", 1],
                          "text": ("photorealistic interior photograph, sharp, "
                                   "natural light, architectural photography, 8k")}},
        "4":  {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["1", 1],
                          "text": "blurry, dark, oversaturated, cartoon, 3d render, flat"}},
        "5":  {"class_type": "LoadImage",
               "inputs": {"image": rgb_name,   "upload": "image"}},
        "6":  {"class_type": "LoadImage",
               "inputs": {"image": depth_name, "upload": "image"}},
        "7":  {"class_type": "ControlNetApply",
               "inputs": {"conditioning": ["3", 0],
                          "control_net":  ["2", 0],
                          "image":        ["6", 0],
                          "strength":     CN_STRENGTH}},
        "8":  {"class_type": "VAEEncode",
               "inputs": {"vae": ["1", 2], "pixels": ["5", 0]}},
        "9":  {"class_type": "KSampler",
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
               "inputs": {"images": ["10", 0], "filename_prefix": "rad_cn_"}},
    }


def submit_workflow(workflow: dict) -> str:
    body = json.dumps({"prompt": workflow}).encode()
    req  = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]


def poll_comfyui(prompt_id: str, cdir: Path, timeout: int = 300) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        resp = urllib.request.urlopen(
            f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        hist = json.loads(resp.read())
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"ComfyUI error on {prompt_id}")
        for node_out in entry.get("outputs", {}).values():
            for img in node_out.get("images", []):
                p = cdir / "output" / img["filename"]
                if p.exists():
                    return p
    raise TimeoutError(f"ComfyUI timed out after {timeout}s")


def run_controlnet_stage(job: dict, render_paths: dict,
                         cdir: Path) -> dict[str, Path]:
    log("  [Stage 1] ComfyUI ControlNet per face...")
    out_dir   = Path(job["output_dir"]) / "stage1_controlnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    face_size = int(job.get("face_size", 512))
    stage1    = {}

    for face in FACES:
        result_path = out_dir / f"cubemap_{face}.png"
        if result_path.exists():
            log(f"    {face}: cache hit")
            stage1[face] = result_path
            continue

        rgb_src   = render_paths[face]["rgb"]
        depth_src = render_paths[face]["depth"]

        # Resize RGB to face_size
        rgb_img = cv2.imread(str(rgb_src))
        if rgb_img is None:
            raise FileNotFoundError(f"Blender RGB missing: {rgb_src}")
        rgb_img = cv2.resize(rgb_img, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4)
        rgb_tmp = out_dir / f"_tmp_rgb_{face}.png"
        cv2.imwrite(str(rgb_tmp), rgb_img)

        # Depth: HDR → inverted PNG
        depth_img = depth_hdr_to_png_inverted(depth_src, face_size)
        depth_tmp = out_dir / f"_tmp_depth_{face}.png"
        cv2.imwrite(str(depth_tmp), depth_img)

        rgb_name   = copy_to_comfyui_input(cdir, rgb_tmp,   f"rad_rgb_{face}.png")
        depth_name = copy_to_comfyui_input(cdir, depth_tmp, f"rad_depth_{face}.png")

        wf  = build_cn_workflow(rgb_name, depth_name)
        pid = submit_workflow(wf)
        log(f"    {face}: submitted {pid}")
        rendered = poll_comfyui(pid, cdir)
        shutil.copy2(rendered, result_path)
        log(f"    {face}: done → {result_path.name}")
        stage1[face] = result_path

    return stage1


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FLUX
# ══════════════════════════════════════════════════════════════════════════════

def cf_multipart(account_id, token, model, fields, files) -> dict:
    boundary = "----rad-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks += [f"--{boundary}\r\n".encode(),
                   f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                   value.encode(), b"\r\n"]
    for name, (filename, content, mime) in files.items():
        chunks += [f"--{boundary}\r\n".encode(),
                   f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                   f"Content-Type: {mime}\r\n\r\n".encode(),
                   content, b"\r\n"]
    chunks.append(f"--{boundary}--\r\n".encode())
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    req = urllib.request.Request(url, data=b"".join(chunks), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type",  f"multipart/form-data; boundary={boundary}")
    req.add_header("accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cloudflare HTTP {e.code}: {e.read().decode(errors='replace')}")


def resize_to_limit(img: np.ndarray, limit: int) -> np.ndarray:
    h, w  = img.shape[:2]
    scale = min(1.0, limit / max(w, h))
    return img if scale == 1.0 else cv2.resize(
        img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def run_flux_stage(job: dict, stage1_paths: dict[str, Path],
                   account_id: str, api_token: str) -> dict[str, Path]:
    log("  [Stage 2] FLUX per face...")
    out_dir  = Path(job["output_dir"]) / "stage2_flux"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt   = FLUX_PROMPT + " " + job.get("style_prompt", "")
    stage2   = {}

    for face in FACES:
        result_path = out_dir / f"cubemap_{face}.png"
        if result_path.exists():
            log(f"    {face}: cache hit")
            stage2[face] = result_path
            continue

        img = cv2.imread(str(stage1_paths[face]))
        if img is None:
            raise FileNotFoundError(f"Stage-1 missing: {stage1_paths[face]}")
        h, w = img.shape[:2]
        ref  = resize_to_limit(img, REF_LIMIT)
        ok, enc = cv2.imencode(".png", ref)
        if not ok:
            raise RuntimeError(f"Cannot encode {face}")

        log(f"    {face}: FLUX {w}×{h}...")
        t0     = time.time()
        result = cf_multipart(
            account_id, api_token,
            job.get("model", CF_MODEL),
            {"prompt": prompt, "width": str(w), "height": str(h),
             "guidance": str(GUIDANCE)},
            {"input_image_0": ("ref.png", enc.tobytes(), "image/png")},
        )
        elapsed = time.time() - t0
        img_b64 = result.get("result", {}).get("image")
        if not img_b64:
            raise RuntimeError(f"No image in FLUX response for {face}: {list(result.keys())}")
        result_path.write_bytes(base64.b64decode(img_b64))
        log(f"    {face}: done ({elapsed:.1f}s)")
        stage2[face] = result_path

    return stage2


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Stitch cubemap → ERP
# ══════════════════════════════════════════════════════════════════════════════

def stitch(cube: dict[str, np.ndarray], width: int) -> np.ndarray:
    """Cubemap → equirectangular.

    World space convention (matches OBJ import Y-forward Z-up):
      front  = +X   right = -Y   back = -X   left = +Y
      top    = +Z   bottom= -Z

    ERP: u=0 → lon=-π (right edge wraps), v=0 → lat=+π/2 (top/+Z pole)
    """
    height = width // 2
    size   = next(iter(cube.values())).shape[0]

    u_px, v_px = np.meshgrid(np.arange(width,  dtype=np.float32),
                              np.arange(height, dtype=np.float32))
    lon =  (u_px / width  - 0.5) * 2 * np.pi   # [-π, +π],  0 = front (+X)
    lat =  (0.5 - v_px / height) * np.pi        # [+π/2, -π/2], +π/2 = top

    cos_lat = np.cos(lat)
    dx =  cos_lat * np.cos(lon)   # +X (front/back dominant)
    dy =  cos_lat * np.sin(lon)   # +Y (left dominant) / -Y (right dominant)
    dz =  np.sin(lat)             # +Z (top) / -Z (bottom)

    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    output = np.zeros((height, width, 3), dtype=np.uint8)

    def remap(mask, u_raw, v_raw, face):
        u = np.clip((u_raw + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.float32)
        v = np.clip((1 - v_raw) * 0.5 * (size - 1), 0, size - 1).astype(np.float32)
        s = cv2.remap(cube[face], u, v, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        output[mask] = s[mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        m = (dx > 0) & (ax >= ay) & (ax >= az)
        remap(m,  dy / np.where(m, dx, 1),  dz / np.where(m, dx, 1),  "front")
        m = (dy < 0) & (ay >= ax) & (ay >= az)
        remap(m,  dx / np.where(m, ay, 1),  dz / np.where(m, ay, 1),  "right")
        m = (dx < 0) & (ax >= ay) & (ax >= az)
        remap(m, -dy / np.where(m, ax, 1),  dz / np.where(m, ax, 1),  "back")
        m = (dy > 0) & (ay >= ax) & (ay >= az)
        remap(m, -dx / np.where(m, ay, 1),  dz / np.where(m, ay, 1),  "left")
        m = (dz > 0) & (az >= ax) & (az >= ay)
        remap(m,  dy / np.where(m, dz, 1), -dx / np.where(m, dz, 1),  "top")
        m = (dz < 0) & (az >= ax) & (az >= ay)
        remap(m,  dy / np.where(m, az, 1),  dx / np.where(m, az, 1),  "bottom")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid(job: dict) -> None:
    out_dir   = Path(job["output_dir"])
    erp_width = int(job.get("erp_width", 4096))

    # Stage 0
    render_paths = run_blender_render(job)

    # Stage 1
    cdir = find_comfyui()
    if not cdir:
        raise RuntimeError(
            "ComfyUI not found. Set RAD_AI360_COMFYUI env var or install to "
            "D:/Projects/skpto360/.local/ComfyUI")
    comfyui_proc = None
    if not is_comfyui_running():
        log("  Starting ComfyUI...")
        comfyui_proc = start_comfyui(cdir, out_dir / "comfyui.log")
    stage1_paths = run_controlnet_stage(job, render_paths, cdir)

    # Stage 2
    account_id, api_token = load_credentials(job)
    stage2_paths = run_flux_stage(job, stage1_paths, account_id, api_token)

    if comfyui_proc:
        comfyui_proc.terminate()

    # Stage 3 — stitch
    log("  [Stage 3] Stitching ERP...")
    cube = {}
    for face in FACES:
        img = cv2.imread(str(stage2_paths[face]))
        if img is None:
            raise FileNotFoundError(f"Stage-2 missing: {stage2_paths[face]}")
        s = min(img.shape[:2])
        cube[face] = img[:s, :s]

    panorama  = stitch(cube, erp_width)
    pano_path = out_dir / "equirectangular_2to1.png"
    if not cv2.imwrite(str(pano_path), panorama):
        raise RuntimeError(f"Cannot write ERP: {pano_path}")
    log(f"  ERP → {pano_path}")

    report = {
        "status":   "complete",
        "mode":     "hybrid",
        "pipeline": "SketchUp OBJ → Blender RGB+depth → ControlNet → FLUX → ERP",
        "panorama": str(pano_path),
        "erp_size": [erp_width, erp_width // 2],
        "stage1":   {k: str(v) for k, v in stage1_paths.items()},
        "stage2":   {k: str(v) for k, v in stage2_paths.items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"  Complete: {pano_path}")


def run_legacy(job: dict) -> None:
    from pathlib import Path as P
    source = P(job["source_dir"])
    output = P(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    working = source
    if job.get("mode") == "generate":
        account_id, api_token = load_credentials(job)
        working = output / "generated_faces"
        working.mkdir(exist_ok=True)
        prompt = ("Transform this cubemap face into a photorealistic interior photo. "
                  "Preserve camera projection, openings and furniture. "
                  + job.get("style_prompt", ""))
        for face in FACES:
            dst = working / f"cubemap_{face}.png"
            if dst.exists():
                continue
            img = cv2.imread(str(source / f"cubemap_{face}.png"))
            if img is None:
                raise FileNotFoundError(f"Missing: {source / f'cubemap_{face}.png'}")
            ref  = resize_to_limit(img, REF_LIMIT)
            ok, enc = cv2.imencode(".png", ref)
            result = cf_multipart(
                account_id, api_token, job.get("model", CF_MODEL),
                {"prompt": prompt,
                 "width": str(img.shape[1]), "height": str(img.shape[0]),
                 "guidance": str(GUIDANCE)},
                {"input_image_0": ("ref.png", enc.tobytes(), "image/png")},
            )
            img_b64 = result.get("result", {}).get("image")
            if not img_b64:
                raise RuntimeError(f"No image for {face}")
            dst.write_bytes(base64.b64decode(img_b64))

    width = int(job.get("erp_width", 4096))
    cube  = {}
    for face in FACES:
        img = cv2.imread(str(working / f"cubemap_{face}.png"))
        if img is None:
            raise FileNotFoundError(f"Missing face: {face}")
        s = min(img.shape[:2])
        cube[face] = img[:s, :s]
    panorama  = stitch(cube, width)
    pano_path = output / "equirectangular_2to1.png"
    cv2.imwrite(str(pano_path), panorama)
    (output / "report.json").write_text(json.dumps({
        "status": "complete", "mode": job.get("mode"),
        "panorama": str(pano_path),
    }, indent=2))
    log(f"Complete: {pano_path}")


def run(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    job["_job_path"] = str(job_path)
    Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    if job.get("mode") == "hybrid":
        run_hybrid(job)
    else:
        run_legacy(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAD AI360 Pipeline")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.job)
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
