#!/usr/bin/env python3
"""Phase 0J — Two-Stage Controlled → FLUX Bake-off.

Three outputs:
  A  RGB-only  FLUX.2 Klein on front-face RGB
  B  ControlNet-only  SD1.5 + ControlNet Depth (Stage 1 only, Phase 0I re-run)
  C  Two-stage  Stage 1 (B) → FLUX.2 Klein

Inputs:
  output/phase0i/input/rgb.png         — 512×512 front face RGB
  output/phase0i/rerun/input/depth_inverted.png
  output/phase0i/rerun/B_rgb_depth/output.png  — Stage 1 result (reused)

Stop condition: A/B/C comparison only. Do not proceed to Phase 0K automatically.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT        = Path(__file__).resolve().parents[1]
PHASE0I_DIR = ROOT / "output" / "phase0i"
OUT         = ROOT / "output" / "phase0j"
SCRIPTS_DIR = Path(__file__).parent

# ── Cloudflare credentials ────────────────────────────────────────────────────
PLUGIN_CONFIG = Path(
    r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins"
) / "rad_ai360_visualizer" / "config.json"

MODEL    = "@cf/black-forest-labs/flux-2-klein-4b"
GUIDANCE = 3.5
TIMEOUT  = 300
REF_LIMIT = 511      # max reference edge; FLUX.2 Klein constraint

# Stage 1 parameters (verbatim from Phase 0I re-run)
STAGE1 = {
    "model": "v1-5-pruned-emaonly.safetensors",
    "controlnet_model": "control_v11f1p_sd15_depth.safetensors",
    "seed": 42,
    "steps": 20,
    "cfg": 7.0,
    "sampler": "euler_ancestral",
    "scheduler": "karras",
    "denoise": 0.40,
    "controlnet_strength": 0.8,
    "controlnet_start": 0.0,
    "controlnet_end": 1.0,
    "depth_convention": "inverted: pixel=255-original (near=white, MiDaS-correct)",
    "resolution": "512x512",
    "comfyui_flags": "--lowvram --cpu-vae",
}

FLUX_PROMPT_POSITIVE = (
    "Enhance this existing architectural interior rendering into a "
    "high-end photorealistic interior photograph. "
    "Preserve the exact architecture, room proportions, camera position, "
    "camera composition, wall geometry, openings, ceiling geometry, "
    "columns, cabinetry, furniture count, furniture placement and all "
    "major object silhouettes from the input image. "
    "Do not redesign, move, remove, resize or add architectural elements, "
    "furniture, doors, openings, cabinets, shelves or fixtures. "
    "Treat the input image as the approved design and geometry. "
    "Improve only: realistic material response, fabric microtexture, "
    "wood grain realism, stone surface realism, subtle reflections, "
    "physically believable daylight, soft indirect illumination, "
    "realistic contact shadows, exposure, photographic tonal range, "
    "fine surface detail, natural architectural photography appearance. "
    "Maintain the original material colors and design intent. "
    "Do not change the camera or perspective. "
    "The result must look like the same exact interior photographed after "
    "construction, not a redesigned interpretation."
)

FLUX_PROMPT_NEGATIVE = (
    "redesign, changed architecture, moved furniture, added furniture, "
    "missing furniture, altered wall openings, changed doorway, changed ceiling, "
    "different cabinetry, different layout, different camera, different perspective, "
    "distorted geometry, extra objects, decorative additions, architectural hallucination"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_cf_creds() -> tuple[str, str]:
    account_id = os.environ.get("CF_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token  = os.environ.get("CF_API_TOKEN")  or os.environ.get("CLOUDFLARE_API_TOKEN")
    if PLUGIN_CONFIG.is_file():
        cfg = json.loads(PLUGIN_CONFIG.read_text(encoding="utf-8"))
        cf  = cfg.get("cloudflare", {})
        account_id = account_id or cf.get("account_id")
        api_token  = api_token  or cf.get("api_token")
    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare credentials missing. Set CF_ACCOUNT_ID / CF_API_TOKEN "
            "or ensure rad_ai360_visualizer/config.json is present."
        )
    return account_id, api_token


def resize_reference(src: Path, dst: Path) -> tuple[int, int]:
    """Resize src so its longest edge ≤ REF_LIMIT. Returns (ref_w, ref_h)."""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = min(1.0, REF_LIMIT / max(w, h))
    rw, rh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)
    return rw, rh


def call_flux(
    account_id: str,
    api_token: str,
    ref_path: Path,
    out_path: Path,
    width: int,
    height: int,
    prompt: str,
) -> tuple[dict[str, Any], float]:
    boundary = "----phase0j-" + uuid.uuid4().hex
    file_bytes = ref_path.read_bytes()
    fields = {
        "prompt":   prompt,
        "width":    str(width),
        "height":   str(height),
        "guidance": str(GUIDANCE),
    }
    chunks: list[bytes] = []
    for name, val in fields.items():
        chunks += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            val.encode(),
            b"\r\n",
        ]
    chunks += [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="input_image_0"; filename="{ref_path.name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(chunks)
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Authorization", "Bearer " + api_token)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("accept", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
            cf_ray = dict(resp.headers).get("cf-ray")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"CF HTTP {e.code}: {e.read().decode(errors='replace')}") from e
    runtime = time.time() - t0
    payload = json.loads(raw.decode())
    result = payload.get("result", payload)
    encoded = result.get("image") if isinstance(result, dict) else None
    if isinstance(encoded, str) and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    img_bytes = base64.b64decode(encoded, validate=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img_bytes)
    return {
        "cf_ray": cf_ray,
        "success": payload.get("success"),
        "errors": payload.get("errors"),
    }, runtime


# ── Input files ───────────────────────────────────────────────────────────────
rgb_in      = PHASE0I_DIR / "input" / "rgb.png"
depth_inv   = PHASE0I_DIR / "rerun" / "input" / "depth_inverted.png"
stage1_out  = PHASE0I_DIR / "rerun" / "B_rgb_depth" / "output.png"

for p, label in [(rgb_in, "rgb_in"), (depth_inv, "depth_inv"), (stage1_out, "stage1_out")]:
    if not p.exists():
        raise FileNotFoundError(f"Required input not found ({label}): {p}")

print("=== Phase 0J — Two-Stage Controlled → FLUX Bake-off ===")
print(f"  Input RGB:     {rgb_in}")
print(f"  Depth (inv):   {depth_inv}")
print(f"  Stage 1 (B):   {stage1_out}")

account_id, api_token = load_cf_creds()
print(f"  Cloudflare credentials: OK")

# Source image dimensions (512×512)
src_img = cv2.imread(str(rgb_in), cv2.IMREAD_COLOR)
src_h, src_w = src_img.shape[:2]
OUT_W, OUT_H = src_w, src_h   # 512×512
print(f"  Output resolution: {OUT_W}×{OUT_H}")

results: dict[str, Any] = {}

# ── TEST A — FLUX on front-face RGB ──────────────────────────────────────────
print("\n=== Test A: FLUX on front-face RGB ===")
a_dir = OUT / "A_flux_rgb_only"
a_dir.mkdir(parents=True, exist_ok=True)
a_ref = a_dir / "input_ref.png"
resize_reference(rgb_in, a_ref)
a_out = a_dir / "output.png"

t0 = time.time()
try:
    cf_meta, runtime_a = call_flux(account_id, api_token, a_ref, a_out, OUT_W, OUT_H, FLUX_PROMPT_POSITIVE)
    gen_a = {
        "test": "A_FLUX_RGB_ONLY",
        "stage": 1,
        "provider": "cloudflare",
        "model": MODEL,
        "input_image": str(rgb_in),
        "input_sha256": sha256(rgb_in),
        "reference_path": str(a_ref),
        "reference_sha256": sha256(a_ref),
        "output": str(a_out),
        "output_sha256": sha256(a_out),
        "prompt": FLUX_PROMPT_POSITIVE,
        "negative_prompt_note": FLUX_PROMPT_NEGATIVE,
        "guidance": GUIDANCE,
        "resolution": f"{OUT_W}x{OUT_H}",
        "runtime_s": round(runtime_a, 2),
        "cf_ray": cf_meta.get("cf_ray"),
        "cf_success": cf_meta.get("success"),
    }
    write_json(a_dir / "generation.json", gen_a)
    results["A"] = {"status": "OK", "t": round(runtime_a, 1), "file": str(a_out)}
    print(f"  DONE {runtime_a:.1f}s → {a_out}")
except Exception as e:
    results["A"] = {"status": "FAIL", "error": str(e)}
    print(f"  FAIL: {e}")

# ── TEST B — ControlNet Depth only (reuse Phase 0I rerun) ────────────────────
print("\n=== Test B: ControlNet Depth only (Stage 1 reuse) ===")
b_dir = OUT / "B_controlnet_depth"
b_dir.mkdir(parents=True, exist_ok=True)
b_out = b_dir / "output.png"
shutil.copy2(stage1_out, b_out)

stage1_gen_src = PHASE0I_DIR / "rerun" / "B_rgb_depth" / "generation.json"
gen_b = {
    "test": "B_CONTROLNET_DEPTH",
    "stage": 1,
    "source": "Phase 0I re-run (RC1 depth inversion + RC2 denoise=0.40)",
    "source_file": str(stage1_out),
    "output": str(b_out),
    "output_sha256": sha256(b_out),
    **STAGE1,
    "input_rgb_sha256": sha256(rgb_in),
    "depth_input_sha256": sha256(depth_inv),
}
if stage1_gen_src.exists():
    phase0i_meta = json.loads(stage1_gen_src.read_text())
    gen_b["phase0i_runtime_s"] = phase0i_meta.get("runtime_s")
write_json(b_dir / "generation.json", gen_b)
results["B"] = {"status": "OK", "t": 0, "file": str(b_out), "reused": True}
print(f"  Copied Stage 1 output → {b_out}")

# ── TEST C — Stage 1 → FLUX ───────────────────────────────────────────────────
print("\n=== Test C: Stage 1 → FLUX (two-stage) ===")
c_dir = OUT / "C_controlnet_then_flux"
c_dir.mkdir(parents=True, exist_ok=True)
c_ref = c_dir / "stage1_ref.png"
resize_reference(b_out, c_ref)
c_out = c_dir / "output.png"

t0 = time.time()
try:
    cf_meta_c, runtime_c = call_flux(account_id, api_token, c_ref, c_out, OUT_W, OUT_H, FLUX_PROMPT_POSITIVE)
    gen_c = {
        "test": "C_CONTROLNET_THEN_FLUX",
        "stage1": STAGE1,
        "stage1_input_rgb": str(rgb_in),
        "stage1_depth_inverted": str(depth_inv),
        "stage1_output": str(b_out),
        "stage1_output_sha256": sha256(b_out),
        "stage2_provider": "cloudflare",
        "stage2_model": MODEL,
        "stage2_input": str(b_out),
        "stage2_input_sha256": sha256(b_out),
        "stage2_reference_path": str(c_ref),
        "stage2_reference_sha256": sha256(c_ref),
        "stage2_prompt": FLUX_PROMPT_POSITIVE,
        "stage2_negative_prompt": FLUX_PROMPT_NEGATIVE,
        "stage2_guidance": GUIDANCE,
        "stage2_resolution": f"{OUT_W}x{OUT_H}",
        "stage2_runtime_s": round(runtime_c, 2),
        "stage2_cf_ray": cf_meta_c.get("cf_ray"),
        "stage2_cf_success": cf_meta_c.get("success"),
        "output": str(c_out),
        "output_sha256": sha256(c_out),
    }
    write_json(c_dir / "generation.json", gen_c)
    results["C"] = {"status": "OK", "t": round(runtime_c, 1), "file": str(c_out)}
    print(f"  DONE {runtime_c:.1f}s → {c_out}")
except Exception as e:
    results["C"] = {"status": "FAIL", "error": str(e)}
    print(f"  FAIL: {e}")

# ── COMPARISON ────────────────────────────────────────────────────────────────
print("\n=== Generating comparison ===")
comp_dir = OUT / "comparison"
comp_dir.mkdir(exist_ok=True)

panels = []
if (a_dir / "output.png").exists():
    panels.append((Image.open(a_dir / "output.png").convert("RGB"), "A — FLUX on RGB"))
if (b_dir / "output.png").exists():
    panels.append((Image.open(b_dir / "output.png").convert("RGB"), "B — ControlNet only"))
if (c_dir / "output.png").exists():
    panels.append((Image.open(c_dir / "output.png").convert("RGB"), "C — ControlNet → FLUX"))

W, H = 512, 512
LABEL_H = 44
PAD = 8
COLS = len(panels)
cw = COLS * W + (COLS + 1) * PAD
ch = H + LABEL_H + 2 * PAD
canvas = Image.new("RGB", (cw, ch), (30, 30, 30))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
except Exception:
    font = ImageFont.load_default()

for i, (img, label) in enumerate(panels):
    x = PAD + i * (W + PAD)
    y = PAD
    canvas.paste(img.resize((W, H)), (x, y))
    draw.rectangle([x, y + H, x + W, y + H + LABEL_H], fill=(50, 50, 50))
    bb = draw.textbbox((0, 0), label, font=font)
    tw = bb[2] - bb[0]
    tx = x + (W - tw) // 2
    ty = y + H + (LABEL_H - (bb[3] - bb[1])) // 2
    draw.text((tx, ty), label, fill=(220, 220, 220), font=font)

comp_path = comp_dir / "side_by_side.png"
canvas.save(comp_path)
print(f"  Saved: {comp_path}  ({canvas.size[0]}x{canvas.size[1]})")

# ── STUB REPORT (evaluation filled after visual inspection) ───────────────────
report = {
    "phase": "0J",
    "title": "Two-Stage Controlled → FLUX Bake-off",
    "date": "2026-09-17",
    "hypothesis": "ControlNet = geometry enforcer; FLUX = realism enhancer",
    "inputs": {
        "front_face_rgb": str(rgb_in),
        "depth_inverted": str(depth_inv),
        "stage1_controlled": str(stage1_out),
    },
    "stage1_parameters": STAGE1,
    "stage2_parameters": {
        "provider": "cloudflare",
        "model": MODEL,
        "guidance": GUIDANCE,
        "timeout_s": TIMEOUT,
        "ref_limit_px": REF_LIMIT,
        "prompt": FLUX_PROMPT_POSITIVE,
        "negative_prompt": FLUX_PROMPT_NEGATIVE,
    },
    "tests": results,
    "comparison_image": str(comp_path),
    "verdict": "PENDING_VISUAL_EVALUATION",
}
write_json(OUT / "phase0j_report.json", report)

print(f"\nResults: A={results.get('A',{}).get('t','?')}s  "
      f"B=reused  C={results.get('C',{}).get('t','?')}s")
print("DONE")
