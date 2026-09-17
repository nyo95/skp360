#!/usr/bin/env python3
"""Phase 0H: ControlNet Depth feasibility test — cubemap front face.

Isolates the question: does depth conditioning reduce geometry drift on a
conventional perspective view?  Uses phase0c/rgb/front.png + front.exr.
Does NOT mix equirectangular panorama limitations into this decision.

Experiment matrix (all share the same SD 1.5 model + seed):
  A  — RGB-only img2img              (control, no ControlNet)
  B  — RGB + LINEAR_DEPTH            deterministic, metric-linear, near=white
  C  — RGB + INVERSE_DEPTH           deterministic, 1/z, near=white
  D  — RGB + LOG_DEPTH               deterministic, log(z), near=white
  E  — RGB + PERCENTILE_CLAMPED      deterministic, p5-p95 clamp, linear, near=white
  F  — RGB + AI_ESTIMATED_DEPTH      MiDaS DPT-Large (compatibility reference only;
                                     NOT treated as geometry truth)

Evaluation criteria (geometry-first, photorealism secondary):
  1. Wall / opening preservation
  2. Furniture position
  3. Fixture count (ceiling lights)
  4. Cabinetry / ceiling preservation
  5. Camera composition
  6. Geometry drift (overall structure)

Convention: near=white (255), far=black (0) — matches MiDaS ControlNet training.
Background / invalid depth masked explicitly and set to 0 (far).
Cross-view consistency: all 6 faces would use the same mapping parameters
derived from the front face percentiles (single shared rule).

VRAM: SD 1.5 + ControlNet ≈ 4 GB — safe for RTX 2060 6 GB.
Quality judgement vs FLUX is NOT the goal; structural drift is.

Usage:
  set HF_HOME=D:\\hf_cache
  python phase0h_controlnet_depth_test.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import OpenEXR
import Imath
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\transformers")

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)
from transformers import pipeline as hf_pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT     = Path(r"D:\Projects\skpto360")
RGB_IN      = PROJECT / "output" / "phase0c" / "rgb"   / "front.png"
DEPTH_EXR   = PROJECT / "output" / "phase0c" / "depth" / "front.exr"
OUT_DIR     = PROJECT / "output" / "phase0h" / "controlnet_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Inference settings ────────────────────────────────────────────────────────
TARGET_SIZE      = 768      # SD 1.5 native range; 768 for better geometry retention
DENOISE          = 0.60     # img2img strength — enough to improve but preserve structure
CONTROLNET_SCALE = 0.75
STEPS            = 30
GUIDANCE         = 7.5
SEED             = 42

PROMPT = (
    "photorealistic interior architecture, living room, "
    "realistic lighting, high fidelity materials, "
    "natural soft shadows, interior photography"
)
NEGATIVE_PROMPT = (
    "cartoon, painting, blurry, watermark, "
    "distorted walls, moved furniture, extra objects, "
    "missing ceiling, missing fixtures, geometry errors"
)


# ─── Depth loading & mapping ───────────────────────────────────────────────────

def load_exr_depth(path: Path) -> np.ndarray:
    """Read single-channel float32 EXR depth. Returns float32 H×W array (meters)."""
    exr = OpenEXR.InputFile(str(path))
    hdr = exr.header()
    dw  = hdr["dataWindow"]
    w   = dw.max.x - dw.min.x + 1
    h   = dw.max.y - dw.min.y + 1
    ch  = list(hdr["channels"].keys())[0]
    raw = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
    arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    return arr


def build_valid_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: True where depth is finite, positive, and < background sentinel."""
    return np.isfinite(arr) & (arr > 0) & (arr < 1e30)


def _to_uint8(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Clip, normalize [0,1], scale to uint8.  Invalid pixels → 0 (far/black)."""
    out = np.zeros_like(x, dtype=np.float32)
    if valid.any():
        v      = x[valid]
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            out[valid] = (x[valid] - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def depth_linear(arr: np.ndarray, valid: np.ndarray,
                 p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """Mapping 1: linear metric depth, percentile-clipped, near=white."""
    v   = arr[valid]
    lo  = float(np.percentile(v, p_lo))
    hi  = float(np.percentile(v, p_hi))
    clipped = np.where(valid, np.clip(arr, lo, hi), 0.0)
    # near=white: invert (large depth → small value → dark)
    inverted = np.where(valid, hi - clipped + lo, 0.0)
    return _to_uint8(inverted, valid)


def depth_inverse(arr: np.ndarray, valid: np.ndarray,
                  p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """Mapping 2: inverse depth (1/z). Naturally near=bright. Clip outliers."""
    inv = np.where(valid, 1.0 / np.where(valid, arr, 1.0), 0.0)
    v   = inv[valid]
    lo  = float(np.percentile(v, p_lo))
    hi  = float(np.percentile(v, p_hi))
    inv_clipped = np.where(valid, np.clip(inv, lo, hi), 0.0)
    return _to_uint8(inv_clipped, valid)


def depth_log(arr: np.ndarray, valid: np.ndarray,
              p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """Mapping 3: log(z). Compresses far range. near=white (invert)."""
    safe = np.where(valid & (arr > 0), arr, 1.0)
    log  = np.where(valid, np.log(safe), 0.0)
    v    = log[valid]
    lo   = float(np.percentile(v, p_lo))
    hi   = float(np.percentile(v, p_hi))
    log_clipped = np.where(valid, np.clip(log, lo, hi), 0.0)
    # invert so near (small log) = bright
    inverted = np.where(valid, hi - log_clipped + lo, 0.0)
    return _to_uint8(inverted, valid)


def depth_percentile_clamped(arr: np.ndarray, valid: np.ndarray,
                             p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    """Mapping 4: aggressive percentile clamp (p5-p95) — robust global mapping.
    Designed for cross-view consistency: params are shared across all faces."""
    v   = arr[valid]
    lo  = float(np.percentile(v, p_lo))
    hi  = float(np.percentile(v, p_hi))
    clipped  = np.where(valid, np.clip(arr, lo, hi), 0.0)
    inverted = np.where(valid, hi - clipped + lo, 0.0)
    return _to_uint8(inverted, valid)


def to_pil_depth(uint8: np.ndarray, size: int) -> Image.Image:
    """Convert uint8 H×W depth to RGB PIL, resized for ControlNet."""
    pil = Image.fromarray(uint8, mode="L").convert("RGB")
    return pil.resize((size, size), Image.LANCZOS)


def generate_midas_depth(rgb_pil: Image.Image, size: int) -> Image.Image:
    """AI_ESTIMATED_DEPTH — MiDaS DPT-Large. Labeled as compatibility test only."""
    print("  [AI_ESTIMATED_DEPTH] Running MiDaS DPT-Large…")
    pipe = hf_pipeline(
        "depth-estimation",
        model="Intel/dpt-large",
        device=0 if torch.cuda.is_available() else -1,
    )
    result = pipe(rgb_pil)
    arr    = np.array(result["depth"]).astype(np.float32)
    valid  = np.isfinite(arr) & (arr > 0)
    # MiDaS: higher value = nearer (already near=bright) — just normalize
    lo, hi = float(np.percentile(arr[valid], 0.5)), float(np.percentile(arr[valid], 99.5))
    norm   = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    uint8  = (norm * 255).astype(np.uint8)
    pil    = Image.fromarray(uint8, mode="L").convert("RGB")
    del pipe
    torch.cuda.empty_cache()
    return pil.resize((size, size), Image.LANCZOS)


# ─── Inference ────────────────────────────────────────────────────────────────

def run_rgb_only(rgb_pil: Image.Image, seed: int) -> tuple[Image.Image, float]:
    print("  Loading SD 1.5 img2img…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t0  = time.time()
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=rgb_pil,
        strength=DENOISE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    )
    rt = time.time() - t0
    del pipe; torch.cuda.empty_cache()
    return out.images[0], rt


def run_with_controlnet(
    rgb_pil:   Image.Image,
    depth_pil: Image.Image,
    seed:      int,
    label:     str,
) -> tuple[Image.Image, float]:
    print(f"  Loading SD 1.5 + ControlNet Depth [{label}]…")
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=cn,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t0  = time.time()
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=rgb_pil,
        control_image=depth_pil,
        strength=DENOISE,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    )
    rt = time.time() - t0
    del pipe, cn; torch.cuda.empty_cache()
    return out.images[0], rt


# ─── Comparison grid ──────────────────────────────────────────────────────────

def save_depth_strip(depths: dict[str, np.ndarray], size: int, out_path: Path) -> None:
    """Save all depth maps side by side for inspection."""
    labels = list(depths.keys())
    n      = len(labels)
    canvas = np.zeros((size + 30, size * n, 3), dtype=np.uint8)
    for i, (label, arr) in enumerate(depths.items()):
        resized = np.array(Image.fromarray(arr, mode="L").resize((size, size)))
        bgr     = np.stack([resized] * 3, axis=-1)
        canvas[30:, i*size:(i+1)*size] = bgr
        cv2_text_workaround(canvas, label, i * size, 20)
    from PIL import Image as _PIL
    _PIL.fromarray(canvas).save(str(out_path))


def cv2_text_workaround(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    """Write label text using PIL (avoid cv2 font dependency)."""
    from PIL import Image as _PIL, ImageDraw, ImageFont
    tmp = _PIL.fromarray(canvas)
    draw = ImageDraw.Draw(tmp)
    draw.text((x + 4, y - 18), text, fill=(255, 200, 50))
    canvas[:] = np.array(tmp)


def save_result_grid(
    experiments: dict[str, Image.Image],
    depths:      dict[str, np.ndarray],
    size:        int,
    out_path:    Path,
) -> None:
    """Two-row grid: depth maps (row 1) | outputs (row 2)."""
    keys   = list(experiments.keys())
    n      = len(keys)
    grid_w = size * n
    grid_h = size * 2 + 60
    canvas = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))

    for i, key in enumerate(keys):
        # Row 1: depth map (or blank for RGB-only)
        if key in depths:
            d_arr  = depths[key]
            d_pil  = Image.fromarray(d_arr, mode="L").convert("RGB").resize((size, size))
            canvas.paste(d_pil, (i * size, 30))
        else:
            # RGB-only: show source RGB as reference
            src = experiments["A_RGB_ONLY"].resize((size, size))
            # dim it slightly to mark as "no depth"
            canvas.paste(src, (i * size, 30))

        # Row 2: inference result
        res = experiments[key].resize((size, size))
        canvas.paste(res, (i * size, size + 60))

    # Labels
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, key in enumerate(keys):
        draw.text((i * size + 6, 6), key, fill=(255, 200, 50))
        draw.text((i * size + 6, size + 38), f"OUT: {key}", fill=(100, 220, 100))

    canvas.save(str(out_path))
    print(f"Result grid saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Phase 0H: ControlNet Depth feasibility — front cubemap face")
    print("=" * 70)

    # Load inputs
    rgb_src  = Image.open(str(RGB_IN)).convert("RGB")
    rgb_pil  = rgb_src.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    rgb_pil.save(str(OUT_DIR / "input_rgb.png"))

    depth_arr = load_exr_depth(DEPTH_EXR)
    valid     = build_valid_mask(depth_arr)

    v = depth_arr[valid]
    exr_stats = {
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std()),
        "valid_pixels": int(valid.sum()), "total_pixels": int(depth_arr.size),
        "invalid_pixels": int((~valid).sum()),
        "percentiles": {str(p): float(np.percentile(v, p))
                        for p in [1, 5, 25, 50, 75, 95, 99]},
    }
    print(f"\nEXR stats: {json.dumps(exr_stats, indent=2)}")

    # Generate 4 deterministic depth maps
    d_linear   = depth_linear(depth_arr, valid)
    d_inverse  = depth_inverse(depth_arr, valid)
    d_log      = depth_log(depth_arr, valid)
    d_pctclamp = depth_percentile_clamped(depth_arr, valid)

    depth_maps_uint8 = {
        "B_LINEAR_DEPTH":          d_linear,
        "C_INVERSE_DEPTH":         d_inverse,
        "D_LOG_DEPTH":             d_log,
        "E_PERCENTILE_CLAMPED":    d_pctclamp,
    }

    # Save individual depth PNGs
    for name, arr in depth_maps_uint8.items():
        Image.fromarray(arr, mode="L").save(str(OUT_DIR / f"depth_{name.lower()}.png"))
        print(f"  Saved depth: {name} — min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")

    # Depth strip preview
    save_depth_strip(
        depth_maps_uint8,
        size=256,
        out_path=OUT_DIR / "depth_strip_preview.png",
    )

    # MiDaS — run before inference models load to free VRAM cleanly
    print("\n[F] Generating AI_ESTIMATED_DEPTH (MiDaS)…")
    midas_pil = generate_midas_depth(rgb_pil, TARGET_SIZE)
    midas_pil.save(str(OUT_DIR / "depth_F_AI_ESTIMATED_DEPTH.png"))

    # Prepare PIL depth images for ControlNet
    depth_pils = {k: to_pil_depth(v, TARGET_SIZE) for k, v in depth_maps_uint8.items()}
    depth_pils["F_AI_ESTIMATED_DEPTH"] = midas_pil

    # Run experiments
    results: dict[str, dict] = {}
    outputs:  dict[str, Image.Image] = {}

    print("\n[A] RGB-only (control)…")
    img_a, rt_a = run_rgb_only(rgb_pil, SEED)
    img_a.save(str(OUT_DIR / "out_A_RGB_ONLY.png"))
    outputs["A_RGB_ONLY"] = img_a
    results["A_RGB_ONLY"] = {"depth_map": "none", "runtime_s": round(rt_a, 1)}
    print(f"  Done {rt_a:.1f}s")

    exp_order = [
        ("B_LINEAR_DEPTH",       "LINEAR_DEPTH"),
        ("C_INVERSE_DEPTH",      "INVERSE_DEPTH"),
        ("D_LOG_DEPTH",          "LOG_DEPTH"),
        ("E_PERCENTILE_CLAMPED", "PERCENTILE_CLAMPED"),
        ("F_AI_ESTIMATED_DEPTH", "AI_ESTIMATED_DEPTH"),
    ]

    for exp_key, depth_key in exp_order:
        print(f"\n[{exp_key[0]}] {exp_key}…")
        img, rt = run_with_controlnet(
            rgb_pil, depth_pils[exp_key], SEED, depth_key
        )
        img.save(str(OUT_DIR / f"out_{exp_key}.png"))
        outputs[exp_key] = img
        results[exp_key] = {
            "depth_map": depth_key,
            "is_deterministic": not exp_key.startswith("F"),
            "runtime_s": round(rt, 1),
        }
        print(f"  Done {rt:.1f}s")

    # Full comparison grid
    all_depths_uint8 = {**depth_maps_uint8}  # A has no depth entry → handled in save_result_grid
    save_result_grid(
        experiments={**{"A_RGB_ONLY": img_a},
                     **{k: outputs[k] for k, _ in exp_order}},
        depths=all_depths_uint8,
        size=TARGET_SIZE,
        out_path=OUT_DIR / "comparison_grid.png",
    )

    # Report
    report = {
        "schema":  "phase0h-controlnet-depth-test",
        "version": 1,
        "inputs": {
            "rgb":         str(RGB_IN),
            "depth_exr":   str(DEPTH_EXR),
            "exr_stats":   exr_stats,
        },
        "settings": {
            "model":             "runwayml/stable-diffusion-v1-5",
            "controlnet":        "lllyasviel/control_v11f1p_sd15_depth",
            "resolution":        TARGET_SIZE,
            "denoise":           DENOISE,
            "controlnet_scale":  CONTROLNET_SCALE,
            "steps":             STEPS,
            "guidance":          GUIDANCE,
            "seed":              SEED,
            "depth_convention":  "near=white(255), far=black(0)",
        },
        "depth_mappings": {
            "B_LINEAR_DEPTH":       "linear metric z, p0.5-p99.5 clamp, inverted",
            "C_INVERSE_DEPTH":      "1/z, naturally near=bright, p0.5-p99.5 clamp",
            "D_LOG_DEPTH":          "log(z), p0.5-p99.5 clamp, inverted",
            "E_PERCENTILE_CLAMPED": "linear metric z, p5-p95 clamp (robust global rule)",
            "F_AI_ESTIMATED_DEPTH": "MiDaS DPT-Large — AI_ESTIMATED_DEPTH, NOT geometry truth",
        },
        "results": results,
        "eval_criteria": [
            "1. Wall / opening preservation",
            "2. Furniture position",
            "3. Fixture count (ceiling recessed lights)",
            "4. Cabinetry / ceiling preservation",
            "5. Camera composition",
            "6. Geometry drift (overall structural accuracy)",
        ],
        "decision_rule": (
            "If any deterministic depth (B-E) shows clearly reduced geometry drift vs A: "
            "retain Blender depth as canonical truth and proceed to higher-quality model. "
            "If F (MiDaS) outperforms B-E: report as ControlNet compatibility advantage "
            "only — do NOT conclude MiDaS is geometrically more accurate. "
            "If A ≈ B-E ≈ F: depth conditioning does not help at this scale; "
            "change generation engine without adding more constraints."
        ),
        "outputs_dir": str(OUT_DIR),
    }

    report_path = OUT_DIR / "phase0h_controlnet_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")
    print("=== DONE — evaluate comparison_grid.png ===")


if __name__ == "__main__":
    main()
