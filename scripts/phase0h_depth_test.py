#!/usr/bin/env python3
"""Phase 0H: RGB-only vs RGB+Depth comparison using SD 1.5 + ControlNet.

Two experiments on the same model/seed:
  A — img2img RGB-only (no ControlNet)
  B — img2img + ControlNet Depth (SketchUp CLAHE depth)
  C — img2img + ControlNet Depth (MiDaS estimated depth from RGB)

Decision rule:
  If B or C has clearly better geometry consistency with SketchUp
  geometry AND photorealism is maintained → depth conditioning is useful.
  Otherwise: abandon depth; change generation engine.

VRAM: SD 1.5 + ControlNet fits in ~4GB (RTX 2060 safe).
Models auto-cached in D:\\hf_cache (set HF_HOME=D:\\hf_cache before running).

Usage:
  set HF_HOME=D:\\hf_cache
  python phase0h_depth_test.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Point HuggingFace cache to D: to avoid filling C:
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\transformers")

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from transformers import pipeline as hf_pipeline

PROJECT = Path(r"D:\Projects\skpto360")
RGB_ERP = PROJECT / "output" / "phase0e" / "input" / "rgb_erp.png"
DEPTH_SKETCHUP = PROJECT / "output" / "phase0h" / "depth_clahe.png"
OUT_DIR = PROJECT / "output" / "phase0h"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SD 1.5 img2img works at 512x512 or 768x768.
# For 2:1 ERP, use 768x384 to keep ratio (SD1.5 handles this).
TARGET_W, TARGET_H = 768, 384

PROMPT = (
    "photorealistic interior architectural visualization, "
    "360 equirectangular panorama, soft natural lighting, "
    "realistic material textures, refined surfaces, interior photography, "
    "high fidelity, maintain exact spatial layout and room proportions"
)
NEGATIVE_PROMPT = (
    "cartoon, painting, blurry, watermark, distorted geometry, "
    "changed room layout, extra furniture, missing walls"
)

DENOISE = 0.60        # img2img denoising strength
CONTROLNET_SCALE = 0.75
STEPS = 30
GUIDANCE = 7.5
SEED = 42


def load_and_resize(path: Path, w: int, h: int) -> Image.Image:
    img = Image.open(str(path)).convert("RGB")
    return img.resize((w, h), Image.LANCZOS)


def generate_midas_depth(rgb_pil: Image.Image) -> Image.Image:
    """Run MiDaS DPT-Large depth estimation on the RGB image."""
    print("  Estimating depth with MiDaS (DPT-Large)…")
    depth_pipe = hf_pipeline(
        "depth-estimation",
        model="Intel/dpt-large",
        device=0 if torch.cuda.is_available() else -1,
    )
    result = depth_pipe(rgb_pil)
    depth_arr = np.array(result["depth"])

    # MiDaS outputs inverse depth (bright = near). Normalize to [0, 255].
    lo, hi = np.percentile(depth_arr, 0.5), np.percentile(depth_arr, 99.5)
    norm = np.clip((depth_arr - lo) / (hi - lo + 1e-6), 0, 1)
    depth_uint8 = (norm * 255).astype(np.uint8)
    return Image.fromarray(depth_uint8).convert("RGB")


def run_rgb_only(rgb_pil: Image.Image, seed: int) -> tuple[Image.Image, float]:
    """Experiment A: img2img RGB-only, no ControlNet."""
    print("Loading SD 1.5 img2img pipeline…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=rgb_pil,
        strength=DENOISE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    )
    runtime = time.time() - t0
    del pipe
    torch.cuda.empty_cache()
    return result.images[0], runtime


def run_with_depth(
    rgb_pil: Image.Image,
    depth_pil: Image.Image,
    seed: int,
    label: str,
) -> tuple[Image.Image, float]:
    """Experiments B/C: img2img + ControlNet Depth."""
    print(f"Loading SD 1.5 + ControlNet Depth ({label})…")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=rgb_pil,
        control_image=depth_pil,
        strength=DENOISE,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    )
    runtime = time.time() - t0
    del pipe, controlnet
    torch.cuda.empty_cache()
    return result.images[0], runtime


def save_comparison(
    rgb: Image.Image,
    exp_a: Image.Image,
    exp_b: Image.Image,
    exp_c: Image.Image,
    depth_sk: Image.Image,
    depth_midas: Image.Image,
    out_path: Path,
) -> None:
    """Save 6-panel comparison: source | depth_sk | depth_midas | A | B | C."""
    w, h = rgb.size
    panel_w, panel_h = w, h
    canvas = Image.new("RGB", (panel_w * 3, panel_h * 2), (30, 30, 30))
    # Row 1: source | depth_sk | depth_midas
    canvas.paste(rgb, (0, 0))
    canvas.paste(depth_sk.resize((panel_w, panel_h)).convert("RGB"), (panel_w, 0))
    canvas.paste(depth_midas.resize((panel_w, panel_h)).convert("RGB"), (panel_w * 2, 0))
    # Row 2: Exp A | Exp B | Exp C
    canvas.paste(exp_a.resize((panel_w, panel_h)), (0, panel_h))
    canvas.paste(exp_b.resize((panel_w, panel_h)), (panel_w, panel_h))
    canvas.paste(exp_c.resize((panel_w, panel_h)), (panel_w * 2, panel_h))

    canvas.save(str(out_path))
    print(f"Comparison saved: {out_path}")


def main() -> None:
    print("=== Phase 0H: RGB-only vs RGB+Depth depth fidelity test ===\n")

    rgb_pil = load_and_resize(RGB_ERP, TARGET_W, TARGET_H)
    depth_sk_raw = Image.open(str(DEPTH_SKETCHUP)).convert("RGB")
    depth_sk_pil = depth_sk_raw.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    rgb_pil.save(str(OUT_DIR / "input_rgb_resized.png"))
    depth_sk_pil.save(str(OUT_DIR / "input_depth_sketchup.png"))

    # MiDaS depth from RGB
    depth_midas_pil = generate_midas_depth(rgb_pil)
    depth_midas_pil.save(str(OUT_DIR / "input_depth_midas.png"))
    torch.cuda.empty_cache()

    results: dict = {}

    # Experiment A: RGB-only
    print("\n[A] RGB-only img2img…")
    img_a, rt_a = run_rgb_only(rgb_pil, SEED)
    img_a.save(str(OUT_DIR / "exp_a_rgb_only.png"))
    results["A_rgb_only"] = {"runtime_s": round(rt_a, 1)}
    print(f"  Done in {rt_a:.1f}s")

    # Experiment B: RGB + SketchUp CLAHE depth
    print("\n[B] RGB + SketchUp CLAHE depth…")
    img_b, rt_b = run_with_depth(rgb_pil, depth_sk_pil, SEED, "SketchUp-CLAHE")
    img_b.save(str(OUT_DIR / "exp_b_rgb_depth_sketchup.png"))
    results["B_rgb_depth_sketchup"] = {"runtime_s": round(rt_b, 1)}
    print(f"  Done in {rt_b:.1f}s")

    # Experiment C: RGB + MiDaS depth
    print("\n[C] RGB + MiDaS depth…")
    img_c, rt_c = run_with_depth(rgb_pil, depth_midas_pil, SEED, "MiDaS")
    img_c.save(str(OUT_DIR / "exp_c_rgb_depth_midas.png"))
    results["C_rgb_depth_midas"] = {"runtime_s": round(rt_c, 1)}
    print(f"  Done in {rt_c:.1f}s")

    # 6-panel comparison
    comparison_path = OUT_DIR / "comparison_6panel.png"
    save_comparison(
        rgb_pil, img_a, img_b, img_c,
        depth_sk_pil, depth_midas_pil,
        comparison_path,
    )

    report = {
        "schema": "phase0h-depth-fidelity-test",
        "version": 1,
        "inputs": {
            "rgb_erp": str(RGB_ERP),
            "depth_sketchup_clahe": str(DEPTH_SKETCHUP),
        },
        "settings": {
            "model": "runwayml/stable-diffusion-v1-5",
            "controlnet": "lllyasviel/control_v11f1p_sd15_depth",
            "resolution": [TARGET_W, TARGET_H],
            "denoise_strength": DENOISE,
            "controlnet_scale": CONTROLNET_SCALE,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "seed": SEED,
        },
        "results": results,
        "outputs": {
            "exp_a": str(OUT_DIR / "exp_a_rgb_only.png"),
            "exp_b": str(OUT_DIR / "exp_b_rgb_depth_sketchup.png"),
            "exp_c": str(OUT_DIR / "exp_c_rgb_depth_midas.png"),
            "comparison": str(comparison_path),
        },
        "eval_criteria": [
            "Geometry consistency: walls/ceiling/floor planes match SketchUp source",
            "Photorealism: materials look real, not AI-hallucinated",
        ],
        "decision_rule": (
            "If B or C shows clearly better geometry than A AND photorealism maintained "
            "→ depth conditioning is useful. Proceed to SDXL + full constraint chain. "
            "Otherwise: change model/engine (no more constraints)."
        ),
    }
    report_path = OUT_DIR / "phase0h_test_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {report_path}")
    print("\n=== DONE — open comparison_6panel.png to evaluate ===")


if __name__ == "__main__":
    main()
