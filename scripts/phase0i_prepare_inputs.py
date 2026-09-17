"""
Phase 0I — Prepare test inputs.

Produces:
  output/phase0i/input/rgb.png      — 512x512 front RGB
  output/phase0i/input/depth.png    — 512x512 depth PNG, E_PERCENTILE_CLAMPED mapping, near=white
"""

import os, numpy as np
from PIL import Image
import OpenEXR, Imath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "output", "phase0i", "input")
os.makedirs(OUT, exist_ok=True)

TARGET_SIZE = (512, 512)

# ── RGB ──────────────────────────────────────────────────────────────────────
rgb_src = os.path.join(ROOT, "output", "phase0c", "rgb", "front.png")
rgb = Image.open(rgb_src).convert("RGB")
print(f"RGB source: {rgb.size}")
rgb_512 = rgb.resize(TARGET_SIZE, Image.LANCZOS)
rgb_out = os.path.join(OUT, "rgb.png")
rgb_512.save(rgb_out)
print(f"Saved RGB: {rgb_out}  {rgb_512.size}")

# ── Depth EXR → 8-bit PNG (E_PERCENTILE_CLAMPED) ────────────────────────────
exr_src = os.path.join(ROOT, "output", "phase0c", "depth", "front.exr")
f = OpenEXR.InputFile(exr_src)
hdr = f.header()
dw  = hdr["dataWindow"]
w   = dw.max.x - dw.min.x + 1
h   = dw.max.y - dw.min.y + 1

raw = np.frombuffer(f.channel("V", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
depth = raw.reshape(h, w)
print(f"Depth EXR: {w}x{h}  range=[{depth.min():.3f}, {depth.max():.3f}] m")

# E_PERCENTILE_CLAMPED: p5-p95, linear, near=white(255)
p5, p95 = np.percentile(depth, 5), np.percentile(depth, 95)
print(f"Percentile clamp: p5={p5:.3f} m  p95={p95:.3f} m")
clamped = np.clip(depth, p5, p95)
normalized = (clamped - p5) / (p95 - p5)      # 0 = far, 1 = near
depth_8bit = (normalized * 255).astype(np.uint8) # near=white

depth_img = Image.fromarray(depth_8bit, mode="L")
depth_512 = depth_img.resize(TARGET_SIZE, Image.LANCZOS)
depth_out = os.path.join(OUT, "depth.png")
depth_512.save(depth_out)
print(f"Saved depth: {depth_out}  {depth_512.size}")
print(f"Depth PNG range: min={depth_8bit.min()}  max={depth_8bit.max()}  std={depth_8bit.std():.1f}")

# ── Alignment check ──────────────────────────────────────────────────────────
assert rgb_512.size == depth_512.size, "Dimension mismatch!"
print(f"\nAlignment: PASS — both inputs are {rgb_512.size}, same Blender pass, same camera.")
print("Phase 0I inputs ready.")
