#!/usr/bin/env python3
"""Phase 0H: Pre-process the SketchUp depth ERP for ControlNet Depth.

Problem: The raw depth_condition.png has range [186, 255] — nearly all white.
ControlNet Depth needs full-range [0, 255] grayscale where:
  - 0   = far (background, empty space)
  - 255 = near (close to camera)

Steps:
  1. Load depth PNG (uint8 grayscale)
  2. Linear stretch [min, max] → [0, 255]
  3. Optionally apply CLAHE for local contrast boost
  4. Save as phase0h/depth_normalized.png
  5. Save debug side-by-side preview
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(r"D:\Projects\skpto360")
DEPTH_IN = PROJECT / "output" / "phase0e" / "input" / "depth_condition.png"
OUT_DIR = PROJECT / "output" / "phase0h"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stretch(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Percentile-based linear stretch to full [0, 255] range."""
    p_lo, p_hi = np.percentile(img, lo), np.percentile(img, hi)
    if p_hi == p_lo:
        return np.zeros_like(img)
    stretched = (img.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def main() -> None:
    raw = cv2.imread(str(DEPTH_IN), cv2.IMREAD_GRAYSCALE)
    assert raw is not None, f"Cannot read: {DEPTH_IN}"

    # Stats on raw
    stats_raw = {
        "min": int(raw.min()),
        "max": int(raw.max()),
        "mean": float(raw.mean().round(2)),
        "std": float(raw.std().round(2)),
        "unique_values": int(len(np.unique(raw))),
        "p1": int(np.percentile(raw, 1)),
        "p5": int(np.percentile(raw, 5)),
        "p50": int(np.percentile(raw, 50)),
        "p95": int(np.percentile(raw, 95)),
        "p99": int(np.percentile(raw, 99)),
    }

    # Stretch to full range
    normalized = stretch(raw)

    # CLAHE for additional local contrast (helps ControlNet read fine structure)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_depth = clahe.apply(normalized)

    # Stats on normalized
    stats_norm = {
        "min": int(normalized.min()),
        "max": int(normalized.max()),
        "mean": float(normalized.mean().round(2)),
        "std": float(normalized.std().round(2)),
        "unique_values": int(len(np.unique(normalized))),
    }

    # Save outputs
    norm_path = OUT_DIR / "depth_normalized.png"
    clahe_path = OUT_DIR / "depth_clahe.png"
    cv2.imwrite(str(norm_path), normalized)
    cv2.imwrite(str(clahe_path), clahe_depth)

    # Side-by-side preview (raw | normalized | clahe)
    preview_w = 1280
    each_w = preview_w // 3
    each_h = int(each_w * raw.shape[0] / raw.shape[1])

    panels = [
        cv2.resize(raw, (each_w, each_h)),
        cv2.resize(normalized, (each_w, each_h)),
        cv2.resize(clahe_depth, (each_w, each_h)),
    ]
    comparison = np.hstack([cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) for p in panels])
    label_y = 30
    for i, label in enumerate(["RAW (186-255)", "NORMALIZED (0-255)", "CLAHE"]):
        cv2.putText(
            comparison, label,
            (i * each_w + 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
    preview_path = OUT_DIR / "depth_comparison_preview.png"
    cv2.imwrite(str(preview_path), comparison)

    report = {
        "schema": "phase0h-depth-preprocess",
        "depth_in": str(DEPTH_IN),
        "stats_raw": stats_raw,
        "stats_normalized": stats_norm,
        "outputs": {
            "normalized": str(norm_path),
            "clahe": str(clahe_path),
            "preview": str(preview_path),
        },
        "recommendation": (
            "Use depth_clahe.png for ControlNet — it has the most local contrast. "
            "Confirm in preview that near objects (floor/ceiling/arches) are clearly "
            "distinguished from far objects (doorway backgrounds, walls at distance)."
        ),
    }
    report_path = OUT_DIR / "phase0h_depth_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nPreview saved: {preview_path}")


if __name__ == "__main__":
    main()
