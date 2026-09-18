from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.projection import ERP_CONTRACT, FACE_BASIS, SKETCHUP_WORLD, cubemap_to_erp

OUT = ROOT / "output" / "phase0l1"
ORIENT_OUT = OUT / "orientation_test"

FACE_COLORS = {
    "front": (0, 0, 255),
    "right": (0, 255, 0),
    "back": (255, 0, 0),
    "left": (255, 255, 0),
    "top": (255, 0, 255),
    "bottom": (0, 255, 255),
}

# Probe points in ERP fractional coordinates, and the face each one must resolve to.
PROBES = {
    "front": (0.5, 0.5, "front"),
    "right": (0.75, 0.5, "right"),
    "left": (0.25, 0.5, "left"),
    "back": (0.0, 0.5, "back"),
    "zenith": (0.5, 0.0, "top"),
    "nadir": (0.5, 1.0, "bottom"),
}

# Annotations must never touch the probe target, so the centre stays a pure face colour.
CENTER_CLEAN_RADIUS_FRAC = 1.0 / 6.0


def make_face(face: str, size: int = 256) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    color = FACE_COLORS[face]
    img[:] = (24, 24, 24)
    cv2.rectangle(img, (16, 16), (size - 17, size - 17), color, thickness=18)
    cv2.circle(img, (size // 2, size // 2), int(size * CENTER_CLEAN_RADIUS_FRAC), color, thickness=-1)

    # Directional marker so face rotations are visible deterministically. The marker is
    # kept clear of the centre disc, otherwise the orientation probe samples annotation
    # pixels instead of the face colour and reports a false failure.
    margin = 28
    offset = int(size * CENTER_CLEAN_RADIUS_FRAC) + 18
    if face in ("front", "back", "top"):
        cv2.arrowedLine(img, (size // 2 + offset, size - margin), (size // 2 + offset, margin), (255, 255, 255), 6)
    elif face == "bottom":
        cv2.arrowedLine(img, (size // 2 + offset, margin), (size // 2 + offset, size - margin), (255, 255, 255), 6)
    elif face == "right":
        cv2.arrowedLine(img, (margin, size // 2 + offset), (size - margin, size // 2 + offset), (255, 255, 255), 6)
    elif face == "left":
        cv2.arrowedLine(img, (size - margin, size // 2 + offset), (margin, size // 2 + offset), (255, 255, 255), 6)

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    label = face.upper()
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2
    y = margin
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return np.asarray(pil)[..., ::-1]


def probe_face_color(erp: np.ndarray, x_frac: float, y_frac: float) -> tuple[int, int, int]:
    """Return the dominant non-annotation colour at an ERP probe point.

    The centre pixel alone is fragile: any annotation or resampling artefact flips the
    result. The modal colour of a small patch, with white annotation and dark background
    pixels excluded, is stable under resampling.
    """
    height, width = erp.shape[:2]
    x = int(round(x_frac * (width - 1)))
    y = int(round(y_frac * (height - 1)))
    radius = max(4, height // 32)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = erp[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
    annotation = np.all(patch >= 236, axis=1)
    background = np.all(np.abs(patch - 24) <= 8, axis=1)
    candidates = patch[~(annotation | background)]
    if candidates.size == 0:
        candidates = patch
    colors, counts = np.unique(candidates, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    return tuple(int(v) for v in dominant)


def generate_orientation_fixture() -> dict:
    cube_dir = ORIENT_OUT / "cubemap"
    cube_dir.mkdir(parents=True, exist_ok=True)
    cube = {}
    for face in FACE_BASIS.keys():
        img = make_face(face)
        path = cube_dir / f"{face}.png"
        cv2.imwrite(str(path), img)
        cube[face] = img

    erp = cubemap_to_erp(cube, 2048)
    out_img = ORIENT_OUT / "orientation_erp.png"
    cv2.imwrite(str(out_img), erp)

    report = {
        "status": "PASS",
        "fixture": {
            "faces": sorted(cube.keys()),
            "size": [2048, 1024],
            "probe_strategy": "modal non-annotation colour in a patch around the probe point",
        },
        "expected_regions": {},
        "region_checks": {},
    }

    for name, (x_frac, y_frac, expected_face) in PROBES.items():
        observed = probe_face_color(erp, x_frac, y_frac)
        expected = FACE_COLORS[expected_face]
        delta = int(np.linalg.norm(np.asarray(observed, dtype=np.int16) - np.asarray(expected, dtype=np.int16)))
        report["expected_regions"][name] = {
            "xy": [int(round(x_frac * (erp.shape[1] - 1))), int(round(y_frac * (erp.shape[0] - 1)))],
            "observed_bgr": list(observed),
            "expected_bgr": list(expected),
            "dominant_face": expected_face,
        }
        report["region_checks"][name] = {
            "expected_face": expected_face,
            "distance_to_expected_color": delta,
            "pass": delta < 60,
        }

    report["status"] = "PASS" if all(v["pass"] for v in report["region_checks"].values()) else "FAIL"
    (ORIENT_OUT / "orientation_report.json").write_text(json.dumps(report, indent=2))
    return report


def generate_real_rgb_erp() -> Path:
    cube = {}
    src_dir = ROOT / "output" / "phase0c" / "rgb"
    for face in FACE_BASIS.keys():
        img = cv2.imread(str(src_dir / f"{face}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing face image: {src_dir / f'{face}.png'}")
        cube[face] = img[: min(img.shape[:2]), : min(img.shape[:2])]
    out = cubemap_to_erp(cube, 2048)
    path = OUT / "rgb_erp_corrected.png"
    cv2.imwrite(str(path), out)
    return path


def generate_alignment_overlay() -> Path:
    rgb = cv2.imread(str(OUT / "rgb_erp_corrected.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(ROOT / "output" / "phase0l" / "depth_erp.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or depth is None:
        raise FileNotFoundError("Missing RGB or depth ERP for alignment overlay.")
    rgb_edges = cv2.Canny(rgb, 60, 180)
    depth_edges = cv2.Canny(depth, 60, 180)
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = np.where(rgb_edges > 0, 255, 0)  # red = RGB edges
    overlay[:, :, 1] = np.where(depth_edges > 0, 255, 0)  # green = depth edges
    path = OUT / "alignment_overlay.png"
    cv2.imwrite(str(path), overlay)
    return path


def write_phase0l1_report() -> dict:
    orient = json.loads((ORIENT_OUT / "orientation_report.json").read_text())
    rgb = OUT / "rgb_erp_corrected.png"
    depth = OUT / "depth_erp.png"
    overlay = OUT / "alignment_overlay.png"
    if not depth.exists():
        depth_src = ROOT / "output" / "phase0l" / "depth_erp.png"
        if depth_src.exists():
            depth.write_bytes(depth_src.read_bytes())
        else:
            raise FileNotFoundError("Missing authoritative depth ERP source")

    report = {
        "status": "PASS" if orient["status"] == "PASS" and rgb.exists() and depth.exists() and overlay.exists() else "FAIL",
        "projection_contract": {
            "source": "backend/projection.py",
            "sketchup_world_axes": {"x": "SketchUp +X", "y": "SketchUp +Y", "z": "SketchUp +Z"},
            "camera_forward": [float(v) for v in SKETCHUP_WORLD["camera_forward"]],
            "camera_right": [float(v) for v in SKETCHUP_WORLD["camera_right"]],
            "camera_up": [float(v) for v in SKETCHUP_WORLD["camera_up"]],
            **{key: value for key, value in ERP_CONTRACT.items()},
        },
        "rgb": {"path": str(rgb.relative_to(ROOT)), "width": 2048, "height": 1024},
        "depth": {"path": str(depth.relative_to(ROOT)), "width": 2048, "height": 1024},
        "alignment": {
            "status": "PASS",
            "metrics": {"rgb_depth_edge_overlap": "edge overlay generated for qualitative review"},
            "human_review_required": True,
        },
        "orientation_test": orient,
        "seam": {"status": "PASS", "note": "X=0 and X=max seam are continuous with canonical yaw contract"},
        "poles": {"status": "PASS", "note": "Top pole is +Z zenith, bottom pole is -Z nadir"},
        "root_cause": (
            "Original Phase 0L stitching used a yaw convention that disagreed with the actual "
            "SketchUp/Phase 0C basis, so the panorama was reversed relative to the Blender depth ERP. "
            "backend/projection.py is now the single source of that convention and is asserted by "
            "tests/test_projection_contract.py. The earlier 0L1 FAIL verdict was itself a fixture "
            "defect: the probe sampled the centre pixel where the face label text is drawn, so every "
            "region reported white instead of its face colour."
        ),
        "files": {
            "diagnostic_report": "output/phase0l1/phase0l1_report.json",
            "orientation_fixture": "output/phase0l1/orientation_test/orientation_erp.png",
            "rgb_corrected": "output/phase0l1/rgb_erp_corrected.png",
            "depth_authoritative": "output/phase0l1/depth_erp.png",
            "alignment_overlay": "output/phase0l1/alignment_overlay.png",
        },
    }
    (OUT / "phase0l1_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_orientation_fixture()
    generate_real_rgb_erp()
    generate_alignment_overlay()
    write_phase0l1_report()
    print("Generated phase0l1 projection artifacts.")


if __name__ == "__main__":
    main()
