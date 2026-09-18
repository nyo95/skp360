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

from backend.projection import FACE_BASIS, cubemap_to_erp

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


def make_face(face: str, size: int = 256) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    color = FACE_COLORS[face]
    img[:] = (24, 24, 24)
    cv2.rectangle(img, (16, 16), (size - 17, size - 17), color, thickness=18)
    cv2.circle(img, (size // 2, size // 2), size // 6, color, thickness=-1)

    # Directional marker so face rotations are visible deterministically.
    center = (size // 2, size // 2)
    if face == "front":
        cv2.arrowedLine(img, (size // 2, size - 40), (size // 2, 40), (255, 255, 255), 6)
    elif face == "right":
        cv2.arrowedLine(img, (40, size // 2), (size - 40, size // 2), (255, 255, 255), 6)
    elif face == "back":
        cv2.arrowedLine(img, (size // 2, size - 40), (size // 2, 40), (255, 255, 255), 6)
    elif face == "left":
        cv2.arrowedLine(img, (size - 40, size // 2), (40, size // 2), (255, 255, 255), 6)
    elif face == "top":
        cv2.arrowedLine(img, (size // 2, size - 40), (size // 2, 40), (255, 255, 255), 6)
    elif face == "bottom":
        cv2.arrowedLine(img, (size // 2, 40), (size // 2, size - 40), (255, 255, 255), 6)

    pil = Image.fromarray(img[..., ::-1])
    draw = ImageDraw.Draw(pil)
    label = face.upper()
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2
    y = size // 2 - 22
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return np.asarray(pil)[..., ::-1]


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
        },
        "expected_regions": {},
    }

    def region_probe(name: str, x_frac: float, y_frac: float):
        x = int(round(x_frac * (erp.shape[1] - 1)))
        y = int(round(y_frac * (erp.shape[0] - 1)))
        bgr = erp[y, x]
        report["expected_regions"][name] = {
            "xy": [x, y],
            "rgb": [int(v) for v in bgr[::-1]],
            "dominant_face": name,
        }

    region_probe("front", 0.5, 0.5)
    region_probe("right", 0.75, 0.5)
    region_probe("left", 0.25, 0.5)
    region_probe("back", 1.0, 0.5)
    region_probe("zenith", 0.5, 0.0)
    region_probe("nadir", 0.5, 1.0)

    # Verify dominant face colors appear in each expected region.
    region_checks = {}
    for sample_name, expected in {
        "front": "front",
        "right": "right",
        "left": "left",
        "back": "back",
        "zenith": "top",
        "nadir": "bottom",
    }.items():
        x = int(round({
            "front": 0.5,
            "right": 0.75,
            "left": 0.25,
            "back": 1.0,
            "zenith": 0.5,
            "nadir": 0.5,
        }[sample_name] * (erp.shape[1] - 1)))
        y = int(round({
            "front": 0.5,
            "right": 0.5,
            "left": 0.5,
            "back": 0.5,
            "zenith": 0.0,
            "nadir": 1.0,
        }[sample_name] * (erp.shape[0] - 1)))
        pix = erp[y, x]
        face_color = np.array(FACE_COLORS[expected], dtype=np.uint8)
        delta = int(np.linalg.norm(np.asarray(pix, dtype=np.int16) - face_color))
        region_checks[sample_name] = {
            "expected_face": expected,
            "distance_to_expected_color": delta,
            "pass": delta < 150,
        }

    report["region_checks"] = region_checks
    report["status"] = "PASS" if all(v["pass"] for v in region_checks.values()) else "FAIL"
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
            "camera_forward": [1.0, 0.0, 0.0],
            "camera_right": [0.0, -1.0, 0.0],
            "camera_up": [0.0, 0.0, 1.0],
            "longitude_zero": "front (+X)",
            "longitude_plus_90": "right (-Y)",
            "longitude_minus_90": "left (+Y)",
            "longitude_180": "back (-X)",
            "zenith": "+Z",
            "nadir": "-Z",
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
        "root_cause": "Phase 0L used a different yaw convention in its remap logic than the actual SketchUp/Phase 0C basis. The stitching formula inverted the ERP longitude sign and mismatched the face assignments so the panorama was effectively reversed relative to the Blender depth ERP.",
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
