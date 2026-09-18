from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pass_pipeline import (
    FACE_ORDER,
    build_manifest,
    edge_overlap,
    pack_rgb,
    radial_depth_condition,
    snap_to_palette,
    stitch_passes,
)
from backend.projection import LABEL_PASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Assemble fast Blender cubemap passes into aligned ERP assets.")
    parser.add_argument("--in", dest="input_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--label-min-face-pixels", type=int, default=8)
    return parser.parse_args()


def read_face(path: Path, flags: int):
    if path.suffix == ".npy":
        image = np.load(path)
        if image.ndim != 2:
            raise ValueError(f"Expected single-channel depth array: {path}")
        return image.astype(np.float32)
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable pass: {path}")
    return image


def label_palette(faces: dict, min_face_pixels: int) -> np.ndarray:
    """Collect the legal ID colours actually present in the rendered faces.

    With anti-aliasing disabled each face contains only true ID colours, so this is an
    exact palette. If anti-aliasing ever leaks back in, rare blended colours fall below
    `min_face_pixels` and are excluded, and the snap report will show a non-zero change.
    """
    counts: dict[int, int] = {}
    for image in faces.values():
        packed = pack_rgb(image[..., :3])
        values, occurrences = np.unique(packed, return_counts=True)
        for value, occurrence in zip(values.tolist(), occurrences.tolist()):
            if occurrence >= min_face_pixels:
                counts[value] = counts.get(value, 0) + occurrence
    if not counts:
        raise ValueError("No stable label colours found in cubemap faces.")
    packed_palette = np.array(sorted(counts), dtype=np.uint32)
    return np.stack(
        ((packed_palette >> 16) & 0xFF, (packed_palette >> 8) & 0xFF, packed_palette & 0xFF),
        axis=1,
    ).astype(np.uint8)


def assemble_pass(input_dir: Path, output_dir: Path, pass_name: str, extension: str, width: int, flags: int):
    faces = {face: read_face(input_dir / "cubemap" / pass_name / f"{face}.{extension}", flags) for face in FACE_ORDER}
    if any(not np.any(image) for image in faces.values()):
        raise ValueError(f"Blank cubemap face in pass: {pass_name}")
    face_shape = faces[FACE_ORDER[0]].shape[:2]
    if face_shape[0] != face_shape[1] or any(image.shape[:2] != face_shape for image in faces.values()):
        raise ValueError(f"Cubemap faces must share one square resolution: {pass_name}")
    erp = stitch_passes(faces, width, pass_name=pass_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pass_name}.{'npy' if pass_name == 'depth_raw' else 'png'}"
    if pass_name == "depth_raw":
        np.save(path, erp.astype(np.float32))
    else:
        cv2.imwrite(str(path), erp)
    return path, faces, erp


def assemble_label_pass(input_dir: Path, output_dir: Path, pass_name: str, width: int, min_face_pixels: int):
    path, faces, erp = assemble_pass(input_dir, output_dir, pass_name, "png", width, cv2.IMREAD_COLOR)
    palette = label_palette(faces, min_face_pixels)
    snapped, stats = snap_to_palette(erp, palette)
    cv2.imwrite(str(path), snapped)
    stats["min_face_pixels"] = min_face_pixels
    stats["stitch_interpolation"] = "nearest"
    return path, faces, snapped, stats



def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.out)
    cubemap_dir = output_dir / "cubemap"
    erp_dir = output_dir / "erp"
    files = {}
    face_counts = {}

    albedo_path, albedo_faces, albedo_erp = assemble_pass(input_dir, erp_dir, "albedo", "png", args.width, cv2.IMREAD_COLOR)
    files["erp/albedo"] = albedo_path
    files.update({f"cubemap/albedo/{face}": input_dir / "cubemap" / "albedo" / f"{face}.png" for face in FACE_ORDER})
    face_counts["albedo"] = len(albedo_faces)

    ao_path, ao_faces, ao_erp = assemble_pass(input_dir, erp_dir, "ao", "png", args.width, cv2.IMREAD_GRAYSCALE)
    files["erp/ao"] = ao_path
    files.update({f"cubemap/ao/{face}": input_dir / "cubemap" / "ao" / f"{face}.png" for face in FACE_ORDER})
    face_counts["ao"] = len(ao_faces)
    albedo_float = albedo_erp.astype(np.float32) / 255.0
    ao_factor = ao_erp.astype(np.float32) / 255.0
    albedo_composed = np.clip(albedo_float * ao_factor[..., None], 0.0, 1.0)
    composed_path = erp_dir / "albedo_composed.png"
    cv2.imwrite(str(composed_path), np.round(albedo_composed * 255.0).astype(np.uint8))
    files["erp/albedo_composed"] = composed_path

    depth_path, depth_faces, depth_erp = assemble_pass(input_dir, erp_dir, "depth_raw", "npy", args.width, cv2.IMREAD_UNCHANGED)
    files["erp/depth_raw"] = depth_path
    files.update({f"cubemap/depth_raw/{face}": input_dir / "cubemap" / "depth_raw" / f"{face}.exr" for face in FACE_ORDER})
    face_counts["depth_raw"] = len(depth_faces)
    depth_condition, depth_stats = radial_depth_condition(depth_erp)
    depth_condition_path = erp_dir / "depth_condition.png"
    cv2.imwrite(str(depth_condition_path), depth_condition)
    files["erp/depth_condition"] = depth_condition_path

    normal_path, normal_faces, _ = assemble_pass(input_dir, erp_dir, "normal", "png", args.width, cv2.IMREAD_UNCHANGED)
    files["erp/normal"] = normal_path
    files.update({f"cubemap/normal/{face}": input_dir / "cubemap" / "normal" / f"{face}.png" for face in FACE_ORDER})
    face_counts["normal"] = len(normal_faces)

    label_stats = {}
    for pass_name in LABEL_PASSES:
        if not (input_dir / "cubemap" / pass_name / f"{FACE_ORDER[0]}.png").is_file():
            continue
        path, label_faces, _, stats = assemble_label_pass(
            input_dir, erp_dir, pass_name, args.width, args.label_min_face_pixels
        )
        files[f"erp/{pass_name}"] = path
        files.update(
            {f"cubemap/{pass_name}/{face}": input_dir / "cubemap" / pass_name / f"{face}.png" for face in FACE_ORDER}
        )
        face_counts[pass_name] = len(label_faces)
        label_stats[pass_name] = stats

    overlay = np.zeros((*albedo_erp.shape[:2], 3), dtype=np.uint8)
    albedo_gray = cv2.cvtColor(albedo_erp, cv2.COLOR_BGR2GRAY)
    albedo_edges = cv2.Canny(albedo_gray, 40, 120)
    depth_edges = cv2.Canny(depth_condition, 40, 120)
    overlay[albedo_edges > 0] = (0, 0, 255)
    overlay[depth_edges > 0] = (255, 255, 0)
    both = (albedo_edges > 0) & (depth_edges > 0)
    overlay[both] = (255, 255, 255)
    overlay_path = output_dir / "alignment_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    files["alignment_overlay"] = overlay_path

    report = {
        "schema": "rad-ai360-assembled-passes",
        "status": "PASS",
        "projection": {
            "source": "backend/projection.py",
            "face_order": FACE_ORDER,
            "face_width": int(albedo_faces[FACE_ORDER[0]].shape[1]),
            "face_height": int(albedo_faces[FACE_ORDER[0]].shape[0]),
            "erp_width": args.width,
            "erp_height": args.width // 2,
        },
        "faces": face_counts,
        "ao": {"raw": "stitched Eevee Ambient Occlusion factor", "composite": "albedo_composed = albedo_rgb * (ao_gray / 255)", "status": "PASS"},
        "depth": {"raw": "true radial distance in metres", "conditioning": "p2 white to p98 black", "statistics": depth_stats},
        "label_passes": label_stats,
        "alignment": edge_overlap(albedo_erp, depth_condition),
        "assets": {name: str(path.relative_to(output_dir)).replace("\\", "/") for name, path in files.items()},
    }
    report["manifest"] = build_manifest(output_dir, files, {"projection_source": "backend/projection.py", "face_order": FACE_ORDER})
    (output_dir / "assembly_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"RAD_FAST_ASSEMBLY_REPORT={output_dir / 'assembly_report.json'}")


if __name__ == "__main__":
    main()