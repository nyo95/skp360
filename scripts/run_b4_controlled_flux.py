from __future__ import annotations

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backend.pass_pipeline import pack_rgb

import controlled_core_harness as harness
import phase0f_cloudflare_flux_experiment_a as flux


OUT = ROOT / "output"
B4_DIR = OUT / "generation_bakeoff" / "B4_controlled_flux"
B3_DIR = OUT / "generation_bakeoff" / "B3_contract_faithful"
PACKAGE_DIR = OUT / "condition_package"
B3_SOURCE = B3_DIR / "output.png"
COMPOSED_SOURCE = OUT / "fast_passes" / "erp" / "albedo_composed.png"
DEPTH_SOURCE = OUT / "fast_passes" / "erp" / "depth_condition.png"
NORMAL_SOURCE = OUT / "fast_passes" / "erp" / "normal.png"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_image(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return image


def make_structural_overlay(albedo_path: Path, depth_path: Path, normal_path: Path, output_path: Path) -> dict:
    albedo = read_image(albedo_path, cv2.IMREAD_COLOR)
    depth = read_image(depth_path, cv2.IMREAD_GRAYSCALE)
    normal = read_image(normal_path, cv2.IMREAD_COLOR)
    size = (albedo.shape[1], albedo.shape[0])
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    normal = cv2.resize(normal, size, interpolation=cv2.INTER_LINEAR)
    depth_edges = cv2.Canny(depth, 40, 120)
    normal_edges = cv2.Canny(cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY), 40, 120)
    structural_edges = (depth_edges > 0) | (normal_edges > 0)
    overlay = albedo.astype(np.float32)
    dark_gray = np.full_like(overlay, 48.0)
    overlay[structural_edges] = overlay[structural_edges] * 0.80 + dark_gray[structural_edges] * 0.20
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(overlay, 0, 255).astype(np.uint8))
    return {
        "path": str(output_path),
        "depth_edge_pixels": int(np.count_nonzero(depth_edges)),
        "normal_edge_pixels": int(np.count_nonzero(normal_edges)),
        "opacity": 0.20,
        "color": "dark_gray",
        "sent_to_flux": False,
    }


def seam_metrics(image: np.ndarray) -> dict:
    delta = np.abs(image[:, 0].astype(np.float32) - image[:, -1].astype(np.float32)) / 255.0
    return {"mean": float(delta.mean()), "max": float(delta.max())}


def orientation_metrics(source: np.ndarray, output: np.ndarray) -> dict:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    source_gray -= source_gray.mean()
    output_gray -= output_gray.mean()
    same = float(np.sum(source_gray * output_gray) / max(np.linalg.norm(source_gray) * np.linalg.norm(output_gray), 1e-8))
    flipped = np.fliplr(source_gray)
    flipped_corr = float(np.sum(flipped * output_gray) / max(np.linalg.norm(flipped) * np.linalg.norm(output_gray), 1e-8))
    return {
        "same_orientation_correlation": same,
        "horizontal_flip_correlation": flipped_corr,
        "status": "PASS" if same >= flipped_corr else "FAIL_ORIENTATION_CHANGED",
    }


def structure_metrics(source: np.ndarray, output: np.ndarray, structural_reference: np.ndarray) -> dict:
    source_edges = structural_reference
    output_edges = cv2.Canny(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), 40, 120)
    union = (source_edges > 0) | (output_edges > 0)
    overlap = (source_edges > 0) & (output_edges > 0)
    return {
        "source_edge_pixels": int(np.count_nonzero(source_edges)),
        "output_edge_pixels": int(np.count_nonzero(output_edges)),
        "edge_iou": float(np.count_nonzero(overlap) / max(np.count_nonzero(union), 1)),
        "edge_difference_density": float(np.mean(source_edges != output_edges)),
        "status": "UNCALIBRATED",
        "interpretation": "Depth/normal-to-RGB edge correspondence has no calibrated acceptance threshold yet; it cannot be used as a PASS gate.",
    }


def material_region_metrics(source: np.ndarray, output: np.ndarray, material_id: np.ndarray) -> dict:
    if material_id.shape[:2] != source.shape[:2]:
        material_id = cv2.resize(material_id, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Vectorised per-label aggregation. The previous implementation looped over every
    # unique colour with a full-frame boolean mask, which does not terminate once the
    # ID pass carries interpolated colours.
    labels = pack_rgb(material_id[..., :3]).reshape(-1)
    unique, inverse = np.unique(labels, return_inverse=True)
    delta = np.abs(source.reshape(-1, 3).astype(np.float32) - output.reshape(-1, 3).astype(np.float32)).mean(axis=1)
    pixel_counts = np.bincount(inverse, minlength=unique.size)
    delta_sums = np.bincount(inverse, weights=delta, minlength=unique.size)
    records = []
    for index, packed in enumerate(unique.tolist()):
        if pixel_counts[index] < 100:
            continue
        records.append(
            {
                "material_id_bgr": [int((packed >> 16) & 0xFF), int((packed >> 8) & 0xFF), int(packed & 0xFF)],
                "pixels": int(pixel_counts[index]),
                "mean_rgb_delta_0_255": float(delta_sums[index] / pixel_counts[index]),
            }
        )
    records.sort(key=lambda item: item["pixels"], reverse=True)
    return {
        "status": "UNCALIBRATED",
        "label_count": int(unique.size),
        "regions": records,
        "interpretation": "material_visual_drift only; RGB drift is not proof of semantic material change.",
    }



def qc_b3_to_b4(source_path: Path, output_path: Path) -> dict:
    source = read_image(source_path)
    output = read_image(output_path)
    if output.shape[:2] != source.shape[:2]:
        output_at_source = cv2.resize(output, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        output_at_source = output
    material_id = read_image(OUT / "fast_passes" / "erp" / "material_id.png")
    depth = cv2.resize(read_image(DEPTH_SOURCE, cv2.IMREAD_GRAYSCALE), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    normal = cv2.resize(read_image(NORMAL_SOURCE, cv2.IMREAD_COLOR), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    structural_reference = (cv2.Canny(depth, 40, 120) > 0) | (cv2.Canny(cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY), 40, 120) > 0)
    pixel_delta = np.abs(source.astype(np.int16) - output_at_source.astype(np.int16))
    orientation = orientation_metrics(source, output_at_source)
    source_seam = seam_metrics(source)
    output_seam = seam_metrics(output)
    dimensions_pass = output.shape[1] == output.shape[0] * 2
    orientation_pass = orientation["status"] == "PASS" and orientation["same_orientation_correlation"] >= orientation["horizontal_flip_correlation"] + 0.02
    seam_pass = output_seam["mean"] <= 0.15 and output_seam["max"] <= 0.75
    structure = structure_metrics(source, output_at_source, structural_reference.astype(np.uint8) * 255)
    material_drift = material_region_metrics(source, output_at_source, material_id)
    hard_gate_failures = []
    if not dimensions_pass:
        hard_gate_failures.append("ERP output is not exactly 2:1")
    if not orientation_pass:
        hard_gate_failures.append("orientation/mirroring gate failed")
    if not seam_pass:
        hard_gate_failures.append("seam delta exceeded hard gate")
    uncalibrated_checks = [structure["status"], material_drift["status"]]
    return {
        "status": "REJECT" if hard_gate_failures else ("UNCALIBRATED" if "UNCALIBRATED" in uncalibrated_checks else "PASS"),
        "hard_gate_failures": hard_gate_failures,
        "gates": {
            "dimensions_ratio_2_to_1": "PASS" if dimensions_pass else "REJECT",
            "orientation": "PASS" if orientation_pass else "REJECT",
            "seam": "PASS" if seam_pass else "REJECT",
            "structural_edge_drift": structure["status"],
        },
        "source": {"path": str(source_path), "dimensions": [int(source.shape[1]), int(source.shape[0])], "seam": source_seam},
        "output": {"path": str(output_path), "dimensions": [int(output.shape[1]), int(output.shape[0])], "ratio_2_to_1": output.shape[1] == output.shape[0] * 2, "seam": output_seam},
        "architectural_geometry_changes": structure,
        "object_displacement_addition_removal": {"status": "UNKNOWN", "reason": "No object detector or semantic correspondence model is used; visual edge drift is reported separately."},
        "opening_glazing_changes": {"status": "UNCALIBRATED", "mask": "condition_package glazing/exterior masks are available for downstream review; RGB-only Flux has no hard mask control."},
        "material_visual_drift": material_drift,
        "erp_orientation_stitching_integrity": orientation,
        "photorealistic_improvement": {"status": "UNCALIBRATED", "reason": "No subjective quality score or human visual claim is encoded in this QC."},
        "pixel_change_summary": {
            "changed_pixel_percent": float(np.mean(np.any(pixel_delta != 0, axis=2)) * 100.0),
            "mean_rgb_delta_0_255": float(pixel_delta.mean()),
            "max_rgb_delta": int(pixel_delta.max()),
        },
    }


def main() -> int:
    required = [B3_SOURCE, DEPTH_SOURCE, NORMAL_SOURCE, PACKAGE_DIR / "manifest.json", OUT / "fast_passes" / "erp" / "material_id.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing B4 input: " + ", ".join(missing))
    if B4_DIR.exists():
        shutil.rmtree(B4_DIR)
    B4_DIR.mkdir(parents=True)

    package = harness.read_json(PACKAGE_DIR / "manifest.json")
    prompt = harness.prompt_from_package("Photorealistic architectural visualization with exact design preservation.", concise=True)
    write_json(B4_DIR / "compiled_prompt.json", {"prompt": prompt, "source": "controlled_core_harness.prompt_from_package"})
    overlay = make_structural_overlay(B3_SOURCE, DEPTH_SOURCE, NORMAL_SOURCE, B4_DIR / "diagnostic_structural_overlay.png")
    provenance = {
        "schema": "rad-ai360-b4-provenance",
        "baseline": "B3_contract_faithful",
        "baseline_sha256": harness.sha256(B3_SOURCE),
        "flux_source": str(B3_SOURCE),
        "flux_source_sha256": harness.sha256(B3_SOURCE),
        "compiled_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "canonical_scene": str(OUT / "obj" / "scene.json"),
        "contracts": str(PACKAGE_DIR / "manifest.json"),
        "projection": "backend/projection.py; unchanged",
        "provider": flux.MODEL,
        "controls": {"image_sources": ["B3 canonical ERP only"], "depth": False, "normal": False, "material_id": False, "controlnet": False},
        "diagnostic_overlay": overlay,
        "supported_parameters": {"guidance": 3.5, "timeout_seconds": 300, "width": "provider-derived", "height": "provider-derived"},
        "unsupported_parameters": {"image_strength_control": "unsupported", "denoise": "unsupported", "seed": "unsupported", "depth_weight": "unsupported", "normal_weight": "unsupported"},
    }
    write_json(B4_DIR / "provenance.json", provenance)

    class Args:
        model = flux.MODEL
        guidance = 3.5
        timeout = 300

    generation = harness.run_flux_job("B4_controlled_flux", B3_SOURCE, prompt, OUT / "generation_bakeoff", Args(), package)
    provenance.update({
        "provider_parameters_sent": generation.get("request_parameters", {}).get("multipart_fields", {}),
        "provider_response_metadata": {key: generation.get(key) for key in ("provider_request_id", "cloudflare_success", "cloudflare_errors", "cloudflare_messages")},
        "generation_status": generation.get("status"),
    })
    write_json(B4_DIR / "provenance.json", provenance)
    output = Path(generation.get("output", ""))
    qc = qc_b3_to_b4(B3_SOURCE, output) if output.is_file() else {"status": "REJECT", "hard_gate_failures": [generation.get("error", "Flux output missing")]} 
    write_json(B4_DIR / "generation.json", generation)
    write_json(B4_DIR / "qc.json", qc)
    write_json(B4_DIR / "comparison_B3_to_B4.json", {
        "baseline": "B3 is immutable and was read only.",
        "B4": "RGB-only Flux reconstruction from B3; no depth/normal conditioning.",
        "qc": qc,
        "conclusion": "RGB-only Flux controllability is insufficient for the next structural conditioning experiment if orientation fails or major geometry drift is observed; otherwise proceed only as a controlled baseline.",
        "created_at_unix": int(time.time()),
    })
    print(f"RAD_B4_OUTPUT={output}")
    print(f"RAD_B4_QC={B4_DIR / 'qc.json'}")
    return 0 if generation.get("status") == "ok" and qc.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
