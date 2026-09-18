from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import controlled_core_harness as harness
import phase0f_cloudflare_flux_experiment_a as flux


OUT = ROOT / "output"
B3_DIR = OUT / "generation_bakeoff" / "B3_contract_faithful"
B1_DIR = OUT / "generation_bakeoff" / "B1_controlled_harness"
B2_DIR = OUT / "generation_bakeoff" / "B2_contract_rich"
PASS_DIR = OUT / "fast_passes"
ALBEDO = PASS_DIR / "erp" / "albedo.png"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def image(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return value


def image_record(path: Path, source: Path | None = None) -> dict:
    return {
        "path": str(path),
        "source": str(source or path),
        "bytes": path.stat().st_size,
        "sha256": harness.sha256(path),
    }


def copy_pass(name: str) -> dict:
    source = PASS_DIR / "erp" / name
    target = B3_DIR / "input" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return image_record(target, source)


def contract_snapshot() -> None:
    source_contracts = B2_DIR / "contracts"
    target_contracts = B3_DIR / "contracts"
    target_contracts.mkdir(parents=True, exist_ok=True)
    for name in ("scene.json", "materials.json", "lighting.json", "glazing.json", "geometry.json"):
        shutil.copy2(source_contracts / name, target_contracts / name)
    write_json(target_contracts / "provenance.json", {
        "schema": "rad-ai360-b3-provenance",
        "canonical_scene": str(OUT / "obj" / "scene.json"),
        "canonical_geometry": str(OUT / "obj" / "scene.obj"),
        "source_pass_report": str(PASS_DIR / "pass_generation_report.json"),
        "projection_report": str(PASS_DIR / "assembly_report.json"),
        "projection_contract": "backend/projection.py frozen after regression gate",
        "renderer": "Blender Eevee fast geometry pass generator",
        "photometric_policy": "No lighting, AO, shadows, exposure, reflections, or geometry edits baked into B3 RGB.",
        "flux_stage": "NOT RUN; B3 is a pre-Flux canonical ERP source.",
        "source_hashes": {
            "scene_json": harness.sha256(OUT / "obj" / "scene.json"),
            "scene_obj": harness.sha256(OUT / "obj" / "scene.obj"),
            "albedo_erp": harness.sha256(ALBEDO),
        },
    })


def compile_prompt() -> str:
    package = harness.read_json(OUT / "condition_package" / "manifest.json")
    prompt = harness.prompt_from_package(
        "Preserve the canonical architectural scene exactly. Use the validated albedo as the faithful color reference; do not redesign, beautify, add, remove, move, or reshape geometry."
    )
    write_json(B3_DIR / "compiled_prompt.json", {
        "prompt": prompt,
        "source": "controlled_core_harness.prompt_from_package",
        "mode": "pre-Flux contract compiler output; no provider call in B3",
        "condition_package": str(OUT / "condition_package" / "manifest.json"),
    })
    write_json(B3_DIR / "compiled_prompt_trace.json", {
        "rules": [
            {"prompt_rule": "preserve exact camera and ERP orientation", "source_contract": "scene_contract", "confidence": 1.0},
            {"prompt_rule": "preserve walls, openings, ceilings, arches, doors, windows, furniture silhouettes and placement", "source_contract": "geometry_contract", "confidence": 1.0},
            {"prompt_rule": "preserve source material identity and base color", "source_contract": "materials_contract", "confidence": 0.8},
            {"prompt_rule": "do not invent fixture positions or photometric precision", "source_contract": "lighting_contract", "confidence": 1.0},
            {"prompt_rule": "preserve known glazing and exterior behavior", "source_contract": "glazing_contract", "confidence": 0.8},
        ]
    })
    return prompt


def qc_b3(canonical: Path, output: Path) -> dict:
    source = image(canonical, cv2.IMREAD_COLOR)
    result = image(output, cv2.IMREAD_COLOR)
    gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    source_edges = cv2.Canny(gray_source, 40, 120)
    result_edges = cv2.Canny(gray_result, 40, 120)
    edge_union = (source_edges > 0) | (result_edges > 0)
    edge_overlap = (source_edges > 0) & (result_edges > 0)
    pixel_delta = np.abs(source.astype(np.int16) - result.astype(np.int16))
    normal_path = PASS_DIR / "erp" / "normal.png"
    normal = image(normal_path, cv2.IMREAD_COLOR)
    seam_delta = float(np.mean(np.abs(source[:, 0].astype(np.float32) - source[:, -1].astype(np.float32))) / 255.0)
    return {
        "status": "PASS" if source.shape == result.shape and not np.any(pixel_delta) else "FAIL",
        "canonical_pixel_identity": {
            "changed_pixels": int(np.count_nonzero(np.any(pixel_delta != 0, axis=2))),
            "mae_0_255": float(pixel_delta.mean()),
            "max_delta": int(pixel_delta.max()),
        },
        "projection": {
            "dimensions": [int(result.shape[1]), int(result.shape[0])],
            "ratio_2_to_1": result.shape[1] == result.shape[0] * 2,
            "orientation": "frozen canonical projection; no mirror or rotation applied in B3",
            "seam_source_mean_delta": seam_delta,
            "seam_status": "PASS" if seam_delta < 0.15 else "WARNING",
        },
        "structural_edges": {
            "source_pixels": int(np.count_nonzero(source_edges)),
            "output_pixels": int(np.count_nonzero(result_edges)),
            "iou": float(np.count_nonzero(edge_overlap) / max(np.count_nonzero(edge_union), 1)),
            "status": "PASS" if np.array_equal(source_edges, result_edges) else "WARNING",
        },
        "normal_erp": {
            "exists": normal_path.is_file(),
            "dimensions": [int(normal.shape[1]), int(normal.shape[0])],
            "aligned_dimensions": normal.shape[:2] == result.shape[:2],
        },
        "contract_checks": {
            "geometry_changes": "PASS: B3 RGB is an exact canonical albedo copy; no geometry operation performed.",
            "object_additions_removals": "PASS: no generation or scene mutation performed.",
            "material_correspondence": "PASS: B3 RGB is byte-identical to authoritative albedo ERP.",
            "glazing_openings": "PASS: no pixel edits or opaque replacement performed.",
            "photometric_baking": "PASS: no lighting/AO/shadow/exposure/reflection pass applied.",
        },
    }


def main() -> int:
    required = [ALBEDO, PASS_DIR / "erp" / "depth_condition.png", PASS_DIR / "erp" / "normal.png", PASS_DIR / "assembly_report.json", B1_DIR / "output.png", B2_DIR / "output.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing B3 input: " + ", ".join(missing))
    if B3_DIR.exists():
        shutil.rmtree(B3_DIR)
    B3_DIR.mkdir(parents=True)
    contract_snapshot()
    compile_prompt()

    assets = {}
    for name in ("albedo.png", "depth_condition.png", "normal.png", "material_id.png", "depth_raw.npy"):
        assets[name] = copy_pass(name)
    output = B3_DIR / "output.png"
    shutil.copy2(ALBEDO, output)
    assets["output.png"] = image_record(output, ALBEDO)

    qc = qc_b3(ALBEDO, output)
    write_json(B3_DIR / "qc.json", qc)
    write_json(B3_DIR / "generation.json", {
        "schema": "rad-ai360-b3-generation",
        "status": "PRE_FLUX_CANONICAL_SOURCE",
        "provider": None,
        "note": "No AI generation run. B3 is the contract-faithful photometric ERP source for a later Flux stage.",
        "source_albedo": image_record(ALBEDO),
        "output": image_record(output, ALBEDO),
    })

    b1_report = harness.read_json(OUT / "generation_bakeoff" / "report.json")
    b2_comparison = harness.read_json(B2_DIR / "comparison_to_B1.json")
    comparison = {
        "schema": "rad-ai360-b1-b2-b3-comparison",
        "baseline_integrity": "B1 and B2 outputs were read only; no overwrite performed.",
        "B1": {"output": str(B1_DIR / "output.png"), "qc": b1_report.get("qc", {}).get("B1")},
        "B2": {"output": str(B2_DIR / "output.png"), "qc": b2_comparison.get("B2")},
        "B3": {"output": str(output), "qc": qc, "mode": "canonical albedo pass; pre-Flux"},
        "selection": "B3 is the recommended pre-Flux canonical source for fidelity, not a realism winner over generated B1/B2 images.",
        "remaining_contract_violations": ["No photometric realism stage is applied yet; this is intentional to protect geometry/material truth."],
        "verdict": "B3 SUITABLE AS PRE-FLUX CANONICAL ERP" if qc["status"] == "PASS" else "B3 FAIL",
        "created_at_unix": int(time.time()),
    }
    write_json(B3_DIR / "comparison_to_B1_B2.json", comparison)
    print(f"RAD_B3_OUTPUT={output}")
    print(f"RAD_B3_QC={B3_DIR / 'qc.json'}")
    return 0 if qc["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
