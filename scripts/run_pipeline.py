#!/usr/bin/env python3
"""Single-click AI360 panorama pipeline.

Chain:
  SketchUp Ruby exporter (menu item) -> output/obj/scene.{obj,json}
  -> Blender headless fast passes (cubemap -> ERP albedo/depth/normal/material_id)
  -> self-contained harness condition package built from the fresh ERP
  -> one Cloudflare FLUX.2 Klein call -> output/pipeline/panorama.png + report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "output"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
OBJ = OUT / "obj" / "scene.obj"
SCENE_JSON = OUT / "obj" / "scene.json"
FAST_PASSES = OUT / "fast_passes"
PIPELINE_OUT = OUT / "pipeline"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
import controlled_core_harness as harness  # noqa: E402
import phase0f_cloudflare_flux_experiment_a as flux  # noqa: E402
from backend.pass_pipeline import pack_rgb, snap_to_palette  # noqa: E402

GLAZING_KEYWORDS = (
    "glass", "kaca", "glazing", "window", "jendela", "transparent",
    "vitrine", "petir", "kaca0",
)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return harness.sha256(path)


def run(cmd: list[str], label: str) -> None:
    print(f"[pipeline] {label}")
    print("[pipeline]   ", " ".join(str(part) for part in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def image_dims(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    return [int(w), int(h)]


def copy_asset(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": str(target), "source": str(source), "bytes": target.stat().st_size, "sha256": sha256(target)}


def stage_blender(args) -> dict:
    if args.skip_blender:
        required = [FAST_PASSES / "erp" / "albedo.png", FAST_PASSES / "assembly_report.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("--skip-blender but fast passes missing: " + ", ".join(missing))
        return {"status": "SKIPPED", "note": "Reused existing output/fast_passes."}
    for path in (BLENDER, OBJ, SCENE_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"Required path not found: {path}")
    FAST_PASSES.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(BLENDER), "--background", "--python", str(SCRIPTS / "blender_passes.py"), "--",
            "--obj", str(OBJ),
            "--json", str(SCENE_JSON),
            "--out", str(FAST_PASSES),
            "--resolution", str(args.face_resolution),
        ],
        "Blender headless fast passes",
    )
    run(
        [
            sys.executable, str(SCRIPTS / "assemble_blender_passes.py"),
            "--in", str(FAST_PASSES),
            "--out", str(FAST_PASSES),
            "--width", str(args.erp_width),
        ],
        "Assemble ERP passes",
    )
    return {"status": "PASS", "report": str(FAST_PASSES / "assembly_report.json")}


def srgb_encode(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


def id_rgb_palette(material_count: int) -> list[tuple[int, int, int]]:
    palette = []
    for index in range(1, material_count + 1):
        digest = hashlib.sha256(f"material:{index}".encode()).digest()
        floats = [digest[offset] / 255.0 * 0.8 + 0.2 for offset in (0, 1, 2)]
        palette.append(tuple(int(round(srgb_encode(channel) * 255.0)) for channel in floats))
    return palette


def nearest_index(color: tuple[int, int, int], palette: list[tuple[int, int, int]]) -> int:
    best, best_index = None, 0
    for index, candidate in enumerate(palette, start=1):
        distance = abs(color[0] - candidate[0]) + abs(color[1] - candidate[1]) + abs(color[2] - candidate[2])
        if distance == 0:
            return index
        if best is None or distance < best:
            best, best_index = distance, index
    return best_index if best == 1 else 0


def material_observations(material_id_path: Path, id_index_path: Path) -> tuple[list[dict], np.ndarray]:
    labels = cv2.imread(str(material_id_path), cv2.IMREAD_COLOR)
    if labels is None:
        raise RuntimeError(f"Could not read material ID ERP: {material_id_path}")
    height, width = labels.shape[:2]

    name_by_index = {}
    if id_index_path.is_file():
        index_data = read_json(id_index_path)
        name_by_index = {int(value): key for key, value in index_data.get("material_indices", {}).items()}
    palette = id_rgb_palette(max(name_by_index) if name_by_index else 64)

    if name_by_index:
        snapped, _ = snap_to_palette(labels, np.array(palette, dtype=np.uint8)[:, ::-1])
    else:
        snapped = labels
    packed = pack_rgb(snapped).reshape(-1)
    values, counts = np.unique(packed, return_counts=True)
    total = packed.size

    records = []
    mask = np.zeros((height, width), dtype=np.uint8)
    manifest = {}
    manifest_path = OUT / "phase0e" / "input" / "material_manifest.json"
    if manifest_path.is_file():
        for item in read_json(manifest_path).get("materials", []):
            manifest[item["source_name"]] = item

    for value, count in zip(values.tolist(), counts.tolist()):
        if count < 100 and not name_by_index:
            continue
        bgr = [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF]
        rgba = (bgr[2], bgr[1], bgr[0])
        found = nearest_index(rgba, palette)
        source_name = name_by_index.get(found, f"Material_{found or len(records) + 1}")
        entry = manifest.get(source_name, {})
        opacity = float(entry.get("opacity", 1.0))
        is_glazing = opacity < 1.0 or any(token in source_name.lower() for token in GLAZING_KEYWORDS)
        records.append({
            "id": f"MAT_{found:03d}" if found else f"OBS_{len(records) + 1:03d}",
            "name": source_name,
            "base_color": [round(float(channel) / 255.0, 4) for channel in (rgba[0], rgba[1], rgba[2])],
            "opacity": opacity,
            "is_glazing": is_glazing,
            "visible": True,
            "visible_pixels": int(count),
            "visible_pixel_coverage": round(count / total, 6),
        })
        if found and is_glazing:
            mask[packed.reshape(height, width) == value] = 255

    records.sort(key=lambda item: item["visible_pixel_coverage"], reverse=True)
    return records, mask


def stage_harness(args) -> dict:
    erp = FAST_PASSES / "erp"
    rgb_source = erp / "rgb_neutral_erp.png"
    if not rgb_source.is_file():
        rgb_source = erp / "albedo.png"
    for required in (rgb_source, erp / "depth_condition.png", erp / "normal.png", erp / "material_id.png"):
        if not required.is_file():
            raise FileNotFoundError(f"Missing harness input from fast passes: {required}")

    pkg = PIPELINE_OUT / "condition_package"
    if pkg.exists():
        shutil.rmtree(pkg)
    for rel in ("source", "masks", "semantics", "qc"):
        (pkg / rel).mkdir(parents=True, exist_ok=True)

    assets = {
        "source": {
            "rgb_erp": copy_asset(rgb_source, pkg / "source" / "rgb_erp.png"),
            "depth_condition": copy_asset(erp / "depth_condition.png", pkg / "source" / "depth_condition.png"),
            "normal_condition": copy_asset(erp / "normal.png", pkg / "source" / "normal_condition.png"),
        },
    }
    materials, glass_mask = material_observations(erp / "material_id.png", FAST_PASSES / "id_index.json")
    cv2.imwrite(str(pkg / "masks" / "material_id.png"), cv2.imread(str(erp / "material_id.png"), cv2.IMREAD_COLOR))
    assets["masks"] = {
        "material_id": {"path": str(pkg / "masks" / "material_id.png"), "sha256": sha256(pkg / "masks" / "material_id.png")},
        "glass": {},
    }
    if bool(np.any(glass_mask)):
        glass_path = pkg / "masks" / "glass.png"
        cv2.imwrite(str(glass_path), glass_mask)
        assets["masks"]["glass"] = {"path": str(glass_path), "sha256": sha256(glass_path)}

    scene_json = read_json(SCENE_JSON)
    camera = scene_json.get("camera", {})
    width, height = image_dims(rgb_source)
    source_hashes = {
        "obj": sha256(OBJ),
        "scene_json": sha256(SCENE_JSON),
        "rgb_erp": assets["source"]["rgb_erp"]["sha256"],
    }
    scene_contract = {
        "schema": "rad-ai360-single-pipeline-scene-contract",
        "version": 1,
        "source_boundary": "Deterministic source truth from fast ERP passes; no legacy phase reports.",
        "camera": camera,
        "projection": {"type": "equirectangular", "width": width, "height": height, "source": "Blender Eevee fast geometry pass assembler"},
        "regions": {
            "interior": {"status": "IMPLICIT_SOURCE_RGB_NON_GLAZING"},
            "glazing": {"status": "NAME_OPACITY_HEURISTIC", "mask": "masks/glass.png" if assets["masks"]["glass"] else None},
            "known_exterior": {"status": "NOT_DETECTED", "regions": []},
            "unknown": {"status": "AVAILABLE_BY_EXCLUSION"},
        },
        "constraints": {
            "geometry": True, "depth": True, "normal": True, "materials": True,
            "glazing": "heuristic_only", "exterior_context": False, "lights": "partial",
        },
        "source_hashes": source_hashes,
    }
    materials_contract = {
        "schema": "rad-ai360-single-pipeline-materials",
        "version": 1,
        "source": "Observed material_id ERP coverage joined to OBJ-import material names via id_index.json.",
        "semantic_interpretation_used": False,
        "materials": materials,
    }
    lights_contract = {
        "schema": "rad-ai360-single-pipeline-lights",
        "version": 1,
        "source_boundary": "SOURCE_TRUTH_ONLY; no light emitters invented.",
        "scene": camera.get("scene_name", "AI360"),
        "lights": [],
        "unresolved_candidates": [],
        "exporter_gap": {
            "reliable_light_positions_available": False,
            "needed_upstream_fields": ["component names", "instance names", "tags/layers", "world-space transforms"],
        },
    }

    write_json(pkg / "semantics" / "scene_contract.json", scene_contract)
    write_json(pkg / "semantics" / "materials.json", materials_contract)
    write_json(pkg / "semantics" / "lights.json", lights_contract)
    write_json(pkg / "qc" / "source_metrics.json", {
        "schema": "rad-ai360-single-pipeline-source-metrics",
        "version": 1,
        "rgb_erp_dims": [width, height],
        "material_labels": len(materials),
        "glazing_label_count": sum(1 for item in materials if item["is_glazing"]),
    })

    manifest = {
        "schema": "rad-ai360-single-pipeline-condition-package",
        "version": 1,
        "pipeline": "CONTROLLED_CORE_FAST",
        "created_at_unix": int(time.time()),
        "source_truth_boundary": {
            "source_truth": ["scene.json camera", "OBJ geometry", "Blender fast ERP RGB/depth/normal", "material_id ERP"],
            "semantic_interpretation": ["none"],
            "ai_generation": ["not part of condition package"],
        },
        "assets": assets,
        "semantics": {"materials": "semantics/materials.json", "lights": "semantics/lights.json", "scene_contract": "semantics/scene_contract.json"},
        "qc": {"source_metrics": "qc/source_metrics.json"},
    }
    write_json(pkg / "manifest.json", manifest)

    harness.PACKAGE_DIR = pkg
    return {
        "status": "PASS",
        "package": str(pkg / "manifest.json"),
        "materials_observed": len(materials),
        "glazing_heuristic_labels": sum(1 for item in materials if item["is_glazing"]),
    }


def stage_flux(args) -> dict:
    pkg = PIPELINE_OUT / "condition_package"
    manifest = read_json(pkg / "manifest.json")
    rgb_source = pkg / "source" / "rgb_erp.png"
    prompt = harness.prompt_from_package(
        "Photorealistic architectural interior panorama faithful to the SketchUp design, generated from the deterministic unlit albedo; improve realism while preserving the exact design.",
        concise=False,
    )
    write_json(PIPELINE_OUT / "compiled_prompt.json", {"prompt": prompt, "source": "controlled_core_harness.prompt_from_package", "mode": "single-click pipeline"})

    class Args:
        model = args.model
        guidance = args.guidance
        timeout = args.timeout

    if args.skip_flux:
        return {"status": "SKIPPED", "prompt": prompt}

    generation = harness.run_flux_job("panorama", rgb_source, prompt, PIPELINE_OUT, Args(), manifest)
    output = Path(generation.get("output", ""))
    panorama_path = PIPELINE_OUT / "panorama.png"
    if output.is_file():
        shutil.copy2(output, panorama_path)
    qc = harness.qc_generation(rgb_source, output, manifest) if output.is_file() else {"status": "NOT_RUN"}
    return {
        "status": "ok" if generation.get("status") == "ok" and output.is_file() else "FAIL",
        "generation": generation,
        "qc": qc,
        "panorama": str(panorama_path),
        "panorama_sha256": sha256(panorama_path) if panorama_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-click AI360 panorama pipeline.")
    parser.add_argument("--out", type=Path, default=PIPELINE_OUT)
    parser.add_argument("--face-resolution", type=int, default=512)
    parser.add_argument("--erp-width", type=int, default=2048)
    parser.add_argument("--model", default=flux.MODEL)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--skip-flux", action="store_true")
    args = parser.parse_args()

    started = time.time()
    stages = {"blender": {}, "harness": {}, "flux": {}}
    try:
        stages["blender"] = stage_blender(args)
        stages["harness"] = stage_harness(args)
        stages["flux"] = stage_flux(args)
    except Exception as error:
        report = {
            "schema": "rad-ai360-single-pipeline-report",
            "version": 1,
            "status": "FAIL",
            "error": str(error),
            "stages": stages,
            "created_at_unix": int(time.time()),
        }
        write_json(PIPELINE_OUT / "report.json", report)
        print(f"RAD_PIPELINE_STATUS=FAIL")
        print(f"RAD_PIPELINE_REPORT={PIPELINE_OUT / 'report.json'}")
        return 1

    panorama = Path(stages["flux"].get("panorama") or "")
    report = {
        "schema": "rad-ai360-single-pipeline-report",
        "version": 1,
        "status": "PASS" if panorama.is_file() else "WARNING_NO_FLUX",
        "chain": "SketchUp Ruby -> OBJ -> Blender headless fast passes -> harness condition package -> Cloudflare FLUX.2 Klein (1 call)",
        "flux_calls": 0 if args.skip_flux else 1,
        "stages": stages,
        "panorama": str(panorama) if panorama.is_file() else None,
        "panorama_sha256": sha256(panorama) if panorama.is_file() else None,
        "prompt": str(PIPELINE_OUT / "compiled_prompt.json"),
        "duration_seconds": round(time.time() - started, 3),
        "created_at_unix": int(time.time()),
    }
    write_json(PIPELINE_OUT / "report.json", report)
    print(f"RAD_PIPELINE_STATUS={report['status']}")
    print(f"RAD_PIPELINE_REPORT={PIPELINE_OUT / 'report.json'}")
    if panorama.is_file():
        print(f"RAD_PANORAMA={panorama}")
    print(f"RAD_PIPELINE_FLUX={0 if args.skip_flux else 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())