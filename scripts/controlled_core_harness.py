#!/usr/bin/env python3
"""Controlled Core v0 condition package and FLUX harness."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import phase0f_cloudflare_flux_experiment_a as flux


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "output"
PACKAGE_DIR = OUT / "condition_package"
BAKEOFF_DIR = OUT / "generation_bakeoff"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_asset(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return file_record(target, source)


def file_record(path: Path, source: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "source": str(source) if source else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def image_dims(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    return [int(w), int(h)]


def make_edges(source_rgb: Path, target: Path) -> dict[str, Any]:
    image = cv2.imread(str(source_rgb), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {source_rgb}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), edges)
    return {
        **file_record(target, source_rgb),
        "method": "Canny edges from authoritative RGB ERP; QC/reference signal only.",
    }


def make_empty_mask(target: Path, width: int, height: int, reason: str) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), np.zeros((height, width), dtype=np.uint8))
    return {**file_record(target), "reason": reason}


def material_contract(material_manifest: dict[str, Any], phase0g: dict[str, Any], visibility: dict[str, Any] | None) -> dict[str, Any]:
    glass_ids = {item["material_id"]: item for item in phase0g["glass"]["detected_materials"]}
    visibility_by_id = {}
    if visibility:
        for item in visibility["outputs"]["coverage"]["materials"]:
            visibility_by_id[item["id"]] = item

    materials = []
    for record in material_manifest["materials"]:
        visible = visibility_by_id.get(record["id"], {})
        glass = record["id"] in glass_ids
        materials.append({
            "id": record["id"],
            "name": record["source_name"],
            "base_color": record.get("base_color"),
            "opacity": record.get("opacity"),
            "visible": bool(visible.get("visible", False)),
            "visible_pixels": visible.get("visible_pixels", 0),
            "visible_pixel_coverage": visible.get("visible_pixel_coverage", 0),
            "is_glazing": glass,
            "is_exterior_facing": None,
            "source_texture": record.get("texture"),
            "semantic_class": None,
            "evidence": (
                glass_ids[record["id"]].get("reason_detected", [])
                if glass else []
            ),
        })
    return {
        "schema": "rad-ai360-controlled-material-contract",
        "version": 1,
        "source": "Phase 0E material manifest + Phase 0G deterministic glass candidates + Controlled material-ID visibility pass.",
        "semantic_interpretation_used": False,
        "materials": materials,
    }


def light_contract(phase0g_light_audit: dict[str, Any], scene_json: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    candidates = []
    for index, item in enumerate(phase0g_light_audit.get("candidates", []), start=1):
        candidates.append({
            "id": f"fixture_candidate_{index:03d}",
            "source_name": item["source_name"],
            "material_id": item.get("material_id"),
            "world_position_mm": None,
            "erp_x": None,
            "erp_y": None,
            "confidence": 0.25 if item.get("confidence") == "low" else 0.5,
            "evidence": item.get("evidence", []),
            "status": "UNKNOWN",
            "reason_not_projected": "Current scene.json/export does not include SketchUp component/group/instance transforms; material-name evidence alone is not reliable fixture position source truth.",
        })
    return {
        "schema": "rad-ai360-controlled-lights",
        "version": 1,
        "source_boundary": "SOURCE_TRUTH_ONLY; no light emitters invented.",
        "scene": scene_json["camera"]["scene_name"],
        "projection": {"type": "equirectangular", "width": width, "height": height},
        "lights": [],
        "unresolved_candidates": candidates,
        "exporter_gap": {
            "reliable_light_positions_available": False,
            "needed_upstream_fields": [
                "component names",
                "group names",
                "instance names",
                "tags/layers",
                "world-space transforms",
                "bounding boxes or insertion points",
            ],
        },
    }


def scene_contract(scene_json: dict[str, Any], phase0d: dict[str, Any], phase0g: dict[str, Any], phase0h: dict[str, Any], source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "rad-ai360-controlled-scene-contract",
        "version": 1,
        "source_boundary": "Deterministic source truth only; no LLM/vision semantic interpretation.",
        "camera": {
            **scene_json["camera"],
            "basis_m": phase0d["camera_basis"],
        },
        "projection": {
            "type": "equirectangular",
            "width": phase0d["render"]["resolution"][0],
            "height": phase0d["render"]["resolution"][1],
            "source": "Blender panoramic equirectangular",
        },
        "regions": {
            "interior": {"status": "IMPLICIT_SOURCE_RGB_NON_GLAZING", "mask": None},
            "glazing": {
                "status": "AVAILABLE",
                "mask": "masks/glass.png",
                "candidates": phase0g["glass"]["detected_materials"],
            },
            "known_exterior": {
                "status": phase0h["exterior_context"]["status"],
                "mask": "masks/exterior_context.png",
                "context": "context/exterior_context.png",
                "regions": phase0h["exterior_context"]["regions"],
            },
            "unknown": {
                "status": "AVAILABLE_BY_EXCLUSION",
                "definition": "Glazing candidates with context_source UNKNOWN or unconstrained regions not backed by deterministic context mask.",
            },
        },
        "constraints": {
            "geometry": True,
            "depth": True,
            "normal": True,
            "materials": True,
            "glazing": True,
            "exterior_context": True,
            "lights": "partial",
        },
        "source_hashes": source_hashes,
    }


def source_metrics(package_assets: dict[str, Any], phase0d: dict[str, Any], phase0h: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "rad-ai360-controlled-source-metrics",
        "version": 1,
        "projection": phase0d["validation"]["rgb_validation"],
        "seam": {
            "source_rgb_mean_lr_delta": phase0d["validation"]["rgb_validation"]["mean_lr_seam_rgb_delta"],
            "source_rgb_max_lr_delta": phase0d["validation"]["rgb_validation"]["max_lr_seam_rgb_delta"],
            "exterior_context_mean_lr_delta": phase0h["local_qc"]["context_panorama"]["mean_rgb_seam_delta"],
        },
        "masks": {
            "glass": image_dims(Path(package_assets["masks"]["glass"]["path"])),
            "exterior_context": image_dims(Path(package_assets["masks"]["exterior_context"]["path"])),
        },
    }


def build_package(args: argparse.Namespace) -> None:
    phase0d = read_json(OUT / "phase0d" / "phase0d_report.json")
    phase0e = read_json(OUT / "phase0e" / "phase0e_report.json")
    phase0g = read_json(OUT / "phase0g" / "phase0g_report.json")
    phase0h = read_json(OUT / "phase0h" / "phase0h_report.json")
    scene_json = read_json(OUT / "obj" / "scene.json")
    material_manifest = read_json(OUT / "phase0e" / "input" / "material_manifest.json")
    light_audit = read_json(OUT / "phase0g" / "light_audit.json")
    visibility_path = OUT / "condition_work" / "material_visibility.json"
    visibility = read_json(visibility_path) if visibility_path.is_file() else None

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    for rel in ("source", "masks", "context", "semantics", "qc"):
        (PACKAGE_DIR / rel).mkdir(parents=True, exist_ok=True)

    assets = {
        "source": {
            "rgb_erp": copy_asset(Path(phase0g["outputs"]["after_source_fix_rgb"]), PACKAGE_DIR / "source" / "rgb_erp.png"),
            "depth_exr": copy_asset(Path(phase0d["outputs"]["depth"]), PACKAGE_DIR / "source" / "depth.exr"),
            "depth_condition": copy_asset(Path(phase0e["inputs"]["files"]["depth_condition"]["path"]), PACKAGE_DIR / "source" / "depth_condition.png"),
            "normal_exr": copy_asset(Path(phase0d["outputs"]["normal"]), PACKAGE_DIR / "source" / "normal.exr"),
            "normal_condition": copy_asset(Path(phase0e["inputs"]["files"]["normal_condition"]["path"]), PACKAGE_DIR / "source" / "normal_condition.png"),
        },
        "masks": {
            "glass": copy_asset(Path(phase0g["outputs"]["glass_mask"]), PACKAGE_DIR / "masks" / "glass.png"),
            "exterior_context": copy_asset(Path(phase0h["exterior_context"]["mask"]), PACKAGE_DIR / "masks" / "exterior_context.png"),
        },
        "context": {
            "exterior_context": copy_asset(Path(phase0h["exterior_context"]["panorama"]), PACKAGE_DIR / "context" / "exterior_context.png"),
        },
    }
    if visibility:
        assets["masks"]["material_id"] = copy_asset(Path(visibility["outputs"]["material_id"]), PACKAGE_DIR / "masks" / "material_id.png")
    else:
        assets["masks"]["material_id"] = {"status": "MISSING", "reason": "Run scripts/run_controlled_material_id.ps1 first."}
    assets["masks"]["geometry_edges"] = make_edges(Path(assets["source"]["rgb_erp"]["path"]), PACKAGE_DIR / "masks" / "geometry_edges.png")
    width, height = image_dims(Path(assets["source"]["rgb_erp"]["path"]))
    assets["masks"]["light_mask"] = make_empty_mask(
        PACKAGE_DIR / "masks" / "light_mask.png",
        width,
        height,
        "No reliable source-truth light fixture positions were available.",
    )

    materials = material_contract(material_manifest, phase0g, visibility)
    lights = light_contract(light_audit, scene_json, width, height)
    source_hashes = {
        "obj": sha256(OUT / "obj" / "scene.obj"),
        "scene_json": sha256(OUT / "obj" / "scene.json"),
        "rgb_erp": assets["source"]["rgb_erp"]["sha256"],
        "depth_exr": assets["source"]["depth_exr"]["sha256"],
        "normal_exr": assets["source"]["normal_exr"]["sha256"],
        "glass_mask": assets["masks"]["glass"]["sha256"],
        "exterior_context_mask": assets["masks"]["exterior_context"]["sha256"],
    }
    scene = scene_contract(scene_json, phase0d, phase0g, phase0h, source_hashes)
    metrics = source_metrics(assets, phase0d, phase0h)

    write_json(PACKAGE_DIR / "semantics" / "materials.json", materials)
    write_json(PACKAGE_DIR / "semantics" / "lights.json", lights)
    write_json(PACKAGE_DIR / "semantics" / "scene_contract.json", scene)
    write_json(PACKAGE_DIR / "qc" / "source_metrics.json", metrics)

    manifest = {
        "schema": "rad-ai360-controlled-condition-package",
        "version": 1,
        "pipeline": "CONTROLLED_CORE",
        "created_at_unix": int(time.time()),
        "source_truth_boundary": {
            "source_truth": ["scene.json camera", "OBJ geometry", "Blender RGB/depth/normal", "deterministic masks", "material manifest"],
            "semantic_interpretation": ["none in v0"],
            "ai_generation": ["not part of condition package"],
        },
        "assets": assets,
        "semantics": {
            "materials": "semantics/materials.json",
            "lights": "semantics/lights.json",
            "scene_contract": "semantics/scene_contract.json",
        },
        "qc": {"source_metrics": "qc/source_metrics.json"},
        "historical_inputs": {
            "phase0d": str(OUT / "phase0d" / "phase0d_report.json"),
            "phase0e": str(OUT / "phase0e" / "phase0e_report.json"),
            "phase0g": str(OUT / "phase0g" / "phase0g_report.json"),
            "phase0h": str(OUT / "phase0h" / "phase0h_report.json"),
        },
    }
    write_json(PACKAGE_DIR / "manifest.json", manifest)
    print(f"RAD_CONTROLLED_PACKAGE={PACKAGE_DIR / 'manifest.json'}")


def prompt_from_package(intent: str) -> str:
    scene = read_json(PACKAGE_DIR / "semantics" / "scene_contract.json")
    materials = read_json(PACKAGE_DIR / "semantics" / "materials.json")
    lights = read_json(PACKAGE_DIR / "semantics" / "lights.json")
    visible_glass = [r for r in scene["regions"]["known_exterior"]["regions"] if r["context_source"] == "SCENE_GEOMETRY"]
    visible_materials = [m for m in materials["materials"] if m["visible"]]
    material_lines = []
    for item in sorted(visible_materials, key=lambda x: x["visible_pixel_coverage"], reverse=True)[:18]:
        material_lines.append(
            f"- {item['id']} {item['name']}: preserve base color {item['base_color']} and opacity {item['opacity']}"
        )
    light_line = (
        "- No reliable source-truth fixture positions are available; do not invent arbitrary new luminaires."
        if not lights["lights"] else
        "- Preserve known fixture positions from lights.json."
    )
    exterior_line = (
        f"- Preserve {len(visible_glass)} glazing regions with deterministic scene geometry visible behind them; do not replace known exterior/context pixels with unrelated scenery."
    )
    sections = [
        "CONTROLLED AI360 GENERATION REQUEST",
        "",
        f"INTENT: {intent}",
        "",
        "PRESERVE:",
        "- Exact equirectangular 2:1 panorama projection and camera composition.",
        "- Architecture, openings, wall/ceiling/floor geometry, cabinetry proportions, fixed furniture placement, and visible fixture count.",
        "- Do not redesign the interior, add furniture, remove objects, move objects, or change room proportions.",
        "",
        "MATERIAL:",
        "- Preserve source material colors, visible material regions, glazing regions, and texture intent.",
        *material_lines,
        "",
        "LIGHTING:",
        light_line,
        "- Improve photographic illumination and reflections without changing source design intent.",
        "",
        "EXTERIOR:",
        exterior_line,
        "- Treat UNKNOWN glazing separately from known exterior/context regions.",
        "",
        "REALISM:",
        "- Improve realistic material response, microtexture, exposure, subtle reflections, and believable architectural photography.",
        "- Preserve the 0/360 panorama seam.",
    ]
    return "\n".join(sections)


def run_flux_job(label: str, source_path: Path, prompt: str, out_dir: Path, args: argparse.Namespace, package_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    exp_dir = out_dir / label
    input_dir = exp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_copy = input_dir / "rgb_erp.png"
    source_copy.write_bytes(source_path.read_bytes())
    source_width, source_height = flux.source_dimensions(source_copy)
    output_width, output_height = flux.output_size_for_source(source_width, source_height)
    reference_path = input_dir / "rgb_erp_cloudflare_reference_511.png"
    reference = flux.resize_reference_to_limit(source_copy, reference_path)
    output_path = exp_dir / "output.png"
    generation_path = exp_dir / "generation.json"

    account_id, api_token, credential_source = flux.load_cloudflare_config()
    available_controls = (
        ["depth", "normal", "glass_mask", "exterior_context", "material_ids", "geometry_edges", "light_positions"]
        if package_manifest else ["rgb"]
    )
    request_parameters = {
        "provider": "cloudflare",
        "model": args.model,
        "endpoint": "https://api.cloudflare.com/client/v4/accounts/<redacted-account-id>/ai/run/" + args.model,
        "credential_source": credential_source,
        "multipart_fields": {
            "prompt": prompt,
            "width": str(output_width),
            "height": str(output_height),
            "guidance": str(args.guidance),
        },
        "multipart_files": {
            "input_image_0": {
                "filename": reference_path.name,
                "mime_type": "image/png",
                "source": "RGB ERP ordinary image reference",
                "sha256": reference["sha256"],
            }
        },
        "provider_control_limit": "Current project Cloudflare FLUX.2 Klein path supports ordinary image reference upload; no verified depth/normal/mask structural control.",
    }
    generation = {
        "schema": "rad-ai360-controlled-generation-job",
        "version": 1,
        "label": label,
        "status": "PENDING",
        "provider": "cloudflare",
        "model": args.model,
        "pipeline": "FAST_BASELINE" if package_manifest is None else "CONTROLLED_CORE",
        "prompt": prompt,
        "negative_prompt": None,
        "condition_package": str(PACKAGE_DIR / "manifest.json") if package_manifest else None,
        "available_controls": available_controls,
        "actually_consumed": ["rgb", "prompt"],
        "not_consumed_by_provider": [item for item in available_controls if item not in {"rgb"}],
        "input_image_hashes": {
            "rgb_erp": sha256(source_copy),
            "cloudflare_reference_511": reference["sha256"],
        },
        "seed": None,
        "strength_or_denoise": None,
        "control_weights": None,
        "resolution": [output_width, output_height],
        "request_parameters": request_parameters,
        "provider_request_id": None,
        "runtime_seconds": None,
        "estimated_or_reported_cost": None,
    }
    started = time.time()
    try:
        if not account_id or not api_token:
            raise RuntimeError("Cloudflare credentials are missing.")
        endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{args.model}"
        payload, headers, runtime = flux.multipart_post(
            endpoint=endpoint,
            token=api_token,
            fields=request_parameters["multipart_fields"],
            files={"input_image_0": (reference_path.name, reference_path.read_bytes(), "image/png")},
            timeout=args.timeout,
        )
        flux.decode_cloudflare_image(payload, output_path)
        generation.update({
            "status": "ok",
            "output": str(output_path),
            "output_sha256": sha256(output_path),
            "provider_request_id": headers.get("cf-ray"),
            "runtime_seconds": round(runtime, 3),
            "cloudflare_success": payload.get("success"),
            "cloudflare_errors": payload.get("errors"),
            "cloudflare_messages": payload.get("messages"),
        })
    except Exception as error:
        generation.update({"status": "FAIL", "error": str(error), "output": str(output_path)})
    generation["total_seconds"] = round(time.time() - started, 3)
    write_json(generation_path, generation)
    return generation


def mask_mean_delta(source: Path, output: Path, mask: Path) -> dict[str, Any]:
    if not output.is_file():
        return {"status": "NOT_RUN"}
    src = cv2.imread(str(source), cv2.IMREAD_COLOR)
    out = cv2.imread(str(output), cv2.IMREAD_COLOR)
    m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if src is None or out is None or m is None:
        return {"status": "UNAVAILABLE"}
    if out.shape[:2] != src.shape[:2]:
        out = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    active = m > 127
    if not np.any(active):
        return {"status": "UNKNOWN", "active_pixels": 0}
    diff = np.abs(src.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
    return {
        "status": "WARNING_SIGNAL_ONLY",
        "active_pixels": int(np.count_nonzero(active)),
        "mean_rgb_delta_0_1": round(float(diff[active].mean()) / 255.0, 6),
    }


def qc_generation(source: Path, output: Path, package: dict[str, Any] | None) -> dict[str, Any]:
    output_qc = flux.validate_png(output) if output.is_file() else None
    seam_status = "UNKNOWN"
    if output_qc:
        seam_status = "PASS" if output_qc["mean_rgb_seam_delta"] < 0.15 else "WARNING"
    qc = {
        "geometry": "WARNING",
        "camera": "PASS" if output_qc and output_qc["ratio_2_to_1"] else "FAIL",
        "projection": "PASS" if output_qc and output_qc["ratio_2_to_1"] else "FAIL",
        "seam": seam_status,
        "glazing": "UNKNOWN",
        "exterior_context": "UNKNOWN",
        "material_regions": "UNKNOWN",
        "lighting_positions": "UNKNOWN",
        "output_validity": output_qc,
        "edge_similarity": flux.edge_similarity(source, output) if output.is_file() else None,
    }
    if package and output.is_file():
        qc["glazing"] = "WARNING"
        qc["exterior_context"] = "WARNING"
        qc["material_regions"] = "WARNING"
        qc["lighting_positions"] = "UNKNOWN"
        qc["mask_checks"] = {
            "glass": mask_mean_delta(source, output, PACKAGE_DIR / "masks" / "glass.png"),
            "known_exterior": mask_mean_delta(source, output, PACKAGE_DIR / "masks" / "exterior_context.png"),
        }
    return qc


def run_bakeoff(args: argparse.Namespace) -> None:
    package_manifest = read_json(PACKAGE_DIR / "manifest.json")
    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    (BAKEOFF_DIR / "comparison").mkdir(parents=True, exist_ok=True)
    source_rgb = PACKAGE_DIR / "source" / "rgb_erp.png"
    baseline_prompt = flux.PROMPT
    controlled_prompt = prompt_from_package(args.intent)
    write_json(BAKEOFF_DIR / "B1_controlled_harness" / "compiled_prompt.json", {"prompt": controlled_prompt})

    b0 = run_flux_job("B0_rgb_baseline", source_rgb, baseline_prompt, BAKEOFF_DIR, args, None)
    b1 = run_flux_job("B1_controlled_harness", source_rgb, controlled_prompt, BAKEOFF_DIR, args, package_manifest)
    b0_output = Path(b0.get("output", ""))
    b1_output = Path(b1.get("output", ""))
    report = {
        "schema": "rad-ai360-controlled-bakeoff",
        "version": 1,
        "status": "COMPLETE" if b0["status"] == "ok" and b1["status"] == "ok" else "WARNING",
        "purpose": "Compare same Cloudflare model/source/settings with RGB-only baseline vs Controlled Core prompt/package/QC harness.",
        "controlled_variables": {
            "provider": "cloudflare",
            "model": args.model,
            "source_rgb": str(source_rgb),
            "guidance": args.guidance,
            "seed": None,
            "note": "Cloudflare path does not expose deterministic seed in current implementation.",
        },
        "B0_rgb_baseline": b0,
        "B1_controlled_harness": b1,
        "qc": {
            "B0": qc_generation(source_rgb, b0_output, None),
            "B1": qc_generation(source_rgb, b1_output, package_manifest),
        },
        "skeptical_result": "Deterministic v0 QC can verify projection/seam/mask-region warning signals and metadata completeness, but cannot prove semantic fidelity improvement without human/semantic QC. Provider still consumes RGB and prompt only.",
    }
    write_json(BAKEOFF_DIR / "report.json", report)
    print(f"RAD_CONTROLLED_BAKEOFF={BAKEOFF_DIR / 'report.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-package")
    bakeoff = sub.add_parser("run-bakeoff")
    bakeoff.add_argument("--model", default=flux.MODEL)
    bakeoff.add_argument("--guidance", type=float, default=3.5)
    bakeoff.add_argument("--timeout", type=int, default=300)
    bakeoff.add_argument("--intent", default="Photorealistic architectural interior panorama faithful to the SketchUp design.")
    args = parser.parse_args()
    if args.command == "build-package":
        build_package(args)
    elif args.command == "run-bakeoff":
        run_bakeoff(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
