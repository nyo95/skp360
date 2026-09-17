#!/usr/bin/env python3
"""Phase 0G source signal validation for AI-ready ERP renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase0d_equirectangular_passes as phase0d  # noqa: E402


GLASS_KEYWORDS = ("glass", "glazing", "translucent", "transparent")
LIGHT_KEYWORDS = ("light", "lamp", "downlight", "spot", "spotlight", "pendant", "led", "track", "ies")

GLASS_SHADER_SETTINGS = {
    "method": "generic_architectural_glazing_signal",
    "alpha": 0.22,
    "roughness": 0.08,
    "metallic": 0.0,
    "transmission_weight_if_available": 0.65,
    "ior_if_available": 1.45,
    "preserve_source_tint": True,
}

WORLD_DAYLIGHT_SETTINGS = {
    "world_color": [0.86, 0.90, 0.96],
    "sun_energy": 1.4,
    "sun_rotation_degrees": [50.0, 0.0, -35.0],
    "intent": "neutral daylight signal, not production lighting",
}

LIGHT_DEFAULTS = {
    "downlight": {"type": "POINT", "power": 60.0, "color": [1.0, 0.93, 0.84]},
    "generic_fixture": {"type": "POINT", "power": 35.0, "color": [1.0, 0.93, 0.84]},
}


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--phase0d-report", required=True)
    parser.add_argument("--material-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv[argv.index("--") + 1 :])


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_vec(vector):
    return [round(float(value), 6) for value in vector]


def material_by_name():
    return {mat.name: mat for mat in bpy.data.materials}


def get_alpha(material):
    alpha = None
    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                socket = node.inputs.get("Alpha")
                if socket:
                    alpha = float(socket.default_value)
                break
    if alpha is None:
        alpha = float(material.diffuse_color[3])
    return alpha


def detect_glass_candidates(manifest):
    materials = material_by_name()
    candidates = []
    for record in manifest["materials"]:
        source_name = record["source_name"]
        mat = materials.get(source_name)
        source_opacity = float(record.get("opacity", get_alpha(mat) if mat else 1.0))
        lower = source_name.lower()
        reasons = []
        if any(keyword in lower for keyword in GLASS_KEYWORDS):
            reasons.append("source material name indicates glass/translucency")
        if source_opacity < 0.999:
            reasons.append("source opacity < 1.0")
        if reasons:
            candidates.append({
                "material_id": record["id"],
                "source_name": source_name,
                "source_opacity": round(source_opacity, 6),
                "source_color": record.get("base_color"),
                "texture": record.get("texture"),
                "reason_detected": reasons,
            })
    return candidates


def set_input_if_exists(node, names, value):
    for name in names:
        socket = node.inputs.get(name)
        if socket:
            socket.default_value = value
            return True
    return False


def apply_glass_override(candidates):
    applied = []
    for candidate in candidates:
        mat = bpy.data.materials.get(candidate["source_name"])
        if not mat:
            continue
        tint = candidate.get("source_color") or [0.75, 0.85, 0.9]
        tint_rgba = (float(tint[0]), float(tint[1]), float(tint[2]), GLASS_SHADER_SETTINGS["alpha"])
        mat.use_nodes = True
        mat.diffuse_color = tint_rgba
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True if hasattr(mat, "use_screen_refraction") else False
        bsdf = None
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    bsdf = node
                    break
        if bsdf:
            set_input_if_exists(bsdf, ("Base Color",), tint_rgba)
            set_input_if_exists(bsdf, ("Alpha",), GLASS_SHADER_SETTINGS["alpha"])
            set_input_if_exists(bsdf, ("Roughness",), GLASS_SHADER_SETTINGS["roughness"])
            set_input_if_exists(bsdf, ("Metallic",), GLASS_SHADER_SETTINGS["metallic"])
            set_input_if_exists(bsdf, ("Transmission Weight", "Transmission"), GLASS_SHADER_SETTINGS["transmission_weight_if_available"])
            set_input_if_exists(bsdf, ("IOR",), GLASS_SHADER_SETTINGS["ior_if_available"])
        applied.append({
            "material_id": candidate["material_id"],
            "source_name": candidate["source_name"],
            "applied": True,
            "shader_settings": GLASS_SHADER_SETTINGS,
        })
    return applied


def create_emission_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def render_material_mask(path, glass_names):
    white = create_emission_material("RAD_PHASE0G_MASK_WHITE", (1, 1, 1, 1))
    black = create_emission_material("RAD_PHASE0G_MASK_BLACK", (0, 0, 0, 1))
    stored = []
    glass_set = set(glass_names)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            original = slot.material
            stored.append((slot, original))
            slot.material = white if original and original.name in glass_set else black
    try:
        return phase0d.render_to_path(path, "PNG", "BW", "8", "Standard")
    finally:
        for slot, original in stored:
            slot.material = original


def configure_phase0g_render(width, height, samples):
    class Args:
        engine = "CYCLES"
    args = Args()
    args.width = width
    args.height = height
    args.samples = samples
    args.denoise = True
    actual = phase0d.configure_render(args)
    scene = bpy.context.scene
    return actual


def add_daylight():
    light_data = bpy.data.lights.new("RAD_PHASE0G_NEUTRAL_DAYLIGHT_SUN", "SUN")
    light_data.energy = WORLD_DAYLIGHT_SETTINGS["sun_energy"]
    light = bpy.data.objects.new("RAD_PHASE0G_NEUTRAL_DAYLIGHT_SUN", light_data)
    bpy.context.collection.objects.link(light)
    rx, ry, rz = [math.radians(v) for v in WORLD_DAYLIGHT_SETTINGS["sun_rotation_degrees"]]
    light.rotation_euler = (rx, ry, rz)
    return {"name": light.name, **WORLD_DAYLIGHT_SETTINGS}


def audit_lights(manifest):
    candidates = []
    for record in manifest["materials"]:
        lower = record["source_name"].lower()
        evidence = [f"material name contains {kw.upper()}" for kw in LIGHT_KEYWORDS if kw in lower]
        if not evidence:
            continue
        confidence = "low"
        classification = "fixture_material_candidate"
        reconstruct = False
        if "downlight" in lower or lower.strip() in {"light", "lights"}:
            confidence = "medium"
            classification = "downlight_or_light_fixture_candidate"
        candidates.append({
            "material_id": record["id"],
            "source_name": record["source_name"],
            "classification": classification,
            "confidence": confidence,
            "evidence": evidence,
            "reconstruct_light": reconstruct,
            "reason_not_reconstructed": "OBJ/material evidence is insufficient to recover fixture transform or emitter intent without risking invented lights.",
        })
    return {
        "method": "deterministic material-name audit only; OBJ contains no renderer light metadata.",
        "keywords": list(LIGHT_KEYWORDS),
        "candidates": candidates,
        "lights_reconstructed": [],
        "rejected_or_uncertain_candidates": candidates,
        "defaults": LIGHT_DEFAULTS,
    }


def exterior_detection(bounds):
    # Current OBJ import is usually one mesh object with no exterior semantic labels.
    # Report only defensible evidence: large scene extents exist, but usable exterior
    # context cannot be identified deterministically from OBJ names/materials.
    return {
        "status": "INDETERMINATE",
        "method": "OBJ object/material metadata inspection and scene bounds",
        "scene_bounds_m": bounds,
        "finding": "No deterministic exterior object names or metadata were available. Neutral daylight world is used; no invented landscape/backdrop was added.",
        "priority_applied": "neutral world/sky daylight",
    }


def validate_same_geometry(before_bounds, after_bounds):
    return {
        "status": "PASS" if before_bounds == after_bounds else "WARNING",
        "before_bounds_m": before_bounds,
        "after_bounds_m": after_bounds,
        "note": "Phase 0G changes materials/world/lights only and does not edit mesh geometry.",
    }


def main():
    args = parse_args()
    start = time.time()
    out = args.out
    comparison_dir = os.path.join(out, "comparison")
    mask_dir = os.path.join(out, "masks")
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    scene_data = read_json(args.json)
    phase0d_report = read_json(args.phase0d_report)
    manifest = read_json(args.material_manifest)

    phase0d.clear_scene()
    import_report = phase0d.import_obj(args.obj)
    bounds = phase0d.scene_bounds(list(bpy.context.scene.objects))
    bounds_report = {
        "min_m": round_vec(Vector(bounds["min_m"])),
        "max_m": round_vec(Vector(bounds["max_m"])),
        "size_m": round_vec(Vector(bounds["size_m"])),
    } if bounds else None

    basis = phase0d.camera_basis(scene_data)
    far_clip = phase0d.far_clip_for_bounds(basis["eye"], bounds)
    engine = configure_phase0g_render(args.width, args.height, args.samples)
    camera = phase0d.create_panorama_camera(basis, phase0d.NEAR_CLIP, far_clip)
    phase0d.add_eye_guide_light(basis["eye"])

    before_path = os.path.join(comparison_dir, "BEFORE.png")
    before_seconds = phase0d.render_to_path(before_path, "PNG", "RGB", "8", "Standard")

    glass_candidates = detect_glass_candidates(manifest)
    glass_candidates_path = os.path.join(out, "glass_candidates.json")
    write_json(glass_candidates_path, {
        "schema": "rad-ai360-phase0g-glass-candidates",
        "version": 1,
        "glass_candidates": glass_candidates,
    })

    glass_applied = apply_glass_override(glass_candidates)
    if bpy.context.scene.world:
        bpy.context.scene.world.color = WORLD_DAYLIGHT_SETTINGS["world_color"]
    daylight = add_daylight()
    light_audit = audit_lights(manifest)
    light_audit_path = os.path.join(out, "light_audit.json")
    write_json(light_audit_path, {
        "schema": "rad-ai360-phase0g-light-audit",
        "version": 1,
        **light_audit,
    })

    mask_path = os.path.join(mask_dir, "glass_mask.png")
    mask_seconds = render_material_mask(mask_path, [item["source_name"] for item in glass_candidates])

    after_path = os.path.join(comparison_dir, "AFTER_SOURCE_FIX.png")
    after_seconds = phase0d.render_to_path(after_path, "PNG", "RGB", "8", "Standard")

    after_bounds = phase0d.scene_bounds(list(bpy.context.scene.objects))
    after_bounds_report = {
        "min_m": round_vec(Vector(after_bounds["min_m"])),
        "max_m": round_vec(Vector(after_bounds["max_m"])),
        "size_m": round_vec(Vector(after_bounds["size_m"])),
    } if after_bounds else None

    outputs = {
        "before_rgb": before_path,
        "after_source_fix_rgb": after_path,
        "glass_mask": mask_path,
    }
    report = {
        "schema": "rad-ai360-phase0g-source-signal-validation",
        "version": 1,
        "status": "PASS",
        "source": {
            "obj": args.obj,
            "scene_json": args.json,
            "phase0d_report": args.phase0d_report,
            "material_manifest": args.material_manifest,
            "source_scene_hashes": {
                "obj_sha256": file_sha256(args.obj),
                "scene_json_sha256": file_sha256(args.json),
            },
        },
        "camera": {
            "scene_name": scene_data["camera"]["scene_name"],
            "eye_m": round_vec(basis["eye"]),
            "forward": round_vec(basis["forward"]),
            "up": round_vec(basis["up"]),
            "right": round_vec(basis["right"]),
            "projection": "panoramic_equirectangular",
            "resolution": [args.width, args.height],
            "camera_type": camera.data.type,
            "panorama_type": camera.data.panorama_type,
        },
        "render": {
            "engine": engine,
            "samples": args.samples,
            "denoise": True,
            "color_management": {
                "view_transform": bpy.context.scene.view_settings.view_transform,
                "look": bpy.context.scene.view_settings.look,
                "exposure": bpy.context.scene.view_settings.exposure,
                "gamma": bpy.context.scene.view_settings.gamma,
            },
        },
        "glass": {
            "candidate_count": len(glass_candidates),
            "glass_candidates_path": glass_candidates_path,
            "detected_materials": glass_candidates,
            "shader_settings": GLASS_SHADER_SETTINGS,
            "applied": glass_applied,
            "mask_path": mask_path,
        },
        "exterior_daylight": exterior_detection(bounds_report),
        "lighting": {
            "audit_path": light_audit_path,
            **light_audit,
        },
        "outputs": {
            "before_rgb": before_path,
            "after_source_fix_rgb": after_path,
            "glass_mask": mask_path,
            "manifest": {
                key: {
                    "path": value,
                    "bytes": os.path.getsize(value),
                    "sha256": file_sha256(value),
                }
                for key, value in outputs.items()
            },
        },
        "supporting_passes": {
            "depth_regenerated": False,
            "normal_regenerated": False,
            "reason": "No mesh geometry or camera change; Phase 0D depth and geometry-derived normal remain source-of-truth.",
            "phase0d_depth_sha256": phase0d_report.get("output_manifest", {}).get("depth", {}).get("sha256"),
            "phase0d_normal_sha256": phase0d_report.get("output_manifest", {}).get("normal", {}).get("sha256"),
        },
        "geometry_validation": validate_same_geometry(bounds_report, after_bounds_report),
        "render_runtime": {
            "import_seconds": import_report["duration_seconds"],
            "before_rgb_seconds": before_seconds,
            "glass_mask_seconds": mask_seconds,
            "after_source_fix_seconds": after_seconds,
            "total_seconds": round(time.time() - start, 3),
        },
        "warnings": [
            "No AI classification used.",
            "No new exterior geometry, landscaping, or photographic backdrop was added.",
            "No light was reconstructed because available OBJ evidence did not provide high-confidence emitter transforms.",
        ],
    }
    report_path = os.path.join(out, "phase0g_report.json")
    write_json(report_path, report)
    print(f"RAD_PHASE0G_REPORT={report_path}")


if __name__ == "__main__":
    main()
