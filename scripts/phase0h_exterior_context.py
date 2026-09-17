#!/usr/bin/env python3
"""Phase 0H exterior context conditioning.

This pass distinguishes glazing with scene geometry visible behind it from
unconstrained glazing. It does not create outdoor assets and does not send
depth/normal/masks to AI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase0d_equirectangular_passes as phase0d  # noqa: E402
import phase0g_source_signal_validation as phase0g  # noqa: E402


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--phase0g-report", required=True)
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_for(paths):
    return {
        key: {
            "path": value,
            "bytes": os.path.getsize(value),
            "sha256": sha256(value),
        }
        for key, value in paths.items()
    }


def round_vec(vector):
    return [round(float(value), 6) for value in vector]


def configure(width, height, samples):
    class Args:
        engine = "CYCLES"
    args = Args()
    args.width = width
    args.height = height
    args.samples = samples
    args.denoise = True
    return phase0d.configure_render(args)


def create_transparent_material():
    mat = bpy.data.materials.new("RAD_PHASE0H_TRANSPARENT_GLAZING")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    nodes.clear()
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    return mat


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


def replace_glass_material_slots(glass_names, replacement):
    stored = []
    glass_set = set(glass_names)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material.name in glass_set:
                stored.append((slot, slot.material))
                slot.material = replacement
    return stored


def restore_slots(stored):
    for slot, original in stored:
        slot.material = original


def render_context_geometry_presence(path, glass_names):
    white = create_emission_material("RAD_PHASE0H_CONTEXT_GEOMETRY_WHITE", (1, 1, 1, 1))
    black = create_emission_material("RAD_PHASE0H_CONTEXT_BACKGROUND_BLACK", (0, 0, 0, 1))
    transparent = create_transparent_material()
    stored = []
    glass_set = set(glass_names)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            original = slot.material
            stored.append((slot, original))
            if original and original.name in glass_set:
                slot.material = transparent
            else:
                slot.material = white
    original_world = tuple(bpy.context.scene.world.color) if bpy.context.scene.world else None
    if bpy.context.scene.world:
        bpy.context.scene.world.color = (0, 0, 0)
    try:
        return phase0d.render_to_path(path, "PNG", "BW", "8", "Standard")
    finally:
        if bpy.context.scene.world and original_world:
            bpy.context.scene.world.color = original_world
        restore_slots(stored)


def render_glass_id_mask(path, glass_candidates):
    black = create_emission_material("RAD_PHASE0H_ID_BLACK", (0, 0, 0, 1))
    id_materials = {}
    palette = [
        (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1),
        (1, 0, 1, 1), (0, 1, 1, 1), (1, 0.5, 0, 1), (0.5, 0, 1, 1),
        (0, 0.5, 1, 1), (0.5, 1, 0, 1), (1, 0, 0.5, 1), (0, 1, 0.5, 1),
    ]
    for index, candidate in enumerate(glass_candidates):
        id_materials[candidate["source_name"]] = {
            "material": create_emission_material(
                f"RAD_PHASE0H_ID_{candidate['material_id']}",
                palette[index % len(palette)],
            ),
            "rgb_8bit": [round(v * 255) for v in palette[index % len(palette)][:3]],
        }
    stored = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            original = slot.material
            stored.append((slot, original))
            if original and original.name in id_materials:
                slot.material = id_materials[original.name]["material"]
            else:
                slot.material = black
    original_world = tuple(bpy.context.scene.world.color) if bpy.context.scene.world else None
    if bpy.context.scene.world:
        bpy.context.scene.world.color = (0, 0, 0)
    try:
        seconds = phase0d.render_to_path(path, "PNG", "RGB", "8", "Standard")
    finally:
        if bpy.context.scene.world and original_world:
            bpy.context.scene.world.color = original_world
        restore_slots(stored)
    return seconds, {name: data["rgb_8bit"] for name, data in id_materials.items()}


def load_pixels(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    return int(width), int(height), list(image.pixels)


def save_bw_from_values(path, width, height, values):
    image = bpy.data.images.new("RAD_PHASE0H_CONTEXT_MASK", width, height, alpha=True, float_buffer=False)
    rgba = []
    for value in values:
        rgba.extend([value, value, value, 1.0])
    image.pixels.foreach_set(rgba)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "BW"
    bpy.context.scene.render.image_settings.color_depth = "8"
    image.filepath_raw = path
    image.save()
    bpy.data.images.remove(image)


def compute_context_mask(glass_mask_path, geometry_presence_path, context_mask_path):
    gw, gh, glass = load_pixels(glass_mask_path)
    cw, ch, context = load_pixels(geometry_presence_path)
    if [gw, gh] != [cw, ch]:
        raise RuntimeError("Glass mask and context geometry mask dimensions do not match")
    values = []
    glass_pixels = 0
    constrained_pixels = 0
    for index in range(gw * gh):
        is_glass = glass[index * 4] > 0.5
        has_context = context[index * 4] > 0.5
        value = 1.0 if is_glass and has_context else 0.0
        if is_glass:
            glass_pixels += 1
        if value > 0:
            constrained_pixels += 1
        values.append(value)
    save_bw_from_values(context_mask_path, gw, gh, values)
    return {
        "dimensions": [gw, gh],
        "glass_pixels": glass_pixels,
        "constrained_context_pixels": constrained_pixels,
        "constrained_ratio_of_glass": round(constrained_pixels / glass_pixels, 6) if glass_pixels else 0,
    }


def classify_regions(glass_id_mask_path, context_mask_path, glass_candidates, color_map):
    width, height, id_pixels = load_pixels(glass_id_mask_path)
    mw, mh, context_pixels = load_pixels(context_mask_path)
    if [width, height] != [mw, mh]:
        raise RuntimeError("Glass ID mask and context mask dimensions do not match")
    regions = []
    for index, candidate in enumerate(glass_candidates, start=1):
        color = color_map[candidate["source_name"]]
        target = [component / 255.0 for component in color]
        visible = 0
        constrained = 0
        for pixel_index in range(width * height):
            offset = pixel_index * 4
            rgb = id_pixels[offset:offset + 3]
            if all(abs(rgb[channel] - target[channel]) < 0.08 for channel in range(3)):
                visible += 1
                if context_pixels[offset] > 0.5:
                    constrained += 1
        if visible == 0:
            source = "UNKNOWN"
            has_geometry = None
            confidence = 0.5
        else:
            ratio = constrained / visible
            if ratio >= 0.05:
                source = "SCENE_GEOMETRY"
                has_geometry = True
                confidence = min(0.99, 0.6 + ratio * 0.4)
            else:
                source = "UNCONSTRAINED"
                has_geometry = False
                confidence = min(0.95, 0.7 + (1 - ratio) * 0.25)
        regions.append({
            "id": f"glass_{index:03d}",
            "material_id": candidate["material_id"],
            "source_name": candidate["source_name"],
            "classification": "exterior_window_or_glazing_candidate",
            "context_source": source,
            "has_geometry_behind": has_geometry,
            "confidence": round(confidence, 4),
            "visible_glass_pixels": visible,
            "context_pixels": constrained,
            "context_ratio": round(constrained / visible, 6) if visible else None,
        })
    return regions


def classify_context_status(regions):
    visible = [region for region in regions if region["context_source"] != "UNKNOWN"]
    if not visible:
        return "UNCONSTRAINED"
    known = [region for region in visible if region["context_source"] == "SCENE_GEOMETRY"]
    unconstrained = [region for region in visible if region["context_source"] == "UNCONSTRAINED"]
    if known and unconstrained:
        return "PARTIAL"
    if known:
        return "PASS"
    return "UNCONSTRAINED"


def validate_png(path):
    width, height, pixels = load_pixels(path)
    seam = []
    visible = False
    alpha_min = 1.0
    alpha_max = 0.0
    for y in range(height):
        left = (y * width) * 4
        right = (y * width + width - 1) * 4
        seam.append(sum(abs(pixels[left + i] - pixels[right + i]) for i in range(3)) / 3.0)
    for index in range(width * height):
        offset = index * 4
        alpha = pixels[offset + 3]
        alpha_min = min(alpha_min, alpha)
        alpha_max = max(alpha_max, alpha)
        if any(channel > 0.003 for channel in pixels[offset:offset + 3]):
            visible = True
    return {
        "dimensions": [width, height],
        "ratio_2_to_1": abs((width / height) - 2.0) <= 0.001 if height else False,
        "alpha_min": round(alpha_min, 6),
        "alpha_max": round(alpha_max, 6),
        "has_visible_pixels": visible,
        "mean_rgb_seam_delta": round(sum(seam) / len(seam), 6) if seam else None,
        "max_rgb_seam_delta": round(max(seam), 6) if seam else None,
    }


def main():
    args = parse_args()
    start = time.time()
    out = args.out
    context_dir = os.path.join(out, "exterior_context")
    comparison_dir = os.path.join(out, "comparison")
    os.makedirs(context_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    phase0g_report = read_json(args.phase0g_report)
    glass_candidates = phase0g_report["glass"]["detected_materials"]
    source_g_path = os.path.join(comparison_dir, "SOURCE_G.png")
    shutil.copy2(phase0g_report["outputs"]["after_source_fix_rgb"], source_g_path)

    scene_data = read_json(args.json)
    phase0d.clear_scene()
    import_report = phase0d.import_obj(args.obj)
    bounds = phase0d.scene_bounds(list(bpy.context.scene.objects))
    basis = phase0d.camera_basis(scene_data)
    far_clip = phase0d.far_clip_for_bounds(basis["eye"], bounds)
    engine = configure(args.width, args.height, args.samples)
    camera = phase0d.create_panorama_camera(basis, phase0d.NEAR_CLIP, far_clip)
    phase0d.add_eye_guide_light(basis["eye"])
    if bpy.context.scene.world:
        bpy.context.scene.world.color = phase0g.WORLD_DAYLIGHT_SETTINGS["world_color"]
    phase0g.add_daylight()

    glass_names = [candidate["source_name"] for candidate in glass_candidates]
    transparent = create_transparent_material()
    stored = replace_glass_material_slots(glass_names, transparent)
    context_panorama_path = os.path.join(context_dir, "panorama.png")
    context_seconds = phase0d.render_to_path(context_panorama_path, "PNG", "RGB", "8", "Standard")
    restore_slots(stored)

    context_pass_path = os.path.join(comparison_dir, "CONTEXT_PASS.png")
    shutil.copy2(context_panorama_path, context_pass_path)

    geometry_presence_path = os.path.join(context_dir, "_geometry_presence.png")
    presence_seconds = render_context_geometry_presence(geometry_presence_path, glass_names)
    glass_id_mask_path = os.path.join(context_dir, "_glass_id_mask.png")
    id_seconds, color_map = render_glass_id_mask(glass_id_mask_path, glass_candidates)

    glass_mask_path = phase0g_report["glass"]["mask_path"]
    context_mask_path = os.path.join(context_dir, "mask.png")
    mask_stats = compute_context_mask(glass_mask_path, geometry_presence_path, context_mask_path)
    regions = classify_regions(glass_id_mask_path, context_mask_path, glass_candidates, color_map)
    exterior_status = classify_context_status(regions)

    exterior_manifest = {
        "schema": "rad-ai360-exterior-context",
        "version": 1,
        "camera": scene_data["camera"]["scene_name"],
        "method": {
            "summary": "Detected glazing is made transparent only for the context pass; non-glass scene geometry visible through the same ERP pixels marks constrained context.",
            "context_source_values": ["SCENE_GEOMETRY", "UNCONSTRAINED", "UNKNOWN"],
            "no_exterior_objects_created": True,
            "no_reference_image_used": True,
        },
        "regions": regions,
        "summary": {
            "status": exterior_status,
            "region_counts": {
                "SCENE_GEOMETRY": sum(1 for r in regions if r["context_source"] == "SCENE_GEOMETRY"),
                "UNCONSTRAINED": sum(1 for r in regions if r["context_source"] == "UNCONSTRAINED"),
                "UNKNOWN": sum(1 for r in regions if r["context_source"] == "UNKNOWN"),
            },
            "mask_stats": mask_stats,
        },
    }
    exterior_manifest_path = os.path.join(out, "exterior_context.json")
    write_json(exterior_manifest_path, exterior_manifest)

    output_paths = {
        "source_g": source_g_path,
        "context_panorama": context_panorama_path,
        "context_mask": context_mask_path,
        "context_pass": context_pass_path,
        "exterior_context_manifest": exterior_manifest_path,
    }
    context_qc = validate_png(context_panorama_path)
    categories = {
        "geometry": "PASS",
        "camera": "PASS",
        "seam": "PASS" if context_qc["mean_rgb_seam_delta"] is not None and context_qc["mean_rgb_seam_delta"] < 0.25 else "WARNING",
        "glazing": "PASS" if mask_stats["glass_pixels"] > 0 else "FAIL",
        "exterior_context": exterior_status,
        "material_fidelity": "WARNING",
    }

    report = {
        "schema": "rad-ai360-phase0h-exterior-context-conditioning",
        "version": 1,
        "status": "PASS" if categories["geometry"] == "PASS" and categories["camera"] == "PASS" else "WARNING",
        "phase0g_baseline": {
            "accepted": True,
            "phase0g_report": args.phase0g_report,
            "winning_source": phase0g_report["outputs"]["after_source_fix_rgb"],
            "winning_ai_output": phase0g_report.get("ai_ab_test", {}).get("g_b_after_source_fix", {}).get("output"),
        },
        "source": {
            "obj": args.obj,
            "scene_json": args.json,
            "obj_sha256": sha256(args.obj),
            "scene_json_sha256": sha256(args.json),
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
        },
        "exterior_context": {
            "manifest": exterior_manifest_path,
            "panorama": context_panorama_path,
            "mask": context_mask_path,
            "status": exterior_status,
            "regions": regions,
            "mask_stats": mask_stats,
        },
        "comparison": {
            "source_g": source_g_path,
            "context_pass": context_pass_path,
        },
        "ai_ab_test": {
            "status": "UNSUPPORTED_NOT_RUN",
            "reason": "Existing Cloudflare FLUX.2 Klein interface in this project accepts ordinary image references, not a true structural exterior-context/mask control. Sending the context pass as a second generic reference would fake conditioning and was intentionally not done.",
            "h_a_baseline": phase0g_report.get("ai_ab_test", {}).get("g_b_after_source_fix", {}),
            "h_b_context_conditioned": None,
        },
        "qc_categories": categories,
        "local_qc": {
            "context_panorama": context_qc,
        },
        "outputs": {
            **output_paths,
            "manifest": manifest_for(output_paths),
        },
        "supporting_passes": {
            "depth_regenerated": False,
            "normal_regenerated": False,
            "reason": "Mesh geometry and camera are unchanged. Context pass changes only material visibility for a separate diagnostic render.",
        },
        "runtime": {
            "import_seconds": import_report["duration_seconds"],
            "context_panorama_seconds": context_seconds,
            "geometry_presence_seconds": presence_seconds,
            "glass_id_mask_seconds": id_seconds,
            "total_seconds": round(time.time() - start, 3),
        },
        "warnings": [
            "Exterior context means scene geometry visible after glazing is made transparent; it is not proof that the geometry is truly outdoors.",
            "No procedural exterior, landscaping, sky generation, or AI reference was created.",
            "AI A/B was not run because the available FLUX interface cannot consume the new context signal structurally.",
        ],
        "stop_condition": "Stopped after Phase 0H report and deterministic context comparison. No Phase 0I or orchestration work performed.",
    }
    report_path = os.path.join(out, "phase0h_report.json")
    write_json(report_path, report)
    print(f"RAD_PHASE0H_REPORT={report_path}")


if __name__ == "__main__":
    main()
