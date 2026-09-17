#!/usr/bin/env python3
"""Controlled Core material-ID visibility pass.

This Blender pass renders one deterministic color per imported SketchUp/OBJ
material in the authoritative ERP camera, then measures visible pixel coverage.
It does not interpret material semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase0d_equirectangular_passes as phase0d  # noqa: E402


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--material-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args(argv[argv.index("--") + 1 :])


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def stable_rgb(index):
    # Deterministic 24-bit palette, avoiding black so background remains distinct.
    value = (index * 2654435761) & 0xFFFFFF
    r = ((value >> 16) & 255) or 1
    g = ((value >> 8) & 255) or 1
    b = (value & 255) or 1
    return [r, g, b]


def create_emission_material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def render_material_id(path, material_manifest):
    black = create_emission_material("RAD_CONTROLLED_MATERIAL_BACKGROUND", [0, 0, 0])
    colors = {}
    replacements = {}
    for index, record in enumerate(material_manifest["materials"], start=1):
        rgb = stable_rgb(index)
        colors[record["source_name"]] = {
            "material_id": record["id"],
            "rgb_8bit": rgb,
        }
        replacements[record["source_name"]] = create_emission_material(
            f"RAD_CONTROLLED_ID_{record['id']}",
            rgb,
        )

    stored = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            original = slot.material
            stored.append((slot, original))
            slot.material = replacements.get(original.name, black) if original else black

    original_world = tuple(bpy.context.scene.world.color) if bpy.context.scene.world else None
    if bpy.context.scene.world:
        bpy.context.scene.world.color = (0, 0, 0)
    try:
        phase0d.render_to_path(path, "PNG", "RGB", "8", "Standard")
    finally:
        if bpy.context.scene.world and original_world:
            bpy.context.scene.world.color = original_world
        for slot, original in stored:
            slot.material = original
    return colors


def load_pixels(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    bpy.data.images.remove(image)
    return int(width), int(height), pixels


def measure_coverage(path, material_manifest, color_map):
    width, height, pixels = load_pixels(path)
    rgba = np.asarray(pixels, dtype=np.float32).reshape((height, width, 4))
    rgb_image = np.clip(np.rint(rgba[:, :, :3] * 255.0), 0, 255).astype(np.uint32)
    counts = {record["id"]: 0 for record in material_manifest["materials"]}
    names_by_id = {record["id"]: record["source_name"] for record in material_manifest["materials"]}
    packed = (
        rgb_image[:, :, 0].astype(np.uint32) << 16
        | rgb_image[:, :, 1].astype(np.uint32) << 8
        | rgb_image[:, :, 2].astype(np.uint32)
    )
    unique_values, unique_counts = np.unique(packed, return_counts=True)
    palette_items = list(color_map.values())
    palette_rgb = np.asarray([item["rgb_8bit"] for item in palette_items], dtype=np.int16)
    palette_ids = [item["material_id"] for item in palette_items]

    # Cycles/color management can perturb flat ID colors by a few values, and
    # anti-aliased boundaries create blended colors. Count every non-black ID
    # pixel against the nearest deterministic palette entry.
    for value, count in zip(unique_values, unique_counts):
        value = int(value)
        rgb = np.asarray([(value >> 16) & 255, (value >> 8) & 255, value & 255], dtype=np.int16)
        if int(rgb.sum()) == 0:
            continue
        deltas = palette_rgb - rgb
        distances = np.einsum("ij,ij->i", deltas, deltas)
        nearest = int(np.argmin(distances))
        counts[palette_ids[nearest]] += int(count)

    total = width * height
    return {
        "dimensions": [width, height],
        "total_pixels": total,
        "materials": [
            {
                "id": material_id,
                "source_name": names_by_id[material_id],
                "visible": count > 0,
                "visible_pixels": count,
                "visible_pixel_coverage": round(count / total, 8),
            }
            for material_id, count in sorted(counts.items())
        ],
    }


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    scene_data = read_json(args.json)
    material_manifest = read_json(args.material_manifest)

    phase0d.clear_scene()
    import_report = phase0d.import_obj(args.obj)
    bounds = phase0d.scene_bounds(list(bpy.context.scene.objects))
    basis = phase0d.camera_basis(scene_data)
    far_clip = phase0d.far_clip_for_bounds(basis["eye"], bounds)

    class RenderArgs:
        engine = "CYCLES"
        width = args.width
        height = args.height
        samples = args.samples
        denoise = True

    engine = phase0d.configure_render(RenderArgs())
    phase0d.create_panorama_camera(basis, phase0d.NEAR_CLIP, far_clip)

    material_id_path = os.path.join(args.out, "material_id.png")
    color_map = render_material_id(material_id_path, material_manifest)
    coverage = measure_coverage(material_id_path, material_manifest, color_map)

    report = {
        "schema": "rad-ai360-controlled-material-visibility",
        "version": 1,
        "status": "PASS",
        "source": {
            "obj": args.obj,
            "scene_json": args.json,
            "material_manifest": args.material_manifest,
        },
        "render": {
            "engine": engine,
            "projection": "panoramic_equirectangular",
            "resolution": [args.width, args.height],
            "samples": args.samples,
        },
        "outputs": {
            "material_id": material_id_path,
            "color_map": color_map,
            "coverage": coverage,
        },
        "import": import_report,
    }
    report_path = os.path.join(args.out, "material_visibility.json")
    write_json(report_path, report)
    print(f"RAD_CONTROLLED_MATERIAL_VISIBILITY={report_path}")


if __name__ == "__main__":
    main()
