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
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACE_ORDER = ("front", "right", "back", "left", "top", "bottom")
PASS_ORDER = ("albedo", "ao", "depth_raw", "normal", "material_id", "object_id")
LABEL_PASSES = ("material_id", "object_id")
NEAR_CLIP = 0.01
AO_DISTANCE_M = 3.0
AO_SAMPLES = 16


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")
    parser = argparse.ArgumentParser(description="Fast deterministic Blender geometry pass generator.")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv[argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def import_obj(path: str) -> dict:
    start = time.time()
    options = {
        "filepath": path,
        "global_scale": 1.0,
        "forward_axis": "Y",
        "up_axis": "Z",
        "use_split_objects": True,
        "use_split_groups": True,
        "validate_meshes": True,
    }
    bpy.ops.wm.obj_import(**options)
    return {"options": options, "duration_seconds": round(time.time() - start, 3)}


def scene_basis(scene_data: dict) -> dict:
    eye = Vector(scene_data["camera"]["eye_mm"]) / 1000.0
    forward = Vector(scene_data["camera"]["direction"]).normalized()
    up = Vector(scene_data["camera"]["up"]).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    return {
        "eye": eye,
        "front": {"forward": forward, "up": up},
        "right": {"forward": right, "up": up},
        "back": {"forward": -forward, "up": up},
        "left": {"forward": -right, "up": up},
        "top": {"forward": up, "up": -forward},
        "bottom": {"forward": -up, "up": forward},
    }


def camera_matrix(eye: Vector, forward: Vector, up: Vector) -> Matrix:
    right = forward.cross(up).normalized()
    corrected_up = right.cross(forward).normalized()
    rotation = Matrix(((right.x, corrected_up.x, -forward.x),
                       (right.y, corrected_up.y, -forward.y),
                       (right.z, corrected_up.z, -forward.z))).to_4x4()
    return Matrix.Translation(eye) @ rotation


def configure_scene(resolution: int):
    scene = bpy.context.scene
    engines = [item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items]
    selected = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.engine = selected
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    # Dithering injects noise into 8-bit output, which destroys discrete ID passes.
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world.color = (0.0, 0.0, 0.0)
    # The Ambient Occlusion shader node needs EEVEE ray tracing to report real occlusion.
    eevee = getattr(scene, "eevee", None)
    ao_settings = {}
    for attribute, value in (("use_raytracing", True), ("use_gtao", True)):
        if eevee is not None and hasattr(eevee, attribute):
            setattr(eevee, attribute, value)
            ao_settings[attribute] = value
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    return selected, ao_settings


def label_render_settings(enabled: bool):
    """Toggle anti-aliasing off for discrete label passes and back on afterwards."""
    scene = bpy.context.scene
    scene.render.filter_size = 0.0 if enabled else 1.5
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 1 if enabled else 16



def create_camera(eye: Vector, resolution: int):
    data = bpy.data.cameras.new("RAD_FAST_PASS_CAMERA")
    camera = bpy.data.objects.new("RAD_FAST_PASS_CAMERA", data)
    bpy.context.collection.objects.link(camera)
    data.type = "PERSP"
    data.angle = math.radians(90.0)
    data.clip_start = NEAR_CLIP
    data.clip_end = 10000.0
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    return camera


def emission_material(name: str, color=(0.0, 0.0, 0.0, 1.0)):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def radial_depth_material(eye: Vector):
    material = emission_material("RAD_FAST_RADIAL_DEPTH")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    emission = next(node for node in nodes if node.bl_idname == "ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    distance = nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    eye_value = nodes.new("ShaderNodeCombineXYZ")
    eye_value.inputs["X"].default_value = eye.x
    eye_value.inputs["Y"].default_value = eye.y
    eye_value.inputs["Z"].default_value = eye.z
    links.new(geometry.outputs["Position"], distance.inputs[0])
    links.new(eye_value.outputs["Vector"], distance.inputs[1])
    links.new(distance.outputs["Value"], emission.inputs["Color"].links[0].to_socket if emission.inputs["Color"].is_linked else emission.inputs["Color"])
    return material


def normal_material():
    material = emission_material("RAD_FAST_WORLD_NORMAL")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    emission = next(node for node in nodes if node.bl_idname == "ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (1.0, 1.0, 1.0)
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 0.5
    links.new(geometry.outputs["Normal"], add.inputs[0])
    links.new(add.outputs["Vector"], scale.inputs[0])
    links.new(scale.outputs["Vector"], emission.inputs["Color"])
    return material


def ao_material():
    material = emission_material("RAD_FAST_AO")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    emission = next(node for node in nodes if node.bl_idname == "ShaderNodeEmission")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    # `samples` is a node property in current Blender, not an input socket.
    if hasattr(ao, "samples"):
        ao.samples = 16
    ao.only_local = True
    if "Distance" in ao.inputs:
        ao.inputs["Distance"].default_value = AO_DISTANCE_M
    links.new(ao.outputs["Color"], emission.inputs["Color"])
    return material


def material_color(index: int) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(f"material:{index}".encode()).digest()
    return tuple((digest[offset] / 255.0 * 0.8 + 0.2) for offset in (0, 1, 2)) + (1.0,)


def object_color(index: int) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(f"object:{index}".encode()).digest()
    return tuple((digest[offset] / 255.0 * 0.8 + 0.2) for offset in (0, 1, 2)) + (1.0,)


def albedo_materials() -> dict:
    result = {}
    for source in sorted((m for m in bpy.data.materials if m.name != "RAD_FAST_RADIAL_DEPTH"), key=lambda item: item.name):
        base_color = source.diffuse_color
        principled = None
        if source.use_nodes:
            principled = next(
                (node for node in source.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"),
                None,
            )
            if principled is not None:
                base_color = principled.inputs["Base Color"].default_value

        material = emission_material(f"RAD_FAST_ALBEDO_{source.name}", base_color)
        if principled is None or not principled.inputs["Base Color"].is_linked:
            result[source.name] = material
            continue

        source_socket = principled.inputs["Base Color"].links[0].from_socket
        source_node = source_socket.node
        image = getattr(source_node, "image", None)
        if source_node.bl_idname != "ShaderNodeTexImage" or image is None or not image.has_data:
            result[source.name] = material
            continue

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        emission = next(node for node in nodes if node.bl_idname == "ShaderNodeEmission")
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        links.new(texture.outputs["Color"], emission.inputs["Color"])
        result[source.name] = material
    return result


def replace_materials(factory):
    stored = []
    for obj in sorted((item for item in bpy.context.scene.objects if item.type == "MESH"), key=lambda item: item.name):
        original = [slot.material for slot in obj.material_slots]
        stored.append((obj, original))
        for slot_index, slot in enumerate(obj.material_slots):
            slot.material = factory(obj, slot_index, slot.material)
    return stored


def restore_materials(stored):
    for obj, materials in stored:
        for index, material in enumerate(materials):
            obj.material_slots[index].material = material


def render(path: Path, file_format: str, color_mode: str, color_depth: str):
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = color_depth
    if path.exists():
        path.unlink()
    bpy.ops.render.render(write_still=True)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Render output missing: {path}")


def save_float_sidecar(path: Path, sidecar_path: Path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    values = [pixels[index] for index in range(0, width * height * 4, 4)]
    import numpy as np

    np.save(sidecar_path, np.flipud(np.asarray(values, dtype=np.float32).reshape((height, width))))
    bpy.data.images.remove(image)


def run(args):
    start = time.time()
    out = Path(args.out)
    for pass_name in PASS_ORDER:
        (out / "cubemap" / pass_name).mkdir(parents=True, exist_ok=True)
    scene_data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    clear_scene()
    import_report = import_obj(args.obj)
    engine, ao_settings = configure_scene(args.resolution)
    basis = scene_basis(scene_data)
    camera = create_camera(basis["eye"], args.resolution)
    albedo = albedo_materials()
    radial = radial_depth_material(basis["eye"])
    normal = normal_material()
    ao = ao_material()
    material_names = sorted({slot.material.name for obj in bpy.context.scene.objects if obj.type == "MESH" for slot in obj.material_slots if slot.material})
    material_indices = {name: index + 1 for index, name in enumerate(material_names)}
    objects = sorted((obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=lambda item: item.name)
    object_indices = {obj.name: index + 1 for index, obj in enumerate(objects)}
    timings = {"import_seconds": import_report["duration_seconds"], "faces": {}}

    # One datablock per ID, reused across all six faces. Creating a material per slot
    # per face leaks thousands of datablocks and slows every later render.
    id_material_cache: dict = {}

    def id_material(kind: str, index: int):
        key = (kind, index)
        if key not in id_material_cache:
            color = material_color(index) if kind == "material" else object_color(index)
            id_material_cache[key] = emission_material(f"RAD_{kind.upper()}_ID_{index}", color)
        return id_material_cache[key]

    for face in FACE_ORDER:
        camera.matrix_world = camera_matrix(basis["eye"], basis[face]["forward"], basis[face]["up"])
        face_start = time.time()
        stored = replace_materials(lambda obj, slot_index, source: albedo.get(source.name, source))
        render(out / "cubemap" / "albedo" / f"{face}.png", "PNG", "RGB", "8")
        restore_materials(stored)

        stored = replace_materials(lambda obj, slot_index, source: ao)
        render(out / "cubemap" / "ao" / f"{face}.png", "PNG", "BW", "8")
        restore_materials(stored)

        stored = replace_materials(lambda obj, slot_index, source: radial)
        depth_path = out / "cubemap" / "depth_raw" / f"{face}.exr"
        render(depth_path, "OPEN_EXR", "BW", "32")
        save_float_sidecar(depth_path, depth_path.with_suffix(".npy"))
        restore_materials(stored)

        stored = replace_materials(lambda obj, slot_index, source: normal)
        render(out / "cubemap" / "normal" / f"{face}.png", "PNG", "RGB", "8")
        restore_materials(stored)

        label_render_settings(True)
        stored = replace_materials(lambda obj, slot_index, source: id_material("material", material_indices.get(source.name, 0) if source else 0))
        render(out / "cubemap" / "material_id" / f"{face}.png", "PNG", "RGB", "8")
        restore_materials(stored)

        stored = replace_materials(lambda obj, slot_index, source: id_material("object", object_indices[obj.name]))
        render(out / "cubemap" / "object_id" / f"{face}.png", "PNG", "RGB", "8")
        restore_materials(stored)
        label_render_settings(False)
        timings["faces"][face] = round(time.time() - face_start, 3)

    report = {
        "schema": "rad-ai360-fast-passes",
        "status": "PASS",
        "legacy_paths": {"phase0c": "LEGACY / R&D", "phase0d": "LEGACY / R&D"},
        "engine": engine,
        "lighting": {"lights_created": False, "lighting_dependent": False},
        "projection": {"face_order": FACE_ORDER, "contract_source": "backend/projection.py", "face_size": args.resolution},
        "camera_basis": {face: {key: [round(float(value), 6) for value in basis[face][key]] for key in ("forward", "up")} for face in FACE_ORDER},
        "depth": {"raw_representation": "world-position to camera-position Euclidean distance", "units": "metres", "conditioning": "backend.pass_pipeline.radial_depth_condition"},
        "ao": {
            "representation": "lighting-independent Eevee Ambient Occlusion node factor",
            "encoding": "8-bit grayscale; white=open, dark=occluded",
            "distance_m": AO_DISTANCE_M,
            "samples": AO_SAMPLES,
            "only_local": True,
            "engine_settings": ao_settings,
        },
        "normal": {"space": "world", "encoding": "RGB = normal * 0.5 + 0.5"},
        "label_passes": {
            "passes": LABEL_PASSES,
            "anti_aliasing": "disabled during label renders (filter_size=0, taa_render_samples=1)",
            "dither_intensity": 0.0,
            "stitch_interpolation": "nearest",
            "material_index_count": len(material_indices),
            "object_index_count": len(object_indices),
        },
        "ids": {"material_strategy": "stable sorted material-name IDs", "object_strategy": "stable sorted imported-object-name IDs", "object_identity_limit": "OBJ import only preserves identities exposed by imported object names/groups"},
        "materials": {"source_material_count": len(material_names), "albedo_fallback": "diffuse_color when no node-based Principled Base Color exists"},
        "timings": timings,
        "duration_seconds": round(time.time() - start, 3),
    }
    report_path = out / "pass_generation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    index_path = out / "id_index.json"
    index_path.write_text(
        json.dumps({"material_indices": material_indices, "object_indices": object_indices}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RAD_FAST_PASS_REPORT={report_path}")


if __name__ == "__main__":
    run(parse_args())