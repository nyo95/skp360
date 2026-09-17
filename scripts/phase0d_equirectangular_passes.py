import argparse
import hashlib
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector


DEFAULT_WIDTH = 2048
DEFAULT_HEIGHT = 1024
NEAR_CLIP = 0.01
SKETCHUP_TO_BLENDER = Matrix.Identity(4)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Render native equirectangular RGB/depth/normal passes from scene.json camera."
    )
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--engine", default="CYCLES", choices=("CYCLES", "BLENDER_EEVEE"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--denoise", action="store_true")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def prepare_output_dirs(out_dir):
    for folder in ("rgb", "depth", "normal"):
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)


def import_obj(obj_path):
    start = time.time()
    kwargs = {
        "filepath": obj_path,
        "global_scale": 1.0,
        "forward_axis": "Y",
        "up_axis": "Z",
        "use_split_objects": True,
        "use_split_groups": False,
        "validate_meshes": True,
    }
    bpy.ops.wm.obj_import(**kwargs)
    return {
        "operator": "bpy.ops.wm.obj_import",
        "options": kwargs,
        "duration_seconds": round(time.time() - start, 3),
    }


def mm_to_meters(values):
    source = Vector((values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0))
    converted = SKETCHUP_TO_BLENDER @ source.to_4d()
    return Vector((converted.x, converted.y, converted.z))


def vector_to_blender(values):
    source = Vector(values).normalized()
    return (SKETCHUP_TO_BLENDER.to_3x3() @ source).normalized()


def camera_basis(scene_data):
    eye = mm_to_meters(scene_data["camera"]["eye_mm"])
    source_forward = vector_to_blender(scene_data["camera"]["direction"])
    source_up = vector_to_blender(scene_data["camera"]["up"])

    forward = source_forward.normalized()
    up = (source_up - forward * source_up.dot(forward)).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()

    return {
        "eye": eye,
        "forward": forward,
        "up": up,
        "right": right,
    }


def camera_matrix_from_vectors(eye, direction, up):
    forward = Vector(direction).normalized()
    up_vector = Vector(up).normalized()
    right = forward.cross(up_vector).normalized()
    corrected_up = right.cross(forward).normalized()
    rotation = Matrix(
        (
            (right.x, corrected_up.x, -forward.x),
            (right.y, corrected_up.y, -forward.y),
            (right.z, corrected_up.z, -forward.z),
        )
    ).to_4x4()
    return Matrix.Translation(eye) @ rotation


def scene_bounds(objects):
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    if not mesh_objects:
        return None

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world_corner.x)
            mins.y = min(mins.y, world_corner.y)
            mins.z = min(mins.z, world_corner.z)
            maxs.x = max(maxs.x, world_corner.x)
            maxs.y = max(maxs.y, world_corner.y)
            maxs.z = max(maxs.z, world_corner.z)

    return {
        "min_m": [mins.x, mins.y, mins.z],
        "max_m": [maxs.x, maxs.y, maxs.z],
        "size_m": [maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z],
        "center_m": [(mins.x + maxs.x) * 0.5, (mins.y + maxs.y) * 0.5, (mins.z + maxs.z) * 0.5],
    }


def far_clip_for_bounds(eye, bounds):
    if not bounds:
        return 1000.0
    mins = Vector(bounds["min_m"])
    maxs = Vector(bounds["max_m"])
    corners = [
        Vector((x, y, z))
        for x in (mins.x, maxs.x)
        for y in (mins.y, maxs.y)
        for z in (mins.z, maxs.z)
    ]
    return max((corner - eye).length for corner in corners) * 1.1


def configure_render(args):
    scene = bpy.context.scene
    requested = args.engine
    actual = requested
    try:
        scene.render.engine = requested
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
        actual = scene.render.engine

    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.film_transparent = False

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = bool(args.denoise)
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    if scene.world:
        scene.world.color = (0.75, 0.75, 0.75)

    add_guide_lights()
    return actual


def add_guide_lights():
    sun_data = bpy.data.lights.new("RAD_GUIDE_SUN", "SUN")
    sun_obj = bpy.data.objects.new("RAD_GUIDE_SUN", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
    sun_data.energy = 2.0

    area_data = bpy.data.lights.new("RAD_GUIDE_AREA", "AREA")
    area_obj = bpy.data.objects.new("RAD_GUIDE_AREA", area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (0, 0, 8)
    area_data.energy = 500.0
    area_data.size = 12.0


def add_eye_guide_light(eye):
    light_data = bpy.data.lights.new("RAD_EYE_GUIDE_POINT", "POINT")
    light_obj = bpy.data.objects.new("RAD_EYE_GUIDE_POINT", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = eye
    light_data.energy = 2800.0
    light_data.shadow_soft_size = 8.0


def create_panorama_camera(basis, near_clip, far_clip):
    cam_data = bpy.data.cameras.new("RAD_AI360_EQUIRECTANGULAR_CAMERA")
    cam_obj = bpy.data.objects.new("RAD_AI360_EQUIRECTANGULAR_CAMERA", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam_data.clip_start = near_clip
    cam_data.clip_end = far_clip
    cam_obj.matrix_world = camera_matrix_from_vectors(
        basis["eye"],
        basis["forward"],
        basis["up"],
    )
    bpy.context.scene.camera = cam_obj
    return cam_obj


def create_depth_material(name, far_clip=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    camera_data = nodes.new("ShaderNodeCameraData")
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")

    if far_clip:
        mapper = nodes.new("ShaderNodeMapRange")
        mapper.inputs["From Min"].default_value = NEAR_CLIP
        mapper.inputs["From Max"].default_value = far_clip
        mapper.inputs["To Min"].default_value = 1.0
        mapper.inputs["To Max"].default_value = 0.0
        mapper.clamp = True
        mat.node_tree.links.new(camera_data.outputs["View Z Depth"], mapper.inputs["Value"])
        mat.node_tree.links.new(mapper.outputs["Result"], emission.inputs["Color"])
    else:
        mat.node_tree.links.new(camera_data.outputs["View Z Depth"], emission.inputs["Color"])

    emission.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_normal_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (1.0, 1.0, 1.0)
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 0.5
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(geometry.outputs["Normal"], add.inputs[0])
    mat.node_tree.links.new(add.outputs["Vector"], scale.inputs[0])
    mat.node_tree.links.new(scale.outputs["Vector"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def apply_material_override(material):
    stored = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        slots = [slot.material for slot in obj.material_slots]
        stored.append((obj, slots))
        if obj.material_slots:
            for slot in obj.material_slots:
                slot.material = material
        else:
            obj.data.materials.append(material)
    return stored


def restore_materials(stored):
    for obj, slots in stored:
        obj.data.materials.clear()
        for material in slots:
            obj.data.materials.append(material)


def render_to_path(path, file_format, color_mode, color_depth, view_transform="Standard"):
    scene = bpy.context.scene
    scene.compositing_node_group = None
    scene.render.filepath = path
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = color_depth
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    if os.path.exists(path):
        os.remove(path)
    start = time.time()
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(path):
        raise RuntimeError(f"Expected render output was not written: {path}")
    return round(time.time() - start, 3)


def render_passes(out_dir):
    rgb_path = os.path.join(out_dir, "rgb", "equirectangular.png")
    depth_path = os.path.join(out_dir, "depth", "equirectangular.exr")
    depth_preview_path = os.path.join(out_dir, "depth", "equirectangular_preview.png")
    normal_path = os.path.join(out_dir, "normal", "equirectangular.exr")
    normal_preview_path = os.path.join(out_dir, "normal", "equirectangular_preview.png")

    durations = {}
    durations["rgb_seconds"] = render_to_path(rgb_path, "PNG", "RGB", "8", "Standard")

    depth_raw = bpy.data.materials["RAD_DEPTH_RAW"]
    stored = apply_material_override(depth_raw)
    durations["depth_raw_seconds"] = render_to_path(depth_path, "OPEN_EXR", "BW", "32", "Raw")
    restore_materials(stored)

    depth_preview = bpy.data.materials["RAD_DEPTH_PREVIEW"]
    stored = apply_material_override(depth_preview)
    durations["depth_preview_seconds"] = render_to_path(depth_preview_path, "PNG", "BW", "8", "Standard")
    restore_materials(stored)

    normal = bpy.data.materials["RAD_NORMAL_ENCODED"]
    stored = apply_material_override(normal)
    durations["normal_raw_seconds"] = render_to_path(normal_path, "OPEN_EXR", "RGB", "32", "Raw")
    durations["normal_preview_seconds"] = render_to_path(normal_preview_path, "PNG", "RGB", "8", "Standard")
    restore_materials(stored)

    return {
        "rgb": rgb_path,
        "depth": depth_path,
        "depth_preview": depth_preview_path,
        "normal": normal_path,
        "normal_preview": normal_preview_path,
        "durations": durations,
    }


def image_dimensions(path):
    image = bpy.data.images.load(path, check_existing=False)
    dimensions = [image.size[0], image.size[1]]
    bpy.data.images.remove(image)
    return dimensions


def validate_png(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    stride = 4
    alpha_values = pixels[3::stride]
    alpha_min = min(alpha_values)
    alpha_max = max(alpha_values)
    has_visible_pixels = False
    seam_samples = []
    top_has_visible = False
    bottom_has_visible = False
    pole_rows = max(1, height // 64)

    def rgba_at(x, y):
        index = ((y * width) + x) * stride
        return pixels[index:index + 4]

    for y in range(height):
        left = rgba_at(0, y)
        right = rgba_at(width - 1, y)
        seam_samples.append(sum(abs(left[i] - right[i]) for i in range(3)) / 3.0)
        for x in range(width):
            rgba = rgba_at(x, y)
            if any(channel > 0.003 for channel in rgba[:3]):
                has_visible_pixels = True
                if y < pole_rows:
                    top_has_visible = True
                if y >= height - pole_rows:
                    bottom_has_visible = True

    bpy.data.images.remove(image)
    return {
        "dimensions": [width, height],
        "ratio_2_to_1": width == height * 2,
        "alpha_min": round(alpha_min, 6),
        "alpha_max": round(alpha_max, 6),
        "has_visible_pixels": has_visible_pixels,
        "mean_lr_seam_rgb_delta": round(sum(seam_samples) / len(seam_samples), 6),
        "max_lr_seam_rgb_delta": round(max(seam_samples), 6),
        "top_pole_has_visible_pixels": top_has_visible,
        "bottom_pole_has_visible_pixels": bottom_has_visible,
    }


def validate_outputs(outputs, width, height):
    checks = []
    rgb_validation = validate_png(outputs["rgb"])
    checks.append({"name": "rgb_ratio_2_to_1", "pass": rgb_validation["ratio_2_to_1"], "details": rgb_validation})
    checks.append({"name": "rgb_no_unwanted_transparency", "pass": rgb_validation["alpha_min"] >= 0.999 and rgb_validation["alpha_max"] >= 0.999})
    checks.append({"name": "rgb_has_visible_pixels", "pass": rgb_validation["has_visible_pixels"]})
    checks.append({"name": "rgb_expected_dimensions", "pass": rgb_validation["dimensions"] == [width, height]})
    checks.append({"name": "rgb_lr_seam_reasonable", "pass": rgb_validation["mean_lr_seam_rgb_delta"] < 25.0, "details": rgb_validation})
    checks.append({"name": "rgb_poles_nonempty", "pass": rgb_validation["top_pole_has_visible_pixels"] and rgb_validation["bottom_pole_has_visible_pixels"], "details": rgb_validation})

    for key in ("depth", "normal"):
        checks.append({
            "name": f"{key}_exr_exists",
            "path": outputs[key],
            "pass": os.path.exists(outputs[key]) and os.path.getsize(outputs[key]) > 0,
        })
    for key in ("depth_preview", "normal_preview"):
        dimensions = image_dimensions(outputs[key])
        checks.append({
            "name": f"{key}_expected_dimensions",
            "path": outputs[key],
            "dimensions": dimensions,
            "pass": dimensions == [width, height],
        })

    return {
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "checks": checks,
        "rgb_validation": rgb_validation,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(outputs):
    manifest = {}
    for key, value in outputs.items():
        if key == "durations":
            continue
        manifest[key] = {
            "path": value,
            "bytes": os.path.getsize(value),
            "sha256": file_sha256(value),
        }
    return manifest


def round_vector(vector):
    return [round(float(value), 6) for value in vector]


def write_report(out_dir, report):
    path = os.path.join(out_dir, "phase0d_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def main():
    args = parse_args()
    start = time.time()
    prepare_output_dirs(args.out)

    with open(args.json, "r", encoding="utf-8") as handle:
        scene_data = json.load(handle)

    clear_scene()
    import_report = import_obj(args.obj)
    bounds = scene_bounds(list(bpy.context.scene.objects))
    basis = camera_basis(scene_data)
    far_clip = far_clip_for_bounds(basis["eye"], bounds)
    engine = configure_render(args)
    add_eye_guide_light(basis["eye"])
    create_depth_material("RAD_DEPTH_RAW")
    create_depth_material("RAD_DEPTH_PREVIEW", far_clip)
    create_normal_material("RAD_NORMAL_ENCODED")
    camera = create_panorama_camera(basis, NEAR_CLIP, far_clip)

    outputs = render_passes(args.out)
    manifest = output_manifest(outputs)
    validation = validate_outputs(outputs, args.width, args.height)
    validation["checks"].append({
        "name": "output_manifest_sha256_present",
        "pass": all(item["sha256"] for item in manifest.values()),
    })
    validation["status"] = "PASS" if all(check["pass"] for check in validation["checks"]) else "FAIL"

    report = {
        "status": validation["status"],
        "source": {
            "obj": args.obj,
            "json": args.json,
            "scene_name": scene_data["camera"]["scene_name"],
        },
        "mode": {
            "name": "phase0d_equirectangular_native",
            "primary_output": "native equirectangular panorama",
            "cubemap_status": "debug/compatibility only",
        },
        "render": {
            "requested_engine": args.engine,
            "actual_engine": engine,
            "projection": "panoramic_equirectangular",
            "resolution": [args.width, args.height],
            "ratio": "2:1",
            "samples": args.samples,
            "denoise": bool(args.denoise),
            "color_management": {
                "view_transform": bpy.context.scene.view_settings.view_transform,
                "look": bpy.context.scene.view_settings.look,
                "exposure": bpy.context.scene.view_settings.exposure,
                "gamma": bpy.context.scene.view_settings.gamma,
            },
        },
        "camera_basis": {
            "eye_m": round_vector(basis["eye"]),
            "forward_from_scene": round_vector(basis["forward"]),
            "up_from_scene": round_vector(basis["up"]),
            "right_derived": round_vector(basis["right"]),
            "camera_type": camera.data.type,
            "panorama_type": camera.data.panorama_type,
        },
        "depth": {
            "raw_representation": "Blender Camera Data View Z Depth rendered through an emission override material and saved as 32-bit OpenEXR.",
            "units": "metres in Blender world/camera depth pass",
            "near_clip_m": NEAR_CLIP,
            "far_clip_m": round(far_clip, 6),
            "preview_normalization": "Shared linear mapping: near_clip -> white, far_clip -> black, clamped.",
        },
        "normal": {
            "raw_representation": "Blender Geometry node Normal rendered through an emission override material and saved as 32-bit RGB OpenEXR.",
            "convention": "World/shading-space normal encoded as RGB = normal * 0.5 + 0.5.",
            "preview_normalization": "Preview PNG uses the same encoded [0, 1] RGB normal representation.",
        },
        "scene_bounds": {
            "min_m": round_vector(Vector(bounds["min_m"])),
            "max_m": round_vector(Vector(bounds["max_m"])),
            "size_m": round_vector(Vector(bounds["size_m"])),
        } if bounds else None,
        "import": import_report,
        "outputs": outputs,
        "output_manifest": manifest,
        "validation": validation,
        "duration_seconds": round(time.time() - start, 3),
    }
    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0D_REPORT={report_path}")


if __name__ == "__main__":
    main()
