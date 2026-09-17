import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector


FACE_ORDER = ("front", "right", "back", "left", "top", "bottom")
RESOLUTION = 1024
NEAR_CLIP = 0.01
SKETCHUP_TO_BLENDER = Matrix.Identity(4)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Generate deterministic 6-face RGB/depth/normal cubemap conditioning passes."
    )
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    return parser.parse_args(argv[argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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


def cubemap_basis(scene_data):
    eye = mm_to_meters(scene_data["camera"]["eye_mm"])
    source_forward = vector_to_blender(scene_data["camera"]["direction"])
    source_up = vector_to_blender(scene_data["camera"]["up"])

    forward = source_forward.normalized()
    up = (source_up - forward * source_up.dot(forward)).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()

    return {
        "eye": eye,
        "front_forward": forward,
        "front_up": up,
        "front_right": right,
        "faces": {
            "front": {"forward": forward, "up": up},
            "right": {"forward": right, "up": up},
            "back": {"forward": -forward, "up": up},
            "left": {"forward": -right, "up": up},
            "top": {"forward": up, "up": -forward},
            "bottom": {"forward": -up, "up": forward},
        },
    }


def create_camera(eye, near_clip, far_clip):
    cam_data = bpy.data.cameras.new("RAD_AI360_CUBEMAP_CAMERA")
    cam_obj = bpy.data.objects.new("RAD_AI360_CUBEMAP_CAMERA", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_data.type = "PERSP"
    cam_data.angle = math.radians(90.0)
    cam_data.clip_start = near_clip
    cam_data.clip_end = far_clip
    bpy.context.scene.camera = cam_obj
    cam_obj.location = eye
    return cam_obj


def configure_render(resolution):
    scene = bpy.context.scene
    available_engines = [
        item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
    ]
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if engine in available_engines:
            scene.render.engine = engine
            break

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 16
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_normal = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    if scene.world:
        scene.world.color = (0.75, 0.75, 0.75)

    add_preview_lights()
    return scene.render.engine


def add_preview_lights():
    light_data = bpy.data.lights.new("RAD_GUIDE_SUN", "SUN")
    light_obj = bpy.data.objects.new("RAD_GUIDE_SUN", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
    light_data.energy = 2.0

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


def prepare_output_dirs(out_dir):
    for folder in ("rgb", "depth", "normal"):
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)


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
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(path):
        raise RuntimeError(f"Expected render output was not written: {path}")
    return path


def render_face(out_dir, face, camera, eye, forward, up, near_clip, far_clip):
    start = time.time()
    scene = bpy.context.scene
    camera.matrix_world = camera_matrix_from_vectors(eye, forward, up)

    rgb_path = os.path.join(out_dir, "rgb", f"{face}.png")
    render_to_path(rgb_path, "PNG", "RGB", "8", "Standard")

    depth_material = bpy.data.materials["RAD_DEPTH_RAW"]
    depth_stored = apply_material_override(depth_material)
    depth_path = os.path.join(out_dir, "depth", f"{face}.exr")
    render_to_path(depth_path, "OPEN_EXR", "BW", "32", "Raw")
    restore_materials(depth_stored)

    depth_preview_material = bpy.data.materials["RAD_DEPTH_PREVIEW"]
    depth_preview_stored = apply_material_override(depth_preview_material)
    depth_preview_path = os.path.join(out_dir, "depth", f"{face}_preview.png")
    render_to_path(depth_preview_path, "PNG", "BW", "8", "Standard")
    restore_materials(depth_preview_stored)

    normal_material = bpy.data.materials["RAD_NORMAL_ENCODED"]
    normal_stored = apply_material_override(normal_material)
    normal_path = os.path.join(out_dir, "normal", f"{face}.exr")
    render_to_path(normal_path, "OPEN_EXR", "RGB", "32", "Raw")
    normal_preview_path = os.path.join(out_dir, "normal", f"{face}_preview.png")
    render_to_path(normal_preview_path, "PNG", "RGB", "8", "Standard")
    restore_materials(normal_stored)

    right = forward.cross(up).normalized()
    return {
        "eye": round_vector(eye),
        "forward": round_vector(forward),
        "up": round_vector(up),
        "right": round_vector(right),
        "fov_degrees": 90.0,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "near_clip_m": near_clip,
        "far_clip_m": round(far_clip, 6),
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "depth_preview_path": depth_preview_path,
        "normal_path": normal_path,
        "normal_preview_path": normal_preview_path,
        "render_duration_seconds": round(time.time() - start, 3),
    }


def round_vector(vector):
    return [round(float(value), 6) for value in vector]


def dot(a, b):
    return Vector(a).dot(Vector(b))


def validate_faces(face_reports):
    checks = []
    eyes = [Vector(face_reports[face]["eye"]) for face in FACE_ORDER]
    first_eye = eyes[0]
    checks.append({
        "name": "all_eyes_identical",
        "pass": all((eye - first_eye).length <= 1e-6 for eye in eyes),
    })

    for face in FACE_ORDER:
        forward = Vector(face_reports[face]["forward"])
        up = Vector(face_reports[face]["up"])
        right = Vector(face_reports[face]["right"])
        checks.append({"name": f"{face}_forward_normalized", "pass": abs(forward.length - 1.0) <= 1e-6})
        checks.append({"name": f"{face}_up_normalized", "pass": abs(up.length - 1.0) <= 1e-6})
        checks.append({"name": f"{face}_right_normalized", "pass": abs(right.length - 1.0) <= 1e-6})
        checks.append({"name": f"{face}_orthogonal_basis", "pass": abs(forward.dot(up)) <= 1e-6 and abs(forward.dot(right)) <= 1e-6 and abs(up.dot(right)) <= 1e-6})
        checks.append({"name": f"{face}_square_resolution", "pass": face_reports[face]["resolution"][0] == face_reports[face]["resolution"][1]})

    checks.extend([
        {"name": "front_back_180_degrees", "pass": dot(face_reports["front"]["forward"], face_reports["back"]["forward"]) <= -0.999999},
        {"name": "right_left_180_degrees", "pass": dot(face_reports["right"]["forward"], face_reports["left"]["forward"]) <= -0.999999},
        {"name": "top_bottom_180_degrees", "pass": dot(face_reports["top"]["forward"], face_reports["bottom"]["forward"]) <= -0.999999},
        {"name": "front_right_90_degrees", "pass": abs(dot(face_reports["front"]["forward"], face_reports["right"]["forward"])) <= 1e-6},
        {"name": "right_back_90_degrees", "pass": abs(dot(face_reports["right"]["forward"], face_reports["back"]["forward"])) <= 1e-6},
        {"name": "back_left_90_degrees", "pass": abs(dot(face_reports["back"]["forward"], face_reports["left"]["forward"])) <= 1e-6},
        {"name": "left_front_90_degrees", "pass": abs(dot(face_reports["left"]["forward"], face_reports["front"]["forward"])) <= 1e-6},
    ])

    for horizontal in ("front", "right", "back", "left"):
        checks.append({"name": f"top_orthogonal_to_{horizontal}", "pass": abs(dot(face_reports["top"]["forward"], face_reports[horizontal]["forward"])) <= 1e-6})
        checks.append({"name": f"bottom_orthogonal_to_{horizontal}", "pass": abs(dot(face_reports["bottom"]["forward"], face_reports[horizontal]["forward"])) <= 1e-6})

    return {
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "checks": checks,
    }


def image_dimensions(path):
    image = bpy.data.images.load(path, check_existing=False)
    dimensions = [image.size[0], image.size[1]]
    bpy.data.images.remove(image)
    return dimensions


def verify_output_dimensions(face_reports):
    checks = []
    for face in FACE_ORDER:
        for key in ("rgb_path", "depth_preview_path", "normal_preview_path"):
            dimensions = image_dimensions(face_reports[face][key])
            checks.append({"name": f"{face}_{key}_dimensions", "path": face_reports[face][key], "dimensions": dimensions, "pass": dimensions == [RESOLUTION, RESOLUTION]})
        for key in ("depth_path", "normal_path"):
            checks.append({"name": f"{face}_{key}_exists", "path": face_reports[face][key], "pass": os.path.exists(face_reports[face][key]) and os.path.getsize(face_reports[face][key]) > 0})
    return checks


def write_report(out_dir, report):
    path = os.path.join(out_dir, "phase0c_report.json")
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
    basis = cubemap_basis(scene_data)
    far_clip = far_clip_for_bounds(basis["eye"], bounds)
    render_engine = configure_render(args.resolution)
    add_eye_guide_light(basis["eye"])
    create_depth_material("RAD_DEPTH_RAW")
    create_depth_material("RAD_DEPTH_PREVIEW", far_clip)
    create_normal_material("RAD_NORMAL_ENCODED")
    camera = create_camera(basis["eye"], NEAR_CLIP, far_clip)

    face_reports = {}
    for face in FACE_ORDER:
        face_basis = basis["faces"][face]
        face_reports[face] = render_face(
            args.out,
            face,
            camera,
            basis["eye"],
            face_basis["forward"],
            face_basis["up"],
            NEAR_CLIP,
            far_clip,
        )

    basis_validation = validate_faces(face_reports)
    dimension_checks = verify_output_dimensions(face_reports)
    output_validation = {
        "status": "PASS" if all(check["pass"] for check in dimension_checks) else "FAIL",
        "checks": dimension_checks,
    }

    report = {
        "status": "PASS" if basis_validation["status"] == "PASS" and output_validation["status"] == "PASS" else "FAIL",
        "source": {
            "obj": args.obj,
            "json": args.json,
            "scene_name": scene_data["camera"]["scene_name"],
        },
        "transport": {
            "format": "OBJ",
            "frozen": True,
            "note": "No DAE/FBX experiments performed in Phase 0C.",
        },
        "render": {
            "engine": render_engine,
            "resolution": [args.resolution, args.resolution],
            "projection": "perspective",
            "fov_degrees": 90.0,
        },
        "depth": {
            "raw_representation": "Blender Camera Data View Z Depth rendered through an emission override material and saved as 32-bit OpenEXR. RGB channels carry the same metric depth value.",
            "units": "metres in Blender world/camera depth pass",
            "near_clip_m": NEAR_CLIP,
            "far_clip_m": round(far_clip, 6),
            "preview_normalization": "Shared linear mapping for all faces: near_clip -> white, far_clip -> black, clamped.",
        },
        "normal": {
            "raw_representation": "Blender Geometry node Normal rendered through an emission override material and saved as 32-bit RGB OpenEXR.",
            "convention": "World/shading-space normal encoded as RGB = normal * 0.5 + 0.5, consistent across all six faces.",
            "preview_normalization": "Preview PNG uses the same encoded [0, 1] RGB normal representation.",
        },
        "camera_basis": {
            "eye_m": round_vector(basis["eye"]),
            "front_from_scene_forward": round_vector(basis["front_forward"]),
            "front_up_from_scene_up": round_vector(basis["front_up"]),
            "front_right_derived": round_vector(basis["front_right"]),
        },
        "scene_bounds": {
            "min_m": round_vector(Vector(bounds["min_m"])),
            "max_m": round_vector(Vector(bounds["max_m"])),
            "size_m": round_vector(Vector(bounds["size_m"])),
        } if bounds else None,
        "import": import_report,
        "faces": face_reports,
        "validation": {
            "basis": basis_validation,
            "outputs": output_validation,
        },
        "duration_seconds": round(time.time() - start, 3),
    }

    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0C_REPORT={report_path}")


if __name__ == "__main__":
    main()
