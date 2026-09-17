import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector


SKETCHUP_TO_BLENDER = Matrix.Identity(4)
SKETCHUP_TO_BLENDER_NOTE = (
    "Identity conversion. The SketchUp exporter writes OBJ in metres with Z-up "
    "and swap_yz=false. Blender OBJ import is configured with forward_axis=Y "
    "and up_axis=Z, so scene.json camera values are converted from mm to m and "
    "placed in the same world space."
)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Import SketchUp OBJ and reconstruct the AI360 camera from scene.json."
    )
    parser.add_argument("--obj", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dae-report", default=None)
    parser.add_argument("--fbx-report", default=None)
    parser.add_argument("--render-preview", action="store_true")
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
        "filepath": obj_path,
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
    if forward.length == 0:
        raise ValueError("Camera direction vector is zero.")
    if up_vector.length == 0:
        raise ValueError("Camera up vector is zero.")

    right = forward.cross(up_vector).normalized()
    if right.length == 0:
        raise ValueError("Camera direction and up vectors are parallel.")

    corrected_up = right.cross(forward).normalized()
    rotation = Matrix(
        (
            (right.x, corrected_up.x, -forward.x),
            (right.y, corrected_up.y, -forward.y),
            (right.z, corrected_up.z, -forward.z),
        )
    ).to_4x4()
    return Matrix.Translation(eye) @ rotation


def create_ai360_camera(scene_data):
    camera_data = scene_data["camera"]
    eye = mm_to_meters(camera_data["eye_mm"])
    direction = vector_to_blender(camera_data["direction"])
    up = vector_to_blender(camera_data["up"])

    cam_data = bpy.data.cameras.new("RAD_AI360_CAMERA")
    cam_object = bpy.data.objects.new("RAD_AI360_CAMERA", cam_data)
    bpy.context.collection.objects.link(cam_object)
    cam_object.matrix_world = camera_matrix_from_vectors(eye, direction, up)

    if camera_data.get("projection") == "parallel":
        cam_data.type = "ORTHO"
    else:
        cam_data.type = "PERSP"
        fov = camera_data.get("fov_degrees")
        if fov:
            cam_data.angle = math.radians(float(fov))

    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000.0
    bpy.context.scene.camera = cam_object
    return cam_object


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
        "min_m": [round(mins.x, 6), round(mins.y, 6), round(mins.z, 6)],
        "max_m": [round(maxs.x, 6), round(maxs.y, 6), round(maxs.z, 6)],
        "size_m": [
            round(maxs.x - mins.x, 6),
            round(maxs.y - mins.y, 6),
            round(maxs.z - mins.z, 6),
        ],
        "center_m": [
            round((mins.x + maxs.x) * 0.5, 6),
            round((mins.y + maxs.y) * 0.5, 6),
            round((mins.z + maxs.z) * 0.5, 6),
        ],
    }


def mesh_stats(objects):
    vertex_count = 0
    triangle_count = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        vertex_count += len(mesh.vertices)
        mesh.calc_loop_triangles()
        triangle_count += len(mesh.loop_triangles)
    return vertex_count, triangle_count


def camera_relation(camera, bounds):
    if not bounds:
        return None
    min_v = Vector(bounds["min_m"])
    max_v = Vector(bounds["max_m"])
    center = Vector(bounds["center_m"])
    location = camera.location
    forward = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    to_center = center - location
    return {
        "inside_bounds": bool(
            min_v.x <= location.x <= max_v.x
            and min_v.y <= location.y <= max_v.y
            and min_v.z <= location.z <= max_v.z
        ),
        "distance_to_scene_center_m": round(to_center.length, 6),
        "forward_dot_to_center": round(forward.normalized().dot(to_center.normalized()), 6)
        if to_center.length > 0
        else None,
    }


def setup_preview_shading():
    scene = bpy.context.scene
    available_engines = [
        item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items
    ]
    for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if engine in available_engines:
            scene.render.engine = engine
            break

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
    except TypeError:
        pass

    if hasattr(scene, "display") and hasattr(scene.display, "shading"):
        try:
            scene.display.shading.light = "STUDIO"
            scene.display.shading.color_type = "MATERIAL"
            scene.display.shading.background_type = "VIEWPORT"
            scene.display.shading.background_color = (0.78, 0.78, 0.78)
        except TypeError:
            pass


def add_preview_light():
    light_data = bpy.data.lights.new("RAD_PREVIEW_SUN", "SUN")
    light_obj = bpy.data.objects.new("RAD_PREVIEW_SUN", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
    light_data.energy = 2.0


def render_camera_preview(out_dir):
    setup_preview_shading()
    add_preview_light()
    path = os.path.join(out_dir, "preview_camera.png")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def add_camera_marker(camera):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.45, location=camera.location)
    marker = bpy.context.object
    marker.name = "RAD_AI360_CAMERA_MARKER"
    marker.color = (1.0, 0.1, 0.05, 1.0)
    return marker


def render_overview(out_dir, bounds, camera):
    if not bounds:
        return None

    marker = add_camera_marker(camera)
    min_v = Vector(bounds["min_m"])
    max_v = Vector(bounds["max_m"])
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    distance = max(size.x, size.y, size.z) * 1.25

    cam_data = bpy.data.cameras.new("RAD_OVERVIEW_CAMERA")
    cam_obj = bpy.data.objects.new("RAD_OVERVIEW_CAMERA", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = center + Vector((distance, -distance, distance * 0.55))
    cam_obj.matrix_world = camera_matrix_from_vectors(
        cam_obj.location,
        center - cam_obj.location,
        Vector((0.0, 0.0, 1.0)),
    )
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(size.x, size.y) * 1.15
    cam_data.clip_start = 0.01
    cam_data.clip_end = distance * 4.0

    previous_camera = bpy.context.scene.camera
    bpy.context.scene.camera = cam_obj
    path = os.path.join(out_dir, "preview_overview.png")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.camera = previous_camera
    bpy.data.objects.remove(marker, do_unlink=True)
    return path


def read_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_report(label, prior, current):
    if not prior:
        return None
    prior_counts = prior.get("counts", {})
    prior_bounds = prior.get("bounds")
    current_counts = current.get("counts", {})
    current_bounds = current.get("bounds")
    comparison = {
        f"{label}_status": prior.get("status"),
        f"{label}_mesh_objects": prior_counts.get("mesh_objects"),
        "obj_mesh_objects": current_counts.get("mesh_objects"),
        f"{label}_materials": prior_counts.get("materials"),
        "obj_materials": current_counts.get("materials"),
        f"{label}_bounds_size_m": prior_bounds.get("size_m") if prior_bounds else None,
        "obj_bounds_size_m": current_bounds.get("size_m") if current_bounds else None,
    }
    if prior_bounds and current_bounds:
        old_size = prior_bounds["size_m"]
        new_size = current_bounds["size_m"]
        comparison["bounds_size_delta_m"] = [
            round(new_size[index] - old_size[index], 6) for index in range(3)
        ]
    return comparison


def write_report(out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase0b_obj_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def fail_report(args, scene_data, error, start):
    report = {
        "status": "FAIL",
        "warnings": [],
        "errors": [str(error)],
        "source": {
            "obj": args.obj,
            "json": args.json,
            "scene_name": scene_data["camera"]["scene_name"],
        },
        "coordinate_conversion": {
            "name": "sketchup_z_up_m_to_blender_z_up_m_identity",
            "note": SKETCHUP_TO_BLENDER_NOTE,
        },
        "counts": {
            "objects_total": 0,
            "mesh_objects": 0,
            "vertices": 0,
            "triangles": 0,
            "materials": 0,
            "images": 0,
            "cameras": 0,
        },
        "bounds": None,
        "camera": {
            "source_scene": scene_data["camera"]["scene_name"],
            "source_eye_mm": scene_data["camera"]["eye_mm"],
            "expected_location_m_before_axis_conversion": [
                round(value / 1000.0, 6) for value in scene_data["camera"]["eye_mm"]
            ],
        },
        "outputs": {
            "blend": None,
            "preview": None,
            "overview": None,
            "preview_error": "Preview skipped because OBJ import failed.",
        },
        "duration_seconds": round(time.time() - start, 3),
    }
    report["dae_proof_comparison"] = compare_report("dae", read_json(args.dae_report), report)
    report["fbx_proof_comparison"] = compare_report("fbx", read_json(args.fbx_report), report)
    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0B_OBJ_REPORT={report_path}")


def main():
    args = parse_args()
    overall_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as handle:
        scene_data = json.load(handle)
    if scene_data.get("units") != "mm":
        raise ValueError("scene.json must use units: mm")

    clear_scene()
    try:
        import_result = import_obj(args.obj)
    except Exception as exc:
        fail_report(args, scene_data, exc, overall_start)
        return

    camera = create_ai360_camera(scene_data)
    objects = list(bpy.context.scene.objects)
    bounds = scene_bounds(objects)
    vertex_count, triangle_count = mesh_stats(objects)

    blend_path = os.path.join(args.out, "phase0b_obj_imported.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    preview_error = None
    preview_path = None
    overview_path = None
    if args.render_preview:
        try:
            preview_path = render_camera_preview(args.out)
            overview_path = render_overview(args.out, bounds, camera)
        except Exception as exc:
            preview_error = str(exc)

    warnings = []
    if not bounds:
        warnings.append("No mesh bounds were computed; OBJ may not have imported geometry.")
    if preview_error:
        warnings.append(f"Preview render failed: {preview_error}")
    if len(bpy.data.images) == 0:
        warnings.append("No image textures were loaded by the OBJ importer.")

    report = {
        "status": "WARN" if warnings else "PASS",
        "warnings": warnings,
        "errors": [],
        "source": {
            "obj": args.obj,
            "json": args.json,
            "scene_name": scene_data["camera"]["scene_name"],
        },
        "coordinate_conversion": {
            "name": "sketchup_z_up_m_to_blender_z_up_m_identity",
            "note": SKETCHUP_TO_BLENDER_NOTE,
        },
        "import": import_result,
        "counts": {
            "objects_total": len(objects),
            "mesh_objects": sum(1 for obj in objects if obj.type == "MESH"),
            "vertices": vertex_count,
            "triangles": triangle_count,
            "materials": len(bpy.data.materials),
            "images": len(bpy.data.images),
            "cameras": len(bpy.data.cameras),
        },
        "bounds": bounds,
        "camera": {
            "name": camera.name,
            "source_scene": scene_data["camera"]["scene_name"],
            "source_eye_mm": scene_data["camera"]["eye_mm"],
            "location_m": [round(v, 6) for v in camera.location],
            "forward": [
                round(v, 6)
                for v in (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1)))
            ],
            "up": [
                round(v, 6)
                for v in (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0)))
            ],
            "fov_degrees": round(math.degrees(camera.data.angle), 6)
            if camera.data.type == "PERSP"
            else None,
            "relation_to_scene": camera_relation(camera, bounds),
        },
        "outputs": {
            "blend": blend_path,
            "preview": preview_path,
            "overview": overview_path,
            "preview_error": preview_error,
        },
        "duration_seconds": round(time.time() - overall_start, 3),
    }
    report["dae_proof_comparison"] = compare_report("dae", read_json(args.dae_report), report)
    report["fbx_proof_comparison"] = compare_report("fbx", read_json(args.fbx_report), report)
    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0B_OBJ_REPORT={report_path}")


if __name__ == "__main__":
    main()
