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
    "Identity conversion. The SketchUp exporter writes FBX in metres with Z-up "
    "and swap_yz=false; Blender's FBX importer places the imported architecture "
    "in Blender Z-up world space. Camera values from scene.json are converted "
    "from mm to m and placed in the same world space."
)


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Import SketchUp FBX and reconstruct the AI360 camera from scene.json."
    )
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dae-report", default=None)
    parser.add_argument("--render-preview", action="store_true")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(fbx_path):
    start = time.time()
    kwargs = {
        "filepath": fbx_path,
        "use_custom_normals": True,
        "use_image_search": True,
        "use_anim": False,
        "use_manual_orientation": False,
    }
    bpy.ops.import_scene.fbx(**kwargs)
    return {
        "operator": "bpy.ops.import_scene.fbx",
        "filepath": fbx_path,
        "options": kwargs,
        "duration_seconds": round(time.time() - start, 3),
    }


def mm_to_meters(values):
    source = Vector((values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0))
    return SKETCHUP_TO_BLENDER @ source.to_4d()


def vector_to_blender(values):
    source = Vector(values).normalized()
    converted = (SKETCHUP_TO_BLENDER.to_3x3() @ source).normalized()
    return converted


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
    eye_4d = mm_to_meters(camera_data["eye_mm"])
    eye = Vector((eye_4d.x, eye_4d.y, eye_4d.z))
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


def add_camera_marker(camera):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.45, location=camera.location)
    marker = bpy.context.object
    marker.name = "RAD_AI360_CAMERA_MARKER"
    marker.color = (1.0, 0.1, 0.05, 1.0)
    return marker


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


def compare_with_dae(dae_report, current):
    if not dae_report:
        return None
    previous_counts = dae_report.get("counts", {})
    previous_bounds = dae_report.get("bounds")
    current_counts = current.get("counts", {})
    current_bounds = current.get("bounds")
    comparison = {
        "dae_status": dae_report.get("status"),
        "dae_mesh_objects": previous_counts.get("mesh_objects"),
        "fbx_mesh_objects": current_counts.get("mesh_objects"),
        "dae_materials": previous_counts.get("materials"),
        "fbx_materials": current_counts.get("materials"),
        "dae_bounds_size_m": previous_bounds.get("size_m") if previous_bounds else None,
        "fbx_bounds_size_m": current_bounds.get("size_m") if current_bounds else None,
    }
    if previous_bounds and current_bounds:
        dae_size = previous_bounds["size_m"]
        fbx_size = current_bounds["size_m"]
        comparison["bounds_size_delta_m"] = [
            round(fbx_size[index] - dae_size[index], 6) for index in range(3)
        ]
    return comparison


def write_report(out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase0b_fbx_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def main():
    args = parse_args()
    overall_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as handle:
        scene_data = json.load(handle)
    if scene_data.get("units") != "mm":
        raise ValueError("scene.json must use units: mm")

    clear_scene()
    import_error = None
    import_result = None
    try:
        import_result = import_fbx(args.fbx)
    except Exception as exc:
        import_error = str(exc)

    if import_error:
        dae_report = read_json(args.dae_report)
        report = {
            "status": "FAIL",
            "warnings": [],
            "errors": [import_error],
            "source": {
                "fbx": args.fbx,
                "json": args.json,
                "scene_name": scene_data["camera"]["scene_name"],
            },
            "coordinate_conversion": {
                "name": "sketchup_z_up_m_to_blender_z_up_m_identity",
                "note": SKETCHUP_TO_BLENDER_NOTE,
            },
            "import": {
                "operator": "bpy.ops.import_scene.fbx",
                "filepath": args.fbx,
                "duration_seconds": round(time.time() - overall_start, 3),
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
                "preview_error": "Preview skipped because FBX import failed.",
            },
            "duration_seconds": round(time.time() - overall_start, 3),
        }
        report["dae_proof_comparison"] = compare_with_dae(dae_report, report)
        report_path = write_report(args.out, report)
        print(f"RAD_PHASE0B_FBX_REPORT={report_path}")
        return

    camera = create_ai360_camera(scene_data)
    objects = list(bpy.context.scene.objects)
    bounds = scene_bounds(objects)
    vertex_count, triangle_count = mesh_stats(objects)

    blend_path = os.path.join(args.out, "phase0b_fbx_imported.blend")
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
        warnings.append("No mesh bounds were computed; FBX may not have imported geometry.")
    if preview_error:
        warnings.append(f"Preview render failed: {preview_error}")
    if len(bpy.data.images) == 0:
        warnings.append("No image textures were loaded by the FBX importer.")

    report = {
        "status": "WARN" if warnings else "PASS",
        "warnings": warnings,
        "source": {
            "fbx": args.fbx,
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
    report["dae_proof_comparison"] = compare_with_dae(read_json(args.dae_report), report)

    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0B_FBX_REPORT={report_path}")


if __name__ == "__main__":
    main()
