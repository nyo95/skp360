import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Import SketchUp DAE and reconstruct the AI360 camera from scene.json."
    )
    parser.add_argument("--dae", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-preview", action="store_true")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def read_dae_asset(dae_path):
    result = {
        "unit_meter": None,
        "unit_name": None,
        "up_axis": None,
    }

    try:
        for _event, elem in ET.iterparse(dae_path, events=("end",)):
            tag = elem.tag.split("}", 1)[-1]
            if tag == "unit":
                meter = elem.attrib.get("meter")
                result["unit_meter"] = float(meter) if meter else None
                result["unit_name"] = elem.attrib.get("name")
            elif tag == "up_axis":
                result["up_axis"] = (elem.text or "").strip()

            if result["unit_meter"] is not None and result["up_axis"]:
                break
            elem.clear()
    except Exception as exc:
        result["read_error"] = str(exc)

    return result


def import_collada(dae_path):
    attempts = [
        {"filepath": dae_path, "import_units": True},
        {"filepath": dae_path},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            bpy.ops.wm.collada_import(**kwargs)
            return kwargs
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break

    try:
        bpy.ops.import_scene.dae(filepath=dae_path)
        return {"filepath": dae_path, "operator": "import_scene.dae"}
    except Exception as exc:
        fallback = import_collada_fallback(dae_path)
        fallback["native_import_error"] = str(last_error or exc)
        return fallback


def local_name(tag):
    return tag.split("}", 1)[-1]


def child_by_name(element, name):
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def children_by_name(element, name):
    return [child for child in list(element) if local_name(child.tag) == name]


def find_descendant_by_id(element, tag_name, element_id):
    for descendant in element.iter():
        if local_name(descendant.tag) == tag_name and descendant.attrib.get("id") == element_id:
            return descendant
    return None


def parse_float_array(text):
    if not text:
        return []
    return [float(value) for value in text.split()]


def parse_int_array(text):
    if not text:
        return []
    return [int(value) for value in text.split()]


def parse_matrix(element, unit_meter):
    values = parse_float_array(element.text)
    if len(values) != 16:
        return Matrix.Identity(4)

    rows = [
        values[0:4],
        values[4:8],
        values[8:12],
        values[12:16],
    ]
    matrix = Matrix(rows)
    matrix[0][3] *= unit_meter
    matrix[1][3] *= unit_meter
    matrix[2][3] *= unit_meter
    return matrix


def convert_geometry_mesh(geometry, unit_meter):
    mesh_node = child_by_name(geometry, "mesh")
    if mesh_node is None:
        return None

    sources = {}
    for source in children_by_name(mesh_node, "source"):
        float_array = child_by_name(source, "float_array")
        if float_array is None:
            continue

        stride = 3
        technique = child_by_name(source, "technique_common")
        accessor = None
        if technique is not None:
            accessor = child_by_name(technique, "accessor")
        if accessor is not None and accessor.attrib.get("stride"):
            stride = int(accessor.attrib["stride"])

        raw_values = parse_float_array(float_array.text)
        values = []
        for index in range(0, len(raw_values), stride):
            if index + 2 >= len(raw_values):
                break
            values.append(
                (
                    raw_values[index] * unit_meter,
                    raw_values[index + 1] * unit_meter,
                    raw_values[index + 2] * unit_meter,
                )
            )
        sources["#" + source.attrib["id"]] = values

    vertices_sources = {}
    for vertices in children_by_name(mesh_node, "vertices"):
        position_source = None
        for input_node in children_by_name(vertices, "input"):
            if input_node.attrib.get("semantic") == "POSITION":
                position_source = input_node.attrib.get("source")
                break
        if position_source:
            vertices_sources["#" + vertices.attrib["id"]] = position_source

    mesh_vertices = []
    mesh_faces = []
    vertex_lookup = {}

    for triangles in children_by_name(mesh_node, "triangles"):
        vertex_source = None
        vertex_offset = 0
        max_offset = 0
        for input_node in children_by_name(triangles, "input"):
            offset = int(input_node.attrib.get("offset", "0"))
            max_offset = max(max_offset, offset)
            if input_node.attrib.get("semantic") == "VERTEX":
                vertex_offset = offset
                vertex_source = vertices_sources.get(input_node.attrib.get("source"))

        positions = sources.get(vertex_source or "")
        p_node = child_by_name(triangles, "p")
        if not positions or p_node is None:
            continue

        stride = max_offset + 1
        indices = parse_int_array(p_node.text)
        for index in range(0, len(indices), stride * 3):
            face = []
            for corner in range(3):
                position_index = indices[index + (corner * stride) + vertex_offset]
                if position_index not in vertex_lookup:
                    vertex_lookup[position_index] = len(mesh_vertices)
                    mesh_vertices.append(positions[position_index])
                face.append(vertex_lookup[position_index])
            mesh_faces.append(tuple(face))

    if not mesh_vertices or not mesh_faces:
        return None

    mesh_name = geometry.attrib.get("name") or geometry.attrib.get("id") or "dae_mesh"
    blender_mesh = bpy.data.meshes.new(mesh_name)
    blender_mesh.from_pydata(mesh_vertices, [], mesh_faces)
    blender_mesh.update()
    return blender_mesh


def collect_geometry_instances(
    node,
    parent_matrix,
    unit_meter,
    instances,
    library_nodes=None,
    node_stack=None,
):
    library_nodes = library_nodes or {}
    node_stack = node_stack or []
    matrix = parent_matrix.copy()
    for child in children_by_name(node, "matrix"):
        matrix = matrix @ parse_matrix(child, unit_meter)

    for instance in children_by_name(node, "instance_geometry"):
        url = instance.attrib.get("url", "")
        if url.startswith("#"):
            instances.append(
                {
                    "geometry_id": url[1:],
                    "name": node.attrib.get("name") or url[1:],
                    "matrix": matrix.copy(),
                }
            )

    for child in children_by_name(node, "node"):
        collect_geometry_instances(
            child,
            matrix,
            unit_meter,
            instances,
            library_nodes,
            node_stack,
        )

    for instance_node in children_by_name(node, "instance_node"):
        url = instance_node.attrib.get("url", "")
        if not url.startswith("#"):
            continue
        node_id = url[1:]
        if node_id in node_stack:
            continue
        referenced_node = library_nodes.get(node_id)
        if referenced_node is None:
            continue
        collect_geometry_instances(
            referenced_node,
            matrix,
            unit_meter,
            instances,
            library_nodes,
            node_stack + [node_id],
        )


def import_collada_fallback(dae_path):
    dae_asset = read_dae_asset(dae_path)
    unit_meter = dae_asset.get("unit_meter") or 1.0

    tree = ET.parse(dae_path)
    root = tree.getroot()

    geometry_nodes = {}
    library_geometries = child_by_name(root, "library_geometries")
    if library_geometries is not None:
        for geometry in children_by_name(library_geometries, "geometry"):
            geometry_id = geometry.attrib.get("id")
            if geometry_id:
                geometry_nodes[geometry_id] = geometry

    library_nodes = {}
    library_nodes_node = child_by_name(root, "library_nodes")
    if library_nodes_node is not None:
        for node in children_by_name(library_nodes_node, "node"):
            node_id = node.attrib.get("id")
            if node_id:
                library_nodes[node_id] = node

    visual_scene = None
    library_visual_scenes = child_by_name(root, "library_visual_scenes")
    if library_visual_scenes is not None:
        visual_scene = child_by_name(library_visual_scenes, "visual_scene")

    if visual_scene is None:
        raise RuntimeError("Fallback DAE import failed: visual_scene not found.")

    instances = []
    for node in children_by_name(visual_scene, "node"):
        collect_geometry_instances(
            node,
            Matrix.Identity(4),
            unit_meter,
            instances,
            library_nodes,
            [],
        )

    mesh_cache = {}
    created_objects = 0
    skipped_instances = 0

    for instance in instances:
        geometry_id = instance["geometry_id"]
        mesh = mesh_cache.get(geometry_id)
        if mesh is None:
            geometry = geometry_nodes.get(geometry_id)
            if geometry is None:
                skipped_instances += 1
                continue
            mesh = convert_geometry_mesh(geometry, unit_meter)
            if mesh is None:
                skipped_instances += 1
                continue
            mesh_cache[geometry_id] = mesh

        obj = bpy.data.objects.new(instance["name"], mesh)
        obj.matrix_world = instance["matrix"]
        bpy.context.collection.objects.link(obj)
        created_objects += 1

    if created_objects == 0:
        raise RuntimeError("Fallback DAE import failed: no mesh objects were created.")

    return {
        "filepath": dae_path,
        "operator": "rad_fallback_dae_parser",
        "unit_meter": unit_meter,
        "geometries": len(geometry_nodes),
        "library_nodes": len(library_nodes),
        "instances": len(instances),
        "created_objects": created_objects,
        "skipped_instances": skipped_instances,
    }


def mm_to_meters(values):
    return Vector((values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0))


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
    translation = Matrix.Translation(eye)
    return translation @ rotation


def create_ai360_camera(scene_data):
    camera_data = scene_data["camera"]
    eye = mm_to_meters(camera_data["eye_mm"])
    direction = Vector(camera_data["direction"])
    up = Vector(camera_data["up"])

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
    }


def add_preview_light():
    light_data = bpy.data.lights.new("RAD_PREVIEW_SUN", "SUN")
    light_obj = bpy.data.objects.new("RAD_PREVIEW_SUN", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(35))
    light_data.energy = 2.0


def render_preview(out_dir):
    scene = bpy.context.scene
    available_engines = [
        item.identifier
        for item in scene.render.bl_rna.properties["engine"].enum_items
    ]
    for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if engine in available_engines:
            scene.render.engine = engine
            break

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
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
            scene.display.shading.color_type = "OBJECT"
            scene.display.shading.background_type = "VIEWPORT"
            scene.display.shading.background_color = (0.78, 0.78, 0.78)
        except TypeError:
            pass

    for index, obj in enumerate(obj for obj in scene.objects if obj.type == "MESH"):
        shade = 0.45 + ((index % 7) * 0.055)
        obj.color = (shade, shade, shade, 1.0)

    if scene.camera:
        scene.camera.data.clip_start = 0.01
        scene.camera.data.clip_end = 10000.0

    add_preview_light()
    preview_path = os.path.join(out_dir, "preview_camera.png")
    scene.render.filepath = preview_path
    bpy.ops.render.render(write_still=True)
    return preview_path


def render_overview(out_dir, bounds):
    if not bounds:
        return None

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
    overview_path = os.path.join(out_dir, "preview_overview.png")
    bpy.context.scene.render.filepath = overview_path
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.camera = previous_camera
    return overview_path


def write_report(out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase0b_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path


def main():
    args = parse_args()
    start = time.time()
    os.makedirs(args.out, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as handle:
        scene_data = json.load(handle)

    if scene_data.get("units") != "mm":
        raise ValueError("scene.json must use units: mm")

    dae_asset = read_dae_asset(args.dae)
    clear_scene()
    import_options = import_collada(args.dae)
    camera = create_ai360_camera(scene_data)

    blend_path = os.path.join(args.out, "phase0b_imported.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    objects = list(bpy.context.scene.objects)
    bounds = scene_bounds(objects)
    preview_path = None
    overview_path = None
    preview_error = None
    if args.render_preview:
        try:
            preview_path = render_preview(args.out)
            overview_path = render_overview(args.out, bounds)
        except Exception as exc:
            preview_error = str(exc)

    warnings = []
    if import_options.get("native_import_error"):
        warnings.append("Native Blender COLLADA importer is unavailable; fallback parser was used.")
    if import_options.get("skipped_instances"):
        warnings.append(f"Skipped {import_options['skipped_instances']} DAE instances without triangle mesh data.")

    report = {
        "status": "WARN" if warnings else "PASS",
        "warnings": warnings,
        "source": {
            "dae": args.dae,
            "json": args.json,
            "scene_name": scene_data["camera"]["scene_name"],
        },
        "dae_asset": dae_asset,
        "import_options": import_options,
        "counts": {
            "objects_total": len(objects),
            "mesh_objects": sum(1 for obj in objects if obj.type == "MESH"),
            "materials": len(bpy.data.materials),
            "cameras": len(bpy.data.cameras),
        },
        "bounds": bounds,
        "camera": {
            "name": camera.name,
            "location_m": [round(v, 6) for v in camera.location],
            "forward": [round(v, 6) for v in (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1)))],
            "up": [round(v, 6) for v in (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0)))],
            "fov_degrees": round(math.degrees(camera.data.angle), 6)
            if camera.data.type == "PERSP"
            else None,
        },
        "outputs": {
            "blend": blend_path,
            "preview": preview_path,
            "overview": overview_path,
            "preview_error": preview_error,
        },
        "duration_seconds": round(time.time() - start, 3),
    }
    report_path = write_report(args.out, report)
    print(f"RAD_PHASE0B_REPORT={report_path}")


if __name__ == "__main__":
    main()
