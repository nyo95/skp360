"""Blender headless depth renderer for RAD AI360 Hybrid pipeline.

Called by ai360_worker.py via subprocess:
  blender.exe --background --python blender_depth_worker.py -- --job job.json

Imports the OBJ, sets up 6 cubemap cameras, renders Z-depth EXR per face.
Depth EXR channel "R" = distance in Blender units (metres).
Writes: <output_dir>/depth/cubemap_{face}.exr  (x6)
        <output_dir>/depth/depth_meta.json
"""
import sys, os, json, math
from pathlib import Path

# ── Blender is available only when running inside blender --background ─────────
import bpy
import mathutils

FACE_NAMES = ("front", "right", "back", "left", "top", "bottom")


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Pass arguments after '--': blender ... -- --job JOB")
    rest = argv[argv.index("--") + 1:]
    job_path = None
    i = 0
    while i < len(rest):
        if rest[i] == "--job" and i + 1 < len(rest):
            job_path = Path(rest[i + 1])
            i += 2
        else:
            i += 1
    if not job_path:
        raise RuntimeError("--job <path> is required")
    return job_path


def load_job(job_path):
    return json.loads(job_path.read_text(encoding="utf-8-sig"))


def inches_to_m(v):
    return v / 39.3701


def setup_scene_for_depth():
    """Set up scene for depth rendering using a Camera Data material override.

    Avoids compositor nodes entirely. Renders Z-depth directly as RGB float EXR:
    R = G = B = Z distance in scene units (metres).
    """
    scene = bpy.context.scene

    # Use EEVEE Next (fastest headless, no GPU needed for depth)
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue

    if "CYCLES" in scene.render.engine:
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False

    scene.render.film_transparent = False
    scene.render.use_compositing  = False
    scene.render.use_sequencer    = False

    # Output: HDR (Radiance RGBE) — float32, readable by cv2 on Windows without OpenEXR
    scene.render.image_settings.file_format = "HDR"
    scene.render.image_settings.color_mode  = "BW"

    # Create depth material: Camera Data Z Depth → Emission → surface
    mat = bpy.data.materials.new(name="_rad_depth_override")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    cam_node  = nodes.new("ShaderNodeCameraData")
    emit_node = nodes.new("ShaderNodeEmission")
    out_node  = nodes.new("ShaderNodeOutputMaterial")

    links.new(cam_node.outputs["View Z Depth"], emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"],    out_node.inputs["Surface"])

    # Apply as view-layer material override (affects all objects, no scene changes)
    vl = scene.view_layers[0]
    vl.material_override = mat

    return mat


def import_obj(obj_path):
    print(f"  Importing OBJ: {obj_path}")
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z")


def make_cube_camera(eye_m, direction, up, fov_deg=90.0):
    """Create a camera at eye_m looking in direction with given up vector."""
    cam_data = bpy.data.cameras.new("depth_cam")
    cam_data.type = "PERSP"
    cam_data.angle = math.radians(fov_deg)
    cam_obj = bpy.data.objects.new("depth_cam_obj", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Build rotation from direction + up
    fwd = mathutils.Vector(direction).normalized()
    up_v = mathutils.Vector(up).normalized()
    right = fwd.cross(up_v).normalized()
    up_v  = right.cross(fwd).normalized()   # reorthogonalise

    # Blender camera looks down -Z, with +Y up, +X right
    # Map: cam_-Z → world_fwd, cam_+Y → world_up, cam_+X → world_right
    rot_matrix = mathutils.Matrix((
        [right.x, up_v.x, -fwd.x, 0],
        [right.y, up_v.y, -fwd.y, 0],
        [right.z, up_v.z, -fwd.z, 0],
        [0,       0,       0,     1],
    ))
    cam_obj.matrix_world = mathutils.Matrix.Translation(eye_m) @ rot_matrix
    return cam_obj


FACE_DIRECTIONS = {
    # (forward, up) — in SketchUp world space, passed straight through
    # (no axis swap: SketchUp X=right, Y=forward, Z=up same as Blender with OBJ Z-up import)
    "front":  (( 1,  0,  0), (0, 0, 1)),
    "right":  (( 0, -1,  0), (0, 0, 1)),
    "back":   ((-1,  0,  0), (0, 0, 1)),
    "left":   (( 0,  1,  0), (0, 0, 1)),
    "top":    (( 0,  0,  1), (-1, 0, 0)),
    "bottom": (( 0,  0, -1), ( 1, 0, 0)),
}


def render_depth_faces(job):
    cam_meta   = job["camera"]
    eye_inches = cam_meta["eye_inches"]
    eye_m      = mathutils.Vector([inches_to_m(v) for v in eye_inches])

    face_size  = int(job.get("face_size", 1024))
    output_dir = Path(job["output_dir"]) / "depth"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x          = face_size
    scene.render.resolution_y          = face_size
    scene.render.resolution_percentage = 100

    results = {}
    for face_name in FACE_NAMES:
        fwd, up = FACE_DIRECTIONS[face_name]

        cam_obj = make_cube_camera(eye_m, fwd, up, fov_deg=90.0)
        scene.camera = cam_obj

        # Blender appends frame number; use a temp subdir then rename
        tmp_dir = output_dir / f"_tmp_{face_name}"
        tmp_dir.mkdir(exist_ok=True)
        scene.render.filepath = str(tmp_dir / "depth_")

        print(f"  Rendering depth: {face_name} ...")
        bpy.ops.render.render(write_still=True)

        # Find whatever Blender wrote (depth_0001.hdr or depth_.hdr etc.)
        candidates = sorted(tmp_dir.glob("*.hdr")) + sorted(tmp_dir.glob("*.exr"))
        final = output_dir / f"cubemap_{face_name}.hdr"
        if candidates:
            candidates[0].rename(final)
            print(f"    -> {final} (from {candidates[0].name})")
            results[face_name] = str(final)
        else:
            raise RuntimeError(f"No HDR/EXR found in {tmp_dir} after render")

        bpy.data.objects.remove(cam_obj, do_unlink=True)

    meta = {
        "schema": "rad-ai360-depth-meta", "version": 1,
        "face_size": face_size,
        "depth_unit": "metres (Blender)",
        "format": "HDR (Radiance RGBE)",
        "channel": "R (= G = B, grayscale Camera Data Z Depth)",
        "convention": "Z-distance (near = small value)",
        "inversion_note": "Apply pixel=255-pixel after E_PERCENTILE_CLAMPED to get near=white for ControlNet",
        "faces": results,
    }
    (output_dir / "depth_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Depth meta: {output_dir / 'depth_meta.json'}")
    return results


def main():
    job_path   = parse_args()
    job        = load_job(job_path)
    obj_path   = Path(job["obj_path"])

    print("=== RAD AI360 Blender Depth Worker ===")
    print(f"  OBJ:    {obj_path}")
    print(f"  Output: {job['output_dir']}")

    # Fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_obj(obj_path)
    setup_scene_for_depth()
    render_depth_faces(job)
    print("=== Depth render complete ===")


main()
