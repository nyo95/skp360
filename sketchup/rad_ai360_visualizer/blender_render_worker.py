"""Blender headless render worker — RGB + depth per cubemap face.

Called by the pipeline exe:
  blender --background --factory-startup --python blender_render_worker.py -- --job job.json

Per face (90° FOV, same camera eye):
  Pass 1 — EEVEE, material override (Camera Data Z Depth) → depth HDR   (fast, ~1s/face)
  Pass 2 — EEVEE, full OBJ materials                      → RGB  PNG   (~2-3s/face)

Writes:
  <output_dir>/blender_render/cubemap_<face>_depth.hdr
  <output_dir>/blender_render/cubemap_<face>_rgb.png
  <output_dir>/blender_render/render_meta.json
"""
import sys, os, json, math
from pathlib import Path

import bpy
import mathutils

FACE_NAMES = ("front", "right", "back", "left", "top", "bottom")

# (forward, up) in SketchUp/OBJ world space: X=right, Y=forward, Z=up
FACE_DIRECTIONS = {
    "front":  (( 1,  0,  0), ( 0,  0,  1)),
    "right":  (( 0, -1,  0), ( 0,  0,  1)),
    "back":   ((-1,  0,  0), ( 0,  0,  1)),
    "left":   (( 0,  1,  0), ( 0,  0,  1)),
    "top":    (( 0,  0,  1), (-1,  0,  0)),
    "bottom": (( 0,  0, -1), ( 1,  0,  0)),
}


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Pass arguments after '--'")
    rest = argv[argv.index("--") + 1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--job" and i + 1 < len(rest):
            return Path(rest[i + 1])
        i += 1
    raise RuntimeError("--job <path> required")


def inches_to_m(v):
    return float(v) / 39.3701


def import_obj(obj_path: Path):
    print(f"  Importing OBJ: {obj_path}")
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z")
    print(f"  Import done — {len(bpy.data.objects)} objects")


def set_engine(name: str):
    scene = bpy.context.scene
    for eng in (name, "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = eng
            return eng
        except TypeError:
            continue
    return scene.render.engine


def make_camera(eye_m, fwd, up, fov=90.0):
    fwd_v  = mathutils.Vector(fwd).normalized()
    up_v   = mathutils.Vector(up).normalized()
    right  = fwd_v.cross(up_v).normalized()
    up_v   = right.cross(fwd_v).normalized()
    rot = mathutils.Matrix((
        [ right.x,  up_v.x, -fwd_v.x, 0],
        [ right.y,  up_v.y, -fwd_v.y, 0],
        [ right.z,  up_v.z, -fwd_v.z, 0],
        [0, 0, 0, 1],
    ))
    cam_data       = bpy.data.cameras.new("_rad_cam")
    cam_data.type  = "PERSP"
    cam_data.angle = math.radians(fov)
    cam_obj        = bpy.data.objects.new("_rad_cam_obj", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.matrix_world = mathutils.Matrix.Translation(eye_m) @ rot
    return cam_obj


def remove_camera(cam_obj):
    bpy.data.cameras.remove(cam_obj.data, do_unlink=True)


def make_depth_material():
    mat = bpy.data.materials.new("_rad_depth_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    cam  = nodes.new("ShaderNodeCameraData")
    emit = nodes.new("ShaderNodeEmission")
    out  = nodes.new("ShaderNodeOutputMaterial")
    links.new(cam.outputs["View Z Depth"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"],    out.inputs["Surface"])
    return mat


def setup_world_light(strength=1.5):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = next((n for n in nodes if n.type == "BACKGROUND"), None)
    if bg:
        bg.inputs["Color"].default_value    = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = strength


def render_face(scene, cam_obj, tmp_dir: Path, fmt: str, ext: str) -> Path:
    scene.camera = cam_obj
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(tmp_dir / "out_")
    bpy.ops.render.render(write_still=True)
    candidates = sorted(tmp_dir.glob(f"*.{ext}"))
    if not candidates:
        candidates = sorted(tmp_dir.glob("out_*"))
    if not candidates:
        raise RuntimeError(f"No render output found in {tmp_dir}")
    return candidates[0]


def main():
    job_path = parse_args()
    job      = json.loads(job_path.read_text(encoding="utf-8-sig"))
    obj_path = Path(job["obj_path"])
    out_root = Path(job["output_dir"]) / "blender_render"
    out_root.mkdir(parents=True, exist_ok=True)

    face_size  = int(job.get("face_size", 1024))
    cam_meta   = job["camera"]
    eye_m      = mathutils.Vector([inches_to_m(v) for v in cam_meta["eye_inches"]])

    print("=== RAD AI360 Blender Render Worker ===")
    print(f"  Eye: {[round(v,3) for v in eye_m]} m")
    print(f"  Face size: {face_size}px")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_obj(obj_path)
    setup_world_light()

    scene = bpy.context.scene
    scene.render.resolution_x          = face_size
    scene.render.resolution_y          = face_size
    scene.render.resolution_percentage = 100
    scene.render.use_compositing       = False
    scene.render.use_sequencer         = False
    scene.render.film_transparent      = False

    depth_mat = make_depth_material()
    vl        = scene.view_layers[0]
    results   = {}

    for face_name in FACE_NAMES:
        fwd, up = FACE_DIRECTIONS[face_name]
        print(f"\n  --- {face_name} ---")

        depth_out = out_root / f"cubemap_{face_name}_depth.hdr"
        rgb_out   = out_root / f"cubemap_{face_name}_rgb.png"
        tmp_d     = out_root / f"_tmp_{face_name}_depth"
        tmp_r     = out_root / f"_tmp_{face_name}_rgb"

        cam_obj = make_camera(eye_m, fwd, up)

        # ── Pass 1: depth (EEVEE, material override, 1 sample) ──────────────
        if not depth_out.exists():
            set_engine("BLENDER_EEVEE_NEXT")
            scene.render.image_settings.file_format = "HDR"
            scene.render.image_settings.color_mode  = "BW"
            vl.material_override = depth_mat
            rendered = render_face(scene, cam_obj, tmp_d, "HDR", "hdr")
            rendered.rename(depth_out)
            print(f"    depth → {depth_out.name}")
        else:
            print(f"    depth → cache hit")

        # ── Pass 2: RGB (EEVEE, full materials) ─────────────────────────────
        if not rgb_out.exists():
            set_engine("BLENDER_EEVEE_NEXT")
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode  = "RGB"
            scene.render.image_settings.color_depth = "8"
            vl.material_override = None
            rendered = render_face(scene, cam_obj, tmp_r, "PNG", "png")
            rendered.rename(rgb_out)
            print(f"    rgb   → {rgb_out.name}")
        else:
            print(f"    rgb   → cache hit")

        remove_camera(cam_obj)
        results[face_name] = {
            "depth": str(depth_out),
            "rgb":   str(rgb_out),
        }

    meta = {
        "schema":    "rad-ai360-render-meta",
        "version":   1,
        "face_size": face_size,
        "depth_fmt": "HDR (Radiance), Camera Data Z Depth, near=small",
        "rgb_fmt":   "PNG 8-bit, EEVEE full materials",
        "faces":     results,
    }
    meta_path = out_root / "render_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  Render meta: {meta_path}")
    print("=== Blender render complete ===")


main()
