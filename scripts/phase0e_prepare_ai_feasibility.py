import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time

import bpy


SCHEMA = "rad-ai360-phase0e-ai-render-feasibility"
MATERIAL_SCHEMA = "rad-ai360-material-manifest"
DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 20.0


RGB_PROMPT = """Transform this existing architectural interior render into a highly photorealistic interior visualization.

Preserve the exact architecture, camera position, room proportions, wall openings, doors, ceiling geometry, furniture placement, fixture count, cabinetry proportions, and spatial layout from the source image.

Preserve the existing material colors and visual material intent.

Improve only realism: realistic material response, natural reflections, surface texture, soft indirect lighting, photographic exposure, subtle imperfections, and believable interior photography.

Do not redesign the room. Do not move objects. Do not remove objects. Do not add furniture. Do not change wall geometry. Do not change openings. Do not modify camera composition.

The source image is a 360-degree equirectangular panorama. Preserve the 2:1 equirectangular panorama layout. The left and right image boundaries represent the same 360-degree seam."""


HUMAN_REVIEW_CHECKLIST = [
    "walls remain unchanged",
    "openings remain unchanged",
    "doors remain unchanged",
    "ceiling remains unchanged",
    "furniture count remains unchanged",
    "furniture positions remain unchanged",
    "cabinetry remains unchanged",
    "bathroom fixtures remain unchanged",
    "major colors/material intent remain unchanged",
    "camera composition remains unchanged",
    "pole regions remain usable",
    "0°/360° seam remains usable",
    "realism materially improved",
]


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected arguments after --")

    parser = argparse.ArgumentParser(
        description="Prepare deterministic Phase 0E AI feasibility inputs and local QC report."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--phase0d-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth-near", type=float, default=DEPTH_NEAR_M)
    parser.add_argument("--depth-far", type=float, default=DEPTH_FAR_M)
    parser.add_argument("--experiment-a-output")
    parser.add_argument("--experiment-b-output")
    parser.add_argument("--experiment-c-output")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def ensure_dirs(out_dir):
    for rel in (
        "input",
        "experiment_a_rgb_only",
        "experiment_b_depth",
        "experiment_c_depth_normal",
        "perspective_control",
    ):
        os.makedirs(os.path.join(out_dir, rel), exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)
    return target


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(obj_path):
    start = time.time()
    bpy.ops.wm.obj_import(
        filepath=obj_path,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
        use_split_objects=True,
        use_split_groups=False,
        validate_meshes=True,
    )
    return round(time.time() - start, 3)


def node_input_value(node, input_name):
    socket = node.inputs.get(input_name)
    if socket is None:
        return None
    value = socket.default_value
    if hasattr(value, "__len__"):
        return [round(float(v), 6) for v in value]
    return round(float(value), 6)


def principled_node(material):
    if not material.use_nodes or not material.node_tree:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def image_texture_paths(material, project_root):
    paths = []
    if not material.use_nodes or not material.node_tree:
        return paths
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or not node.image:
            continue
        raw_path = bpy.path.abspath(node.image.filepath) if node.image.filepath else node.image.name
        if raw_path and os.path.exists(raw_path):
            try:
                rel = os.path.relpath(raw_path, project_root)
                if not rel.startswith(".."):
                    raw_path = rel.replace("\\", "/")
            except ValueError:
                pass
        if raw_path and raw_path not in paths:
            paths.append(raw_path)
    return paths


def extract_material_manifest(obj_path, project_root):
    clean_scene()
    import_duration = import_obj(obj_path)

    materials = []
    source_materials = sorted(
        [mat for mat in bpy.data.materials if not mat.name.startswith("RAD_")],
        key=lambda mat: mat.name.lower(),
    )

    for index, material in enumerate(source_materials, start=1):
        node = principled_node(material)
        base_color = None
        opacity = None
        if node:
            base_color = node_input_value(node, "Base Color")
            alpha_value = node_input_value(node, "Alpha")
            opacity = alpha_value if isinstance(alpha_value, float) else None
        if base_color is None:
            base_color = [round(float(v), 6) for v in material.diffuse_color[:4]]
        if opacity is None:
            opacity = round(float(material.diffuse_color[3]), 6)

        textures = image_texture_paths(material, project_root)
        record = {
            "id": f"MAT_{index:03d}",
            "source_name": material.name,
            "base_color": base_color[:3],
            "opacity": opacity,
        }
        if textures:
            record["texture"] = textures[0]
            if len(textures) > 1:
                record["additional_textures"] = textures[1:]
        materials.append(record)

    return {
        "schema": MATERIAL_SCHEMA,
        "version": 1,
        "source": {
            "obj": obj_path,
            "extraction": "Blender native OBJ import material data; no AI interpretation.",
            "import_duration_seconds": import_duration,
        },
        "identity_rule": "Material ID is deterministic by sorted imported source material name for this source OBJ.",
        "materials": materials,
    }


def save_image_png(image, path, color_mode="RGB"):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = "8"
    image.filepath_raw = path
    image.save()


def image_pixels_rgba(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = list(image.pixels)
    return image, int(width), int(height), pixels


def create_depth_condition(depth_exr_path, target_png_path, near_m, far_m):
    image, width, height, pixels = image_pixels_rgba(depth_exr_path)
    total = width * height
    out_pixels = [0.0] * (total * 4)
    sampled_depths = []
    clipped_near = 0
    clipped_far = 0
    invalid = 0
    denom = far_m - near_m
    if denom <= 0:
        raise ValueError("depth far must be greater than near")

    for i in range(total):
        depth = float(pixels[i * 4])
        if not math.isfinite(depth):
            depth = far_m
            invalid += 1
        sampled_depths.append(depth)
        if depth <= near_m:
            clipped_near += 1
            mapped = 1.0
        elif depth >= far_m:
            clipped_far += 1
            mapped = 0.0
        else:
            mapped = 1.0 - ((depth - near_m) / denom)
        out_pixels[i * 4] = mapped
        out_pixels[i * 4 + 1] = mapped
        out_pixels[i * 4 + 2] = mapped
        out_pixels[i * 4 + 3] = 1.0

    output = bpy.data.images.new("RAD_PHASE0E_DEPTH_CONDITION", width, height, alpha=True, float_buffer=False)
    output.pixels.foreach_set(out_pixels)
    save_image_png(output, target_png_path, "BW")
    bpy.data.images.remove(output)

    finite_depths = [v for v in sampled_depths if math.isfinite(v)]
    return {
        "path": target_png_path,
        "mapping": "linear inverse depth for image conditioning: near -> white, far -> black, clamped",
        "near_m": near_m,
        "far_m": far_m,
        "source_exr": depth_exr_path,
        "source_min_m": round(min(finite_depths), 6) if finite_depths else None,
        "source_max_m": round(max(finite_depths), 6) if finite_depths else None,
        "source_mean_m": round(statistics.fmean(finite_depths), 6) if finite_depths else None,
        "clipped_near_pixels": clipped_near,
        "clipped_far_pixels": clipped_far,
        "invalid_pixels": invalid,
    }


def create_normal_condition(normal_exr_path, target_png_path):
    image, width, height, pixels = image_pixels_rgba(normal_exr_path)
    total = width * height
    out_pixels = [0.0] * (total * 4)
    for i in range(total):
        out_pixels[i * 4] = min(1.0, max(0.0, float(pixels[i * 4])))
        out_pixels[i * 4 + 1] = min(1.0, max(0.0, float(pixels[i * 4 + 1])))
        out_pixels[i * 4 + 2] = min(1.0, max(0.0, float(pixels[i * 4 + 2])))
        out_pixels[i * 4 + 3] = 1.0
    output = bpy.data.images.new("RAD_PHASE0E_NORMAL_CONDITION", width, height, alpha=True, float_buffer=False)
    output.pixels.foreach_set(out_pixels)
    save_image_png(output, target_png_path, "RGB")
    bpy.data.images.remove(output)
    return {
        "path": target_png_path,
        "source_exr": normal_exr_path,
        "normal_space": "Blender world/shading-space normal encoded from Phase 0D",
        "channel_interpretation": "RGB = normal * 0.5 + 0.5",
        "conversion": "32-bit EXR values clamped to [0, 1] and saved as 8-bit PNG reference/condition image.",
    }


def validate_png(path):
    image, width, height, pixels = image_pixels_rgba(path)
    total = width * height
    alpha_values = pixels[3::4]
    has_visible = False
    luminance_samples = []
    seam_samples = []
    top_visible = False
    bottom_visible = False
    for y in range(height):
        left = (y * width) * 4
        right = (y * width + width - 1) * 4
        seam_samples.append(
            (
                abs(pixels[left] - pixels[right])
                + abs(pixels[left + 1] - pixels[right + 1])
                + abs(pixels[left + 2] - pixels[right + 2])
            )
            / 3.0
        )
    for i in range(total):
        r, g, b, a = pixels[i * 4 : i * 4 + 4]
        if a > 0.001 and (r > 0.001 or g > 0.001 or b > 0.001):
            has_visible = True
        luminance_samples.append((r + g + b) / 3.0)
    for x in range(width):
        top = x * 4
        bottom = ((height - 1) * width + x) * 4
        if pixels[top + 3] > 0.001 and sum(pixels[top : top + 3]) > 0.001:
            top_visible = True
        if pixels[bottom + 3] > 0.001 and sum(pixels[bottom : bottom + 3]) > 0.001:
            bottom_visible = True

    return {
        "path": path,
        "dimensions": [width, height],
        "ratio": round(width / height, 6) if height else None,
        "ratio_2_to_1": height > 0 and abs((width / height) - 2.0) <= 0.001,
        "alpha_min": round(min(alpha_values), 6) if alpha_values else None,
        "alpha_max": round(max(alpha_values), 6) if alpha_values else None,
        "has_visible_pixels": has_visible,
        "blank_warning": (max(luminance_samples) - min(luminance_samples)) < 0.001 if luminance_samples else True,
        "mean_rgb_seam_delta": round(statistics.fmean(seam_samples), 6) if seam_samples else None,
        "max_rgb_seam_delta": round(max(seam_samples), 6) if seam_samples else None,
        "top_pole_has_visible_pixels": top_visible,
        "bottom_pole_has_visible_pixels": bottom_visible,
    }


def edge_similarity_warning(source_path, output_path):
    if not os.path.exists(output_path):
        return None
    src, width, height, src_pixels = image_pixels_rgba(source_path)
    out, out_width, out_height, out_pixels = image_pixels_rgba(output_path)
    if [width, height] != [out_width, out_height]:
        return {
            "status": "UNSUPPORTED_DIMENSION_MISMATCH",
            "source_dimensions": [width, height],
            "output_dimensions": [out_width, out_height],
        }

    # Lightweight warning signal only: compare Sobel-like luminance gradient magnitude on a coarse grid.
    step = max(1, min(width, height) // 256)

    def lum(pixels, x, y):
        idx = (y * width + x) * 4
        return (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3.0

    diffs = []
    for y in range(step, height - step, step):
        for x in range(step, width - step, step):
            src_grad = abs(lum(src_pixels, x + step, y) - lum(src_pixels, x - step, y)) + abs(lum(src_pixels, x, y + step) - lum(src_pixels, x, y - step))
            out_grad = abs(lum(out_pixels, x + step, y) - lum(out_pixels, x - step, y)) + abs(lum(out_pixels, x, y + step) - lum(out_pixels, x, y - step))
            diffs.append(abs(src_grad - out_grad))
    mean_delta = statistics.fmean(diffs) if diffs else 0.0
    return {
        "status": "WARNING_SIGNAL_ONLY",
        "method": "coarse luminance gradient magnitude difference; not semantic architecture understanding",
        "sample_step_px": step,
        "mean_gradient_delta": round(mean_delta, 6),
    }


def file_manifest(paths):
    result = {}
    for key, path in paths.items():
        result[key] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
        }
    return result


def generation_stub(path, experiment_name, status, reason, prompt=None, inputs=None):
    data = {
        "experiment": experiment_name,
        "status": status,
        "reason": reason,
        "provider": None,
        "model": None,
        "prompt": prompt,
        "negative_prompt": None,
        "input_image_hashes": inputs or {},
        "seed": None,
        "strength_or_denoise": None,
        "control_weights": None,
        "resolution": None,
        "runtime_seconds": None,
        "provider_request_id": None,
        "estimated_or_reported_cost": None,
    }
    write_json(path, data)
    return data


def optional_ai_output_qc(rgb_source_path, output_path):
    if not output_path:
        return {
            "status": "NOT_PROVIDED",
            "reason": "No AI output path was provided for local deterministic QC.",
        }
    if not os.path.exists(output_path):
        return {
            "status": "MISSING",
            "path": output_path,
        }
    png_qc = validate_png(output_path)
    return {
        "status": "PASS" if (
            png_qc["ratio_2_to_1"]
            and png_qc["has_visible_pixels"]
            and png_qc["alpha_min"] is not None
            and png_qc["alpha_min"] >= 0.999
            and not png_qc["blank_warning"]
        ) else "WARNING",
        "output_validity": png_qc,
        "edge_similarity": edge_similarity_warning(rgb_source_path, output_path),
        "note": "Deterministic warning signals only; final architecture fidelity judgment remains human review.",
    }


def main():
    args = parse_args()
    start = time.time()
    ensure_dirs(args.out)

    phase0d = load_json(args.phase0d_report)
    input_dir = os.path.join(args.out, "input")
    rgb_source = phase0d["outputs"]["rgb"]
    depth_source = phase0d["outputs"]["depth"]
    normal_source = phase0d["outputs"]["normal"]

    rgb_target = copy_file(rgb_source, os.path.join(input_dir, "rgb_erp.png"))
    depth_target = os.path.join(input_dir, "depth_condition.png")
    normal_target = os.path.join(input_dir, "normal_condition.png")
    material_manifest_path = os.path.join(input_dir, "material_manifest.json")

    material_manifest = extract_material_manifest(args.obj, args.project_root)
    write_json(material_manifest_path, material_manifest)

    depth_condition = create_depth_condition(depth_source, depth_target, args.depth_near, args.depth_far)
    normal_condition = create_normal_condition(normal_source, normal_target)

    input_paths = {
        "rgb_erp": rgb_target,
        "depth_condition": depth_target,
        "normal_condition": normal_target,
        "material_manifest": material_manifest_path,
    }
    input_manifest = file_manifest(input_paths)

    rgb_qc = validate_png(rgb_target)
    depth_qc = validate_png(depth_target)
    normal_qc = validate_png(normal_target)

    experiment_a = generation_stub(
        os.path.join(args.out, "experiment_a_rgb_only", "generation.json"),
        "Experiment A — RGB-only ERP baseline",
        "NOT_RUN",
        "No project-integrated AI image backend was found in D:\\Projects\\skpto360 during Phase 0E preparation. The experiment is defined and ready, but no output is fabricated.",
        RGB_PROMPT,
        {"rgb_erp": input_manifest["rgb_erp"]["sha256"]},
    )
    experiment_b = generation_stub(
        os.path.join(args.out, "experiment_b_depth", "generation.json"),
        "Experiment B — true depth conditioning",
        "UNSUPPORTED",
        "No selected AI stack with genuine depth/control conditioning is available in the project. A generic second reference image is not counted as structural conditioning.",
        RGB_PROMPT,
        {
            "rgb_erp": input_manifest["rgb_erp"]["sha256"],
            "depth_condition": input_manifest["depth_condition"]["sha256"],
        },
    )
    experiment_c = generation_stub(
        os.path.join(args.out, "experiment_c_depth_normal", "generation.json"),
        "Experiment C — true depth + normal conditioning",
        "UNSUPPORTED",
        "No selected AI stack with genuine depth and normal structural conditioning is available in the project.",
        RGB_PROMPT,
        {
            "rgb_erp": input_manifest["rgb_erp"]["sha256"],
            "depth_condition": input_manifest["depth_condition"]["sha256"],
            "normal_condition": input_manifest["normal_condition"]["sha256"],
        },
    )

    report = {
        "schema": SCHEMA,
        "version": 1,
        "status": "PREPARED_NO_AI_BACKEND",
        "project_root": args.project_root,
        "phase": "0E",
        "goal": "Assess whether AI can improve realism while preserving authoritative SketchUp-derived geometry and camera.",
        "source_of_truth": {
            "geometry": "SketchUp OBJ imported by Blender; Phase 0D native equirectangular render is the visual source.",
            "camera": "scene.json via Phase 0D report",
            "phase0d_report": args.phase0d_report,
            "scene_name": phase0d["source"]["scene_name"],
            "projection": phase0d["render"]["projection"],
            "resolution": phase0d["render"]["resolution"],
            "camera_basis": phase0d["camera_basis"],
        },
        "inputs": {
            "files": input_manifest,
            "depth_condition": depth_condition,
            "normal_condition": normal_condition,
            "material_manifest": {
                "path": material_manifest_path,
                "material_count": len(material_manifest["materials"]),
                "rule": material_manifest["identity_rule"],
            },
        },
        "experiments": {
            "a_rgb_only": experiment_a,
            "b_depth": experiment_b,
            "c_depth_normal": experiment_c,
            "perspective_control": {
                "status": "NOT_RUN",
                "reason": "Direct ERP AI generation was not run, so the optional perspective fallback comparison is not triggered.",
            },
        },
        "local_qc": {
            "input_rgb_erp": rgb_qc,
            "depth_condition": depth_qc,
            "normal_condition": normal_qc,
            "geometry_drift_qc": {
                "status": "READY",
                "method": "Output validity, ERP ratio, seam metrics, and coarse edge-similarity warning against source image.",
                "a_rgb_only": optional_ai_output_qc(rgb_target, args.experiment_a_output),
                "b_depth": optional_ai_output_qc(rgb_target, args.experiment_b_output),
                "c_depth_normal": optional_ai_output_qc(rgb_target, args.experiment_c_output),
            },
        },
        "pass_fail_logic": {
            "phase0e_not_auto_pass": True,
            "strong_candidate_requires": [
                "realism improves significantly",
                "major geometry remains faithful",
                "camera remains faithful",
                "output remains usable as panorama",
            ],
            "current_gate_result": "BLOCKED_FOR_AI_GENERATION_BACKEND",
        },
        "human_review_checklist": [{"label": item, "checked": False} for item in HUMAN_REVIEW_CHECKLIST],
        "notes": [
            "No GPT/Astra/Claude/VLM/material semantic classifier was added.",
            "No Cloudflare/FLUX/image-generation provider code was added to the project.",
            "Phase 0D raw metric depth EXR and normal EXR were not overwritten.",
            "Cubemap remains compatibility/debug only; Phase 0E input uses native ERP from Phase 0D.",
        ],
        "duration_seconds": round(time.time() - start, 3),
    }
    report_path = os.path.join(args.out, "phase0e_report.json")
    write_json(report_path, report)
    print(f"RAD_PHASE0E_REPORT={report_path}")


if __name__ == "__main__":
    main()
