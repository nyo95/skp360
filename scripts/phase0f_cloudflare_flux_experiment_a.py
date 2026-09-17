#!/usr/bin/env python3
"""Phase 0F: Cloudflare FLUX.2 Klein RGB-only ERP feasibility run.

This intentionally reuses the request shape from the experimental
RAD AI360 Visualizer worker without modifying that SketchUp extension.
Only the Phase 0D/0E RGB ERP is sent as an ordinary image reference.
Depth and normal are not sent as fake structural conditioning.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
PLUGIN_CONFIG = Path(
    r"C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins"
) / "rad_ai360_visualizer" / "config.json"
REFERENCE_LIMIT_PX = 511
MAX_OUTPUT_EDGE_PX = 1920

PROMPT = """Transform this existing architectural interior render into a highly photorealistic interior visualization.

Preserve the exact architecture, camera position, room proportions, wall openings, doors, ceiling geometry, furniture placement, fixture count, cabinetry proportions, and spatial layout from the source image.

Preserve the existing material colors and visual material intent.

Improve only realism: realistic material response, natural reflections, surface texture, soft indirect lighting, photographic exposure, subtle imperfections, and believable interior photography.

Do not redesign the room. Do not move objects. Do not remove objects. Do not add furniture. Do not change wall geometry. Do not change openings. Do not modify camera composition.

The source image is a 360-degree equirectangular panorama. Preserve the 2:1 equirectangular panorama layout. The left and right image boundaries represent the same 360-degree seam."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0F Experiment A with Cloudflare FLUX.2 Klein."
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--phase0e-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cloudflare_config() -> tuple[str | None, str | None, str]:
    account_id = (
        os.environ.get("CF_ACCOUNT_ID")
        or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    )
    api_token = (
        os.environ.get("CF_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
    )
    source = "environment"
    if PLUGIN_CONFIG.is_file():
        config = json.loads(PLUGIN_CONFIG.read_text(encoding="utf-8"))
        cloudflare = config.get("cloudflare", {})
        if not account_id:
            account_id = cloudflare.get("account_id")
            source = "rad_ai360_visualizer config.json"
        if not api_token:
            api_token = cloudflare.get("api_token")
            source = "rad_ai360_visualizer config.json"
    return account_id, api_token, source


def source_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def output_size_for_source(width: int, height: int) -> tuple[int, int]:
    if height <= 0 or abs((width / height) - 2.0) > 0.001:
        raise RuntimeError(f"Source ERP is not 2:1: {width}x{height}")
    output_width = min(width, MAX_OUTPUT_EDGE_PX)
    if output_width % 2:
        output_width -= 1
    return output_width, output_width // 2


def resize_reference_to_limit(source_path: Path, destination_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read source RGB ERP: {source_path}")
    height, width = image.shape[:2]
    scale = min(1.0, REFERENCE_LIMIT_PX / max(width, height))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(destination_path), resized)
    if not ok:
        raise RuntimeError(f"Could not write AI reference: {destination_path}")
    ref_height, ref_width = resized.shape[:2]
    return {
        "path": str(destination_path),
        "source_dimensions": [width, height],
        "reference_dimensions": [ref_width, ref_height],
        "resize_limit_px": REFERENCE_LIMIT_PX,
        "sha256": sha256(destination_path),
    }


def multipart_post(
    endpoint: str,
    token: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, str], float]:
    boundary = "----rad-ai360-phase0f-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, content, mime_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {mime_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(endpoint, data=b"".join(chunks), method="POST")
    request.add_header("Authorization", "Bearer " + token)
    request.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    request.add_header("accept", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cloudflare API connection failed: {error}") from error
    duration = time.time() - started
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Cloudflare response was not JSON") from error
    return payload, headers, duration


def decode_cloudflare_image(payload: dict[str, Any], output_path: Path) -> None:
    result = payload.get("result", payload)
    encoded = result.get("image") if isinstance(result, dict) else None
    if isinstance(encoded, str) and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Cloudflare response did not contain a base64 image: "
            + json.dumps(payload)[:2000]
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Cloudflare output was written but is not a valid image")


def validate_png(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"path": str(path), "exists": path.is_file(), "valid_image": False}
    height, width = image.shape[:2]
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        alpha = None
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = image[:, :, 3]
    else:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = None
    luminance = rgb.astype(np.float32).mean(axis=2)
    seam = np.abs(
        rgb[:, 0, :].astype(np.float32) - rgb[:, width - 1, :].astype(np.float32)
    ).mean(axis=1)
    alpha_min = float(alpha.min() / 255.0) if alpha is not None else 1.0
    alpha_max = float(alpha.max() / 255.0) if alpha is not None else 1.0
    return {
        "path": str(path),
        "exists": path.is_file(),
        "valid_image": True,
        "dimensions": [int(width), int(height)],
        "ratio": round(width / height, 6) if height else None,
        "ratio_2_to_1": height > 0 and abs((width / height) - 2.0) <= 0.001,
        "alpha_min": round(alpha_min, 6),
        "alpha_max": round(alpha_max, 6),
        "no_unexpected_alpha": alpha_min >= 0.999 and alpha_max >= 0.999,
        "blank_warning": bool(float(luminance.max() - luminance.min()) < 1.0),
        "has_visible_pixels": bool(float(luminance.max()) > 1.0),
        "mean_rgb_seam_delta": round(float(seam.mean()) / 255.0, 6),
        "max_rgb_seam_delta": round(float(seam.max()) / 255.0, 6),
        "top_pole_has_visible_pixels": bool(float(luminance[0, :].max()) > 1.0),
        "bottom_pole_has_visible_pixels": bool(float(luminance[-1, :].max()) > 1.0),
    }


def edge_similarity(source_path: Path, output_path: Path) -> dict[str, Any]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if source is None or output is None:
        return {"status": "UNAVAILABLE"}
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if source_gray.shape != output_gray.shape:
        output_gray = cv2.resize(
            output_gray,
            (source_gray.shape[1], source_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    source_edges = cv2.Canny(source_gray, 80, 160)
    output_edges = cv2.Canny(output_gray, 80, 160)
    diff = cv2.absdiff(source_edges, output_edges)
    source_density = float(np.count_nonzero(source_edges)) / float(source_edges.size)
    output_density = float(np.count_nonzero(output_edges)) / float(output_edges.size)
    diff_density = float(np.count_nonzero(diff)) / float(diff.size)
    return {
        "status": "WARNING_SIGNAL_ONLY",
        "method": "Canny edge density/difference after resizing output to source dimensions when needed.",
        "source_edge_density": round(source_density, 6),
        "output_edge_density": round(output_density, 6),
        "edge_difference_density": round(diff_density, 6),
        "note": "This is not semantic architecture understanding; use as a local drift warning only.",
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = args.out
    input_dir = out_dir / "input"
    exp_dir = out_dir / "experiment_a_rgb_only"
    input_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    phase0e = load_json(args.phase0e_report)
    source_rgb = Path(phase0e["inputs"]["files"]["rgb_erp"]["path"])
    rgb_copy = input_dir / "rgb_erp.png"
    if source_rgb.resolve() != rgb_copy.resolve():
        rgb_copy.write_bytes(source_rgb.read_bytes())

    source_width, source_height = source_dimensions(rgb_copy)
    output_width, output_height = output_size_for_source(source_width, source_height)
    reference_path = input_dir / "rgb_erp_cloudflare_reference_511.png"
    reference = resize_reference_to_limit(rgb_copy, reference_path)

    output_path = exp_dir / "output.png"
    generation_path = exp_dir / "generation.json"
    report_path = out_dir / "phase0f_report.json"

    account_id, api_token, credential_source = load_cloudflare_config()
    request_parameters = {
        "provider": "cloudflare",
        "model": args.model,
        "endpoint": (
            "https://api.cloudflare.com/client/v4/accounts/"
            "<redacted-account-id>/ai/run/" + args.model
        ),
        "credential_source": credential_source,
        "multipart_fields": {
            "prompt": PROMPT,
            "width": str(output_width),
            "height": str(output_height),
            "guidance": str(args.guidance),
        },
        "multipart_files": {
            "input_image_0": {
                "filename": reference_path.name,
                "mime_type": "image/png",
                "source": "Phase 0D/0E RGB ERP only",
                "sha256": reference["sha256"],
            }
        },
        "explicitly_not_sent": ["depth_condition.png", "normal_condition.png"],
    }

    generation: dict[str, Any] = {
        "experiment": "Experiment A — RGB-only ERP baseline",
        "status": "PENDING",
        "provider": "cloudflare",
        "model": args.model,
        "prompt": PROMPT,
        "negative_prompt": None,
        "input_image_hashes": {
            "rgb_erp": sha256(rgb_copy),
            "cloudflare_reference_511": reference["sha256"],
        },
        "seed": None,
        "strength_or_denoise": None,
        "control_weights": None,
        "resolution": [output_width, output_height],
        "request_parameters": request_parameters,
        "provider_request_id": None,
        "runtime_seconds": None,
        "estimated_or_reported_cost": None,
    }

    try:
        if not account_id or not api_token:
            raise RuntimeError(
                "Cloudflare credentials are missing. Set CF_ACCOUNT_ID/CF_API_TOKEN "
                "or provide rad_ai360_visualizer/config.json."
            )
        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{args.model}"
        )
        payload, headers, runtime = multipart_post(
            endpoint=endpoint,
            token=api_token,
            fields=request_parameters["multipart_fields"],
            files={
                "input_image_0": (
                    reference_path.name,
                    reference_path.read_bytes(),
                    "image/png",
                )
            },
            timeout=args.timeout,
        )
        decode_cloudflare_image(payload, output_path)
        generation.update(
            {
                "status": "ok",
                "output": str(output_path),
                "output_sha256": sha256(output_path),
                "provider_request_id": headers.get("cf-ray"),
                "runtime_seconds": round(runtime, 3),
                "cloudflare_success": payload.get("success"),
                "cloudflare_errors": payload.get("errors"),
                "cloudflare_messages": payload.get("messages"),
            }
        )
    except Exception as error:
        generation.update(
            {
                "status": "FAIL",
                "error": str(error),
                "output": str(output_path),
            }
        )

    write_json(generation_path, generation)

    source_qc = validate_png(rgb_copy)
    output_qc = validate_png(output_path) if output_path.is_file() else None
    drift = edge_similarity(rgb_copy, output_path) if output_path.is_file() else None
    qc_status = "NOT_RUN"
    if output_qc:
        qc_status = (
            "PASS"
            if output_qc["valid_image"]
            and output_qc["ratio_2_to_1"]
            and output_qc["no_unexpected_alpha"]
            and output_qc["has_visible_pixels"]
            and not output_qc["blank_warning"]
            else "WARNING"
        )

    report = {
        "schema": "rad-ai360-phase0f-cloudflare-flux-rgb-only",
        "version": 1,
        "status": "COMPLETE" if generation["status"] == "ok" else "FAILED",
        "phase": "0F",
        "scope": "Experiment A RGB-only ERP only; no depth/normal conditioning.",
        "source": {
            "phase0e_report": str(args.phase0e_report),
            "rgb_erp": str(rgb_copy),
            "source_dimensions": [source_width, source_height],
            "source_sha256": sha256(rgb_copy),
            "cloudflare_reference": reference,
        },
        "experiment_a_rgb_only": generation,
        "local_qc": {
            "status": qc_status,
            "source_rgb_erp": source_qc,
            "output": output_qc,
            "geometry_drift_warning": drift,
        },
        "request_preservation": {
            "generation_json": str(generation_path),
            "all_non_secret_request_parameters_recorded": True,
            "secrets_recorded": False,
            "output_hash_recorded": output_path.is_file(),
        },
        "stopped_after": "Experiment A RGB-only ERP",
        "duration_seconds": round(time.time() - started, 3),
    }
    write_json(report_path, report)
    print(f"RAD_PHASE0F_REPORT={report_path}")
    print(f"RAD_PHASE0F_STATUS={report['status']}")
    return 0 if generation["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
