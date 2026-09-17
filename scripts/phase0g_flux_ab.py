#!/usr/bin/env python3
"""Run Phase 0G controlled FLUX A/B test with the existing Phase 0F backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase0f_cloudflare_flux_experiment_a as flux  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0g-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=flux.MODEL)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_one(label, source_path, out_dir, args):
    exp_dir = out_dir / label
    input_dir = exp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_copy = input_dir / "rgb_erp.png"
    source_copy.write_bytes(Path(source_path).read_bytes())

    source_width, source_height = flux.source_dimensions(source_copy)
    output_width, output_height = flux.output_size_for_source(source_width, source_height)
    reference_path = input_dir / "rgb_erp_cloudflare_reference_511.png"
    reference = flux.resize_reference_to_limit(source_copy, reference_path)
    output_path = exp_dir / "output.png"

    account_id, api_token, credential_source = flux.load_cloudflare_config()
    request_parameters = {
        "provider": "cloudflare",
        "model": args.model,
        "endpoint": "https://api.cloudflare.com/client/v4/accounts/<redacted-account-id>/ai/run/" + args.model,
        "credential_source": credential_source,
        "multipart_fields": {
            "prompt": flux.PROMPT,
            "width": str(output_width),
            "height": str(output_height),
            "guidance": str(args.guidance),
        },
        "multipart_files": {
            "input_image_0": {
                "filename": reference_path.name,
                "mime_type": "image/png",
                "source": "Phase 0G RGB ERP only",
                "sha256": reference["sha256"],
            }
        },
        "explicitly_not_sent": ["depth_condition.png", "normal_condition.png", "glass_mask.png"],
    }
    generation = {
        "experiment": label,
        "status": "PENDING",
        "provider": "cloudflare",
        "model": args.model,
        "prompt": flux.PROMPT,
        "negative_prompt": None,
        "input_image_hashes": {
            "rgb_erp": flux.sha256(source_copy),
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
    started = time.time()
    try:
        if not account_id or not api_token:
            raise RuntimeError("Cloudflare credentials are missing.")
        endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{args.model}"
        payload, headers, runtime = flux.multipart_post(
            endpoint=endpoint,
            token=api_token,
            fields=request_parameters["multipart_fields"],
            files={"input_image_0": (reference_path.name, reference_path.read_bytes(), "image/png")},
            timeout=args.timeout,
        )
        flux.decode_cloudflare_image(payload, output_path)
        generation.update({
            "status": "ok",
            "output": str(output_path),
            "output_sha256": flux.sha256(output_path),
            "provider_request_id": headers.get("cf-ray"),
            "runtime_seconds": round(runtime, 3),
            "cloudflare_success": payload.get("success"),
            "cloudflare_errors": payload.get("errors"),
            "cloudflare_messages": payload.get("messages"),
        })
    except Exception as error:
        generation.update({"status": "FAIL", "error": str(error), "output": str(output_path)})

    output_qc = flux.validate_png(output_path) if output_path.is_file() else None
    generation["local_qc"] = {
        "status": "PASS" if output_qc and output_qc["valid_image"] and output_qc["ratio_2_to_1"] and output_qc["no_unexpected_alpha"] and output_qc["has_visible_pixels"] and not output_qc["blank_warning"] else "WARNING",
        "output": output_qc,
        "geometry_drift_warning": flux.edge_similarity(source_copy, output_path) if output_path.is_file() else None,
    }
    generation["total_seconds"] = round(time.time() - started, 3)
    write_json(exp_dir / "generation.json", generation)
    return generation


def main():
    args = parse_args()
    start = time.time()
    report = read_json(args.phase0g_report)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    before = report["outputs"]["before_rgb"]
    after = report["outputs"]["after_source_fix_rgb"]
    result_a = run_one("G-A_BEFORE_RGB_ONLY", before, out_dir, args)
    result_b = run_one("G-B_AFTER_SOURCE_FIX_RGB_ONLY", after, out_dir, args)

    report["ai_ab_test"] = {
        "status": "COMPLETE" if result_a["status"] == "ok" and result_b["status"] == "ok" else "FAILED",
        "scope": "Two controlled Cloudflare FLUX.2 Klein RGB-only generations. No depth, normal, or glass mask sent.",
        "identical_settings": {
            "provider": "cloudflare",
            "model": args.model,
            "prompt": flux.PROMPT,
            "guidance": args.guidance,
            "resolution": result_a.get("resolution"),
        },
        "g_a_before": result_a,
        "g_b_after_source_fix": result_b,
        "human_review_focus": [
            "windows still read as openings",
            "glazing remains transparent / believable",
            "exterior/daylight is preserved",
            "AI does not close windows with opaque surfaces",
            "window/door frames remain correct",
            "ceiling light fixture count remains correct",
            "fixture positions remain correct",
            "artificial lighting looks plausible",
            "major furniture remains unchanged",
            "architecture remains unchanged",
            "ERP remains valid",
            "seam does not materially worsen",
        ],
        "duration_seconds": round(time.time() - start, 3),
    }
    report["status"] = "PASS" if report.get("status") == "PASS" and report["ai_ab_test"]["status"] == "COMPLETE" else "WARNING"
    report["stop_condition"] = "Stopped after glass audit, light audit, glass mask, BEFORE/AFTER RGB, two controlled FLUX A/B outputs, and Phase 0G report."
    write_json(args.phase0g_report, report)
    print(f"RAD_PHASE0G_AB_REPORT={args.phase0g_report}")
    print(f"RAD_PHASE0G_AB_STATUS={report['ai_ab_test']['status']}")
    return 0 if report["ai_ab_test"]["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
