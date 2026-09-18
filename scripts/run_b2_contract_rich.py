from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import controlled_core_harness as harness
import phase0f_cloudflare_flux_experiment_a as flux


BAKEOFF = ROOT / "output" / "generation_bakeoff"
B2_DIR = BAKEOFF / "B2_contract_rich"
B1_DIR = BAKEOFF / "B1_controlled_harness"
PACKAGE_DIR = ROOT / "output" / "condition_package"
ALBEDO = ROOT / "output" / "fast_passes" / "erp" / "albedo.png"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def image_dimensions(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return [int(image.shape[1]), int(image.shape[0])]


def build_contract_snapshot() -> None:
    contracts = B2_DIR / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    sources = {
        "scene.json": PACKAGE_DIR / "semantics" / "scene_contract.json",
        "materials.json": PACKAGE_DIR / "semantics" / "materials.json",
        "lighting.json": PACKAGE_DIR / "semantics" / "lights.json",
    }
    for name, source in sources.items():
        shutil.copy2(source, contracts / name)

    glazing = {
        "schema": "rad-ai360-b2-glazing-contract",
        "source": "condition_package/semantics/scene_contract.json",
        "regions": json.loads(sources["scene.json"].read_text(encoding="utf-8"))["regions"],
        "uncertainty": "Unknown through-glass regions remain unconstrained; known scene geometry is preserved.",
    }
    geometry = {
        "schema": "rad-ai360-b2-geometry-contract",
        "source": "condition_package/semantics/scene_contract.json",
        "preserve": {
            "CRITICAL": ["wall boundaries", "openings", "doors/windows", "ceiling/scallop geometry", "stairs", "fixed cabinetry", "major furniture placement", "custom furniture shapes"],
            "HIGH": ["fixture count", "dining table/chair count", "TV wall composition", "bathroom fixture placement"],
            "SOFT": ["cushions", "small accessories", "plants", "micro decorative details"],
        },
        "policy": "Critical and high geometry outrank decorative realism; do not redesign or relocate source elements.",
    }
    provenance = {
        "schema": "rad-ai360-b2-provenance",
        "source_albedo": str(ALBEDO),
        "source_albedo_sha256": harness.sha256(ALBEDO),
        "b1_reference": str(B1_DIR),
        "projection_frozen": True,
        "depth_control": "NOT USED: provider structural depth control is not independently proven for this B2.",
        "provider": flux.MODEL,
    }
    write_json(contracts / "glazing.json", glazing)
    write_json(contracts / "geometry.json", geometry)
    write_json(contracts / "provenance.json", provenance)


def main() -> int:
    required = [ALBEDO, PACKAGE_DIR / "manifest.json", B1_DIR / "output.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing B2 input: " + ", ".join(missing))
    if B2_DIR.exists():
        shutil.rmtree(B2_DIR)
    B2_DIR.mkdir(parents=True)
    build_contract_snapshot()

    package = harness.read_json(PACKAGE_DIR / "manifest.json")
    prompt = harness.prompt_from_package(
        "Photorealistic architectural interior panorama using the validated lighting-independent albedo source; improve realism while preserving the exact design."
    )
    write_json(B2_DIR / "compiled_prompt.json", {"prompt": prompt, "source": "controlled_core_harness.prompt_from_package"})
    trace = [
        {"prompt_rule": "preserve exact equirectangular 2:1 panorama projection", "source_contract": "scene_contract", "evidence": "scene.json camera and frozen projection regression", "confidence": 1.0},
        {"prompt_rule": "preserve critical architectural geometry", "source_contract": "geometry_contract", "evidence": "structured B2 geometry preservation policy", "confidence": 1.0},
        {"prompt_rule": "preserve visible material colors and texture intent", "source_contract": "materials_contract", "evidence": "condition package material manifest and visibility data", "confidence": 0.8},
        {"prompt_rule": "do not invent fixture positions or photometric precision", "source_contract": "lighting_contract", "evidence": "lights.json unresolved fixture candidates", "confidence": 1.0},
        {"prompt_rule": "preserve known glazing/exterior geometry", "source_contract": "glazing_contract", "evidence": "scene contract glazing and exterior regions", "confidence": 0.8},
        {"prompt_rule": "preserve the 0/360 seam", "source_contract": "scene_contract", "evidence": "validated ERP ratio and seam checks", "confidence": 1.0},
    ]
    write_json(B2_DIR / "compiled_prompt_trace.json", {"rules": trace})

    class Args:
        model = flux.MODEL
        guidance = 3.5
        timeout = 300

    started = time.time()
    generation = harness.run_flux_job("B2_contract_rich", ALBEDO, prompt, BAKEOFF, Args(), package)
    b2_output = Path(generation.get("output", ""))
    qc = harness.qc_generation(ALBEDO, b2_output, package) if b2_output.is_file() else {"status": "NOT_RUN"}
    write_json(B2_DIR / "qc.json", qc)

    b1_report = harness.read_json(BAKEOFF / "report.json")
    comparison = {
        "schema": "rad-ai360-b1-b2-comparison",
        "baseline": "B1_controlled_harness is immutable and was not overwritten.",
        "controlled_difference": "B2 changes only the RGB source to fast albedo; provider, model, guidance, prompt compiler family, and output sizing are held constant. Depth/normal/IDs are not sent.",
        "B1": {"output": str(B1_DIR / "output.png"), "qc": b1_report.get("qc", {}).get("B1"), "runtime_seconds": b1_report.get("B1_controlled_harness", {}).get("runtime_seconds")},
        "B2": {"output": str(b2_output), "qc": qc, "runtime_seconds": generation.get("runtime_seconds"), "status": generation.get("status")},
        "criteria": ["realism", "architectural fidelity", "ceiling/scallop preservation", "openings", "furniture placement/count", "material intent", "glazing/exterior behavior", "lighting plausibility", "panorama seam", "runtime"],
        "verdict": "INCONCLUSIVE",
        "verdict_reason": "Automated QC can verify image validity, seam, projection, and local edge signals, but cannot prove semantic visual superiority. Human comparison is required for realism and design fidelity.",
        "created_at_unix": int(time.time()),
        "total_seconds": round(time.time() - started, 3),
    }
    write_json(B2_DIR / "comparison_to_B1.json", comparison)
    write_json(B2_DIR / "generation.json", generation)
    print(f"RAD_B2_OUTPUT={B2_DIR / 'output.png'}")
    print(f"RAD_B2_COMPARISON={B2_DIR / 'comparison_to_B1.json'}")
    return 0 if generation.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
