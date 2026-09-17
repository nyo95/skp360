#!/usr/bin/env python3
"""Phase 0K Step 9 — K-A vs K-B Narrow Test.

K-A: Phase 0J Test C (ControlNet → FLUX, Phase 0J prompt) — reused, no new generation.
K-B: Same ControlNet Stage 1 output → FLUX, Phase 0J prompt + LIGHTING CONSTRAINTS appended.

Outputs:
  output/phase0k/K-A/output.png        (copied from Phase 0J Test C)
  output/phase0k/K-B/output.png        (new FLUX call with lighting-constrained prompt)
  output/phase0k/comparison/side_by_side.png
  output/phase0k/phase0k_ab_report.json
"""
from __future__ import annotations

import json, uuid, time, shutil, base64, http.client, urllib.parse
from pathlib import Path
from PIL import Image

ROOT    = Path(__file__).resolve().parents[1]
PHASE0K = ROOT / "output" / "phase0k"
PHASE0K.mkdir(parents=True, exist_ok=True)

# Source paths
KA_SOURCE      = ROOT / "output" / "phase0j" / "C_controlnet_then_flux" / "output.png"
STAGE1_SOURCE  = ROOT / "output" / "phase0j" / "B_controlnet_depth"     / "output.png"
LIGHTS_JSON    = PHASE0K / "lights" / "lights.json"

# Cloudflare credentials
CONFIG_JSON    = (
    Path.home() / "AppData" / "Roaming" / "SketchUp" / "SketchUp 2021" /
    "SketchUp" / "Plugins" / "rad_ai360_visualizer" / "config.json"
)

MODEL    = "@cf/black-forest-labs/flux-2-klein-4b"
GUIDANCE = 3.5
REF_LIMIT = 511

FLUX_PROMPT_BASE = (
    "Enhance this existing architectural interior rendering into a "
    "high-end photorealistic interior photograph. "
    "Preserve the exact room layout, all furniture positions, all wall openings, "
    "cabinetry, and ceiling height precisely as shown. "
    "Upgrade materials: warm walnut wood grain on TV unit and shelving, "
    "polished concrete floor with subtle reflections, "
    "linen-textured sofa, painted plaster walls. "
    "Lighting: warm directional daylight from the left with soft indirect fill; "
    "physically plausible illumination and soft shadows. "
    "Camera: fixed — do not alter the perspective, framing, or field of view. "
    "Style: contemporary residential interior photography, wide-angle architectural lens."
)


def load_credentials():
    data = json.loads(CONFIG_JSON.read_text())
    cf = data["cloudflare"]
    return cf["account_id"], cf["api_token"]


def resize_for_flux(img_path: Path, limit: int) -> bytes:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > limit:
        scale = limit / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def call_flux(account_id: str, api_token: str, ref_bytes: bytes,
              out_path: Path, width: int, height: int, prompt: str) -> float:
    boundary = "----phase0k-" + uuid.uuid4().hex
    crlf = b"\r\n"

    def part(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = b""
    body += part("prompt",   prompt)
    body += part("width",    str(width))
    body += part("height",   str(height))
    body += part("guidance", str(GUIDANCE))
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="input_image_0"; filename="ref.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + ref_bytes + crlf
    body += f"--{boundary}--\r\n".encode()

    url_path = f"/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    conn = http.client.HTTPSConnection("api.cloudflare.com", timeout=300)
    t0 = time.time()
    conn.request("POST", url_path, body=body, headers={
        "Authorization": f"Bearer {api_token}",
        "Content-Type":  f"multipart/form-data; boundary={boundary}",
    })
    resp = conn.getresponse()
    elapsed = time.time() - t0
    raw = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"FLUX HTTP {resp.status}: {raw[:200]}")
    payload = json.loads(raw)
    img_b64 = payload.get("result", {}).get("image") or payload.get("images", [None])[0]
    if not img_b64:
        raise RuntimeError(f"No image in response: {list(payload.keys())}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(img_b64))
    return elapsed


def load_prompt_constraint() -> str | None:
    if LIGHTS_JSON.exists():
        data = json.loads(LIGHTS_JSON.read_text())
        return data.get("prompt_constraint")
    return None


def make_comparison(ka_path: Path, kb_path: Path, out_path: Path):
    ka = Image.open(ka_path).convert("RGB")
    kb = Image.open(kb_path).convert("RGB")
    w, h = ka.size
    kb = kb.resize((w, h), Image.LANCZOS)
    pad = 16
    label_h = 36
    total_w = w * 2 + pad * 3
    total_h = h + label_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    canvas.paste(ka, (pad, label_h + pad))
    canvas.paste(kb, (w + pad * 2, label_h + pad))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((pad, 8), "K-A: ControlNet→FLUX (Phase 0J prompt)", fill=(220, 220, 220), font=font)
    draw.text((w + pad * 2, 8), "K-B: ControlNet→FLUX + Lighting Constraints", fill=(220, 220, 220), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    print("=== Phase 0K Step 9 — K-A vs K-B Narrow Test ===")

    # K-A — reuse Phase 0J Test C (no new generation)
    ka_dir = PHASE0K / "K-A"
    ka_dir.mkdir(parents=True, exist_ok=True)
    ka_out = ka_dir / "output.png"
    if not KA_SOURCE.exists():
        raise FileNotFoundError(f"K-A source not found: {KA_SOURCE}")
    shutil.copy2(KA_SOURCE, ka_out)
    print(f"  K-A: copied from Phase 0J Test C → {ka_out}")

    # Load prompt constraint from light contract
    constraint = load_prompt_constraint()
    if constraint:
        print(f"  Lighting constraint loaded ({len(constraint)} chars).")
    else:
        print("  WARNING: lights.json not found — run phase0k_light_contract.py first.")
        print("    Generating K-B with hardcoded no-fixture constraint.")
        constraint = (
            "LIGHTING CONSTRAINTS:\n"
            "No confirmed light fixture positions are available from source metadata.\n"
            "Do not add any new ceiling lights, downlights, pendants, track lights,\n"
            "  wall lights, sconces, or luminaires.\n"
            "Do not create bright hotspots that imply nonexistent architectural luminaires.\n"
            "Illumination must derive from light sources visible in the input image."
        )

    kb_prompt = FLUX_PROMPT_BASE + "\n\n" + constraint
    print(f"  K-B prompt length: {len(kb_prompt)} chars")

    # K-B — new FLUX call with lighting-constrained prompt
    account_id, api_token = load_credentials()
    kb_dir = PHASE0K / "K-B"
    kb_dir.mkdir(parents=True, exist_ok=True)
    kb_out = kb_dir / "output.png"

    if not STAGE1_SOURCE.exists():
        raise FileNotFoundError(f"Stage 1 source not found: {STAGE1_SOURCE}")

    src_img = Image.open(STAGE1_SOURCE).convert("RGB")
    w, h = src_img.size
    ref_bytes = resize_for_flux(STAGE1_SOURCE, REF_LIMIT)

    print(f"  Running FLUX for K-B (ref={STAGE1_SOURCE.name}, {w}x{h})...")
    elapsed = call_flux(account_id, api_token, ref_bytes, kb_out, w, h, kb_prompt)
    print(f"  K-B done: {kb_out} ({elapsed:.1f}s)")

    # Comparison
    comp_path = PHASE0K / "comparison" / "side_by_side.png"
    make_comparison(ka_out, kb_out, comp_path)
    print(f"  Comparison saved: {comp_path}")

    # Report
    report = {
        "phase": "0K",
        "step": 9,
        "title": "K-A vs K-B Narrow Light Contract Test",
        "date": "2026-09-17",
        "ka": {
            "id": "K-A",
            "source": "Phase 0J Test C (ControlNet→FLUX, Phase 0J base prompt)",
            "output": str(ka_out.relative_to(ROOT)),
            "new_generation": False,
        },
        "kb": {
            "id": "K-B",
            "source": "Phase 0J Stage 1 (ControlNet output) → FLUX with lighting constraints",
            "stage1_source": str(STAGE1_SOURCE.relative_to(ROOT)),
            "output": str(kb_out.relative_to(ROOT)),
            "runtime_s": round(elapsed, 1),
            "prompt_length": len(kb_prompt),
            "lighting_constraint_applied": True,
            "confirmed_fixtures": 0,
            "constraint_type": "zero-fixture negative constraint",
            "new_generation": True,
        },
        "comparison": str(comp_path.relative_to(ROOT)),
        "evaluation_criteria": {
            "geometry_preservation": "Does K-B maintain same geometry contract as K-A?",
            "hallucinated_luminaires": "Does K-B suppress fixture invention vs K-A?",
            "lighting_realism": "Is K-B lighting still physically plausible?",
            "overall_quality": "Is K-B visually comparable to K-A?",
        },
        "constraint_source": "lights.json (phase0k_light_contract.py output)",
    }
    report_path = PHASE0K / "phase0k_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {report_path}")
    print("\n  *** STOP — Review K-A vs K-B comparison before proceeding to Step 10 ***")


if __name__ == "__main__":
    main()
