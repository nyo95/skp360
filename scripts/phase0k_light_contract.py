#!/usr/bin/env python3
"""Phase 0K Steps 2-8 — Deterministic Light Position Contract.

Reads scene_entities.json (from Ruby exporter Step 1) if present.
If absent, documents the gap and produces a zero-fixture contract.

Outputs:
  output/phase0k/lights/light_candidates.json
  output/phase0k/lights/light_overlay.png
  output/condition_package/masks/light_fixture_mask.png
  output/condition_package/semantics/lights.json   (updated)
"""
from __future__ import annotations

import json, math, hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT      = Path(__file__).resolve().parents[1]
PKG       = ROOT / "output" / "condition_package"
PHASE0K   = ROOT / "output" / "phase0k" / "lights"
PHASE0K.mkdir(parents=True, exist_ok=True)

SCENE_ENTITIES_JSON = ROOT / "output" / "obj" / "scene_entities.json"
ERP_SOURCE          = PKG / "source" / "rgb_erp.png"
SCENE_JSON          = ROOT / "output" / "obj" / "scene.json"

# ERP dimensions
ERP_W, ERP_H = 2048, 1024

# Camera from scene.json
CAMERA_EYE_M  = [9.464429, 10.229564, 2.185749]
CAMERA_FWD    = [1.0,  0.0,  0.0]
CAMERA_UP     = [0.0,  0.0,  1.0]
CAMERA_RIGHT  = [0.0, -1.0,  0.0]

# Light keyword detection (from brief)
LIGHT_KEYWORDS = {
    "light":        0.80,
    "lamp":         0.85,
    "downlight":    0.95,
    "spotlight":    0.90,
    "pendant":      0.90,
    "tracklight":   0.90,
    "track light":  0.90,
    "track_light":  0.90,
    "ceiling light":0.90,
    "wall light":   0.85,
    "sconce":       0.90,
    "led":          0.65,
    "fixture":      0.85,
    "luminaire":    0.90,
}

# Source-field confidence weights
FIELD_WEIGHT = {
    "definition_name": 1.0,
    "instance_name":   0.90,
    "tag":             0.75,
    "material":        0.30,
}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def mm_to_m(pt_mm):
    return [v / 1000.0 for v in pt_mm]


def project_to_erp(pos_m, cam_eye_m, cam_fwd, cam_up, cam_right, w, h):
    """Project a 3D world position to ERP pixel coordinates."""
    d = np.array(pos_m) - np.array(cam_eye_m)
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return None
    d = d / dist
    # Spherical relative to camera
    fwd  = np.array(cam_fwd)
    up   = np.array(cam_up)
    rght = np.array(cam_right)
    lon = math.atan2(np.dot(d, rght), np.dot(d, fwd))   # [-π, +π]
    lat = math.asin(float(np.clip(np.dot(d, up), -1.0, 1.0)))  # [-π/2, +π/2]
    u   = lon / (2 * math.pi) + 0.5   # [0, 1], 0.5 = forward
    v   = 0.5 - lat / math.pi         # [0, 1], 0 = top
    px  = int(u * w) % w
    py  = int(v * h)
    in_front = np.dot(d, fwd) > 0.0
    return {"u": round(u, 6), "v": round(v, 6), "px": px, "py": py,
            "visible_from_camera": in_front}


FACE_VECTORS = {
    "front":  {"forward": [1,0,0],  "up": [0,0,1],   "right": [0,-1,0]},
    "right":  {"forward": [0,-1,0], "up": [0,0,1],   "right": [-1,0,0]},
    "back":   {"forward": [-1,0,0], "up": [0,0,1],   "right": [0,1,0]},
    "left":   {"forward": [0,1,0],  "up": [0,0,1],   "right": [1,0,0]},
    "top":    {"forward": [0,0,1],  "up": [-1,0,0],  "right": [0,-1,0]},
    "bottom": {"forward": [0,0,-1], "up": [1,0,0],   "right": [0,-1,0]},
}


def project_to_cubemap(pos_m, cam_eye_m):
    """Project world position to the best cubemap face + UV."""
    d = np.array(pos_m) - np.array(cam_eye_m)
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return None
    d = d / dist
    best_face, best_dot = None, -2.0
    for face_name, vecs in FACE_VECTORS.items():
        dot = float(np.dot(d, vecs["forward"]))
        if dot > best_dot:
            best_dot, best_face = dot, face_name
    if best_face is None or best_dot <= 0:
        return None
    vecs = FACE_VECTORS[best_face]
    fwd  = np.array(vecs["forward"])
    rght = np.array(vecs["right"])
    up   = np.array(vecs["up"])
    f    = float(np.dot(d, fwd))
    s    = float(np.dot(d, rght)) / f
    t    = float(np.dot(d, up))   / f
    face_u = (s + 1) / 2
    face_v = 1 - (t + 1) / 2
    return {"face": best_face, "u": round(face_u, 6), "v": round(face_v, 6)}


# ── Candidate detection ────────────────────────────────────────────────────────

def detect_candidates_from_entities(entities):
    candidates = []
    for ent in entities:
        fields = {
            "definition_name": ent.get("definition_name") or "",
            "instance_name":   ent.get("instance_name")   or "",
            "tag":             ent.get("tag")              or "",
            "material":        ent.get("material")         or "",
        }
        matched_kws = []
        for kw, kw_conf in LIGHT_KEYWORDS.items():
            for field_name, field_val in fields.items():
                if kw in field_val.lower():
                    matched_kws.append({
                        "keyword": kw,
                        "field": field_name,
                        "value": field_val,
                        "kw_conf": kw_conf,
                        "field_weight": FIELD_WEIGHT[field_name],
                    })
        if not matched_kws:
            continue

        # Confidence = max(kw_conf * field_weight) per match
        conf = max(m["kw_conf"] * m["field_weight"] for m in matched_kws)
        evidence = [f"{m['field']}:{m['value']!r} contains keyword '{m['keyword']}'"
                    for m in matched_kws]

        if conf >= 0.80:
            status = "CONFIRMED_CANDIDATE"
        elif conf >= 0.50:
            status = "LOW_CONFIDENCE"
        else:
            status = "UNKNOWN"

        pos_mm = ent.get("world_position_mm") or ent.get("world_bounding_box_mm", {}).get("center")
        if not pos_mm:
            continue

        pos_m = mm_to_m(pos_mm)
        erp_proj = project_to_erp(pos_m, CAMERA_EYE_M, CAMERA_FWD,
                                   CAMERA_UP, CAMERA_RIGHT, ERP_W, ERP_H)
        cube_proj = project_to_cubemap(pos_m, CAMERA_EYE_M)

        candidates.append({
            "id": f"fixture_{len(candidates):04d}",
            "source_name": (ent.get("definition_name") or
                            ent.get("instance_name") or "UNNAMED"),
            "entity_type": ent.get("entity_type"),
            "world_position_mm": pos_mm,
            "blender_position_m": [round(v, 6) for v in pos_m],
            "confidence": round(conf, 3),
            "evidence": evidence,
            "status": status,
            "erp_projection": erp_proj,
            "cubemap_projection": cube_proj,
        })
    return candidates


def detect_from_light_audit_fallback():
    """Fall back to Phase 0G material-name audit if no entity metadata."""
    audit = ROOT / "output" / "phase0g" / "light_audit.json"
    if not audit.exists():
        return []
    data = json.loads(audit.read_text())
    result = []
    for c in data.get("candidates", []):
        result.append({
            "id": f"fixture_fallback_{len(result):04d}",
            "source_name": c.get("source_name", "UNKNOWN"),
            "entity_type": "material_only",
            "world_position_mm": None,
            "blender_position_m": None,
            "confidence": 0.25,
            "evidence": c.get("evidence", []),
            "status": "UNKNOWN",
            "erp_projection": None,
            "cubemap_projection": None,
        })
    return result


# ── Mask generation ────────────────────────────────────────────────────────────

def make_light_mask(confirmed, erp_w, erp_h, radius=18):
    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    for fix in confirmed:
        proj = fix.get("erp_projection")
        if not proj or not proj.get("visible_from_camera"):
            continue
        px, py = proj["px"], proj["py"]
        cv2.circle(mask, (px, py), radius, 255, -1)
    return mask


def make_overlay(confirmed, erp_source_path, radius=18):
    img = Image.open(erp_source_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for fix in confirmed:
        proj = fix.get("erp_projection")
        if not proj:
            continue
        px, py = proj["px"], proj["py"]
        colour = (255, 220, 0) if fix["status"] == "CONFIRMED_CANDIDATE" else (180, 180, 0)
        draw.ellipse([(px-radius, py-radius), (px+radius, py+radius)],
                     outline=colour, width=3)
        draw.text((px+radius+3, py-7), fix["source_name"][:20], fill=colour, font=font)
    return img


# ── Prompt compiler ────────────────────────────────────────────────────────────

def compile_prompt_constraint(confirmed):
    visible = [f for f in confirmed
               if f.get("erp_projection", {}) and
               f.get("erp_projection", {}).get("visible_from_camera")]
    n = len(visible)
    lines = ["LIGHTING CONSTRAINTS:"]
    if n == 0:
        lines += [
            "No confirmed light fixture positions are available from source metadata.",
            "Do not add any new ceiling lights, downlights, pendants, track lights,",
            "  wall lights, sconces, or luminaires.",
            "Do not create bright hotspots that imply nonexistent architectural luminaires.",
            "Illumination must derive from light sources visible in the input image.",
        ]
    else:
        lines.append(f"Preserve the {n} visible confirmed light fixture(s) at their exact source positions.")
        for fix in visible:
            proj = fix["erp_projection"]
            lines.append(f"  - {fix['source_name']} at ERP ({proj['px']}, {proj['py']})")
        lines += [
            "Do not add any additional ceiling lights, pendants, or luminaires.",
            "Do not relocate or remove existing fixtures.",
            "Do not create highlights implying nonexistent architectural light sources.",
        ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 0K — Light Position Contract ===")

    entity_metadata_available = SCENE_ENTITIES_JSON.exists()
    if entity_metadata_available:
        print(f"  Loading entity metadata: {SCENE_ENTITIES_JSON}")
        entities = json.loads(SCENE_ENTITIES_JSON.read_text())["entities"]
        candidates = detect_candidates_from_entities(entities)
        source = "scene_entities.json (SketchUp Ruby entity export)"
    else:
        print(f"  scene_entities.json NOT found — SketchUp entity export not yet run.")
        print(f"    In SketchUp: Extensions → RAD AI360 Visualizer → Export Entity Metadata...")
        print(f"    Save to: {ROOT / 'output' / 'obj'}")
        candidates = detect_from_light_audit_fallback()
        source = "light_audit.json fallback (material names only — no world positions)"

    confirmed   = [c for c in candidates if c["status"] == "CONFIRMED_CANDIDATE"]
    low_conf    = [c for c in candidates if c["status"] == "LOW_CONFIDENCE"]
    unknown     = [c for c in candidates if c["status"] == "UNKNOWN"]

    print(f"  Candidates: {len(candidates)} total — "
          f"{len(confirmed)} CONFIRMED, {len(low_conf)} LOW_CONFIDENCE, "
          f"{len(unknown)} UNKNOWN")

    # light_candidates.json
    light_cand = {
        "schema": "rad-ai360-phase0k-light-candidates",
        "version": 1,
        "date": "2026-09-17",
        "source": source,
        "entity_metadata_available": entity_metadata_available,
        "candidates_total": len(candidates),
        "confirmed_count": len(confirmed),
        "low_confidence_count": len(low_conf),
        "unknown_count": len(unknown),
        "exporter_gap": not entity_metadata_available,
        "exporter_gap_note": (
            "scene_entities.json was not found. Run 'Export Entity Metadata...' "
            "from SketchUp to populate fixture world positions. "
            "Until then, confirmed_count=0 and light positions are UNKNOWN."
        ) if not entity_metadata_available else None,
        "confirmed": confirmed,
        "low_confidence": low_conf,
        "unknown": unknown,
    }
    cand_path = PHASE0K / "light_candidates.json"
    cand_path.write_text(json.dumps(light_cand, indent=2))
    print(f"  Saved: {cand_path}")

    # light_fixture_mask.png (all zeros if no confirmed)
    mask = make_light_mask(confirmed, ERP_W, ERP_H)
    mask_path = PKG / "masks" / "light_fixture_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"  Saved mask: {mask_path} (nonzero={int(mask.any())})")
    # Also copy to phase0k/lights
    import shutil; shutil.copy2(mask_path, PHASE0K / "light_fixture_mask.png")

    # light_overlay.png
    overlay_path = PHASE0K / "light_overlay.png"
    if ERP_SOURCE.exists():
        overlay = make_overlay(confirmed + low_conf, ERP_SOURCE)
        overlay.save(overlay_path)
        print(f"  Saved overlay: {overlay_path}")
    else:
        print(f"  WARNING: ERP source not found, skipping overlay")

    # Prompt constraint
    prompt_constraint = compile_prompt_constraint(confirmed)
    print(f"\n  Prompt constraint compiled:\n{prompt_constraint}")

    # Update condition_package/semantics/lights.json
    lights_json = {
        "schema": "rad-ai360-lights",
        "version": 2,
        "date": "2026-09-17",
        "source_boundary": "SOURCE_TRUTH_ONLY; no light emitters invented.",
        "entity_metadata_available": entity_metadata_available,
        "scene": "AI360_1",
        "projection": {"type": "equirectangular", "width": ERP_W, "height": ERP_H},
        "fixtures": [
            {
                "id": f["id"],
                "source_name": f["source_name"],
                "classification": "light_fixture",
                "classification_confidence": f["confidence"],
                "world_position_mm": f["world_position_mm"],
                "blender_position_m": f["blender_position_m"],
                "visible_from_camera": (f.get("erp_projection") or {}).get("visible_from_camera"),
                "erp_uv": ([f["erp_projection"]["u"], f["erp_projection"]["v"]]
                           if f.get("erp_projection") else None),
                "erp_pixel": ([f["erp_projection"]["px"], f["erp_projection"]["py"]]
                              if f.get("erp_projection") else None),
                "cubemap_projection": f.get("cubemap_projection"),
                "evidence": f["evidence"],
                "status": f["status"],
                "properties": {"cct": None, "wattage": None, "beam_angle": None},
            }
            for f in confirmed
        ],
        "unresolved_candidates": [
            {
                "id": c["id"],
                "source_name": c["source_name"],
                "world_position_mm": c["world_position_mm"],
                "confidence": c["confidence"],
                "evidence": c["evidence"],
                "status": c["status"],
                "reason_not_promoted": (
                    "Material-name evidence only; world position unavailable until "
                    "SketchUp entity export is run." if not entity_metadata_available
                    else "Confidence below CONFIRMED_CANDIDATE threshold."
                ),
            }
            for c in low_conf + unknown
        ],
        "exporter_gap": {
            "reliable_light_positions_available": len(confirmed) > 0,
            "entity_metadata_run": entity_metadata_available,
            "needed_upstream_action": (
                "Run Extensions → RAD AI360 Visualizer → Export Entity Metadata "
                f"and save to {ROOT / 'output' / 'obj'}"
            ) if not entity_metadata_available else None,
        },
        "prompt_constraint": prompt_constraint,
    }
    lights_path = PKG / "semantics" / "lights.json"
    lights_path.write_text(json.dumps(lights_json, indent=2))
    # Also copy to phase0k/lights
    shutil.copy2(lights_path, PHASE0K / "lights.json")
    print(f"\n  Saved lights.json: {lights_path}")

    print(f"\n  CONFIRMED fixtures: {len(confirmed)}")
    print(f"  Entity metadata available: {entity_metadata_available}")
    if not entity_metadata_available:
        print(f"\n  *** ACTION REQUIRED ***")
        print(f"  Open SketchUp with the model.")
        print(f"  Run: Extensions → RAD AI360 Visualizer → Export Entity Metadata...")
        print(f"  Save to: {ROOT / 'output' / 'obj'}")
        print(f"  Then re-run this script to populate fixture positions.")

    return prompt_constraint


if __name__ == "__main__":
    main()
