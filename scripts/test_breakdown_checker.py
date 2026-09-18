#!/usr/bin/env python3
"""Breakdown checker — validate the 6-face cubemap breakdown against ground truth.

Primary test: stitch phase0c normal faces to ERP and compare with the native
Blender equirectangular normal ERP from phase0d. World-space normals are
renderer-independent, so any mismatch indicates a breakdown/stitch convention bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHASE0C = ROOT / "output" / "phase0c"
PHASE0D = ROOT / "output" / "phase0d"
OUT_DIR = ROOT / "output" / "test_breakdown"

FACES = ("front", "right", "back", "left", "top", "bottom")

# Face basis vectors recorded in phase0c_report.json
FACE_VECTORS = {
    "front":  {"forward": np.array([1.0, 0.0, 0.0]),  "up": np.array([0.0, 0.0, 1.0]),  "right": np.array([0.0, -1.0, 0.0])},
    "right":  {"forward": np.array([0.0, -1.0, 0.0]), "up": np.array([0.0, 0.0, 1.0]),  "right": np.array([-1.0, 0.0, 0.0])},
    "back":   {"forward": np.array([-1.0, 0.0, 0.0]), "up": np.array([0.0, 0.0, 1.0]),  "right": np.array([0.0, 1.0, 0.0])},
    "left":   {"forward": np.array([0.0, 1.0, 0.0]),  "up": np.array([0.0, 0.0, 1.0]),  "right": np.array([1.0, 0.0, 0.0])},
    "top":    {"forward": np.array([0.0, 0.0, 1.0]),  "up": np.array([-1.0, 0.0, 0.0]), "right": np.array([0.0, -1.0, 0.0])},
    "bottom": {"forward": np.array([0.0, 0.0, -1.0]), "up": np.array([1.0, 0.0, 0.0]),  "right": np.array([0.0, -1.0, 0.0])},
}


def load_face_images(subdir: str) -> dict[str, np.ndarray]:
    """Load 6 PNG faces as float RGB [0,1]."""
    out = {}
    for face in FACES:
        path = PHASE0C / subdir / f"{face}_preview.png" if "normal" in subdir else PHASE0C / subdir / f"{face}.png"
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        out[face] = img
    return out


def load_erp(subdir: str, name: str) -> np.ndarray:
    path = PHASE0D / subdir / name
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img


def basis_validation() -> dict:
    """Orthonormality + handedness of each recorded face basis."""
    checks = []
    for face in FACES:
        f = FACE_VECTORS[face]["forward"]
        u = FACE_VECTORS[face]["up"]
        r = FACE_VECTORS[face]["right"]
        checks.append({"face": face, "test": "normalized",
                       "pass": bool(abs(np.linalg.norm(f) - 1) < 1e-6 and abs(np.linalg.norm(u) - 1) < 1e-6 and abs(np.linalg.norm(r) - 1) < 1e-6)})
        checks.append({"face": face, "test": "orthogonal",
                       "pass": bool(abs(np.dot(f, u)) < 1e-6 and abs(np.dot(f, r)) < 1e-6 and abs(np.dot(u, r)) < 1e-6)})
        cross = np.cross(f, u)
        checks.append({"face": face, "test": "right_handed_right_is_f_cross_up",
                       "pass": bool(np.allclose(cross, r, atol=1e-6))})
    status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    return {"status": status, "checks": checks}


def erp_ray_grid(w: int, h: int):
    """Return (lon_phi, lat_theta) grids and cos/sin for an ERP."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    phi = (xs / w - 0.5) * 2 * np.pi
    theta = (0.5 - ys / h) * np.pi
    return phi, theta, np.cos(theta), np.sin(theta)


def stitch_phase0k(cube: dict[str, np.ndarray], w: int, h: int) -> np.ndarray:
    size = next(iter(cube.values())).shape[0]
    phi, theta, cos_t, sin_t = erp_ray_grid(w, h)
    dx = cos_t * np.cos(phi)
    dy = cos_t * np.sin(phi)
    dz = sin_t
    d_world = np.stack([dx, -dy, dz], axis=-1)
    out = np.zeros((h, w, 3), dtype=np.float32)
    for face in FACES:
        f = FACE_VECTORS[face]["forward"]
        r = FACE_VECTORS[face]["right"]
        u = FACE_VECTORS[face]["up"]
        dot_f = d_world @ f
        mask = dot_f > 1e-4
        s = np.divide(d_world @ r, dot_f, out=np.zeros_like(dot_f), where=mask)
        t = np.divide(d_world @ u, dot_f, out=np.zeros_like(dot_f), where=mask)
        face_u = np.clip((s + 1) / 2, 0.0, 1.0)
        face_v = np.clip(1 - (t + 1) / 2, 0.0, 1.0)
        px = np.clip(face_u * (size - 1), 0, size - 1).astype(np.int32)
        py = np.clip(face_v * (size - 1), 0, size - 1).astype(np.int32)
        out[mask] = cube[face][py[mask], px[mask]]
    return out


def stitch_phase0l(cube: dict[str, np.ndarray], w: int, h: int) -> np.ndarray:
    size = next(iter(cube.values())).shape[0]
    phi, theta, cos_t, sin_t = erp_ray_grid(w, h)
    X = cos_t * np.cos(phi)
    Y = -cos_t * np.sin(phi)
    Z = sin_t
    aX, aY, aZ = np.abs(X), np.abs(Y), np.abs(Z)
    out = np.zeros((h, w, 3), dtype=np.float32)

    def remap(mask, ur, vr, face):
        u = np.clip((ur + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.int32)
        v = np.clip((vr + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.int32)
        out[mask] = cube[face][v[mask], u[mask]]

    with np.errstate(divide="ignore", invalid="ignore"):
        m = (X > 0) & (aX >= aY) & (aX >= aZ)
        remap(m, -np.divide(Y, X, out=np.zeros_like(X), where=m), -np.divide(Z, X, out=np.zeros_like(X), where=m), "front")
        m = (Y < 0) & (aY >= aX) & (aY >= aZ)
        remap(m, np.divide(X, Y, out=np.zeros_like(Y), where=m), np.divide(Z, Y, out=np.zeros_like(Y), where=m), "right")
        m = (X < 0) & (aX >= aY) & (aX >= aZ)
        remap(m, -np.divide(Y, X, out=np.zeros_like(X), where=m), -np.divide(Z, X, out=np.zeros_like(X), where=m), "back")
        m = (Y > 0) & (aY >= aX) & (aY >= aZ)
        remap(m, np.divide(X, Y, out=np.zeros_like(Y), where=m), -np.divide(Z, Y, out=np.zeros_like(Y), where=m), "left")
        m = (Z > 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, -np.divide(Y, Z, out=np.zeros_like(Z), where=m), np.divide(X, Z, out=np.zeros_like(Z), where=m), "top")
        m = (Z < 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, np.divide(Y, Z, out=np.zeros_like(Z), where=m), np.divide(X, Z, out=np.zeros_like(Z), where=m), "bottom")
    return out


def stitch_worker(cube: dict[str, np.ndarray], w: int, h: int) -> np.ndarray:
    size = next(iter(cube.values())).shape[0]
    phi, theta, cos_t, sin_t = erp_ray_grid(w, h)
    dx = cos_t * np.cos(phi)
    dy = cos_t * np.sin(phi)
    dz = sin_t
    aX, aY, aZ = np.abs(dx), np.abs(dy), np.abs(dz)
    out = np.zeros((h, w, 3), dtype=np.float32)

    def remap(mask, ur, vr, face):
        u = np.clip((ur + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.int32)
        v = np.clip((1 - vr) * 0.5 * (size - 1), 0, size - 1).astype(np.int32)
        out[mask] = cube[face][v[mask], u[mask]]

    z = np.zeros_like(dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (dx > 0) & (aX >= aY) & (aX >= aZ)
        remap(m, np.divide(dy, dx, out=z, where=m), np.divide(dz, dx, out=z, where=m), "front")
        m = (dy < 0) & (aY >= aX) & (aY >= aZ)
        remap(m, np.divide(dx, aY, out=z, where=m), np.divide(dz, aY, out=z, where=m), "right")
        m = (dx < 0) & (aX >= aY) & (aX >= aZ)
        remap(m, -np.divide(dy, aX, out=z, where=m), np.divide(dz, aX, out=z, where=m), "back")
        m = (dy > 0) & (aY >= aX) & (aY >= aZ)
        remap(m, -np.divide(dx, aY, out=z, where=m), np.divide(dz, aY, out=z, where=m), "left")
        m = (dz > 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, np.divide(dy, dz, out=z, where=m), -np.divide(dx, dz, out=z, where=m), "top")
        m = (dz < 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, np.divide(dy, dz, out=z, where=m), np.divide(dx, dz, out=z, where=m), "bottom")
    return out


def stitch_search(cube: dict[str, np.ndarray], w: int, h: int, a_axis, b_axis, c_axis, u_flip=1, v_flip=1) -> np.ndarray:
    """Generic ERP: d = cosT*(cosP*A + sinP*B) + sinT*C, then per-face sample top=+up/right=+right."""
    size = next(iter(cube.values())).shape[0]
    phi, theta, cos_t, sin_t = erp_ray_grid(w, h)
    d_world = (cos_t * np.cos(phi))[..., None] * a_axis + (cos_t * np.sin(phi))[..., None] * b_axis + sin_t[..., None] * c_axis
    out = np.zeros((h, w, 3), dtype=np.float32)
    for face in FACES:
        f = FACE_VECTORS[face]["forward"]
        r = FACE_VECTORS[face]["right"]
        u = FACE_VECTORS[face]["up"]
        dot_f = d_world @ f
        mask = dot_f > 1e-4
        s = u_flip * np.divide(d_world @ r, dot_f, out=np.zeros_like(dot_f), where=mask)
        t = v_flip * np.divide(d_world @ u, dot_f, out=np.zeros_like(dot_f), where=mask)
        face_u = np.clip((s + 1) / 2, 0.0, 1.0)
        face_v = np.clip(1 - (t + 1) / 2, 0.0, 1.0)
        px = np.clip(face_u * (size - 1), 0, size - 1).astype(np.int32)
        py = np.clip(face_v * (size - 1), 0, size - 1).astype(np.int32)
        out[mask] = cube[face][py[mask], px[mask]]
    return out


def decode_normal(img_rgb: np.ndarray) -> np.ndarray:
    n = img_rgb * 2.0 - 1.0
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = norms[..., 0] > 0.5
    return n, ok


def angular_error_map(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, pok = decode_normal(pred)
    t, tok = decode_normal(truth)
    ok = pok & tok
    dot = np.sum(p * t, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    err = np.degrees(np.arccos(dot))
    return err, ok


def per_face_metrics(err: np.ndarray, ok: np.ndarray, w: int, h: int) -> dict:
    phi, theta, cos_t, sin_t = erp_ray_grid(w, h)
    dx = cos_t * np.cos(phi)
    dy = cos_t * np.sin(phi)
    dz = sin_t
    aX, aY, aZ = np.abs(dx), np.abs(dy), np.abs(dz)
    labels = np.empty((h, w), dtype=np.int16)
    labels[:] = -1
    label_map = {face: i for i, face in enumerate(FACES)}
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (dx > 0) & (aX >= aY) & (aX >= aZ); labels[m] = label_map["front"]
        m = (dy < 0) & (aY >= aX) & (aY >= aZ); labels[m] = label_map["right"]
        m = (dx < 0) & (aX >= aY) & (aX >= aZ); labels[m] = label_map["back"]
        m = (dy > 0) & (aY >= aX) & (aY >= aZ); labels[m] = label_map["left"]
        m = (dz > 0) & (aZ >= aX) & (aZ >= aY); labels[m] = label_map["top"]
        m = (dz < 0) & (aZ >= aX) & (aZ >= aY); labels[m] = label_map["bottom"]
    metrics = {}
    mask_any = np.ones_like(ok)
    valid = ok
    total_sq = np.sum(valid)
    valid_err = err[valid]
    metrics["ALL"] = {
        "mean_deg": round(float(valid_err.mean()), 3),
        "p95_deg": round(float(np.percentile(valid_err, 95)), 3),
        "bad_pct": round(float(np.mean(valid_err > 5.0) * 100), 3),
        "valid_pixels": int(total_sq),
    }
    for face in FACES:
        lm = (labels == label_map[face])
        vm = lm & valid
        if vm.sum() == 0:
            metrics[face] = {"mean_deg": None, "p95_deg": None, "bad_pct": None, "valid_pixels": 0}
            continue
        fe = err[vm]
        metrics[face] = {
            "mean_deg": round(float(fe.mean()), 3),
            "p95_deg": round(float(np.percentile(fe, 95)), 3),
            "bad_pct": round(float(np.mean(fe > 5.0) * 100), 3),
            "valid_pixels": int(vm.sum()),
        }
    return metrics


def edge_continuity(cube: dict[str, np.ndarray]) -> dict:
    """Compare adjacent faces along shared edges. Min circular-shift search
    resolves unknown azimuth offset between a horizontal edge and the top/bottom
    face full-circle edge."""
    checks = []
    for a, b in (("front", "right"), ("right", "back"), ("back", "left"), ("left", "front")):
        left_edge = cube[a][:, -1:].astype(np.float32)
        right_edge = cube[b][:, :1].astype(np.float32)
        diff = float(np.abs(left_edge - right_edge).mean())
        checks.append({"pair": f"{a}|{b}", "mean_rgb_delta": round(diff, 4),
                       "pass": diff < 0.08})

    def min_roll_shift(row_a, row_b):
        rb = np.array(row_b)
        best = float(np.inf)
        for roll in range(rb.shape[1]):
            rolled = np.roll(rb, roll, axis=1)
            best = min(best, float(np.abs(row_a - rolled).mean()))
        return round(best, 4)

    for a in ("front", "right", "back", "left"):
        top_row = cube[a][:1].astype(np.float32)
        top_edge = cube["top"][-1:].astype(np.float32)
        checks.append({"pair": f"{a}|top", "min_mean_rgb_delta": min_roll_shift(top_row, top_edge)})
        bot_row = cube[a][-1:].astype(np.float32)
        bot_edge = cube["bottom"][:1].astype(np.float32)
        checks.append({"pair": f"{a}|bottom", "min_mean_rgb_delta": min_roll_shift(bot_row, bot_edge)})

    status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    return {"status": status, "checks": checks}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    report["basis"] = basis_validation()
    print("basis:", report["basis"]["status"])

    normals = load_face_images("normal")
    truth_n = load_erp("normal", "equirectangular_preview.png")
    h, w, _ = truth_n.shape

    compares = {}

    for name, fn in (("phase0k_build_erp", stitch_phase0k),
                     ("phase0l_stitch", stitch_phase0l),
                     ("ai360_worker_stitch", stitch_worker)):
        print(f"stitching [{name}] ...")
        pred = fn(normals, w, h)
        err, ok = angular_error_map(pred, truth_n)
        metrics = per_face_metrics(err, ok, w, h)
        compares[name] = metrics
        Image.fromarray(np.clip(err / 90.0 * 255, 0, 255).astype(np.uint8)).save(OUT_DIR / f"err_{name}.png")
        m = metrics["ALL"]
        print(f"  {name}: mean={m['mean_deg']}deg p95={m['p95_deg']}deg bad={m['bad_pct']}%")

    f = FACE_VECTORS["front"]["forward"]
    r = FACE_VECTORS["front"]["right"]
    u = FACE_VECTORS["front"]["up"]
    search_runs = []
    flips = [1, -1]
    best = None
    for a in (f, -f):
        for b in (r, -r):
            for c in (u, -u):
                for uf in flips:
                    for vf in flips:
                        pred = stitch_search(normals, w, h, a, b, c, uf, vf)
                        err, ok = angular_error_map(pred, truth_n)
                        ma = float(err[ok].mean())
                        search_runs.append({"a": a.tolist(), "b": b.tolist(), "c": c.tolist(),
                                            "u_flip": uf, "v_flip": vf, "mean_deg": round(ma, 3)})
                        if best is None or ma < best["mean_deg"]:
                            best = search_runs[-1]

    print("best convention:", best)
    report["candidates"] = {"phase0k_build_erp": compares["phase0k_build_erp"],
                            "phase0l_stitch": compares["phase0l_stitch"],
                            "ai360_worker_stitch": compares["ai360_worker_stitch"]}
    report["best_search"] = best
    report["edge_continuity"] = edge_continuity(load_face_images("rgb"))
    report["conclusion"] = ("Best convention search hit mean error %.3f deg. "
                            "phase0k=%.3f / phase0l=%.3f / worker=%.3f deg. q=max error." % (
        best["mean_deg"],
        compares["phase0k_build_erp"]["ALL"]["mean_deg"],
        compares["phase0l_stitch"]["ALL"]["mean_deg"],
        compares["ai360_worker_stitch"]["ALL"]["mean_deg"],
    ))
    report_path = OUT_DIR / "breakdown_test_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print("report:", report_path)


if __name__ == "__main__":
    main()