from __future__ import annotations

import math
from typing import Mapping

import numpy as np


SKETCHUP_WORLD = {
    "x": "+X",
    "y": "+Y",
    "z": "+Z",
    "camera_forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "camera_right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "camera_up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}

FACE_BASIS = {
    "front": {
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    "right": {
        "forward": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "right": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    "back": {
        "forward": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    "left": {
        "forward": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    "top": {
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "up": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    },
    "bottom": {
        "forward": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "up": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    },
}

ERP_CONTRACT = {
    "longitude_zero": "front (+X)",
    "longitude_plus_90": "left (+Y)",
    "longitude_minus_90": "right (-Y)",
    "longitude_180": "back (-X)",
    "zenith": "+Z",
    "nadir": "-Z",
    "erp_x_direction": "from front toward left (+90°) as U advances; seam at 180°/−180°",
    "erp_y_direction": "from zenith to nadir with 0° at the top pole",
    "cubemap_face_direction": "world direction of face forward vector",
    "cubemap_face_up": "world up vector for that face",
}


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Zero-length vector is not valid for projection math.")
    return v / norm


def world_direction_to_erp(direction: np.ndarray | tuple[float, float, float]) -> tuple[float, float]:
    """Return (longitude, latitude) in radians for a world direction.

    Canonical contract (matches the actual SketchUp/Phase 0C camera basis):
      - front = +X
      - right = -Y
      - back = -X
      - left = +Y
      - top = +Z
      - bottom = -Z
      - longitude 0 = front
    - longitude +pi/2 = left
    - longitude -pi/2 = right
      - longitude pi = back
      - latitude +pi/2 = top zenith
      - latitude -pi/2 = bottom nadir
    """
    v = _normalize(np.asarray(direction, dtype=np.float64))
    x, y, z = v
    lon = math.atan2(-y, x)
    lat = math.asin(np.clip(z, -1.0, 1.0))
    return lon, lat


def erp_to_world_direction(longitude: float, latitude: float) -> np.ndarray:
    """Invert world_direction_to_erp for the canonical contract."""
    x = math.cos(latitude) * math.cos(longitude)
    y = -math.cos(latitude) * math.sin(longitude)
    z = math.sin(latitude)
    return np.array([x, y, z], dtype=np.float64)


def cubemap_face_uv(face: str, direction: np.ndarray | tuple[float, float, float]) -> tuple[float, float]:
    """Map a world direction to normalized face UV coordinates.

    Each cubemap face is defined in a right-handed basis where the face forward axis is the
    camera-facing direction, its right axis is computed from the face basis, and the face's up
    axis is orthogonal to that. The resulting face UV coordinates use the conventional image
    space where U = 0..1 corresponds to left-to-right across the face and V = 0..1 is top-to-bottom.
    """
    face_basis = FACE_BASIS[face]
    fwd = _normalize(np.asarray(face_basis["forward"]))
    right = _normalize(np.asarray(face_basis["right"]))
    up = _normalize(np.asarray(face_basis["up"]))

    v = _normalize(np.asarray(direction, dtype=np.float64))
    local_x = float(np.dot(v, right))
    local_y = float(np.dot(v, up))
    local_z = float(np.dot(v, fwd))

    # Each face is a square tangent projection; UVs are centered on the forward vector.
    u = 0.5 + (local_x / max(abs(local_z), 1e-8)) * 0.5
    v_uv = 0.5 - (local_y / max(abs(local_z), 1e-8)) * 0.5
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v_uv, 0.0, 1.0))


def cubemap_to_erp(cube: Mapping[str, np.ndarray], width: int) -> np.ndarray:
    """Stitch a six-face cubemap into an equirectangular panorama using the canonical contract."""
    if width <= 0 or width % 2 != 0:
        raise ValueError("ERP width must be even and positive.")
    height = width // 2
    first_face = np.asarray(next(iter(cube.values())))
    size = first_face.shape[0]
    if first_face.shape[0] != first_face.shape[1]:
        raise ValueError("Cubemap faces must be square.")
    u_px, v_px = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    lon = (u_px / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - v_px / height) * np.pi
    dx = np.cos(lat) * np.cos(lon)
    dy = -np.cos(lat) * np.sin(lon)
    dz = np.sin(lat)
    direction = np.stack((dx, dy, dz), axis=-1)
    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    out_shape = (height, width) if first_face.ndim == 2 else (height, width, first_face.shape[2])
    out = np.zeros(out_shape, dtype=first_face.dtype)

    def remap(mask, face):
        import cv2

        basis = FACE_BASIS[face]
        forward = np.asarray(basis["forward"], dtype=np.float32)
        right = np.asarray(basis["right"], dtype=np.float32)
        up = np.asarray(basis["up"], dtype=np.float32)
        local_x = np.sum(direction * right, axis=-1)
        local_y = np.sum(direction * up, axis=-1)
        local_z = np.sum(direction * forward, axis=-1)
        denominator = np.maximum(np.abs(local_z), 1e-8)
        u = np.clip((0.5 + (local_x / denominator) * 0.5) * (size - 1), 0, size - 1).astype(np.float32)
        v = np.clip((0.5 - (local_y / denominator) * 0.5) * (size - 1), 0, size - 1).astype(np.float32)
        sample = cv2.remap(np.asarray(cube[face]), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[mask] = sample[mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        m = (dx > 0) & (ax >= ay) & (ax >= az)
        remap(m, "front")
        m = (dy < 0) & (ay >= ax) & (ay >= az)
        remap(m, "right")
        m = (dx < 0) & (ax >= ay) & (ax >= az)
        remap(m, "back")
        m = (dy > 0) & (ay >= ax) & (ay >= az)
        remap(m, "left")
        m = (dz > 0) & (az >= ax) & (az >= ay)
        remap(m, "top")
        m = (dz < 0) & (az >= ax) & (az >= ay)
        remap(m, "bottom")

    return out


# Import lazily to avoid import-time heavy dependency issues.
def _ensure_cv2():
    import cv2

    return cv2
