from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .projection import cubemap_to_erp


FACE_ORDER = ("front", "right", "back", "left", "top", "bottom")


def radial_depth_condition(
	depth_m: np.ndarray,
	valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
	"""Convert visible radial metres to a robust white-near/black-far PNG."""
	depth = np.asarray(depth_m, dtype=np.float32)
	valid = np.isfinite(depth) & (depth > 0.0)
	if valid_mask is not None:
		valid &= np.asarray(valid_mask, dtype=bool)
	values = depth[valid]
	if values.size < 32:
		raise ValueError("Radial depth has too few visible pixels for conditioning.")

	p2, p50, p98 = np.percentile(values, (2.0, 50.0, 98.0)).astype(float)
	if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
		raise ValueError("Radial depth has no usable dynamic range.")

	clipped_near = float(np.mean(values <= p2) * 100.0)
	clipped_far = float(np.mean(values >= p98) * 100.0)
	normalized = np.zeros(depth.shape, dtype=np.float32)
	normalized[valid] = np.clip((p98 - depth[valid]) / (p98 - p2), 0.0, 1.0)
	condition = np.round(normalized * 255.0).astype(np.uint8)
	unique_values = int(np.unique(condition[valid]).size)
	if unique_values < 16:
		raise ValueError("Conditioning depth is nearly flat.")

	stats = {
		"min_m": float(np.min(values)),
		"max_m": float(np.max(values)),
		"p2_m": float(p2),
		"p50_m": float(p50),
		"p98_m": float(p98),
		"mean_m": float(np.mean(values)),
		"std_m": float(np.std(values)),
		"unique_8bit_values": unique_values,
		"clipped_near_percent": clipped_near,
		"clipped_far_percent": clipped_far,
		"valid_pixel_count": int(values.size),
		"status": "PASS",
	}
	return condition, stats


def stitch_passes(cube: Mapping[str, np.ndarray], width: int) -> np.ndarray:
	missing = [face for face in FACE_ORDER if face not in cube]
	if missing:
		raise ValueError(f"Missing cubemap faces: {', '.join(missing)}")
	return cubemap_to_erp({face: cube[face] for face in FACE_ORDER}, width)


def sha256_file(path: str | Path) -> str:
	digest = hashlib.sha256()
	with open(path, "rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def build_manifest(root: str | Path, files: Mapping[str, str | Path], metadata: Mapping) -> dict:
	root_path = Path(root)
	assets = {}
	for name, file_path in sorted(files.items()):
		path = Path(file_path)
		assets[name] = {
			"path": path.relative_to(root_path).as_posix(),
			"bytes": path.stat().st_size,
			"sha256": sha256_file(path),
		}
	return {"schema": "rad-ai360-pass-manifest", "metadata": dict(metadata), "assets": assets}


def write_manifest(path: str | Path, manifest: Mapping) -> None:
	Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def edge_overlap(albedo: np.ndarray, depth_condition: np.ndarray) -> dict:
	albedo_gray = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY) if albedo.ndim == 3 else albedo
	depth_edges = cv2.Canny(depth_condition, 40, 120)
	albedo_edges = cv2.Canny(albedo_gray, 40, 120)
	overlap = (albedo_edges > 0) & (depth_edges > 0)
	union = (albedo_edges > 0) | (depth_edges > 0)
	return {
		"albedo_edge_pixels": int(np.count_nonzero(albedo_edges)),
		"depth_edge_pixels": int(np.count_nonzero(depth_edges)),
		"overlap_pixels": int(np.count_nonzero(overlap)),
		"iou": float(np.count_nonzero(overlap) / max(np.count_nonzero(union), 1)),
		"status": "PASS" if np.count_nonzero(albedo_edges) and np.count_nonzero(depth_edges) else "FAIL",
	}