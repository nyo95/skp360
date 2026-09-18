from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .projection import LABEL_PASSES, cubemap_to_erp


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


def pack_rgb(image: np.ndarray) -> np.ndarray:
	"""Pack an 8-bit 3-channel image into one uint32 label per pixel."""
	rgb = np.asarray(image, dtype=np.uint32)
	if rgb.ndim != 3 or rgb.shape[2] < 3:
		raise ValueError("Label packing expects a 3-channel image.")
	return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def snap_to_palette(image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, dict]:
	"""Force every pixel of a label pass onto its nearest legal palette entry.

	Blender renders ID passes through a reconstruction filter, so even a single face
	contains anti-aliased colours that are not real IDs. Snapping restores a discrete
	label image whose unique colour count cannot exceed the palette size.
	"""
	source = np.asarray(image)
	if source.ndim != 3 or source.shape[2] < 3:
		raise ValueError("Label snapping expects a 3-channel image.")
	legal = np.asarray(palette, dtype=np.int16).reshape(-1, 3)
	if legal.size == 0:
		raise ValueError("Label palette is empty.")
	flat = source[..., :3].reshape(-1, 3).astype(np.int16)
	before = int(np.unique(pack_rgb(source[..., :3])).size)
	# Chunked nearest-palette search keeps peak memory bounded for 2K ERP inputs.
	nearest = np.empty(flat.shape[0], dtype=np.int32)
	chunk = 1 << 18
	for start in range(0, flat.shape[0], chunk):
		block = flat[start : start + chunk]
		distance = np.abs(block[:, None, :] - legal[None, :, :]).sum(axis=2)
		nearest[start : start + chunk] = np.argmin(distance, axis=1)
	snapped = legal[nearest].astype(np.uint8).reshape(source.shape[0], source.shape[1], 3)
	after = int(np.unique(pack_rgb(snapped)).size)
	stats = {
		"palette_size": int(legal.shape[0]),
		"unique_colors_before": before,
		"unique_colors_after": after,
		"changed_pixel_percent": float(np.mean(np.any(snapped != source[..., :3], axis=2)) * 100.0),
		"status": "PASS" if after <= legal.shape[0] else "FAIL",
	}
	if stats["status"] != "PASS":
		raise ValueError(f"Label snapping did not produce a discrete label image: {stats}")
	return snapped, stats


def stitch_passes(cube: Mapping[str, np.ndarray], width: int, pass_name: str | None = None) -> np.ndarray:
	missing = [face for face in FACE_ORDER if face not in cube]
	if missing:
		raise ValueError(f"Missing cubemap faces: {', '.join(missing)}")
	interpolation = "nearest" if pass_name in LABEL_PASSES else "linear"
	return cubemap_to_erp({face: cube[face] for face in FACE_ORDER}, width, interpolation=interpolation)


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