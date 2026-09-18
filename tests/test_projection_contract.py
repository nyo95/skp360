import numpy as np
import pytest

from backend.projection import (
    ERP_CONTRACT,
    FACE_BASIS,
    LABEL_PASSES,
    cubemap_face_uv,
    cubemap_to_erp,
    erp_to_world_direction,
    world_direction_to_erp,
)
from backend.pass_pipeline import pack_rgb, radial_depth_condition, snap_to_palette, stitch_passes


FACE_COLORS = {
    "front": (0, 0, 255),
    "right": (0, 255, 0),
    "back": (255, 0, 0),
    "left": (255, 255, 0),
    "top": (255, 0, 255),
    "bottom": (0, 255, 255),
}


def _flat_cube(size: int = 64) -> dict:
    return {face: np.full((size, size, 3), color, dtype=np.uint8) for face, color in FACE_COLORS.items()}



def test_face_basis_matches_scene_json():
    assert np.allclose(FACE_BASIS["front"]["forward"], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(FACE_BASIS["front"]["right"], np.array([0.0, -1.0, 0.0]))
    assert np.allclose(FACE_BASIS["front"]["up"], np.array([0.0, 0.0, 1.0]))

    assert np.allclose(FACE_BASIS["right"]["forward"], np.array([0.0, -1.0, 0.0]))
    assert np.allclose(FACE_BASIS["right"]["right"], np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(FACE_BASIS["left"]["forward"], np.array([0.0, 1.0, 0.0]))
    assert np.allclose(FACE_BASIS["left"]["right"], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(FACE_BASIS["back"]["forward"], np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(FACE_BASIS["top"]["forward"], np.array([0.0, 0.0, 1.0]))
    assert np.allclose(FACE_BASIS["bottom"]["forward"], np.array([0.0, 0.0, -1.0]))


def test_face_uv_axes_define_top_and_bottom_rotation():
    for face in ("front", "right", "back", "left", "top", "bottom"):
        forward = FACE_BASIS[face]["forward"]
        right = FACE_BASIS[face]["right"]
        up = FACE_BASIS[face]["up"]
        assert np.allclose(cubemap_face_uv(face, forward), (0.5, 0.5))
        right_uv = cubemap_face_uv(face, forward + right * 0.25)
        up_uv = cubemap_face_uv(face, forward + up * 0.25)
        assert right_uv[0] > 0.5 and np.isclose(right_uv[1], 0.5)
        assert np.isclose(up_uv[0], 0.5) and up_uv[1] < 0.5


def test_erps_are_consistent_with_scene_camera_basis():
    for direction in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ):
        lon, lat = world_direction_to_erp(direction)
        got = erp_to_world_direction(lon, lat)
        assert np.allclose(got, direction, atol=1e-7)


def test_longitude_contract():
    lon_front, _ = world_direction_to_erp(np.array([1.0, 0.0, 0.0]))
    lon_right, _ = world_direction_to_erp(np.array([0.0, -1.0, 0.0]))
    lon_left, _ = world_direction_to_erp(np.array([0.0, 1.0, 0.0]))
    lon_back, _ = world_direction_to_erp(np.array([-1.0, 0.0, 0.0]))

    assert np.isclose(lon_front, 0.0)
    assert np.isclose(lon_right, np.pi / 2.0)
    assert np.isclose(lon_left, -np.pi / 2.0)
    assert np.isclose(abs(abs(lon_back) - np.pi), 0.0)


def test_cubemap_to_erp_horizontal_contract_matches_face_identity():
    cube = _flat_cube()

    erp = cubemap_to_erp(cube, 256)

    for expected_face, x_frac, y_frac in (
        ("front", 0.5, 0.5),
        ("left", 0.25, 0.5),
        ("right", 0.75, 0.5),
        ("back", 0.0, 0.5),
        ("top", 0.5, 0.0),
        ("bottom", 0.5, 1.0),
    ):
        x = int(round(x_frac * (erp.shape[1] - 1)))
        y = int(round(y_frac * (erp.shape[0] - 1)))
        sampled = tuple(int(v) for v in erp[y, x])
        assert sampled == FACE_COLORS[expected_face], (
            f"Expected {expected_face} at ({x_frac}, {y_frac}) but got {sampled}"
        )


def test_erp_contract_labels_match_the_projection_math():
    """The documented contract must be derived from the same math the code uses.

    The Phase 0L yaw regression was caused by human-written labels drifting away from
    `world_direction_to_erp`, so the labels are asserted here rather than trusted.
    """
    axis_for_face = {
        "front": np.array([1.0, 0.0, 0.0]),
        "right": np.array([0.0, -1.0, 0.0]),
        "left": np.array([0.0, 1.0, 0.0]),
        "back": np.array([-1.0, 0.0, 0.0]),
    }
    longitude = {face: world_direction_to_erp(axis)[0] for face, axis in axis_for_face.items()}

    assert np.isclose(longitude["front"], 0.0)
    assert np.isclose(longitude["right"], np.pi / 2.0)
    assert np.isclose(longitude["left"], -np.pi / 2.0)
    assert np.isclose(abs(longitude["back"]), np.pi)

    assert ERP_CONTRACT["longitude_zero"].startswith("front")
    assert ERP_CONTRACT["longitude_plus_90"].startswith("right")
    assert ERP_CONTRACT["longitude_minus_90"].startswith("left")
    assert ERP_CONTRACT["longitude_180"].startswith("back")
    assert ERP_CONTRACT["zenith"] == "+Z"
    assert ERP_CONTRACT["nadir"] == "-Z"


def test_erp_horizontal_order_is_back_left_front_right():
    """Guards the U-advance order that ERP_CONTRACT["erp_x_direction"] claims."""
    erp = cubemap_to_erp(_flat_cube(), 256)
    inverse = {color: face for face, color in FACE_COLORS.items()}
    row = erp.shape[0] // 2

    observed = [
        inverse[tuple(int(v) for v in erp[row, int(round(x_frac * (erp.shape[1] - 1)))])]
        for x_frac in (0.0, 0.25, 0.5, 0.75)
    ]

    assert observed == ["back", "left", "front", "right"]


def test_label_passes_are_stitched_without_interpolation():
    """A label pass must never gain colours that do not exist in the source faces."""
    size = 64
    cube = {}
    for index, face in enumerate(FACE_BASIS):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, : size // 2] = (10 + index, 20 + index, 30 + index)
        image[:, size // 2 :] = (200 - index, 190 - index, 180 - index)
        cube[face] = image
    legal = {int(value) for image in cube.values() for value in np.unique(pack_rgb(image))}

    nearest = stitch_passes(cube, 256, pass_name="material_id")
    linear = stitch_passes(cube, 256, pass_name="albedo")

    assert set(np.unique(pack_rgb(nearest)).tolist()) <= legal
    assert not set(np.unique(pack_rgb(linear)).tolist()) <= legal
    assert "material_id" in LABEL_PASSES and "object_id" in LABEL_PASSES


def test_snap_to_palette_restores_a_discrete_label_image():
    palette = np.array([[10, 20, 30], [200, 190, 180]], dtype=np.uint8)
    blended = np.zeros((8, 8, 3), dtype=np.uint8)
    blended[:, :4] = (12, 22, 33)
    blended[:, 4:] = (198, 188, 178)

    snapped, stats = snap_to_palette(blended, palette)

    assert stats["unique_colors_after"] <= stats["palette_size"]
    assert stats["unique_colors_before"] == 2
    assert stats["status"] == "PASS"
    assert set(np.unique(pack_rgb(snapped)).tolist()) <= set(np.unique(pack_rgb(palette.reshape(1, -1, 3))).tolist())


def test_cubemap_to_erp_rejects_unknown_interpolation():
    with pytest.raises(ValueError, match="interpolation"):
        cubemap_to_erp(_flat_cube(), 256, interpolation="cubic")



def test_radial_depth_condition_uses_visible_percentiles_and_has_range():
    depth = np.linspace(1.0, 20.0, 256, dtype=np.float32).reshape(16, 16)
    condition, stats = radial_depth_condition(depth)

    assert condition.shape == depth.shape
    assert condition.dtype == np.uint8
    assert stats["p2_m"] > 1.0
    assert stats["p98_m"] < 20.0
    assert stats["unique_8bit_values"] > 16
    assert condition[0, 0] > condition[-1, -1]


def test_radial_depth_condition_rejects_nearly_flat_input():
    depth = np.full((16, 16), 4.0, dtype=np.float32)

    try:
        radial_depth_condition(depth)
    except ValueError as error:
        assert "dynamic range" in str(error) or "flat" in str(error)
    else:
        raise AssertionError("Flat radial depth must fail the conditioning gate")
