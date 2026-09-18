import numpy as np

from backend.projection import FACE_BASIS, cubemap_face_uv, cubemap_to_erp, erp_to_world_direction, world_direction_to_erp
from backend.pass_pipeline import radial_depth_condition


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
    face_colors = {
        "front": (0, 0, 255),
        "right": (0, 255, 0),
        "back": (255, 0, 0),
        "left": (255, 255, 0),
        "top": (255, 0, 255),
        "bottom": (0, 255, 255),
    }
    cube = {}
    for face, color in face_colors.items():
        cube[face] = np.full((64, 64, 3), color, dtype=np.uint8)

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
        assert sampled == face_colors[expected_face], (
            f"Expected {expected_face} at ({x_frac}, {y_frac}) but got {sampled}"
        )


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
