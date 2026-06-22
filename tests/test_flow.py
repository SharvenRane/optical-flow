import numpy as np
import pytest

from src.flow import (
    image_gradients,
    lucas_kanade,
    horn_schunck,
    make_shifted_pair,
)


def _central_region(field, margin):
    """Return the interior of a field, dropping ``margin`` pixels per side.

    Optical flow estimates are unreliable near image borders because the
    aggregation window and the smoothness averaging reach outside the image.
    Evaluating on the interior is the standard way to judge accuracy.
    """
    return field[margin:-margin, margin:-margin]


# ---------------------------------------------------------------------------
# Gradient helper
# ---------------------------------------------------------------------------

def test_image_gradients_constant_image_is_zero():
    img = np.full((20, 20), 7.0)
    ix, iy, it = image_gradients(img, img)
    assert np.allclose(ix, 0.0)
    assert np.allclose(iy, 0.0)
    assert np.allclose(it, 0.0)


def test_image_gradients_horizontal_ramp():
    # Intensity increases by 1 per column, so Ix should be 1 and Iy zero.
    ramp = np.tile(np.arange(30, dtype=float), (30, 1))
    ix, iy, it = image_gradients(ramp, ramp)
    inner_ix = _central_region(ix, 2)
    inner_iy = _central_region(iy, 2)
    assert np.allclose(inner_ix, 1.0)
    assert np.allclose(inner_iy, 0.0)
    assert np.allclose(it, 0.0)


def test_image_gradients_shape_mismatch_raises():
    with pytest.raises(ValueError):
        image_gradients(np.zeros((10, 10)), np.zeros((10, 12)))


# ---------------------------------------------------------------------------
# Identical frames give near zero flow
# ---------------------------------------------------------------------------

def test_lucas_kanade_identical_frames_zero_flow():
    frame, _ = make_shifted_pair(shape=(48, 48), shift=(0.0, 0.0), seed=1)
    u, v = lucas_kanade(frame, frame, window_size=11)
    mag = np.sqrt(u * u + v * v)
    assert mag.max() < 1e-6


def test_horn_schunck_identical_frames_zero_flow():
    frame, _ = make_shifted_pair(shape=(48, 48), shift=(0.0, 0.0), seed=2)
    u, v = horn_schunck(frame, frame, alpha=1.0, num_iterations=100)
    mag = np.sqrt(u * u + v * v)
    assert mag.max() < 1e-6


# ---------------------------------------------------------------------------
# Recovering a known translation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)])
def test_lucas_kanade_recovers_translation(shift):
    dx, dy = shift
    frame1, frame2 = make_shifted_pair(shape=(80, 80), shift=shift, seed=3)
    u, v = lucas_kanade(frame1, frame2, window_size=21, sigma=1.0)

    u_in = _central_region(u, 12)
    v_in = _central_region(v, 12)

    assert np.median(u_in) == pytest.approx(dx, abs=0.25)
    assert np.median(v_in) == pytest.approx(dy, abs=0.25)


@pytest.mark.parametrize("shift", [(1.0, 0.0), (0.0, 1.0), (0.8, -0.8)])
def test_horn_schunck_recovers_translation(shift):
    dx, dy = shift
    frame1, frame2 = make_shifted_pair(shape=(80, 80), shift=shift, seed=4)
    u, v = horn_schunck(frame1, frame2, alpha=1.0, num_iterations=400, sigma=1.0)

    u_in = _central_region(u, 12)
    v_in = _central_region(v, 12)

    assert np.median(u_in) == pytest.approx(dx, abs=0.3)
    assert np.median(v_in) == pytest.approx(dy, abs=0.3)


def test_flow_direction_sign_is_correct():
    # A purely rightward shift must give positive u and near zero v.
    frame1, frame2 = make_shifted_pair(shape=(80, 80), shift=(1.5, 0.0), seed=5)
    u, v = lucas_kanade(frame1, frame2, window_size=21, sigma=1.0)
    u_in = _central_region(u, 14)
    v_in = _central_region(v, 14)
    assert np.median(u_in) > 0.5
    assert abs(np.median(v_in)) < 0.25


def test_lucas_kanade_and_horn_schunck_agree():
    # Two independent estimators should land on a similar answer.
    frame1, frame2 = make_shifted_pair(shape=(80, 80), shift=(1.0, 0.5), seed=6)
    u_lk, v_lk = lucas_kanade(frame1, frame2, window_size=21, sigma=1.0)
    u_hs, v_hs = horn_schunck(frame1, frame2, alpha=1.0, num_iterations=400, sigma=1.0)

    u_lk_m = np.median(_central_region(u_lk, 14))
    v_lk_m = np.median(_central_region(v_lk, 14))
    u_hs_m = np.median(_central_region(u_hs, 14))
    v_hs_m = np.median(_central_region(v_hs, 14))

    assert abs(u_lk_m - u_hs_m) < 0.3
    assert abs(v_lk_m - v_hs_m) < 0.3


# ---------------------------------------------------------------------------
# Output shape and validation
# ---------------------------------------------------------------------------

def test_output_shapes_match_input():
    frame1, frame2 = make_shifted_pair(shape=(40, 56), shift=(1.0, 0.0), seed=7)
    u, v = lucas_kanade(frame1, frame2)
    assert u.shape == frame1.shape
    assert v.shape == frame1.shape

    u2, v2 = horn_schunck(frame1, frame2, num_iterations=20)
    assert u2.shape == frame1.shape
    assert v2.shape == frame1.shape


def test_estimators_reject_mismatched_shapes():
    a = np.zeros((10, 10))
    b = np.zeros((10, 11))
    with pytest.raises(ValueError):
        lucas_kanade(a, b)
    with pytest.raises(ValueError):
        horn_schunck(a, b)
