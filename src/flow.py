"""Lucas Kanade and Horn Schunck optical flow implemented from scratch.

The flow convention used here is image coordinate based. ``u`` is the
horizontal displacement (along columns, the x axis) and ``v`` is the vertical
displacement (along rows, the y axis). A positive ``u`` means content moved to
the right between the first and second frame, a positive ``v`` means content
moved down.

Both estimators rely on the brightness constancy assumption, which gives the
linearised optical flow constraint

    Ix * u + Iy * v + It = 0

where ``Ix`` and ``Iy`` are spatial gradients of the image and ``It`` is the
temporal gradient between the two frames.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def _as_float(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as a contiguous 2D float64 array."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {arr.shape}")
    return arr


def image_gradients(frame1: np.ndarray, frame2: np.ndarray):
    """Compute spatial and temporal gradients of an image pair.

    The spatial gradients are averaged across the two frames, which is the
    standard choice for the Horn Schunck and Lucas Kanade derivations because
    the brightness constancy constraint is evaluated between the frames rather
    than at a single instant.

    Parameters
    ----------
    frame1, frame2:
        Two grayscale frames of identical shape.

    Returns
    -------
    (Ix, Iy, It):
        Horizontal spatial gradient, vertical spatial gradient and temporal
        gradient. All three share the shape of the input frames.
    """
    f1 = _as_float(frame1)
    f2 = _as_float(frame2)
    if f1.shape != f2.shape:
        raise ValueError(
            f"frames must have the same shape, got {f1.shape} and {f2.shape}"
        )

    # np.gradient uses central differences in the interior and one sided
    # differences at the borders. axis 0 is rows (y), axis 1 is columns (x).
    iy1, ix1 = np.gradient(f1)
    iy2, ix2 = np.gradient(f2)

    ix = 0.5 * (ix1 + ix2)
    iy = 0.5 * (iy1 + iy2)
    it = f2 - f1
    return ix, iy, it


def lucas_kanade(
    frame1: np.ndarray,
    frame2: np.ndarray,
    window_size: int = 15,
    sigma: float = 1.0,
    regularization: float = 1e-3,
):
    """Estimate dense optical flow with the Lucas Kanade method.

    The method assumes the flow is locally constant inside a window. For every
    pixel it solves the 2x2 weighted least squares system

        [ sum(Ix*Ix)  sum(Ix*Iy) ] [u]   [ -sum(Ix*It) ]
        [ sum(Ix*Iy)  sum(Iy*Iy) ] [v] = [ -sum(Iy*It) ]

    where the sums run over the local window. Aggregation over the window is
    done with a uniform box filter, which is equivalent to summing the per
    pixel products inside the window.

    Parameters
    ----------
    frame1, frame2:
        Two grayscale frames of identical shape.
    window_size:
        Side length in pixels of the square aggregation window.
    sigma:
        Standard deviation of an optional Gaussian pre smoothing applied to
        each frame. Set to ``0`` to disable smoothing.
    regularization:
        Small value added to the diagonal of the structure tensor so that the
        linear system stays solvable in flat regions where it would otherwise
        be singular.

    Returns
    -------
    (u, v):
        Two arrays with the same shape as the input frames holding the
        horizontal and vertical flow per pixel.
    """
    f1 = _as_float(frame1)
    f2 = _as_float(frame2)
    if f1.shape != f2.shape:
        raise ValueError(
            f"frames must have the same shape, got {f1.shape} and {f2.shape}"
        )

    if sigma and sigma > 0:
        f1 = gaussian_filter(f1, sigma)
        f2 = gaussian_filter(f2, sigma)

    ix, iy, it = image_gradients(f1, f2)

    # Per pixel products of the structure tensor and the right hand side.
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy
    ixt = ix * it
    iyt = iy * it

    # Sum each product over the local window. uniform_filter computes a mean,
    # multiplying by the window area turns it back into a sum. The scaling
    # cancels between the matrix and the right hand side, so for solving the
    # system the mean would suffice, but keeping it as a sum matches the
    # textbook derivation.
    area = float(window_size * window_size)
    s_ixx = uniform_filter(ixx, window_size) * area
    s_iyy = uniform_filter(iyy, window_size) * area
    s_ixy = uniform_filter(ixy, window_size) * area
    s_ixt = uniform_filter(ixt, window_size) * area
    s_iyt = uniform_filter(iyt, window_size) * area

    # Solve the 2x2 system at every pixel in closed form using Cramer's rule.
    a = s_ixx + regularization
    d = s_iyy + regularization
    b = s_ixy
    det = a * d - b * b

    rhs_u = -s_ixt
    rhs_v = -s_iyt

    u = (d * rhs_u - b * rhs_v) / det
    v = (-b * rhs_u + a * rhs_v) / det
    return u, v


def horn_schunck(
    frame1: np.ndarray,
    frame2: np.ndarray,
    alpha: float = 1.0,
    num_iterations: int = 300,
    sigma: float = 1.0,
):
    """Estimate dense optical flow with the Horn Schunck method.

    Horn Schunck poses flow estimation as a global energy minimisation that
    balances the brightness constancy constraint against a smoothness term.
    The Euler Lagrange equations lead to the iterative update

        u = u_avg - Ix * (Ix*u_avg + Iy*v_avg + It) / (alpha**2 + Ix**2 + Iy**2)
        v = v_avg - Iy * (Ix*u_avg + Iy*v_avg + It) / (alpha**2 + Ix**2 + Iy**2)

    where ``u_avg`` and ``v_avg`` are local averages of the current flow. The
    ``alpha`` weight controls how strongly the smoothness term pulls the flow
    toward its neighbourhood average.

    Parameters
    ----------
    frame1, frame2:
        Two grayscale frames of identical shape.
    alpha:
        Smoothness regularisation weight. Larger values produce smoother flow.
    num_iterations:
        Number of Jacobi style update iterations.
    sigma:
        Standard deviation of an optional Gaussian pre smoothing applied to
        each frame. Set to ``0`` to disable smoothing.

    Returns
    -------
    (u, v):
        Two arrays with the same shape as the input frames holding the
        horizontal and vertical flow per pixel.
    """
    f1 = _as_float(frame1)
    f2 = _as_float(frame2)
    if f1.shape != f2.shape:
        raise ValueError(
            f"frames must have the same shape, got {f1.shape} and {f2.shape}"
        )

    if sigma and sigma > 0:
        f1 = gaussian_filter(f1, sigma)
        f2 = gaussian_filter(f2, sigma)

    ix, iy, it = image_gradients(f1, f2)

    u = np.zeros_like(f1)
    v = np.zeros_like(f1)

    # 3x3 averaging kernel that excludes the centre pixel, as used in the
    # original Horn Schunck paper.
    kernel = np.array(
        [
            [1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0],
            [1.0 / 6.0, 0.0, 1.0 / 6.0],
            [1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0],
        ]
    )

    denom = alpha * alpha + ix * ix + iy * iy

    for _ in range(int(num_iterations)):
        u_avg = _convolve3x3(u, kernel)
        v_avg = _convolve3x3(v, kernel)

        numer = ix * u_avg + iy * v_avg + it
        common = numer / denom

        u = u_avg - ix * common
        v = v_avg - iy * common

    return u, v


def _convolve3x3(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``field`` with a 3x3 ``kernel`` using edge replication.

    A small dependency free convolution keeps the Horn Schunck update self
    contained. Edges are handled by replicating the border, which avoids
    artificial zero flow being pulled in at the boundary.
    """
    padded = np.pad(field, 1, mode="edge")
    out = np.zeros_like(field)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight == 0.0:
                continue
            out += weight * padded[dy : dy + field.shape[0], dx : dx + field.shape[1]]
    return out


def make_shifted_pair(
    shape=(64, 64),
    shift=(0.0, 0.0),
    seed: int = 0,
    smooth_sigma: float = 2.0,
):
    """Build a synthetic frame pair related by a known sub pixel translation.

    A random texture is generated and smoothed so that it has well defined
    gradients, then shifted by ``shift`` to produce the second frame. The
    shift is the ground truth optical flow that an estimator should recover.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the generated frames.
    shift:
        Ground truth displacement as ``(dx, dy)`` where ``dx`` is horizontal
        (columns) and ``dy`` is vertical (rows). These match the ``(u, v)``
        convention returned by the estimators.
    seed:
        Seed for the random texture so the pair is reproducible.
    smooth_sigma:
        Gaussian smoothing applied to the random texture. Smoothing spreads
        energy across spatial frequencies the linearised flow constraint can
        actually track.

    Returns
    -------
    (frame1, frame2):
        Two float images. ``frame2`` is ``frame1`` translated by ``shift``.
    """
    from scipy.ndimage import shift as ndi_shift

    height, width = shape
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((height, width))
    base = gaussian_filter(base, smooth_sigma)

    # Normalise to a comfortable intensity range.
    base = base - base.min()
    if base.max() > 0:
        base = base / base.max()
    base = base * 255.0

    dx, dy = shift
    # ndi_shift takes the offset in array axis order (rows, cols) = (dy, dx).
    # A positive shift value moves content toward higher indices, so shifting
    # by (dy, dx) makes frame2 the content of frame1 displaced down and right,
    # which corresponds to flow (u, v) = (dx, dy).
    frame1 = base
    frame2 = ndi_shift(base, shift=(dy, dx), order=3, mode="reflect")
    return frame1, frame2
