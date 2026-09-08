"""Reveal coma structure with a radially variable spatial filter."""

import numpy as np
import numpy.typing as npt

from .enhance import _prepare_image, _subpixel_offsets

__all__ = ["radially_variable_filter"]


def radially_variable_filter(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    a: float,
    b: float,
    n: float,
    log: bool = False,
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Apply the CometCIEF radially variable eight-neighbor filter.

    Choose `a`, `b`, and `n` for the feature sizes in your image. The kernel
    half-width in pixels is ``a + b * rho**n``. Larger widths reveal broader
    structure. The input is never modified.

    Parameters
    ----------
    image : array-like
        Nonempty, real 2-D image. Masked-array masks and nonfinite pixels
        are excluded. Pass CCDData pixels and its mask separately.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image. Fractional
        positions are supported. Subtract one from CometCIEF coordinates.
    a, b : float
        Finite, nonnegative kernel coefficients. `a` sets the central
        half-width; `b` controls its growth with radius.
    n : float
        Finite, positive radial exponent. Select explicitly for your image.
    log : bool, optional
        Apply ``log10(image)`` before filtering. Every unmasked finite pixel
        must be positive; otherwise raise `ValueError`.
    subpixels : int, optional
        Midpoint samples per pixel axis. The default `10` reproduces the
        source's 10-by-10 width sampling. Use `1` for pixel-center sampling.
    mask : array-like of bool or None, optional
        Additional exclusion mask with the same shape as `image`.

    Returns
    -------
    ndarray
        Float64 image with the same shape. Pixels whose stencil touches
        missing data or the image exterior are `NaN`. No flux normalization,
        clipping, or uncertainty propagation is performed.

    Notes
    -----
    At each subpixel, round the kernel half-width to the nearest pixel
    (half-integers round up), sample four axial and four diagonal neighbors,
    and evaluate ``1024*I - 192*sum(axial) - 64*sum(diagonal)``. Average these
    filter values across the pixel. The coefficients sum to zero, so a
    constant image gives zero wherever the stencil is valid.

    All eight neighbors contribute even when the width rounds to zero.
    This preserves the zero response for constant images at small widths;
    the legacy Fortran code instead drops collapsed axial neighbors.
    Missing neighbors produce `NaN` rather than artificial bright borders.

    This filter changes intensities and can produce negative lobes around
    stars and jets. Use the original calibrated image for photometry.

    References
    ----------
    Samarasinha, Martin, and Larson (2013), Cometary Coma Image Enhancement
    Facility, https://cie.psi.edu/. The source attributes this algorithm
    to Steve Larson. Larson-Sekanina rotational filtering is a separate
    method and is not implemented here.

    Examples
    --------
    A constant image has no local structure:

    >>> from astroimred.viz import radially_variable_filter
    >>> data = np.ones((7, 7))
    >>> result = radially_variable_filter(data, (3, 3), a=1, b=0, n=1)
    >>> bool(result[3, 3] == 0)
    True
    """
    data, (cx, cy) = _prepare_image(image, center, mask)
    offsets = _subpixel_offsets(subpixels)
    for name, value in (("a", a), ("b", b), ("n", n)):
        if np.ndim(value) != 0 or not np.isrealobj(value):
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            valid = np.isfinite(value) and (value > 0 if name == "n" else value >= 0)
        except TypeError as exc:
            raise ValueError(f"{name} must be a finite real scalar") from exc
        if not valid:
            comparison = "positive" if name == "n" else "nonnegative"
            raise ValueError(f"{name} must be finite and {comparison}")
    if not isinstance(log, (bool, np.bool_)):
        raise ValueError("log must be a bool")
    if log:
        valid_pixels = np.isfinite(data)
        if np.any(data[valid_pixels] <= 0):
            raise ValueError("log=True requires positive unmasked finite pixels")
        data[valid_pixels] = np.log10(data[valid_pixels])

    height, width = data.shape
    yy, xx = np.indices(data.shape)
    result = np.zeros(data.shape, dtype=np.float64)
    neighbors = (
        (0, -1, 192),
        (0, 1, 192),
        (-1, 0, 192),
        (1, 0, 192),
        (-1, -1, 64),
        (-1, 1, 64),
        (1, -1, 64),
        (1, 1, 64),
    )
    for dy in offsets:
        for dx in offsets:
            rho = np.hypot(xx - cx + dx, yy - cy + dy)
            with np.errstate(over="ignore", invalid="ignore"):
                half_width = a + b * rho**n if b != 0 else np.full(data.shape, a)
            # A step as large as an image axis cannot have a valid stencil.
            # Bound it before converting to integers.
            usable = np.isfinite(half_width) & (half_width < min(height, width))
            step = np.floor(np.where(usable, half_width, 0) + 0.5).astype(np.intp)
            filtered = np.zeros(data.shape, dtype=np.float64)
            for sy, sx, weight in neighbors:
                py, px = yy + sy * step, xx + sx * step
                inside = usable & (py >= 0) & (py < height) & (px >= 0) & (px < width)
                sampled = data[np.clip(py, 0, height - 1), np.clip(px, 0, width - 1)]
                # This difference form cancels a constant field before scaling.
                filtered += weight * (data - np.where(inside, sampled, np.nan))
            result += filtered / (subpixels * subpixels)
    return result
