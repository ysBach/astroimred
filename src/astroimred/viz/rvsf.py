"""Reveal coma structure with a radially variable spatial filter."""

import numpy as np
import numpy.typing as npt

from ._enhance_common import (
    _finite_nonnegative,
    _finite_positive,
    _positive_integer,
    _prepare_image,
    _subpixel_offsets,
)

__all__ = ["radially_variable_filter"]


_NEIGHBORS = (
    (0, -1, 192),
    (0, 1, 192),
    (-1, 0, 192),
    (1, 0, 192),
    (-1, -1, 64),
    (-1, 1, 64),
    (1, -1, 64),
    (1, 1, 64),
)


def _accumulate_step(
    data: npt.NDArray[np.float64],
    steps: npt.NDArray[np.intp],
    usable: npt.NDArray[np.bool_],
    result: npt.NDArray[np.float64],
    invalid: npt.NDArray[np.bool_],
    max_step: npt.NDArray[np.intp],
    previous_steps: npt.NDArray[np.intp],
    previous_stencil: npt.NDArray[np.float64],
    *,
    width: int,
    scale: float,
) -> None:
    """Accumulate one midpoint stencil and reuse unchanged integer widths.

    A pixel's stencil depends only on its rounded integer width. The previous
    width and stencil are cached per pixel, so a repeated width skips the
    eight neighbor gathers. Pixels with an invalid center, neighbor, or image
    boundary remain invalid for later midpoint samples.
    """
    data_flat = data.ravel()
    steps_flat = steps.ravel()
    usable_flat = usable.ravel()
    result_flat = result.ravel()
    invalid_flat = invalid.ravel()
    max_step_flat = max_step.ravel()

    valid = usable_flat & (steps_flat <= max_step_flat) & np.isfinite(data_flat)
    invalid_flat[~valid] = True
    active = valid & ~invalid_flat
    changed = active & (steps_flat != previous_steps)
    changed_pixels = np.flatnonzero(changed)
    if changed_pixels.size:
        step_values = steps_flat[changed_pixels]
        center_values = data_flat[changed_pixels]
        filtered = np.zeros(changed_pixels.size, dtype=np.float64)
        stencil_valid = np.ones(changed_pixels.size, dtype=bool)
        for sy, sx, weight in _NEIGHBORS:
            neighbor = changed_pixels + step_values * (sy * width + sx)
            neighbor_values = data_flat[neighbor]
            finite = np.isfinite(neighbor_values)
            stencil_valid &= finite
            filtered += weight * np.where(
                finite,
                center_values - neighbor_values,
                0.0,
            )

        invalid_pixels = changed_pixels[~stencil_valid]
        invalid_flat[invalid_pixels] = True
        valid_pixels = changed_pixels[stencil_valid]
        previous_steps[valid_pixels] = steps_flat[valid_pixels]
        previous_stencil[valid_pixels] = filtered[stencil_valid]
        active[invalid_pixels] = False

    if not np.any(active):
        return
    result_flat[active] += previous_stencil[active] / scale


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
    """Apply CometCIEF's radially variable eight-neighbor filter.

    The kernel half-width at radius ``rho`` is ``a + b * rho**n`` pixels.
    Larger widths reveal broader structure. The input is never modified.

    Parameters
    ----------
    image : array-like, shape (height, width)
        Nonempty, real 2-D image. Masked-array masks and nonfinite pixels are
        excluded. Pass CCDData pixels and its mask separately.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image. Fractional
        positions are supported. Subtract one from CometCIEF coordinates.
    a, b : float
        Finite, nonnegative kernel coefficients. `a` sets the central
        half-width and `b` controls growth with radius.
    n : float
        Finite, positive radial exponent. Choose it for the image structure.
    log : bool, optional
        Apply ``log10(image)`` before filtering. Every unmasked finite pixel
        must be positive; otherwise raise `ValueError`.
    subpixels : int, optional
        Midpoint samples per pixel axis. The default `10` reproduces the
        source's 10-by-10 width sampling. Use `1` for pixel-center sampling.
    mask : array-like or None, optional
        Additional boolean-convertible exclusion mask with the same shape as
        `image`.

    Returns
    -------
    ndarray
        Float64 image with the same shape. Pixels whose stencil touches
        missing data or the image exterior are `NaN`. No flux normalization,
        clipping, or uncertainty propagation is performed.

    Notes
    -----
    For every midpoint, the half-width is rounded to the nearest pixel;
    half-integers round up. Four axial and four diagonal neighbors then give
    ``1024*I - 192*sum(axial) - 64*sum(diagonal)``. The filter values are
    averaged over the pixel. The coefficients sum to zero, so a constant
    image gives zero wherever the stencil is valid.

    All eight neighbors contribute even when the width rounds to zero.
    This preserves the zero response for constant images at small widths;
    the legacy Fortran code instead drops collapsed axial neighbors.
    Missing neighbors produce `NaN` rather than artificial bright borders.

    The filter changes intensities and can produce negative lobes around
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
    a = _finite_nonnegative(a, "a")
    b = _finite_nonnegative(b, "b")
    n = _finite_positive(n, "n")
    subpixels = _positive_integer(subpixels, "subpixels")
    if not isinstance(log, (bool, np.bool_)):
        raise ValueError("log must be a bool")
    if log:
        valid_pixels = np.isfinite(data)
        if np.any(data[valid_pixels] <= 0):
            raise ValueError("log=True requires positive unmasked finite pixels")
        data[valid_pixels] = np.log10(data[valid_pixels])

    height, width = data.shape
    minimum_axis = min(height, width)
    result = np.zeros(data.shape, dtype=np.float64)
    invalid = np.zeros(data.shape, dtype=bool)
    row_indices = np.arange(height, dtype=np.intp)
    column_indices = np.arange(width, dtype=np.intp)
    row_max_step = np.minimum(row_indices, height - 1 - row_indices)[:, None]
    column_max_step = np.minimum(column_indices, width - 1 - column_indices)[None, :]
    max_step = np.minimum(row_max_step, column_max_step)
    previous_steps = np.full(data.size, -1, dtype=np.intp)
    previous_stencil = np.zeros(data.size, dtype=np.float64)

    if b == 0.0:
        # The width and stencil are identical at every midpoint when b is zero.
        # One evaluation is therefore exactly the subpixel average.
        usable = np.full(data.shape, a < minimum_axis, dtype=bool)
        step = np.zeros(data.shape, dtype=np.intp)
        if a < minimum_axis:
            step.fill(int(np.floor(a + 0.5)))
        _accumulate_step(
            data,
            step,
            usable,
            result,
            invalid,
            max_step,
            previous_steps,
            previous_stencil,
            width=width,
            scale=1.0,
        )
    else:
        offsets = _subpixel_offsets(subpixels)
        x_distances = np.arange(width, dtype=np.float64)[None, :] - cx
        y_distances = np.arange(height, dtype=np.float64)[:, None] - cy
        sample_count = float(subpixels * subpixels)
        for dy in offsets:
            for dx in offsets:
                rho = np.hypot(x_distances + dx, y_distances + dy)
                with np.errstate(over="ignore", invalid="ignore"):
                    half_width = a + b * rho**n
                # A step as large as an image axis cannot have a valid stencil.
                # Bound it before converting to integers.
                usable = np.isfinite(half_width) & (half_width < minimum_axis)
                step = np.zeros(data.shape, dtype=np.intp)
                step[usable] = np.floor(half_width[usable] + 0.5).astype(np.intp)
                _accumulate_step(
                    data,
                    step,
                    usable,
                    result,
                    invalid,
                    max_step,
                    previous_steps,
                    previous_stencil,
                    width=width,
                    scale=sample_count,
                )
    result[invalid] = np.nan
    return result
