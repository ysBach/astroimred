"""Reveal coma structure with Larson–Sekanina rotational differences."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from ._enhance_common import _prepare_image

__all__ = ["larson_sekanina"]


def _finite_scalar(value: float, name: str) -> float:
    """Validate a signed real scalar without accepting booleans or strings."""
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _lerp(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
    fraction: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate finite endpoints without overflow or zero-weight NaNs.

    Same-sign differences are bounded; opposite-sign weighted sums are
    bounded. The difference form also preserves constant fields exactly.
    Select endpoints explicitly so unused missing neighbors cannot leak in.
    """
    same_sign = np.signbit(first) == np.signbit(second)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.where(
            same_sign,
            first + fraction * (second - first),
            (1.0 - fraction) * first + fraction * second,
        )
    return np.where(fraction == 0.0, first, np.where(fraction == 1.0, second, result))


def _sample(
    data: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    interpolation: Literal["linear", "nearest"],
) -> npt.NDArray[np.float64]:
    """Gather Cartesian samples with strict bounds and finite support."""
    height, width = data.shape
    inside = valid & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    # Bound coordinates before conversion to avoid overflow in integer indices.
    x = np.where(inside, x, 0.0)
    y = np.where(inside, y, 0.0)
    if interpolation == "nearest":
        result = data[
            np.floor(y + 0.5).astype(np.intp), np.floor(x + 0.5).astype(np.intp)
        ]
    else:
        x0, y0 = x.astype(np.intp), y.astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        result = _lerp(
            _lerp(data[y0, x0], data[y0, x1], fx),
            _lerp(data[y1, x0], data[y1, x1], fx),
            fy,
        )
    result[~inside] = np.nan
    return result


def larson_sekanina(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    angle: float,
    radial_shift: float = 0.0,
    interpolation: Literal["linear", "nearest"] = "linear",
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Subtract two oppositely rotated, radially shifted coma images.

    Parameters
    ----------
    image : array-like
        Nonempty, real 2-D image. Masked and nonfinite pixels are excluded.
        The input is never modified.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
        Fractional coordinates are supported; indexing is ``image[y, x]``.
        Angles increase from +x toward +y (counterclockwise when displayed
        with ``origin="lower"``); both rotation directions contribute.
    angle : float
        Angular shift in degrees. Both positive and negative rotations are
        sampled, so changing the sign gives the same result. Full turns are
        reduced modulo 360. Zero gives twice a purely radial difference.
    radial_shift : float, optional
        Radial displacement in pixels. Positive values sample inward at
        ``r - radial_shift``; negative values sample outward. Default `0`
        selects a purely rotational difference.
    interpolation : {"linear", "nearest"}, optional
        Default `"linear"` uses bilinear interpolation. `"nearest"` selects
        the nearest pixel, rounding half-integers upward. Every contributor
        with nonzero weight must be finite; missing neighbors with zero
        weight are ignored. Neither method extrapolates beyond pixel centers.
    mask : array-like, optional
        Exclusion mask matching `image`; `True` marks missing pixels.
        Combined with the masked-array mask and nonfinite values.

    Returns
    -------
    ndarray
        Float64 residual with the input shape and intensity units. Missing
        original pixels or either missing shifted sample give `NaN`.
        Shifted radii with ``r - radial_shift < 0`` are undefined. At ``r = 0``, a
        nonzero radial shift is undefined because no direction is available.
        Results outside float64 range are also `NaN`.

    Raises
    ------
    TypeError
        If `image` or `center` is not real numeric data.
    ValueError
        If shapes, coordinates, shifts, or `interpolation` are invalid.

    Notes
    -----
    Implements Larson & Sekanina (1984), equation (1), with ``dr`` in pixels
    and ``da`` in degrees::

        L(r, theta) = (I(r, theta) - I(r - dr, theta + da))
                    + (I(r, theta) - I(r - dr, theta - da))

    The two differences are summed, not averaged. Cartesian sampling avoids
    a polar-grid round trip. Right-angle rotations use exact sine/cosine
    values; no coordinate snapping or boundary tolerance is applied.

    Rotational differences emphasize angular structure; radial displacement
    also reveals concentric structure. Choose shifts for the feature scale
    in your image. Interpolation smooths fine structure, and the filter can
    create negative lobes around stars, jets, or centering errors. Use the
    original calibrated image for photometry. No logarithm or background
    subtraction is applied.

    See Also
    --------
    radially_variable_filter : Filter with a radius-dependent spatial width.
    azimuthal_normalize : Divide by an estimated azimuthal brightness level.

    References
    ----------
    Larson, S. M. & Sekanina, Z. (1984), AJ 89, 571–578, equation (1).
    https://doi.org/10.1086/113551

    Examples
    --------
    A constant field has no rotational structure:

    >>> from astroimred.viz import larson_sekanina
    >>> data = np.ones((9, 9))
    >>> residual = larson_sekanina(data, (4, 4), angle=10)
    >>> bool(residual[4, 4] == 0)
    True
    """
    data, (cx, cy) = _prepare_image(image, center, mask)
    angle = abs(_finite_scalar(angle, "angle")) % 360.0
    radial_shift = _finite_scalar(radial_shift, "radial_shift")
    if not isinstance(interpolation, str) or interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    if angle == 0.0 and radial_shift == 0.0:
        # Identity geometry: avoid coordinate roundoff beside masked pixels.
        data[np.isfinite(data)] = 0.0
        return data

    if angle % 90.0 == 0.0:
        cosine, sine = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[
            int(angle / 90.0)
        ]
    else:
        radians = np.deg2rad(angle)
        cosine, sine = np.cos(radians), np.sin(radians)

    y, x = np.ogrid[: data.shape[0], : data.shape[1]]
    dx, dy = x - cx, y - cy
    valid = np.isfinite(data)
    if radial_shift != 0.0:
        radius = np.hypot(dx, dy)
        shifted_radius = radius - radial_shift
        valid &= (radius > 0.0) & (shifted_radius >= 0.0)
        # Unit directions avoid an unbounded shifted_radius / radius ratio.
        ux = np.divide(dx, radius, out=np.zeros_like(radius), where=radius > 0.0)
        uy = np.divide(dy, radius, out=np.zeros_like(radius), where=radius > 0.0)
        dx, dy = ux * shifted_radius, uy * shifted_radius

    with np.errstate(over="ignore", invalid="ignore"):
        plus = _sample(
            data,
            cx + cosine * dx - sine * dy,
            cy + sine * dx + cosine * dy,
            valid,
            interpolation,
        )
        if sine == 0.0:
            minus = plus  # Both source positions coincide at 0 and 180 degrees.
        else:
            minus = _sample(
                data,
                cx + cosine * dx + sine * dy,
                cy - sine * dx + cosine * dy,
                valid,
                interpolation,
            )
        result = (data - plus) + (data - minus)
        overflow = valid & np.isfinite(plus) & np.isfinite(minus) & ~np.isfinite(result)
        # Recover representable residuals whose intermediate difference overflowed.
        result[overflow] = 2.0 * (
            (data[overflow] / 2.0 - plus[overflow] / 2.0)
            + (data[overflow] / 2.0 - minus[overflow] / 2.0)
        )
    result[~np.isfinite(result)] = np.nan
    return result
