"""Comet image enhancement helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["enhance_rho", "azimuthal_normalize", "azimuthal_renormalize"]


def _prepare_image(
    image: npt.ArrayLike,
    center: tuple[float, float],
    mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], tuple[float, float]]:
    """Prepare an image and validate its zero-based ``(x, y)`` center.

    Parameters
    ----------
    image : array-like
        A nonempty, real, two-dimensional numeric image. Masked-array masks
        and nonfinite values are converted to `NaN`.
    center : tuple of float
        Zero-based ``(x, y)`` coordinates of the image center. Both
        coordinates must lie within the pixel-center extent of the image.
    mask : array-like, optional
        An additional exclusion mask with the same shape as `image`. Values
        are converted to `bool`.

    Returns
    -------
    data : ndarray
        A private float64 copy of `image`, with excluded pixels set to `NaN`.
    center : tuple of float
        The validated zero-based ``(x, y)`` center.

    Raises
    ------
    TypeError
        If `image` or `center` is not real numeric data.
    ValueError
        If the image, mask, or center has an invalid shape or value.
    """
    masked_image = np.ma.asarray(image)
    raw_image = np.ma.getdata(masked_image)
    if raw_image.ndim != 2 or 0 in raw_image.shape:
        raise ValueError("image must be a nonempty 2-D array")
    if not np.issubdtype(raw_image.dtype, np.number) or np.iscomplexobj(raw_image):
        raise TypeError("image must be real numeric data")

    data = np.array(raw_image, dtype=np.float64, copy=True)
    excluded = np.ma.getmaskarray(masked_image).copy()
    if mask is not None:
        external_mask = np.asarray(mask)
        if external_mask.shape != data.shape:
            raise ValueError(
                "mask must have the same shape as image; "
                f"got {external_mask.shape} and {data.shape}"
            )
        try:
            excluded |= np.asarray(external_mask, dtype=bool)
        except (TypeError, ValueError) as exc:
            raise TypeError("mask must be convertible to a boolean array") from exc
    data[excluded | ~np.isfinite(data)] = np.nan

    raw_center = np.asarray(center)
    if raw_center.shape != (2,):
        raise ValueError("center must contain exactly two coordinates in (x, y) order")
    if raw_center.dtype.kind not in "iuf" or np.iscomplexobj(raw_center):
        raise TypeError("center must contain two real numeric coordinates")
    center_values = np.asarray(raw_center, dtype=np.float64)
    if not np.all(np.isfinite(center_values)):
        raise ValueError("center coordinates must be finite")
    cx, cy = (float(value) for value in center_values)
    height, width = data.shape
    if not (0.0 <= cx <= width - 1 and 0.0 <= cy <= height - 1):
        raise ValueError(
            "center coordinates must lie within the image pixel-center extent "
            f"[0, {width - 1}] x [0, {height - 1}]"
        )

    return data, (cx, cy)


def _subpixel_offsets(subpixels: int) -> npt.NDArray[np.float64]:
    """Return midpoint offsets for one pixel axis.

    Parameters
    ----------
    subpixels : int
        Positive number of equally spaced midpoint samples per pixel axis.

    Returns
    -------
    ndarray
        Offsets from the pixel center, from ``-0.5`` to ``0.5``.

    Raises
    ------
    ValueError
        If `subpixels` is not a positive integer.
    """
    if isinstance(subpixels, (bool, np.bool_)) or not isinstance(
        subpixels, (int, np.integer)
    ):
        raise ValueError("subpixels must be a positive integer")
    subpixels = int(subpixels)
    if subpixels < 1:
        raise ValueError("subpixels must be a positive integer")
    return (np.arange(subpixels, dtype=np.float64) + 0.5) / subpixels - 0.5


def _positive_integer(value: int, name: str) -> int:
    """Validate and normalize a positive integer parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    """Validate a finite nonnegative scalar."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from exc
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return value_float


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite positive scalar."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value_float


def _nearest_integer(values: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Round coordinates to the nearest zero-based pixel index."""
    return np.floor(values + 0.5).astype(np.intp)


def _polar_sample(
    data: npt.NDArray[np.float64],
    center: tuple[float, float],
    nrad: int,
    ntheta: int,
    subpixels: int,
) -> npt.NDArray[np.float64]:
    """Sample `data` into a ``(radius, angle)`` polar image.

    Each radius is an integer ring from 1 through `nrad`. Each angular bin
    has equal angular width. The midpoint grid uses `subpixels` samples in
    both radial and angular directions; a ring/angle sample is undefined if
    any one of its midpoint samples falls outside the image or on a missing
    pixel.
    """
    offsets = _subpixel_offsets(subpixels)
    height, width = data.shape
    cx, cy = center
    radii = np.arange(1, nrad + 1, dtype=np.float64)
    theta_index = np.arange(ntheta, dtype=np.float64)
    total = np.zeros((nrad, ntheta), dtype=np.float64)
    valid = np.ones((nrad, ntheta), dtype=bool)

    for radial_offset in offsets:
        radius = radii + radial_offset
        for angular_offset in offsets:
            theta = 2.0 * np.pi * (theta_index + 0.5 + angular_offset) / ntheta
            x = cx - radius[:, None] * np.sin(theta)[None, :]
            y = cy + radius[:, None] * np.cos(theta)[None, :]
            x_pixel = _nearest_integer(x)
            y_pixel = _nearest_integer(y)
            inside = (
                (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            )
            safe_x = np.clip(x_pixel, 0, width - 1)
            safe_y = np.clip(y_pixel, 0, height - 1)
            sampled = data[safe_y, safe_x]
            sample_valid = inside & np.isfinite(sampled)
            valid &= sample_valid
            total += np.where(sample_valid, sampled, 0.0)

    polar = np.full((nrad, ntheta), np.nan, dtype=np.float64)
    polar[valid] = total[valid] / float(subpixels * subpixels)
    return polar


def _polar_reconstruct(
    polar: npt.NDArray[np.float64],
    center: tuple[float, float],
    shape: tuple[int, int],
    subpixels: int,
) -> npt.NDArray[np.float64]:
    """Reconstruct a Cartesian image from a sampled polar image."""
    offsets = _subpixel_offsets(subpixels)
    height, width = shape
    nrad, ntheta = polar.shape
    cx, cy = center
    yy, xx = np.indices(shape, dtype=np.float64)
    total = np.zeros(shape, dtype=np.float64)
    valid = np.ones(shape, dtype=bool)
    twopi = 2.0 * np.pi

    for y_offset in offsets:
        for x_offset in offsets:
            x_distance = xx + x_offset - cx
            y_distance = yy + y_offset - cy
            radius = np.hypot(x_distance, y_distance)
            radius_index = _nearest_integer(radius)
            angle = np.mod(np.arctan2(-x_distance, y_distance), twopi)
            theta_index = np.floor(angle * ntheta / twopi).astype(np.intp)
            theta_index %= ntheta
            inside = (radius > 0.0) & (radius_index >= 1) & (radius_index <= nrad)
            safe_radius = np.clip(radius_index - 1, 0, nrad - 1)
            sampled = polar[safe_radius, theta_index]
            sample_valid = inside & np.isfinite(sampled)
            valid &= sample_valid
            total += np.where(sample_valid, sampled, 0.0)

    result = np.full(shape, np.nan, dtype=np.float64)
    result[valid] = total[valid] / float(subpixels * subpixels)
    return result


def _mean_and_std(values: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Return a mean and sample standard deviation for finite `values`."""
    if values.size == 0:
        return np.nan, np.nan
    reference = values[0]
    centered = values - reference
    mean = float(reference + np.mean(centered))
    if values.size < 2:
        return mean, 0.0
    return mean, float(np.std(centered, ddof=1))


def _rejected_mean(
    values: npt.NDArray[np.float64], sigma: float | None
) -> tuple[float, float]:
    """Return one-pass sigma-rejected mean and sample standard deviation."""
    mean, std = _mean_and_std(values)
    if sigma is None or not np.isfinite(mean) or not np.isfinite(std):
        return mean, std
    if std == 0.0:
        return mean, std
    lower = mean - sigma * std
    upper = mean + sigma * std
    retained = values[(values >= lower) & (values <= upper)]
    if retained.size == 0:
        return np.nan, np.nan
    return _mean_and_std(retained)


def _normalize_polar(
    polar: npt.NDArray[np.float64],
    statistic: Literal["mean", "median"],
    sigma: float | None,
) -> npt.NDArray[np.float64]:
    """Normalize each polar ring by its selected angular statistic."""
    scales = np.full(polar.shape[0], np.nan, dtype=np.float64)
    for index, row in enumerate(polar):
        values = row[np.isfinite(row)]
        if values.size == 0:
            continue
        if statistic == "median":
            scales[index] = float(np.median(values))
        else:
            scales[index] = _rejected_mean(values, sigma)[0]

    normalized = np.full_like(polar, np.nan, dtype=np.float64)
    for index, scale in enumerate(scales):
        if np.isfinite(scale) and scale != 0.0:
            finite = np.isfinite(polar[index])
            normalized[index, finite] = polar[index, finite] / scale
    return normalized


def enhance_rho(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Multiply image values by their subpixel-averaged radius.

    Parameters
    ----------
    image : array-like
        A nonempty, real, two-dimensional image. Nonfinite and masked pixels
        are treated as missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    subpixels : int, optional
        Midpoint samples per pixel axis. The default `10` follows the
        CometCIEF source's 10-by-10 radius quadrature. Use `1` for center
        sampling.
    mask : array-like, optional
        Additional pixels to treat as missing. Values are converted to
        `bool`.

    Returns
    -------
    ndarray
        A float64 image with the same shape as `image`. The input is never
        modified, and missing input pixels remain `NaN`.

    Notes
    -----
    The source routine first rescales the whole image to mean 100. This API
    omits that presentation-only rescaling so the operation is a direct
    radius multiplication and remains composable with calibrated data.
    """
    data, (cx, cy) = _prepare_image(image, center, mask)
    offsets = _subpixel_offsets(subpixels)
    yy, xx = np.indices(data.shape, dtype=np.float64)
    radius_sum = np.zeros(data.shape, dtype=np.float64)
    for y_offset in offsets:
        for x_offset in offsets:
            radius_sum += np.hypot(xx + x_offset - cx, yy + y_offset - cy)
    result = data * (radius_sum / float(subpixels * subpixels))
    result[np.isnan(data)] = np.nan
    return result


def azimuthal_normalize(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    statistic: Literal["mean", "median"] = "median",
    sigma: float | None = None,
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Divide each radius by its azimuthal mean or median.

    Parameters
    ----------
    image : array-like
        A nonempty, real, two-dimensional image. Nonfinite and masked pixels
        are treated as missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    nrad : int
        Number of integer radius rings, starting at radius 1 pixel.
    ntheta : int
        Number of equal-width angular bins in each radius ring.
    statistic : {``"mean"``, ``"median"``}, optional
        Angular statistic used as the ring divisor. The default is
        ``"median"``.
    sigma : float or None, optional
        One-pass rejection threshold in sample standard deviations for the
        ``"mean"`` statistic. It must be `None` for ``"median"``.
    subpixels : int, optional
        Midpoint samples per pixel axis in both polar sampling and Cartesian
        reconstruction. The default `10` follows the source quadrature.
    mask : array-like, optional
        Additional pixels to treat as missing. Values are converted to
        `bool`.

    Returns
    -------
    ndarray
        A float64 image with the same shape as `image`. Reconstruction samples
        below radius `0.5` or at or above radius `nrad + 0.5` are undefined;
        a pixel is `NaN` if any of its reconstruction subpixels is undefined.
        The input is never modified.

    Raises
    ------
    ValueError
        If `nrad`, `ntheta`, or `subpixels` is invalid, `statistic` is
        unknown, or `sigma` is supplied for the median statistic.

    Notes
    -----
    Each polar sample averages equal-angle midpoint samples over a radial
    ring. A ring/angle sample is undefined if any midpoint lands outside the
    image or on a missing input pixel. A zero ring divisor produces `NaN`
    rather than using an epsilon floor.

    Polar angle `0` is along `+y` and increases counterclockwise toward `-x`.
    Each angular cell is centered at the half-bin angle
    ``(j + 0.5) * 2*pi / ntheta`` before its configurable midpoint samples
    are applied.
    """
    nrad = _positive_integer(nrad, "nrad")
    ntheta = _positive_integer(ntheta, "ntheta")
    _subpixel_offsets(subpixels)
    if not isinstance(statistic, str):
        raise ValueError("statistic must be either 'mean' or 'median'")
    statistic_key = statistic.lower()
    if statistic_key not in {"mean", "median"}:
        raise ValueError("statistic must be either 'mean' or 'median'")
    sigma_value = None if sigma is None else _finite_nonnegative(sigma, "sigma")
    if statistic_key == "median" and sigma_value is not None:
        raise ValueError("sigma is only supported with statistic='mean'")

    data, center_value = _prepare_image(image, center, mask)
    polar = _polar_sample(data, center_value, nrad, ntheta, int(subpixels))
    result = _polar_reconstruct(
        _normalize_polar(polar, statistic_key, sigma_value),
        center_value,
        data.shape,
        int(subpixels),
    )
    result[np.isnan(data)] = np.nan
    return result


def azimuthal_renormalize(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    nsigma: float,
    sigma: float | None = None,
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Scale each radius using its mean plus or minus sample deviations.

    Parameters
    ----------
    image : array-like
        A nonempty, real, two-dimensional image. Nonfinite and masked pixels
        are treated as missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    nrad : int
        Number of integer radius rings, starting at radius 1 pixel.
    ntheta : int
        Number of equal-width angular bins in each radius ring.
    nsigma : float
        Positive number of sample standard deviations around the ring mean
        defining the lower and upper scaling bounds.
    sigma : float or None, optional
        One-pass rejection threshold in sample standard deviations used while
        estimating the ring mean and standard deviation.
    subpixels : int, optional
        Midpoint samples per pixel axis in both polar sampling and Cartesian
        reconstruction. The default `10` follows the source quadrature.
    mask : array-like, optional
        Additional pixels to treat as missing. Values are converted to
        `bool`.

    Returns
    -------
    ndarray
        A float64 image with the same shape as `image`. Values are scaled by
        ``(value - (mean - nsigma*std)) / (2*nsigma*std)`` without clipping.
        Reconstruction samples below radius `0.5` or at or above radius
        `nrad + 0.5` are undefined; a pixel is `NaN` if any of its
        reconstruction subpixels is undefined. Zero variance also produces
        `NaN`.

    Raises
    ------
    ValueError
        If any integer or scalar parameter is invalid.

    Notes
    -----
    The source routine floors the lower bound and maps to the byte range
    0--255. This API uses the documented mean-minus/plus-`nsigma` sample
    standard deviation bounds and maps to the unit interval, preserving
    values outside that interval instead of clipping them.

    Polar angle `0` is along `+y` and increases counterclockwise toward `-x`.
    Each angular cell is centered at the half-bin angle
    ``(j + 0.5) * 2*pi / ntheta`` before its configurable midpoint samples
    are applied.
    """
    nrad = _positive_integer(nrad, "nrad")
    ntheta = _positive_integer(ntheta, "ntheta")
    _subpixel_offsets(subpixels)
    nsigma_value = _finite_positive(nsigma, "nsigma")
    sigma_value = None if sigma is None else _finite_nonnegative(sigma, "sigma")

    data, center_value = _prepare_image(image, center, mask)
    polar = _polar_sample(data, center_value, nrad, ntheta, int(subpixels))
    normalized_polar = np.full_like(polar, np.nan, dtype=np.float64)
    for index, row in enumerate(polar):
        values = row[np.isfinite(row)]
        if values.size == 0:
            continue
        mean, std = _rejected_mean(values, sigma_value)
        if not np.isfinite(mean) or not np.isfinite(std) or std == 0.0:
            continue
        lower = mean - nsigma_value * std
        width = 2.0 * nsigma_value * std
        if not np.isfinite(lower) or not np.isfinite(width) or width == 0.0:
            continue
        finite = np.isfinite(row)
        normalized_polar[index, finite] = (row[finite] - lower) / width

    result = _polar_reconstruct(
        normalized_polar,
        center_value,
        data.shape,
        int(subpixels),
    )
    result[np.isnan(data)] = np.nan
    return result
