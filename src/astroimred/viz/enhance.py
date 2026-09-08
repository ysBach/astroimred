"""Comet image enhancement helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._enhance_common import (
    _finite_positive,
    _positive_integer,
    _prepare_image,
    _subpixel_offsets,
    _update_mean,
)
from ._enhance_stats import _ring_statistics, _validate_statistics

__all__ = [
    "enhance_rho",
    "azimuthal_normalize",
    "azimuthal_renormalize",
    "annular_normalize",
]


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

    angles = [2.0 * np.pi * (theta_index + 0.5 + offset) / ntheta for offset in offsets]
    directions = [(np.sin(theta)[None, :], np.cos(theta)[None, :]) for theta in angles]
    sample_count = 0
    for radial_offset in offsets:
        radius = radii + radial_offset
        for sin_theta, cos_theta in directions:
            x = cx - radius[:, None] * sin_theta
            y = cy + radius[:, None] * cos_theta
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
            sample_count += 1
            _update_mean(total, np.where(sample_valid, sampled, 0.0), sample_count)

    polar = np.full((nrad, ntheta), np.nan, dtype=np.float64)
    polar[valid] = total[valid]
    return polar


def _polar_reconstruct(
    polar: npt.NDArray[np.float64],
    center: tuple[float, float],
    shape: tuple[int, int],
    subpixels: int,
) -> npt.NDArray[np.float64]:
    """Reconstruct only pixels whose midpoint radii all lie in the polar grid."""
    offsets = _subpixel_offsets(subpixels)
    nrad, ntheta = polar.shape
    cx, cy = center
    x_distances = np.arange(shape[1])[None, :] + offsets[:, None] - cx
    y_distances = np.arange(shape[0])[None, :] + offsets[:, None] - cy
    absolute_x = np.abs(x_distances)
    absolute_y = np.abs(y_distances)
    # The Cartesian product of midpoint coordinates has separable extrema.
    nearest = np.hypot(
        np.min(absolute_x, axis=0)[None, :],
        np.min(absolute_y, axis=0)[:, None],
    )
    farthest = np.hypot(
        np.max(absolute_x, axis=0)[None, :],
        np.max(absolute_y, axis=0)[:, None],
    )
    yy, xx = np.nonzero(
        (nearest > 0.0)
        & (_nearest_integer(nearest) >= 1)
        & (_nearest_integer(farthest) <= nrad)
    )
    total = np.zeros(yy.size)
    valid = np.ones(yy.size, dtype=bool)
    twopi = 2.0 * np.pi
    sample_count = 0
    for y_distance in y_distances:
        dy = y_distance[yy]
        for x_distance in x_distances:
            dx = x_distance[xx]
            radius_index = _nearest_integer(np.hypot(dx, dy))
            angle = np.mod(np.arctan2(-dx, dy), twopi)
            theta_index = np.floor(angle * ntheta / twopi).astype(np.intp) % ntheta
            # Check rounded indices too: floating addition at the outer edge
            # can round a radius just below nrad + 0.5 into the next ring.
            inside = (radius_index >= 1) & (radius_index <= nrad)
            sampled = polar[np.clip(radius_index - 1, 0, nrad - 1), theta_index]
            sample_valid = inside & np.isfinite(sampled)
            valid &= sample_valid
            sample_count += 1
            _update_mean(total, np.where(sample_valid, sampled, 0.0), sample_count)
    result = np.full(shape, np.nan)
    result[yy[valid], xx[valid]] = total[valid]
    return result


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
    subpixels = offsets.size
    yy = np.arange(data.shape[0], dtype=np.float64)[:, None]
    xx = np.arange(data.shape[1], dtype=np.float64)[None, :]
    x_distances = [xx + offset - cx for offset in offsets]
    y_distances = [yy + offset - cy for offset in offsets]
    radius_sum = np.zeros(data.shape, dtype=np.float64)
    for y_distance in y_distances:
        for x_distance in x_distances:
            radius_sum += np.hypot(x_distance, y_distance)
    result = data * (radius_sum / float(subpixels * subpixels))
    result[np.isnan(data)] = np.nan
    return result


def azimuthal_normalize(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    statistic: Literal["mean", "median", "mode"]
    | Callable[[npt.NDArray[np.float64]], float] = "median",
    sigma: float | tuple[float, float] | None = None,
    maxiters: int | None = 1,
    clip_center: Literal["mean", "median"] = "mean",
    clip_scale: Literal["std", "mad_std"] = "std",
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Divide polar rings by a mean, median, mode estimate, or custom statistic.

    Use `statistic="mode", sigma=3, maxiters=5` for a sigma-clipped mode
    estimate. Clipping changes the ring divisor; bright features remain in
    the image being normalized.

    Parameters
    ----------
    image : array-like
        Nonempty, real 2-D image. Masked and nonfinite pixels are missing.
        Pass CCDData pixels and its mask separately.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    nrad : int
        Number of integer radius rings, starting at radius 1 pixel.
    ntheta : int
        Number of equal-angle cells per radius.
    statistic : {"mean", "median", "mode"} or callable, optional
        Ring divisor. `"mode"` estimates ``3*median - 2*mean``; it is an
        empirical approximation, not a histogram mode, and may lie outside
        the observed range. A callable receives a private finite 1-D array
        after clipping and must return a real scalar. Empty rings skip the
        callable. Nonfinite results make the ring undefined.
    sigma : float, tuple of float, or None, optional
        Nonnegative clipping threshold, or ``(lower, upper)`` thresholds.
        `None` disables clipping. Every statistic supports clipping.
    maxiters : int or None, optional
        Maximum clipping passes. The default `1` preserves the original
        one-pass mean rejection. `None` clips until no values are rejected.
    clip_center : {"mean", "median"}, optional
        Location used for clipping bounds, independently of `statistic`.
    clip_scale : {"std", "mad_std"}, optional
        Clipping spread: sample standard deviation (``ddof=1``) or
        Gaussian-consistent median absolute deviation. A singleton has zero
        spread. Zero-width bounds retain only samples equal to the center.
    subpixels : int, optional
        Midpoints per axis for polar sampling and reconstruction. The
        default `10` preserves the source's 10-by-10 quadrature.
    mask : array-like or None, optional
        Additional exclusion mask with the image shape.

    Returns
    -------
    ndarray
        Float64 image with unchanged shape. The input is never modified.
        Zero divisors and incomplete sampling neighborhoods produce `NaN`.
        Negative finite divisors are retained.

    Notes
    -----
    Each polar cell averages equal-angle, equal-radius midpoint samples,
    using nearest-pixel lookup. A missing midpoint invalidates that cell.
    Ring statistics use the remaining finite cells. Reconstruction again
    averages midpoint samples; all must have valid polar cells. Radii below
    0.5 or at or above `nrad + 0.5` have no ring.

    Polar angle zero lies along `+y` and increases toward `-x`. Angular
    cells are centered at ``(j + 0.5) * 2*pi / ntheta``.
    This polar round trip changes spatial sampling. Use
    `annular_normalize` for direct normalization of pixel-center annuli.

    The mode formula matches the MMM estimator described by Photutils:
    https://photutils.readthedocs.io/en/stable/api/photutils.background.MMMBackground.html.
    No automatic estimator fallback or denominator floor is applied.
    """
    nrad = _positive_integer(nrad, "nrad")
    ntheta = _positive_integer(ntheta, "ntheta")
    subpixels = _positive_integer(subpixels, "subpixels")
    options = _validate_statistics(statistic, sigma, maxiters, clip_center, clip_scale)
    data, center_value = _prepare_image(image, center, mask)
    polar = _polar_sample(data, center_value, nrad, ntheta, subpixels)
    scales, _ = _ring_statistics(polar, *options)
    normalized = np.full_like(polar, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        np.divide(
            polar,
            scales[:, None],
            out=normalized,
            where=np.isfinite(scales[:, None]) & (scales[:, None] != 0),
        )
    result = _polar_reconstruct(normalized, center_value, data.shape, subpixels)
    result[np.isnan(data)] = np.nan
    return result


def azimuthal_renormalize(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    nsigma: float,
    statistic: Literal["mean", "median", "mode"]
    | Callable[[npt.NDArray[np.float64]], float] = "mean",
    sigma: float | tuple[float, float] | None = None,
    maxiters: int | None = 1,
    clip_center: Literal["mean", "median"] = "mean",
    clip_scale: Literal["std", "mad_std"] = "std",
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Map each ring's location plus/minus sample deviations to zero and one.

    Parameters
    ----------
    image : array-like
        Nonempty, real 2-D image. Masked and nonfinite pixels are missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    nrad, ntheta : int
        Number of integer radius rings and equal-angle cells per ring.
    nsigma : float
        Finite positive number of sample deviations defining each half-range.
    statistic : {"mean", "median", "mode"} or callable, optional
        Ring location. Default `"mean"` preserves the original behavior.
        `"mode"` is ``3*median - 2*mean``. A callable receives finite
        retained samples and returns a real scalar; empty rings skip it.
    sigma : float, tuple of float, or None, optional
        Clipping threshold or ``(lower, upper)`` pair; `None` disables it.
        Clipping affects the location and spread, not the output samples.
    maxiters : int or None, optional
        Maximum clipping passes; `1` by default. `None` means convergence.
    clip_center : {"mean", "median"}, optional
        Location for clipping bounds, independently of `statistic`.
    clip_scale : {"std", "mad_std"}, optional
        Sample deviation or Gaussian-consistent MAD for clipping only.
        The output range always uses the retained sample standard deviation.
    subpixels : int, optional
        Midpoint samples per axis in polar sampling and reconstruction.
    mask : array-like or None, optional
        Additional exclusion mask with the image shape.

    Returns
    -------
    ndarray
        Float64 image of unchanged shape. Ring values follow
        ``0.5 + (value - location) / (2*nsigma*std)``, without clipping.
        Zero spread, missing neighborhoods, and uncovered radii yield `NaN`.
        The input is never modified.

    Notes
    -----
    Sampling and estimator definitions match `azimuthal_normalize`.
    The source floors its lower bound and scales to bytes; this function
    preserves values outside zero to one and uses no denominator floor.
    A retained sample deviation exceeding the float64 range makes its ring
    undefined, even when a scaled ratio could be represented.
    """
    nrad = _positive_integer(nrad, "nrad")
    ntheta = _positive_integer(ntheta, "ntheta")
    subpixels = _positive_integer(subpixels, "subpixels")
    nsigma = _finite_positive(nsigma, "nsigma")
    options = _validate_statistics(statistic, sigma, maxiters, clip_center, clip_scale)
    data, center_value = _prepare_image(image, center, mask)
    polar = _polar_sample(data, center_value, nrad, ntheta, subpixels)
    location, std = _ring_statistics(polar, *options, return_std=True)
    normalized = np.full_like(polar, np.nan)
    usable = np.isfinite(location) & np.isfinite(std) & (std > 0)
    # Divide in stages so a representable result does not need a finite 2*std.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized[usable] = (polar[usable] / 2.0 - location[usable, None] / 2.0) / std[
            usable, None
        ] / nsigma + 0.5
    result = _polar_reconstruct(normalized, center_value, data.shape, subpixels)
    result[np.isnan(data)] = np.nan
    return result


def annular_normalize(
    image: npt.ArrayLike,
    center: tuple[float, float],
    *,
    radii: npt.ArrayLike,
    statistic: Literal["mean", "median", "mode"]
    | Callable[[npt.NDArray[np.float64]], float] = "median",
    sigma: float | tuple[float, float] | None = None,
    maxiters: int | None = 1,
    clip_center: Literal["mean", "median"] = "mean",
    clip_scale: Literal["std", "mad_std"] = "std",
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Divide original pixels by statistics in explicit radial intervals.

    Parameters
    ----------
    image : array-like
        Nonempty, real 2-D image. Masked and nonfinite pixels are missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image.
    radii : array-like
        At least two finite, strictly increasing nonnegative bin edges in
        pixels. Each annulus includes its inner edge and excludes its outer
        edge. Include zero to normalize the central pixel too.
    statistic : {"mean", "median", "mode"} or callable, optional
        Divisor for each annulus. `"mode"` is ``3*median - 2*mean``.
        A callable receives finite retained pixels and returns a real scalar.
        Empty annuli skip the callable.
    sigma : float, tuple of float, or None, optional
        Clipping threshold or ``(lower, upper)`` pair. `None` disables it.
        Rejected pixels remain in the output; only the divisor uses clipping.
    maxiters : int or None, optional
        Maximum clipping passes; `1` by default. `None` means convergence.
    clip_center : {"mean", "median"}, optional
        Location used for clipping bounds.
    clip_scale : {"std", "mad_std"}, optional
        Sample deviation or Gaussian-consistent MAD used for clipping.
    mask : array-like or None, optional
        Additional exclusion mask with the image shape.

    Returns
    -------
    ndarray
        Float64 image with unchanged shape. Missing pixels, pixels outside
        the radial intervals, and zero/nonfinite divisors produce `NaN`.
        The input is never modified.

    Notes
    -----
    Each finite pixel contributes once with equal weight, according to its
    center radius. There is no polar resampling or fractional area weighting.
    This is a different estimator from `azimuthal_normalize`, particularly
    near the nucleus, masks, and boundaries. Incomplete annuli use their
    available finite pixels and can therefore give biased divisors.

    Pixel radii are computed once and grouped once, avoiding a separate
    full bounding-box scan for every annulus.

    Examples
    --------
    Normalize two annuli using a clipped mode estimate:

    >>> data = np.ones((9, 9))
    >>> result = annular_normalize(
    ...     data, (4, 4), radii=[0, 2, 4], statistic="mode", sigma=3
    ... )
    >>> bool(result[4, 4] == 1.0)
    True
    """
    options = _validate_statistics(statistic, sigma, maxiters, clip_center, clip_scale)
    edges = np.asarray(radii)
    if edges.ndim != 1 or edges.size < 2 or edges.dtype.kind not in "iuf":
        raise ValueError("radii must be a 1-D array of at least two real bin edges")
    edges = np.asarray(edges, dtype=np.float64)
    if not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("radii must be finite, nonnegative, and strictly increasing")
    data, (cx, cy) = _prepare_image(image, center, mask)
    yy = np.arange(data.shape[0], dtype=np.float64)[:, None] - cy
    xx = np.arange(data.shape[1], dtype=np.float64)[None, :] - cx
    bins = np.searchsorted(edges, np.hypot(xx, yy), side="right") - 1
    selected = np.flatnonzero(np.isfinite(data) & (bins >= 0) & (bins < edges.size - 1))
    groups = bins.ravel()[selected]
    order = np.argsort(groups, kind="stable")
    selected = selected[order]
    groups = groups[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, groups.size]
    result = np.full(data.shape, np.nan)
    flat_data = data.ravel()
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        if start == stop:
            continue
        indices = selected[start:stop]
        values = flat_data[indices]
        levels, _ = _ring_statistics(values[None, :], *options)
        if np.isfinite(levels[0]) and levels[0] != 0:
            with np.errstate(over="ignore", invalid="ignore"):
                result.ravel()[indices] = values / levels[0]
    return result
