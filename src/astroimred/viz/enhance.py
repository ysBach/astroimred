"""Enhance comet images with radial, polar, and annular filters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
import reducers as rd

from ._enhance_common import (
    _finite_positive,
    _finite_scalar,
    _positive_integer,
    _prepare_image,
    _subpixel_offsets,
    _update_mean,
)
from ._enhance_result import (
    EnhancementResult,
    NormStats,
    _validate_result_options,
)
from ._enhance_stats import _ring_statistics, _validate_statistics

__all__ = [
    "EnhancementResult",
    "NormStats",
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
    statistic_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample `data` into a ``(radius, angle)`` polar image.

    Each radius is an integer ring from 1 through `nrad`. Each angular bin
    has equal angular width. The midpoint grid uses `subpixels` samples in
    both radial and angular directions; a ring/angle sample is undefined if
    any one of its midpoint samples falls outside the image or on a missing
    pixel. Return the output samples and the estimation samples. An
    estimation mask excludes a whole polar cell if any midpoint touches it;
    output samples keep their original values.
    """
    offsets = _subpixel_offsets(subpixels)
    height, width = data.shape
    cx, cy = center
    radii = np.arange(1, nrad + 1, dtype=np.float64)
    theta_index = np.arange(ntheta, dtype=np.float64)
    total = np.zeros((nrad, ntheta), dtype=np.float64)
    valid = np.ones((nrad, ntheta), dtype=bool)
    statistic_valid = None if statistic_mask is None else np.ones_like(valid)

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
            if statistic_valid is not None:
                statistic_valid &= ~statistic_mask[safe_y, safe_x]
            sample_count += 1
            _update_mean(total, np.where(sample_valid, sampled, 0.0), sample_count)

    polar = np.full((nrad, ntheta), np.nan, dtype=np.float64)
    polar[valid] = total[valid]
    estimates = (
        polar if statistic_valid is None else np.where(statistic_valid, polar, np.nan)
    )
    return polar, estimates


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
    xmin, xmax = rd.minmax(absolute_x, axis=0)
    ymin, ymax = rd.minmax(absolute_y, axis=0)
    nearest = np.hypot(
        xmin[None, :],
        ymin[:, None],
    )
    farthest = np.hypot(
        xmax[None, :],
        ymax[:, None],
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
    power: float = -1.0,
    subpixels: int = 10,
    mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Compensate a radial brightness profile proportional to ``rho**power``.

    Parameters
    ----------
    image : array-like, shape (height, width)
        Nonempty, real, two-dimensional image. Nonfinite and masked pixels
        are treated as missing.
    center : tuple of float
        Zero-based ``(x, y)`` nucleus coordinates within the image. Fractional
        positions are supported.
    power : float, optional
        Finite exponent of the assumed brightness profile. Multiply by the
        midpoint mean of ``rho**(-power)``, with radius in pixels. Default
        `-1` retains multiplication by mean radius; `-2` uses mean squared
        radius. `0` leaves valid pixel values unchanged.
    subpixels : int, optional
        Midpoint samples per pixel axis. The default `10` follows the
        CometCIEF source's 10-by-10 quadrature; `1` uses pixel centers.
    mask : array-like or None, optional
        Additional boolean-convertible mask with the same shape as `image`.

    Returns
    -------
    ndarray, shape (height, width)
        Float64 image. The input is never modified. Missing inputs, any
        nonfinite midpoint correction, and overflowed products become `NaN`.

    Raises
    ------
    TypeError
        If `image` or `center` is not real numeric data.
    ValueError
        If image/mask geometry, `center`, `power`, or `subpixels` is invalid.

    Notes
    -----
    Each output is ``image * mean(rho**(-power))`` over its midpoint grid,
    not ``image * mean(rho)**(-power)``. This is a discrete midpoint average,
    not analytic pixel-area integration.

    At a sampled ``rho=0``, negative `power` gives zero correction, `power=0`
    gives unity, and positive `power` is singular (`NaN`). A grid that avoids
    zero can return a finite value; near-nucleus results for positive powers
    depend strongly on sampling. For ``power >= 2``, the continuous area
    integral diverges in pixels containing the nucleus even if the discrete
    midpoint mean is finite. No radius floor is added.

    The source routine first rescales the whole image to mean 100; this API
    omits that presentation-only step so calibrated data remains composable.
    No clipping or uncertainty propagation is applied.

    Examples
    --------
    Compensate a ``1/rho**1.5`` profile:

    >>> image = np.ones((3, 3))
    >>> result = enhance_rho(image, (1, 1), power=-1.5, subpixels=1)
    >>> bool(result[1, 2] == 1.0)
    True
    """
    data, (cx, cy) = _prepare_image(image, center, mask)
    power = _finite_scalar(power, "power")
    offsets = _subpixel_offsets(subpixels)
    subpixels = offsets.size
    if power == 0.0:
        return data
    yy = np.arange(data.shape[0], dtype=np.float64)[:, None]
    xx = np.arange(data.shape[1], dtype=np.float64)[None, :]
    x_distances = [xx + offset - cx for offset in offsets]
    y_distances = [yy + offset - cy for offset in offsets]
    correction = np.zeros(data.shape, dtype=np.float64)
    if power == -1.0:
        # Retain the original accumulation order and avoid a unit power call.
        for y_distance in y_distances:
            for x_distance in x_distances:
                correction += np.hypot(x_distance, y_distance)
        correction /= float(subpixels * subpixels)
    else:
        valid = np.ones(data.shape, dtype=bool)
        sample_count = 0
        for y_distance in y_distances:
            for x_distance in x_distances:
                factor = np.hypot(x_distance, y_distance)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    np.power(factor, -power, out=factor)
                finite = np.isfinite(factor)
                valid &= finite
                sample_count += 1
                _update_mean(correction, np.where(finite, factor, 0.0), sample_count)
        correction[~valid] = np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        result = data * correction
    result[~np.isfinite(result)] = np.nan
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
    statistic_mask: npt.ArrayLike | None = None,
    return_stats: bool = False,
) -> npt.NDArray[np.float64] | EnhancementResult:
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
    statistic_mask : array-like or None, optional
        Exclude these pixels from ring estimates only. Original valid samples
        remain in the output. A polar cell is excluded from estimation if
        any midpoint touches this mask. The hard `mask` still excludes output.
    return_stats : bool, optional
        Return an `EnhancementResult` with `.image` and `.stats` when `True`.
        The statistics include levels, sample standard deviations, and counts.
        The default `False` returns only the image and avoids diagnostic work.

    Returns
    -------
    ndarray or EnhancementResult
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
    statistic_mask = _validate_result_options(statistic_mask, data.shape, return_stats)
    polar, estimates = _polar_sample(
        data, center_value, nrad, ntheta, subpixels, statistic_mask
    )
    if return_stats:
        scales, std, n_valid, n_used = _ring_statistics(
            estimates, *options, return_std=True, return_counts=True
        )
    else:
        scales, _ = _ring_statistics(estimates, *options)
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
    if return_stats:
        return EnhancementResult(
            result,
            NormStats(
                radius=np.arange(1, nrad + 1, dtype=np.float64),
                statistic=scales,
                std=std,
                n_samples=np.full(nrad, ntheta, dtype=np.intp),
                n_valid=n_valid,
                n_used=n_used,
                sampling="polar",
            ),
        )
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
    statistic_mask: npt.ArrayLike | None = None,
    return_stats: bool = False,
) -> npt.NDArray[np.float64] | EnhancementResult:
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
    statistic_mask : array-like or None, optional
        Exclude pixels from estimation only, preserving valid output samples.
        A polar cell is excluded if any midpoint touches this mask.
    return_stats : bool, optional
        Return an `EnhancementResult` with `.image` and `.stats` when `True`.
        Default `False` returns an image. Counts refer to polar cells, not
        independent source pixels; see `NormStats`.

    Returns
    -------
    ndarray or EnhancementResult
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
    statistic_mask = _validate_result_options(statistic_mask, data.shape, return_stats)
    polar, estimates = _polar_sample(
        data, center_value, nrad, ntheta, subpixels, statistic_mask
    )
    if return_stats:
        location, std, n_valid, n_used = _ring_statistics(
            estimates, *options, return_std=True, return_counts=True
        )
    else:
        location, std = _ring_statistics(estimates, *options, return_std=True)
    normalized = np.full_like(polar, np.nan)
    usable = np.isfinite(location) & np.isfinite(std) & (std > 0)
    # Divide in stages so a representable result does not need a finite 2*std.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized[usable] = (polar[usable] / 2.0 - location[usable, None] / 2.0) / std[
            usable, None
        ] / nsigma + 0.5
    result = _polar_reconstruct(normalized, center_value, data.shape, subpixels)
    result[np.isnan(data)] = np.nan
    if return_stats:
        return EnhancementResult(
            result,
            NormStats(
                radius=np.arange(1, nrad + 1, dtype=np.float64),
                statistic=location,
                std=std,
                n_samples=np.full(nrad, ntheta, dtype=np.intp),
                n_valid=n_valid,
                n_used=n_used,
                sampling="polar",
            ),
        )
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
    statistic_mask: npt.ArrayLike | None = None,
    return_stats: bool = False,
) -> npt.NDArray[np.float64] | EnhancementResult:
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
    statistic_mask : array-like or None, optional
        Exclude these pixels from each bin's estimate, while dividing their
        original valid values by that estimate. The hard `mask` still
        excludes pixels from both estimation and output.
    return_stats : bool, optional
        Return an `EnhancementResult` with `.image` and `.stats` when `True`.
        Default `False` returns only the image. Coverage is relative to
        in-frame pixel centers, not the area of a complete annulus.

    Returns
    -------
    ndarray or EnhancementResult
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
    statistic_mask = _validate_result_options(statistic_mask, data.shape, return_stats)
    yy = np.arange(data.shape[0], dtype=np.float64)[:, None] - cy
    xx = np.arange(data.shape[1], dtype=np.float64)[None, :] - cx
    bins = np.searchsorted(edges, np.hypot(xx, yy), side="right") - 1
    if return_stats:
        nbins = edges.size - 1
        bin_levels = np.full(nbins, np.nan)
        bin_std = np.full(nbins, np.nan)
        n_valid = np.zeros(nbins, dtype=np.intp)
        n_used = np.zeros(nbins, dtype=np.intp)
        n_samples = np.bincount(bins[(bins >= 0) & (bins < nbins)], minlength=nbins)
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
        estimates = values
        if statistic_mask is not None:
            estimates = np.where(statistic_mask.ravel()[indices], np.nan, values)
        if return_stats:
            levels, std, valid, used = _ring_statistics(
                estimates[None, :], *options, return_std=True, return_counts=True
            )
            bin_index = groups[start]
            bin_levels[bin_index] = levels[0]
            bin_std[bin_index] = std[0]
            n_valid[bin_index] = valid[0]
            n_used[bin_index] = used[0]
        else:
            levels, _ = _ring_statistics(estimates[None, :], *options)
        if np.isfinite(levels[0]) and levels[0] != 0:
            with np.errstate(over="ignore", invalid="ignore"):
                result.ravel()[indices] = values / levels[0]
    if return_stats:
        return EnhancementResult(
            result,
            NormStats(
                radius=edges[:-1] + np.diff(edges) / 2.0,
                statistic=bin_levels,
                std=bin_std,
                n_samples=n_samples,
                n_valid=n_valid,
                n_used=n_used,
                sampling="pixel_center",
                bin_edges=edges.copy(),
            ),
        )
    return result
