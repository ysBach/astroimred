"""Estimate ring levels with finite-only reductions and optional clipping."""

from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import reducers as rd

from ._enhance_common import _positive_integer

Statistic = str | Callable[[npt.NDArray[np.float64]], float]
Sigma = float | tuple[float, float] | None
_MAD_TO_STD = 1.0 / NormalDist().inv_cdf(0.75)


def _validate_statistics(
    statistic: Statistic,
    sigma: Sigma,
    maxiters: int | None,
    clip_center: str,
    clip_scale: str,
) -> tuple[Statistic, tuple[float, float] | None, int | None, str, str]:
    """Validate estimator and clipping options before sampling an image."""
    if isinstance(statistic, str):
        statistic = statistic.lower()
        if statistic not in {"mean", "median", "mode"}:
            raise ValueError("statistic must be 'mean', 'median', 'mode', or callable")
    elif not callable(statistic):
        raise ValueError("statistic must be 'mean', 'median', 'mode', or callable")
    thresholds = None
    if sigma is not None:
        raw = np.asarray(sigma)
        if raw.dtype.kind not in "iuf" or raw.shape not in {(), (2,)}:
            raise ValueError(
                "sigma must be a nonnegative scalar or (lower, upper) pair"
            )
        values = np.broadcast_to(raw, (2,)).astype(np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("sigma thresholds must be finite and nonnegative")
        thresholds = (float(values[0]), float(values[1]))
    if maxiters is not None:
        maxiters = _positive_integer(maxiters, "maxiters")
    if not isinstance(clip_center, str) or clip_center not in {"mean", "median"}:
        raise ValueError("clip_center must be 'mean' or 'median'")
    if not isinstance(clip_scale, str) or clip_scale not in {"std", "mad_std"}:
        raise ValueError("clip_scale must be 'std' or 'mad_std'")
    return statistic, thresholds, maxiters, clip_center, clip_scale


def _mean_std(
    values: npt.NDArray[np.float64], *, return_std: bool = True
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Center before scaling, preserving narrow spreads on a bright pedestal."""
    low, high = rd.nanminmax(values, axis=1, ignore_inf=True)
    reference = low / 2.0 + high / 2.0
    scale = np.maximum(high - reference, reference - low)
    scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
    centered = (values - reference[:, None]) / scale[:, None]
    mean = reference + scale * rd.nanmean(centered, axis=1)
    std = np.full_like(mean, np.nan)
    if return_std:
        with np.errstate(over="ignore", invalid="ignore"):
            std = scale * rd.nanstd(centered, axis=1, ddof=1)
        std[np.count_nonzero(np.isfinite(values), axis=1) == 1] = 0.0
    return mean, std


def _median(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Select raw values, repairing overflow of an even pair's finite sum."""
    median = rd.nanmedian(values, axis=1)
    overflow = np.isinf(median)
    if np.any(overflow):
        lower = rd.lmedian(values[overflow], axis=1)
        upper = -rd.lmedian(-values[overflow], axis=1)
        midpoint = lower.copy()
        distinct = lower != upper
        midpoint[distinct] = lower[distinct] / 2.0 + upper[distinct] / 2.0
        median[overflow] = midpoint
    return median


def _ring_statistics(
    samples: npt.NDArray[np.float64],
    statistic: Statistic,
    sigma: tuple[float, float] | None,
    maxiters: int | None,
    clip_center: str,
    clip_scale: str,
    *,
    return_std: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reduce rows; clipping affects estimates, never the original samples.

    Keep original values for selection and clipping. Center and scale only
    the variance calculation: a large reference must not erase a faint
    median in a ring containing a bright star.
    """
    retained = np.where(np.isfinite(samples), samples, np.nan)
    if sigma is not None:
        iteration = 0
        while maxiters is None or iteration < maxiters:
            if clip_center == "mean" or clip_scale == "std":
                location, spread = _mean_std(retained, return_std=clip_scale == "std")
            if clip_center == "median":
                location = _median(retained)
            if clip_scale == "mad_std":
                median = _median(retained)
                with np.errstate(over="ignore", invalid="ignore"):
                    spread = _MAD_TO_STD * _median(np.abs(retained - median[:, None]))
            with np.errstate(over="ignore", invalid="ignore"):
                # Zero sigma means an exact bound even if spread overflowed.
                lower = location if sigma[0] == 0 else location - sigma[0] * spread
                upper = location if sigma[1] == 0 else location + sigma[1] * spread
            rejected = (retained < lower[:, None]) | (retained > upper[:, None])
            if not np.any(rejected):
                break
            retained[rejected] = np.nan
            iteration += 1

    std = np.full(samples.shape[0], np.nan)
    if return_std or (isinstance(statistic, str) and statistic in {"mean", "mode"}):
        mean, std = _mean_std(retained, return_std=return_std)
    if callable(statistic):
        levels = np.full(samples.shape[0], np.nan)
        for index, row in enumerate(retained):
            values = row[np.isfinite(row)]
            if values.size:
                result = np.asarray(statistic(values))
                if result.shape != () or result.dtype.kind not in "iuf":
                    raise ValueError("a statistic callable must return a real scalar")
                levels[index] = float(result)
    elif statistic == "mean":
        levels = mean
    else:
        levels = _median(retained)
        if statistic == "mode":
            with np.errstate(over="ignore", invalid="ignore"):
                levels += 2.0 * (levels - mean)
    levels[~np.isfinite(levels)] = np.nan
    return levels, std
