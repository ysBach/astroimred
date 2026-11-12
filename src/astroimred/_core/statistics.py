"""Estimator policies shared by photometry and image enhancement."""

from typing import Literal

import numpy as np
import numpy.typing as npt


def _mode_from_stats(
    mean: float | npt.NDArray[np.float64],
    median: float | npt.NDArray[np.float64],
    std: float | npt.NDArray[np.float64] | None = None,
    *,
    method: Literal["mmm", "iraf", "sex"] = "mmm",
) -> float | npt.NDArray[np.float64]:
    """Estimate a mode from already computed scalar or array statistics.

    Parameters
    ----------
    mean, median : float or ndarray
        Location summaries of the same samples. Arrays must broadcast.
    std : float or ndarray or None, optional
        Spread of those samples. Required only for `method="sex"`.
    method : {"mmm", "iraf", "sex"}, optional
        `"mmm"` uses ``3*median - 2*mean``. `"iraf"` uses that formula
        unless the mean is below the median, when it returns the mean.
        `"sex"` uses ``2.5*median - 1.5*mean``, or the median when
        ``std > 0`` and ``(mean - median)/std > 0.3``.

    Returns
    -------
    float or ndarray
        Estimated mode. Inputs are never changed. Subtracting the mean
        before scaling avoids overflow for equal, very bright summaries.

    Notes
    -----
    Sampling, clipping, finite-value policy, and degrees of freedom belong
    to the caller. These formulas are not the iterative DAOPHOT algorithm.
    """
    if method not in {"mmm", "iraf", "sex"}:
        raise ValueError(f"unknown mode method: {method!r}")
    if method == "sex" and std is None:
        raise ValueError("std is required for method='sex'")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = median - mean
        if method == "mmm":
            return median + 2.0 * difference
        if method == "iraf":
            if np.isscalar(mean) and np.isscalar(median):
                return mean if mean < median else median + 2.0 * difference
            result = np.where(mean < median, mean, median + 2.0 * difference)
        else:
            if np.isscalar(std) and np.isscalar(difference):
                use_median = std > 0.0 and -difference / std > 0.3
                return median if use_median else median + 1.5 * difference
            use_median = (std > 0.0) & (-difference / std > 0.3)
            result = np.where(use_median, median, median + 1.5 * difference)
    return float(result) if result.ndim == 0 else result
