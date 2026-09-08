"""Inspect the radial estimates used to enhance an image."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RadialProfile:
    """Read the estimates, sample counts, and coverage of each radial bin.

    Attributes
    ----------
    radius : ndarray
        Integer ring radii for polar sampling, or bin-edge midpoints for
        direct annuli, in pixels.
    statistic : ndarray
        Retained-sample divisor or location, before image transformation.
        Zero/negative values are reported even if unsuitable for division.
    std : ndarray
        Retained sample standard deviation (``ddof=1``), regardless of the
        clipping scale. One sample has zero spread; empty bins have `NaN`.
    n_samples : ndarray
        Total sampling opportunities. Polar rings have `ntheta` angular
        cells. Direct annuli count all in-frame pixel centers in each bin,
        including masked/nonfinite pixels; off-image area is not counted.
    n_valid : ndarray
        Finite samples after both masks, before sigma clipping.
    n_used : ndarray
        Samples retained after clipping. A callable may use these samples
        internally in its own way; that does not change this count.
    sampling : {"polar", "pixel_center"}
        Whether counts refer to polar cells or original pixel centers.
    bin_edges : ndarray or None
        Direct annular edges in pixels. `None` for integer polar rings.

    Notes
    -----
    Counts satisfy ``0 <= n_used <= n_valid <= n_samples``. They describe
    estimation support, not valid output pixels. In polar sampling, repeated
    lookups of one source pixel are not independent measurements.

    Arrays belong to the result and can be edited without changing the input
    image. Only `n_valid` and `n_used` exclude hard and estimation masks;
    `n_samples` counts opportunities before masking. `n_rejected` counts
    sigma rejections only.
    """

    radius: npt.NDArray[np.float64]
    statistic: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    n_samples: npt.NDArray[np.intp]
    n_valid: npt.NDArray[np.intp]
    n_used: npt.NDArray[np.intp]
    sampling: Literal["polar", "pixel_center"]
    bin_edges: npt.NDArray[np.float64] | None = None

    @property
    def n_rejected(self) -> npt.NDArray[np.intp]:
        """Number of samples removed by sigma clipping in each bin."""
        return self.n_valid - self.n_used

    @property
    def coverage(self) -> npt.NDArray[np.float64]:
        """Fraction of sampling opportunities available before clipping.

        Direct annular coverage is relative to in-frame pixel centers, not
        a complete geometric annulus. A bin with no in-frame pixels has `NaN`
        coverage. Use `n_used` to assess support after clipping.
        """
        coverage = np.full(self.n_samples.shape, np.nan)
        np.divide(
            self.n_valid,
            self.n_samples,
            out=coverage,
            where=self.n_samples > 0,
        )
        return coverage


@dataclass(frozen=True)
class EnhancementResult:
    """Access an enhanced image and the radial estimates that produced it.

    Attributes
    ----------
    image : ndarray
        Float64 enhanced image, as returned without profile diagnostics.
    profile : RadialProfile
        Radial estimates, spreads, and sample counts. These are visualization
        diagnostics, not propagated measurement uncertainties.
    """

    image: npt.NDArray[np.float64]
    profile: RadialProfile


def _validate_profile_options(
    statistic_mask: npt.ArrayLike | None,
    shape: tuple[int, int],
    return_profile: bool,
) -> npt.NDArray[np.bool_] | None:
    """Validate opt-in diagnostics and an estimation-only exclusion mask."""
    if not isinstance(return_profile, (bool, np.bool_)):
        raise ValueError("return_profile must be a bool")
    if statistic_mask is None:
        return None
    mask = np.asarray(statistic_mask)
    if mask.shape != shape:
        raise ValueError("statistic_mask must have the same shape as image")
    try:
        return np.asarray(mask, dtype=bool)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "statistic_mask must be convertible to a boolean array"
        ) from exc
