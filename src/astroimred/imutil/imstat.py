"""IRAF IMSTAT-like image statistics and uncertainty helpers."""

import os
from collections.abc import Sequence

import numpy as np
import reducers.lowlevel as rdl
from astro_ndslice import slicefy
from astropy import units as u
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from .._core.types import SectionLike, StrPathLike
from ..logging import logger

try:
    import numexpr as ne

    HAS_NE = True
except ImportError:
    HAS_NE = False

__all__ = [
    "errormap",
    "give_stats",
]

_MAD_STD_NORMAL_SCALE = 1.482602218505602


def _data_header_from_array_or_path(
    item: np.ndarray | StrPathLike,
    extension: int | str | None = None,
) -> tuple[np.ndarray, fits.Header | None]:
    if isinstance(item, np.ndarray):
        return item, None
    if isinstance(item, (str, os.PathLike)):
        with fits.open(item) as hdul:
            data = hdul[extension if extension is not None else 0].data.copy()
            hdr = hdul[extension if extension is not None else 0].header.copy()
        return data, hdr
    raise TypeError(
        "imstat helpers accept numpy.ndarray or path-like FITS inputs. "
        f"Received {type(item)}."
    )


def errormap(
    ccd_biassub: np.ndarray | StrPathLike,
    gain_epadu: float | u.Quantity = 1,
    rdnoise_electron: float | u.Quantity = 0,
    subtracted_dark: float | np.ndarray = 0.0,
    flat: float | np.ndarray = 1.0,
    dark_std: float | np.ndarray = 0.0,
    flat_err: float | np.ndarray = 0.0,
    dark_std_min: float | str = "rdnoise",
    return_variance: bool = False,
) -> np.ndarray:
    r"""Propagate independent detector and calibration errors through a flat.

    Parameters
    ----------
    ccd_biassub : ndarray or path-like
        Bias-subtracted signal D in ADU, before flat division. If a dark was
        subtracted, D must be the dark-subtracted signal and `subtracted_dark`
        must contain the removed counts. For CCDData/HDU input, pass `.data`.
    gain_epadu : float or Quantity, optional
        Detector gain g in electrons per ADU.
    rdnoise_electron : float or Quantity, optional
        Science-exposure read noise R in electrons.
    subtracted_dark : float or ndarray, optional
        Removed dark counts in pre-flat ADU. Restores their science-exposure
        Poisson noise; this is distinct from dark-calibration uncertainty.
    flat : float or ndarray, optional
        Dimensionless divisor F. Use 1 for no flat division. An already
        flat-divided signal must first be multiplied by its original F.
    dark_std : float or ndarray, optional
        Independent uncertainty s_D of the subtracted dark calibration in
        pre-flat ADU, after any exposure scaling. Excludes science-exposure
        Poisson and read noise, which are included separately.
    flat_err : float or ndarray, optional
        Absolute uncertainty s_F of F. Zero treats the flat as exact but
        still propagates detector noise through division by F.
    dark_std_min : float or str, optional
        Floor applied to ndarray `dark_std`. ``"rdnoise"`` means R/g.
        Scalar `dark_std` is used as supplied, retaining the zero default
        when no dark-calibration uncertainty is requested.
    return_variance : bool, optional
        Return variance in ADU squared instead of standard deviation in ADU.

    Returns
    -------
    ndarray
        Variance or standard deviation of D/F.

    Notes
    -----
    Assuming independent noise sources, the variance is

    .. math::

        V = \frac{\max(D + D_{dark}, 0)/g + s_D^2 + (R/g)^2}{F^2}
            + \left(\frac{D s_F}{F^2}\right)^2.

    Only the estimated Poisson counts are clipped to zero; negative D still
    contributes to flat-calibration uncertainty. F must be nonzero.
    """
    data, _ = _data_header_from_array_or_path(ccd_biassub)
    data = np.asarray(data, dtype=np.result_type(data.dtype, np.float64))
    flat = np.asarray(flat, dtype=np.result_type(flat, np.float64))
    poisson = np.maximum(data + subtracted_dark, 0)

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value
    elif isinstance(gain_epadu, str):
        gain_epadu = float(gain_epadu)

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron).value
    elif isinstance(rdnoise_electron, str):
        rdnoise_electron = float(rdnoise_electron)

    if dark_std_min == "rdnoise":
        dark_std_min = rdnoise_electron / gain_epadu
    if isinstance(dark_std, np.ndarray):
        dark_std = np.maximum(dark_std, dark_std_min)

    # Calculate the full variance map
    # restore dark for Poisson term calculation
    if HAS_NE:
        eval_str = (
            "poisson/(gain_epadu*flat**2)"
            "+ (dark_std/flat)**2"
            "+ (data*flat_err/flat**2)**2"
            "+ (rdnoise_electron/(gain_epadu*flat))**2"
        )
        if return_variance:
            return ne.evaluate(eval_str)
        else:  # Sqrt is the most time-consuming part...
            return ne.evaluate(f"sqrt({eval_str})")
    else:
        variance = (
            poisson / (gain_epadu * flat**2)
            + (dark_std / flat) ** 2
            + (data * flat_err / flat**2) ** 2
            + (rdnoise_electron / (gain_epadu * flat)) ** 2
        )
        if return_variance:
            return variance
        else:
            return np.sqrt(variance)


def _finite_reducer_values(data: np.ndarray) -> np.ndarray:
    """Return native-endian finite 1-D values accepted by reducers."""
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    dtype = data.dtype
    if dtype.byteorder not in {"=", "|"}:
        data = data.astype(dtype.newbyteorder("="), copy=False)
    return np.ascontiguousarray(data)


def _normalize_num_extrema(
    num_extrema: tuple[int, int] | Sequence[int] | None,
) -> tuple[int, int] | None:
    """Return validated low/high extrema counts."""
    if num_extrema is None:
        return None
    try:
        n_lo, n_hi = num_extrema
    except (TypeError, ValueError) as err:
        raise ValueError("num_extrema must be None or a 2-item sequence.") from err
    n_lo = int(n_lo)
    n_hi = int(n_hi)
    if n_lo < 0 or n_hi < 0:
        raise ValueError("num_extrema counts must be non-negative.")
    return n_lo, n_hi


# TODO: add sigma-clipped statistics option (hdr key can be using "SIGC", e.g., SIGCAVG.)
def give_stats(
    item: np.ndarray | StrPathLike,
    mask: np.ndarray | None = None,
    extension: int | str | None = None,
    statsecs: SectionLike | list[SectionLike] = None,
    percentiles: Sequence[float] | None = (1, 99),
    num_extrema: tuple[int, int] | Sequence[int] | None = (1, 1),
    return_header: bool = False,
) -> dict | tuple[dict, fits.Header]:
    """Calculates simple statistics.

    ``item`` is now intentionally accepted as either `~numpy.ndarray` or
    path-like FITS input. For CCDData/HDU inputs, pass their `.data` explicitly.

    Parameters
    ----------
    percentiles : sequence of float or `None`, optional
        Percentiles to calculate. If `None`, percentile calculation is skipped.
    num_extrema : 2-tuple of int or `None`, optional
        Number of low and high extreme values to report as ``(n_lo, n_hi)``. If
        `None`, extrema calculation is skipped.
    """
    data, hdr = _data_header_from_array_or_path(item, extension=extension)
    if mask is not None:
        data = np.array(data, copy=True)
        data[mask] = np.nan

    if statsecs is not None:
        statsecs = [statsecs] if isinstance(statsecs, str) else list(statsecs)
        data = np.array([data[slicefy(sec)] for sec in statsecs])

    data = _finite_reducer_values(data)

    std, mean = rdl.std_mean_valid(data, ddof=1)
    d_min, d_max = rdl.minmax_valid(data)
    med = rdl.median_valid(data)
    d_zmin, d_zmax = ZScaleInterval().get_limits(data)

    result = {
        "num": np.size(data),
        "min": d_min,
        "max": d_max,
        "avg": mean,
        "med": med,
        "std": std,
        "slices": statsecs,
    }
    result["madstd"] = _MAD_STD_NORMAL_SCALE * rdl.median_valid_in_place(
        np.abs(data - med)
    )
    if percentiles is not None:
        result["percentiles"] = percentiles
        result["pct"] = rdl.percentiles_valid_in_place(data, percentiles)
    # d_pct = np.percentile(data, percentiles)
    # for i, pct in enumerate(percentiles):
    #     result[f"percentile_{round(pct, 4)}"] = d_pct[i]

    result["zmin"] = d_zmin
    result["zmax"] = d_zmax

    extrema_counts = _normalize_num_extrema(num_extrema)
    if extrema_counts is not None:
        n_lo, n_hi = extrema_counts
        if n_lo + n_hi > result["num"]:
            logger.warning(
                "Extrema overlaps (n_lo + n_hi (%s) > N_pix (%s))",
                n_lo + n_hi,
                result["num"],
            )
        if n_lo == 0:
            d_los = np.array([], dtype=data.dtype)
        else:
            d_los = np.sort(np.partition(data, n_lo - 1)[:n_lo])
        if n_hi == 0:
            d_his = np.array([], dtype=data.dtype)
        else:
            d_his = np.sort(np.partition(data, -n_hi)[-n_hi:])
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    if return_header and hdr is not None:
        hdr["STATNPIX"] = (result["num"], "Number of pixels used in statistics below")
        hdr["STATMIN"] = (result["min"], "Minimum value of the pixels")
        hdr["STATMAX"] = (result["max"], "Maximum value of the pixels")
        hdr["STATAVG"] = (result["avg"], "Average value of the pixels")
        hdr["STATMED"] = (result["med"], "Median value of the pixels")
        hdr["STATSTD"] = (
            result["std"],
            "Sample standard deviation value of the pixels",
        )
        hdr["STATZMIN"] = (result["zmin"], "zscale minimum value of the pixels")
        hdr["STATZMAX"] = (result["zmax"], "zscale maximum value of the pixels")
        if percentiles is not None:
            for i, p in enumerate(percentiles):
                hdr[f"PERCTS{i + 1:02d}"] = (p, "The percentile used in STATPCii")
                hdr[f"STATPC{i + 1:02d}"] = (
                    result["pct"][i],
                    "Percentile value at PERCTSii",
                )

        if statsecs is not None:
            for i, sec in enumerate(statsecs):
                hdr[f"STATSEC{i + 1:01d}"] = (sec, "Sections used for statistics")

        if extrema_counts is not None:
            n_lo, n_hi = extrema_counts
            if max(n_lo, n_hi) > 99:
                logger.warning("num_extrema > 99 may not work properly in header.")
            for i in range(n_lo):
                hdr[f"STATLO{i + 1:02d}"] = (
                    result["ext_lo"][i],
                    f"Lower extreme values (num_extrema={extrema_counts})",
                )
            for i in range(n_hi):
                hdr[f"STATHI{i + 1:02d}"] = (
                    result["ext_hi"][i],
                    f"Upper extreme values (num_extrema={extrema_counts})",
                )
        return result, hdr
    return result
