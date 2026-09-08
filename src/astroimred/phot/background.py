from collections.abc import Callable

import astroapers as aap
import numpy as np
import reducers.lowlevel as rdl
from astropy.nddata import CCDData
from astropy.table import Table

from astroimred._core.numeric import _as_reducer_1d_values, sigma_clipper
from astroimred._core.statistics import _mode_from_stats

__all__ = ["annul2values", "quick_sky_circ", "sky_fit", "mmm_dao"]


def _data_and_mask(
    ccd: CCDData | np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return image data and the combined CCD/external mask."""
    if isinstance(ccd, CCDData):
        arr = np.asarray(ccd.data)
        ccd_mask = ccd.mask
        base_mask = None if ccd_mask is None else np.asarray(ccd_mask, dtype=bool)
    else:
        arr = np.asarray(ccd)
        base_mask = None

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        base_mask = mask if base_mask is None else (base_mask | mask)
    return arr, base_mask


def annul2values(
    ccd: CCDData | np.ndarray,
    annulus,
    mask: np.ndarray | None = None,
    *,
    flat: bool = False,
) -> list[np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Return center-selected annulus values using the current astroapers API.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Image data to sample.
    annulus : astroapers aperture-like
        Object exposing ``sampled_values(data, mask=..., flat=...)``.
    mask : `~numpy.ndarray`, optional
        Additional boolean mask where `True` pixels are excluded.
    flat : bool, optional
        If `True`, return ``(flat_values, offsets)`` as provided by
        `astroapers`.
    """
    arr, base_mask = _data_and_mask(ccd, mask)
    return annulus.sampled_values(arr, mask=base_mask, flat=flat)


def quick_sky_circ(
    ccd: CCDData | np.ndarray,
    pos: tuple[float, float],
    r_in: float = 10,
    r_out: float = 20,
    mask: np.ndarray | None = None,
    **kwargs,
) -> Table:
    """Estimate sky in a circular annulus using quick presets.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or ndarray-like
        Image data from which to extract annular sky values.

    pos : tuple of float
        Annulus center in ``(x, y)`` order.

    r_in, r_out : float, optional
        Inner and outer annulus radii in pixels. Defaults are `10` and `20`.

    mask : array-like or None, optional
        Boolean mask with the same shape as `ccd`; `True` pixels are excluded.

    **kwargs :
        Keyword arguments forwarded to `sky_fit`.
    """
    annulus = aap.CircAn(pos, r_in=r_in, r_out=r_out)
    return sky_fit(ccd, annulus, mask=mask, **kwargs)


def sky_fit(
    ccd: CCDData | np.ndarray,
    annulus=None,
    mask: np.ndarray | None = None,
    method: str | Callable[[np.ndarray, float], float] = "sex",
    sky_clipper: Callable[..., np.ndarray] | None = sigma_clipper,
    std_ddof: int = 1,
    to_table: bool = True,
    return_skyarr: bool = False,
    **kwargs,
) -> Table | tuple[Table, np.ndarray]:
    """Estimate sky values from an image or one or more annuli.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`, HDU, or ndarray
        Image data from which to extract sky values.

    annulus : annulus object, optional
        Annulus used for the estimate. If `None` (default), use the whole
        image.

    method : {"sex", "IRAF", "MMM", "mean", "median"}, callable, optional
        Sky estimator. Names are case-insensitive:

        * ``"mean"``/``"median"``: simple nanmean/nanmedian.
        * ``"sex"``: ``(med_factor, mean_factor) = (2.5, 1.5)``.
        * ``"IRAF"``: ``(med_factor, mean_factor) = (3, 2)``, or the mean
          when the mean is below the median.
        * ``"MMM"``: ``(med_factor, mean_factor) = (3, 2)``.

        The three compound estimators use
        ``msky = med_factor * med - mean_factor * mean``. ``"MMM"`` uses the
        same formula as visualization's `statistic="mode"`; it is not the
        iterative `mmm_dao` algorithm. For ``"sex"``, use the median instead
        when ``(mean - median) / std > 0.3``, matching SExtractor.

        A callable must have signature ``func(skyarr, ssky) -> msky``. It
        receives the 1-D clipped values and their sample standard deviation.
        Example: ``method = lambda x, s: np.median(x)``. Default is ``"sex"``.

    sky_clipper : callable, `None`, optional
        Function used before estimating the sky. It must have signature
        ``func(skyarr, **kwargs) -> clipped_skyarr`` and return a plain ndarray
        without NaNs: remove clipped values rather than replacing them with
        NaN. This is required by the `reducers` kernels used by `_sky_fit`.
        The default `~astroimred._core.numeric.sigma_clipper` removes values
        and uses `imcombiners` rejection kernels for compatible 1-D calls.
        If `None`, apply no clipping beyond removing NaNs
        (``lambda x: x[~np.isnan(x)]``).

    std_ddof : int, optional
        Delta degrees of freedom for the sky standard deviation.

    to_table : bool, optional
        Return a `~astropy.table.Table` when `True`; otherwise return a list of
        dictionaries. Default is `True`.

    return_skyarr : bool, optional
        Also return the 1-D sky array, or a list of arrays for multiple annuli.
        Default is `False`.

    **kwargs :
        Keyword arguments forwarded to `sky_clipper`.

    Returns
    -------
    skytable : `~astropy.table.Table` or list of dict
        One row or dictionary per annulus with the following fields:

        msky : float
            Estimated sky value after clipping.

        ssky : float
            Sample standard deviation after clipping.

        nsky : int
            Number of pixels used after clipping.

        nrej : int
            Number of pixels rejected by clipping.

    skys : ndarray or list of ndarray
        Returned only when `return_skyarr=True`: the 1-D sky values, or one
        array per annulus.

    """
    skydicts = []
    if annulus is None:
        try:  # CCDData or HDU
            arr = np.asarray(ccd.data)
            ccd_mask = getattr(ccd, "mask", None)
        except AttributeError:  # ndarray
            arr = np.asarray(ccd)
            ccd_mask = None

        base_mask = None
        if ccd_mask is not None:
            base_mask = np.asarray(ccd_mask, dtype=bool)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            base_mask = mask if base_mask is None else (base_mask | mask)

        skys = [arr[~base_mask].ravel() if base_mask is not None else arr.ravel()]
    else:
        flat_sky, offsets = annul2values(ccd, annulus, mask=mask, flat=True)
        sky_slices = (
            flat_sky[start:stop]
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        )
        skys = list(sky_slices) if return_skyarr else sky_slices

    if sky_clipper is None:

        def sky_clipper(x, **kwargs):
            return x[~np.isnan(x)]

    for _, sky in enumerate(skys):
        skydict = {}
        msky, std, nsky, nrej = _sky_fit(
            sky,
            method=method,
            sky_clipper=sky_clipper,
            std_ddof=std_ddof,
            **kwargs,
        )
        skydict["msky"] = msky
        skydict["ssky"] = std
        skydict["nsky"] = nsky
        skydict["nrej"] = nrej
        skydicts.append(skydict)

    if to_table:
        if return_skyarr:
            return Table(skydicts), skys
        return Table(skydicts)

    return (skydicts, skys) if return_skyarr else skydicts


def _sky_fit(
    sky,
    method="sex",
    sky_clipper=sigma_clipper,
    std_ddof=1,
    **kwargs,
):
    sky_clipped = _clip_sky_values(sky, sky_clipper, kwargs)
    if sky_clipped.size == 0:
        return np.nan, np.nan, 0, sky.size
    sky_clipped = _as_reducer_1d_values(sky_clipped)

    if isinstance(method, str):
        method = method.lower()
        std, mean = rdl.std_mean_valid(
            sky_clipped,
            ddof=std_ddof,
            copy=False,
        )
        if method == "median":
            msky = rdl.median_valid(sky_clipped, copy=False)
        elif method == "mean":
            msky = mean
        elif method in {"sex", "iraf", "mmm"}:
            med = rdl.median_valid(sky_clipped, copy=False)
            msky = _mode_from_stats(mean, med, std, method=method)
        else:
            raise ValueError(f"{method=} not understood")

    else:
        std = rdl.std_valid(sky_clipped, ddof=std_ddof, copy=False)
        msky = method(sky_clipped, std)

    nsky = sky_clipped.size
    nrej = sky.size - nsky

    return msky, std, nsky, nrej


def _clip_sky_values(
    sky: np.ndarray,
    sky_clipper: Callable[..., np.ndarray] | None,
    kwargs: dict,
) -> np.ndarray:
    """Return clipped sky values."""
    sky = np.asarray(sky).reshape(-1)
    if sky_clipper is None:
        return sky[~np.isnan(sky)]
    return sky_clipper(sky, **kwargs)


def mmm_dao(
    sky: np.ndarray,
    highbad: bool | None = None,
    min_nsky: int = 20,
    maxiter: int = 50,
    readnoise: float = 0,
) -> tuple[float, float, float]:
    """Estimate a sky value with the iterative MMM-DAO algorithm.

    Notes
    -----
    Prefer `sky_fit` for routine use. This implementation is slower and may
    change in the future.

    MMM-DAO is a robust sky estimator based on the MMM modification of DAOFIND
    in DAOPHOT. See Section 2.2 of
    `Stetson (1987) <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_.

    Parameters
    ----------
    sky : array-like
        Sky values, usually the image after object masking or subtraction.

    highbad : float or None, optional
        Optional value that limits the upper side of the initial symmetric
        interval. The current implementation applies it as
        ``min(cut1, highbad - skymid)``. `None` leaves that interval unchanged.
        The annotation still accepts `bool` for compatibility, although this
        branch uses the value arithmetically. Default is `None`.

    min_nsky : float or int, optional
        Minimum number of sky values. Default is `20` (`minsky` in DAOPHOT MMM).

    maxiter : int, optional
        Maximum number of iterations. Default is `50`.

    readnoise : float, optional
        Image read noise. Default is `0`.

    Returns
    -------
    sky : float
        Estimated sky value.

    """
    sky = np.array(sky)
    if (nsky := sky.size) < min_nsky:
        sigma = -1.0
        raise ValueError(f"Input vector must contain at least {min_nsky = } elements.")

    # do the MMM sky estimation similar to DAOPHOT (MMM.pro of https://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/mmm.pro)
    sky = np.sort(sky)
    skymid = 0.5 * sky[(nsky - 1) // 2] + 0.5 * sky[nsky // 2]
    cut1 = min([skymid - sky[0], sky[-1] - skymid])
    if highbad is not None:
        cut1 = min(cut1, highbad - skymid)
    cut2 = skymid + cut1
    cut1 = skymid - cut1
    goodmask = (cut1 <= sky) & (sky <= cut2)
    if np.count_nonzero(goodmask) == 0:
        sigma = -1.0
        raise ValueError(f"No sky values survived: {cut1:.4f}<=sky<={cut2:.4f}")

    skydelta = sky[goodmask] - skymid
    deltasum = np.sum(skydelta)
    deltasumsq = np.sum(skydelta**2)
    _goodidx = np.where(goodmask)[0]
    max_idx = _goodidx[-1]
    min_idx = _goodidx[0] - 1

    # compute mean and standard deviation
    skymed = 0.5 * (
        sky[(min_idx + max_idx + 1) // 2] + sky[(min_idx + max_idx) // 2 + 1]
    )

    skymn = float(deltasum / (max_idx - min_idx))
    sigma = np.sqrt(deltasumsq / (max_idx - min_idx) - skymn**2)
    skymn = skymn + skymid

    skymod = 3 * skymed - 2 * skymn if skymed < skymn else skymn

    # Rejection and recomputation loop
    niter = 0
    clamp = 1
    old = 0

    # Python version of the IDL loop (START_LOOP)
    while True:
        niter += 1
        if niter > maxiter:
            sigma = -1.0
            raise ValueError(f"Too many ({maxiter}) iterations, unable to compute sky.")

        if max_idx - min_idx < min_nsky:
            sigma = -1.0
            # raise ValueError(f"Too few ({max_idx - min_idx}) valid sky elements.")
            # It seems robust to just return the current estimate
            break

        # Compute Chauvenet rejection criterion
        r = np.log10(float(max_idx - min_idx))
        r = max(2.0, (-0.1042 * r + 1.1695) * r + 0.8895)

        # Compute rejection limits
        cut = r * sigma + 0.5 * np.abs(skymn - skymod)
        if issubclass(sky.dtype.type, np.integer):
            cut = max(cut, 1.5)

        cut1 = skymod - cut
        cut2 = skymod + cut

        # Recompute mean and sigma
        redo = False
        newmin = min_idx

        # Adjust min_idx (minimm)
        tst_min = sky[newmin + 1] >= cut1
        done = (newmin == -1) and tst_min

        if not done:
            done = (sky[max(newmin, 0)] < cut1) and tst_min

        if not done:
            istep = 1 - 2 * int(tst_min)
            while True:
                newmin += istep
                done = (newmin == -1) or (newmin == (nsky - 1))
                if not done:
                    done = (sky[newmin] <= cut1) and (sky[newmin + 1] >= cut1)
                if done:
                    break

            if tst_min:
                delta = sky[newmin + 1 : min_idx + 1] - skymid
            else:
                delta = sky[min_idx + 1 : newmin + 1] - skymid

            deltasum = deltasum - istep * np.sum(delta)
            deltasumsq = deltasumsq - istep * np.sum(delta**2)
            redo = True
            min_idx = newmin

        # Adjust max_idx (maximm)
        newmax = max_idx
        tst_max = sky[max_idx] <= cut2
        done = (max_idx == nsky - 1) and tst_max
        if not done:
            done = tst_max and (sky[min(max_idx + 1, nsky - 1)] > cut2)

        if not done:
            istep = -1 + 2 * int(tst_max)
            while True:
                newmax += istep
                done = (newmax == nsky - 1) or (newmax == -1)
                if not done:
                    done = (sky[newmax] <= cut2) and (sky[newmax + 1] >= cut2)
                if done:
                    break

            if tst_max:
                delta = sky[max_idx + 1 : newmax + 1] - skymid
            else:
                delta = sky[newmax + 1 : max_idx + 1] - skymid

            deltasum = deltasum + istep * np.sum(delta)
            deltasumsq = deltasumsq + istep * np.sum(delta**2)
            redo = True
            max_idx = newmax

        # Compute mean and sigma
        nsky_curr = max_idx - min_idx
        if nsky_curr < min_nsky:
            sigma = -1.0
            # raise ValueError("Outlier rejection left too few sky elements.")
            break

        skymn = float(deltasum / nsky_curr)
        sigma = float(np.sqrt(max(0, deltasumsq / nsky_curr - skymn**2)))
        skymn = skymn + skymid

        # Determine robust median
        center = (min_idx + 1 + max_idx) / 2.0
        side = round(0.2 * (max_idx - min_idx)) / 2.0 + 0.25
        J = int(round(center - side))
        K = int(round(center + side))

        if readnoise > 0:
            L = int(round(center - 0.25))
            M = int(round(center + 0.25))
            R = 0.25 * readnoise
            while (
                (J > 0)
                and (nsky - 1 > K)
                and ((sky[L] - sky[J] < R) or (sky[K] - sky[M] < R))
            ):
                J -= 1
                K += 1

        skymed = np.sum(sky[J : K + 1]) / (K - J + 1)

        dmod = 3 * skymed - 2 * skymn - skymod if skymed < skymn else skymn - skymod

        if dmod * old < 0:
            clamp = 0.5 * clamp

        skymod = skymod + clamp * dmod
        old = dmod

        if not redo:
            break

    return skymod
