from collections.abc import Callable

import astroapers as aap
import numpy as np
import reducers.lowlevel as rdl
from astropy.nddata import CCDData
from astropy.table import Table

from astroimred._core.numeric import _as_reducer_1d_values, sigma_clipper

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
    """Estimate sky with crude presets.

    Parameters
    ----------
    ccd: `~astropy.nddata.CCDData`, astropy HDU-like, ndarray-like
        The image data to extract sky at given annulus.

    pos: tuple of float
        The (x, y) position of the center of the annulus.

    r_in, r_out: float, optional
        The inner and outer radius of the annulus in pixel.
        Default is ``10``, ``20``.

    mask: None or array_like, optional
        A boolean mask with the same shape as `ccd`. The pixels with True
        values will be masked.

    kwargs : dict, optional
        The keyword arguments for `sky_fit`.
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
    """Estimate the sky value from image and annulus.

    Parameters
    ----------
    ccd: `~astropy.nddata.CCDData`, HDU, ndarray
        The image data to extract sky at given annulus.

    annulus: annulus object, optional
        The annulus which will be used to estimate sky values.
        If `None` (default), the whole image will be used.

    method : {"sex", "IRAF", "MMM", "mean", "median"}, callable, optional
        The method to estimate sky value.

          * ``"mean"``/``"median"``: simple nanmean/nanmedian of the sky values.
          * ``"sex"``  == (med_factor, mean_factor) = (2.5, 1.5)
          * ``"IRAF"`` == (med_factor, mean_factor) = (3, 2)
          * ``"MMM"``  == (med_factor, mean_factor) = (3, 2)

        where ``msky = (med_factor * med) - (mean_factor * mean)``.
        For the ``"sex"`` method, if (mean - median)/std > 0.3, median is used
        instead of the above formula, which mimics the SExtractor's way.
        If a callable is given, it should have the signature
        ``func(skyarr, ssky) -> msky`` where `skyarr` is the 1-d array of sky
        values and `ssky` is the sample standard deviation after using
        `sky_clipper`. Example: ``method = lambda x, s: np.median(x)``.
        The default is ``"sex"``.

    sky_clipper : callable, `None`, optional
        The function to be used to clip the sky values before estimating the
        sky value. The function should have the signature
        ``func(skyarr, **kwargs) -> clipped_skyarr`` where `skyarr` is the 1-d
        array of sky values. The returned array must be a plain ndarray with
        no NaN values — clipped elements should be removed, not replaced with
        NaN — because `_sky_fit` passes the result to `reducers` statistics
        kernels. `~astroimred._core.numeric.sigma_clipper` (the default)
        removes clipped values and uses `imcombiners` rejection kernels for
        compatible 1-D calls.
        If `None`, no clipping will be applied
        (i.e., ``lambda x: x[~np.isnan(x)]`` is used).
        The default is `astroimred._core.numeric.sigma_clipper`.

    std_ddof : int, optional
        The "delta-degrees of freedom" for sky standard deviation calculation.

    to_table : bool, optional
        If `True`, the output will be a `~astropy.table.Table` object. Otherwise,
        return a `dict`.

    return_skyarr : bool, optional
        If `True`, the 1-d array (or list of such if multiple annuli) of sky
        values will be returned.

    kwargs : dict, optional
        The keyword arguments for `sky_clipper`.

    Returns
    -------
    skytable: `~astropy.table.Table` or dict
        The table or dict of the followings.

        msky : float
            The estimated sky value within the all_sky data, after sigma clipping.

        ssky : float
            The sample standard deviation of sky value within the all_sky data,
            after sigma clipping.

        nsky : int
            The number of pixels which were used for sky estimation after the
            sigma clipping.

        nrej : int
            The number of pixels which are rejected after sigma clipping.

    skys : ndarray or list of ndarray
        The 1-d array (or list of such if multiple annuli) of sky values.

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
        if method in {"sex", "iraf", "mmm"}:
            med = rdl.median_valid(sky_clipped, copy=False)
        if method == "sex":
            use_median = std > 0.0 and (mean - med) / std > 0.3
            msky = med if use_median else (2.5 * med) - (1.5 * mean)
        elif method == "median":
            msky = rdl.median_valid(sky_clipped, copy=False)
        elif method == "mean":
            msky = mean
        elif method == "iraf":
            msky = mean if (mean < med) else 3 * med - 2 * mean
        elif method == "mmm":
            msky = 3 * med - 2 * mean
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
    """Use the MMM-DAO algorithm to estimate the sky value.

    Notes
    -----
    Please use sky_fit. This function is less optimized for calculation.
    Maybe updated in the future.

    The MMM-DAO algorithm is a robust estimator of the sky value. This
    algorithm is based on the MMM algorithm, which is a modification of the
    DAOFIND algorithm in DAOPHOT. This algorithm is described in Section 2.2
    of `Stetson (1987) <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_.

    Parameters
    ----------
    sky : array-like
        The sky values. This is usually the difference between the image and
        the object mask.

    highbad : bool, optional
        If `True`, the sky values are assumed to be high values, and the
        algorithm will search for low values. If `False`, the sky values are
        assumed to be low values, and the algorithm will search for high
        values. If `None`, the algorithm will search for both high and low
        values. The default is None.

    min_nsky : float or int, optional
        The minimum number of sky values to use. The default is 20. (`minsky` in DAOPHOT MMM)

    maxiter : int, optional
        The maximum number of iterations. The default is 50.

    readnoise : float, optional
        The read noise of the image. The default is 0.

    Returns
    -------
    sky : float
        The estimated sky value.

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
