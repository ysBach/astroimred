from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import QTable

from ..logging import logger
from .background import sky_fit

__all__ = ["apphot_annulus"]


@dataclass(frozen=True)
class PhotometryResult:
    """Aperture photometry arrays."""

    positions: np.ndarray
    apsum: np.ndarray
    apsum_err: np.ndarray | None
    apsum_npix: np.ndarray
    nbadpix: np.ndarray


def _normalize_apertures(apertures: Any) -> list[Any]:
    """Return one or more astroapers aperture objects."""
    if isinstance(apertures, np.ndarray):
        return list(apertures.ravel())
    if isinstance(apertures, (list, tuple)):
        return list(np.asarray(apertures, dtype=object).ravel())
    return [apertures]


def _normalize_positions(aperture: Any) -> np.ndarray:
    """Return aperture positions as an ``(N, 2)`` float array."""
    positions = np.asarray(aperture.positions, dtype=np.float64)
    return positions.reshape(1, 2) if positions.ndim == 1 else positions


def _aperture_apsum(
    aperture: Any,
    data: np.ndarray,
    *,
    mask: np.ndarray | None,
    method: str,
    return_npix: bool = True,
):
    if method == "exact":
        return aperture.apsum_exact(data, mask=mask, return_npix=return_npix)
    if method == "center":
        return aperture.apsum_center(data, mask=mask, return_npix=return_npix)
    raise ValueError("method must be 'exact' or 'center'")


def _aperture_npix(
    aperture: Any,
    shape: tuple[int, int],
    *,
    method: str,
    mask: np.ndarray | None = None,
):
    if method == "exact":
        return aperture.npix_exact(shape, mask=mask)
    if method == "center":
        return aperture.npix_center(shape, mask=mask)
    raise ValueError("method must be 'exact' or 'center'")


def photometer(
    data: np.ndarray,
    apertures: Any,
    *,
    error: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    method: str = "exact",
) -> PhotometryResult:
    """Measure aperture sums, errors, used aperture support, and masked support."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array.")
    err = None if error is None else np.asarray(error)
    if err is not None and err.shape != arr.shape:
        raise ValueError("error must have the same shape as data.")
    bad = None if mask is None else np.asarray(mask, dtype=bool)
    if bad is not None and bad.shape != arr.shape:
        raise ValueError("mask must have the same shape as data.")

    positions: list[np.ndarray] = []
    sums: list[float] = []
    errs: list[float] = []
    apsum_npixs: list[float] = []
    nbadpix: list[float] = []

    for aperture in _normalize_apertures(apertures):
        ap_positions = _normalize_positions(aperture)
        ap_sums, ap_npixs = _aperture_apsum(
            aperture,
            arr,
            mask=bad,
            method=method,
            return_npix=True,
        )
        ap_sums = np.asarray(ap_sums, dtype=np.float64).reshape(-1)
        ap_npixs = np.asarray(ap_npixs, dtype=np.float64).reshape(-1)
        if len(ap_positions) != len(ap_sums):
            raise ValueError("aperture positions and sums have inconsistent lengths.")

        positions.extend(ap_positions)
        sums.extend(ap_sums)
        apsum_npixs.extend(ap_npixs)
        if bad is None:
            nbadpix.extend(np.zeros_like(ap_npixs))
        else:
            total_npix = _aperture_npix(aperture, arr.shape, method=method)
            total_npix = np.asarray(total_npix, dtype=np.float64).reshape(-1)
            nbadpix.extend(_nonnegative_nbadpix(total_npix - ap_npixs))
        if err is not None:
            err_sums = _aperture_apsum(
                aperture,
                np.square(err),
                mask=bad,
                method=method,
                return_npix=False,
            )
            errs.extend(np.sqrt(np.asarray(err_sums, dtype=np.float64).reshape(-1)))

    return PhotometryResult(
        positions=np.asarray(positions, dtype=np.float64),
        apsum=np.asarray(sums, dtype=np.float64),
        apsum_err=None if err is None else np.asarray(errs, dtype=np.float64),
        apsum_npix=np.asarray(apsum_npixs, dtype=np.float64),
        nbadpix=np.asarray(nbadpix, dtype=np.float64),
    )


def _nonnegative_nbadpix(values: np.ndarray) -> np.ndarray:
    tiny = 1.0e-12
    return np.where((values < 0.0) & (np.abs(values) <= tiny), 0.0, values)


# TODO: Put centroiding into this apphot_annulus ?
# TODO: use variance instead of error
# TODO: one_aperture_per_row : bool, optional
def apphot_annulus(
    ccd: CCDData,
    aperture,
    annulus=None,
    gain: str | float = "GAIN",
    rdnoise: str | float = "RDNOISE",
    t_exposure: float | None = None,
    exposure_key: str = "EXPTIME",
    error: np.ndarray | u.Quantity | None = None,
    mask: np.ndarray | None = None,
    sky_keys: dict | None = None,
    sky_min: float | None = None,
    npix_mask_ap: int = 2,
    pandas: bool = True,
    **kwargs,
) -> pd.DataFrame | QTable:
    """Do aperture photometry using annulus.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The data to be photometried. Preferably in ADU.

    aperture, annulus : astroapers aperture or list of such, optional
        The aperture and annulus to be used for aperture photometry.

        .. note::
            For a multi-position aperture, use, e.g.,
            ``aap.CircAp(positions, r=10)``. For multiple radii, use, e.g.,
            ``[aap.CircAp(positions, r=r) for r in radii]``.

    gain : str, float, optional
        The gain of the CCD in electrons per ADU. If str, gain will be
        found from the header by tke key, and if not found, defaults to 1. If
        float, it should be in electrons per ADU. Used only if `error` is
        `None`.
        Default is ``'GAIN'``.

    rdnoise : str, float, optional
        The readout noise of the CCD in electrons. If str, readout noise will
        be found from the header by tke key, and if not found, defaults to ``0``.
        If float, it should be in electrons. Used only if `error` is `None`.
        Default is ``'RDNOISE'``.

    exposure_key : str, optional
        The key for exposure time. Together with `t_exposure_unit`, the
        function will normalize the signal to exposure time. If `t_exposure`
        is not None, this will be ignored.

    error : array-like or `~astropy.units.Quantity`, optional
        The pixel-wise error map to be propagated to magnitude error.

    sky_keys : dict, optional
        args/kwargs of `sky_fit`. If `None`(default), 3-sigma 5-iters clipping
        with ``ddof=1`` is performed, and then the modal sky value is estimated
        by SExtractor estimator; see `~astroimred.phot.background.sky_fit`.

    sky_min : float, optional
        The minimum value of the sky to be used for sky subtraction.

    npix_mask_ap : int, optional
        If the weighted masked aperture support is greater than `npix_mask_ap`,
        the column ``"bad"`` will be marked as ``1``.

        .. note::
            Currently it is not checked for annulus (works only for aperture)

    pandas : bool, optional
        Whether to convert to `~pandas.DataFrame`.

    **kwargs :
        Reserved for backward compatibility. ``method`` may be supplied and
        defaults to ``"exact"``.

    Returns
    -------
    phot_f: `~astropy.table.Table`
        The photometry result.

    bad code
      * 1 (2^0) : weighted masked support ``> npix_mask_ap`` within aperture.

    Notes
    -----
    If `error` is given, the error is propagated to magnitude error by
    quadratically summing the error. The final source variance is `error**2`
    plus ``apsum_npix*sky_stddev**2``. ``apsum_npix`` is the weighted in-image,
    unmasked aperture support used for the aperture sum.

    If `error` is not given, ``error=sqrt(data/gain + (rdnoise/gain)**2)`` is
    used, assuming dark=0 (also, digitization error is ignored. cf. Merline &
    Howell (1995) ExpA).
    """

    def _propagate_ccdmask(ccd, additional_mask=None):
        """Propagate the CCDData's mask and additional mask.

        Parameters
        ----------
        ccd : `~astropy.nddata.CCDData`, ndarray
            The ccd to extract mask. If ndarray, it will only return a copy of
            `additional_mask`.

        additional_mask : mask-like, None
            The mask to be propagated.

        Notes
        -----
        The original ``ccd.mask`` is not modified. To do so,
        >>> ccd.mask = propagate_ccdmask(ccd, additional_mask=mask2)
        """
        from copy import deepcopy

        if additional_mask is None:
            try:
                mask = ccd.mask.copy()
            except AttributeError:  # i.e., if ccd.mask is None
                mask = None
        else:
            if ccd.mask is None:
                mask = deepcopy(additional_mask)
            else:
                mask = ccd.mask | additional_mask
            # except (TypeError, AttributeError):  # i.e., if ccd.mask is None:
        return mask

    if isinstance(ccd, CCDData):
        _ccd = ccd.copy()
        _arr = _ccd.data
        _mask = _propagate_ccdmask(_ccd, additional_mask=mask)
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                logger.warning(
                    "The exposure time info not given and not found from the header "
                    f"({exposure_key}). Setting it to 1 sec."
                )
    else:  # ndarray
        _ccd = CCDData(data=ccd, unit="adu")
        _arr = _ccd.data
        _mask = mask
        if t_exposure is None:
            t_exposure = 1
            logger.warning("The exposure time info not given. Setting it to 1 sec.")

    method = kwargs.pop("method", "exact")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported apphot_annulus keyword(s): {unknown}")

    apertures = _normalize_apertures(aperture)

    flag_bad = True
    nbads = []
    bads = []
    if _mask is None:
        flag_bad = False

    if mask is not None:
        _mask = np.asarray(mask, dtype=bool) if _mask is None else (_mask | mask)

    if error is not None:
        logger.info(
            "Ignore any uncertainty extension in the original CCD, "
            + "and use provided uncertainty map."
        )
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            gn = float(_ccd.header.get(gain, 1) if isinstance(gain, str) else gain)
            rd = float(
                _ccd.header.get(rdnoise, 0) if isinstance(rdnoise, str) else rdnoise
            )
            logger.info(f"Making errormap from {gn} [e/ADU], {rd} [e]")
            err = np.sqrt(_arr / gn + (rd / gn) ** 2)

    measured = photometer(_arr, apertures, error=err, mask=_mask, method=method)
    if flag_bad:
        bads = (measured.nbadpix > npix_mask_ap).astype(int)
        nbads = measured.nbadpix
    else:
        bads = np.zeros_like(measured.nbadpix)
        nbads = np.zeros_like(measured.nbadpix)

    _phot = QTable()
    _phot["id"] = np.arange(1, len(measured.apsum) + 1)
    _phot["xcenter"] = measured.positions[:, 0]
    _phot["ycenter"] = measured.positions[:, 1]
    _phot["apsum"] = measured.apsum
    _phot["apsum_err"] = measured.apsum_err

    if annulus is not None:
        if sky_keys is None:
            skys = sky_fit(
                _arr,
                annulus,
                mask=_mask,
                method="sex",
                sigma=3,
                maxiters=5,
                std_ddof=1,
            )
        else:
            skys = sky_fit(_arr, annulus, mask=_mask, **sky_keys)
        for c in skys.colnames:
            values = np.asarray(skys[c])
            if values.size == 1 and len(_phot) != 1:
                values = np.repeat(values, len(_phot))
            _phot[c] = values
    else:
        _phot["msky"] = np.zeros(len(_phot), dtype=float)
        _phot["nsky"] = np.ones(len(_phot), dtype=int)
        _phot["nrej"] = np.zeros(len(_phot), dtype=int)
        _phot["ssky"] = np.zeros(len(_phot), dtype=float)

    phot = _phot

    if sky_min is not None:
        phot["msky"][phot["msky"] < sky_min] = sky_min

    phot["apsum_npix"] = measured.apsum_npix
    phot["nbadpix"] = np.atleast_1d(nbads)
    phot["srcsum"] = phot["apsum"] - phot["apsum_npix"] * phot["msky"]

    # see, e.g., http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?radprof.hlp
    # Poisson + RDnoise (Poisson includes signal + sky + dark) :
    var_errmap = phot["apsum_err"] ** 2
    # Sum of apsum_npix Gaussians (kind of random walk):
    var_skyrand = phot["apsum_npix"] * phot["ssky"] ** 2
    # The CLT error (although not correct, let me denote it as "systematic"
    # error for simplicity) of the mean estimation is ssky/sqrt(nsky), and that
    # is propagated for apsum_npix pixels, so we have
    # std = apsum_npix*ssky/sqrt(nsky), so
    # variance is:
    # var_sky = (phot["apsum_npix"] * phot["ssky"])**2 / phot["nsky"]
    # This error term is used in IRAF APPHOT, but this is wrong and thus
    # ignored here.
    phot["srcsum_err"] = np.sqrt(var_errmap + var_skyrand)
    snr = np.array(phot["srcsum"]) / np.array(phot["srcsum_err"])
    snr[snr < 0] = 0
    phot["mag"] = -2.5 * np.log10(phot["srcsum"] / t_exposure)
    phot["merr"] = 2.5 / np.log(10) * (1 / snr)
    phot["snr"] = snr
    phot["bad"] = np.atleast_1d(bads)

    if pandas:
        # This takes about 0.7 us for 1-row table. MBP 14" [2021, macOS 13.1,
        # M1Pro(6P+2E/G16c/N16c/32G)], 2023-06-19 22:46:08 (KST: GMT+09:00)
        # -YPB
        phot = phot.to_pandas()
        return phot.drop(["id"], axis=1)
    else:
        return phot


# TODO: make this...
def apphot_ellip_sep(
    ccd,
    x,
    y,
    a,
    a_in,
    a_out,
    bpa=1,
    theta=0,
    t_exposure=None,
    exposure_key="EXPTIME",
    error=None,
    mask=None,
    sky_keys=None,
    t_exposure_unit=u.s,
    pandas=False,
    **kwargs,
):
    """Similar to apphot_annulus but use sep to speedup.
    bpa : float
        b per a (ellipticity)
    """
    if sky_keys is None:
        sky_keys = {}

    _ccd = ccd.copy()

    if isinstance(_ccd, CCDData):
        _arr = _ccd.data
        _mask = _ccd.mask
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                logger.warning(
                    "The exposure time info not given and not found from the"
                    f"header({exposure_key}). Setting it to 1 sec."
                )
    else:  # ndarray
        _arr = np.array(_ccd)
        _mask = None
        if t_exposure is None:
            t_exposure = 1
            logger.warning("The exposure time info not given. Setting it to 1 sec.")

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    if error is not None:
        logger.info(
            "Ignore any uncertainty extension in the original CCD and use provided error."
        )
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            logger.warning(
                "Couldn't find Uncertainty extension in ccd. "
                + "Will not calculate errors."
            )
            err = np.zeros_like(_arr)

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    elif x.size > 1:
        pass

    a = np.atleast_1d(a)
    bpa = np.atleast_1d(bpa)
    theta = np.atleast_1d(theta)
    if (a.size > 1) + (bpa.size > 1) + (theta.size > 1) > 1:
        raise ValueError("Only one of a, bpa, theta can have size > 1.")

    num_apertures = max(a.size, bpa.size, theta.size)
    a = np.repeat(a, num_apertures)
    bpa = np.repeat(bpa, num_apertures)
    theta = np.repeat(theta, num_apertures)
    a * bpa

    a_in = np.atleast_1d(a_in)
    a_out = np.atleast_1d(a_out)
    if a_in.size > 1 or a_out.size > 1 or bpa.size > 1:
        raise ValueError("multiple annuli not allowed yet.")

    pass
