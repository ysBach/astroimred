from collections.abc import Callable

import numpy as np
import pandas as pd
from astro_ndslice import listify, slicefy
from astropy import units as u
from astropy.nddata import CCDData
from astropy.time import Time

from astroimred._core.types import FITSLike, StrPathLike
from astroimred.fitsmgmt.header import cmt2hdr, hdrval, update_process, update_tlm
from astroimred.fitsmgmt.io import _parse_image, load_ccd
from astroimred.imutil.ccdops import CCDData_astype, imslice
from astroimred.imutil.pixels import fixpix
from astroimred.logging import logger

from .crrej import LACOSMIC_CRREJ, crrej, medfilt_bpm

__all__ = [
    "crrej",
    "medfilt_bpm",
    "biascor",
    "darkcor",
    "flatcor",
    "frincor",
    "ccdred",
    "run_reduc_plan",
]


def _addfrm(ccd, name, path):
    ccd.header[f"{name.upper()[:4]}FRM"] = (path, f"Applied {name.upper()} frame")


def scancor(
    ccd: CCDData,
    overscan: np.ndarray | None = None,
    scansec: str | None = None,
    scanax: int | None = 0,
    fit_func: str = "legendre",
    fit_order: int = 1,
    fit_kw: dict | None = None,
) -> CCDData:
    """Reserve overscan fitting; currently raises `NotImplementedError`.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to correct.
    overscan : `~numpy.ndarray`, optional
        Overscan array, for example ``ccd.data[:, :overscan]``. Give exactly one
        of `overscan` and `scansec`. Default: ``None``.
    scansec : `str`, optional
        FITS-style section of the overscan region. Give exactly one of `overscan`
        and `scansec`. Default: ``None``.
    scanax : `int` or `None`, optional
        Axis along which overscan values are combined. Axis numbers use Python
        order. ``None`` selects the shortest overscan dimension; a square region
        uses axis 1. Default: ``0``.
    fit_func : `str`, optional
        Fitting function name. Default: ``"legendre"``.
    fit_order : `int`, optional
        Polynomial order. Default: ``1``.
    fit_kw : `dict`, optional
        Keywords for the fitting function. Default: ``None``.

        The planned preset is ``dict(sigma=(3, 3), maxiters=1, grow=0)``;
        no fitting is performed by the current implementation.

    Raises
    ------
    NotImplementedError
        Overscan correction is reserved for a future implementation.
    """
    raise NotImplementedError("scancor is not implemented yet.")
    # if fit_kw is None:
    #     fit_kw = dict(sigma=(3, 3), maxiters=1, grow=0)
    # pass


def biascor(
    ccd: CCDData,
    mbias: CCDData | np.ndarray | None = None,
    mbiaspath: StrPathLike | None = None,
    copy: bool = True,
    verbose: int = 1,
) -> CCDData:
    """Subtract a master bias frame.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to correct.
    mbias : `~astropy.nddata.CCDData` or `~numpy.ndarray`, optional
        Master bias frame. Default: ``None``.
    mbiaspath : path-like, optional
        Path to the master bias frame. Default: ``None``.
    copy : `bool`, optional
        Return a copy when ``True``; when ``False``, modify the input in place.
        The latter is slightly faster and uses less memory. Default: ``True``.
    verbose : `int`, optional
        Header logging level. Default: ``1``.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Bias-corrected CCD. When both master inputs are ``None``, this is an
        unchanged copy or the original object according to ``copy``.
    """
    if mbias is None and mbiaspath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    mbias, mbiasname, _ = _parse_image(mbias or mbiaspath, name=mbiaspath)
    # For BIAS, header information is not needed at all... I guess?
    nccd.data = nccd.data - mbias
    _addfrm(nccd, "BIAS", mbiasname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[biascor] Bias subtracted (BIASFRM = {mbiasname})",
    )
    update_process(nccd.header, "B")
    return nccd


def darkcor(
    ccd: CCDData,
    mdark: CCDData | np.ndarray | None = None,
    mdarkpath: StrPathLike | None = None,
    exptime_key: str = "EXPTIME",
    exptime_data: float | None = None,
    exptime_dark: float | None = None,
    dark_scale: bool = False,
    copy: bool = True,
    verbose: int = 1,
) -> CCDData:
    """Subtract a master dark frame, optionally scaled by exposure time.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to correct.
    mdark : `~astropy.nddata.CCDData` or `~numpy.ndarray`, optional
        Master dark frame. Default: ``None``.
    mdarkpath : path-like, optional
        Path to the master dark frame. Default: ``None``.
    exptime_key : `str`, optional
        Header keyword for exposure time. Used only when ``dark_scale=True``.
        Default: ``"EXPTIME"``.
    exptime_data, exptime_dark : numeric, optional
        Exposure times of the data and dark frame in the same unit. If either is
        ``None``, ``header.get(exptime_key, 1)`` supplies that value; explicit
        values override the header. Ignored when ``dark_scale=False``. Default:
        ``None``.
    dark_scale : `bool`, optional
        Multiply the dark frame by
        ``exptime_data / exptime_dark`` before subtraction. Default: ``False``.
    copy : `bool`, optional
        Return a copy when ``True``; when ``False``, modify the input in place.
        The latter is slightly faster and uses less memory. Default: ``True``.
    verbose : `int`, optional
        Header logging level. Default: ``1``.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Dark-corrected CCD. When both master inputs are ``None``, this is an
        unchanged copy or the original object according to ``copy``.
    """

    if mdark is None and mdarkpath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    use_ccddata = dark_scale and exptime_dark is None
    mdark, mdarkname, _ = _parse_image(
        mdark or mdarkpath, name=mdarkpath, force_ccddata=use_ccddata
    )

    if dark_scale:
        exptime_data = exptime_data or ccd.header.get(exptime_key, None)
        exptime_dark = exptime_dark or mdark.header.get(exptime_key, None)

        msg = "[darkcor] Dark scaled by exptime: "
        if exptime_data is None or exptime_dark is None:
            logger.warning(
                "exptime_data=%s, exptime_dark=%s. Fix scale=1.",
                exptime_data,
                exptime_dark,
            )
            scale = 1
            msg += "Fixed scale=1 (metadata missing)."
        else:
            scale = exptime_data / exptime_dark
            msg += f"({exptime_data:.3f}/{exptime_dark:.3f}) = {scale:.3f}"

        mdark = mdark.data * scale if use_ccddata else mdark * scale
        # ^ mdark is now ndarray regardless of use_ccddata
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=msg,
        )
    nccd.data = nccd.data - mdark
    _addfrm(nccd, "DARK", mdarkname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[darkcor] Dark subtracted (DARKFRM = {mdarkname})",
    )
    update_process(nccd.header, "D")
    return nccd


# add flat_norm_value
def flatcor(
    ccd: CCDData,
    mflat: CCDData | np.ndarray | None = None,
    mflatpath: StrPathLike | None = None,
    flat_mask: float | np.ndarray | None = 0,
    flat_fill: float = 1,
    copy: bool = True,
    flat_norm_value: float | None = 1,
    verbose: int = 1,
) -> CCDData:
    """Divide by a master flat frame.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to correct.
    mflat : `~astropy.nddata.CCDData` or `~numpy.ndarray`, optional
        Master flat frame. Default: ``None``.
    mflatpath : path-like, optional
        Path to the master flat frame. Default: ``None``.
    flat_mask : numeric, `~numpy.ndarray`, or `None`, optional
        Pixels replaced by `flat_fill`. For a numeric value, pixels with
        ``mflat < flat_mask`` are selected. Skipped when ``None``. Default: ``0``.
    flat_fill : numeric, optional
        Value used for selected flat pixels. Default: ``1``.
    copy : `bool`, optional
        Return a copy when ``True``; when ``False``, modify the input in place.
        The latter is slightly faster and uses less memory. Default: ``True``.
    flat_norm_value : numeric or `None`, optional
        If ``None``, divide the flat by its mean; otherwise divide it by this
        value before correction. Default: ``1``.
    verbose : `int`, optional
        Header logging level. Default: ``1``.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Flat-corrected CCD. When both master inputs are ``None``, this is an
        unchanged copy or the original object according to ``copy``.
    """
    if mflat is None and mflatpath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    mflat, mflatname, _ = _parse_image(
        mflat or mflatpath, name=mflatpath, force_ccddata=False
    )
    # For FLAT, header information is not needed at all... I guess?
    if flat_mask is not None:
        mflat = mflat.copy()
        if isinstance(flat_mask, np.ndarray):
            maskstr = "Flat pixels with `value < flat_mask (User-provided ndarray)`"
        else:
            maskstr = f"Flat pixels with `value < {flat_mask = }`"
            flat_mask = mflat < flat_mask
        mflat[flat_mask] = flat_fill
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=(f"[flatcor] {maskstr} are replaced by `{flat_fill = }`."),
        )

    if flat_norm_value is None:
        mflat /= np.mean(mflat)
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=("[flatcor] Flat normalized by its mean."),
        )
    elif float(flat_norm_value) != 1.0:
        mflat /= float(flat_norm_value)
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=(f"[flatcor] Flat divided by {flat_norm_value = }."),
        )

    nccd.data = nccd.data / mflat
    _addfrm(nccd, "FLAT", mflatname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[flatcor] Flat corrected (FLATFRM = {mflatname})",
    )
    update_process(nccd.header, "F")

    return nccd


def frincor(
    ccd: CCDData,
    mfrin: CCDData | np.ndarray | None,
    mfrinpath: StrPathLike | None = None,
    fringe_scale: int | float | np.ndarray | str | Callable | None = None,
    fringe_scale_region: np.ndarray | str | None = None,
    fringe_scale_kw: dict | None = None,
    exptime_key: str = "EXPTIME",
    exptime_data: float | None = None,
    exptime_frin: float | None = None,
    copy: bool = True,
    verbose: int = 1,
) -> CCDData:
    """Subtract a master fringe frame, optionally scaled to the image.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        CCD image to correct.
    mfrin : `~astropy.nddata.CCDData` or `~numpy.ndarray` or `None`
        Fringe frame, or ``None`` when `mfrinpath` supplies it. Default: ``None``.
    mfrinpath : path-like, optional
        Path to the fringe frame. Default: ``None``.
    fringe_scale : numeric, `~numpy.ndarray`, callable, `str`, or `None`, optional
        Scale applied before subtraction. A number or array is multiplied
        directly. A callable receives ``image[fringe_scale_region] /
        fringe[fringe_scale_region]`` and `fringe_scale_kw`. ``"exp"``,
        ``"exposure"``, and ``"exptime"`` use exposure times. ``None`` leaves
        the fringe unchanged. Default: ``None``.
    fringe_scale_region : `~numpy.ndarray` or `str`, optional
        Boolean mask or FITS-style section used to estimate a callable scale. An
        array is converted to boolean. Default: ``None`` (use all pixels).
    fringe_scale_kw : `dict`, optional
        Keywords passed to a callable `fringe_scale`. Default: ``None`` (treated
        as ``{}``).
    exptime_key : `str`, optional
        Header keyword for exposure time. Used only for exposure scaling.
        Default: ``"EXPTIME"``.
    exptime_data, exptime_frin : numeric or `None`, optional
        Exposure times of the data and fringe frame in the same unit. Missing
        values use ``header.get(exptime_key, 1)``. Used only for exposure scaling.
        Default: ``None``.
    copy : `bool`, optional
        Return a copy when ``True``; when ``False``, modify the input in place.
        The latter is slightly faster and uses less memory. Default: ``True``.
    verbose : `int`, optional
        Header logging level. Default: ``1``.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Fringe-corrected CCD. When both fringe inputs are ``None``, this is an
        unchanged copy or the original object according to ``copy``.
    """
    if mfrin is None and mfrinpath is None:
        return ccd.copy() if copy else ccd
    if fringe_scale_kw is None:
        fringe_scale_kw = {}

    def _str(_ccd, frm, sec=None, fun=None, scal=None):
        str1 = f"[frincor] Fringe subtracted (FRINFRM = {frm})"
        _ccd.header["FRINFRM"] = (frm, "Fringe frame")
        noscal = scal is None
        nosec = sec is None
        nofun = fun is None
        if noscal and nofun and nosec:
            return str1

        str2 = "[frincor] IMAGE - FRINSCAL*FRINFRM "
        elems = []
        if not noscal:  # scal is not None
            _ccd.header["FRINSCAL"] = (scal, "Scale FRINFUNC(FRINFRM[FRINSECT])")
            elems.append(f"`FRINSCAL = {scal}`")
        if not nofun:
            _ccd.header["FRINFUNC"] = (fun, "Function used to get FRINSCAL")
            elems.append(f"`FRINFUNC = {fun}`")
        if not nosec:
            _ccd.header["FRINSECT"] = (sec, "The region used to get FRINSCAL")
            elems.append(f"`FRINSECT = {sec}`")
        return [str1, str2, ",".join(elems)]

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd

    mfrin_input = mfrin if mfrin is not None else mfrinpath
    mfrin, mfrinname, _ = _parse_image(mfrin_input, name=mfrinpath, force_ccddata=True)
    #                                                                ^^^^^^^^^^^^^^^^^^
    # Converting an ndarray to CCDData (or vice versa) is very quick, so just
    # force CCDData for the sake of simplicity.

    if fringe_scale is None:
        nccd.data -= mfrin.data
        infostr = _str(nccd, mfrinname)
    elif isinstance(fringe_scale, (int, float)):
        nccd.data -= fringe_scale * mfrin.data
        infostr = _str(
            nccd, mfrinname, fun=type(fringe_scale).__name__, scal=fringe_scale
        )
    elif isinstance(fringe_scale, np.ndarray):
        nccd.data -= fringe_scale * mfrin.data
        infostr = _str(nccd, mfrinname, fun=f"{type(fringe_scale).__name__}")
    elif isinstance(fringe_scale, str):
        if fringe_scale.lower() in ["exp", "exposure", "exptime"]:
            exptime_data = exptime_data or nccd.header.get(exptime_key, 1)
            exptime_frin = exptime_frin or mfrin.header.get(exptime_key, 1)
            scale = exptime_data / exptime_frin
            nccd.data -= scale * mfrin.data
            infostr = _str(nccd, mfrinname, fun="EXPTIME", scal=scale)
        else:
            raise ValueError(
                f'`{fringe_scale=}` not in {{"exp", "exposure", "exptime"}}.'
            )
    else:  # Function
        if isinstance(fringe_scale_region, str):
            reg = slicefy(fringe_scale_region)
            sec = fringe_scale_region
        elif isinstance(fringe_scale_region, np.ndarray):
            reg = fringe_scale_region.astype(bool)
            sec = "User-provided mask"
        else:
            reg = None  # All
            sec = None
        scale = fringe_scale(ccd.data[reg] / mfrin.data[reg], **fringe_scale_kw)
        nccd.data -= scale * mfrin.data
        infostr = _str(
            nccd,
            mfrinname,
            fun=f"{fringe_scale.__name__} with {fringe_scale_kw}",
            sec=sec,
            scal=scale,
        )

    # FRINSCAL=FRINFUNC(FRINFRM[FRINSECT])
    _addfrm(nccd, "FRIN", mfrinname)
    cmt2hdr(nccd.header, "h", verbose=verbose, t_ref=_t, s=infostr)
    update_process(nccd.header, "R")

    return nccd


def illumcor(
    ccd: CCDData,
) -> CCDData:
    """Apply illumination correction.

    This placeholder is reserved for a future implementation.
    """
    raise NotImplementedError("illumcor is not implemented yet.")


# TODO: add overscan
# TODO: add normalization (e.g., `normalize` = {"mean", "median", "mode",
# "sum", "exptime", })
def ccdred(
    ccd: FITSLike,
    output: StrPathLike | None = None,
    extension: int | str | None = None,
    trimsec: str | None = None,
    mbiaspath: str | None = None,
    mdarkpath: str | None = None,
    mflatpath: str | None = None,
    mfrinpath: str | None = None,
    mbias: CCDData | None = None,
    mdark: CCDData | None = None,
    mflat: CCDData | None = None,
    mfrin: CCDData | None = None,
    fringe_flat_fielded: bool = True,
    fringe_scale: int | float | np.ndarray | str | Callable | None = None,
    fringe_scale_region: str | None = None,
    fringe_scale_kw: dict | None = None,
    gain: float | u.Unit = 1,
    gain_key: str = "GAIN",
    gain_unit: u.Unit = u.electron / u.adu,
    rdnoise: float | u.Unit = 0,
    rdnoise_key: str = "RDNOISE",
    rdnoise_unit: u.Unit = u.electron,
    exptime_key: str = "EXPTIME",
    exptime_frin: float | None = None,
    exptime_dark: float | None = None,
    exptime_data: float | None = None,
    dark_scale: bool = False,
    pixel_min: float | None = None,
    pixel_min_fill: float = 0,
    pixel_max: float | None = None,
    pixel_max_fill: float = 65535,
    flat_mask: float | int = 0,
    flat_fill: float = 1,
    flat_norm_value: float = 1,
    do_crrej: bool = False,
    crrej_kw: dict | None = None,
    propagate_crmask: bool = False,
    verbose_crrej: bool = False,
    verbose_bdf: int = 1,
    output_verify: str = "fix",
    overwrite: bool = True,
    dtype: str = "float32",
) -> CCDData:
    """Run the basic CCD calibration sequence.

    The operations run in this order when their master frame is supplied:
    trim, bias subtraction, dark subtraction, fringe subtraction (before or after
    flat correction according to `fringe_flat_fielded`), flat correction, and
    optional cosmic-ray rejection. Pixel limits and output dtype are applied last.

    Parameters
    ----------
    ccd : FITS-like
        CCDData, FITS HDU, array, path, or number accepted by ``_parse_image``.
    output : path-like or `None`, optional
        Save the reduced CCD to this path. Default: ``None``.
    extension : extension selector, optional
        FITS extension passed to ``_parse_image``. Default: ``None``.
    trimsec : `str` or `None`, optional
        FITS-style section applied to the science and supplied calibration frames.
        Default: ``None``.
    mbiaspath, mdarkpath, mflatpath, mfrinpath : path-like or `None`, optional
        Paths to master bias, dark, flat, and fringe frames. A corresponding
        ``m*`` object takes precedence when both are supplied; the path is still
        recorded in the output header. Defaults: ``None``.
    mbias, mdark, mflat, mfrin : `~astropy.nddata.CCDData` or `None`, optional
        In-memory master calibration frames. Defaults: ``None``.
    fringe_flat_fielded : `bool`, optional
        If ``True``, subtract the fringe after flat correction. If ``False``,
        subtract it before flat correction. Default: ``True``.
    fringe_scale : numeric, `~numpy.ndarray`, callable, `str`, or `None`, optional
        Fringe scaling passed to ``frincor``. Exposure modes are
        ``"exp"``, ``"exposure"``, and ``"exptime"``. Default: ``None``.
    fringe_scale_region : `~numpy.ndarray` or `str`, optional
        Mask or FITS-style section used by a callable fringe scale. Default:
        ``None``.
    fringe_scale_kw : `dict` or `None`, optional
        Keywords passed to a callable fringe scale. Default: ``None``.
    gain, rdnoise : numeric or `~astropy.units.Quantity`, optional
        Detector gain and read noise used only for cosmic-ray rejection. Values
        without units use `gain_unit` and `rdnoise_unit`. Defaults: `1` and `0`.
    gain_key, rdnoise_key : `str`, optional
        Header keywords used when `gain` or `rdnoise` is not supplied. Defaults:
        ``"GAIN"`` and ``"RDNOISE"``.
    gain_unit, rdnoise_unit : `~astropy.units.Unit`, optional
        Units for header or explicit gain and read-noise values. Defaults:
        ``u.electron / u.adu`` and ``u.electron``.
    exptime_key : `str`, optional
        Header keyword used for dark/fringe exposure scaling. Default:
        ``"EXPTIME"``.
    exptime_frin, exptime_dark, exptime_data : numeric or `None`, optional
        Explicit fringe, dark, and science exposure times. Missing values are
        read from `exptime_key` when the relevant scaling is enabled. Defaults:
        ``None``.
    dark_scale : `bool`, optional
        Scale the dark by ``exptime_data / exptime_dark`` before subtraction.
        Default: ``False``.
    pixel_min, pixel_max : numeric or `None`, optional
        Replace science values below/above the limit with `pixel_min_fill`/
        `pixel_max_fill`. Defaults: ``None``.
    pixel_min_fill, pixel_max_fill : numeric, optional
        Replacement values for the pixel limits. Defaults: ``0`` and ``65535``.
    flat_mask : numeric or `~numpy.ndarray`, optional
        Flat pixels selected by this threshold or mask are replaced with
        `flat_fill`. Default: ``0``.
    flat_fill : numeric, optional
        Replacement value for selected flat pixels. Default: ``1``.
    flat_norm_value : numeric or `None`, optional
        Divide the flat by this value before correction. ``None`` uses its mean.
        Default: ``1``.
    do_crrej : `bool`, optional
        Run ``crrej`` after bias, dark, and flat processing. Default: ``False``.
    crrej_kw : `dict` or `None`, optional
        Keyword arguments passed to ``crrej``. ``None`` uses the package's
        L.A. Cosmic defaults and logs a warning. Default: ``None``.

        Choose parameters for the data; defaults are suitable only for a quick
        check and differ across IRAF, IDL, and Python implementations. Inspect
        ``astroscrappy.detect_cosmics.__doc__`` for available options, or see
        https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb
    propagate_crmask : `bool`, optional
        Add the cosmic-ray mask to the CCD mask. Default: ``False``.
    verbose_crrej : `bool`, optional
        Enable cosmic-ray rejection logging. Default: ``False``.
    verbose_bdf : `int`, optional
        Verbosity for bias, dark, flat, fringe, and output logging. Default: ``1``.
    output_verify : `str`, optional
        FITS verification mode: ``"fix"``, ``"silentfix"``, ``"ignore"``,
        ``"warn"``, or ``"exception"``. Combine ``"fix"`` or ``"silentfix"``
        with ``"+ignore"``, ``"+warn"``, or ``"+exception"`` as needed, for
        example ``"fix+warn"``. Default: ``"fix"``. See
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify
    overwrite : `bool`, optional
        Overwrite `output` when it exists. Default: ``True``.
    dtype : dtype-like, optional
        Output data type. Default: ``"float32"``.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Reduced CCD. The input is copied before processing.

    Notes
    -----
    Existing uncertainty attributes on input and calibration frames are ignored.

    The old names `calc_err`, `unit`, `dark_exposure`, `data_exposure`,
    `exposure_key`, `exposure_unit`, `normalize_exposure`, `normalize_average`,
    `normalize_median`, `flat_min_value`, and `crrej_kwargs` are not parameters of
    this API. Use `exptime_dark`, `exptime_data`, and `exptime_key` for available
    exposure scaling, `flat_norm_value` for flat normalization, and `crrej_kw` for
    cosmic-ray options. Supply units on the input `CCDData`; no separate `unit`
    argument is accepted.
    """

    # This reduction process will ignore `uncertainty` attribute of all
    # input/master calibration frames. This is because (1) speed matters more
    # than such an error calculation for cases when this simple generalized
    # function is used (2) such uncertainties are anyway not accurate in most
    # cases.
    def _load_master(path, master):
        if path is None and master is None:
            return False, None, None

        if path is not None and master is None:
            master = load_ccd(
                path,
                ccddata=False,
                use_wcs=False,
                extension_uncertainty=None,
                extension_mask=None,
            )  # because it will be forced to CCDData

        do = True
        master, imname, _ = _parse_image(master, name=path, force_ccddata=True)
        return do, master, imname

    # ******************************************************************************** #
    # *                                  INITIAL SETTING                             * #
    # ******************************************************************************** #
    ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
    proc = ccd.copy()

    # == Set for BIAS ================================================================ #
    do_b, mbias, mbiaspath = _load_master(mbiaspath, mbias)  # mbias in ndarray
    do_d, mdark, mdarkpath = _load_master(mdarkpath, mdark)  # mdark in ndarray
    do_f, mflat, mflatpath = _load_master(mflatpath, mflat)  # mflat in ndarray
    do_r, mfrin, mfrinpath = _load_master(mfrinpath, mfrin)  # mfrin in ndarray

    # ******************************************************************************** #
    # *                                 RUN PREPROCESSING                            * #
    # ******************************************************************************** #
    # == Do TRIM ===================================================================== #
    if trimsec is not None:
        sect = {"trimsec": trimsec, "fill_value": None, "update_header": False}
        proc = imslice(proc, trimsec=trimsec, fill_value=None)  # update header
        mbias = imslice(mbias, **sect) if do_b else None
        mdark = imslice(mdark, **sect) if do_d else None
        mflat = imslice(mflat, **sect) if do_f else None
        mfrin = imslice(mfrin, **sect) if do_r else None

    prockw = {"copy": False, "verbose": verbose_bdf}
    # == Do BIAS ===================================================================== #
    if do_b:
        proc = biascor(proc, mbias=mbias, mbiaspath=mbiaspath, **prockw)

    # == Do DARK ===================================================================== #
    if do_d:
        proc = darkcor(
            proc,
            mdark=mdark,
            mdarkpath=mdarkpath,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_dark=exptime_dark,
            dark_scale=dark_scale,
            **prockw,
        )

    # == Do FRINGE **before** flat if not `fringe_flat_fielded` ====================== #
    if do_r and not fringe_flat_fielded:
        proc = frincor(
            proc,
            mfrin,
            mfrinpath=mfrinpath,
            fringe_scale=fringe_scale,
            fringe_scale_region=fringe_scale_region,
            fringe_scale_kw=fringe_scale_kw,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_frin=exptime_frin,
            **prockw,
        )

    # == Do FLAT ===================================================================== #
    if do_f:
        proc = flatcor(
            proc,
            mflat=mflat,
            mflatpath=mflatpath,
            flat_mask=flat_mask,
            flat_fill=flat_fill,
            flat_norm_value=flat_norm_value,
            **prockw,
        )

    # == Do FRINGE **after** flat if `fringe_flat_fielded` =========================== #
    if do_r and fringe_flat_fielded:
        proc = frincor(
            proc,
            mfrin,
            mfrinpath=mfrinpath,
            fringe_scale=fringe_scale,
            fringe_scale_region=fringe_scale_region,
            fringe_scale_kw=fringe_scale_kw,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_frin=exptime_frin,
            **prockw,
        )

    # == Do CRREJ ==================================================================== #
    if do_crrej:
        if crrej_kw is None:
            crrej_kw = LACOSMIC_CRREJ.copy()
            logger.warning("Using default CR-rejection parameters.")

        _proc = proc.header["PROCESS"]
        if (("B" in _proc) + ("D" in _proc) + ("F" in _proc)) < 2:
            logger.warning(
                "L.A. Cosmic should be run AFTER bias, dark, flat process. "
                "You have only done %s. "
                "See http://www.astro.yale.edu/dokkum/lacosmic/notes.html",
                proc.header["PROCESS"],
            )

        proc, _ = crrej(
            proc,
            propagate_crmask=propagate_crmask,
            update_header=True,
            gain=hdrval(gain, proc.header, gain_key, 1, unit=gain_unit),
            rdnoise=hdrval(rdnoise, proc.header, rdnoise_key, 0, unit=rdnoise_unit),
            verbose=verbose_crrej,
            **crrej_kw,
        )

    # ******************************************************************************** #
    # *                                  PREPARE OUTPUT                              * #
    # ******************************************************************************** #
    # To avoid ``pssl`` in cr rejection, subtract fringe AFTER the CRREJ.
    if pixel_min is not None:
        proc.data[proc.data < pixel_min] = pixel_min_fill
    if pixel_max is not None:
        proc.data[proc.data > pixel_max] = pixel_max_fill
    proc = CCDData_astype(proc, dtype=dtype)
    update_tlm(proc.header)

    if output is not None:
        if verbose_bdf:
            logger.info("Writing FITS to %s...", output)
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            logger.info("Saved.")
    return proc


def run_reduc_plan(
    plan: pd.DataFrame,
    output: StrPathLike | list[StrPathLike] | None = None,
    extension: int | str | None = None,
    col_file: str = "file",
    col_bias: str = "BIASFRM",
    col_dark: str = "DARKFRM",
    col_flat: str = "FLATFRM",
    col_mask: str = "MASKFILE",
    col_fringe: str = "FRINFRM",
    fixpix_kw: dict | None = None,
    do_crrej: bool = False,
    preload_cals: bool = False,
    return_ccd: bool = False,
    verbose: bool = False,
) -> list[CCDData] | None:
    """Run `ccdred` for each row in a reduction plan.

    Parameters
    ----------
    plan : `~pandas.DataFrame`
        Table containing input files and calibration-frame columns.
    output : path-like or sequence of path-like, optional
        Output path for each row. Required unless ``return_ccd=True``.
    extension : int, str, or tuple, optional
        FITS extension to load.
    col_file, col_bias, col_dark, col_flat, col_mask, col_fringe : str, optional
        The column names for bias, dark, flat, mask, and fringe frames in
        `plan`. Default values follow IRAF convention.
    fixpix_kw, do_crrej : optional
        Options for bad-pixel fixing and cosmic-ray rejection.
    preload_cals, return_ccd, verbose : bool, optional
        Whether to preload calibration frames, return reduced CCDs, and log
        progress.

    Returns
    -------
    list of `~astropy.nddata.CCDData` or None
        Reduced CCDs if ``return_ccd=True``; otherwise writes outputs and
        returns `None`.
    """

    def _get_frms(df, col):
        if col not in df:
            return {}
        ccds = {}
        for fpath in df[col].unique():
            if pd.isna(fpath):
                continue
            ccds[fpath] = load_ccd(
                fpath,
                use_wcs=False,
                extension_uncertainty=None,
                extension_mask=None,
            )
        return ccds

    def _get_path(row, col):
        value = row.get(col)
        return None if pd.isna(value) else value

    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a ~pandas.DataFrame.")

    if output is None and not return_ccd:
        raise ValueError(
            "No output file and return_ccd is False. "
            + "Nothing will be saved and nothing will be returned."
        )

    output = [None] * len(plan) if output is None else listify(output)

    if len(output) != len(plan):
        raise ValueError("output must have the same length as the plan.")
    if fixpix_kw is None:
        fixpix_kw = {"priority": None, "verbose": False}

    mbiass = _get_frms(plan, col_bias) if preload_cals else {}
    mdarks = _get_frms(plan, col_dark) if preload_cals else {}
    mflats = _get_frms(plan, col_flat) if preload_cals else {}
    mfrins = _get_frms(plan, col_fringe) if preload_cals else {}
    mmasks = _get_frms(plan, col_mask) if preload_cals else {}

    if return_ccd:
        ccds = []

    for (_, row), outpath in zip(plan.iterrows(), output, strict=False):
        mbiaspath = _get_path(row, col_bias)  # either path or None
        mdarkpath = _get_path(row, col_dark)  # either path or None
        mflatpath = _get_path(row, col_flat)  # either path or None
        mfrinpath = _get_path(row, col_fringe)  # either path or None
        maskpath = _get_path(row, col_mask)
        ccd = ccdred(
            load_ccd(row[col_file]),
            extension=extension,
            mbias=mbiass.get(mbiaspath),  # = None if not preload_cals or mbiaspath=None
            mdark=mdarks.get(mdarkpath),  # (same as above)
            mflat=mflats.get(mflatpath),  # (same as above)
            mfrin=mfrins.get(mfrinpath),  # (same as above)
            mbiaspath=mbiaspath,
            mdarkpath=mdarkpath,
            mflatpath=mflatpath,
            mfrinpath=mfrinpath,
            verbose_bdf=verbose,
        )
        # ^ Better to load as CCDData here rather than pass filepath to
        #   `ccdred`, to avoid parsing overhead in `_parse_image`.

        mask_ccd = (
            mmasks.get(maskpath)
            if preload_cals
            else (
                load_ccd(
                    maskpath,
                    use_wcs=False,
                    extension_uncertainty=None,
                    extension_mask=None,
                )
                if maskpath is not None
                else None
            )
        )
        ccd = fixpix(
            ccd,
            mask_ccd,
            maskpath=maskpath,
            **fixpix_kw,
        )

        if do_crrej:
            ccd, _ = crrej(
                ccd,
                mask=mask_ccd,
                gain=row.get("gain", LACOSMIC_CRREJ.get("gain")),
                rdnoise=row.get("rdnoise", LACOSMIC_CRREJ.get("rdnoise")),
                sigclip=row.get("sigclip", LACOSMIC_CRREJ.get("sigclip")),
                sigfrac=row.get("sigfrac", LACOSMIC_CRREJ.get("sigfrac")),
                objlim=row.get("objlim", LACOSMIC_CRREJ.get("objlim")),
                satlevel=row.get("satlevel", LACOSMIC_CRREJ.get("satlevel")),
                niter=row.get("niter", LACOSMIC_CRREJ.get("niter")),
                sepmed=row.get("sepmed", LACOSMIC_CRREJ.get("sepmed")),
                cleantype=row.get("cleantype", LACOSMIC_CRREJ.get("cleantype")),
                fs=row.get("fs", LACOSMIC_CRREJ.get("fs")),
                psffwhm=row.get("psffwhm", LACOSMIC_CRREJ.get("psffwhm")),
                psfsize=row.get("psfsize", LACOSMIC_CRREJ.get("psfsize")),
                psfbeta=row.get("psfbeta", LACOSMIC_CRREJ.get("psfbeta")),
                verbose=verbose,
            )

        if outpath is not None:
            ccd.write(outpath, overwrite=True, output_verify="fix")
        if return_ccd:
            ccds.append(ccd)

    if return_ccd:
        return ccds
