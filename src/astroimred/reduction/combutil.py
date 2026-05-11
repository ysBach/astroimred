"""
Archaic utilities before I developed imcombine.py.
This was a thin wrapper around ccdproc.combine.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import ccdproc
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from astropy.time import Time
from ccdproc import combine

from astroimred._core.numeric import sstd, weighted_avg
from astroimred._core.types import HDUExt, StrPathLike
from astroimred.fitsmgmt.header import cmt2hdr
from astroimred.fitsmgmt.io import _parse_extension, load_ccd
from astroimred.fitsmgmt.table import SummaryInput, fits_summary, select_fits
from astroimred.imutil.ccdops import CCDData_astype, imslice
from astroimred.logging import logger

__all__ = [
    "sstd",
    "weighted_mean",
    "combine_ccd",
]


# FIXME: Add this to Ccdproc esp. for mem_limit


def weighted_mean(
    ccds: Sequence[CCDData],
    unit: str | u.Unit = "adu",
) -> CCDData:
    """Combine CCDs with inverse-variance weights.

    Parameters
    ----------
    ccds : sequence of `~astropy.nddata.CCDData`
        Input CCDs with ``uncertainty`` arrays.
    unit : str or `~astropy.units.Unit`, optional
        Unit for the returned CCD.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Weighted mean image with propagated standard-deviation uncertainty.
    """
    datas = []
    errs = []
    for ccd in ccds:
        if ccd.uncertainty is None:
            raise ValueError("All CCDs must have uncertainty arrays.")
        datas.append(ccd.data)
        errs.append(ccd.uncertainty.array)
    wmean, wuncert = weighted_avg(np.array(datas), np.array(errs), axis=0)
    nccd = CCDData(data=wmean, header=ccds[0].header, unit=unit)
    nccd.uncertainty = StdDevUncertainty(wuncert)
    return nccd


def combine_ccd(
    fitslist: SummaryInput | None = None,
    summary_table: pd.DataFrame | Table | None = None,
    table_filecol: str = "file",
    trimsec: str | None = None,
    output: StrPathLike | None = None,
    unit: str | u.Unit | None = None,
    subtract_frame: CCDData | np.ndarray | None = None,
    combine_method: str = "median",
    reject_method: str | None = None,
    normalize_exposure: bool = False,
    normalize_average: bool = False,
    normalize_median: bool = False,
    exposure_key: str = "EXPTIME",
    mem_limit: float = 2e9,
    combine_uncertainty_function: Callable[..., np.ndarray] | None = None,
    extension: HDUExt = None,
    type_key: str | list[str] | None = None,
    type_val: object = None,
    dtype: str = "float32",
    uncertainty_dtype: str = "float32",
    output_verify: str = "fix",
    overwrite: bool = False,
    verbose: bool = True,
    **kwargs: object,
) -> CCDData:
    """Combining images -- slight variant from ccdproc.

    Parameters
    ----------
    fitslist: path-like, `list` of path-like, or `list` of `~astropy.nddata.CCDData`
        The `list` of path to FITS files or the `list` of `~astropy.nddata.CCDData` to be stacked. It
        is useful to give `list` of `~astropy.nddata.CCDData` if you have already stacked/loaded
        FITS file into a `list` by your own criteria. If `None` (default), you
        must give `fitslist` or `summary_table`. If it is not `None`, this
        function will do very similar job to that of `ccdproc.combine`.
        Although it is not a good idea, a mixed `list` of `~astropy.nddata.CCDData` and paths to
        the files is also acceptable.

    summary_table: `~pandas.DataFrame` or `~astropy.table.Table`
        The table which contains the metadata of files. If there are many FITS
        files and you want to use stacking many times, it is better to make a
        summary table by `~astroimred.fits_summary` and use that instead of
        opening FITS files' headers every time you call this function. If you
        want to use `summary_table` instead of `fitslist` and have set
        ``ccddata=True``, you must not have `None` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    table_filecol: `str`
        The column name of the `summary_table` which contains the path to the
        FITS files.

    trimsec : `str`, [`list` of] `int`, [`list` of] slice, optional
        Section of the data to be extracted by `~astroimred.imutil.ccdops.imslice`.
        Default is `None`.

    output : path-like or `None`, optional.
        The path if you want to save the resulting `~astropy.nddata.CCDData`
        object.
        Default is `None`.

    unit : `~astropy.units.Unit` or `str`, optional.
        The units of the data.
        Default is `None`.

    subtract_frame : array-like, optional.
        The frame you want to subtract from the image after the combination. It
        can be, e.g., dark frame, because it is easier to calculate Poisson
        error before the dark subtraction and subtract the dark later.
        TODO: This maybe unnecessary.
        Default is `None`.

    combine_method : `str` or `None`, optinal.
        The `method` for `ccdproc.combine`, i.e., ``{'average', 'median', 'sum'}``
        Default is `None`.

    reject_method : `str`
        Made for simple use of `ccdproc.combine`, [`None`, 'minmax', 'sigclip' ==
        'sigma_clip', 'extrema' == 'ext']. Automatically turns on the option,
        e.g., ``clip_extrema = True`` or ``sigma_clip = True``. Leave it blank
        for no rejection.
        Default is `None`.

    normalize_exposure : `bool`, optional.
        Whether to normalize the values by the exposure time of each frame
        before combining.
        Default is `False`.

    normalize_average, normalize_median : `bool`, optional.
        Whether to normalize the values by the average or median value of each
        frame before combining. Only up to one of these must be `True`.
        Default is `False`.

    exposure_key : `str`, optional
        The header keyword for the exposure time.
        Default is ``"EXPTIME"``.

    combine_uncertainty_function : callable, `None`, optional
        The uncertainty calculation function of `~ccdproc.combine`. If `None`
        use the default uncertainty func when using average, median or sum
        combine, otherwise use the function provided.
        Default is `None`.

    extension: `int`, `str`, (`str`, `int`)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    dtype : `str` or `numpy.dtype` or `None`, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If `None` it uses ``np.float64``.
        Default is `None`.

    type_key, type_val: `str`, `list` of `str`
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named `hdu`, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the
        `fitslist` will be used.

    output_verify : `str`
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``. May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    mem_limit : `float`, optional
        Maximum memory which should be used while combining (in bytes).
        Default is ``2.e9``.

    **kwarg:
        kwargs for the `ccdproc.combine`. See its documentation. This includes
        (RHS are the default values)

        .. code-block:: python

            weights=None,
            scale=None,
            mem_limit=16000000000.0,
            clip_extrema=False,
            nlow=1,
            nhigh=1,
            minmax_clip=False,
            minmax_clip_min=None,
            minmax_clip_max=None,
            sigma_clip=False,
            sigma_clip_low_thresh=3,
            sigma_clip_high_thresh=3,
            sigma_clip_func=<numpy.ma.core._frommethod instance>,
            sigma_clip_dev_func=<numpy.ma.core._frommethod instance>,
            combine_uncertainty_function=None, **ccdkwargs


    Returns
    -------
    master: astropy.nddata.CCDData
        Resulting combined ccd.
    """
    # def _normalize_exptime(ccdlist, exposure_key):
    #     _ccdlist = ccdlist.copy()
    #     exptimes = []
    #     for i in range(len(_ccdlist)):
    #         exptime = _ccdlist[i].header[exposure_key]
    #         exptimes.append(exptime)
    #         _ccdlist[i] = _ccdlist[i].divide(exptime)
    #     if verbose:
    #         if len(np.unique(exptimes)) != 1:
    #             print('There are more than one exposure times:\n\t', end=' ')
    #             print(np.unique(exptimes), end=' ')
    #             print('seconds')
    #         print(f'Normalized images by exposure time ("{exposure_key}").')
    #     return _ccdlist

    def _set_reject_method(reject_method):
        """Convenience function for ccdproc.combine reject switches"""
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method in ["extrema", "ext"]:
            clip_extrema = True
        elif reject_method in ["minmax"]:
            minmax_clip = True
        elif reject_method in ["sigma_clip", "sigclip"]:
            sigma_clip = True
        else:
            if reject_method not in [None, "no"]:
                raise KeyError(
                    "reject must be one of "
                    + "[None, 'minmax', sigclip'=='sigma_clip', 'extrema'=='ext']"
                )

        return clip_extrema, minmax_clip, sigma_clip

    # def _print_info(combine_method, Nccd, reject_method, **kwargs):
    #     if reject_method is None:
    #         reject_method = 'no'

    #     info_str = ('"{:s}" combine {:d} images by "{:s}" rejection')

    #     print(info_str.format(combine_method, Nccd, reject_method))
    #     print(dict(**kwargs))
    #     return

    def _add_and_log(s, header, verbose):
        header.add_history(s)
        if verbose:
            logger.info(s)

    # Give only one
    if (fitslist is not None) + (summary_table is not None) != 1:
        raise ValueError("One and only one of [fitslist, summary_table] must be given.")

    # If fitslist
    if fitslist is not None:
        # === a single CCDData ======================================================= #
        if isinstance(fitslist, CCDData):
            fitslist = [fitslist]
        else:
            # === a single path-like ================================================= #
            try:
                fitslist = [Path(fitslist)]
            except TypeError:
                # === a list of path-like or CCDData ================================= #
                try:
                    fitslist = list(fitslist)
                except TypeError as err:
                    raise TypeError(
                        f"fitslist must be list-like. It's now {type(fitslist)}."
                    ) from err

    # If summary_table
    if (
        summary_table is not None
        and (not isinstance(summary_table, Table))
        and (not isinstance(summary_table, pd.DataFrame))
    ):
        raise TypeError(
            "summary_table must be an astropy Table or Pandas DataFrame. "
            + f"It's now {type(summary_table)}."
        )

    # Check for type_key and type_val
    if (type_key is None) ^ (type_val is None):
        raise ValueError("type_key and type_val must be both specified or both None.")

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            if verbose:
                logger.info("%s already exists: But will be overridden.", output)
        else:
            if verbose:
                logger.info("%s already exists", output)
                logger.info("Loading the existing %s...", output)
            master = CCDData.read(output)
            if verbose:
                logger.info("Done")
            return master

    # Do we really need to accept all three of normalize & scale?
    # if scale is None:
    #     scale = np.ones(len(ccdlist))
    if ((normalize_average) + (normalize_exposure) + (normalize_median)) > 1:
        raise ValueError(
            "Only up to one of [normalize_average, normalize_exposure, normalize_median] "
            + "is acceptable."
        )

    # Set history messages
    str_history = (
        '{:d} images with {:s} = {:s} are "{:s}" combined '
        + 'using "{:s}" rejection (additional kwargs: {})'
    )
    str_nexp = "Each frame will be normalized by exposure time before combine."
    str_navg = "Each frame will be normalized by average before combine."
    str_nmed = "Each frame will be normalized by median before combine."
    str_subt = "Subtracted a user-provided frame"

    if reject_method is None:
        reject_method = "no"

    extension = _parse_extension(extension)

    # Select CCDs by
    ccdlist = select_fits(
        inputs=fitslist if fitslist is not None else summary_table,
        table_filecol=table_filecol,
        extension=extension,
        # extension will be parsed within fits_summary/load_ccd (no need to care here)
        unit=unit,
        type_key=type_key,
        type_val=type_val,
        prefer_ccddata=False,
        verbose=verbose,
    )
    # prefer_ccddata=False: Loading CCD here may cause memory blast...

    try:
        header = ccdlist[0].header
    except AttributeError:
        header = fits.getheader(ccdlist[0])

    # if verbose:
    #     _print_info(
    #         combine_method=combine_method,
    #         Nccd=len(ccdlist),
    #         reject_method=reject_method,
    #         dtype=dtype,
    #         **kwargs)

    _t = Time.now()
    scale = None
    # Normalize by exposure
    # TODO: Let it accept summary table as well as fitslist
    if normalize_exposure:
        tmp = fits_summary(
            fitslist=fitslist, keywords=[exposure_key], verbose=False, sort_by=None
        )
        exptimes = tmp[exposure_key].tolist()
        scale = 1 / np.array(exptimes)
        cmt2hdr(header, "h", str_nexp, verbose=verbose)

    # Normalize by pixel average
    if normalize_average:

        def invavg(a):
            return 1 / np.mean(a)

        scale = invavg
        cmt2hdr(header, "h", str_navg, verbose=verbose)

    # Normalize by pixel median
    if normalize_median:

        def invmed(a):
            return 1 / np.median(a)

        scale = invmed
        cmt2hdr(header, "h", str_nmed, verbose=verbose)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    if len(ccdlist) == 1:
        if isinstance(ccdlist[0], CCDData):
            master = ccdlist[0]
        else:
            # extension will be parsed within load_ccd (no need to care here)
            master = load_ccd(ccdlist[0], extension=extension, unit=unit)
    else:
        master = combine(
            img_list=ccdlist,
            method=combine_method,
            clip_extrema=clip_extrema,
            minmax_clip=minmax_clip,
            sigma_clip=sigma_clip,
            mem_limit=mem_limit,
            combine_uncertainty_function=combine_uncertainty_function,
            unit=unit,  # user-given unit is already applied by select_fits
            hdu=extension,
            scale=scale,
            dtype=dtype,
            **kwargs,
        )

    header["COMBVER"] = (ccdproc.__version__, "ccdproc version used for combine.")
    # NCOMBINE from ccdproc has no comment so I duplicate this...
    ncombine = len(ccdlist)
    header["NCOMBINE"] = (ncombine, "Number of combined images")
    header["COMBMETH"] = (combine_method, "Combining method")

    s = str_history.format(
        ncombine,
        str(type_key),
        str(type_val),
        str(combine_method),
        str(reject_method),
        kwargs,
    )
    cmt2hdr(header, "h", s, verbose=verbose, t_ref=_t)
    # header.add_history(str_history.format(ncombine,
    #                                       str(type_key),
    #                                       str(type_val),
    #                                       str(combine_method),
    #                                       str(reject_method),
    #                                       kwargs))

    if subtract_frame is not None:
        _t = Time.now()
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        cmt2hdr(header, "h", str_subt, header, verbose=verbose, t_ref=_t)

    if trimsec is not None:
        master = imslice(master, trimsec, verbose=verbose)

    master.header = header
    master = CCDData_astype(master, dtype=dtype, uncertainty_dtype=uncertainty_dtype)
    # update_tlm is done incide CCDData_astype

    if output is not None:
        if verbose:
            logger.info("Writing FITS to %s...", output)
        master.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose:
            logger.info("Saved.")

    return master
