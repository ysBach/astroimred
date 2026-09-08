"""FITS image loading helpers."""

import glob
from collections.abc import Sequence
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pandas as pd
from astro_ndslice import is_list_like, listify, slicefy
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import WCS

from .._core.system import get_size
from .._core.types import HDUExt, HDULike, HDULike_types, StrPathLike
from ..logging import logger

try:
    import fitsio

    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

InputListInput = (
    StrPathLike
    | HDULike
    | Sequence[StrPathLike | HDULike]
    | pd.DataFrame
    | Table
    | dict[str, object]
)

__all__ = [
    "_parse_data_header",
    "_parse_image",
    "_has_header",
    "_parse_extension",
    "get_size",
    "write2fits",
    "inputs2list",
    "load_ccd",
    "load_ccds",
]


# ************************************************************************************ #
# *                                         PARSERS                                  * #
# ************************************************************************************ #
def _parse_data_header(
    ccdlike,
    extension: HDUExt = None,
    parse_data: bool = True,
    parse_header: bool = True,
    copy: bool = True,
):
    """Parse data and/or a header from an image-like input.

    Parameters
    ----------
    ccdlike : `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`, `~astropy.io.fits.Header`, `~numpy.ndarray`, number-like, path-like, or `None`
        Input object. ``None`` and an empty string return ``(None, None)``.
    extension : `int`, `str`, or (`str`, `int`), optional
        FITS extension: a zero-based index, an ``EXTNAME`` string, or an
        ``(EXTNAME, EXTVER)`` tuple. ``None`` selects the first extension with data.
    parse_data, parse_header : `bool`, optional
        Select which parts to return. Both default to ``True``.
    copy : `bool`, optional
        Whether to copy returned data and headers. Set to ``False`` only when
        references to the input objects are intentional. Default: ``True``.

    Returns
    -------
    data : `~numpy.ndarray`, scalar, or `None`
        Parsed data, or ``None`` when ``parse_data=False`` or the input has no data.
    hdr : `~astropy.io.fits.Header` or `None`
        Parsed header, or ``None`` when ``parse_header=False`` or no header exists.

    Notes
    -----
    Use ``_parse_data_header`` for a quick data/header copy or header check. Use
    ``_parse_image`` when normalizing image data or requesting
    `~astropy.nddata.CCDData`.
    """
    if ccdlike is None or (isinstance(ccdlike, str) and ccdlike == ""):
        data = None
        hdr = None
    elif isinstance(ccdlike, HDULike_types):
        data = (ccdlike.data.copy() if copy else ccdlike.data) if parse_data else None
        if parse_header:
            hdr = ccdlike.header.copy() if copy else ccdlike.header
        else:
            hdr = None
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension) if (parse_data or parse_header) else 0
        # ^ don't even do _parse_extension if both are False
        if parse_data:
            data = ccdlike[extension].data.copy() if copy else ccdlike[extension].data
        else:
            data = None
        if parse_header:
            hdr = (
                ccdlike[extension].header.copy() if copy else ccdlike[extension].header
            )
        else:
            hdr = None
    elif isinstance(ccdlike, (np.ndarray, list, tuple)):
        data = np.array(ccdlike, copy=copy) if parse_data else None
        hdr = None  # regardless of parse_header
    elif isinstance(ccdlike, fits.Header):
        data = None  # regardless of parse_data
        hdr = (ccdlike.copy() if copy else ccdlike) if parse_header else None
    elif HAS_FITSIO and isinstance(ccdlike, fitsio.FITSHDR):
        import copy

        data = None  # regardless of parse_data
        hdr = (copy.deepcopy(ccdlike) if copy else ccdlike) if parse_header else None
    else:
        try:
            data = float(ccdlike) if (parse_data or parse_header) else None
            hdr = None
        except (ValueError, TypeError):  # Path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            extension = _parse_extension(extension) if parse_data or parse_header else 0
            # fits.getheader is ~ 10-20 times faster than load_ccd.
            # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
            try:
                if parse_header:
                    with fits.open(Path(ccdlike), memmap=False) as hdul:
                        hdu = hdul[extension]
                        data = (
                            (
                                hdu.data.copy()
                                if copy and hdu.data is not None
                                else hdu.data
                            )
                            if parse_data
                            else None
                        )
                        hdr = hdu.header.copy() if copy else hdu.header
                else:
                    if isinstance(extension, tuple):
                        if HAS_FITSIO:
                            data = fitsio.read(
                                Path(ccdlike), ext=extension[0], extver=extension[1]
                            )
                        else:
                            data = fits.getdata(Path(ccdlike), *extension)
                    else:
                        if HAS_FITSIO:
                            data = fitsio.read(Path(ccdlike), ext=extension)
                            # fitsio returns None for empty HDUs (e.g., PrimaryHDU
                            # with no data). Fall back to first extension with data.
                            if data is None:
                                with fits.open(Path(ccdlike), memmap=False) as hdul:
                                    for hdu in hdul:
                                        if hdu.data is not None:
                                            data = hdu.data
                                            break
                        else:
                            data = fits.getdata(Path(ccdlike), extension)
                    hdr = None
            except TypeError as err:
                raise TypeError(
                    f"ccdlike type ({type(ccdlike)}) is not acceptable "
                    + "to find header and data."
                ) from err

    return data, hdr


# TODO: str(pathlibPath)
def _parse_image(
    ccdlike,
    extension: HDUExt = None,
    name: str | None = None,
    force_ccddata: bool = False,
    prefer_ccddata: bool = False,
    copy: bool = True,
):
    """Parse an image-like input as an array or `~astropy.nddata.CCDData`.

    Parameters
    ----------
    ccdlike : `~astropy.nddata.CCDData`-like, `~numpy.ndarray`, path-like, or number-like
        The image to parse. A value accepted by ``float(ccdlike)`` is treated as a
        scalar; other path-like values are read as FITS files.
    extension : `int`, `str`, or (`str`, `int`), optional
        FITS extension: a zero-based index, an ``EXTNAME`` string, or an
        ``(EXTNAME, EXTVER)`` tuple. ``None`` selects the first extension with data.
        Default: ``None``.
    name : `str`, optional
        Name used in the returned ``imname`` and in log messages. If omitted, a
        descriptive name is generated from the input. Default: ``None``.
    force_ccddata, prefer_ccddata : `bool`, optional
        ``force_ccddata=True`` always returns `~astropy.nddata.CCDData`, which is
        useful when metadata or error propagation is needed. With
        ``prefer_ccddata=True``, CCDData/HDU/path inputs return CCDData while arrays
        and numbers remain arrays. Both default to ``False``.
    copy : `bool`, optional
        Whether to copy array, header, and CCDData inputs. Default: ``True``.

    Returns
    -------
    new_im : `~numpy.ndarray` or `~astropy.nddata.CCDData`
        Parsed image, selected by ``force_ccddata`` and ``prefer_ccddata``.
    imname : `str`
        The supplied or generated image name.
    imtype : `str`
        Input category: ``"CCDData"``, ``"hdu"``, ``"HDUList"``, ``"ndarray"``,
        ``"num"``, or ``"path"``.

    Notes
    -----
    ``_parse_data_header`` is the quick data/header helper. Reading a path with
    ``force_ccddata=True`` is slower because Astropy parses its metadata; use an
    array or ``force_ccddata=False`` when metadata is unnecessary.

    Historical timing records
    -------------------------
    These measurements are retained for comparison only; they are not performance
    contracts.

    Timing on MBP 14" [2021, macOS 12.2.1, M1Pro(6P+2E/G16c/N16c/32G)]:


    >>> np.random.RandomState(123)
    >>> data = np.random.normal(size=(100,100))
    >>> ccd = CCDData(data, unit='adu')
    >>> fpath = "img/0001.fits"  # doctest: +SKIP
    >>> %timeit air._parse_image(data, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(ccd, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=True) # doctest: +SKIP
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=False)[0]*1.0 # doctest: +SKIP
    # 14.2 µs +- 208 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    # 16.6 µs +- 298 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    # 20.8 ms +- 133 µs per loop (mean +- std. dev. of 7 runs, 10000 loops each)
    # 156 µs +- 3.3 µs per loop (mean +- std. dev. of 7 runs, 10000 loops each)

    `fpath` contains a FITS file of 276KB. Note that path with `force_ccddata =
    True` consumes tremendous amount of time, because of astropy's header
    parsing scheme.

    Timing on MBP 14" [2024, macOS 26.4.1,
    M4Pro(8P+4E/G20c/N16c/48G)], 2026-05-27:

    >>> rng = np.random.default_rng(123)
    >>> data = rng.normal(size=(100, 100))
    >>> ccd = CCDData(data, unit="adu")
    >>> fpath = tmp_path / "parse_0001.fits"  # doctest: +SKIP
    >>> fits.writeto(fpath, rng.normal(size=(256, 256)).astype("float32"))  # doctest: +SKIP
    >>> %timeit air._parse_image(data, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(ccd, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=True)  # doctest: +SKIP
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=False)[0] * 1.0  # doctest: +SKIP
    # 5.21 µs ± 182 ns per loop (7 runs, 2000 loops each)
    # 6.87 µs ± 148 ns per loop (7 runs, 2000 loops each)
    # 1.06 ms ± 29.6 µs per loop (7 runs, 20 loops each)
    # 176 µs ± 2.26 µs per loop (7 runs, 200 loops each)
    """

    def __extract_extension(ext):
        extension = _parse_extension(ext)
        if extension is None:
            extstr = ""
        else:
            if isinstance(extension, (tuple, list)):
                extstr = f"[{extension[0]}, {extension[1]}]"
            else:
                extstr = f"[{extension}]"
        return extension, extstr

    def __extract_from_hdu(hdu, force_ccddata, prefer_ccddata):
        if force_ccddata or prefer_ccddata:
            unit = ccdlike.header.get("BUNIT", default=u.adu)
            if isinstance(unit, str):
                unit = unit.lower()
            if copy:
                return CCDData(
                    data=hdu.data.copy(), header=hdu.header.copy(), unit=unit
                )
            else:
                return CCDData(data=hdu.data, header=hdu.header, unit=unit)
            # The two lines above took ~ 5 us and 10-30 us for the simplest
            # header and 1x1 pixel data case (regardless of BUNIT exists), on
            # MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
            # GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
        else:
            return hdu.data.copy() if copy else hdu.data

    ccd_kw = {"force_ccddata": force_ccddata, "prefer_ccddata": prefer_ccddata}
    has_no_name = name is None
    extension, extstr = __extract_extension(extension)
    imname = (
        f"User-provided {ccdlike.__class__.__name__}{extstr}" if has_no_name else name
    )

    if isinstance(ccdlike, CCDData):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        if force_ccddata or prefer_ccddata:
            new_im = ccdlike.copy() if copy else ccdlike
        else:
            new_im = ccdlike.data.copy() if copy else ccdlike.data
        imtype = "CCDData"
        imname = str(imname).replace("[0]", "")
    elif isinstance(ccdlike, (fits.PrimaryHDU, fits.ImageHDU)):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike, **ccd_kw)
        imtype = "hdu"
        imname = str(imname).replace("[0]", "")
    elif isinstance(ccdlike, fits.HDUList):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike[extension], **ccd_kw)
        imtype = "HDUList"
    elif isinstance(ccdlike, np.ndarray):
        # force_ccddata: CCDData // prefer_ccddata: ndarray // else: ndarray
        if copy:
            new_im = (
                CCDData(data=ccdlike.copy(), unit="adu")
                if force_ccddata
                else ccdlike.copy()
            )
        else:
            new_im = CCDData(data=ccdlike, unit="adu") if force_ccddata else ccdlike
        imtype = "ndarray"
    else:
        try:  # IF number (ex: im = 1.3)
            # force_ccddata: CCDData // prefer_ccddata: array // else: array
            imname = f"{imname} {ccdlike}" if has_no_name else name
            _im = float(ccdlike)
            new_im = CCDData(data=_im, unit="adu") if force_ccddata else np.asarray(_im)
            imtype = "num"
            # imname can be "int", "float", "str", etc, so imtype might be useful.
        except (ValueError, TypeError):
            try:
                fpath = Path(ccdlike)
            except TypeError as err:
                raise TypeError(
                    "input must be CCDData-like, ndarray, path-like (to FITS), or a number."
                ) from err

            # If we are here, it is a path-like.
            imname = f"{str(fpath)}{extstr}" if has_no_name else name
            # set redundant extensions to None so that only the part
            # specified by `extension` be loaded:
            new_im = load_ccd(
                fpath,
                extension,
                ccddata=prefer_ccddata or force_ccddata,
                extension_uncertainty=None,
                extension_mask=None,
            )
            imtype = "path"

    return new_im, imname, imtype


def _has_header(ccdlike, extension: HDUExt = None, open_if_file: bool = True) -> bool:
    """Return whether an object has, or points to, a FITS-like header.

    Parameters
    ---------
    ccdlike : `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`, `~numpy.ndarray`, number-like, path-like
        The object to be parsed into data and header.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used. Used only if `ccdlike` is `~astropy.io.fits.HDUList` or
        path-like.
        Default: `None`.

    open_if_file : `bool`, optional.
        Whether to open the file to check if it has a header when `ccdlike` is
        path-like. Any FITS file has a header, so this means it will check the
        existence and validity of the file. If set to `False`, all path-like
        input will return `False` because the path itself has no header.
        Default: `True`.

    Notes
    -----
    It first checks if the input is one of ``(`~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`,
    `~astropy.io.fits.ImageHDU`)``, then if `~astropy.io.fits.HDUList`, then if `np.ndarray`, then if
    number-like, and then finally if path-like. Although this has a bit of
    disadvantage considering we may use file-path for most of the time, the
    overhead is only ~ 1 us, tested on MBP 15" [2018, macOS 10.14.6, i7-8850H
    (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].
    """
    hashdr = True
    if isinstance(ccdlike, HDULike_types):  # extension not used
        try:
            hashdr = ccdlike.header is not None
        except AttributeError:
            hashdr = False
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension)
        try:
            hashdr = ccdlike[extension].header is not None
        except AttributeError:
            hashdr = False
    elif is_list_like(ccdlike):
        hashdr = False
    else:
        try:  # if number-like
            _ = float(ccdlike)
            hashdr = False
        except (ValueError, TypeError):  # if path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            if open_if_file:
                try:
                    # fits.getheader is ~ 10-20 times faster than load_ccd.
                    # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
                    _ = fits.getheader(Path(ccdlike), extension)
                except (AttributeError, FileNotFoundError):
                    hashdr = False
            else:
                hashdr = False

    return hashdr


def _parse_extension(
    *args, ext: HDUExt = None, extname: str | None = None, extver: int | None = None
) -> HDUExt:
    """
    Open the input file, return the `~astropy.io.fits.HDUList` and the extension.

    This supports several different styles of extension selection.  See the
    :func:`getdata()` documentation for the different possibilities.

    Direct copy from astropy, but removing "opening `~astropy.io.fits.HDUList`" part
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L988

    This is essential for fits_ccddata_reader, because it only has `hdu`, not
    all three of ext, extname, and extver.

    Notes
    -----
    extension parser itself is not a time-consuming process:

    %timeit air._parse_extension()
    # 1.52 µs +- 69.3 ns per loop (mean +- std. dev. of 7 runs, 1000000 loops each)
    """

    err_msg = "Redundant/conflicting extension arguments(s): {}".format(
        {"args": args, "ext": ext, "extname": extname, "extver": extver}
    )

    # This code would be much simpler if just one way of specifying an
    # extension were picked.  But now we need to support all possible ways for
    # the time being.
    if len(args) == 1:
        # Must be either an extension number, an extension name, or an
        # (extname, extver) tuple
        if isinstance(args[0], (int, np.integer)) or (
            isinstance(ext, tuple) and len(ext) == 2
        ):
            if ext is not None or extname is not None or extver is not None:
                raise TypeError(err_msg)
            ext = args[0]
        elif isinstance(args[0], str):
            # The first arg is an extension name; it could still be valid to
            # provide an extver kwarg
            if ext is not None or extname is not None:
                raise TypeError(err_msg)
            extname = args[0]
        else:
            # Take whatever we have as the ext argument; we'll validate it below
            ext = args[0]
    elif len(args) == 2:
        # Must be an extname and extver
        if ext is not None or extname is not None or extver is not None:
            raise TypeError(err_msg)
        extname = args[0]
        extver = args[1]
    elif len(args) > 2:
        raise TypeError("Too many positional arguments.")

    if ext is not None and not (
        isinstance(ext, (int, np.integer))
        or (
            isinstance(ext, tuple)
            and len(ext) == 2
            and isinstance(ext[0], str)
            and isinstance(ext[1], (int, np.integer))
        )
    ):
        raise ValueError(
            "The ext keyword must be either an extension number (zero-indexed) "
            + "or a (extname, extver) tuple."
        )
    if extname is not None and not isinstance(extname, str):
        raise ValueError("The extname argument must be a string.")
    if extver is not None and not isinstance(extver, (int, np.integer)):
        raise ValueError("The extver argument must be an integer.")

    if ext is None and extname is None and extver is None:
        ext = 0
    elif ext is not None and (extname is not None or extver is not None):
        raise TypeError(err_msg)
    elif extname:
        ext = (extname, extver) if extver else (extname, 1)
    elif extver and extname is None:
        raise TypeError("extver alone cannot specify an extension.")

    return ext


def _parse_extension_or_none(ext: HDUExt) -> HDUExt:
    """Return `None` if ext is `None`, otherwise parse it."""
    if ext is None:
        return None
    return _parse_extension(ext)


def _build_ccd_reader_kwargs(
    path: StrPathLike,
    extension: HDUExt,
    extension_uncertainty: str | None,
    extension_mask: str | None,
    extension_flags: str | None,
    key_uncertainty_type: str,
    memmap: bool,
    use_wcs: bool,
    **kwd,
) -> dict:
    reader_kw = dict(
        hdu=extension,
        hdu_uncertainty=_parse_extension_or_none(extension_uncertainty),
        hdu_mask=_parse_extension_or_none(extension_mask),
        hdu_flags=_parse_extension_or_none(extension_flags),
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        **kwd,
    )
    # ^ If hdu_flags is not None, CCDData raises this Error:
    #   NotImplementedError: loading flags is currently not supported.

    # FIXME: Remove this if block in the future if WCS issue is resolved.
    if use_wcs:  # Because of the TPV WCS issue
        hdr = fits.getheader(path)
        reader_kw["wcs"] = WCS(hdr)
        del hdr

    return reader_kw


def _load_ccd_astropy(
    path: StrPathLike,
    extension: HDUExt,
    trimsec: str | None,
    unit: u.Unit | None,
    extension_uncertainty: str | None,
    extension_mask: str | None,
    extension_flags: str | None,
    key_uncertainty_type: str,
    memmap: bool,
    use_wcs: bool,
    **kwd,
) -> tuple[CCDData, dict]:
    reader_kw = _build_ccd_reader_kwargs(
        path=path,
        extension=extension,
        extension_uncertainty=extension_uncertainty,
        extension_mask=extension_mask,
        extension_flags=extension_flags,
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        use_wcs=use_wcs,
        **kwd,
    )

    try:  # Use BUNIT if unit is None
        ccd = CCDData.read(path, unit=unit, **reader_kw)
    except ValueError:  # e.g., user did not give unit and there's no BUNIT
        ccd = CCDData.read(path, unit=u.adu, **reader_kw)

    if trimsec is not None:
        # Do imslice AFTER loading the data to easily add LTV/LTM...
        from ..imutil.ccdops import imslice

        ccd = imslice(ccd, trimsec=trimsec)

    return ccd, reader_kw


def _read_fitsio_extension(
    hdul, ext: HDUExt, trimsec: str | None = None
) -> np.ndarray | None:
    if ext is None:
        return None
    ext = _parse_extension_or_none(ext)
    try:
        if trimsec is not None:
            sl = slicefy(trimsec)
            if is_list_like(ext):
                # length == 2 is already checked in _parse_extension.
                arr = hdul[ext[0], ext[1]].read()[sl]
            else:
                arr = hdul[ext].read()[sl]
        else:
            arr = hdul[ext[0], ext[1]].read() if is_list_like(ext) else hdul[ext].read()
        return arr
    except (OSError, ValueError) as e:
        logger.debug("Error reading extension: %s", e)
        # "Extension `{ext}` is not found (file: `{path}`)")
        return None


def _load_ccd_fitsio(
    path: StrPathLike,
    extension: HDUExt,
    trimsec: str | None,
    full: bool,
    extension_uncertainty: str | None,
    extension_mask: str | None,
    extension_flags: str | None,
):
    # Use fitsio and only load the data as soon as possible.
    # This is much quicker than astropy's getdata
    with fitsio.FITS(path) as hdul:
        if full:
            dat = _read_fitsio_extension(hdul, extension, trimsec)
            unc = _read_fitsio_extension(hdul, extension_uncertainty, trimsec)
            msk = _read_fitsio_extension(hdul, extension_mask, trimsec)
            flg = _read_fitsio_extension(hdul, extension_flags, trimsec)
            return dat, unc, msk, flg

        return _read_fitsio_extension(hdul, extension, trimsec)


def _astropy_raw_return(
    ccd: CCDData,
    full: bool,
    extension_uncertainty: HDUExt,
    extension_mask: HDUExt,
    extension_flags: HDUExt,
):
    if full:
        try:
            unc = (
                None
                if extension_uncertainty is None
                else np.array(ccd.uncertainty.array)
            )
        except AttributeError:
            unc = None
        try:
            mask = None if extension_mask is None else np.array(ccd.mask.array)
        except AttributeError:
            mask = None
        flag = None if extension_flags is None else np.array(ccd.flags)
        return ccd.data, unc, mask, flag
    return ccd.data


def load_ccd(
    path: StrPathLike,
    extension: HDUExt = None,
    trimsec: str | None = None,
    ccddata: bool = True,
    use_wcs: bool = True,
    unit: u.Unit | None = None,
    extension_uncertainty: str | None = "UNCERT",
    extension_mask: str | None = "MASK",
    extension_flags: str | None = None,
    full: bool = False,
    key_uncertainty_type: str = "UTYPE",
    memmap: bool = False,
    **kwd,
) -> CCDData | np.ndarray | tuple[object, ...]:
    """Load one FITS image, with optional uncertainty and mask extensions.

    Parameters
    ----------
    path : path-like
        FITS file to load.
    extension : `int`, `str`, or (`str`, `int`), optional
        FITS extension: a zero-based index, an ``EXTNAME`` string, or an
        ``(EXTNAME, EXTVER)`` tuple. ``None`` selects the first extension with data.
        Default: ``None``.
    trimsec : `str`, optional
        FITS-style section to apply after loading the selected image. Default:
        ``None``.
    ccddata : `bool`, optional
        Return `~astropy.nddata.CCDData` when ``True`` (default). When ``False``,
        return only the image array unless ``full=True``.
    use_wcs : `bool`, optional
        Build the WCS with `~astropy.io.fits.getheader` instead of Astropy's CCDData
        reader. This keeps TPV WCS usable but can be slower. Used only for the
        Astropy/CCDData path. Default: ``True``.
    unit : `~astropy.units.Unit`, optional
        Override the image unit. If omitted, ``BUNIT`` is used when present;
        otherwise Astropy falls back to ADU. Used for CCDData loading only.
    full : `bool`, optional
        Return ``(data, unc, mask, flag)`` for raw loading, or
        ``(ccd, uncertainty, mask, flags)`` when ``ccddata=True``. Missing
        extensions produce ``None``. Default: ``False``.
    extension_uncertainty, extension_mask, extension_flags : extension selector or `None`, optional
        Extensions used to initialize the corresponding CCDData components or raw
        arrays. Defaults are ``"UNCERT"``, ``"MASK"``, and ``None``. Flags are not
        supported by Astropy's CCDData reader.
    key_uncertainty_type : `str`, optional
        Header key naming the uncertainty class. Used only for CCDData loading.
        Default: ``"UTYPE"``.
    memmap : `bool`, optional
        Whether Astropy should memory-map FITS data. Default: ``False`` (the
        opposite of Astropy's default configuration).
    **kwd
        Additional keywords passed to `~astropy.nddata.CCDData.read` in the Astropy
        path. They are ignored by the direct `fitsio` path used when
        ``ccddata=False`` and `fitsio` is installed.

    Returns
    -------
    `~astropy.nddata.CCDData`, `~numpy.ndarray`, or tuple
        The selected image, or the full tuple requested by ``full``.

    Notes
    -----
    When `fitsio` is installed and ``ccddata=False``, the direct reader uses
    ``extension``, ``trimsec``, ``full``, and the three extension selectors; WCS,
    unit, uncertainty-class, and memory-map options do not apply in that branch.
    """
    try:
        path = Path(path)
    except TypeError as err:
        raise TypeError(f"You must provide Path-like, not {type(path)}.") from err

    if "as_ccd" in kwd:
        raise TypeError("`as_ccd` was removed. Use `ccddata` instead.")

    extension = _parse_extension(extension)

    if HAS_FITSIO and not ccddata:
        return _load_ccd_fitsio(
            path=path,
            extension=extension,
            trimsec=trimsec,
            full=full,
            extension_uncertainty=extension_uncertainty,
            extension_mask=extension_mask,
            extension_flags=extension_flags,
        )

    e_u = _parse_extension_or_none(extension_uncertainty)
    e_m = _parse_extension_or_none(extension_mask)
    e_f = _parse_extension_or_none(extension_flags)

    ccd, _ = _load_ccd_astropy(
        path=path,
        extension=extension,
        trimsec=trimsec,
        unit=unit,
        extension_uncertainty=extension_uncertainty,
        extension_mask=extension_mask,
        extension_flags=extension_flags,
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        use_wcs=use_wcs,
        **kwd,
    )

    # Force them to be None if extension is not specified
    # (astropy.NDData.CCDData forces them to be loaded, which is not desirable imho)
    ccd.uncertainty = None if e_u is None else ccd.uncertainty
    ccd.mask = None if e_m is None else ccd.mask

    if ccddata:
        if full:  # Just for API consistency
            return ccd, ccd.uncertainty, ccd.mask, ccd.flags
        return ccd

    return _astropy_raw_return(
        ccd,
        full=full,
        extension_uncertainty=e_u,
        extension_mask=e_m,
        extension_flags=e_f,
    )


def load_ccds(
    paths: StrPathLike | list[StrPathLike],
    extension: HDUExt = None,
    trimsec: str | None = None,
    ccddata: bool = True,
    use_wcs: bool = True,
    unit: u.Unit | None = None,
    extension_uncertainty: str | None = "UNCERT",
    extension_mask: str | None = "MASK",
    extension_flags: str | None = None,
    full: bool = False,
    key_uncertainty_type: str = "UTYPE",
    memmap: bool = False,
    **kwd,
) -> list:
    """Load multiple FITS files with ``load_ccd``.

    Parameters
    ----------
    paths : path-like or sequence of path-like
        A file path, glob pattern, or sequence of paths and glob patterns.
    extension, trimsec, ccddata, use_wcs, unit, extension_uncertainty,
    extension_mask, extension_flags, full, key_uncertainty_type, memmap
        Passed unchanged to ``load_ccd`` for every resolved path.
    **kwd
        Additional keywords passed to ``load_ccd``.

    Returns
    -------
    list
        One result from ``load_ccd`` per resolved path. Paths are expanded and
        sorted by ``inputs2list`` before loading.

    Historical timing records
    -------------------------
    These measurements are retained for comparison only; they are not performance
    contracts.

    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)] using 10
    FITS (each 4.3 MB) with ~ 100 header cards:

    .. code-block:: python

        %timeit ccds = air.load_ccds("h_20191021_000*")
        105 ms +- 2.11 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)

    Timing on MBP 14" [2024, macOS 26.4.1,
    M4Pro(8P+4E/G20c/N16c/48G)], 2026-05-27, using 10 FITS
    files (each 4.2 MB):

    .. code-block:: python

        %timeit air.load_ccds(pattern)
        # 11.0 ms ± 603 µs per loop (5 runs, 2 loops each)
        %timeit air.load_ccds(
        ...     pattern,
        ...     use_wcs=False,
        ...     extension_uncertainty=None,
        ...     extension_mask=None,
        ... )
        # 6.52 ms ± 247 µs per loop (7 runs, 3 loops each)
    """
    paths2load = []
    for p in listify(paths):
        paths2load += inputs2list(p, sort=True, accept_ccdlike=False)
    return [
        load_ccd(
            p,
            extension=extension,
            trimsec=trimsec,
            ccddata=ccddata,
            use_wcs=use_wcs,
            unit=unit,
            extension_uncertainty=extension_uncertainty,
            extension_mask=extension_mask,
            extension_flags=extension_flags,
            full=full,
            key_uncertainty_type=key_uncertainty_type,
            memmap=memmap,
            **kwd,
        )
        for p in np.array(paths2load).ravel()
    ]


def write2fits(
    data: np.ndarray,
    header: fits.Header,
    output: StrPathLike,
    return_ccd: bool = False,
    **kwargs,
) -> CCDData | None:
    """Write data and a header to a FITS file via `~astropy.nddata.CCDData`.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data

    header : `~astropy.io.fits.Header`
        The header

    output : path-like
        The output file path

    return_ccd : `bool`, optional.
        Whether to return the generated `~astropy.nddata.CCDData`.

    **kwargs :
        Keyword arguments passed to `~astropy.nddata.CCDData.write`, such as
        ``output_verify="fix"`` or ``overwrite=True``.
    """
    ccd = CCDData(data=data, header=header, unit=header.get("BUNIT", "adu"))

    try:
        ccd.write(output, **kwargs)
    except fits.VerifyError:
        logger.warning("Try using output_verify='fix' to avoid this error.")
    if return_ccd:
        return ccd


def inputs2list(
    inputs: InputListInput,
    sort: bool = True,
    accept_ccdlike: bool = True,
    path_to_text: bool = False,
    check_coherency: bool = False,
) -> list | None:
    """Normalize file, table, and CCD-like inputs to a list.

    Parameters
    ----------
    inputs : `str`, path-like, CCD-like, table-like, or sequence
        A glob pattern, path, sequence of paths/CCD-like objects, or a table-like
        object with a ``"file"`` column. A string beginning with ``"@"`` is read as
        a newline-separated IRAF-style list file.
    sort : `bool`, optional
        Sort path results. Default: ``True``.
    accept_ccdlike : `bool`, optional
        Accept a CCD-like object as a single-item list. If ``False``, such input
        raises `TypeError`. Default: ``True``.
    path_to_text : `bool`, optional
        Convert path-like entries in a sequence to strings instead of `Path`
        objects. Default: ``False``.
    check_coherency : `bool`, optional
        Require every item in a sequence to have the type of the first item.
        Default: ``False``.

    Returns
    -------
    list or `None`
        Normalized inputs, or ``None`` when ``inputs`` is ``None``.

    Raises
    ------
    KeyError
        A table-like input has no ``"file"`` column.
    TypeError
        A CCD-like object is rejected or an unsupported input type is given.
    """
    contains_ccdlike = False
    if inputs is None:
        return None
    elif isinstance(inputs, str):
        if inputs.startswith("@"):
            with open(inputs[1:]) as ff:
                outlist = ff.read().splitlines()
        else:
            # If str, "dir/file.fits" --> [Path("dir/file.fits")]
            #         "dir/*.fits"    --> [Path("dir/file.fits"), ...]
            outlist = glob.glob(inputs)
    elif isinstance(inputs, (PosixPath, WindowsPath)):
        # If Path, ``TOP/"file*.fits"`` --> [Path("top/file1.fits"), ...]
        outlist = glob.glob(str(inputs))
    elif isinstance(inputs, HDULike_types):
        if accept_ccdlike:
            outlist = [inputs]
        else:
            raise TypeError(
                f"{type(inputs)} is given as `inputs`. "
                + "Turn off accept_ccdlike or use path-like."
            )
    elif isinstance(inputs, (Table, dict, pd.DataFrame)):
        # Do this before is_list_like because DataFrame returns True in
        # is_list_like as it is iterable.
        try:
            outlist = list(inputs["file"])
        except KeyError as err:
            raise KeyError(
                "If inputs is DataFrame convertible, it must have column named 'file'."
            ) from err
    elif is_list_like(inputs):
        type_ref = type(inputs[0])
        outlist = []
        for i, item in enumerate(inputs):
            if check_coherency and not isinstance(item, type_ref):
                raise TypeError(
                    f"The 0-th item has {type_ref} while {i}-th has {type(item)}."
                )
            if isinstance(item, HDULike_types):
                contains_ccdlike = True
                if accept_ccdlike:
                    outlist.append(item)
                else:
                    raise TypeError(
                        f"{type(item)} is given in the {i}-th element. "
                        + "Turn off accept_ccdlike or use path-like."
                    )
            else:  # assume it is path-like
                if path_to_text:
                    outlist.append(str(item))
                else:
                    outlist.append(Path(item))
    else:
        raise TypeError(f"inputs type ({type(inputs)})not accepted.")

    if sort and not contains_ccdlike:
        outlist.sort()

    return outlist
