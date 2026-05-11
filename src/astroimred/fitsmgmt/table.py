"""FITS file summary and table-selection helpers."""

import contextlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from astro_ndslice import listify
from astropy import units as u
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData
from astropy.table import Table

from .._core.types import HDUExt, HDULike, StrPathLike
from ..logging import logger
from .header import chk_keyval
from .io import _parse_extension, inputs2list, load_ccd

__all__ = [
    "fits_summary",
    "df_selector",
    "group_fits",
    "select_fits",
]

SummaryInput = (
    StrPathLike | HDULike | pd.DataFrame | Table | Sequence[StrPathLike | HDULike]
)


def _write_summary(
    output: StrPathLike, summarytab: pd.DataFrame, verbose: bool = True
) -> None:
    """Write a summary table, choosing format from the file suffix."""
    output = Path(output)
    if verbose:
        logger.info('Saving the summary to "%s"', output)

    suffix = output.suffix.lower()
    if suffix in {".parq", ".parquet"}:
        summarytab.to_parquet(output, index=False)
    else:
        summarytab.to_csv(output, index=False)


def fits_summary(
    inputs: SummaryInput | None = None,
    extension: HDUExt = None,
    verify_fix: bool = False,
    fname_option: str = "relative",
    output: StrPathLike | None = None,
    keywords: list[str] | str | None = None,
    example_header: StrPathLike | None = None,
    sort_by: str = "file",
    sort_map: dict | None = None,
    fullmatch: dict | None = None,
    flags: int = 0,
    querystr: str | None = None,
    negate_fullmatch: bool = False,
    nonunique_keys: bool = False,
    verbose: bool = True,
    **kwargs: object,
) -> pd.DataFrame | None:
    """Extract summary rows from FITS headers.

    Parameters
    ----------
    inputs : glob pattern, `list`-like of path-like, `list`-like of `~astropy.nddata.CCDData`, `~pandas.DataFrame` convertible, optional.
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list` of
        files (each element must be path-like or `~astropy.nddata.CCDData`). Although it is not a
        good idea, a mixed `list` of `~astropy.nddata.CCDData` and paths to the files is also
        acceptable. If a `~pandas.DataFrame` or convertible (especially
        `~astropy.table.Table`) is given, it finds the ``"file"`` column and
        use it as the input files, make a summary table from the headers of
        those files.
        If `inputs` is `None`, any `output` is ignored and `None` is returned.
        Default: `None`.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    verify_fix : `bool`, optional.
        Whether to do ``.verify('fix')`` to all FITS files to avoid
        VerifyError. It may take some time if turned on. Default is `False`.

    fname_option : `str` ``{'absolute', 'relative', 'name'}``, optional
        Whether to save full absolute/relative path or only the filename.
        Default: ``'relative'``.

    output : `str` or path-like, optional
        Output summary file. ``.parq`` and ``.parquet`` use parquet; other
        suffixes use CSV.
        Default: `None`.

    keywords : `list` or `str`(``"*"``), optional
        The `list` of the keywords to extract (keywords should be in `str`).
        Default: `None`.

    example_header : `None` or path-like, optional
        The path including the filename of the output summary text file. If
        specified, the header of the 0-th element of `inputs` will be extracted
        (if glob-pattern is given, the 0-th element is random, so be careful)
        and saved to `example_header`. Use `None` (default) to skip this.
        Default: `None`.

    sort_by : `str`, optional
        The column name to sort the results. It can be any element of
        `keywords` or `'file'`, which sorts the table by the file name.
        Default: ``'file'``.

    sort_map: `dict`, optional
        A subset of `key` parameter in `pandas.DataFrame.sort_values()`. If a
        `dict` is given, then ``key = lambda x: x.map(sort_map)`` is passed into
        `.sort_values()`.
        Default: `None`.

    fullmatch : `dict`, optional
        The ``{column: regex}`` style `dict` to be used for selecting rows by
        ``summarytab[column].str.fullmatch(regex, case=True)``.
        Default: `None`

    negate_fullmatch: `bool`, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.
        Default: `False`.

    flags: `int`, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0

    querystr : `str`, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.
        Default: `None`.

    nonunique_keys : `bool`, optional
        Whether to remove the keys that have only one unique value throughout
        *ALL* input objects. Even if they are unique, keys specified in `keywords`
        will not be removed.
        Default is `False`.

    verbose : `bool`, optional
        Whether to print the progress. Default is `True`.

    **kwargs :
        The keyword arguments to be passed to `~astropy.io.fits.open`.

    Returns
    -------
    summarytab : `~pandas.DataFrame`
        Summary table with one row per input FITS file.

    Notes
    -----
    I want to use ccdproc.ImageFileCollection instead of this, but it is about
    4 times slower than my `~astroimred.fitsmgmt.table.fits_summary`, so I cannot use it yet.

    Examples
    -------

    >>> from pathlib import Path
    >>> import astroimred as air
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]
    >>> # actually it is case-insensitive
    >>> # The keywords you want to extract
    >>> # (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # list of all the fits files in TOPPATH/rawdata:
    >>> summary = air.fits_summary(
    >>>     TOPPATH/"rawdata/*.fits",
    >>>     keywords=keys,
    >>>     fname_option='name',
    >>>     sort_by="DATE-OBS",
    >>>     output=savepath
    >>> )

    Select all rows with ``OBJECT`` starts with "DA":

    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:

    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]"
    """
    if inputs is None:
        return None

    if nonunique_keys:
        summ = fits_summary(
            inputs=inputs,
            extension=extension,
            verify_fix=verify_fix,
            fname_option=fname_option,
            output=None,
            keywords=keywords,
            example_header=example_header,
            sort_by=sort_by,
            sort_map=sort_map,
            fullmatch=fullmatch,
            flags=flags,
            querystr=querystr,
            negate_fullmatch=negate_fullmatch,
            nonunique_keys=False,
            verbose=verbose,
            **kwargs,
        )
        if verbose:
            logger.info("Unique keys that will be removed:")
        for key in list(summ.columns):
            if keywords is not None and key in keywords:
                continue
            if len(_uniq := summ[key].unique()) == 1:
                if verbose:
                    logger.info(" * %-8s: %s", key, _uniq[0])
                summ.pop(key)
        if output is not None:
            _write_summary(output, summ, verbose=verbose)
        return summ

    # Although there's no need to sort here because the real "sort" will be
    # done later based on ``sort_by`` column, I did it here because the full
    # header keys will be inferred from the 0-th element (if `keywords` is not
    # given)
    fitslist = inputs2list(
        inputs, sort=True, accept_ccdlike=True, check_coherency=False
    )

    if len(fitslist) == 0:
        if verbose:
            logger.info("No FITS file found.")
        return None

    def _get_fname_fsize_hdr(item, idx, extension):
        if isinstance(item, CCDData):
            # NOTE: CCDData does not support extension (only available when it
            #   is being read)!
            fname = f"CCDData in fitslist[{idx:d}]"
            fsize = None
            hdr = item.header
        else:
            if fname_option == "relative":
                fname = str(item)
            elif fname_option == "absolute":
                fname = str(item.absolute())
            elif fname_option == "name":
                fname = item.name
            else:
                raise ValueError(f"fname_option `{fname_option}`not understood.")
            fsize = Path(item).stat().st_size
            # Don't change to MB/GB, which will make it float...
            with fits.open(item, **kwargs) as hdul:
                if verify_fix:
                    hdul.verify("fix")
                hdr = hdul[extension].header.copy()

        return fname, fsize, hdr

    skip_keys = ["COMMENT", "HISTORY"]

    if verbose and keywords is not None:
        if keywords == "*":
            logger.info("Extracting all keywords...")
        else:
            logger.info("Extracting keys: %s", keywords)

    extension = _parse_extension(extension)

    first_info = None
    if example_header is not None or keywords is None or keywords == "*":
        first_info = _get_fname_fsize_hdr(fitslist[0], 0, extension=extension)

    # Save example header
    if example_header is not None:
        fname0, _, hdr0 = first_info
        if verbose:
            logger.info("Header of 0-th: %s -> %s", fname0, example_header)
        hdr0.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords is None) or (keywords is not None and keywords == "*"):
        fname0, _, hdr0 = first_info
        num_hkeys = len(hdr0.cards)
        keywords = []

        for i in range(num_hkeys):
            try:
                key_i = hdr0.cards[i][0]
            except VerifyError as err:
                raise VerifyError("Use verify_fix=True.") from err
            if key_i in skip_keys:
                continue
            elif key_i in keywords:
                logger.warning(
                    "Key %s is duplicated! Only the first one will be saved.",
                    key_i,
                )
                continue
            keywords.append(key_i)

        if verbose:
            logger.info(
                "All %d keywords (guessed from %s) will be loaded.",
                len(keywords),
                fname0,
            )

    # Initialize
    summarytab = {"file": [], "filesize": []}
    missing_keys = set()
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for i, item in enumerate(fitslist):
        if i == 0 and first_info is not None:
            fname, fsize, hdr = first_info
        else:
            fname, fsize, hdr = _get_fname_fsize_hdr(item, i, extension=extension)
        summarytab["file"].append(fname)
        summarytab["filesize"].append(fsize)
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    str_keyerror_fill = (
                        "Key {:s} not found for {:s}, filling with None."
                    )
                    if isinstance(item, CCDData):
                        logger.warning(str_keyerror_fill.format(k, f"fitslist[{i}]"))
                    else:
                        logger.warning(str_keyerror_fill.format(k, str(item)))
                summarytab[k].append(None)
                missing_keys.add(k)

    summarytab = pd.DataFrame.from_dict(summarytab)
    summarytab = df_selector(
        summarytab,
        fullmatch=fullmatch,
        flags=flags,
        querystr=querystr,
        negate_fullmatch=negate_fullmatch,
    )
    if sort_by is not None:
        key = None if sort_map is None else lambda x: x.map(sort_map)
        summarytab.sort_values(sort_by, inplace=True, key=key)
    summarytab.reset_index(drop=True, inplace=True)
    for k in missing_keys:
        summarytab[k] = (
            summarytab[k].astype(object).where(pd.notna(summarytab[k]), None)
        )

    if output is not None:
        _write_summary(output, summarytab, verbose=verbose)

    return summarytab


def df_selector(
    summarytab: pd.DataFrame,
    fullmatch: dict | None = None,
    flags: int = 0,
    negate_fullmatch: bool = False,
    querystr: str | None = None,
    columns: str | list[str] | None = None,
    columns_drop: str | list[str] | None = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Select rows from a summary table.

    Parameters
    ----------
    summarytab : `~pandas.DataFrame`
        The summary table to select from. Normally the table made from header
        information.
    fullmatch : `dict`, optional
        The ``{column: regex}`` style `dict` to be used for selecting rows by
        ``summarytab[column].str.fullmatch(regex, case=True)``. An example:
        ``{"OBJECT": "Ves.*"}``. All corresponding columns must have dtype of
        `str` to apply regex.
        Default: `None`
    negate_fullmatch: `bool`, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.
        Default: `False`.
    flags: `int`, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0
    querystr : `str`, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.
    columns, columns_drop: `str`, `list`, optional.
        The `list` of columns to be returned/dropped after selection. No need to
        setup both, but no Error will be raised even the user does so.
        Default: `None`.

    reset_index : `bool`, optional.
        Whether to reset the DataFrame index after selection.
        Default: `True`.

    Returns
    -------
    summarytab
        The final summary table after selection. If everything is `None` (the
        default), the original summary table is returned.

    Raises
    ------
    AttributeError
        The column dtype is not `str`
    TypeError
        fullmatch must be in `dict`.

    Examples
    --------
    Select all rows with ``OBJECT`` starts with "DA":

    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:

    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]"

    """
    df = summarytab.copy()

    if fullmatch is not None:
        if not isinstance(fullmatch, dict):
            raise TypeError("fullmatch must be a dict.")

        select_mask = np.ones(len(df), dtype=bool)
        for k, v in fullmatch.items():
            try:
                select_mask &= df[k].str.fullmatch(v, flags=flags, case=True)
            except AttributeError:
                try:
                    select_mask &= df[k] == v
                except (ValueError, TypeError, AttributeError) as err:
                    raise TypeError(
                        "Both ``summarytab[k].str.fullmatch(v)`` and "
                        + f"``summarytab[{k}] == {v}`` failed.\n"
                        + "Maybe use `querystr` instead?"
                    ) from err
        df = df[~select_mask] if negate_fullmatch else df[select_mask]

    if querystr is not None:
        df = df.query(querystr)

    if columns is not None:
        df = df[listify(columns)]

    if columns_drop is not None:
        df.drop(listify(columns_drop), axis=1, inplace=True)

    if reset_index:
        df = df.reset_index(drop=True)

    return df.copy()


def group_fits(
    summary_table: pd.DataFrame | Table,
    type_key: str | list[str] | None = None,
    type_val: object = None,
    group_key: str | list[str] | None = None,
    table_filecol: str = "file",
    verbose: bool = False,
) -> tuple[pd.core.groupby.DataFrameGroupBy, list[str]]:
    """Organize the group_by and type_key for select_fits

    Parameters
    ----------
    summary_table: `~pandas.DataFrame` or `~astropy.table.Table`
        The table which contains the metadata (header) of files. If it is in
        the astropy table format, it will be converted to `~pandas.DataFrame`
        object.

    type_key, type_val: `None`, `str`, `list` of `str`, optional
        The header keyword for the ccd type, and the value you want to match.

    group_key : `None`, `str`, `list` of `str`, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.
        Default: `None`.

    Returns
    -------
    grouped : `~pandas.DataFrameGroupBy`
        The table after the grouping process.

    group_type_key : `list` of `str`
        The `type_key` that can directly be used for `select_fits` for each
        element of `grouped.groups`. Basically this is ``type_key +
        group_key``.

    Examples
    --------

    >>> allfits = list(Path('.').glob("*.fits"))
    >>> import astroimred as air
    >>> summary_table = fm.fits_summary(allfits)
    >>> type_key = ["OBJECT"]
    >>> type_val = ["dark"]
    >>> group_key = ["EXPTIME"]
    >>> gs, g_key = group_fits(summary_table,
    ...                        type_key,
    ...                        type_val,
    ...                        group_key)
    >>> for g_val, group in gs:
    >>>     _ = combine_ccd(group["file"],
    ...                     type_key=g_key,
    ...                     type_val=g_val)
    """
    if isinstance(summary_table, Table):
        st = summary_table.copy().to_pandas()
    elif isinstance(summary_table, pd.DataFrame):
        st = summary_table.copy()
    else:
        raise TypeError(
            "summary_table must be an astropy Table or Pandas DataFrame. "
            + f"It's now {type(summary_table)}."
        )

    type_key, type_val, group_key = chk_keyval(
        type_key=type_key, type_val=type_val, group_key=group_key
    )

    if len(group_key + type_key) == 0:
        raise ValueError("At least one of type_key and group_key should not be empty!")

    # For simplicity, crop the original data by type_key and type_val first.
    if type_key and type_val:  # if not empty list
        fpaths = select_fits(
            st,
            table_filecol=table_filecol,
            prefer_ccddata=False,
            type_key=type_key,
            type_val=type_val,
            verbose=verbose,
            path_to_text=True,
        )
        st = st[st[table_filecol].isin(fpaths)]
    group_type_key = type_key + group_key
    grouped = st.groupby(group_key)

    return grouped, group_type_key


def select_fits(
    inputs: SummaryInput,
    extension: HDUExt = None,
    unit: str | u.Unit | None = None,
    trimsec: str | None = None,
    table_filecol: str = "file",
    prefer_ccddata: bool = False,
    type_key: str | list[str] | None = None,
    type_val: object = None,
    path_to_text: bool = False,
    verbose: bool = True,
) -> list[Path] | list[CCDData]:
    """Stacks the FITS files specified in fitslist

    Parameters
    ----------
    inputs : path-like, `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~pandas.DataFrame` or `~astropy.table.Table`
        If it is path-like, it must contain FITS files to extract header. If
        CCD-like, the header information will be used for selecting elements to
        select.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Ignored if `inputs` is table-like.
        Default: `None`.

    unit: `~astropy.units.Unit` or `str`, optional
        The unit of the CCDs to be loaded.
        Used only when `fitslist` is not a `list` of `~astropy.nddata.CCDData`
        and `prefer_ccddata` is `True`.
        Ignored if `inputs` is table-like.
        Default: `None`.

    trimsec : `str`, [`list` of] `int`, [`list` of] slice, optional
        Section of the data to be extracted by `~astroimred.imutil.ccdops.imslice`.
        Default is `None`.
        Ignored if `inputs` is table-like.

    table_filecol : `str`, optional.
        The column name of the `summary_table` which contains the path to the
        FITS files. Ignored if `inputs` is CCD-like.
        Default: ``'file'``.

    prefer_ccddata: `bool`, optional
        Whether to prefer to return `~astropy.nddata.CCDData` objects if possible. If `True`,
        path-like, `~numpy.ndarray`, or table-like input will return a `list` of `~astropy.nddata.CCDData`.
        If `False` (default), only the paths will be returned unless the
        `inputs` is consist of `~astropy.nddata.CCDData.` Ignored if `inputs` is already
        CCD-like.

    type_key, type_val: `str`, `list` of `str`
        The header keyword for the ccd type, and the value you want to match.
        Default: `False`.

    Returns
    -------
    matched: `list` of `~pathlib.Path` or `list` of `~astropy.nddata.CCDData`
        `list` containing `~pathlib.Path` to files if `prefer_ccddata` is `False`. Otherwise
        it is a `list` containing loaded `~astropy.nddata.CCDData` after loading the files. If
        `ccdlist` is given a priori, `list` of `~astropy.nddata.CCDData` will be returned
        regardless of `prefer_ccddata`.
    """
    from astroimred.imutil.ccdops import imslice

    def _parse_val(value):
        val = str(value)
        if val.lstrip("+-").isdigit():  # if int
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    def _check_mismatch(row, keys, values):
        mismatch = False
        for k, v in zip(keys, values, strict=False):
            hdr_val = _parse_val(row[k])
            parse_v = _parse_val(v)
            if hdr_val != parse_v:
                mismatch = True
                break
        return mismatch

    # Check for type_key and type_val
    type_key, type_val, _ = chk_keyval(
        type_key=type_key, type_val=type_val, group_key=None
    )
    # I made this but think it is unnecessary as all string type_val must be subject to
    # regex.. I am leaving it here just in case in the future I find it necessary.
    #   YPBach 2021-01-08 17:36:53 (KST: GMT+09:00)
    # regex : bool or list of bool optional.
    #     Whether to use regex for `type_val` matching. Default is `False`. If it
    #     is a list, it must have the identical length to `type_key` and
    #     `type_val`. An example is that you want to select ``OBJECT`` with regex
    #     of ``'NGC.*'``, but ``EXPTIME`` of ``120``, which is a numeric. Sometimes
    #     the header will have ``'120.0'``, which may not be easily selected by
    #     regex. In that case, an internal parser is easier to use to catch any
    #     numeric values that must be regarded as the same thing. A possible usage
    #     is: ``type_key=["OBJECT", "EXPTIME"], type_val=["NGC*", 120],
    #     regex=[True, False]``.
    # if isinstance(regex, bool):
    #     regex = [regex]*len(type_key)
    # else:
    #     try:
    #         if len(regex) != len(type_key):
    #             raise ValueError("Length of regex differ from type_key and type_val.")
    #         if not all(isinstance(r, bool) for r in regex):
    #             raise TypeError("If regex is not bool, it must be list of bool.")
    #     except TypeError:
    #         raise TypeError("If regex is not bool, it must be list of bool.")

    # Setting whether we have to select a subset from the list
    selecting = len(type_key) > 0

    if verbose:
        logger.info("Analyzing FITS...")

    if isinstance(inputs, Table):
        summary_table = inputs.to_pandas()
        fitslist = summary_table[table_filecol].to_list()
    elif isinstance(inputs, pd.DataFrame):
        summary_table = inputs
        fitslist = summary_table[table_filecol].to_list()
    else:
        # No need to sort here because the real "sort" will be done later in fits_summary
        fitslist = inputs2list(
            inputs,
            sort=False,
            accept_ccdlike=True,
            check_coherency=False,
            path_to_text=path_to_text,
        )
        if selecting:
            summary_table = fits_summary(
                fitslist,
                extension=extension,
                # extension will be parsed within fits_summary (no need to care here)
                verbose=verbose,
                fname_option="relative",
                keywords=type_key,
                sort_by=None,
            )
        else:
            summary_table = None

    if summary_table is not None:
        with contextlib.suppress(ValueError):
            summary_table.reset_index(inplace=True, drop=True)

    if verbose:
        logger.info("Done.")

    # ******************************************************************************** #
    # *                             SELECT AND LOAD TO MATCHED                       * #
    # ******************************************************************************** #
    # == Do regex matching if type_val[i] is string ================================== #
    _type_key = []
    _type_val = []
    if selecting:
        for k, v in zip(type_key, type_val, strict=False):
            if isinstance(v, str):
                match_mask = summary_table[k].str.match(v)
                summary_table = summary_table[match_mask]
                fitslist = np.array(fitslist)[match_mask].tolist()
                # NOTE: Is there a better way to do this?
                with contextlib.suppress(ValueError):
                    summary_table.reset_index(inplace=True, drop=True)
            else:  # not used as regex
                _type_key.append(k)
                _type_val.append(v)
                continue  # need to do _check_mismatch below

    matched = []
    if selecting:
        # == Select FITS based on type_key and type_val ============================== #
        for i, row in summary_table.iterrows():
            # I intentionally used iterrows instead of making mask, because for
            # some cases the keyword (e.g., an angle) can contain both str and
            # float among CCDs.
            #   For example, if we want to select ``angle == 0.0``, masking
            # cannot work because the column has dtype of object
            # (``summary_table[column].dtype`` is `object``).
            #   Instead, _check_mismatch tries to convert the value found in
            # the header to int, and if it fails, tries float, and finally uses
            # str. This is the most natural way I could think of.
            # ysBach, 2020-05-15 09:44:13 (KST: GMT+09:00)
            mismatch = _check_mismatch(row, _type_key, _type_val)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            item = fitslist[i]
            if isinstance(item, CCDData):
                if trimsec is None:
                    matched.append(item)
                else:
                    matched.append(imslice(item, trimsec=trimsec))
            else:  # it must be a path to a file
                fpath = Path(item)
                if prefer_ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    matched.append(ccd_i)
                else:
                    if path_to_text:
                        matched.append(str(fpath))
                    else:
                        matched.append(fpath)
    else:
        # == Use all item in fitslist ================================================ #
        # summary_table is not used.
        for item in fitslist:
            if isinstance(item, CCDData):
                if trimsec is None:
                    matched.append(item)
                else:
                    matched.append(imslice(item, trimsec=trimsec))
            else:  # it must be a path to a file
                if prefer_ccddata:
                    # extension will be parsed within load_ccd (no need to care here)
                    ccd_i = load_ccd(item, extension=extension, unit=unit)
                    if trimsec is not None:
                        ccd_i = imslice(ccd_i, trimsec=trimsec)
                    matched.append(ccd_i)
                else:  # TODO: Is it better to remove Path here?
                    if path_to_text:
                        matched.append(str(item))
                    else:
                        matched.append(Path(item))

    # ******************************************************************************** #
    # *                           PRINT INFO MESSAGE OR WARNING                      * #
    # ******************************************************************************** #
    if len(matched) == 0:
        if selecting:
            logger.warning(
                'No FITS file had "%s = %s". Maybe int/float/str confusing?',
                type_key,
                type_val,
            )
        else:
            logger.warning("No FITS file found")
    else:
        if selecting:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if verbose:
                if prefer_ccddata:
                    logger.info('%d FITS files with "%s = %s" are loaded.', N, ks, vs)
                else:
                    logger.info('%d FITS files with "%s = %s" are selected.', N, ks, vs)
        else:
            if verbose and prefer_ccddata:
                logger.info("%d FITS files are loaded.", len(matched))

    return matched


# TODO: accept the input like ``sigma_clip_func='median'``, etc.
