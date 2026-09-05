from pathlib import Path

import imcombiners as imc
import numpy as np
import numpy.typing as npt
import pandas as pd
from astro_ndslice import is_list_like, listify, offseted_shape
from astropy.nddata import CCDData
from astropy.time import Time

from astroimred._core.astropy_helpers import str_now
from astroimred._core.types import HDUExt, SectionLike, StrPathLike
from astroimred.fitsmgmt.header import cmt2hdr
from astroimred.fitsmgmt.io import _parse_extension, inputs2list, load_ccd
from astroimred.fitsmgmt.table import SummaryInput, fits_summary, group_fits
from astroimred.logging import logger

from . import _docstrings as docstrings
from ._util_fits import (
    apply_output_offsets,
    calculate_zsw,
    check_stack_memory,
    extract_stack_metadata,
    load_full_stack,
    load_stack_chunk,
    log_zsw_table,
    update_hdr,
    write_imcombine_outputs,
)

__all__ = ["group_combine", "group_save", "imcombine"]

"""
removed : headers, project, masktype, maskvalue, sigscale, grow
partial removal:
    * combine in ["quadrature", "nmodel"]
replaced
    * reject in ["crreject", "avsigclip"] --> ccdclip with certain params
    * offsets in ["grid", <filename>]  --> offsets in `~numpy.ndarray`

bpmasks                : ?
rejmask                : output_mask
nrejmasks              : output_nrej
expmasks               : Should I implement???
sigma                  : output_std
outtype                : dtype
outlimits              : trimsec
expname                : exposure_key

# ALGORITHM PARAMETERS ====================================================== #
lthreshold, hthreshold : thresholds (`tuple`)
nlow      , nhigh      : n_minmax (`tuple`)
nkeep                  : nkeep & maxrej
                        (IRAF nkeep > 0 && < 0 case, resp.)
mclip                  : cenfunc
lsigma    , hsigma     : sigma uple
"""


def group_combine(
    inputs: SummaryInput,
    type_key: str | list[str] | None = None,
    type_val: object = None,
    group_key: str | list[str] | None = None,
    fmt: str | None = None,
    outdir: StrPathLike | None = None,
    verbose: int = 1,
    **kwargs,
) -> dict[tuple, CCDData]:
    """Combine sub-groups of FITS files from the given input.

    Parameters
    ----------
    inputs : `~pandas.DataFrame`, glob pattern, `list`-like of path-like
        If `DataFrame`, it must be the summary table made by ``fm.fits_summary``.
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list` of
        files (each element must be path-like or `~astropy.nddata.CCDData`).
        Although it is not a good idea, a mixed `list` of
        `~astropy.nddata.CCDData` and paths to the files is also acceptable.
        For the purpose of ``imred.imcombine``, the best use is to use the
        `~glob` pattern or `list` of paths.

    type_key, type_val : `str`, `list` of `str`
        The header keyword for the ccd type, and the value you want to match.

    group_key : `None`, `str`, `list` of `str`, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.
        Default: `None`.

    verbose : `int`, optional.
        Larger number means it becomes more verbose:

        * 0: print nothing
        * 1: only essential messages from this function
        * 2: also pass verbose mode to ``imred.imcombine``

        Default: ``1``.

    fmt : `str`, optional
        The f-string for the output file names.

        Example: if `group_key="EXPTIME"` and there are two groups where
          ``EXPTIME`` is 1.0 and 2.0,

        * ``"dark_{:.1f}s"`` gives ``dark_1.0s.fits`` and ``dark_2.0s.fits``.
        * For `float`, non-specification such as ``"d{}"`` is not recommended
          because filenames can contain long floating-point representations.

        If two `group_key` values are used, resulting in ``("B", 2.0)``,
        ``("V", 12.0)``, ...:

        * ``"flat_{2:04.1f}_{1:s}"`` gives ``"flat_02.0_B.fits"`` and
          ``"flat_12.0_V.fits"``.

        Default: `None`.

    outdir : path-like, optional
        The directory where the output fits files will be saved.

    **kwargs :
        The keyword arguments for `imcombine`.
        Default: `None`.

    Returns
    -------
    combined : `dict` of `~astropy.nddata.CCDData`
        The `dict` object where keys are the header value of the `group_key`
        and the values are the combined images in `~astropy.nddata.CCDData`
        object. If multiple keys for `group_key` is given, the key of this
        `dict` is a `tuple`.
    """

    def _group_save(ccd, groupname, fmt=None, verbose=1, outdir=None):
        """Saves the results."""
        outdir = Path(".") if outdir is None else Path(outdir)
        if verbose >= 1 and not outdir.exists():
            logger.info(
                "Output directory: '%s' <- does not exist! It will be newly made.",
                outdir,
            )

        outdir.mkdir(exist_ok=True, parents=True)

        if fmt is None:
            nk = len(group_key) if is_list_like(group_key) else 1  # 1 if str
            fmt = "_".join(["{}"] * nk)
            if verbose >= 1:
                logger.warning("fmt is not specified! Output file names might be ugly.")

        if isinstance(groupname, tuple):
            fname = fmt.format(*groupname) + ".fits"
        else:
            fname = fmt.format(groupname) + ".fits"

        fname = fname.replace(".fits.fits", ".fits")

        fpath = outdir / fname
        if verbose >= 1:
            if fpath.exists():
                logger.info("%s will be overridden.", fpath)
            else:
                logger.info("%s", fpath)
        ccd.write(fpath, overwrite=True)

    _t = Time.now()

    if isinstance(inputs, pd.DataFrame):
        load_fits = True
        summary = inputs.copy()
    elif isinstance(inputs, str):  # glob pattern
        load_fits = True
        summary = fits_summary(inputs)
    else:
        inputs = listify(inputs)
        load_fits = not isinstance(inputs[0], CCDData)
        # Assume all are CCDData if the first element is CCDData
        summary = fits_summary(inputs)

    gs, gt_key = group_fits(
        summary, type_key=type_key, type_val=type_val, group_key=group_key
    )
    if verbose >= 1:
        logger.info("Group and combine by %s (total %d groups)", group_key, len(gs))

    combined = {}
    for g_val, group in gs:
        if is_list_like(g_val) and len(g_val) == 1:
            g_val = g_val[0]
        files = group["file"].to_list()
        if verbose >= 1:
            logger.info("* %s... (%d files)", g_val, len(files))
        if len(files) == 0:
            if verbose >= 1:
                logger.info("No FITS to combine.")
            combined[g_val] = None
        elif len(files) == 1:
            if verbose >= 1:
                logger.info(
                    "Only 1 FITS to combine -- returning it without any modification."
                )
            combined[g_val] = load_ccd(files[0]) if load_fits else inputs[0]
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)
        else:
            combined[g_val] = imcombine(
                files if load_fits else inputs,
                verbose=verbose >= 2,
                full=False,
                **kwargs,
            )
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)

    if verbose >= 1:
        logger.info(str_now(t_ref=_t))

    return combined


def group_save(
    combined: dict,
    fmt: str = "",
    verbose: int = 1,
    outdir: StrPathLike | None = None,
) -> None:
    """Saves the group_combine results.
    Parameters
    ---------
    combined : `dict`
        The result from `group_combine` function.
    """
    outdir = Path(".") if outdir is None else Path(outdir)
    if verbose and not outdir.exists():
        logger.info(
            "Output directory: '%s' <- does not exist! It will be newly made.", outdir
        )

    outdir.mkdir(exist_ok=True, parents=True)

    if not fmt:
        fmt = "_".join(["{}"] * len(list(combined.keys())[0]))
        if verbose:
            logger.warning("fmt is not specified! Output file names might be ugly.")

    for k, ccd in combined.items():
        if isinstance(k, tuple):
            fname = fmt.format(*k) + ".fits"
        else:
            fname = fmt.format(k) + ".fits"
        fpath = outdir / fname
        if verbose >= 1 and fpath.exists():
            logger.info("The pre-existing file %s will be overridden.", fpath)
        ccd.write(fpath, overwrite=True)


def _mask_total_from_parts(
    input_mask: np.ndarray,
    mask_rej: np.ndarray | None,
    mask_thresh: np.ndarray | None,
) -> np.ndarray:
    """Return the total per-sample mask used for diagnostic output products."""
    mask_total = np.asarray(input_mask, dtype=bool).copy()
    if mask_thresh is not None:
        mask_total |= mask_thresh
    if mask_rej is not None:
        mask_total |= mask_rej
    return mask_total


def _set_int_dtype(ncombine: int) -> type[np.integer]:
    if ncombine < 255:
        return np.uint8
    if ncombine > 65535:
        return np.uint32
    return np.uint16


def _set_reject_name(reject: str | None) -> str | None:
    if reject is None:
        return None

    reject_key = reject.lower()
    if reject_key in {"none", "no", "null"}:
        return None
    if reject_key in {"sig", "sc", "sigclip", "sigma", "sigma clip", "sigmaclip"}:
        return "sigclip"
    if reject_key in {"mm", "minmax"}:
        return "minmax"
    if reject_key in {"ccd", "ccdclip", "ccdc"}:
        return "ccdclip"
    if reject_key in {"pclip", "pc", "percentile"}:
        return "pclip"
    raise ValueError("reject not understood.")


def _as_scalar_ccdclip_parameter(name: str, value: str | npt.ArrayLike | None) -> float:
    """Return a scalar CCD clipping parameter accepted by `imcombiners`."""
    arr = np.asarray(0.0 if value is None else value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    first = float(arr[0])
    if not np.allclose(arr, first, rtol=0.0, atol=0.0):
        raise ValueError(
            f"imcombiners ccdclip currently accepts scalar {name}; "
            f"got per-image values {arr.tolist()}."
        )
    return first


def _subset_plane_values_for_active_chunk(
    values: npt.ArrayLike,
    active_indices: np.ndarray,
    *,
    reference_to_0th: bool,
    operation: str,
) -> np.ndarray:
    """Return active per-plane values while preserving image-0 rebasing."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Per-plane values must not be empty.")
    if reference_to_0th:
        if operation == "subtract":
            arr = arr - arr[0]
        elif operation == "divide":
            arr = arr / arr[0]
        else:
            raise ValueError(f"Unknown plane-value operation: {operation}")
    return arr[active_indices]


def _ndckw_for_active_chunk(
    ndc_kw: dict[str, object], active_indices: np.ndarray
) -> dict[str, object]:
    """Subset plane-wise kwargs for a compact chunk stack."""
    chunk_kw = dict(ndc_kw)
    chunk_kw["zero"] = _subset_plane_values_for_active_chunk(
        ndc_kw["zero"],
        active_indices,
        reference_to_0th=bool(ndc_kw["zero_to_0th"]),
        operation="subtract",
    )
    chunk_kw["scale"] = _subset_plane_values_for_active_chunk(
        ndc_kw["scale"],
        active_indices,
        reference_to_0th=bool(ndc_kw["scale_to_0th"]),
        operation="divide",
    )
    if ndc_kw["weight"] is not None:
        chunk_kw["weight"] = np.asarray(ndc_kw["weight"], dtype=float).reshape(-1)[
            active_indices
        ]
    chunk_kw["zero_to_0th"] = False
    chunk_kw["scale_to_0th"] = False
    return chunk_kw


def imcombine(
    inputs: SummaryInput,
    mask: np.ndarray | None = None,
    extension: HDUExt = None,
    extension_uncertainty: HDUExt = None,
    extension_mask: HDUExt = None,
    uncertainty_type: str = "stddev",
    trimsec: SectionLike = None,
    blank: float = np.nan,
    offsets: str | npt.ArrayLike | None = None,
    thresholds: tuple[float, float] | list[float] | None = None,
    zero: str | npt.ArrayLike | None = None,
    zero_to_0th: bool = True,
    zero_section: SectionLike = None,
    scale: str | npt.ArrayLike | None = None,
    scale_to_0th: bool = True,
    scale_section: SectionLike = None,
    zero_kw: dict[str, object] | None = None,
    scale_kw: dict[str, object] | None = None,
    weight: str | npt.ArrayLike | None = None,
    reject: str | None = None,
    sigma: float | tuple[float, float] | list[float] | None = None,
    cenfunc: str = "median",
    maxiters: int = 50,
    ddof: int = 1,
    nkeep: int = 1,
    maxrej: int | None = None,
    n_minmax: tuple[int, int] | list[int] | None = None,
    rdnoise: str | npt.ArrayLike | None = 0.0,
    gain: str | npt.ArrayLike | None = 1.0,
    snoise: str | npt.ArrayLike | None = 0.0,
    pclip: float = -0.5,
    combine: str = "average",
    dtype: str = "float32",
    dtype_std: str = "float32",
    dtype_low: str | None = None,
    dtype_upp: str | None = None,
    memlimit: float | None = 2.5e9,
    verbose: bool = False,
    full: bool = False,
    imcmb_key: str = "$I",
    exposure_key: str = "EXPTIME",
    output: StrPathLike | None = None,
    output_mask: StrPathLike | None = None,
    output_nrej: StrPathLike | None = None,
    output_std: StrPathLike | None = None,
    output_low: StrPathLike | None = None,
    output_upp: StrPathLike | None = None,
    output_flags: StrPathLike | None = None,
    return_dict: bool = False,
    output_verify: str = "exception",
    overwrite: bool = False,
    checksum: bool = False,
) -> CCDData | dict:
    if extension_uncertainty is not None:
        raise NotImplementedError(
            "extension_uncertainty is not supported: imcombine does not propagate "
            "input uncertainties. Use extension_uncertainty=None for data-only "
            "combination; output_std describes clipping spread, not propagated error."
        )

    if uncertainty_type != "stddev":
        raise NotImplementedError(
            "uncertainty_type is not supported: imcombine does not propagate "
            "input uncertainties. Leave uncertainty_type='stddev'."
        )

    if combine.lower() in {"weighted_average", "wvg"}:
        combine = "average"
    if weight is not None and combine.lower() not in {"average", "mean", "avg"}:
        raise ValueError("weight can only be used with mean combination.")

    # === 1. Normalize defaults that must not use mutable signature values ===
    thresholds = None if thresholds is None else list(thresholds)
    zero_kw = None if zero_kw is None else dict(zero_kw)
    scale_kw = None if scale_kw is None else dict(scale_kw)
    sigma = [3.0, 3.0] if sigma is None else sigma
    n_minmax = [1, 1] if n_minmax is None else n_minmax

    if verbose:
        _t1 = Time.now()
        logger.info(_t1.iso)
        logger.info("- Organizing...")

    # === 2. Organize inputs and output mode ===
    full = (
        full
        or output_mask is not None
        or output_nrej is not None
        or output_std is not None
        or output_low is not None
        or output_upp is not None
        or output_flags is not None
    )

    items = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=True)
    ncombine = len(items)
    rejname = _set_reject_name(reject)
    int_dtype = _set_int_dtype(ncombine)
    extension = _parse_extension(extension)
    # If extensions are given as `None`, don't parse them and leave it as `None`.
    e_u = (
        None
        if extension_uncertainty is None
        else _parse_extension(extension_uncertainty)
    )
    e_m = None if extension_mask is None else _parse_extension(extension_mask)

    # === 3. Read only the metadata needed to plan the full output stack ===
    metadata = extract_stack_metadata(
        items=items,
        ncombine=ncombine,
        extension=extension,
        trimsec=trimsec,
        imcmb_key=imcmb_key,
        scale=scale,
        exposure_key=exposure_key,
        reject_fullname=rejname,
        gain=gain,
        rdnoise=rdnoise,
        snoise=snoise,
        dtype=dtype,
        offsets=offsets,
    )
    hdr0 = metadata["hdr0"]
    ndim = metadata["ndim"]
    shapes = metadata["shapes"]
    raw_shapes = metadata["raw_shapes"]
    offsets = metadata["offsets"]
    offset_mode = metadata["offset_mode"]
    use_wcs = metadata["use_wcs"]
    use_phy = metadata["use_phy"]
    imcmb_val = metadata["imcmb_val"]
    extract_exptime = metadata["extract_exptime"]
    scales = metadata["scales"]
    gns = metadata["gns"]
    rds = metadata["rds"]
    sns = metadata["sns"]

    # == Check the size of the temporary array for combination ======================= #
    offsets, sh_comb = offseted_shape(
        shapes, offsets, method="outer", offset_order_xyz=False, intify_offsets=True
    )
    # Rejection counts and diagnostics depend on the full input-plane axis,
    # including NaNs where offset images do not cover an output pixel.
    compact_chunks = rejname is None

    mem_req, num_chunk, chunks = check_stack_memory(
        ncombine=ncombine,
        sh_comb=sh_comb,
        dtype=dtype,
        combine=combine,
        memlimit=memlimit,
        offsets=offsets if compact_chunks else None,
        shapes=shapes if compact_chunks else None,
        full=full,
        reject=rejname,
        thresholds=thresholds is not None,
    )
    if verbose:
        logger.info("Done.")
        if num_chunk > 1:
            logger.info("memlimit reached: Split combine by %d chunks.", num_chunk)

    if verbose:
        logger.info("- Loading, calculating offsets with zero/scale...")

    _t = Time.now()

    if num_chunk == 1:
        # == Setup offset-ed array =================================================== #
        # NOTE: Using NaN does not set array with dtype of int... Any solution?
        arr_full, mask_full, var_full, zeros, scales, weights = load_full_stack(
            items=items,
            offsets=offsets,
            shapes=shapes,
            sh_comb=sh_comb,
            dtype=dtype,
            mask=mask,
            trimsec=trimsec,
            extension=extension,
            extension_mask=e_m,
            extension_uncertainty=e_u,
            extract_exptime=extract_exptime,
            scale=scale,
            zero=zero,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section,
            scale_section=scale_section,
            scales=scales,
        )
    else:
        zeros, scales, weights = calculate_zsw(
            items=items,
            dtype=dtype,
            trimsec=trimsec,
            extension=extension,
            extension_mask=e_m,
            extension_uncertainty=e_u,
            extract_exptime=extract_exptime,
            scale=scale,
            zero=zero,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section,
            scale_section=scale_section,
            scales=scales,
        )

    log_zsw_table(items, zeros, scales, weights, verbose)
    # -------------------------------------------------------------------------------- #

    cmt2hdr(
        hdr0,
        "h",
        t_ref=_t,
        verbose=verbose,
        s=f"Loaded {ncombine} FITS, calculated zero, scale, weights",
    )

    ndc_kw = {
        "combine": combine,
        "reject": rejname,
        "scale": scales,  # it is scales , NOT scale , as it was updated above.
        "zero": zeros,  # it is zeros  , NOT zero  , as it was updated above.
        "weight": weights if weight is not None else None,
        "zero_to_0th": zero_to_0th,
        "scale_to_0th": scale_to_0th,
        "scale_sigclip_kwargs": scale_kw,
        "zero_sigclip_kwargs": zero_kw,
        "thresholds": thresholds,
        "n_minmax": n_minmax,
        "nkeep": nkeep,
        "maxrej": maxrej,
        "cenfunc": cenfunc,
        "sigma": sigma,
        "maxiters": maxiters,
        "ddof": ddof,
        "rdnoise": rds,  # it is rds, not rdnoise, as it was updated above.
        "gain": gns,  # it is gns, not gain   , as it was updated above.
        "snoise": sns,  # it is sns, not snoise , as it was updated above.
        "pclip": pclip,
        "diagnostics": "simple" if full else None,
    }
    if rejname == "ccdclip":
        ndc_kw["rdnoise"] = _as_scalar_ccdclip_parameter("rdnoise", rds)
        ndc_kw["gain"] = _as_scalar_ccdclip_parameter("gain", gns)
        ndc_kw["snoise"] = _as_scalar_ccdclip_parameter("snoise", sns)

    # == Combine with rejection! ===================================================== #
    _t = Time.now()

    if num_chunk == 1:
        comb = imc.ndcombine(arr=arr_full, mask=mask_full, **ndc_kw)

        if full:  # unpack the output
            comb, mask_rej, mask_thresh, std, low, upp, nit, output_flags_data = comb
            mask_total = _mask_total_from_parts(mask_full, mask_rej, mask_thresh)
        else:
            std = low = upp = mask_total = output_flags_data = None
    else:
        if verbose:
            logger.info("- Combining by %d chunks", num_chunk)

        comb = np.empty(sh_comb, dtype=dtype)
        std = mask_total = mask_rej = mask_thresh = low = upp = nit = None
        output_flags_data = None

        for i_chunk, chunk_slices in enumerate(chunks, start=1):
            if verbose:
                logger.info("-- chunk %d/%d: %s", i_chunk, num_chunk, chunk_slices)

            arr_i, mask_i, var_i, active_indices = load_stack_chunk(
                items=items,
                offsets=offsets,
                shapes=shapes,
                raw_shapes=raw_shapes,
                chunk_slices=chunk_slices,
                dtype=dtype,
                mask=mask,
                trimsec=trimsec,
                extension=extension,
                extension_mask=e_m,
                extension_uncertainty=e_u,
                compact=compact_chunks,
            )

            if active_indices.size == 0:
                # Use the same array cast as the final full-stack output.
                # Assigning a Python NaN directly to integer arrays raises.
                # An empty sum is zero; other combinations return NaN.
                comb[chunk_slices] = np.asarray(
                    0.0 if combine.lower() == "sum" else np.nan
                ).astype(dtype)
                if full and mask_total is None:
                    mask_total = np.zeros((ncombine, *sh_comb), dtype=bool)
                del arr_i, mask_i, var_i
                continue

            chunk_kw = _ndckw_for_active_chunk(ndc_kw, active_indices)
            combined_i = imc.ndcombine(arr=arr_i, mask=mask_i, **chunk_kw)

            if full:
                comb_i = combined_i[0]
                mask_rej_i = combined_i[1]
                mask_thresh_i = combined_i[2]
                std_i = combined_i[3]
                low_i = combined_i[4]
                upp_i = combined_i[5]
                nit_i = combined_i[6]
                output_flags_i = combined_i[7]
                mask_total_i = _mask_total_from_parts(mask_i, mask_rej_i, mask_thresh_i)
                mask_slices = (slice(None), *chunk_slices)

                if mask_total is None:
                    mask_shape = (ncombine, *sh_comb)
                    mask_total = np.zeros(mask_shape, dtype=bool)
                if std_i is not None and std is None:
                    std = np.full(sh_comb, np.nan, dtype=std_i.dtype)
                if mask_rej_i is not None and mask_rej is None:
                    mask_rej = np.zeros((ncombine, *sh_comb), dtype=bool)
                if mask_thresh_i is not None and mask_thresh is None:
                    mask_thresh = np.zeros((ncombine, *sh_comb), dtype=bool)
                if low_i is not None and low is None:
                    low = np.full(sh_comb, np.nan, dtype=low_i.dtype)
                if upp_i is not None and upp is None:
                    upp = np.full(sh_comb, np.nan, dtype=upp_i.dtype)
                if nit_i is not None and nit is None:
                    nit = np.zeros(sh_comb, dtype=np.asarray(nit_i).dtype)
                if output_flags_i is not None and output_flags_data is None:
                    output_flags_data = np.zeros(
                        sh_comb, dtype=np.asarray(output_flags_i).dtype
                    )

                comb[chunk_slices] = comb_i
                if std_i is not None:
                    std[chunk_slices] = std_i
                mask_total[mask_slices] = False
                mask_total[(active_indices, *chunk_slices)] = mask_total_i
                if mask_rej_i is not None:
                    mask_rej[mask_slices] = False
                    mask_rej[(active_indices, *chunk_slices)] = mask_rej_i
                if mask_thresh_i is not None:
                    mask_thresh[mask_slices] = False
                    mask_thresh[(active_indices, *chunk_slices)] = mask_thresh_i
                if low_i is not None:
                    low[chunk_slices] = low_i
                if upp_i is not None:
                    upp[chunk_slices] = upp_i
                if nit_i is not None:
                    nit[chunk_slices] = nit_i
                if output_flags_i is not None:
                    output_flags_data[chunk_slices] = output_flags_i
            else:
                comb[chunk_slices] = combined_i

            # Release chunk outputs before the next input chunk is loaded.
            # Otherwise the previous tuple and its unpacked views stay resident.
            if full:
                del (
                    comb_i,
                    mask_rej_i,
                    mask_thresh_i,
                    std_i,
                    low_i,
                    upp_i,
                    nit_i,
                    output_flags_i,
                    mask_total_i,
                )
            del combined_i, arr_i, mask_i, var_i

        if not full:
            std = low = upp = mask_total = output_flags_data = None

    # == Update header properly ====================================================== #
    # Update WCS or PHYSICAL keywords so that "lock frame wcs", etc, on SAO
    # ds9, for example, to give proper visualization:
    apply_output_offsets(hdr0, ndim, offsets, use_wcs, use_phy)

    update_hdr(
        hdr0,
        ncombine,
        imcmb_key=imcmb_key,
        imcmb_val=imcmb_val,
        offset_mode=offset_mode,
        offsets=offsets,
        zeros=zeros,
        scales=scales,
        weights=weights,
    )

    try:
        unit = hdr0["BUNIT"].lower()
    except (KeyError, IndexError):
        unit = "adu"

    cmt2hdr(hdr0, "h", t_ref=_t, verbose=verbose, s="Rejection and combination done")
    comb = comb.astype(dtype, copy=False)
    comb = CCDData(data=comb, header=hdr0, unit=unit)

    if verbose:
        logger.info("- Writing output FITS...")

    # == Save FITS files ============================================================= #
    write_imcombine_outputs(
        comb=comb,
        hdr0=hdr0,
        output=output,
        output_std=output_std,
        output_low=output_low,
        output_upp=output_upp,
        output_nrej=output_nrej,
        output_mask=output_mask,
        output_flags=output_flags,
        std=std,
        low=low,
        upp=upp,
        mask_total=mask_total,
        output_flags_data=output_flags_data,
        int_dtype=int_dtype,
        dtype=dtype,
        dtype_std=dtype_std,
        dtype_low=dtype_low,
        dtype_upp=dtype_upp,
        output_verify=output_verify,
        overwrite=overwrite,
        checksum=checksum,
    )

    if verbose:
        logger.info("Done.")

    # == Return memory... ============================================================ #
    if num_chunk == 1:
        del arr_full, mask_full
    del hdr0

    if verbose:
        _t2 = Time.now()
        logger.info("")
        logger.info("%s (TOTAL dt = %.3f sec)", _t2.iso, (_t2 - _t1).sec)

    # == Return ====================================================================== #
    if full:
        if return_dict:
            return {
                "comb": comb,
                "mask_total": mask_total,
                "mask_rej": mask_rej,
                "mask_thresh": mask_thresh,
                "std": std,
                "low": low,
                "upp": upp,
                "nit": nit,
                "output_flags": output_flags_data,
            }
        else:
            return (
                comb,
                mask_rej,
                mask_thresh,
                std,
                low,
                upp,
                nit,
                output_flags_data,
            )
    else:
        return comb


imcombine.__doc__ = f"""A FITS-file helper for ``imcombiners.ndcombine``.

    {docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4)}

    Parameters
    ----------

    inputs : glob pattern, `list`-like of path-like, `list`-like of `~astropy.nddata.CCDData`-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list`
        of files (each element must be path-like or `~astropy.nddata.CCDData`).
        Although it is not a good idea, a mixed `list` of
        `~astropy.nddata.CCDData` and paths to the files is also acceptable.
        For the purpose of ``imred.imcombine``, the best use is to use the
        `~glob` pattern or `list` of paths.

    mask : `~numpy.ndarray`, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

        .. note::
            If the user ever want to use masking, it's more convenient to use
            ``'MASK'`` extension to the FITS files or replace bad pixel to very
            large or small numbers and use thresholds.

    extension, extension_uncertainty, extension_mask : `int`, `str`, (`str`, `int`)
        The extension of FITS, uncertainty, and mask to be used. It can be
        given as integer (0-indexing) of the extension, ``EXTNAME`` (single
        `str`), or a `tuple` of `str` and `int`: ``(EXTNAME, EXTVER)``. If
        `None` (default), the *first extension with data* will be used. If
        `extension_uncertainty` or `extension_mask` is `None` (default),
        uncertainty and mask extensions are ignored (turned off).
        A non-`None` `extension_uncertainty` raises `NotImplementedError` before
        input I/O: uncertainty propagation is not supported. Attached CCDData
        uncertainties are ignored without copying. `output_std` is clipping
        spread, not propagated measurement uncertainty.

    {docstrings.OFFSETS_LONG(indent=4)}

    {docstrings.NDCOMB_PARAMETERS_COMMON(indent=4)}

    weight : `float`, array-like, `str`, or `None`, optional
        Finite, nonzero weights for mean/average combination; signs are retained.
        Use a shared scalar, one value per image, or a statistic such as
        ``"mean"`` computed within `scale_section` on each trimmed input.
        Vectors follow sorted file-path order; CCDData lists retain input order.
        Each pixel uses only finite, unmasked, unrejected samples; `None` gives
        equal weights. No valid samples gives NaN; valid weights summing to zero
        raise `ZeroDivisionError`. Weights do not affect rejection.

    imcmb_key : `str`
        The thing to add as ``IMCMBnnn`` in the output FITS file header. If
        ``"$I"``, following the default of IRAF, the file's name will be added.
        Otherwise, it should be a header keyword. If the key does not exist in
        ``nnn``-th file, a null string will be added. If a null string
        (``imcmb_key=""``), it does not set the ``IMCMBnnn`` keywords nor
        deletes any existing keyword.

        .. warning::
            If more than 999 files are combined, only the first 999 files will
            be recorded in the header.

    exposure_key : `str`, optional.
        The header keyword which contains the information about the exposure
        time of each FITS file. This is used only if scaling is done for
        exposure time (see `scale`).

    uncertainty_type : `str`, optional
        Reserved for future uncertainty propagation. Only the default
        ``"stddev"`` is accepted; other values raise `NotImplementedError`.

    memlimit : `float` or `None`, optional
        Approximate byte budget for input data, working buffers, and outputs.
        If the estimate exceeds it, read and combine spatial chunks.
        The final image and requested diagnostics remain in memory.
        Statistical normalization may still read a full input image.
        `None` or non-positive values disable chunking.

    output : path-like, optional
        The path to the final combined FITS file. It has dtype of `dtype` and
        dimension identical to each input image. Optional keyword arguments for
        ``fits.writeto()`` can be provided as ``**kwargs``.

    output_xxx : path-like, optional
        The output path to the mask, number of rejected pixels at each
        position, ``std`` diagnostic, lower and upper bounds for rejection,
        and integer output flags (see `mask_rej`, `mask_thresh`, `std`, `low`,
        `upp`, `nit`, and `output_flags` in Returns.)

    return_dict : `bool`, optional.
        Whether to return the results as `dict` (works only if ``full=True``).

    Returns
    -------
    Returns the followings depending on `full` and `return_dict`.

    comb : `astropy.nddata.CCDData` (dtype `dtype`)
        The combined data.

    {docstrings.NDCOMB_RETURNS_COMMON(indent=4)}

    {docstrings.IMCOMBINE_LINK(indent=4)}
    """
