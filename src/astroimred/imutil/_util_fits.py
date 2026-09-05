"""FITS I/O helpers for :func:`astroimred.imutil.imcombine`."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Any

import imcombiners as imc
import numpy as np
import numpy.typing as npt
from astro_ndslice import (
    calc_offset_physical,
    calc_offset_wcs,
    slice_from_string,
    slicefy,
)
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData
from astropy.wcs import WCS

from astroimred._core.types import HDUExt, StrPathLike
from astroimred.fitsmgmt.header import update_tlm
from astroimred.fitsmgmt.io import _parse_data_header, write2fits
from astroimred.logging import logger

Key_or_Val = str | npt.ArrayLike | None


def _normalize_stat_name(value: Any) -> Any:
    """Normalize legacy zero/scale statistic aliases to imcombiners names."""
    if not isinstance(value, str):
        return value
    aliases = {
        "avg_sc": "mean_sc",
        "average_sc": "mean_sc",
        "medi_sc": "median_sc",
    }
    return aliases.get(value.lower(), value)


def _normalize_sigclip_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return imcombiners-compatible sigma-clipped-statistic kwargs."""
    if kwargs is None:
        return None
    out = dict(kwargs)
    out.pop("axis", None)
    if "std_ddof" in out and "ddof" not in out:
        out["ddof"] = out.pop("std_ddof")
    else:
        out.pop("std_ddof", None)
    stdfunc = out.pop("stdfunc", "std")
    if stdfunc != "std":
        raise ValueError("imcombiners sigma-clipped statistics support std only.")
    return out


def _stat_workspace(arr: np.ndarray, section: str | None) -> np.ndarray:
    """Return an ``(N, M, 1)`` workspace for imcombiners plane statistics."""
    if section is None:
        selected = np.asarray(arr)
    else:
        try:
            stat_slices = slice_from_string(section, fits_convention=True)
        except (AttributeError, ValueError):
            step = int(section)
            selected = np.asarray(arr).reshape(arr.shape[0], -1)[:, ::step]
            return selected.reshape(selected.shape[0], -1, 1)
        selected = np.asarray(arr)[(slice(None), *stat_slices)]
    return selected.reshape(selected.shape[0], -1, 1)


def _resolve_plane_values(
    name: str,
    arr: np.ndarray,
    value: Key_or_Val,
    *,
    sigclip_kwargs: dict[str, Any] | None = None,
    section: str | None = None,
    nonzero: bool = False,
) -> np.ndarray:
    """Resolve scalar, vector, callable, or named-stat values via imcombiners."""
    if value is None:
        fill = 1.0 if nonzero else 0.0
        return np.full(arr.shape[0], fill, dtype=float)

    if isinstance(value, str) or callable(value):
        workspace = _stat_workspace(np.asarray(arr), section)
        # The resolver casts its result to the workspace dtype. Integer image
        # storage must not truncate fractional statistics such as the mean.
        workspace = workspace.astype(
            np.result_type(workspace.dtype, np.nan), copy=False
        )
    else:
        # Numeric calibration values are independent of the stored pixel dtype.
        workspace = np.empty((arr.shape[0], 1, 1), dtype=float)
    resolved = imc.resolve_zero_scale(
        name,
        _normalize_stat_name(value),
        workspace,
        nonzero=nonzero,
        sigclip_kwargs=_normalize_sigclip_kwargs(sigclip_kwargs),
    )
    if resolved is None:
        fill = 1.0 if nonzero else 0.0
        return np.full(arr.shape[0], fill, dtype=float)
    return np.asarray(resolved, dtype=float).reshape(-1)


def _resolve_zsw(
    arr: np.ndarray,
    *,
    zero: Key_or_Val,
    scale: Key_or_Val,
    weight: Key_or_Val,
    zero_kw: dict[str, Any] | None,
    scale_kw: dict[str, Any] | None,
    zero_section: str | None,
    scale_section: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve zero/scale/weight values with imcombiners plane statistics."""
    return (
        _resolve_plane_values(
            "zero",
            arr,
            zero,
            sigclip_kwargs=zero_kw,
            section=zero_section,
        ),
        _resolve_plane_values(
            "scale",
            arr,
            scale,
            sigclip_kwargs=scale_kw,
            section=scale_section,
            nonzero=True,
        ),
        _resolve_plane_values(
            "weight",
            arr,
            weight,
            section=scale_section,
            nonzero=True,
        ),
    )


def _image_parameter(name: str, value: Key_or_Val, index: int, ncombine: int) -> Any:
    """Select one image's numeric value; leave statistics for the image loader.

    A scalar applies to every image. A vector must have one entry per input,
    even though the statistic resolver receives only one image at a time.
    """
    if value is None or isinstance(value, str) or callable(value):
        return value
    values = np.asarray(value).reshape(-1)
    if values.size not in (1, ncombine):
        raise ValueError(f"{name} length must be 1 or match the number of images.")
    return values[0 if values.size == 1 else index]


def _expand_ccdclip_parameter(
    name: str,
    value: Key_or_Val,
    ncombine: int,
    dtype: npt.DTypeLike,
) -> tuple[bool, np.ndarray]:
    """Return whether to read a header keyword and its default vector."""
    if isinstance(value, str):
        return True, np.ones(ncombine, dtype=dtype)

    arr = np.atleast_1d(value).astype(dtype)
    if arr.size == ncombine:
        arr = arr.ravel()
    elif arr.size == 1:
        arr = np.full(ncombine, arr.item(), dtype=dtype)
    else:
        raise ValueError(f"{name} size must be 1 or equal to ncombine.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf.")
    return False, arr


def _trim_slices(trimsec: str | None, shape: Sequence[int]) -> tuple[slice, ...]:
    """Convert a FITS-style trim section to Python slices for `shape`."""
    if trimsec is None:
        return tuple(slice(None) for _ in shape)
    return tuple(slicefy(trimsec, ndim=len(shape)))


def _trimmed_shape(shape: Sequence[int], trimsec: str | None) -> tuple[int, ...]:
    """Return the data shape after applying `trimsec` to an array shape."""
    _slices = _trim_slices(trimsec, shape)
    # shape produced by applying `slices` to an array shape:
    return tuple(
        len(range(*sl.indices(int(size))))
        for size, sl in zip(shape, _slices, strict=False)
    )


def _compose_trim_data_slices(
    trimsec: str | None,
    data_slices: tuple[slice, ...],
    raw_shape: Sequence[int],
) -> tuple[slice, ...]:
    """Map chunk-local slices in trimmed coordinates back to raw data slices.

    Parameters
    ----------
    trimsec : str or None
        FITS-style section applied to the input before combination.
    data_slices : tuple of slice
        Region requested in the already-trimmed input-image coordinate system.
    raw_shape : tuple of int
        Untrimmed data shape in Python order.

    Returns
    -------
    tuple of slice
        Slices that can be applied directly to the raw FITS/CCDData image.
    """
    trim_slices = _trim_slices(trimsec, raw_shape)
    slices = []
    for raw_size, trim_slice, data_slice in zip(
        raw_shape, trim_slices, data_slices, strict=False
    ):
        t_start, _t_stop, t_step = trim_slice.indices(int(raw_size))
        if t_step <= 0:
            raise ValueError("Negative-step trimsec is not supported in chunked load.")

        trimmed_size = len(range(*trim_slice.indices(int(raw_size))))
        d_start, d_stop, d_step = data_slice.indices(trimmed_size)
        if d_step != 1:
            raise ValueError("Non-unit chunk slice steps are not supported.")
        slices.append(
            slice(t_start + d_start * t_step, t_start + d_stop * t_step, t_step)
        )
    return tuple(slices)


def _hdu_has_data(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> bool:
    """Return whether an HDU has a non-empty image data array."""
    return hdu.header.get("NAXIS", 0) > 0 and all(
        hdu.header.get(f"NAXIS{i}", 0) > 0
        for i in range(1, hdu.header.get("NAXIS", 0) + 1)
    )


def _get_image_hdu(
    hdul: fits.HDUList,
    extension: HDUExt,
) -> fits.hdu.base.ExtensionHDU | fits.PrimaryHDU | None:
    """Return the requested image HDU, falling back from primary to image HDU.

    When `extension` is 0 but the primary HDU has no data, this follows the
    package convention of using the first later HDU that contains image data.
    """
    try:
        hdu = hdul[extension]
    except (KeyError, IndexError, TypeError):
        return None

    if _hdu_has_data(hdu):
        return hdu

    if extension == 0:
        for hdu in hdul:
            if _hdu_has_data(hdu):
                return hdu
    return None


def _read_hdul_section(
    hdul: fits.HDUList,
    extension: HDUExt,
    section: tuple[slice, ...],
) -> np.ndarray | None:
    """Read a section from an HDUList without forcing a full-image load."""
    if extension is None:
        return None

    hdu = _get_image_hdu(hdul, extension)
    if hdu is None:
        return None
    return np.asarray(hdu.section[section])


def _parse_imc_data_header(
    item: Any,
    extension: HDUExt,
    parse_data: bool = True,
    parse_header: bool = True,
    copy: bool = True,
) -> tuple[np.ndarray | None, fits.Header | None]:
    """Parse data/header while applying imcombine's image-HDU fallback.

    Unrequested parts are `None`. For `CCDData`, a private header receives
    dimensions from the array, since its metadata need not contain FITS cards.
    """
    if isinstance(item, CCDData):
        data = (item.data.copy() if copy else item.data) if parse_data else None
        header = None
        if parse_header:
            header = fits.Header(item.header)
            header["NAXIS"] = item.data.ndim
            for axis, size in enumerate(reversed(item.data.shape), start=1):
                header[f"NAXIS{axis}"] = size
        return data, header

    try:
        path = Path(item)
    except TypeError:
        return _parse_data_header(
            item,
            extension=extension,
            parse_data=parse_data,
            parse_header=parse_header,
            copy=copy,
        )

    if not (parse_data or parse_header):
        return None, None

    with fits.open(path, memmap=False) as hdul:
        hdu = _get_image_hdu(hdul, extension)
        if hdu is None:
            raise ValueError(f"No image data found in {path}.")
        data = None
        if parse_data:
            data = hdu.data.copy() if copy else hdu.data
        header = None
        if parse_header:
            header = hdu.header.copy() if copy else hdu.header
    return data, header


def update_hdr(
    header: fits.Header,
    ncombine: int,
    imcmb_key: str | None,
    imcmb_val: Sequence[Any],
    offset_mode: str | None = None,
    offsets: np.ndarray | None = None,
    zeros: npt.ArrayLike | None = None,
    scales: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> None:
    """Update an imcombine output header in place.

    Adds the number of combined images, optional ``IMCMBnnn`` provenance,
    offset mode/values, zero/scale/weight summaries, and an IRAF-like TLM
    timestamp. Existing numbered cards with the same base names are replaced.
    """

    def __rm_and_add(
        hdr: fits.Header, keybase: str, values: Sequence[Any] | np.ndarray
    ) -> None:
        for i in range(999):
            if f"{keybase}{i + 1:03d}" in hdr:
                del hdr[f"{keybase}{i + 1:03d}"]
            else:
                break

        for i in range(min(999, len(values))):
            hdr[f"{keybase}{i + 1:03d}"] = values[i]

        return

    header["NCOMBINE"] = (ncombine, "Number of combined images")
    if imcmb_key != "":
        header["IMCMBKEY"] = (imcmb_key, "Key used in IMCMBiii ('$I': filepath)")
        __rm_and_add(header, "IMCMB", imcmb_val)
        # remove header keyword IMCMBiii if it exists:
        for i in range(999):
            if f"IMCMB{i + 1:03d}" in header:
                del header[f"IMCMB{i + 1:03d}"]
            else:
                break

        for i in range(min(999, len(imcmb_val))):
            header[f"IMCMB{i + 1:03d}"] = imcmb_val[i]

    if offset_mode is not None:
        if offsets is None:
            raise ValueError("offsets is required when offset_mode is set.")
        header["OFFSTMOD"] = (offset_mode, "Offset method used for combine.")
        for i in range(min(999, len(imcmb_val))):
            header[f"OFFST{i:03d}"] = str(offsets[i,][::-1].tolist())

    if zeros is not None and not np.all(zeros == 0):
        __rm_and_add(header, "ZERO", np.atleast_1d(zeros))

    if scales is not None and not np.all(scales == 1):
        __rm_and_add(header, "SCALE", np.atleast_1d(scales))

    if weights is not None and not np.all(weights == 1):
        __rm_and_add(header, "WEIGH", np.atleast_1d(weights))

    # Add "IRAF-TLM" like header key for continuity with IRAF.
    update_tlm(header)


def setup_offsets(
    offsets: str | npt.ArrayLike | None,
    ncombine: int,
    ndim: int,
    hdr0: fits.Header,
) -> tuple[np.ndarray, str | None, bool, bool, WCS | None]:
    """Normalize the requested offset mode and allocate the offset array.

    Parameters
    ----------
    offsets : None, str, or array-like
        User input passed to ``imcombine(offsets=...)``. String modes currently
        include WCS/world and physical/LTV offsets.
    ncombine, ndim : int
        Number of images and dimensionality of the data.
    hdr0 : `~astropy.io.fits.Header`
        Header of the first image, used as the WCS reference when needed.

    Returns
    -------
    offsets : ndarray
        Raw offsets in Python axis order, one row per input image.
    offset_mode : str or None
        Label written to the output header/log.
    use_wcs, use_phy : bool
        Flags for later metadata extraction and output-header updates.
    w_ref : `~astropy.wcs.WCS` or None
        Reference WCS for WCS-derived offsets.
    """
    use_wcs, use_phy = False, False
    w_ref = None

    if isinstance(offsets, str):
        if offsets.lower() in ["world", "wcs"]:
            w_ref = WCS(hdr0)
            use_wcs = True
            offset_mode = "WCS"
            offsets = np.zeros((ncombine, ndim))
        elif offsets.lower() in ["physical", "phys", "phy"]:
            use_phy = True
            offset_mode = "Physical"
            offsets = np.zeros((ncombine, ndim))
        else:
            raise ValueError("offsets not understood.")
    elif offsets is None:
        offset_mode = None
        offsets = np.zeros((ncombine, ndim))
    else:
        offsets = np.asarray(offsets)
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")
        offset_mode = "User"

    return offsets, offset_mode, use_wcs, use_phy, w_ref


def extract_stack_metadata(
    items: Sequence[Any],
    ncombine: int,
    extension: HDUExt,
    trimsec: str | None,
    imcmb_key: str | None,
    scale: Key_or_Val,
    exposure_key: str,
    reject_fullname: str | None,
    gain: Key_or_Val,
    rdnoise: Key_or_Val,
    snoise: Key_or_Val,
    dtype: npt.DTypeLike,
    offsets: str | npt.ArrayLike | None,
) -> dict[str, Any]:
    """Collect headers, shapes, offsets, and calibration metadata.

    This is the metadata-only prepass for ``imcombine``. It determines the
    dimensionality, raw and trimmed image shapes, requested offset convention,
    exposure scaling, and CCD-noise keywords needed for ``ccdclip``. FITS paths
    are inspected through headers without loading image pixels.

    Returns
    -------
    dict
        Metadata consumed by the full-stack and chunked loading paths.
    """
    # === Extract header info ======================================================== #
    # TODO: if offsets is None and `fsize_tot` << memlimit, why not
    # just load all data here?
    hdr0 = _parse_imc_data_header(items[0], extension=extension, parse_data=False)[1]
    if hdr0 is None:
        raise ValueError("Could not read header from the first input image.")
    ndim = hdr0["NAXIS"]
    # N x ndim. sizes[i, :] = images[i].shape
    shapes = np.ones((ncombine, ndim), dtype=int)
    raw_shapes = np.ones((ncombine, ndim), dtype=int)
    extract_hdr = imcmb_key not in [None, "", "$I"]

    extract_exptime = False
    if isinstance(scale, str) and scale.lower() in [
        "exp",
        "expos",
        "exposure",
        "exptime",
    ]:
        extract_exptime = True

    # === 1. Determine which calibration keywords are needed for rejection ===
    if reject_fullname == "ccdclip":
        extract_gain, gns = _expand_ccdclip_parameter(
            "gain", gain, ncombine, dtype=dtype
        )
        extract_rdnoise, rds = _expand_ccdclip_parameter(
            "rdnoise", rdnoise, ncombine, dtype=dtype
        )
        extract_snoise, sns = _expand_ccdclip_parameter(
            "snoise", snoise, ncombine, dtype=dtype
        )
    else:
        extract_gain, gns = False, 1
        extract_rdnoise, rds = False, 0
        extract_snoise, sns = False, 0

    # === 2. Interpret offset mode and initialize per-image offsets ===
    offsets, offset_mode, use_wcs, use_phy, w_ref = setup_offsets(
        offsets, ncombine, ndim, hdr0
    )

    scales = np.ones(shape=ncombine)
    imcmb_val = []
    extract_hdr = (
        extract_hdr
        or extract_exptime
        or extract_gain
        or extract_rdnoise
        or extract_snoise
        or use_wcs
        or use_phy
    )

    for i, item in enumerate(items):
        if extract_hdr:
            _, hdr = _parse_imc_data_header(
                item, extension=extension, parse_data=False, copy=False
            )
            if hdr is None:
                raise ValueError(f"Could not read header from input {i}.")
            if imcmb_key not in [None, ""]:
                if imcmb_key == "$I":
                    try:
                        imcmb_val.append(Path(item).name)
                    except TypeError:
                        imcmb_val.append(f"User-provided {type(item)}")
                else:
                    imcmb_val.append(hdr.get(imcmb_key, ""))

            if extract_exptime:
                scales[i] = float(hdr[exposure_key])
            if extract_gain:
                gns[i] = float(hdr[gain])
            if extract_rdnoise:
                rds[i] = float(hdr[rdnoise])
            if extract_snoise:
                sns[i] = float(hdr[snoise])

            if hdr["NAXIS"] != ndim:
                raise ValueError(
                    "All FITS files must have the identical ndim, "
                    + "though they can have different sizes."
                )

            # Update offsets if WCS or Physical should be used
            if use_wcs:
                # Code if using WCS, which may be much slower (but accurate?)
                # Find the center's pixel position in w_ref, in nearest integer value.
                offsets[i,] = calc_offset_wcs(
                    WCS(hdr),
                    w_ref,
                    intify_offset=True,
                    loc_target="center",
                    loc_reference="center",
                    order_xyz=False,
                )
                # For IRAF-like calculation, use
                #   offsets[i, ] = [hdr[f'CRPIX{i}'] for i in range(ndim, 0, -1)]
            elif use_phy:
                offsets[i,] = calc_offset_physical(
                    hdr, None, intify_offset=True, order_xyz=False, ignore_ltm=True
                )

            # NOTE: the indexing in python is [z, y, x] order!!
            raw_shape = tuple(int(hdr[f"NAXIS{i}"]) for i in range(ndim, 0, -1))
            raw_shapes[i,] = raw_shape
            shapes[i,] = _trimmed_shape(raw_shape, trimsec)
        else:
            if imcmb_key == "$I":
                try:
                    imcmb_val.append(Path(item).name)
                except TypeError:
                    imcmb_val.append(f"User-provided {type(item)}")

            try:
                Path(item)
            except TypeError:
                data = _parse_imc_data_header(
                    item, extension=extension, parse_header=False, copy=False
                )[0]
                if data is None:
                    raise ValueError(f"Could not read data from input {i}.") from None
                raw_shape = data.shape
                if data.ndim != ndim:
                    raise ValueError(
                        "All input images must have identical ndim."
                    ) from None
            else:
                _, hdr = _parse_imc_data_header(
                    item, extension=extension, parse_data=False, copy=False
                )
                if hdr is None:
                    raise ValueError(f"Could not read header from input {i}.")
                if hdr["NAXIS"] != ndim:
                    raise ValueError(
                        "All FITS files must have the identical ndim, "
                        + "though they can have different sizes."
                    )
                raw_shape = tuple(int(hdr[f"NAXIS{i}"]) for i in range(ndim, 0, -1))

            raw_shapes[i,] = raw_shape
            if trimsec is not None:
                shapes[i,] = _trimmed_shape(raw_shape, trimsec)
            else:
                shapes[i,] = raw_shape

    return {
        "hdr0": hdr0,
        "ndim": ndim,
        "shapes": shapes,
        "raw_shapes": raw_shapes,
        "offsets": offsets,
        "offset_mode": offset_mode,
        "use_wcs": use_wcs,
        "use_phy": use_phy,
        "imcmb_val": imcmb_val,
        "extract_exptime": extract_exptime,
        "scales": scales,
        "gns": gns,
        "rds": rds,
        "sns": sns,
    }


def _expand_chunk_slices(
    chunk: tuple[slice, ...], shape: tuple[int, ...], halo: int
) -> tuple[slice, ...]:
    """Include spatial neighbors needed for growth, clipped at image edges."""
    return tuple(
        slice(max(0, sl.start - halo), min(size, sl.stop + halo))
        for sl, size in zip(chunk, shape, strict=True)
    )


def check_stack_memory(
    ncombine: int,
    sh_comb: tuple[int, ...],
    dtype: npt.DTypeLike,
    combine: str,
    memlimit: float | None,
    offsets: np.ndarray | None = None,
    shapes: np.ndarray | None = None,
    *,
    full: bool = False,
    reject: str | None = None,
    thresholds: bool = False,
    sample_flags: bool = False,
    halo: int = 0,
) -> tuple[float, int, list[tuple[slice, ...]]]:
    """Plan input sections from the estimated memory needed for combination.

    Before loading the full input stack, estimate the memory for that stack,
    temporary computation buffers, and retained outputs. If the estimate exceeds
    `memlimit`, choose spatial sections to read and combine one at a time.
    The final image and diagnostic arrays stay in memory for every chunk, so
    subtract their sizes first. Divide the remaining budget by the estimated
    input and working bytes per pixel to choose how much of each image to read.

    Parameters
    ----------
    ncombine : `int`
        Number of input images.
    sh_comb : `tuple` of `int`
        Final output shape after offsets.
    dtype : dtype-like
        Output dtype. Integer outputs use a floating workspace to hold NaNs.
    combine : `str`
        Combine method. The legacy working-memory estimate uses 4.5 times the
        input stack size for median, and 3 times for other methods. These are
        rough allowances, not measured bounds for the current backend.
    memlimit : `float` or `None`
        Approximate budget in bytes for loaded input sections, temporary
        computation buffers, and retained output arrays. `None` or non-positive
        values disable chunking. This is not a process RSS limit.
    offsets, shapes : `~numpy.ndarray`, optional
        Normalized origins and image shapes, used to count active chunk planes.
    full : `bool`, optional
        Reserve the full diagnostic outputs as well as the combined image.
    reject : `str` or `None`, optional
        Normalized rejection name, determining which diagnostics are retained.
    thresholds : `bool`, optional
        Whether a threshold mask is requested.
    sample_flags : `bool`, optional
        Reserve detailed diagnostics, including one byte per input sample.
    halo : `int`, optional
        Additional pixels read on each side of a chunk for rejection growth.

    Returns
    -------
    mem_req : `float`
        Estimated bytes to load and combine all inputs at once, including
        temporary computation buffers and retained outputs.
    num_chunk : `int`
        Number of output chunks.
    chunks : `list` of `tuple` of `slice`
        Output sections, with contiguous FITS slabs preferred.

    Notes
    -----
    Caller-owned data, FITS decoding, full-image zero/scale statistics, output
    serialization, and backend/thread scratch are outside this estimate.
    Diagnostic masks always reserve all input planes, including sparse offsets.
    """
    pixels = prod(sh_comb)
    itemsize = np.result_type(np.dtype(dtype), np.nan).itemsize
    persistent = pixels * np.dtype(dtype).itemsize  # final combined image
    if full or sample_flags:
        persistent += pixels * ncombine  # total mask, one bool per input sample
        if thresholds:
            persistent += pixels * ncombine  # threshold mask
        if reject is not None and str(reject).lower() != "none":
            persistent += pixels * ncombine  # rejection mask
            persistent += pixels * (2 * itemsize + 2)  # bounds + uint8 counts/flags
            if reject in {"sigclip", "ccdclip"}:
                persistent += pixels * itemsize  # clipping standard deviation
        if sample_flags:
            persistent += pixels * ncombine  # uint8 sample provenance

    # Preserve the original approximate workspace allowance. Only the stack
    # region shrinks with chunking; the output allocation above does not.
    memory_factor = 4.5 if str(combine).lower() in {"med", "median"} else 3.0
    # Growth also needs temporary sample flags to distinguish true rejection
    # seeds from pre-existing exclusions, even when flags are not returned.
    bytes_per_sample = memory_factor * itemsize + int(sample_flags or halo > 0)
    temporary_pixel = ncombine * bytes_per_sample
    mem_req = float(persistent + pixels * temporary_pixel)
    full_chunk = tuple(slice(0, size) for size in sh_comb)
    if memlimit is None or memlimit <= 0 or mem_req <= memlimit:
        return mem_req, 1, [full_chunk]

    available = memlimit - persistent
    if available <= 0:
        raise ValueError(
            f"memlimit ({memlimit:g} bytes) cannot hold persistent outputs "
            f"({persistent} bytes) plus a FITS chunk. Increase memlimit or "
            "disable full/diagnostic outputs."
        )
    if halo or (offsets is not None and shapes is not None):
        chunks = _plan_offset_chunks(
            full_chunk=full_chunk,
            offsets=(
                np.zeros((ncombine, len(sh_comb)), dtype=int)
                if offsets is None
                else np.asarray(offsets, dtype=int)
            ),
            shapes=(
                np.broadcast_to(sh_comb, (ncombine, len(sh_comb)))
                if shapes is None
                else np.asarray(shapes, dtype=int)
            ),
            bytes_per_sample=bytes_per_sample,
            memlimit=available,
            halo=halo,
        )
        return mem_req, len(chunks), chunks

    # Preserve full fast axes when a slab fits. Otherwise bisect dimensions
    # until even multidimensional small-budget chunks can be represented.
    for axis in range(len(sh_comb)):
        bytes_per_axis_pixel = temporary_pixel * prod(
            sh_comb[:axis] + sh_comb[axis + 1 :]
        )
        size = int(available // bytes_per_axis_pixel)
        if size >= 1:
            chunks = []
            for start in range(0, sh_comb[axis], size):
                slices = list(full_chunk)
                slices[axis] = slice(start, min(start + size, sh_comb[axis]))
                chunks.append(tuple(slices))
            return mem_req, len(chunks), chunks

    chunks = _plan_offset_chunks(
        full_chunk,
        np.zeros((ncombine, len(sh_comb)), dtype=int),
        np.broadcast_to(sh_comb, (ncombine, len(sh_comb))),
        bytes_per_sample=bytes_per_sample,
        memlimit=available,
    )
    return mem_req, len(chunks), chunks


def _plan_offset_chunks(
    full_chunk: tuple[slice, ...],
    offsets: np.ndarray,
    shapes: np.ndarray,
    *,
    bytes_per_sample: float,
    memlimit: float,
    halo: int = 0,
) -> list[tuple[slice, ...]]:
    """Return offset-aware chunks whose active stack estimates fit memory."""

    def chunk_memory(chunk: tuple[slice, ...]) -> float:
        chunk = _expand_chunk_slices(chunk, tuple(sl.stop for sl in full_chunk), halo)
        starts = np.array([sl.start for sl in chunk])
        stops = np.array([sl.stop for sl in chunk])
        image_stops = offsets + shapes
        active = np.count_nonzero(
            np.all(np.minimum(stops, image_stops) > np.maximum(starts, offsets), axis=1)
        )
        pixels = prod(sl.stop - sl.start for sl in chunk)
        return float(pixels * active * bytes_per_sample)

    def split_score(chunk: tuple[slice, ...], axis: int) -> tuple[float, float, int]:
        start = chunk[axis].start
        stop = chunk[axis].stop
        mid = (start + stop) // 2
        children = []
        for child_start, child_stop in [(start, mid), (mid, stop)]:
            slices = list(chunk)
            slices[axis] = slice(child_start, child_stop)
            children.append(tuple(slices))
        child_memory = [chunk_memory(child) for child in children]
        return max(child_memory), sum(child_memory), axis

    chunks = []
    stack = [full_chunk]
    while stack:
        chunk = stack.pop()
        if chunk_memory(chunk) <= memlimit:
            chunks.append(chunk)
            continue

        splittable_axes = [
            axis for axis, sl in enumerate(chunk) if sl.stop - sl.start > 1
        ]
        if not splittable_axes:
            raise ValueError(
                "Remaining memlimit after persistent outputs is too small for one FITS chunk: "
                f"{memlimit:g} bytes available, {chunk_memory(chunk):g} bytes needed. "
                "Increase memlimit."
            )

        split_axis = min(splittable_axes, key=lambda axis: split_score(chunk, axis))
        start = chunk[split_axis].start
        stop = chunk[split_axis].stop
        mid = (start + stop) // 2
        upper = list(chunk)
        upper[split_axis] = slice(mid, stop)
        lower = list(chunk)
        lower[split_axis] = slice(start, mid)
        stack.append(tuple(upper))
        stack.append(tuple(lower))

    return chunks


def calculate_zsw(
    items: Sequence[Any],
    dtype: npt.DTypeLike,
    trimsec: str | None,
    extension: HDUExt,
    extension_mask: HDUExt,
    extension_uncertainty: HDUExt,
    extract_exptime: bool,
    scale: Key_or_Val,
    zero: Key_or_Val,
    weight: Key_or_Val,
    zero_kw: dict[str, Any] | None,
    scale_kw: dict[str, Any] | None,
    zero_section: str | None,
    scale_section: str | None,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate global zero, scale, and weight values before chunking.

    Chunked combination must use the same zero/scale/weight values as the
    full-stack path. When a statistic name is requested, this function loads
    each input image once and evaluates the statistic on the full trimmed image,
    not per chunk.
    """
    ncombine = len(items)
    zeros = np.zeros(shape=ncombine)
    weights = np.ones(shape=ncombine)

    needs_data = any(
        isinstance(value, str) or callable(value)
        for value in (zero, None if extract_exptime else scale, weight)
    )
    for i, item in enumerate(items):
        if needs_data:
            # Preserve the legacy global zero/scale/weight semantics.  These
            # statistics must not be recalculated per chunk.
            data, _var, _mask = load_imcombine_item(
                item,
                trimsec=trimsec,
                extension=extension,
                extension_mask=extension_mask,
                extension_uncertainty=extension_uncertainty,
            )
            workspace = data[None, :]
        else:
            # Numeric arguments only need a plane count and dtype for validation.
            workspace = np.empty((1, 1, 1), dtype=float)

        z_i, s_i, w_i = _resolve_zsw(
            arr=workspace,
            zero=_image_parameter("zero", zero, i, ncombine),
            scale=(
                scales[i]
                if extract_exptime
                else _image_parameter("scale", scale, i, ncombine)
            ),
            weight=_image_parameter("weight", weight, i, ncombine),
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section if needs_data else None,
            scale_section=scale_section if needs_data else None,
        )
        zeros[i], scales[i], weights[i] = z_i[0], s_i[0], w_i[0]
        # Do not keep the previous image alive while reading the next one.
        del workspace
        if needs_data:
            del data, _var, _mask

    return zeros, scales, weights


def load_imcombine_item(
    item: Any,
    trimsec: str | None,
    extension: HDUExt,
    extension_mask: HDUExt,
    extension_uncertainty: HDUExt,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load one complete imcombine input as data, variance, and mask arrays.

    FITS inputs use the same section decoder as chunked reads, so scaling and
    BLANK-to-NaN conversion agree for every memory budget. CCDData inputs are
    sliced directly. Uncertainties are copied only when explicitly requested.
    """
    try:
        path = Path(item)
    except TypeError as err:
        if isinstance(item, CCDData):
            slices = _trim_slices(trimsec, item.data.shape)
            data = item.data[slices].copy()
            if item.mask is None:
                mask = np.zeros(data.shape, dtype=bool)
            else:
                mask = item.mask[slices].copy()
            var = (
                None
                if extension_uncertainty is None or item.uncertainty is None
                else np.asarray(item.uncertainty.array)[slices].copy()
            )
        else:
            raise ValueError("Each item is not path-like or CCDData.") from err
        return data, var, mask

    with fits.open(path, memmap=False) as hdul:
        hdu = _get_image_hdu(hdul, extension)
        if hdu is None:
            raise ValueError(f"No image data found in {path}.")
        section = _trim_slices(trimsec, hdu.shape)
        data = _read_hdul_section(hdul, extension, section)
        var = _read_hdul_section(hdul, extension_uncertainty, section)
        mask = _read_hdul_section(hdul, extension_mask, section)
    mask = (
        np.zeros(data.shape, dtype=bool)
        if mask is None
        else mask.astype(bool, copy=False)
    )
    return data, var, mask


def load_imcombine_item_region(
    item: Any,
    data_slices: tuple[slice, ...],
    raw_shape: Sequence[int],
    trimsec: str | None,
    extension: HDUExt,
    extension_mask: HDUExt,
    extension_uncertainty: HDUExt,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load only one region of an imcombine input.

    `data_slices` are expressed in the trimmed input-image coordinates. They
    are converted back to raw FITS/CCDData coordinates before reading, so
    ``trimsec`` and chunk boundaries compose correctly.
    """
    section = _compose_trim_data_slices(trimsec, data_slices, raw_shape)
    try:
        path = Path(item)
    except TypeError as err:
        if not isinstance(item, CCDData):
            raise ValueError("Each item is not path-like or CCDData.") from err
        data = item.data[section].copy()
        if item.mask is None:
            mask = np.zeros(data.shape, dtype=bool)
        else:
            mask = item.mask[section].copy()
        var = (
            None
            if extension_uncertainty is None or item.uncertainty is None
            else np.asarray(item.uncertainty.array)[section].copy()
        )
        return data, var, mask

    # Section reads stay partial without memory mapping, and allow Astropy to
    # apply BSCALE/BZERO/BLANK to scaled and unsigned FITS data.
    with fits.open(path, memmap=False) as hdul:
        data = _read_hdul_section(hdul, extension, section)
        if data is None:
            raise ValueError(f"No image data found in {path}.")
        var = _read_hdul_section(hdul, extension_uncertainty, section)
        mask = _read_hdul_section(hdul, extension_mask, section)
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)
    else:
        mask = mask.astype(bool, copy=False)
    return data, var, mask


def load_full_stack(
    items: Sequence[Any],
    offsets: np.ndarray,
    shapes: np.ndarray,
    sh_comb: tuple[int, ...],
    dtype: npt.DTypeLike,
    mask: np.ndarray | None,
    trimsec: str | None,
    extension: HDUExt,
    extension_mask: HDUExt,
    extension_uncertainty: HDUExt,
    extract_exptime: bool,
    scale: Key_or_Val,
    zero: Key_or_Val,
    weight: Key_or_Val,
    zero_kw: dict[str, Any] | None,
    scale_kw: dict[str, Any] | None,
    zero_section: str | None,
    scale_section: str | None,
    scales: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray
]:
    """Load all images into one offset-expanded stack.

    This is the legacy/non-chunked loading path. Each input image is trimmed,
    inserted at its normalized offset location, and used to calculate
    zero/scale/weight values before insertion into the final stack.
    """
    ncombine = len(items)
    zeros = np.zeros(shape=ncombine)
    weights = np.ones(shape=ncombine)
    # Match NaN * zeros promotion for integer output dtypes without creating
    # a second full-sized allocation.
    stack_dtype = np.result_type(np.dtype(dtype), np.nan)
    var_full = None
    if extension_uncertainty is not None:
        var_full = np.full((ncombine, *sh_comb), np.nan, dtype=stack_dtype)

    arr_full = np.full((ncombine, *sh_comb), np.nan, dtype=stack_dtype)
    mask_full = np.zeros(shape=(ncombine, *sh_comb), dtype=bool)

    for i, (item, offset, shape) in enumerate(
        zip(items, offsets, shapes, strict=False)
    ):
        # --- Set slice -------------------------------------------------------------- #
        # offsets2slice is introduced much later than the code below was written,
        # so not used here..
        # offset & size at each j-th dimension axis
        insert_slices = tuple(
            slice(offset_j, offset_j + shape_j, None)
            for offset_j, shape_j in zip(offset, shape, strict=False)
        )
        slices = (i, *insert_slices)

        # --- Load data -------------------------------------------------------------- #
        data, var, item_mask = load_imcombine_item(
            item,
            trimsec=trimsec,
            extension=extension,
            extension_mask=extension_mask,
            extension_uncertainty=extension_uncertainty,
        )

        if mask is not None:
            item_mask |= mask[i,]

        # --- zero and scale --------------------------------------------------------- #
        # better to calculate here than from full array, as the
        # latter may contain too many NaNs due to offest shifting.
        # TODO: let get_zsw to get functionals for zsw, so _set_calc_zsw
        # will not be repeated for every iteration.
        z_i, s_i, w_i = _resolve_zsw(
            arr=np.asarray(data[None, :]),  # make a fake (N+1)-D array
            zero=_image_parameter("zero", zero, i, ncombine),
            scale=(
                scales[i]
                if extract_exptime
                else _image_parameter("scale", scale, i, ncombine)
            ),
            weight=_image_parameter("weight", weight, i, ncombine),
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section,
            scale_section=scale_section,
        )
        zeros[i] = z_i[0]
        scales[i] = s_i[0]
        weights[i] = w_i[0]

        # --- Insertion -------------------------------------------------------------- #
        arr_full[slices] = data
        mask_full[slices] = item_mask
        if var is not None and var_full is not None:
            var_full[slices] = var

    return arr_full, mask_full, var_full, zeros, scales, weights


def load_stack_chunk(
    items: Sequence[Any],
    offsets: np.ndarray,
    shapes: np.ndarray,
    raw_shapes: np.ndarray,
    chunk_slices: tuple[slice, ...],
    dtype: npt.DTypeLike,
    mask: np.ndarray | None,
    trimsec: str | None,
    extension: HDUExt,
    extension_mask: HDUExt,
    extension_uncertainty: HDUExt,
    compact: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Load one output-image chunk into an offset-expanded mini stack.

    When ``compact`` is `True`, the returned arrays have shape
    ``(nactive, *chunk_shape)`` and contain only input images that overlap the
    chunk. ``active_indices`` maps local stack planes back to the original
    image index. When ``compact`` is `False`, all input planes are retained and
    non-overlap regions stay as NaN/False to preserve rank-rejection semantics.
    """
    ncombine = len(items)
    chunk_shape = tuple(sl.stop - sl.start for sl in chunk_slices)
    chunk_starts = np.array([sl.start for sl in chunk_slices])
    chunk_stops = np.array([sl.stop for sl in chunk_slices])

    image_starts_all = offsets
    image_stops_all = offsets + shapes
    starts_all = np.maximum(chunk_starts, image_starts_all)
    stops_all = np.minimum(chunk_stops, image_stops_all)
    overlaps = np.all(stops_all > starts_all, axis=1)
    active_indices = (
        np.flatnonzero(overlaps) if compact else np.arange(ncombine, dtype=int)
    )
    nactive = active_indices.size

    stack_dtype = np.result_type(np.dtype(dtype), np.nan)
    var_chunk = None
    if extension_uncertainty is not None:
        var_chunk = np.full((nactive, *chunk_shape), np.nan, dtype=stack_dtype)

    arr_chunk = np.full((nactive, *chunk_shape), np.nan, dtype=stack_dtype)
    mask_chunk = np.zeros(shape=(nactive, *chunk_shape), dtype=bool)

    for plane, _i in enumerate(active_indices):
        item = items[_i]
        _o = offsets[_i]
        _s = shapes[_i]
        _s0 = raw_shapes[_i]
        image_starts = _o
        image_stops = _o + _s
        starts = np.maximum(chunk_starts, image_starts)
        stops = np.minimum(chunk_stops, image_stops)
        if np.any(stops <= starts):
            continue

        data_slices = tuple(
            slice(int(_a - _c), int(_b - _c))
            for _a, _b, _c in zip(starts, stops, image_starts, strict=False)
        )
        insert_slices = tuple(
            slice(int(_a - _c), int(_b - _c))
            for _a, _b, _c in zip(starts, stops, chunk_starts, strict=False)
        )

        data, var, item_mask = load_imcombine_item_region(
            item=item,
            data_slices=data_slices,
            raw_shape=_s0,
            trimsec=trimsec,
            extension=extension,
            extension_mask=extension_mask,
            extension_uncertainty=extension_uncertainty,
        )

        if mask is not None:
            item_mask |= mask[_i,][data_slices]

        full_insert_slices = (plane, *insert_slices)
        arr_chunk[full_insert_slices] = data
        mask_chunk[full_insert_slices] = item_mask
        if var is not None and var_chunk is not None:
            var_chunk[full_insert_slices] = var

    return arr_chunk, mask_chunk, var_chunk, active_indices


def log_zsw_table(
    items: Sequence[Any],
    zeros: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    verbose: bool,
) -> None:
    """Write a zero/scale/weight summary to the package logger."""
    if not verbose:
        return
    logger.info("Done.")
    if isinstance(items[0], str):
        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "{:^45s}|{:^9s}|{:^9s}|{:^9s}".format("input", "zero", "scale", "weight")
        )
        logger.info("-" * 80)
        for item, z, s, w in zip(items, zeros, scales, weights, strict=False):
            logger.info(f"{item[-45:]:>45s}|{z:3e}|{s:3e}|{w:3e}")
        logger.info("-" * 80)
        logger.info("")


def apply_output_offsets(
    header: fits.Header,
    ndim: int,
    offsets: np.ndarray,
    use_wcs: bool,
    use_phy: bool,
) -> None:
    """Shift output WCS/physical reference keywords after offset combination.

    The combined image is written in the normalized output frame. For WCS or
    physical-offset modes, the first image's reference keywords must be shifted
    by its normalized offset so viewers such as ds9 align the result.
    """
    if use_wcs:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            header[f"CRPIX{i}"] += offsets[0][ndim - i]

    if use_phy:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            header[f"LTV{i}"] += offsets[0][ndim - i]


def write_imcombine_outputs(
    comb: CCDData,
    hdr0: fits.Header,
    output: StrPathLike | None,
    output_std: StrPathLike | None,
    output_low: StrPathLike | None,
    output_upp: StrPathLike | None,
    output_nrej: StrPathLike | None,
    output_mask: StrPathLike | None,
    output_flags: StrPathLike | None,
    std: np.ndarray | None,
    low: np.ndarray | None,
    upp: np.ndarray | None,
    mask_total: np.ndarray | None,
    output_flags_data: np.ndarray | None,
    int_dtype: npt.DTypeLike,
    dtype: npt.DTypeLike,
    dtype_std: npt.DTypeLike,
    dtype_low: npt.DTypeLike | None,
    dtype_upp: npt.DTypeLike | None,
    output_verify: str,
    overwrite: bool,
    checksum: bool,
    *,
    output_sample_flags: StrPathLike | None = None,
    sample_flags: np.ndarray | None = None,
) -> None:
    """Write the main combined image and any requested diagnostic FITS files.

    Diagnostic arrays are written only when their corresponding output path is
    provided. ``output_nrej`` is derived from the total mask, while
    ``output_mask`` stores the per-input total mask as an unsigned byte array
    because FITS image data cannot store booleans directly.
    """
    write_kw = {
        "output_verify": output_verify,
        "overwrite": overwrite,
        "checksum": checksum,
    }
    if output is not None:
        try:
            comb.write(output, **write_kw)
        except VerifyError as err:
            raise VerifyError("Use output_verify='fix'") from err

    if output_std is not None:
        if std is None:
            raise ValueError("std is required when output_std is requested.")
        std = std.astype(dtype_std)
        write2fits(std, hdr0, output_std, return_ccd=False, **write_kw)

    if output_low is not None:
        if low is None:
            raise ValueError("low is required when output_low is requested.")
        low = low.astype(dtype) if dtype_low is None else low.astype(dtype_low)
        write2fits(low, hdr0, output_low, return_ccd=False, **write_kw)

    if output_upp is not None:
        if upp is None:
            raise ValueError("upp is required when output_upp is requested.")
        upp = upp.astype(dtype) if dtype_upp is None else upp.astype(dtype_upp)
        write2fits(upp, hdr0, output_upp, return_ccd=False, **write_kw)

    if output_nrej is not None:  # Do this BEFORE output_mask!!
        if mask_total is None:
            raise ValueError("mask_total is required when output_nrej is requested.")
        nrej = np.count_nonzero(mask_total, axis=0).astype(int_dtype)
        write2fits(nrej, hdr0, output_nrej, return_ccd=False, **write_kw)

    if output_mask is not None:  # Do this AFTER output_nrej!!
        if mask_total is None:
            raise ValueError("mask_total is required when output_mask is requested.")
        # FITS does not accept boolean. We need uint8.
        write2fits(
            mask_total.astype(np.uint8), hdr0, output_mask, return_ccd=False, **write_kw
        )

    if output_flags is not None:
        if output_flags_data is None:
            raise ValueError(
                "output_flags_data is required when output_flags is requested."
            )
        write2fits(output_flags_data, hdr0, output_flags, return_ccd=False, **write_kw)

    if output_sample_flags is not None:
        if sample_flags is None:
            raise ValueError(
                "sample_flags is required when output_sample_flags is requested."
            )
        write2fits(
            sample_flags, hdr0, output_sample_flags, return_ccd=False, **write_kw
        )
