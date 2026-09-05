"""Pixel mask, interpolation, saturation, and extrema helpers."""

import numpy as np
from astro_ndslice import bezel2slice
from astropy.nddata import CCDData
from astropy.time import Time

from .._core.types import FITSLike, HDUExt, StrPathLike
from ..fitsmgmt import header as headers
from ..fitsmgmt import io as _io

__all__ = [
    "fixpix",
    "find_extpix",
    "find_satpix",
]


def _fixpix_interpolate_span(
    data: np.ndarray,
    naxis: tuple[int, ...],
    pos: tuple[int, ...],
    interp_ax: int,
    start: int,
    stop: int,
) -> None:
    init = start - 1
    last = stop + 1
    delta = last - init

    if init < 0 and last >= naxis[interp_ax]:
        return
    if init < 0:
        init = last
    elif last >= naxis[interp_ax]:
        last = init

    coord_init = list(pos)
    coord_last = list(pos)
    coord_slice = []
    for axis, coord in enumerate(pos):
        if axis == interp_ax:
            coord_init[axis] = init
            coord_last[axis] = last
            coord_slice.append(slice(start, stop + 1))
        else:
            coord_slice.append(slice(coord, coord + 1))

    val_init = data.item(tuple(coord_init))
    val_last = data.item(tuple(coord_last))
    grid = np.arange(1, delta, 1)
    data[tuple(coord_slice)].flat = (val_last - val_init) / delta * grid + val_init


def _fixpix_run_spans(
    data: np.ndarray,
    mask: np.ndarray,
    priority: tuple[int, ...],
) -> None:
    # Scan the *flattened* mask once, then expand only the selected
    # coordinates. C traversal is preserved even when the original mask is
    # strided.
    flat_positions = np.flatnonzero(mask)
    nfix = flat_positions.size
    if nfix == 0:
        return
    coords = np.unravel_index(flat_positions, mask.shape)
    # ^^^
    # Used NumPy's optimized 1-D boolean nonzero path, then convert selected
    # indices in one vectorized unravel_index call. This was ~2x faster (0.4ms
    # vs 0.2ms for 512x512 mask) locally than N-D nonzero, but needs extra
    # indices and may copy strided masks.

    chosen_axis = np.empty(nfix, dtype=np.intp)
    chosen_start = np.empty(nfix, dtype=np.intp)
    chosen_stop = np.empty(nfix, dtype=np.intp)
    chosen_length = np.full(nfix, np.iinfo(np.intp).max, dtype=np.intp)

    for axis in priority:
        other_axes = [a for a in range(data.ndim) if a != axis]
        # Group each axis-parallel line, then order its masked coordinates.
        order = np.lexsort([coords[axis], *(coords[a] for a in reversed(other_axes))])
        along = coords[axis][order]
        breaks = np.empty(nfix, dtype=bool)
        breaks[0] = True
        breaks[1:] = along[1:] != along[:-1] + 1
        for other in other_axes:
            line_coord = coords[other][order]
            breaks[1:] |= line_coord[1:] != line_coord[:-1]
        run_indices = np.flatnonzero(breaks)
        lengths = np.diff(np.append(run_indices, nfix))
        starts = np.repeat(along[run_indices], lengths)
        span_lengths = np.repeat(lengths, lengths)
        # Strictly shorter wins; ties retain the first axis in priority order.
        shorter = span_lengths < chosen_length[order]
        selected = order[shorter]
        chosen_axis[selected] = axis
        chosen_start[selected] = starts[shorter]
        chosen_length[selected] = span_lengths[shorter]
        chosen_stop[selected] = starts[shorter] + span_lengths[shorter] - 1

    # Do not deduplicate spans: crossing runs can overwrite earlier results.
    for i, pos in enumerate(zip(*coords, strict=True)):
        _fixpix_interpolate_span(
            data,
            data.shape,
            pos,
            int(chosen_axis[i]),
            int(chosen_start[i]),
            int(chosen_stop[i]),
        )


def fixpix(
    ccd: FITSLike,
    mask: FITSLike | None = None,
    maskpath: StrPathLike | None = None,
    extension: HDUExt = None,
    mask_extension: HDUExt = None,
    priority: tuple[int, ...] | None = None,
    update_header: bool = True,
    verbose: bool = True,
) -> CCDData:
    """Interpolate the masked location (N-D generalization of IRAF PROTO.FIXPIX)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like
        The CCD data to be "fixed".

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, optional.
        The mask to be used for fixing pixels (pixels to be fixed are where
        `mask` is `True`). If `None`, nothing will happen and `ccd` is
        returned.

    extension, mask_extension: `int`, `str`, (`str`, `int`), `None`
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    priority: `tuple` of `int`, `None`, optional.
        The priority of axis as a `tuple` of non-repeating `int` from ``0`` to
        `ccd.ndim`. It will be used if the mask has the same size along two or
        more of the directions. To specify, use the integers for axis
        directions, descending priority. For example,  ``(2, 1, 0)`` will be
        identical to `priority=None` (default) for 3-D images.
        Default is `None` to follow IRAF's PROTO.FIXPIX: Priority is higher for
        larger axis number (e.g., in 2-D, x-axis (axis=1) has higher priority
        than y-axis (axis=0)).

    Notes
    -----
    Run boundaries are computed from the original mask with storage proportional
    to the number of masked pixels times the dimension. Interpolation retains
    C-order traversal and repeated writes where runs cross. Precomputation adds
    index storage and sorting overhead, including for isolated bad pixels.

    Examples
    --------
    Timing test: MBP 15" [2018, macOS 11.4, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)], 2021-11-05 11:14:04 (KST:
    GMT+09:00)

    >>> np.random.RandomState(123)  # RandomState(MT19937) at 0x7FAECA768D40
    >>> data = np.random.normal(size=(1000, 1000))
    >>> mask = np.zeros_like(data).astype(bool)
    >>> mask[10, 10] = True
    >>> %timeit air.fixpix(data, mask)
    19.7 ms +- 1.53 ms per loop (mean +- std. dev. of 7 runs, 100 loops each)

    Same benchmark on MBP 14" [2024, macOS 26.4.1,
    M4Pro(8P+4E/G20c/N16c/48G)], 2026-05-27:

    >>> rng = np.random.default_rng(123)
    >>> data = rng.normal(size=(1000, 1000))
    >>> mask = np.zeros_like(data, dtype=bool)
    >>> mask[10, 10] = True
    >>> %timeit air.fixpix(data, mask)
    8.55 ms ± 124 µs per loop (7 runs, 20 loops each)

    Same M4Pro benchmark after run-span interpolation was added:

    >>> %timeit air.fixpix(data, mask, update_header=False, verbose=False)
    166 µs ± 8 µs per loop (5 runs, 200 loops each)

    >>> mask100 = np.zeros_like(data, dtype=bool)
    >>> mask100.ravel()[::10000] = True
    >>> %timeit air.fixpix(data, mask100, update_header=False, verbose=False)
    620 µs ± 25 µs per loop (5 runs, 100 loops each)

    >>> print(data[9:12, 9:12], air.fixpix(data, mask)[9:12, 9:12])
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621  1.37965928  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]]
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621 -0.32934328  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]] adu
    """
    if mask is None:
        return ccd.copy()

    if update_header:
        _t_start = Time.now()
    _ccd, _, _ = _io._parse_image(ccd, extension=extension, force_ccddata=True)
    mask, maskpath, _ = _io._parse_image(
        mask, extension=mask_extension, name=maskpath, force_ccddata=True
    )
    mask = mask.data.astype(bool, copy=False)
    data = _ccd.data

    if _ccd.shape != mask.shape:
        raise ValueError(
            f"ccd and mask must have the identical shape; now {_ccd.shape} VS {mask.shape}."
        )

    ndim = data.ndim

    if priority is None:
        priority = tuple(list(range(ndim))[::-1])
    elif len(priority) != ndim:
        raise ValueError(
            "len(priority) and ccd.ndim must be the same; "
            + f"now {len(priority)} VS {ccd.ndim}."
        )
    elif not isinstance(priority, tuple):
        priority = tuple(priority)
    elif (np.min(priority) != 0) or (np.max(priority) != ndim - 1):
        raise ValueError(
            f"`priority` must be a tuple of int (0 <= int <= {ccd.ndim-1=}). "
            + f"Now it's {priority=}"
        )

    nfix = np.count_nonzero(mask)
    _fixpix_run_spans(data, mask, priority)

    if update_header:
        _ccd.header["MASKNPIX"] = (nfix, "No. of pixels masked (fixed) by fixpix.")
        _ccd.header["MASKFILE"] = (maskpath, "Applied mask for fixpix.")
        _ccd.header["MASKORD"] = (
            str(priority),
            "Axis priority for fixpix (python order)",
        )
        # MASKFILE: name identical to IRAF
        # add as history
        headers.cmt2hdr(
            _ccd.header,
            "h",
            t_ref=_t_start,
            verbose=verbose,
            s="[air.fixpix] Pixel values interpolated.",
        )
        headers.update_process(_ccd.header, "P")

    return _ccd


def find_extpix(
    ccd: CCDData,
    mask: CCDData | np.ndarray | None = None,
    npixs: tuple[int | None, int | None] = (1, 1),
    bezels: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    order_xyz: bool = True,
    sort: bool = True,
    update_header: bool = True,
    verbose: int = 0,
) -> list[np.ndarray | None]:
    """Finds the N extrema pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`
        The ccd to find extreme values

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    npixs : length-2 `tuple` of `int`, optional
        The numbers of extrema to find, in the form of ``[small, large]``, so
        that ``small`` number of smallest and ``large`` number of largest pixel
        values will be found. If `None`, no extrema is found (`None` is
        returned for that extremum).
        Default: ``(1, 1)`` (find minimum and maximum)
        Default: ``(1, 1)``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    sort: `bool`, optional.
        Whether to sort the extrema in ascending order.
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    if not len(npixs) == 2:
        raise ValueError("npixs must be a length-2 tuple of int.")
    _t = Time.now()
    data = ccd.data.copy().astype("float32")  # Not float64 to reduce memory usage
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]

    if mask is None:
        maskname = "No mask"
        mask = ~np.isfinite(data)
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data | ~np.isfinite(data)
        else:
            maskname = "User-provided mask"

    exts = []
    for npix, sign, minmaxval in zip(npixs, [1, -1], [np.inf, -np.inf], strict=False):
        if npix is None:
            exts.append(None)
            continue
        data[mask] = minmaxval
        # ^ if getting maximum/minimum pix vals, replace with minimum/maximum
        extvals = np.partition(data.ravel(), sign * npix)
        #         ^^^^^^^^^^^^
        # bn.partitoin has virtually no speed gain.
        extvals = extvals[:npix] if sign > 0 else extvals[-npix:]
        if sort:
            extvals = np.sort(extvals)[::sign]
        exts.append(extvals)

    if update_header:
        for ext, mm in zip(exts, ["min", "max"], strict=False):
            if ext is not None:
                for i, extval in enumerate(ext):
                    ccd.header.set(
                        f"{mm.upper()}V{i + 1:03d}", extval, f"{mm} pixel value"
                    )
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        headers.cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=(
                "[air.find_extpix] Extrema pixel values found N(smallest, largest) = "
                + f"{npixs} excluding mask ({maskname}){bezstr}. See MINViii and MAXViii."
            ),
        )
    return exts


def find_satpix(
    ccd: CCDData | np.ndarray,
    mask: CCDData | np.ndarray | None = None,
    satlevel: float = 65535,
    bezels: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
    order_xyz: bool = True,
    update_header: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    """Finds saturated pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`, `~numpy.ndarray`
        The ccd to find extreme values. If `ndarray`, `update_header` will
        automatically be set to `False`.

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    satlevel: numeric, optional.
        The saturation level. Pixels >= `satlevel` will be treated as
        saturated pixels, except for those masked by `mask`.
        Default: ``65535``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    _t = Time.now()
    if isinstance(ccd, CCDData):
        data = ccd.data.copy()
    else:
        data = ccd.copy()
        update_header = False
    satmask = np.zeros(data.shape, dtype=bool)
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]
    else:
        sls = [slice(None, None, None) for _ in range(data.ndim)]

    if mask is None:
        maskname = "No mask"
        satmask[sls] = data >= satlevel
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data
        else:
            maskname = "User-provided mask"
        satmask[sls] = (data >= satlevel) & (~mask)  # saturated && not masked

    if update_header:
        nsat = np.count_nonzero(satmask[sls])
        ccd.header["NSATPIX"] = (nsat, "No. of saturated pix")
        ccd.header["SATLEVEL"] = (satlevel, "Saturation: pixels >= this value")
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        headers.cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s="[air.find_satpix] Saturated pixels calculated based on satlevel = "
            + f"{satlevel}, excluding mask ({maskname}){bezstr}. "
            + "See NSATPIX and SATLEVEL.",
        )
    return satmask
