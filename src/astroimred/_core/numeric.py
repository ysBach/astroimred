"""Standalone array math helpers."""

from collections.abc import Callable
from typing import Literal

import imcombiners.kernels as imck
import numpy as np
import reducers as rd
from astropy.stats import sigma_clip
from numpy.typing import ArrayLike

__all__ = [
    "sqsum",
    "quad_sum",
    "magsum",
    "sigma_clipper",
    "normalize",
    "wvg",
    "quantile_lh",
    "quantile_sigma",
    "binning",
    "dB2epadu",
    "epadu2dB",
]


def _native_dtype(dtype: np.dtype) -> np.dtype:
    """Return dtype normalized to native byte order where applicable."""
    dtype = np.dtype(dtype)
    if dtype.byteorder not in {"=", "|"}:
        return dtype.newbyteorder("=")
    return dtype


def _as_reducer_1d_values(values: ArrayLike) -> np.ndarray:
    """Return native-endian contiguous 1-D values accepted by reducers."""
    arr = np.asarray(values).ravel()
    dtype = _native_dtype(arr.dtype)
    if dtype != arr.dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _as_reducer_array(values: ArrayLike) -> np.ndarray:
    """Return native-endian contiguous values accepted by reducers."""
    arr = np.asarray(values)
    dtype = _native_dtype(arr.dtype)
    if dtype != arr.dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def sqsum(*args: object) -> object:
    """Return the sum of squares of all inputs."""
    total = 0
    for arg in args:
        total += arg**2
    return total


def quad_sum(*args: object) -> object:
    """Return the square root of `sqsum(*args)`."""
    return np.sqrt(sqsum(*args))


def magsum(*args: object) -> object:
    """Return the flux-equivalent sum of astronomical magnitudes."""
    return -2.5 * np.log10(np.sum(10 ** (-0.4 * np.array(args))))


def sigma_clipper(
    data: ArrayLike,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int | None = 5,
    cenfunc: Literal["median", "mean"] | Callable = "median",
    stdfunc: Literal["std", "mad_std"] | Callable = "std",
) -> ArrayLike | tuple[ArrayLike, float, float] | tuple[ArrayLike, ...]:
    """Return sigma-clipped data with clipped values removed.

    Compatible 1-D calls use `imcombiners.kernels.sigclip_mask_1d`; other calls
    fall back to `astropy.stats.sigma_clip` with ``masked=False`` hard-coded, so
    rejected elements are physically removed from the returned ndarray. The
    default ``stdfunc`` ignores NaNs because Astropy may inject NaNs into
    intermediate arrays during iterative clipping. For 1-D robust clipping,
    use ``stdfunc="mad_std"``; this is translated to the installed fast
    backend's internal MAD name when needed.
    """
    arr = np.asarray(data)
    if arr.ndim == 1 and arr.size == 0:
        return arr[~np.isnan(arr)]
    if (
        arr.ndim == 1
        and maxiters is not None
        and isinstance(cenfunc, str)
        and cenfunc in {"median", "mean"}
        and isinstance(stdfunc, str)
        and stdfunc in {"std", "mad_std"}
    ):
        sigma_pair = (
            float(sigma if sigma_lower is None else sigma_lower),
            float(sigma if sigma_upper is None else sigma_upper),
        )
        mask_rej = imck.sigclip_mask_1d(
            _as_reducer_1d_values(arr),
            sigma=sigma_pair,
            maxiters=int(maxiters),
            cenfunc=cenfunc,
            stdfunc="mad" if stdfunc == "mad_std" else stdfunc,
            nkeep=0,
            validate=False,
        )
        return arr[~mask_rej]

    return sigma_clip(
        data,
        masked=False,
        sigma=sigma,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc=cenfunc,
        stdfunc=stdfunc,
    )


def normalize(
    num: float, lower: float = 0, upper: float = 360, b: bool = False
) -> float:
    """Normalize number to range [lower, upper) or [lower, upper].

    Parameters
    ----------
    num : float
        The number to be normalized.

    lower, upper : numeric
        Lower/upper limit of range. Default: ``0`` and ``360``.

    b : bool
        Type of normalization. Default is `False`. See notes. When b=True, the
        range must be symmetric about 0. When b=False, the range must be
        symmetric about 0 or ``lower`` must be equal to 0.

    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].

    Raises
    ------
    ValueError
      If lower >= upper.

    Notes
    -----
    From phn: https://github.com/phn/angles

    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle, with the
    lower and upper ends sitting on top of each other. Moving past one limit,
    takes the number into the beginning of the other end. For example, if range
    is [0 - 360), then 361 becomes 1 and 360 becomes 0. Negative numbers move
    from higher to lower numbers. So, -1 normalized to [0 - 360) becomes 359.
    When b=False range must be symmetric about 0 or lower=0. If the keyword `b
    == True`, then the given number is considered to "bounce" between the two
    limits. So, -91 normalized to [-90, 90], becomes -89, instead of 89. In
    this case the range is [lower, upper]. This code is based on the function
    `fmt_delta` of `TPM`. When b=True range must be symmetric about 0.

    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(181, -180, 180, b=True)
    179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90)
    80.0
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
    """
    if lower >= upper:
        raise ValueError("lower must be lesser than upper")
    if not b:
        if not ((lower + upper == 0) or (lower == 0)):
            raise ValueError("When b=False lower=0 or range must be symmetric about 0.")
    else:
        if not (lower + upper == 0):
            raise ValueError("When b=True range must be symmetric about 0.")

    from math import ceil, floor

    res = num
    if not b:
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0

    return res


def wvg(
    val: ArrayLike,
    err: ArrayLike | None = None,
    var: ArrayLike | None = None,
    ivar: ArrayLike | None = None,
    axis: int | tuple[int, ...] | None = None,
    return_se: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the inverse-variance weighted mean.

    Parameters
    ----------
    val : array-like
        Values to average.
    err, var, ivar : array-like, optional
        Supply exactly one uncertainty representation: 1-sigma error,
        variance, or inverse variance. Scalars apply to every measurement;
        arrays must broadcast to `val`. For a single integer `axis`, a 1-D
        uncertainty array of the matching length supplies weights along that
        axis, as in ``numpy.average``. Errors are assumed independent.
    axis : int, tuple of int, or None, optional
        Axis or axes along which to combine. If `None`, combine all values.
    return_se : bool, optional
        If `True`, also return the weighted standard error (``sqrt(1/sum(ivar,
        axis=axis))``).

    Returns
    -------
    mean or (mean, stderr)
        Weighted mean, optionally with standard error.
    """
    val = np.asarray(val)
    provided = sum(x is not None for x in (err, var, ivar))
    if provided != 1:
        raise ValueError("Exactly one of err, var, or ivar must be provided.")

    unc = np.asarray(ivar if ivar is not None else (var if var is not None else err))
    # Convert integer errors before squaring, where fixed-width integers overflow.
    if unc.dtype.kind in "biu":
        unc = unc.astype(np.float64)
    weight = unc if ivar is not None else (1 / (unc if var is not None else unc**2))

    axis_weights = (
        isinstance(axis, (int, np.integer))
        and weight.ndim == 1
        and weight.size == val.shape[axis]
    )
    # reducers reuses axis vectors without allocating a full weight array.
    if not axis_weights:
        weight = np.broadcast_to(weight, val.shape)
    try:
        weighted_sum, sum_weight = rd.sum(
            val, weights=weight, axis=axis, return_sum_weights=True
        )
    except (NotImplementedError, TypeError) as exc:
        # Tuple axes and complex inputs currently raise TypeError in reducers.
        if isinstance(exc, TypeError) and not (
            isinstance(axis, tuple) or np.iscomplexobj(val) or np.iscomplexobj(weight)
        ):
            raise
        if axis_weights:
            weight_shape = [1] * val.ndim
            weight_shape[axis] = weight.size
            weight = weight.reshape(weight_shape)
        mean, sum_weight = np.average(
            val, weights=np.broadcast_to(weight, val.shape), axis=axis, returned=True
        )
    else:
        mean = np.divide(weighted_sum, sum_weight)
    return (mean, 1 / np.sqrt(sum_weight)) if return_se else mean


def quantile_lh(
    a: np.ndarray,
    lq: float,
    hq: float,
    axis: int | None = None,
    nanfunc: bool = False,
) -> np.ndarray:
    """Return lower and upper quantiles.

    Parameters
    ----------
    a : array-like
        Input data.

    lq, hq : `float`
        Lower and upper quantiles to compute, which must be between 0 and 1
        inclusive.

    axis : {`int`, `None`}, optional
        Axis along which the quantiles are computed. Supported values are
        `None`, 0, -1, and ``a.ndim - 1``.

    nanfunc : `bool`, optional
        Whether to use `~reducers.nanquantile` instead of `~reducers.quantile`.
        Default: `False`.

    """
    a = np.asarray(a)

    try:
        lq = float(lq)
        hq = float(hq)
    except TypeError as err:
        raise TypeError("lq and hq must be floats, not array-like.") from err

    qfunc = rd.nanquantile if nanfunc else rd.quantile
    return qfunc(a, (lq, hq), axis=axis)


def quantile_sigma(
    a: np.ndarray,
    axis: int | None = None,
    nanfunc: bool = False,
) -> np.ndarray:
    """Estimate sigma from the 15.87 and 84.13 percent quantiles."""
    low, upp = quantile_lh(
        a,
        0.1587,
        0.8413,
        axis=axis,
        nanfunc=nanfunc,
    )
    return np.abs(upp - low) / 2


def _validate_binning_factor(factor, axis):
    if factor is None:
        return None
    if isinstance(factor, (bool, np.bool_)):
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    if not isinstance(factor, (int, np.integer)):
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    return factor


def _normalize_binning_factors(arr_shape, factors, order_xyz):
    ndim = len(arr_shape)
    if factors is None:
        return np.ones(ndim, dtype=np.intp)

    raw_factors = list(np.asarray(factors, dtype=object).ravel())
    if len(raw_factors) != ndim:
        raise ValueError(
            f"factors must have the same length as arr.ndim ({ndim}); "
            f"got {len(raw_factors)}."
        )
    if order_xyz:
        raw_factors = raw_factors[::-1]

    normalized = []
    for axis, factor in enumerate(raw_factors):
        factor = arr_shape[axis] if factor is None else factor
        normalized.append(_validate_binning_factor(factor, axis))
    return np.asarray(normalized, dtype=np.intp)


def _reducer_stack_binning(
    reshaped: np.ndarray,
    nbin: np.ndarray,
    factors: np.ndarray,
    reducer: Callable,
) -> np.ndarray:
    """Bin a reshaped block view through a reducers axis-0 kernel."""
    bin_axes = tuple(range(0, reshaped.ndim, 2))
    factor_axes = tuple(range(1, reshaped.ndim, 2))
    stack = np.transpose(reshaped, (*factor_axes, *bin_axes)).reshape(
        int(np.prod(factors)),
        *(int(n) for n in nbin),
    )
    return reducer(_as_reducer_array(stack), axis=0, validate=False)


def _binning_reducer(binfunc: Callable, arr: np.ndarray) -> Callable | None:
    """Return the matching reducers function for supported binning reducers."""
    dtype = _native_dtype(arr.dtype)

    if binfunc is np.mean:
        return rd.mean
    if binfunc is np.nanmean:
        return rd.nanmean
    if binfunc is np.median:
        return rd.median
    if binfunc is np.nanmedian:
        return rd.nanmedian
    if np.issubdtype(dtype, np.floating):
        if binfunc is np.sum:
            return rd.sum
        if binfunc is np.nansum:
            return rd.nansum
    return None


def binning(
    arr: np.ndarray,
    factors: tuple[int, ...] | None = None,
    order_xyz: bool = True,
    binfunc: Callable = np.mean,
    trim_end: bool = False,
) -> np.ndarray:
    """Bin an array by integer factors.

    Parameters
    ---------
    arr : array-like
        Input array.

    factors : tuple of `int`, optional.
        The factors in pythonic axis order (``order_xyz=False``) or in xyz-style
        order (``order_xyz=True``), which is reversed into NumPy axis order. The
        number of factors must match ``arr.ndim``. If any factor is ``None``,
        that factor is replaced by the size of the array along that axis, i.e.,
        collapse along that axis.
        Default: `None`.

    binfunc : callable, optional
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
        Default: ``np.mean``.

    trim_end : `bool`, optional.
        Whether to trim the end of each axis so that the trimmed shape is
        divisible by the binning factors.
        Default: `False`.

    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("arr must not be empty.")

    factors = _normalize_binning_factors(
        arr.shape,
        factors,
        order_xyz,
    )
    shape = np.asarray(arr.shape, dtype=np.intp)

    if np.any(factors > shape):
        axis = int(np.flatnonzero(factors > shape)[0])
        raise ValueError(
            f"factor for axis {axis} ({factors[axis]}) is larger than "
            f"the axis length ({shape[axis]})."
        )

    remainder = shape % factors
    if trim_end:
        trim_shape = shape - remainder
        if np.any(trim_shape == 0):
            axis = int(np.flatnonzero(trim_shape == 0)[0])
            raise ValueError(
                f"factor for axis {axis} ({factors[axis]}) trims the axis to length 0."
            )
        slices = tuple(slice(None, int(size)) for size in trim_shape)
        arr = arr[slices]
        shape = trim_shape
    elif np.any(remainder):
        axis = int(np.flatnonzero(remainder)[0])
        raise ValueError(
            f"array shape along axis {axis} ({shape[axis]}) is not divisible "
            f"by factor {factors[axis]}; use trim_end=True to trim trailing "
            "elements."
        )

    nbin = shape // factors
    newshape = tuple(
        int(item) for pair in zip(nbin, factors, strict=True) for item in pair
    )
    reshaped = arr.reshape(newshape)
    if reducer := _binning_reducer(binfunc, arr):
        return _reducer_stack_binning(reshaped, nbin, factors, reducer)
    return binfunc(reshaped, axis=tuple(range(1, reshaped.ndim, 2)))


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB: float) -> float:
    """Convert gain from decibels to electron/ADU."""
    return 5 / 10 ** (gain_dB / 20)


def epadu2dB(gain_epadu: float) -> float:
    """Convert gain from electron/ADU to decibels."""
    return 20 * np.log10(5 / gain_epadu)
