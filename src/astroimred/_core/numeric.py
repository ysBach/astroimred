"""Standalone array math helpers."""

import importlib.util
from collections.abc import Callable

import numpy as np

_numba_available = importlib.util.find_spec("numba") is not None
_sstd_nan_1d_nb = None

__all__ = [
    "sstd",
    "sqsum",
    "quad_sum",
    "magsum",
    "normalize",
    "wvg",
    "quantile_lh",
    "quantile_sigma",
    "min_max_med_1d",
    "mean_std_1d",
    "binning",
    "dB2epadu",
    "epadu2dB",
]


def _get_sstd_nan_1d_nb():
    global _sstd_nan_1d_nb
    if _sstd_nan_1d_nb is None:
        import numba as nb

        @nb.njit
        def _kernel(arr, ddof):
            count = 0
            mean = 0.0
            m2 = 0.0
            for value in arr:
                if not np.isnan(value):
                    count += 1
                    delta = value - mean
                    mean += delta / count
                    m2 += delta * (value - mean)
            if count <= ddof:
                return count, np.nan
            return count, np.sqrt(m2 / (count - ddof))

        _sstd_nan_1d_nb = _kernel
    return _sstd_nan_1d_nb


def sstd(
    a: np.ndarray,
    ddof: int = 1,
    axis: int | tuple[int, ...] | None = None,
    nan: bool = False,
    **kwargs: object,
) -> np.ndarray:
    """Return standard deviation, optionally ignoring NaNs.

    ``nan=True`` ignores NaNs. For flattened arrays, a lazy optional Numba
    kernel is used when available; axis-aware calculations use `numpy.nanstd`.

    Notes
    -----
    Timing on MBP 14" [2024, macOS 26.4.1,
    M4Pro(8P+4E/G20c/N16c/48G)], 2026-05-27:

    >>> arr = np.random.default_rng(100).normal(size=100)
    >>> arr[::97] = np.nan
    >>> %timeit air.sstd(arr, nan=True, ddof=1)
    >>> # 0.89 µs per loop
    >>> %timeit np.nanstd(arr, ddof=1)
    >>> # 10.2 µs per loop

    >>> arr = np.random.default_rng(10_000).normal(size=10_000)
    >>> arr[::97] = np.nan
    >>> %timeit air.sstd(arr, nan=True, ddof=1)
    >>> # 35.0 µs per loop
    >>> %timeit np.nanstd(arr, ddof=1)
    >>> # 28.0 µs per loop

    >>> stack = np.random.default_rng(20).normal(size=(20, 512, 512))
    >>> stack[0, ::31, ::37] = np.nan
    >>> %timeit air.sstd(stack, nan=True, axis=0, ddof=1)
    >>> # 13.0 ms per loop
    >>> %timeit np.nanstd(stack, axis=0, ddof=1)
    >>> # 10.6 ms per loop
    """
    if not nan:
        return np.std(a, ddof=ddof, axis=axis, **kwargs)

    arr = np.asarray(a)
    if axis is None and _numba_available:
        arr_1d = np.ravel(arr)
        if not np.issubdtype(arr_1d.dtype, np.inexact):
            arr_1d = arr_1d.astype(float)
        count, std = _get_sstd_nan_1d_nb()(arr_1d, ddof)
        if count <= ddof:
            return np.array([], dtype=float)
        return std

    if axis is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.nanstd(arr, ddof=ddof, **kwargs)
        if np.isnan(std) and np.count_nonzero(~np.isnan(arr)) <= ddof:
            return np.array([], dtype=float)
        return std

    count = np.sum(~np.isnan(arr), axis=axis)
    if np.any(np.asarray(count) <= ddof):
        return np.array([], dtype=float)

    return np.nanstd(arr, ddof=ddof, axis=axis, **kwargs)


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
    val: np.ndarray,
    err: np.ndarray | None = None,
    var: np.ndarray | None = None,
    ivar: np.ndarray | None = None,
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
        variance, or inverse variance. The code prefers ivar > var > err when
        multiple are given (because ivar is the most directly useful for
        weighting, and ivar = 1/var = 1/err^2).
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

    if ivar is not None:
        weight = np.asarray(ivar)
    elif var is not None:
        weight = 1 / np.asarray(var)
    else:
        weight = 1 / (np.asarray(err) ** 2)

    wsum = np.sum(weight, axis=axis)
    mean = np.sum(weight * val, axis=axis) / wsum
    if return_se:
        return mean, 1 / np.sqrt(wsum)
    return mean


def quantile_lh(
    a: np.ndarray,
    lq: float,
    hq: float,
    axis: int | tuple[int, ...] | None = None,
    nanfunc: bool = False,
    interpolation: str = "linear",
    linterp: str | None = None,
    hinterp: str | None = None,
) -> list[np.ndarray]:
    """Return lower and upper quantiles.

    Parameters
    ----------
    a : array-like
        Input data.

    lq, hq : array_like of `float`
        Quantile or sequence of quantiles to compute, which must be between 0
        and 1 inclusive.

    axis : {`int`, `tuple` of `int`, `None`}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.

    nanfunc : `bool`, optional
        Whether to use `~numpy.nanquantile` instead of `~numpy.quantile`.
        Default: `False`.

    interpolation, linterp, hinterp : ``{'linear', 'lower', 'higher', 'midpoint', 'nearest'}``, optional.
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the
          fractional part of the index surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles
        individually, set `linterp` and `hinterp` separately. An idea is to use
        ``linterp='higher', hinterp='lower'`` to estimate the robust standard
        deviation estimate.
    """
    a = np.asarray(a)
    linterp = interpolation if linterp is None else linterp
    hinterp = interpolation if hinterp is None else hinterp

    qfunc = np.nanquantile if nanfunc else np.quantile

    try:
        lq = float(lq)
        hq = float(hq)
    except TypeError as err:
        raise TypeError("lq and hq must be floats, not array-like.") from err

    if linterp == hinterp:
        out = qfunc(a, (lq, hq), axis=axis, interpolation=linterp)
    else:
        out_l = qfunc(a, lq, axis=axis, interpolation=linterp)
        out_h = qfunc(a, hq, axis=axis, interpolation=hinterp)
        out = [out_l, out_h]

    return out


def quantile_sigma(
    a: np.ndarray,
    axis: int | tuple[int, ...] | None = None,
    nanfunc: bool = False,
    interpolation: str = "linear",
    linterp: str | None = None,
    hinterp: str | None = None,
) -> np.ndarray:
    """Estimate sigma from the 15.87 and 84.13 percent quantiles."""
    low, upp = quantile_lh(
        a,
        0.1587,
        0.8413,
        axis=axis,
        nanfunc=nanfunc,
        interpolation=interpolation,
        linterp=linterp,
        hinterp=hinterp,
    )
    return np.abs(upp - low) / 2


def min_max_med_1d(arr: np.ndarray) -> tuple[float, float, float]:
    """Return the minimum, maximum, and median of a 1-D array."""
    arr = np.asarray(arr)
    if arr.size < 1000:
        _a = np.sort(arr)
        mid = _a.size // 2
        med = _a[mid] if _a.size % 2 else 0.5 * (_a[mid] + _a[mid - 1])
        return _a[0], _a[-1], med
    else:
        return np.min(arr), np.max(arr), np.median(arr)


def mean_std_1d(
    arr: np.ndarray,
    ddof: int = 0,
    std: bool = True,
    var: bool = False,
) -> tuple[float, ...]:
    """Return mean with standard deviation and/or variance.

    Parameters
    ----------
    arr : array-like
        Input values.
    ddof : int, optional
        Delta degrees of freedom for variance normalization.
    std, var : bool, optional
        Select whether to include standard deviation and/or variance.
    """
    arr = np.asarray(arr)
    sum_a = np.sum(arr)
    sqsum = np.sum(arr**2)
    inv_n = 1.0 / arr.size
    inv_d = 1.0 / (arr.size - ddof) if ddof > 0 else inv_n
    mean = sum_a * inv_n
    var_value = sqsum * inv_d - mean * sum_a * inv_d
    if var:
        if std:
            return mean, np.sqrt(var_value), var_value
        return mean, var_value
    if std:
        return mean, np.sqrt(var_value)
    raise ValueError("At least one of `std` or `var` must be True.")


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
    return binfunc(reshaped, axis=tuple(range(1, reshaped.ndim, 2)))


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB: float) -> float:
    """Convert gain from decibels to electron/ADU."""
    return 5 / 10 ** (gain_dB / 20)


def epadu2dB(gain_epadu: float) -> float:
    """Convert gain from electron/ADU to decibels."""
    return 20 * np.log10(5 / gain_epadu)
