import warnings

import numpy as np

"""
a = np.array(
    [[[ 0.,  1.],
    [ 2., np.nan],
    [np.nan, np.nan]],

   [[ 6.,  7.],
    [ 8., np.nan],
    [np.nan, np.nan]],

   [[12., 13.],
    [14., 15.],
    [np.nan, np.nan]],

   [[18., np.nan],
    [20., 21.],
    [22., np.nan]],

   [[24., np.nan],
    [np.nan, 27.],
    [28., np.nan]]]
)

nanmedian(a, axis=0, choose='mean')
nanmedian(a, axis=0, choose='lower')
nanmedian(a, axis=0, choose='upper')

numpy(mean):   lower:         upper:
  [[12, 7 ],    [[12, 7 ],      [[12, 7 ],
   [11, 21],     [8 , 21],       [14, 21],
   [25, nan]]    [22, nan]]      [28, nan]]

"""

__all__ = ["lmedian", "nanlmedian", "median", "nanmedian", "ma_median"]


def _validate_choose(choose):
    if choose not in {"mean", "lower", "upper"}:
        raise ValueError("choose not understood")
    return choose


def _normalize_axes(axis, ndim):
    if axis is None:
        return None

    axes = []
    for ax in np.atleast_1d(axis):
        ax = int(ax)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise np.exceptions.AxisError(ax, ndim=ndim)
        axes.append(ax)

    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    return tuple(axes)


def _reshape_for_axes(a, axis):
    axes = _normalize_axes(axis, a.ndim)
    if axes is None:
        return a.ravel(), 0, tuple(range(a.ndim))
    if len(axes) == 1:
        return a, axes[0], axes

    keep_axes = tuple(ax for ax in range(a.ndim) if ax not in axes)
    moved = np.moveaxis(a, axes, range(a.ndim - len(axes), a.ndim))
    reduce_size = int(np.prod([a.shape[ax] for ax in axes]))
    return moved.reshape((*[a.shape[ax] for ax in keep_axes], reduce_size)), -1, axes


def _restore_keepdims(result, original_ndim, axes):
    if axes is None:
        return np.reshape(result, (1,) * original_ndim)

    out = result
    for ax in sorted(axes):
        out = np.expand_dims(out, axis=ax)
    return out


def _write_out(result, out):
    if out is not None:
        out[...] = result
        return out
    return result


def _take_order_statistic(a, axis, choose, overwrite_input=False):
    n = a.shape[axis]
    if n == 0:
        return np.mean(a, axis=axis)

    if choose == "lower":
        kth = (n - 1) // 2
    elif choose == "upper":
        kth = n // 2
    else:
        raise ValueError("choose not understood")

    if overwrite_input:
        part = a
        part.partition(kth, axis=axis)
    else:
        part = np.partition(a, kth, axis=axis)
    return np.take(part, kth, axis=axis)


def median(
    a, axis=None, out=None, overwrite_input=False, keepdims=False, choose="mean"
):
    """Return the median or selected middle order statistic.

    Parameters
    ----------
    a : array-like
        Input data.
    axis : int, tuple of int, or None, optional
        Axis or axes to reduce. If `None`, reduce the flattened input.
    out : ndarray, optional
        Output array.
    overwrite_input, keepdims : bool, optional
        Passed through with NumPy-like semantics.
    choose : {"mean", "lower", "upper"}, optional
        For even sample counts, return the mean of the two middle values,
        the lower middle value, or the upper middle value.

    Returns
    -------
    ndarray or scalar
        Median result.
    """
    choose = _validate_choose(choose)
    a = np.asanyarray(a)

    if choose == "mean":
        result = np.median(
            a,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            keepdims=keepdims,
        )
        return result

    work, work_axis, axes = _reshape_for_axes(a, axis)
    result = _take_order_statistic(
        work, work_axis, choose=choose, overwrite_input=overwrite_input
    )
    if keepdims:
        result = _restore_keepdims(result, a.ndim, axes)
    return _write_out(result, out)


def lmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Return the lower median.

    This is equivalent to ``median(..., choose="lower")``.
    """
    return median(
        a,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
        choose="lower",
    )


def _take_nan_order_statistic(a, axis, choose, overwrite_input=False):
    if a.shape[axis] == 0:
        return np.nanmean(a, axis=axis)

    if overwrite_input:
        sorted_a = a
        sorted_a.sort(axis=axis)
    else:
        sorted_a = np.sort(a, axis=axis)

    valid = ~np.isnan(sorted_a)
    counts = np.sum(valid, axis=axis)
    all_nan = counts == 0

    if choose == "lower":
        kth = (counts - 1) // 2
    elif choose == "upper":
        kth = counts // 2
    else:
        raise ValueError("choose not understood")

    kth = np.where(all_nan, 0, kth)
    result = np.take_along_axis(
        sorted_a, np.expand_dims(kth, axis=axis), axis=axis
    ).squeeze(axis=axis)
    result = np.asarray(result)
    result = np.where(all_nan, np.nan, result)

    if np.any(all_nan):
        for _ in range(np.count_nonzero(all_nan)):
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=4)

    return result


def nanmedian(
    a, axis=None, out=None, overwrite_input=False, keepdims=False, choose="mean"
):
    """Return the median while ignoring NaNs.

    Parameters are the same as `median`. For all-NaN slices, returns NaN and
    emits a `RuntimeWarning`.
    """
    choose = _validate_choose(choose)
    a = np.asanyarray(a)

    if choose == "mean":
        return np.nanmedian(
            a,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            keepdims=keepdims,
        )

    work, work_axis, axes = _reshape_for_axes(a, axis)
    result = _take_nan_order_statistic(
        work, work_axis, choose=choose, overwrite_input=overwrite_input
    )
    if keepdims:
        result = _restore_keepdims(result, a.ndim, axes)
    return _write_out(result, out)


def nanlmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Return the lower median while ignoring NaNs."""
    return nanmedian(
        a,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        keepdims=keepdims,
        choose="lower",
    )


def ma_median(
    a, axis=None, out=None, overwrite_input=False, keepdims=False, choose="mean"
):
    """Return the median of a masked array.

    Parameters are the same as `median`, with masked values ignored.
    """
    choose = _validate_choose(choose)
    a = np.ma.asarray(a)

    if choose == "mean":
        return np.ma.median(
            a,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            keepdims=keepdims,
        )

    work, work_axis, axes = _reshape_for_axes(a, axis)
    if overwrite_input:
        sorted_a = work
        sorted_a.sort(axis=work_axis)
    else:
        sorted_a = np.ma.sort(work, axis=work_axis)

    counts = np.ma.count(sorted_a, axis=work_axis)
    all_masked = counts == 0
    kth = (counts - 1) // 2 if choose == "lower" else counts // 2

    kth = np.where(all_masked, 0, kth)
    result = np.ma.take_along_axis(
        sorted_a, np.expand_dims(kth, axis=work_axis), axis=work_axis
    ).squeeze(axis=work_axis)
    result = np.ma.array(result, copy=False)
    result[all_masked] = np.ma.masked

    if keepdims:
        result = _restore_keepdims(result, a.ndim, axes)
    return _write_out(result, out)
