"""Validate image geometry shared by comet enhancements."""

import numpy as np
import numpy.typing as npt


def _update_mean(
    mean: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    count: int,
) -> None:
    """Update midpoint averages without overflowing sums or differences.

    Halving both operands before subtraction bounds the difference for
    opposite-sign finite values. Assign the first sample directly so a
    constant field, including subnormal intensities, stays exactly constant.
    """
    if count == 1:
        mean[...] = values
    else:
        mean += (values / 2.0 - mean / 2.0) / (count / 2.0)


def _prepare_image(
    image: npt.ArrayLike,
    center: tuple[float, float],
    mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], tuple[float, float]]:
    """Prepare an image and validate its zero-based ``(x, y)`` center.

    Parameters
    ----------
    image : array-like
        A nonempty, real, two-dimensional numeric image. Masked-array masks
        and nonfinite values are converted to `NaN`.
    center : tuple of float
        Zero-based ``(x, y)`` coordinates of the image center. Both
        coordinates must lie within the pixel-center extent of the image.
    mask : array-like, optional
        An additional exclusion mask with the same shape as `image`. Values
        are converted to `bool`.

    Returns
    -------
    data : ndarray
        A private float64 copy of `image`, with excluded pixels set to `NaN`.
    center : tuple of float
        The validated zero-based ``(x, y)`` center.

    Raises
    ------
    TypeError
        If `image` or `center` is not real numeric data.
    ValueError
        If the image, mask, or center has an invalid shape or value.
    """
    masked_image = np.ma.asarray(image)
    raw_image = np.ma.getdata(masked_image)
    if raw_image.ndim != 2 or 0 in raw_image.shape:
        raise ValueError("image must be a nonempty 2-D array")
    if not np.issubdtype(raw_image.dtype, np.number) or np.iscomplexobj(raw_image):
        raise TypeError("image must be real numeric data")

    data = np.array(raw_image, dtype=np.float64, copy=True)
    excluded = np.ma.getmaskarray(masked_image).copy()
    if mask is not None:
        external_mask = np.asarray(mask)
        if external_mask.shape != data.shape:
            raise ValueError(
                "mask must have the same shape as image; "
                f"got {external_mask.shape} and {data.shape}"
            )
        try:
            excluded |= np.asarray(external_mask, dtype=bool)
        except (TypeError, ValueError) as exc:
            raise TypeError("mask must be convertible to a boolean array") from exc
    data[excluded | ~np.isfinite(data)] = np.nan

    raw_center = np.asarray(center)
    if raw_center.shape != (2,):
        raise ValueError("center must contain exactly two coordinates in (x, y) order")
    if raw_center.dtype.kind not in "iuf" or np.iscomplexobj(raw_center):
        raise TypeError("center must contain two real numeric coordinates")
    center_values = np.asarray(raw_center, dtype=np.float64)
    if not np.all(np.isfinite(center_values)):
        raise ValueError("center coordinates must be finite")
    cx, cy = (float(value) for value in center_values)
    height, width = data.shape
    if not (0.0 <= cx <= width - 1 and 0.0 <= cy <= height - 1):
        raise ValueError(
            "center coordinates must lie within the image pixel-center extent "
            f"[0, {width - 1}] x [0, {height - 1}]"
        )

    return data, (cx, cy)


def _subpixel_offsets(subpixels: int) -> npt.NDArray[np.float64]:
    """Return midpoint offsets for one pixel axis.

    Parameters
    ----------
    subpixels : int
        Positive number of equally spaced midpoint samples per pixel axis.

    Returns
    -------
    ndarray
        Offsets from the pixel center, strictly between ``-0.5`` and
        ``0.5``.

    Raises
    ------
    ValueError
        If `subpixels` is not a positive integer.
    """
    if isinstance(subpixels, (bool, np.bool_)) or not isinstance(
        subpixels, (int, np.integer)
    ):
        raise ValueError("subpixels must be a positive integer")
    subpixels = int(subpixels)
    if subpixels < 1:
        raise ValueError("subpixels must be a positive integer")
    return (np.arange(subpixels, dtype=np.float64) + 0.5) / subpixels - 0.5


def _positive_integer(value: int, name: str) -> int:
    """Validate and normalize a positive integer parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_scalar(value: float, name: str) -> float:
    """Validate a signed real scalar without accepting booleans or strings."""
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    """Validate a finite nonnegative scalar."""
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from exc
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return value_float


def _finite_positive(value: float, name: str) -> float:
    """Validate a finite positive scalar."""
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value_float
