"""Independent scalar oracles for comet enhancement sampling semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astroimred.viz import (
    azimuthal_normalize,
    azimuthal_renormalize,
    enhance_rho,
    radially_variable_filter,
)


def _midpoint_offsets(count: int) -> tuple[float, ...]:
    """Return scalar midpoint offsets without using the implementation helper."""
    return tuple((index + 0.5) / count - 0.5 for index in range(count))


def _nearest_pixel(value: float) -> int:
    """Round a coordinate using the documented half-up rule."""
    return math.floor(value + 0.5)


def _scalar_polar_sample(
    image: np.ndarray,
    center: tuple[float, float],
    nrad: int,
    ntheta: int,
    subpixels: int,
) -> np.ndarray:
    """Compute the polar image through scalar documented sampling steps."""
    height, width = image.shape
    cx, cy = center
    offsets = _midpoint_offsets(subpixels)
    polar = np.full((nrad, ntheta), np.nan, dtype=np.float64)

    for radial_index in range(nrad):
        radius_center = float(radial_index + 1)
        for theta_index in range(ntheta):
            total = 0.0
            valid = True
            for radial_offset in offsets:
                radius = radius_center + radial_offset
                for angular_offset in offsets:
                    theta = (
                        2.0 * math.pi * (theta_index + 0.5 + angular_offset) / ntheta
                    )
                    x = cx - radius * math.sin(theta)
                    y = cy + radius * math.cos(theta)
                    x_pixel = _nearest_pixel(x)
                    y_pixel = _nearest_pixel(y)
                    if not (
                        0 <= x_pixel < width
                        and 0 <= y_pixel < height
                        and math.isfinite(float(image[y_pixel, x_pixel]))
                    ):
                        valid = False
                        continue
                    total += float(image[y_pixel, x_pixel])
            if valid:
                polar[radial_index, theta_index] = total / (subpixels * subpixels)
    return polar


def _scalar_polar_reconstruct(
    polar: np.ndarray,
    center: tuple[float, float],
    shape: tuple[int, int],
    subpixels: int,
) -> np.ndarray:
    """Reconstruct every Cartesian output pixel through scalar lookups."""
    height, width = shape
    nrad, ntheta = polar.shape
    cx, cy = center
    offsets = _midpoint_offsets(subpixels)
    result = np.full(shape, np.nan, dtype=np.float64)
    twopi = 2.0 * math.pi

    for y_pixel in range(height):
        for x_pixel in range(width):
            total = 0.0
            valid = True
            for y_offset in offsets:
                for x_offset in offsets:
                    x_distance = x_pixel + x_offset - cx
                    y_distance = y_pixel + y_offset - cy
                    radius = math.hypot(x_distance, y_distance)
                    radius_index = _nearest_pixel(radius)
                    angle = math.atan2(-x_distance, y_distance) % twopi
                    theta_index = math.floor(angle * ntheta / twopi) % ntheta
                    inside = radius > 0.0 and 1 <= radius_index <= nrad
                    if not inside:
                        valid = False
                        continue
                    sampled = float(polar[radius_index - 1, theta_index])
                    if not math.isfinite(sampled):
                        valid = False
                        continue
                    total += sampled
            if valid:
                result[y_pixel, x_pixel] = total / (subpixels * subpixels)
    return result


def _scalar_azimuthal_normalize(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    statistic: str,
    subpixels: int,
) -> np.ndarray:
    """Build a normalized image without calling private implementation code."""
    polar = _scalar_polar_sample(image, center, nrad, ntheta, subpixels)
    normalized = np.full_like(polar, np.nan, dtype=np.float64)

    for radial_index, row in enumerate(polar):
        values = [float(value) for value in row if math.isfinite(float(value))]
        if not values:
            continue
        if statistic == "median":
            ordered = sorted(values)
            midpoint = len(ordered) // 2
            if len(ordered) % 2:
                scale = ordered[midpoint]
            else:
                scale = 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
        else:
            scale = sum(values) / len(values)
        if not math.isfinite(scale) or scale == 0.0:
            continue
        for theta_index, value in enumerate(row):
            if math.isfinite(float(value)):
                normalized[radial_index, theta_index] = float(value) / scale

    return _scalar_polar_reconstruct(normalized, center, image.shape, subpixels)


def _scalar_azimuthal_renormalize(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    nrad: int,
    ntheta: int,
    nsigma: float,
    subpixels: int,
) -> np.ndarray:
    """Build a renormalized image through scalar ring statistics."""
    polar = _scalar_polar_sample(image, center, nrad, ntheta, subpixels)
    normalized = np.full_like(polar, np.nan, dtype=np.float64)

    for radial_index, row in enumerate(polar):
        values = [float(value) for value in row if math.isfinite(float(value))]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
        if not math.isfinite(std) or std == 0.0:
            continue
        lower = mean - nsigma * std
        width = 2.0 * nsigma * std
        if not math.isfinite(lower) or not math.isfinite(width) or width == 0.0:
            continue
        for theta_index, value in enumerate(row):
            if math.isfinite(float(value)):
                normalized[radial_index, theta_index] = (float(value) - lower) / width

    return _scalar_polar_reconstruct(normalized, center, image.shape, subpixels)


_RVSF_NEIGHBORS = (
    (0, -1, 192.0),
    (0, 1, 192.0),
    (-1, 0, 192.0),
    (1, 0, 192.0),
    (-1, -1, 64.0),
    (-1, 1, 64.0),
    (1, -1, 64.0),
    (1, 1, 64.0),
)


def _scalar_rvsf(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    a: float,
    b: float,
    n: float,
    subpixels: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the variable stencil with scalar loops and explicit validity."""
    data = np.array(image, dtype=np.float64, copy=True)
    data[~np.isfinite(data)] = np.nan
    if mask is not None:
        data[np.asarray(mask, dtype=bool)] = np.nan

    height, width = data.shape
    cx, cy = center
    offsets = _midpoint_offsets(subpixels)
    minimum_axis = min(height, width)
    result = np.full(data.shape, np.nan, dtype=np.float64)

    for y_pixel in range(height):
        for x_pixel in range(width):
            center_value = float(data[y_pixel, x_pixel])
            if not math.isfinite(center_value):
                continue
            total = 0.0
            valid = True
            for y_offset in offsets:
                for x_offset in offsets:
                    rho = math.hypot(
                        x_pixel + x_offset - cx,
                        y_pixel + y_offset - cy,
                    )
                    half_width = a + b * rho**n
                    usable = math.isfinite(half_width) and half_width < minimum_axis
                    step = _nearest_pixel(half_width) if usable else 0
                    filtered = 0.0
                    for dy, dx, weight in _RVSF_NEIGHBORS:
                        y_neighbor = y_pixel + dy * step
                        x_neighbor = x_pixel + dx * step
                        if not (
                            usable
                            and 0 <= y_neighbor < height
                            and 0 <= x_neighbor < width
                            and math.isfinite(float(data[y_neighbor, x_neighbor]))
                        ):
                            valid = False
                            continue
                        filtered += weight * (
                            center_value - float(data[y_neighbor, x_neighbor])
                        )
                    if valid:
                        total += filtered
            if valid:
                result[y_pixel, x_pixel] = total / (subpixels * subpixels)
    return result


def _sampling_image(shape: tuple[int, int] = (9, 10)) -> np.ndarray:
    """Return an asymmetric image that distinguishes every sample direction."""
    yy, xx = np.indices(shape, dtype=np.float64)
    return 1.0 + 97.0 * yy + 13.0 * xx + 0.125 * xx * yy + 0.01 * xx**2


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_azimuthal_normalize_matches_independent_full_output_oracle(statistic):
    """The entire polar round trip honors fractional centers and angle bins."""
    image = _sampling_image()
    center = (4.25, 3.2)
    expected = _scalar_azimuthal_normalize(
        image,
        center,
        nrad=3,
        ntheta=7,
        statistic=statistic,
        subpixels=2,
    )

    result = azimuthal_normalize(
        image,
        center,
        nrad=3,
        ntheta=7,
        statistic=statistic,
        subpixels=2,
    )

    assert result.dtype == np.float64
    assert np.isfinite(result).any()
    assert np.isnan(result).any()
    assert_allclose(result, expected, equal_nan=True)


def test_azimuthal_renormalize_matches_independent_full_output_oracle():
    """The renormalization round trip honors sample deviations at every pixel."""
    image = _sampling_image()
    center = (4.25, 3.2)
    expected = _scalar_azimuthal_renormalize(
        image,
        center,
        nrad=3,
        ntheta=7,
        nsigma=1.5,
        subpixels=2,
    )

    result = azimuthal_renormalize(
        image,
        center,
        nrad=3,
        ntheta=7,
        nsigma=1.5,
        subpixels=2,
    )

    assert result.dtype == np.float64
    assert np.isfinite(result).any()
    assert np.isnan(result).any()
    assert_allclose(result, expected, equal_nan=True)


def test_rvsf_fractional_center_variable_width_and_mask_match_scalar_oracle():
    """Variable widths, fractional centers, boundaries, and masks agree globally."""
    image = _sampling_image()
    center = (4.25, 3.2)
    mask = np.zeros(image.shape, dtype=bool)
    mask[3, 7] = True
    mask[6, 2] = True
    kwargs = {"a": 0.5, "b": 0.9, "n": 0.7, "subpixels": 3}
    expected = _scalar_rvsf(image, center, mask=mask, **kwargs)

    result = radially_variable_filter(image, center, mask=mask, **kwargs)

    assert result.dtype == np.float64
    assert np.isfinite(result).any()
    assert np.isnan(result).any()
    assert_allclose(result, expected, equal_nan=True)


def test_rvsf_half_integer_width_rounds_up():
    """A half-pixel width selects the same stencil as a one-pixel width."""
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 3] = 2.0
    half_width = radially_variable_filter(
        image,
        (3.0, 3.0),
        a=0.5,
        b=0.0,
        n=1.0,
        subpixels=1,
    )
    one_pixel = radially_variable_filter(
        image,
        (3.0, 3.0),
        a=1.0,
        b=0.0,
        n=1.0,
        subpixels=1,
    )

    assert_allclose(half_width, one_pixel, equal_nan=True)
    assert half_width[3, 3] == pytest.approx(2048.0)


@pytest.mark.parametrize(
    ("subpixels", "python_count"),
    [(np.int8(12), 12), (np.uint8(16), 16)],
)
def test_enhance_rho_numpy_integer_subpixels_match_python_int(subpixels, python_count):
    """Accepted NumPy integer counts preserve the radius quadrature."""
    image = _sampling_image((4, 5))
    center = (2.3, 1.4)
    expected = enhance_rho(image, center, subpixels=python_count)
    result = enhance_rho(image, center, subpixels=subpixels)

    assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    ("subpixels", "python_count"),
    [(np.int8(12), 12), (np.uint8(16), 16)],
)
def test_rvsf_numpy_integer_subpixels_match_python_int(subpixels, python_count):
    """Accepted NumPy integer counts preserve variable-filter averaging."""
    image = _sampling_image((7, 8))
    center = (3.2, 2.6)
    kwargs = {"a": 0.5, "b": 0.35, "n": 0.7}
    expected = radially_variable_filter(
        image,
        center,
        subpixels=python_count,
        **kwargs,
    )
    result = radially_variable_filter(
        image,
        center,
        subpixels=subpixels,
        **kwargs,
    )

    assert_allclose(result, expected, equal_nan=True)
