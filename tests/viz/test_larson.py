"""Tests for Larson–Sekanina rotational differences."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred import viz


def _scalar_lerp(first: float, second: float, fraction: float) -> float:
    """Interpolate scalar endpoints while ignoring zero-weight endpoints."""
    if fraction == 0.0:
        return first
    if fraction == 1.0:
        return second
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.nan
    return (1.0 - fraction) * first + fraction * second


def _scalar_sample(image: np.ndarray, x: float, y: float, method: str) -> float:
    """Sample one Cartesian coordinate with the public interpolation rules."""
    height, width = image.shape
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        return math.nan
    if method == "nearest":
        x_index = math.floor(x + 0.5)
        y_index = math.floor(y + 0.5)
        value = float(image[y_index, x_index])
        return value if math.isfinite(value) else math.nan

    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    top = _scalar_lerp(float(image[y0, x0]), float(image[y0, x1]), fx)
    bottom = _scalar_lerp(float(image[y1, x0]), float(image[y1, x1]), fx)
    return _scalar_lerp(top, bottom, fy)


def _scalar_trig(angle: float) -> tuple[float, float]:
    """Return exact cardinal sine/cosine values for the reference calculation."""
    reduced = abs(angle) % 360.0
    if reduced % 90.0 == 0.0:
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(reduced / 90.0)]
    radians = math.radians(reduced)
    return math.cos(radians), math.sin(radians)


def _scalar_larson(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    angle: float,
    radial_shift: float = 0.0,
    interpolation: str = "linear",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the complete filter output without calling implementation code."""
    data = np.array(image, dtype=np.float64, copy=True)
    data[~np.isfinite(data)] = np.nan
    if mask is not None:
        data[np.asarray(mask, dtype=bool)] = np.nan

    height, width = data.shape
    cx, cy = center
    cosine, sine = _scalar_trig(angle)
    result = np.full((height, width), np.nan, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            value = float(data[y, x])
            if not math.isfinite(value):
                continue

            dx, dy = x - cx, y - cy
            radius = math.hypot(dx, dy)
            if radial_shift != 0.0:
                shifted_radius = radius - radial_shift
                if radius == 0.0 or shifted_radius < 0.0:
                    continue
                dx *= shifted_radius / radius
                dy *= shifted_radius / radius

            plus = _scalar_sample(
                data,
                cx + cosine * dx - sine * dy,
                cy + sine * dx + cosine * dy,
                interpolation,
            )
            if sine == 0.0:
                minus = plus
            else:
                minus = _scalar_sample(
                    data,
                    cx + cosine * dx + sine * dy,
                    cy - sine * dx + cosine * dy,
                    interpolation,
                )
            if math.isfinite(plus) and math.isfinite(minus):
                result[y, x] = (value - plus) + (value - minus)

    return result


def _polynomial_image(shape: tuple[int, int] = (9, 10)) -> np.ndarray:
    """Return an asymmetric quadratic image for whole-output comparisons."""
    yy, xx = np.indices(shape, dtype=np.float64)
    return 1.0 + 3.0 * xx + 7.0 * yy + 0.125 * xx * yy + 0.01 * xx**2


def test_larson_matches_independent_scalar_oracle_for_fractional_quadratic_data() -> (
    None
):
    """Bilinear samples and the whole residual agree with scalar calculations."""
    image = _polynomial_image()
    center = (4.25, 3.2)
    kwargs = {
        "angle": 37.0,
        "radial_shift": -0.35,
        "interpolation": "linear",
    }
    expected = _scalar_larson(image, center, **kwargs)

    result = viz.larson_sekanina(image, center, **kwargs)

    assert result.dtype == np.float64
    assert result.shape == image.shape
    assert np.isfinite(result).any()
    assert np.isnan(result).any()
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_larson_constant_image_has_zero_residual_and_center_is_defined() -> None:
    """A constant image has no rotational structure, including at the center."""
    image = np.full((7, 7), 3.0, dtype=np.float32)

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=37.0)

    finite = np.isfinite(result)
    assert finite.any()
    assert_allclose(result[finite], 0.0)
    assert result[3, 3] == 0.0


def test_larson_zero_shifts_are_zero_with_fractional_center() -> None:
    """Zero rotation and radial shift sample each original pixel exactly."""
    image = _polynomial_image((6, 7)).astype(np.float32)
    original = image.copy()

    result = viz.larson_sekanina(image, (2.25, 3.4), angle=0.0)

    assert_allclose(result, 0.0)
    assert_array_equal(image, original)


def test_larson_cardinal_rotation_uses_the_sum_of_two_differences() -> None:
    """A right-angle ramp verifies orientation and the unhalved sum formula."""
    yy, xx = np.indices((7, 7), dtype=np.float64)
    image = xx + 10.0 * yy

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0, interpolation="nearest")

    # At (x, y)=(4, 3), the two sources are (3, 4) and (3, 2):
    # (34 - 43) + (34 - 23) = 2, rather than the half-sum value 1.
    assert result[3, 4] == pytest.approx(2.0)
    assert result[3, 3] == 0.0


def test_larson_rotationally_symmetric_radial_coma_has_no_residual() -> None:
    """A 90-degree rotation of a concentric image leaves every pixel unchanged."""
    yy, xx = np.indices((9, 9), dtype=np.float64)
    image = (xx - 4.0) ** 2 + (yy - 4.0) ** 2

    result = viz.larson_sekanina(image, (4.0, 4.0), angle=90.0, interpolation="nearest")

    assert np.isfinite(result).all()
    assert_allclose(result, 0.0)


def test_larson_compact_jet_has_positive_source_and_negative_rotated_lobes() -> None:
    """A compact jet produces one positive source and two opposite lobes."""
    image = np.zeros((9, 9), dtype=np.float64)
    image[4, 6] = 1.0

    result = viz.larson_sekanina(image, (4.0, 4.0), angle=90.0, interpolation="nearest")

    assert result[4, 6] == pytest.approx(2.0)
    assert result[2, 4] == pytest.approx(-1.0)
    assert result[6, 4] == pytest.approx(-1.0)


def test_larson_angle_sign_and_full_turns_are_equivalent() -> None:
    """Opposite rotations form the same pair, and angles are reduced modulo 360."""
    image = _polynomial_image((7, 7))
    center = (3.0, 3.0)
    reference = viz.larson_sekanina(image, center, angle=90.0, interpolation="nearest")

    for angle in (-90.0, 270.0, 450.0, -450.0):
        result = viz.larson_sekanina(
            image, center, angle=angle, interpolation="nearest"
        )
        assert_allclose(result, reference, equal_nan=True)

    assert_allclose(
        viz.larson_sekanina(image, center, angle=360.0),
        viz.larson_sekanina(image, center, angle=0.0),
        equal_nan=True,
    )


def test_larson_radial_shift_sign_selects_inward_or_outward_sources() -> None:
    """Positive shifts sample inward and negative shifts sample outward."""
    yy, xx = np.indices((7, 7), dtype=np.float64)
    image = xx + 10.0 * yy

    inward = viz.larson_sekanina(
        image,
        (3.0, 3.0),
        angle=0.0,
        radial_shift=1.0,
        interpolation="nearest",
    )
    outward = viz.larson_sekanina(
        image,
        (3.0, 3.0),
        angle=0.0,
        radial_shift=-1.0,
        interpolation="nearest",
    )

    assert inward[3, 4] == pytest.approx(2.0)  # source (3, 3)
    assert outward[3, 4] == pytest.approx(-2.0)  # source (5, 3)
    assert np.isnan(inward[3, 3])
    assert np.isnan(outward[3, 3])


def test_larson_nearest_rounds_half_coordinates_up() -> None:
    """Nearest interpolation uses ``floor(coordinate + 0.5)`` at ties."""
    image = np.arange(25.0).reshape(5, 5)

    result = viz.larson_sekanina(
        image,
        (1.5, 1.0),
        angle=90.0,
        interpolation="nearest",
    )

    # Target (2, 1) has sources (1.5, 1.5) and (1.5, 0.5). Rounding up
    # selects values 12 and 7, so (7 - 12) + (7 - 7) = -5.
    assert result[1, 2] == pytest.approx(-5.0)


def test_larson_linear_interpolation_uses_fractional_source_values() -> None:
    """A fractional center produces bilinear source values rather than a snap."""
    yy, xx = np.indices((5, 5), dtype=np.float64)
    image = xx + 10.0 * yy

    result = viz.larson_sekanina(
        image,
        (1.5, 1.0),
        angle=90.0,
        interpolation="linear",
    )

    # At (2, 1), the sources are (1.5, 1.5) -> 16.5 and
    # (1.5, 0.5) -> 6.5; the target is 12.
    assert result[1, 2] == pytest.approx(1.0)


def test_larson_linear_ignores_nan_at_zero_weight_integer_neighbor() -> None:
    """A missing neighbor with zero bilinear weight does not invalidate a sample."""
    yy, xx = np.indices((7, 7), dtype=np.float64)
    image = xx + 10.0 * yy
    image[4, 4] = np.nan  # zero-weight neighbor of source (3, 4)

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0)

    assert result[3, 4] == pytest.approx(2.0)


def test_larson_linear_rejects_nan_with_positive_fractional_weight() -> None:
    """A missing bilinear contributor with positive weight invalidates its target."""
    yy, xx = np.indices((7, 7), dtype=np.float64)
    image = xx + 10.0 * yy
    image[4, 3] = np.nan  # contributor to source (3.25, 3.75)

    result = viz.larson_sekanina(image, (3.25, 3.0), angle=90.0)

    assert np.isnan(result[3, 4])


def test_larson_nonfinite_nonzero_weight_source_invalidates_output() -> None:
    """A NaN source with positive interpolation weight invalidates its target."""
    yy, xx = np.indices((7, 7), dtype=np.float64)
    image = xx + 10.0 * yy
    image[4, 3] = np.nan  # exact plus source for target (4, 3)

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0)

    assert np.isnan(result[3, 4])


def test_larson_mask_preserves_original_invalid_pixel_and_rejects_source_mask() -> None:
    """Masks affect both source validity and the corresponding output pixel."""
    image = np.ones((7, 7), dtype=np.float64)
    mask = np.zeros_like(image, dtype=bool)
    mask[3, 4] = True
    mask[4, 3] = True

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0, mask=mask)

    assert np.isnan(result[3, 4])
    assert np.isfinite(result[3, 5])
    assert_allclose(image, 1.0)
    assert not mask[3, 3]


def test_larson_combines_masked_array_and_external_mask() -> None:
    """Masked-array and explicit masks both invalidate original output pixels."""
    image = np.ma.array(np.ones((7, 7), dtype=np.float64), mask=False)
    image.mask[3, 4] = True
    mask = np.zeros(image.shape, dtype=bool)
    mask[4, 3] = True

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0, mask=mask)

    assert np.isnan(result[3, 4])
    assert np.isnan(result[4, 3])
    assert result[3, 3] == 0.0


@pytest.mark.parametrize(
    ("shape", "center", "center_index"),
    [((1, 5), (2.0, 0.0), (0, 2)), ((5, 1), (0.0, 2.0), (2, 0))],
)
def test_larson_degenerate_dimension_does_not_extrapolate(
    shape: tuple[int, int], center: tuple[float, float], center_index: tuple[int, int]
) -> None:
    """Pure rotation on a one-pixel axis leaves only the center sample defined."""
    image = np.ones(shape, dtype=np.float64)

    result = viz.larson_sekanina(image, center, angle=90.0)

    assert result[center_index] == 0.0
    assert np.count_nonzero(np.isfinite(result)) == 1


def test_larson_edges_are_nan_when_rotated_sources_leave_the_image() -> None:
    """The filter does not clip or extrapolate source coordinates."""
    image = np.arange(25.0).reshape(5, 5)

    result = viz.larson_sekanina(image, (2.0, 2.0), angle=45.0)

    assert np.isnan(result[0, 0])
    assert result[2, 2] == 0.0


def test_larson_source_exactly_on_frame_edge_is_valid() -> None:
    """Coordinates on the inclusive image frame are sampled without extrapolation."""
    image = np.arange(25.0).reshape(5, 5)

    result = viz.larson_sekanina(image, (2.0, 2.0), angle=90.0, interpolation="nearest")

    assert result[0, 2] == pytest.approx(-20.0)


def test_larson_rejects_negative_shifted_radii_but_keeps_outer_radii() -> None:
    """Pixels with ``r - radial_shift < 0`` are undefined, not extrapolated."""
    image = np.ones((9, 9), dtype=np.float64)

    result = viz.larson_sekanina(image, (4.0, 4.0), angle=0.0, radial_shift=2.5)

    assert np.isnan(result[4, 4])
    assert np.isnan(result[4, 6])
    assert np.isfinite(result[4, 7])


def test_larson_large_constant_does_not_overflow_two_times_the_image() -> None:
    """A finite maximum constant remains a finite zero residual."""
    image = np.full((7, 7), np.finfo(np.float64).max)

    result = viz.larson_sekanina(image, (3.0, 3.0), angle=90.0, interpolation="nearest")

    assert np.isfinite(result).all()
    assert_array_equal(result, np.zeros_like(image))


@pytest.mark.parametrize("level", [np.finfo(np.float64).max, np.nextafter(0.0, 1.0)])
def test_larson_extreme_constant_levels_stay_zero_with_linear_sampling(level) -> None:
    """Finite extreme constants remain exact zeros at noncardinal angles."""
    image = np.full((9, 9), level)

    result = viz.larson_sekanina(image, (4.0, 4.0), angle=37.0)

    finite = np.isfinite(result)
    assert finite.any()
    assert_allclose(result[finite], 0.0)


def test_larson_recovers_representable_cancellation_and_marks_overflow() -> None:
    """Stable differences preserve a representable result and reject an overflow."""
    finite_max = np.finfo(np.float64).max
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 3] = finite_max / 4.0
    image[3, 2] = -finite_max
    image[1, 2] = finite_max

    result = viz.larson_sekanina(image, (2.0, 2.0), angle=90.0, interpolation="nearest")

    assert result[2, 3] == pytest.approx(finite_max / 2.0)

    image[2, 3] = finite_max
    overflow = viz.larson_sekanina(
        image, (2.0, 2.0), angle=90.0, interpolation="nearest"
    )
    assert np.isnan(overflow[2, 3])


def test_larson_returns_float64_without_mutating_input() -> None:
    """The public result owns float64 storage and leaves the input unchanged."""
    image = _polynomial_image((6, 8)).astype(np.float32)
    original = image.copy()

    result = viz.larson_sekanina(
        image,
        (3.25, 2.4),
        angle=23.0,
        radial_shift=-0.2,
    )

    assert result.dtype == np.float64
    assert result.shape == image.shape
    assert_array_equal(image, original)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize(
    "bad_angle", [True, np.bool_(False), "90", [90], np.array([90.0]), np.nan, np.inf]
)
def test_larson_rejects_nonfinite_or_non_scalar_angles(bad_angle) -> None:
    """Angle must be one finite real scalar, not a bool, string, or sequence."""
    with pytest.raises(ValueError, match="angle"):
        viz.larson_sekanina(np.ones((3, 3)), (1.0, 1.0), angle=bad_angle)


@pytest.mark.parametrize(
    "bad_shift", [True, np.bool_(False), "1", [1], np.array([1.0]), np.nan, -np.inf]
)
def test_larson_rejects_nonfinite_or_non_scalar_radial_shifts(bad_shift) -> None:
    """Radial displacement must be a finite real scalar."""
    with pytest.raises(ValueError, match="radial_shift"):
        viz.larson_sekanina(
            np.ones((3, 3)), (1.0, 1.0), angle=0.0, radial_shift=bad_shift
        )


@pytest.mark.parametrize("bad_interpolation", ["cubic", None, True, ["linear"]])
def test_larson_rejects_unknown_interpolation(bad_interpolation) -> None:
    """Only nearest and bilinear interpolation are supported."""
    with pytest.raises(ValueError):
        viz.larson_sekanina(
            np.ones((3, 3)), (1.0, 1.0), angle=0.0, interpolation=bad_interpolation
        )


def test_larson_requires_angle_as_a_keyword_argument() -> None:
    """The angular shift is required and keyword-only."""
    image = np.ones((3, 3))

    with pytest.raises(TypeError):
        viz.larson_sekanina(image, (1.0, 1.0))
    with pytest.raises(TypeError):
        viz.larson_sekanina(image, (1.0, 1.0), 0.0)


@pytest.mark.parametrize(
    ("image", "center", "mask"),
    [
        (np.ones(3), (1.0, 1.0), None),
        (np.ones((0, 3)), (0.0, 0.0), None),
        (np.ones((3, 3), dtype=complex), (1.0, 1.0), None),
        (np.ones((3, 3)), (3.0, 1.0), None),
        (np.ones((3, 3)), (1.0, 1.0), np.zeros((2, 2), dtype=bool)),
    ],
)
def test_larson_reuses_image_geometry_validation(image, center, mask) -> None:
    """Invalid image, center, and mask geometry raises before sampling."""
    with pytest.raises((TypeError, ValueError)):
        viz.larson_sekanina(image, center, angle=0.0, mask=mask)
