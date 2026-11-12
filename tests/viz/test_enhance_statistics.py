"""Behavior tests for enhancement statistics and direct annular normalization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred.viz import (
    annular_normalize,
    azimuthal_normalize,
    azimuthal_renormalize,
)
from astroimred.viz._enhance_stats import _ring_statistics


def _diagonal_ring_image(
    values: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return a radius-one image with known midpoint samples."""
    image = np.ones((7, 7), dtype=np.float64)
    center = (3.0, 3.0)
    for (row, column), value in zip(
        ((2, 2), (2, 4), (4, 4), (4, 2)), values, strict=True
    ):
        image[row, column] = value
    return image, center


def test_ring_statistics_mode_is_mmm_estimator():
    """Mode uses three times the median minus twice the mean."""
    samples = np.array([[1.0, 2.0, 3.0, 100.0]])

    level, std = _ring_statistics(
        samples, "mode", None, 1, "mean", "std", return_std=True
    )

    expected = 3.0 * np.median(samples[0]) - 2.0 * np.mean(samples[0])
    assert level[0] == pytest.approx(expected)
    assert std[0] == pytest.approx(np.std(samples[0], ddof=1))


def test_ring_statistics_callable_gets_private_finite_1d_values():
    """A callable sees finite one-dimensional samples and cannot mutate input."""
    samples = np.array([[1.0, np.nan, 3.0], [np.nan, np.inf, -np.inf]])
    original = samples.copy()
    calls: list[np.ndarray] = []

    def statistic(values: np.ndarray) -> float:
        assert values.ndim == 1
        assert values.dtype == np.float64
        assert np.isfinite(values).all()
        calls.append(values.copy())
        result = float(np.mean(values))
        values[...] = -999.0
        return result

    level, _ = _ring_statistics(samples, statistic, None, 1, "mean", "std")

    assert level[0] == pytest.approx(2.0)
    assert np.isnan(level[1])
    assert len(calls) == 1
    assert_array_equal(samples, original)


@pytest.mark.parametrize(
    "statistic",
    [
        lambda values: np.array([1.0]),
        lambda values: 1.0 + 0.0j,
    ],
)
def test_ring_statistics_rejects_non_real_scalar_callable_result(statistic):
    """Custom estimators must return a real scalar."""
    samples = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="real scalar"):
        _ring_statistics(samples, statistic, None, 1, "mean", "std")


def test_ring_statistics_empty_and_singleton_rows_have_defined_widths():
    """Empty rows are undefined while singleton rows have zero sample spread."""
    samples = np.array([[np.nan, np.nan], [5.0, np.nan]])

    level, std = _ring_statistics(
        samples, "mean", None, 1, "mean", "std", return_std=True
    )

    assert np.isnan(level[0])
    assert np.isnan(std[0])
    assert level[1] == pytest.approx(5.0)
    assert std[1] == pytest.approx(0.0)


def test_sigma_scalar_and_asymmetric_pair_clipping():
    """Scalar clipping is symmetric, while a pair supports separate bounds."""
    samples = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])

    symmetric, _ = _ring_statistics(samples, "mean", (1.0, 1.0), 1, "median", "std")
    lower_only, _ = _ring_statistics(samples, "mean", (0.0, 1.0), 1, "median", "std")

    assert symmetric[0] == pytest.approx(2.5)
    assert lower_only[0] == pytest.approx(3.5)


def test_maxiters_none_repeats_clipping_until_convergence():
    """Converged clipping can reject values exposed by an earlier rejection."""
    samples = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 20.0]])

    one_pass, _ = _ring_statistics(samples, "mean", (1.0, 1.0), 1, "mean", "std")
    converged, _ = _ring_statistics(samples, "mean", (1.0, 1.0), None, "mean", "std")

    assert one_pass[0] == pytest.approx(16.0 / 6.0)
    assert converged[0] == pytest.approx(2.5)


def test_clip_center_mean_and_median_select_different_samples():
    """The clipping center is independent of the requested estimator."""
    samples = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]])

    mean_center, _ = _ring_statistics(samples, "mean", (0.45, 0.45), 1, "mean", "std")
    median_center, _ = _ring_statistics(
        samples, "mean", (0.45, 0.45), 1, "median", "std"
    )

    assert mean_center[0] == pytest.approx(2.5)
    assert median_center[0] == pytest.approx(1.5)


def test_clip_scale_std_and_mad_std_select_different_samples():
    """MAD clipping is robust to a bright outlier that inflates sample std."""
    samples = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]])

    std_clip, _ = _ring_statistics(samples, "mean", (3.0, 3.0), 1, "median", "std")
    mad_clip, _ = _ring_statistics(samples, "mean", (3.0, 3.0), 1, "median", "mad_std")

    assert std_clip[0] == pytest.approx(21.2)
    assert mad_clip[0] == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("statistic", "location"),
    [
        (None, 26.75),
        ("mean", 26.75),
        ("median", 3.0),
        ("mode", -44.5),
    ],
)
def test_azimuthal_renormalize_uses_selected_location_and_sample_std(
    statistic, location
):
    """Renormalization changes location while retaining sample-std width."""
    image, center = _diagonal_ring_image((100.0, 1.0, 2.0, 4.0))
    kwargs = {} if statistic is None else {"statistic": statistic}

    result = azimuthal_renormalize(
        image,
        center,
        nrad=1,
        ntheta=4,
        nsigma=1.0,
        subpixels=1,
        **kwargs,
    )

    spread = np.std([100.0, 1.0, 2.0, 4.0], ddof=1)
    expected = 0.5 + (100.0 - location) / (2.0 * spread)
    assert result[2, 2] == pytest.approx(expected)


def test_azimuthal_normalize_mode_uses_same_mmm_location():
    """Azimuthal division applies the mode estimator to the polar ring."""
    image, center = _diagonal_ring_image((100.0, 1.0, 2.0, 4.0))

    result = azimuthal_normalize(
        image, center, nrad=1, ntheta=4, statistic="mode", subpixels=1
    )

    assert result[2, 2] == pytest.approx(100.0 / -44.5)


@pytest.mark.parametrize("function", [azimuthal_normalize, azimuthal_renormalize])
def test_both_azimuthal_functions_accept_shared_clipping_options(function):
    """Both polar APIs accept pair thresholds and median/MAD clipping."""
    image, center = _diagonal_ring_image((100.0, 1.0, 2.0, 4.0))
    kwargs = {
        "nrad": 1,
        "ntheta": 4,
        "statistic": "median",
        "sigma": (1.0, 1.0),
        "maxiters": None,
        "clip_center": "median",
        "clip_scale": "mad_std",
        "subpixels": 1,
    }
    if function is azimuthal_renormalize:
        kwargs["nsigma"] = 1.0

    result = function(image, center, **kwargs)

    assert result.dtype == np.float64
    assert np.isfinite(result).any()


def test_azimuthal_renormalize_clipping_keeps_sample_std_width():
    """Clipping changes location but output width remains retained sample std."""
    image, center = _diagonal_ring_image((100.0, 1.0, 2.0, 4.0))

    result = azimuthal_renormalize(
        image,
        center,
        nrad=1,
        ntheta=4,
        nsigma=1.0,
        statistic="median",
        sigma=1.0,
        maxiters=1,
        subpixels=1,
    )

    retained = np.array([1.0, 2.0, 4.0])
    expected_std = np.std(retained, ddof=1)
    expected = 0.5 + (100.0 - 2.0) / (2.0 * expected_std)
    assert result[2, 2] == pytest.approx(expected)


def test_annular_normalize_uses_pixel_centers_and_half_open_edges():
    """Direct annuli include the inner edge and exclude the outer edge."""
    image = np.array([[1.0, 2.0, 4.0, 8.0]])

    result = annular_normalize(image, (0.0, 0.0), radii=[0.0, 1.0, 3.0, 4.0])

    expected = np.array([[1.0, 2.0 / 3.0, 4.0 / 3.0, 1.0]])
    assert_allclose(result, expected)


def test_annular_normalize_clipping_changes_divisor_only():
    """Rejected bright pixels remain in the numerator of direct normalization."""
    image = np.array([[1.0, 2.0], [3.0, 100.0]])
    original = image.copy()

    result = annular_normalize(
        image,
        (0.5, 0.5),
        radii=[0.0, 2.0],
        statistic="mean",
        sigma=1.0,
        maxiters=1,
    )

    assert_allclose(result, image / 2.0)
    assert_array_equal(image, original)


def test_annular_normalize_median_handles_bright_outlier():
    """The even median selects the central pair, leaving a bright numerator."""
    image = np.array([[1.0, 1.0], [1.0, 1e100]])

    result = annular_normalize(image, (0.5, 0.5), radii=[0.0, 2.0])

    assert_allclose(result, image)


def test_annular_normalize_marks_empty_and_zero_divisor_pixels_nan():
    """Unrepresented intervals and zero-valued divisors remain undefined."""
    empty = annular_normalize(
        np.array([[1.0, 2.0, 4.0]]),
        (0.0, 0.0),
        radii=[0.0, 0.5, 0.75, 1.5],
    )
    zero = annular_normalize(
        np.array([[0.0, 1.0, 0.0]]),
        (0.0, 0.0),
        radii=[0.0, 1.0, 3.0],
    )

    assert_allclose(empty[0, :2], [1.0, 1.0])
    assert np.isnan(empty[0, 2])
    assert np.isnan(zero[0, 0])
    assert_allclose(zero[0, 1:], [2.0, 0.0])


def test_annular_normalize_all_masked_is_nan():
    """A bin with no finite unmasked pixels has no divisor."""
    image = np.ones((2, 2))
    mask = np.ones_like(image, dtype=bool)

    result = annular_normalize(image, (0.5, 0.5), radii=[0.0, 2.0], mask=mask)

    assert np.isnan(result).all()


def test_annular_normalize_callable_is_scalar_and_does_not_mutate_image():
    """Custom annular estimators receive a private finite sample array."""
    image = np.array([[1.0, 2.0, 4.0, 8.0]])
    original = image.copy()
    calls: list[np.ndarray] = []

    def statistic(values: np.ndarray) -> float:
        calls.append(values.copy())
        result = float(np.mean(values))
        values[...] = -100.0
        return result

    result = annular_normalize(image, (0.0, 0.0), radii=[0.0, 4.0], statistic=statistic)

    assert len(calls) == 1
    assert calls[0].ndim == 1
    assert np.isfinite(calls[0]).all()
    assert_allclose(result, image / 3.75)
    assert_array_equal(image, original)


@pytest.mark.parametrize(
    "level",
    [np.finfo(np.float64).max, np.nextafter(0.0, 1.0)],
)
@pytest.mark.parametrize("subpixels", [1, 10])
def test_azimuthal_normalize_extreme_constant_images_remain_unity(level, subpixels):
    """Overflow-safe running means preserve finite constant normalization."""
    image = np.full((9, 9), level, dtype=np.float64)

    result = azimuthal_normalize(
        image,
        (4.0, 4.0),
        nrad=3,
        ntheta=8,
        statistic="mean",
        subpixels=subpixels,
    )

    finite = np.isfinite(result)
    assert finite.any()
    assert_allclose(result[finite], 1.0)


def test_ring_statistics_large_opposite_signs_have_representable_std():
    """Centering before variance avoids overflow for large opposite signs."""
    value = 1e307
    level, std = _ring_statistics(
        np.array([[-value, value]]),
        "mean",
        None,
        1,
        "mean",
        "std",
        return_std=True,
    )

    assert level[0] == pytest.approx(0.0)
    assert np.isfinite(std[0])
    assert std[0] == pytest.approx(np.sqrt(2.0) * value)


def test_ring_statistics_preserves_small_spread_on_bright_pedestal():
    """Sample standard deviation retains sub-pixel spread on a large baseline."""
    samples = np.array([[1e12, 1e12 + 0.125, 1e12 + 0.25]])

    level, std = _ring_statistics(
        samples, "mean", None, 1, "mean", "std", return_std=True
    )

    assert level[0] == pytest.approx(1e12 + 0.125)
    assert std[0] == pytest.approx(0.125, rel=1e-12, abs=1e-12)


def test_ring_statistics_median_keeps_faint_pair_below_bright_outlier():
    """An even raw median uses the central faint pair despite a bright outlier."""
    samples = np.array([[1.0, 2.0, 3.0, 1e30]])

    level, _ = _ring_statistics(samples, "median", None, 1, "mean", "std")

    assert level[0] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"statistic": "unknown"},
        {"statistic": 3},
        {"sigma": -1.0},
        {"sigma": (1.0, 2.0, 3.0)},
        {"sigma": (1.0, np.nan)},
        {"maxiters": 0},
        {"maxiters": True},
        {"clip_center": "mode"},
        {"clip_scale": "mad"},
    ],
)
def test_azimuthal_normalize_rejects_invalid_statistic_options(kwargs):
    """Invalid estimator and clipping options fail before sampling."""
    with pytest.raises(ValueError):
        azimuthal_normalize(
            np.ones((5, 5)),
            (2.0, 2.0),
            nrad=1,
            ntheta=4,
            subpixels=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "radii",
    [[], [0.0], [1.0, 0.0], [0.0, np.nan], [-1.0, 0.0], [0.0, 1.0, 1.0]],
)
def test_annular_normalize_rejects_invalid_edges(radii):
    """Annular edges must be finite, nonnegative, and strictly increasing."""
    with pytest.raises(ValueError):
        annular_normalize(np.ones((3, 3)), (1.0, 1.0), radii=radii)
