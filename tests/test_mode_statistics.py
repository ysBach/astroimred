"""Tests for the shared mode estimators and their public callers."""

from __future__ import annotations

import astroapers as aap
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred._core.statistics import _mode_from_stats
from astroimred.phot.background import sky_fit
from astroimred.phot.radprof import radial_profile
from astroimred.viz import annular_normalize


@pytest.mark.parametrize("method", ["mmm", "iraf", "sex"])
def test_mode_vector_matches_scalar_results_and_fallback_boundaries(method):
    """Array evaluation matches scalar evaluation at each mode boundary."""
    mean = np.array([1.0, 4.0, 2.0, 3.3, 4.0])
    median = np.full(5, 3.0)
    std = np.array([1.0, 1.0, 0.0, np.nan, 1.0])
    vector_std = std if method == "sex" else None

    vector = _mode_from_stats(mean, median, vector_std, method=method)
    if method == "mmm":
        expected = np.array([7.0, 1.0, 5.0, 2.4, 1.0])
    elif method == "iraf":
        expected = np.array([1.0, 1.0, 2.0, 2.4, 1.0])
    else:
        expected = np.array([6.0, 3.0, 4.5, 2.55, 3.0])
    assert_allclose(vector, expected)

    for index, value in enumerate(vector):
        scalar_std = None if method != "sex" else float(std[index])
        scalar = _mode_from_stats(
            float(mean[index]),
            float(median[index]),
            scalar_std,
            method=method,
        )
        assert scalar == pytest.approx(value)


def test_mode_sex_requires_standard_deviation():
    """The SExtractor fallback cannot be evaluated without a spread."""
    with pytest.raises(ValueError, match="std"):
        _mode_from_stats(1.0, 1.0, method="sex")


def test_mode_sex_uses_strict_ratio_threshold_with_scalar_vector_parity():
    """The SExtractor median fallback starts strictly above ratio ``0.3``."""
    mean = np.array([0.3, np.nextafter(0.3, np.inf)])
    median = np.zeros(2)
    std = np.ones(2)

    vector = _mode_from_stats(mean, median, std, method="sex")

    assert_allclose(vector, [-0.45, 0.0], atol=0.0)
    for index, value in enumerate(vector):
        scalar = _mode_from_stats(
            float(mean[index]), float(median[index]), float(std[index]), method="sex"
        )
        assert scalar == pytest.approx(value)


def test_mode_rejects_unknown_method():
    """Only the documented mode policies are accepted."""
    with pytest.raises(ValueError, match="unknown"):
        _mode_from_stats(1.0, 1.0, method="histogram")


@pytest.mark.parametrize("method", ["mmm", "iraf", "sex"])
def test_mode_equal_extreme_moments_remain_finite(method):
    """Equal float64 extrema do not overflow during centered evaluation."""
    moments = np.array([np.finfo(np.float64).max, -np.finfo(np.float64).max])
    std = np.ones_like(moments) if method == "sex" else None

    result = _mode_from_stats(moments, moments, std, method=method)

    assert_array_equal(result, moments)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("method", ["mmm", "iraf", "sex"])
def test_mode_does_not_modify_input_statistics(method):
    """Mode evaluation keeps caller-owned location and spread arrays intact."""
    mean = np.array([1.0, 4.0, 2.0])
    median = np.array([3.0, 3.0, 3.0])
    std = np.array([1.0, 1.0, 0.0])
    original = (mean.copy(), median.copy(), std.copy())

    _mode_from_stats(mean, median, std if method == "sex" else None, method=method)

    assert_array_equal(mean, original[0])
    assert_array_equal(median, original[1])
    assert_array_equal(std, original[2])


@pytest.mark.parametrize(
    ("public_method", "helper_method"),
    [("MMM", "mmm"), ("iraf", "iraf"), ("sex", "sex")],
)
def test_sky_fit_modes_match_shared_statistics(public_method, helper_method):
    """Sky mode aliases use the shared formulas after no clipping."""
    sky = np.array([1.0, 2.0, 4.0, 8.0])
    row = sky_fit(
        sky,
        annulus=None,
        method=public_method,
        sky_clipper=None,
        to_table=False,
    )[0]
    mean = np.mean(sky)
    median = np.median(sky)
    std = np.std(sky, ddof=1)
    expected = _mode_from_stats(mean, median, std, method=helper_method)

    assert row["msky"] == pytest.approx(expected)
    assert row["ssky"] == pytest.approx(std)
    assert row["nsky"] == sky.size
    assert row["nrej"] == 0


@pytest.mark.parametrize(
    ("method", "expected"),
    [("mean", 26.75), ("median", 3.0)],
)
def test_sky_fit_mean_and_median_remain_direct_reductions(method, expected):
    """The mode helper does not alter the direct mean/median policies."""
    sky = np.array([1.0, 2.0, 4.0, 100.0])
    row = sky_fit(
        sky,
        annulus=None,
        method=method,
        sky_clipper=None,
        to_table=False,
    )[0]

    assert row["msky"] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["MMM", "IRAF", "sex"])
def test_sky_fit_modes_preserve_finite_large_singleton(method):
    """Public mode aliases retain a finite large singleton despite NaN spread."""
    sky = np.array([8e307])
    row = sky_fit(
        sky,
        annulus=None,
        method=method,
        sky_clipper=None,
        to_table=False,
    )[0]

    assert row["msky"] == pytest.approx(8e307)
    assert np.isfinite(row["msky"])
    assert np.isnan(row["ssky"])
    assert row["nsky"] == 1
    assert row["nrej"] == 0


def test_radial_profile_mmm_matches_sky_fit_for_the_same_annulus():
    """Radial profile forwards MMM and preserves the same annulus samples."""
    image = np.arange(81.0).reshape(9, 9)
    center = (4.0, 4.0)
    annulus = aap.CircAn(center, r_in=2.0, r_out=4.0)
    sky_row = sky_fit(
        image,
        annulus,
        method="MMM",
        sky_clipper=None,
        to_table=False,
    )[0]

    profile, center_value = radial_profile(
        image,
        center,
        radii=[3.0],
        thickness=2,
        method="MMM",
        sky_clipper=None,
    )

    row = profile.iloc[0]
    assert center_value == image[4, 4]
    assert row["mpix"] == pytest.approx(sky_row["msky"])
    assert row["spix"] == pytest.approx(sky_row["ssky"])
    assert row["npix"] == sky_row["nsky"]
    assert row["spix_n"] == pytest.approx(sky_row["ssky"] / np.sqrt(sky_row["nsky"]))


def test_annular_mode_profile_uses_the_same_finite_samples():
    """Direct annular mode diagnostics report the finite source samples."""
    image = np.array([[1.0, 2.0, 4.0, 8.0, np.nan]])
    original = image.copy()
    finite = image[0, :4]
    expected = _mode_from_stats(np.mean(finite), np.median(finite), method="mmm")

    result = annular_normalize(
        image,
        (0.0, 0.0),
        radii=[0.0, 4.5],
        statistic="mode",
        return_profile=True,
    )

    profile = result.profile
    assert profile.statistic[0] == pytest.approx(expected)
    assert profile.n_samples[0] == 5
    assert profile.n_valid[0] == finite.size
    assert profile.n_used[0] == finite.size
    assert profile.std[0] == pytest.approx(np.std(finite, ddof=1))
    assert_allclose(result.image[0, :4], finite / expected)
    assert np.isnan(result.image[0, 4])
    assert_array_equal(image, original)


def test_singleton_sky_and_visual_profile_keep_their_spread_conventions():
    """Photometric sky keeps NaN ddof-one spread while viz defines zero width."""
    sky_row = sky_fit(
        np.array([5.0]),
        annulus=None,
        method="mean",
        sky_clipper=None,
        to_table=False,
    )[0]
    viz = annular_normalize(
        np.array([[5.0]]),
        (0.0, 0.0),
        radii=[0.0, 1.0],
        statistic="mean",
        return_profile=True,
    )

    assert sky_row["msky"] == pytest.approx(5.0)
    assert np.isnan(sky_row["ssky"])
    assert viz.profile.statistic[0] == pytest.approx(5.0)
    assert viz.profile.std[0] == pytest.approx(0.0)
    assert viz.profile.n_valid[0] == 1
