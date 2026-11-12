"""Tests for opt-in radial enhancement statistics and estimation masks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred.viz import (
    EnhancementResult,
    NormStats,
    annular_normalize,
    azimuthal_normalize,
    azimuthal_renormalize,
)


def _result_image(shape: tuple[int, int] = (9, 9)) -> np.ndarray:
    """Return a positive, asymmetric image for whole-output parity checks."""
    yy, xx = np.indices(shape, dtype=np.float64)
    return 10.0 + 0.5 * yy + 0.25 * xx + 0.01 * xx * yy


def test_stats_are_opt_in_and_expose_public_dataclasses():
    """The default stays an ndarray, while opt-in diagnostics are public."""
    image = _result_image()
    original = image.copy()

    default = azimuthal_normalize(image, (4.0, 4.0), nrad=3, ntheta=8, subpixels=1)
    wrapped = azimuthal_normalize(
        image,
        (4.0, 4.0),
        nrad=3,
        ntheta=8,
        subpixels=1,
        return_stats=True,
    )

    assert isinstance(default, np.ndarray)
    assert isinstance(wrapped, EnhancementResult)
    assert isinstance(wrapped.stats, NormStats)
    assert_array_equal(wrapped.image, default)
    assert wrapped.image.dtype == np.float64
    assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (azimuthal_normalize, {"nrad": 3, "ntheta": 8, "subpixels": 1}),
        (
            azimuthal_renormalize,
            {"nrad": 3, "ntheta": 8, "nsigma": 1.5, "subpixels": 1},
        ),
        (annular_normalize, {"radii": [0.0, 2.0, 4.0, 8.0]}),
    ],
)
@pytest.mark.parametrize("statistic", ["mean", "median", "mode"])
def test_return_stats_preserves_whole_image_for_each_statistic(
    function, kwargs, statistic
):
    """Wrapping diagnostics does not alter mean, median, or mode output."""
    image = _result_image()
    default = function(image, (4.0, 4.0), statistic=statistic, **kwargs)
    wrapped = function(
        image,
        (4.0, 4.0),
        statistic=statistic,
        return_stats=True,
        **kwargs,
    )

    assert isinstance(wrapped, EnhancementResult)
    assert_array_equal(wrapped.image, default)


@pytest.mark.parametrize("statistic", ["mean", "mode"])
@pytest.mark.parametrize("kind", ["mixed_sign", "bright_pedestal", "extreme"])
def test_polar_stats_preserves_output_for_numerically_difficult_values(statistic, kind):
    """Statistics diagnostics preserve divisors through cancellation and scale extremes."""
    yy, xx = np.indices((9, 9), dtype=np.float64)
    if kind == "mixed_sign":
        image = 5.0 + 6.0 * np.where((xx.astype(int) + yy.astype(int)) % 2, -1.0, 1.0)
        image += 0.1 * xx + 0.2 * yy
    elif kind == "bright_pedestal":
        image = 1.0e12 + ((xx + 2.0 * yy) % 3.0) * 0.125
    else:
        image = np.full((9, 9), 1.0e300)
        image[3, 3] = -1.0e300

    default = azimuthal_normalize(
        image,
        (4.0, 4.0),
        nrad=3,
        ntheta=7,
        statistic=statistic,
        subpixels=2,
    )
    wrapped = azimuthal_normalize(
        image,
        (4.0, 4.0),
        nrad=3,
        ntheta=7,
        statistic=statistic,
        subpixels=2,
        return_stats=True,
    )

    assert_array_equal(wrapped.image, default)
    assert np.isfinite(wrapped.stats.statistic).any()


def test_polar_renormalize_combined_masks_keep_estimator_only_pixel_output():
    """Hard and estimation masks remain distinct in polar renormalization."""
    image = _result_image((7, 7))
    original = image.copy()
    hard_mask = np.zeros_like(image, dtype=bool)
    hard_mask[2, 2] = True
    statistic_mask = np.zeros_like(image, dtype=bool)
    statistic_mask[3, 2] = True

    default = azimuthal_renormalize(
        image,
        (3.0, 3.0),
        nrad=2,
        ntheta=4,
        nsigma=1.0,
        subpixels=2,
        mask=hard_mask,
        statistic_mask=statistic_mask,
    )
    wrapped = azimuthal_renormalize(
        image,
        (3.0, 3.0),
        nrad=2,
        ntheta=4,
        nsigma=1.0,
        subpixels=2,
        mask=hard_mask,
        statistic_mask=statistic_mask,
        return_stats=True,
    )

    assert_array_equal(wrapped.image, default)
    assert np.isnan(wrapped.image[2, 2])
    assert np.isfinite(wrapped.image[3, 2])
    assert np.any(wrapped.stats.n_valid < wrapped.stats.n_samples)
    assert np.all(wrapped.stats.n_used <= wrapped.stats.n_valid)
    assert_array_equal(image, original)


def test_polar_stats_reports_counts_metadata_and_coverage():
    """Polar diagnostics use one opportunity per angle and sample spread."""
    image = np.full((7, 7), 5.0)
    result = azimuthal_normalize(
        image,
        (3.0, 3.0),
        nrad=2,
        ntheta=4,
        subpixels=1,
        statistic="mean",
        return_stats=True,
    )
    stats = result.stats

    assert stats.sampling == "polar"
    assert stats.bin_edges is None
    assert_allclose(stats.radius, [1.0, 2.0])
    assert_allclose(stats.statistic, [5.0, 5.0])
    assert_allclose(stats.std, [0.0, 0.0])
    assert_array_equal(stats.n_samples, [4, 4])
    assert_array_equal(stats.n_valid, [4, 4])
    assert_array_equal(stats.n_used, [4, 4])
    assert_array_equal(stats.n_rejected, [0, 0])
    assert_allclose(stats.coverage, [1.0, 1.0])


def test_polar_stats_reports_partial_coverage_at_image_boundary():
    """Polar opportunity counts remain fixed while boundary cells go missing."""
    result = azimuthal_normalize(
        np.ones((5, 5)),
        (0.0, 0.0),
        nrad=2,
        ntheta=8,
        subpixels=1,
        statistic="mean",
        return_stats=True,
    )
    stats = result.stats

    assert_array_equal(stats.n_samples, [8, 8])
    assert np.all((stats.n_valid >= 0) & (stats.n_valid <= stats.n_samples))
    assert np.any(stats.n_valid < stats.n_samples)
    assert_allclose(stats.coverage, stats.n_valid / stats.n_samples)


def test_polar_statistic_mask_excludes_cells_but_keeps_valid_output_samples():
    """An estimation mask changes counts without acting as a hard output mask."""
    image = _result_image((7, 7))
    original = image.copy()
    statistic_mask = np.zeros_like(image, dtype=bool)
    statistic_mask[3, 2] = True

    result = azimuthal_normalize(
        image,
        (3.0, 3.0),
        nrad=1,
        ntheta=4,
        statistic="mean",
        subpixels=2,
        statistic_mask=statistic_mask,
        return_stats=True,
    )
    stats = result.stats

    assert stats.n_samples[0] == 4
    assert 0 <= stats.n_valid[0] < stats.n_samples[0]
    assert stats.n_used[0] == stats.n_valid[0]
    assert np.isfinite(result.image[3, 2])
    assert_array_equal(image, original)


def test_annular_stats_reports_edges_midpoints_counts_and_std():
    """Direct annuli report half-open pixel-center bins and sample spread."""
    image = np.array([[1.0, 2.0, 4.0, 8.0]])
    result = annular_normalize(
        image,
        (0.0, 0.0),
        radii=[0.0, 1.0, 3.0, 4.0],
        statistic="mean",
        return_stats=True,
    )
    stats = result.stats

    assert stats.sampling == "pixel_center"
    assert_allclose(stats.bin_edges, [0.0, 1.0, 3.0, 4.0])
    assert_allclose(stats.radius, [0.5, 2.0, 3.5])
    assert_allclose(stats.statistic, [1.0, 3.0, 8.0])
    assert_allclose(stats.std, [0.0, np.sqrt(2.0), 0.0])
    assert_array_equal(stats.n_samples, [1, 2, 1])
    assert_array_equal(stats.n_valid, [1, 2, 1])
    assert_array_equal(stats.n_used, [1, 2, 1])
    assert_array_equal(stats.n_rejected, [0, 0, 0])
    assert_allclose(stats.coverage, [1.0, 1.0, 1.0])
    assert_allclose(result.image, [[1.0, 2.0 / 3.0, 4.0 / 3.0, 1.0]])


def test_annular_stats_distinguishes_empty_nonfinite_and_singleton_bins():
    """Coverage is NaN for empty bins and zero for unavailable finite values."""
    image = np.array([[1.0, np.nan, 3.0]])
    result = annular_normalize(
        image,
        (0.0, 0.0),
        radii=[0.0, 0.5, 0.75, 1.5, 2.5],
        statistic="mean",
        return_stats=True,
    )
    stats = result.stats

    assert_array_equal(stats.n_samples, [1, 0, 1, 1])
    assert_array_equal(stats.n_valid, [1, 0, 0, 1])
    assert_array_equal(stats.n_used, [1, 0, 0, 1])
    assert_allclose(stats.statistic[[0, 3]], [1.0, 3.0])
    assert np.isnan(stats.statistic[1:3]).all()
    assert_allclose(stats.std[[0, 3]], [0.0, 0.0])
    assert np.isnan(stats.std[1:3]).all()
    assert_allclose(stats.coverage[[0, 2, 3]], [1.0, 0.0, 1.0])
    assert np.isnan(stats.coverage[1])
    assert np.isnan(result.image[0, 1])


def test_annular_stats_counts_clipping_and_keeps_outlier_in_numerator():
    """Statistics counts expose rejection while normalization keeps the source pixel."""
    image = np.array([[1.0, 2.0], [3.0, 100.0]])
    original = image.copy()
    result = annular_normalize(
        image,
        (0.5, 0.5),
        radii=[0.0, 2.0],
        statistic="mean",
        sigma=1.0,
        maxiters=1,
        return_stats=True,
    )
    stats = result.stats

    assert_array_equal(stats.n_samples, [4])
    assert_array_equal(stats.n_valid, [4])
    assert_array_equal(stats.n_used, [3])
    assert_array_equal(stats.n_rejected, [1])
    assert_allclose(stats.coverage, [1.0])
    assert stats.statistic[0] == pytest.approx(2.0)
    assert stats.std[0] == pytest.approx(1.0)
    assert_allclose(result.image, image / 2.0)
    assert_array_equal(image, original)


def test_annular_statistic_mask_preserves_output_where_hard_mask_does_not():
    """Both masks alter estimates, but only the hard mask erases output pixels."""
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = image.copy()
    exclusion = np.array([[True, False], [False, False]])
    estimate_only = annular_normalize(
        image,
        (0.5, 0.5),
        radii=[0.0, 2.0],
        statistic="mean",
        statistic_mask=exclusion,
        return_stats=True,
    )
    hard = annular_normalize(
        image,
        (0.5, 0.5),
        radii=[0.0, 2.0],
        statistic="mean",
        mask=exclusion,
        return_stats=True,
    )

    assert_array_equal(estimate_only.stats.n_samples, [4])
    assert_array_equal(estimate_only.stats.n_valid, [3])
    assert_array_equal(estimate_only.stats.n_used, [3])
    assert_array_equal(hard.stats.n_samples, [4])
    assert_array_equal(hard.stats.n_valid, [3])
    assert_array_equal(hard.stats.n_used, [3])
    assert np.isfinite(estimate_only.image[0, 0])
    assert np.isnan(hard.image[0, 0])
    assert_allclose(
        estimate_only.image,
        [[1.0 / 3.0, 2.0 / 3.0], [1.0, 4.0 / 3.0]],
    )
    assert_allclose(hard.image[~exclusion], estimate_only.image[~exclusion])
    assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (azimuthal_normalize, {"nrad": 1, "ntheta": 4, "subpixels": 1}),
        (
            azimuthal_renormalize,
            {"nrad": 1, "ntheta": 4, "nsigma": 1.0, "subpixels": 1},
        ),
        (annular_normalize, {"radii": [0.0, 2.0]}),
    ],
)
def test_all_estimator_masked_stats_has_no_divisor(function, kwargs):
    """Excluding every estimate yields NaN output and zero valid support."""
    image = np.arange(1.0, 5.0).reshape(2, 2)
    statistic_mask = np.ones_like(image, dtype=bool)
    result = function(
        image,
        (0.5, 0.5),
        statistic_mask=statistic_mask,
        return_stats=True,
        **kwargs,
    )
    stats = result.stats

    assert np.isnan(result.image).all()
    assert_array_equal(stats.n_valid, [0])
    assert_array_equal(stats.n_used, [0])
    assert_array_equal(stats.n_rejected, [0])
    assert np.isnan(stats.statistic[0])
    assert np.isnan(stats.std[0])
    assert stats.coverage[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (azimuthal_normalize, {"nrad": 1, "ntheta": 4, "subpixels": 1}),
        (
            azimuthal_renormalize,
            {"nrad": 1, "ntheta": 4, "nsigma": 1.0, "subpixels": 1},
        ),
        (annular_normalize, {"radii": [0.0, 2.0]}),
    ],
)
def test_stats_rejects_statistic_mask_with_wrong_shape(function, kwargs):
    """Estimation masks must align with the image geometry."""
    with pytest.raises(ValueError, match="statistic_mask"):
        function(
            np.ones((2, 2)),
            (0.5, 0.5),
            statistic_mask=np.zeros((3, 3), dtype=bool),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (azimuthal_normalize, {"nrad": 1, "ntheta": 4, "subpixels": 1}),
        (
            azimuthal_renormalize,
            {"nrad": 1, "ntheta": 4, "nsigma": 1.0, "subpixels": 1},
        ),
        (annular_normalize, {"radii": [0.0, 2.0]}),
    ],
)
def test_stats_rejects_non_bool_return_stats(function, kwargs):
    """The opt-in switch accepts booleans only."""
    with pytest.raises(ValueError, match="return_stats"):
        function(
            np.ones((2, 2)),
            (0.5, 0.5),
            return_stats=1,
            **kwargs,
        )
