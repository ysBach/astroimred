"""Behavior tests for the comet image enhancement helpers."""

import hashlib
import lzma
from importlib.resources import files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred.viz import (
    azimuthal_normalize,
    azimuthal_renormalize,
    enhance_rho,
    radially_variable_filter,
)


def _diagonal_ring_image(
    values: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return a small image whose radius-one polar samples are known."""
    image = np.ones((7, 7), dtype=np.float64)
    center = (3.0, 3.0)
    for (row, column), value in zip(
        ((2, 2), (2, 4), (4, 4), (4, 2)), values, strict=True
    ):
        image[row, column] = value
    return image, center


def test_enhance_rho_uses_fractional_center_without_global_rescaling():
    """The inverse-rho helper uses the supplied fractional x/y center."""
    yy, xx = np.indices((5, 5), dtype=np.float64)
    image = np.ones((5, 5), dtype=np.float32)
    original = image.copy()

    result = enhance_rho(image, center=(2.25, 1.5), subpixels=1)

    assert result.dtype == np.float64
    assert_allclose(result, np.hypot(xx - 2.25, yy - 1.5))
    assert_array_equal(image, original)
    assert not np.shares_memory(result, image)


def test_enhance_rho_averages_radius_over_subpixel_midpoints():
    """The radius factor is the mean of midpoint radii, not center radius."""
    image = np.ones((2, 2), dtype=np.float64)
    center = (0.25, 0.75)
    offsets = (-0.25, 0.25)
    expected = np.array(
        [
            [
                np.mean(
                    [
                        np.hypot(x + dx - center[0], y + dy - center[1])
                        for dx in offsets
                        for dy in offsets
                    ]
                )
                for x in range(2)
            ]
            for y in range(2)
        ]
    )

    result = enhance_rho(image, center, subpixels=2)

    assert_allclose(result, expected)


def test_enhance_rho_propagates_nan_and_masked_pixels():
    """Invalid source pixels remain undefined in the direct transform."""
    image = np.ones((5, 5), dtype=np.float64)
    image[0, 0] = np.nan
    mask = np.zeros_like(image, dtype=bool)
    mask[4, 4] = True

    result = enhance_rho(image, center=(2.0, 2.0), subpixels=1, mask=mask)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[4, 4])
    assert result[2, 2] == 0.0


def test_enhance_rho_accepts_a_masked_array():
    """A masked-array pixel is treated as missing in the float64 result."""
    image = np.ma.array(np.ones((5, 5), dtype=np.float32), mask=False)
    image.mask[1, 4] = True

    result = enhance_rho(image, center=(2.0, 2.0), subpixels=1)

    assert result.dtype == np.float64
    assert np.isnan(result[1, 4])
    assert not image.mask[2, 2]


@pytest.mark.parametrize("power", [-2.0, -0.5, 0.5, 2.0])
def test_enhance_rho_compensates_the_modeled_profile_power(power: float) -> None:
    """A power-law image becomes constant under pixel-center compensation."""
    yy, xx = np.indices((5, 5), dtype=np.float64)
    center = (2.25, 1.5)
    radius = np.hypot(xx - center[0], yy - center[1])

    result = enhance_rho(7.0 * radius**power, center, power=power, subpixels=1)

    assert_allclose(result, 7.0)


def test_enhance_rho_averages_powered_midpoints() -> None:
    """Square correction uses the mean square radius, not the squared mean."""
    yy, xx = np.indices((3, 4), dtype=np.float64)
    center = (1.25, 0.75)
    # Two midpoints at +/-0.25 per axis add 2 * 0.25**2 to mean rho**2.
    expected = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + 0.125

    result = enhance_rho(np.ones((3, 4)), center, power=-2, subpixels=2)

    assert_allclose(result, expected)


def test_enhance_rho_zero_power_preserves_values_and_combines_masks() -> None:
    """A flat modeled profile leaves finite pixels unchanged, including rho=0."""
    image = np.ma.array(np.arange(9.0).reshape(3, 3) - 4, mask=False)
    image.mask[0, 0] = True
    image.data[0, 1] = np.inf
    mask = np.zeros((3, 3), dtype=bool)
    mask[2, 2] = True
    expected = image.data.copy()
    expected[image.mask | mask | ~np.isfinite(expected)] = np.nan

    result = enhance_rho(image, (1, 1), power=0, mask=mask)

    assert_array_equal(result, expected)
    assert not np.shares_memory(result, image.data)


@pytest.mark.parametrize("subpixels", [1, 3])
def test_enhance_rho_singular_sample_is_nan(subpixels: int) -> None:
    """A positive profile exponent cannot use a midpoint exactly at rho=0."""
    result = enhance_rho(np.ones((3, 3)), (1, 1), power=0.5, subpixels=subpixels)

    assert np.isnan(result[1, 1])
    assert np.isfinite(result[0, 0])


def test_enhance_rho_negative_power_keeps_zero_radius_zero() -> None:
    """A negative profile exponent gives zero correction at the nucleus."""
    result = enhance_rho(np.ones((3, 3)), (1, 1), power=-0.5, subpixels=1)

    assert result[1, 1] == 0.0


def test_enhance_rho_power_average_avoids_overflowing_finite_sum() -> None:
    """Large finite correction samples are averaged without summing overflow."""
    # Every sampled rho is 2**(-3/2), so this correction is 2**1022.
    # Four copies would overflow an ordinary sum, but their mean is finite.
    power = 2044.0 / 3.0
    expected = np.hypot(0.25, 0.25) ** (-power)

    result = enhance_rho(np.ones((1, 1)), (0, 0), power=power, subpixels=2)

    assert np.isfinite(result[0, 0])
    assert_allclose(result[0, 0], expected)


def test_enhance_rho_nonfinite_correction_or_output_is_nan() -> None:
    """Unrepresentable factors and multiplied intensities are undefined."""
    with np.errstate(all="raise"):
        factor_overflow = enhance_rho(np.ones((1, 3)), (0, 0), power=-2048, subpixels=1)
        output_overflow = enhance_rho(
            np.full((1, 3), np.finfo(float).max), (0, 0), subpixels=1
        )

    assert np.isnan(factor_overflow[0, 2])
    assert np.isnan(output_overflow[0, 2])


@pytest.mark.parametrize("power", [True, "-1", [-1], None, 1j, np.inf, np.nan])
def test_enhance_rho_rejects_invalid_profile_power(power: object) -> None:
    """The profile exponent must be one finite real scalar."""
    with pytest.raises(ValueError, match="power"):
        enhance_rho(np.ones((3, 3)), (1, 1), power=power)


def test_enhance_rho_zero_power_still_validates_sampling() -> None:
    """Identity enhancement does not silently accept an invalid sampling grid."""
    with pytest.raises(ValueError, match="subpixels"):
        enhance_rho(np.ones((3, 3)), (1, 1), power=0, subpixels=0)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_azimuthal_normalize_constant_image_is_unity_inside_rings(statistic):
    """A constant image stays one after division by its radial statistic."""
    image = np.full((5, 5), 7.0, dtype=np.float32)
    original = image.copy()

    result = azimuthal_normalize(
        image,
        center=(2.0, 2.0),
        nrad=2,
        ntheta=8,
        statistic=statistic,
        subpixels=1,
    )

    assert result.dtype == np.float64
    assert np.isnan(result[2, 2])
    assert np.isnan(result[0, 0])
    finite = np.isfinite(result)
    assert finite[2, 3]
    assert finite[0, 2]
    assert_allclose(result[finite], 1.0)
    assert_array_equal(image, original)


def test_azimuthal_mean_rejects_one_outlier_but_median_does_not_shift():
    """One-pass sigma rejection recovers the baseline radial mean."""
    image, center = _diagonal_ring_image((100.0, 1.0, 1.0, 1.0))

    median = azimuthal_normalize(
        image, center, nrad=1, ntheta=4, statistic="median", subpixels=1
    )
    mean_rejected = azimuthal_normalize(
        image,
        center,
        nrad=1,
        ntheta=4,
        statistic="mean",
        sigma=1.0,
        subpixels=1,
    )
    mean_unfiltered = azimuthal_normalize(
        image, center, nrad=1, ntheta=4, statistic="mean", subpixels=1
    )

    assert_allclose(mean_rejected, median, equal_nan=True)
    assert np.nanmax(mean_rejected) == pytest.approx(100.0)
    assert np.nanmax(mean_unfiltered) == pytest.approx(100.0 / 25.75)


def test_azimuthal_normalize_supports_clipped_median_and_rejects_unknown_statistic():
    """Every supported statistic can use clipping; unknown names fail clearly."""
    image = np.ones((5, 5), dtype=np.float64)

    result = azimuthal_normalize(
        image,
        (2.0, 2.0),
        nrad=1,
        ntheta=4,
        statistic="median",
        sigma=2.0,
        subpixels=1,
    )
    assert np.isfinite(result).any()
    assert_allclose(result[np.isfinite(result)], 1.0)
    with pytest.raises(ValueError):
        azimuthal_normalize(
            image,
            (2.0, 2.0),
            nrad=1,
            ntheta=4,
            statistic="unknown",
            subpixels=1,
        )


def test_azimuthal_normalize_marks_missing_polar_cells_nan():
    """A missing polar cell invalidates finite Cartesian pixels using it."""
    image = np.ones((7, 7), dtype=np.float64)
    center = (3.0, 3.0)
    image[2, 2] = np.nan
    mask = np.zeros_like(image, dtype=bool)
    mask[2, 4] = True

    result = azimuthal_normalize(
        image, center, nrad=1, ntheta=4, statistic="median", subpixels=1, mask=mask
    )

    assert np.isnan(result[3, 1])
    assert np.isnan(result[3, 5])
    assert np.isfinite(result[4, 4])


def test_azimuthal_renormalize_maps_range_without_clipping():
    """Mean-plus/minus-sigma bounds may produce values outside zero to one."""
    image, center = _diagonal_ring_image((1.0, 3.0, 1.0, 3.0))

    result = azimuthal_renormalize(
        image, center, nrad=1, ntheta=4, nsigma=0.5, subpixels=1
    )

    spread = np.std([1.0, 3.0, 1.0, 3.0], ddof=1)
    lower = 2.0 - 0.5 * spread
    upper = 2.0 + 0.5 * spread
    expected_low = (1.0 - lower) / (upper - lower)
    expected_high = (3.0 - lower) / (upper - lower)
    assert np.nanmin(result) == pytest.approx(expected_low)
    assert np.nanmax(result) == pytest.approx(expected_high)
    assert np.nanmin(result) < 0.0
    assert np.nanmax(result) > 1.0


@pytest.mark.parametrize("level", [4.0, 1e-100])
def test_azimuthal_renormalize_zero_spread_is_undefined(level):
    """A row with no azimuthal spread has no renormalization range."""
    image = np.full((21, 21), level, dtype=np.float64)

    result = azimuthal_renormalize(image, (10.0, 10.0), nrad=8, ntheta=48, nsigma=3.0)

    assert np.isnan(result).all()


@pytest.mark.parametrize("statistic", ["mean", "median"])
@pytest.mark.parametrize("level", [0.0, -2.0])
def test_azimuthal_division_retains_negative_values_and_marks_zero_undefined(
    statistic, level
):
    """Zero divisors are undefined; negative finite rings remain valid."""
    image = np.full((9, 9), level)
    result = azimuthal_normalize(image, (4, 4), nrad=3, ntheta=16, statistic=statistic)
    if level == 0:
        assert np.isnan(result).all()
    else:
        assert np.isfinite(result).any()
        assert_allclose(result[np.isfinite(result)], 1.0)


@pytest.mark.parametrize(
    ("image", "center", "kwargs", "error"),
    [
        (np.ones(3), (0, 0), {}, ValueError),
        (np.empty((0, 3)), (0, 0), {}, ValueError),
        (np.ones((3, 3), dtype=complex), (1, 1), {}, TypeError),
        (np.ones((3, 3)), (np.nan, 1), {}, ValueError),
        (np.ones((3, 3)), (3, 1), {}, ValueError),
        (np.ones((3, 3)), (1, 1), {"mask": np.zeros((2, 2))}, ValueError),
        (np.ones((3, 3)), (1, 1), {"subpixels": True}, ValueError),
    ],
)
def test_enhancement_rejects_undefined_geometry(image, center, kwargs, error):
    """Reject inputs that cannot define a real image sampling geometry."""
    with pytest.raises(error):
        enhance_rho(image, center, **kwargs)


def test_rvsf_collapsed_kernel_retains_all_eight_stencil_terms():
    """A zero-width kernel cancels a flat field through all eight neighbors."""
    image = np.full((5, 5), 3.0, dtype=np.float32)
    original = image.copy()

    result = radially_variable_filter(
        image, (2.0, 2.0), a=0.0, b=0.0, n=1.0, subpixels=1
    )

    assert result.dtype == np.float64
    assert_allclose(result, 0.0)
    assert_array_equal(image, original)


def test_rvsf_default_quadrature_preserves_a_flat_field():
    """The default 10-by-10 width sampling keeps a valid flat field at zero."""
    image = np.full((9, 9), 3.0, dtype=np.float32)

    result = radially_variable_filter(image, (4.0, 4.0), a=1.0, b=0.0, n=1.0)

    assert_allclose(result[1:-1, 1:-1], 0.0)
    assert np.isnan(result[0]).all()
    assert np.isnan(result[-1]).all()
    assert np.isnan(result[:, 0]).all()
    assert np.isnan(result[:, -1]).all()


def test_rvsf_uses_edge_and_corner_coefficients():
    """An impulse exercises the four edge and four corner coefficients."""
    image = np.zeros((7, 7), dtype=np.float64)
    image[3, 3] = 2.0

    result = radially_variable_filter(
        image, (3.0, 3.0), a=1.0, b=0.0, n=1.0, subpixels=1
    )

    assert result[3, 3] == pytest.approx(2048.0)
    assert result[3, 4] == pytest.approx(-384.0)
    assert result[4, 4] == pytest.approx(-128.0)
    assert result[1, 1] == pytest.approx(0.0)
    assert np.isnan(result[0]).all()
    assert np.isnan(result[:, 0]).all()


def test_rvsf_propagates_missing_stencil_samples():
    """Any masked sample in an eight-neighbor stencil makes the target NaN."""
    image = np.ones((7, 7), dtype=np.float64)
    mask = np.zeros_like(image, dtype=bool)
    mask[3, 3] = True

    result = radially_variable_filter(
        image, (3.0, 3.0), a=1.0, b=0.0, n=1.0, subpixels=1, mask=mask
    )

    assert np.isnan(result[3, 3])
    assert np.isnan(result[2, 2])
    assert np.isfinite(result[1, 1])


def test_rvsf_log_option_matches_log10_input():
    """The log option applies log10 before filtering."""
    image = np.arange(1.0, 50.0).reshape(7, 7)
    kwargs = {"center": (3.0, 3.0), "a": 1.0, "b": 0.0, "n": 1.0, "subpixels": 1}

    logged = radially_variable_filter(image, log=True, **kwargs)
    explicit = radially_variable_filter(np.log10(image), **kwargs)

    assert_allclose(logged, explicit, equal_nan=True)


def test_rvsf_subpixel_quadrature_matches_quadratic_stencil_oracle():
    """A quadratic field exposes variable widths averaged over subpixels."""
    yy, xx = np.indices((11, 11), dtype=np.float64)
    image = xx**2 + yy**2
    center = (5.0, 5.0)
    offsets = (np.arange(4, dtype=np.float64) + 0.5) / 4.0 - 0.5
    radii = np.array(
        [np.hypot(dx, dy) for dy in offsets for dx in offsets], dtype=np.float64
    )
    steps = np.floor(0.5 + 0.5 + 2.0 * radii).astype(np.intp)
    expected = -1280.0 * np.mean(steps**2)

    result = radially_variable_filter(
        image,
        center,
        a=0.5,
        b=2.0,
        n=1.0,
        subpixels=4,
    )

    assert result[5, 5] == pytest.approx(expected)


def test_rvsf_allows_nonpositive_values_only_when_masked_for_log():
    """Masked nonpositive pixels are invalid samples rather than log errors."""
    image = np.ones((7, 7), dtype=np.float64)
    image[3, 3] = -1.0
    mask = np.zeros_like(image, dtype=bool)
    mask[3, 3] = True

    result = radially_variable_filter(
        image,
        (3.0, 3.0),
        a=1.0,
        b=0.0,
        n=1.0,
        log=True,
        subpixels=1,
        mask=mask,
    )

    assert np.isnan(result[3, 3])


def test_rvsf_log_rejects_unmasked_nonpositive_values():
    """The log option rejects zero or negative valid samples."""
    image = np.ones((7, 7), dtype=np.float64)
    image[3, 3] = 0.0

    with pytest.raises(ValueError):
        radially_variable_filter(
            image, (3.0, 3.0), a=1.0, b=0.0, n=1.0, log=True, subpixels=1
        )


def test_packaged_comet_example_decompresses_to_the_original_byte_stream():
    """The compressed example preserves all original FITS bytes."""
    resource = files("astroimred.viz").joinpath("data/CometCIEF_test.fits.xz")

    original = lzma.decompress(resource.read_bytes())

    assert len(original) == 2_574_720
    assert hashlib.sha256(original).hexdigest() == (
        "b9459e7484cb60e72ac53b14a620eef6d73449a2d6bd1f99e061bda67d114496"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": -1.0, "b": 0.0, "n": 1.0},
        {"a": 0.0, "b": -1.0, "n": 1.0},
        {"a": 0.0, "b": 0.0, "n": 0.0},
        {"a": 0.0, "b": 0.0, "n": np.nan},
        {"a": np.inf, "b": 0.0, "n": 1.0},
    ],
)
def test_rvsf_rejects_invalid_kernel_parameters(kwargs):
    """Kernel terms must be finite, nonnegative, and have positive n."""
    image = np.ones((5, 5), dtype=np.float64)

    with pytest.raises(ValueError):
        radially_variable_filter(image, (2.0, 2.0), **kwargs, subpixels=1)
