"""
Tests for astroimred.phot.background module.

All expected values are analytically derived.
"""

import astroapers as aap
import numpy as np
import pytest
from astropy.nddata import CCDData
from numpy.testing import assert_allclose

from astroimred._core.astropy_helpers import sigma_clipper
from astroimred.phot.background import mmm_dao, quick_sky_circ, sky_fit


def _ellip_an(positions, a_in, a_out, b_out, theta=0, **kwargs):
    import astropy.units as u

    if hasattr(theta, "to_value"):
        theta = theta.to_value(u.rad)
    b_in = kwargs.pop("b_in", b_out * a_in / a_out)
    if kwargs:
        raise TypeError(f"unexpected keyword(s): {sorted(kwargs)}")
    return aap.EllipAn(
        positions,
        a_in=a_in,
        b_in=b_in,
        a_out=a_out,
        b_out=b_out,
        theta_in=theta,
    )


# =============================================================================
# Tests for sky_fit
# =============================================================================
class TestSkyFit:
    """Tests for sky_fit function."""

    def test_sky_fit_uniform_mean(self, uniform_100x100):
        """
        Test sky_fit with method='mean' on uniform array.

        Expected: msky = 10.0, ssky = 0.0
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert_allclose(result["ssky"][0], 0.0, atol=1e-10)

    def test_sky_fit_uniform_median(self, uniform_100x100):
        """
        Test sky_fit with method='median' on uniform array.

        Expected: msky = 10.0, ssky = 0.0
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method="median")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert_allclose(result["ssky"][0], 0.0, atol=1e-10)

    def test_sky_fit_uniform_sex(self, uniform_100x100):
        """
        Test sky_fit with method='sex' (SExtractor) on uniform array.

        For uniform array: mean = median = 10.0
        Since (mean - median)/std is undefined (std=0), should return median.
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method="sex")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)

    def test_sky_fit_with_noise(self, uniform_with_noise):
        """
        Test sky_fit on noisy data recovers approximate mean.

        Data: N(100, 10), method='mean'
        Expected: msky ≈ 100 (within several std/sqrt(n))
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method="mean")

        # Should be close to 100, allow 3-sigma tolerance
        # With ~500 pixels, std of mean ≈ 10/sqrt(500) ≈ 0.45
        assert_allclose(result["msky"][0], 100.0, atol=3.0)

    def test_sky_fit_nsky_nrej(self, uniform_100x100):
        """Test nsky and nrej are correctly reported."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=15)
        result = sky_fit(uniform_100x100, an, method="mean")

        # nsky should be positive
        assert result["nsky"][0] > 0
        # nrej should be 0 for uniform array (no sigma clipping rejects)
        assert result["nrej"][0] == 0

    def test_sky_fit_no_annulus(self, uniform_100x100):
        """
        Test sky_fit with annulus=None uses whole image.
        """
        result = sky_fit(uniform_100x100, annulus=None, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert result["nsky"][0] == 100 * 100  # whole image

    def test_sky_fit_no_annulus_with_mask(self, uniform_100x100):
        """Test sky_fit with annulus=None excludes externally masked pixels."""
        data = uniform_100x100.copy()
        data[0, 0] = 1000.0
        data[0, 1] = 1000.0
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True

        result = sky_fit(data, annulus=None, mask=mask, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert result["nsky"][0] == data.size - 2

    def test_sky_fit_no_annulus_with_ccddata_mask(self, uniform_100x100):
        """Test sky_fit with annulus=None excludes CCDData.mask pixels."""
        data = uniform_100x100.copy()
        data[0, 0] = 1000.0
        internal_mask = np.zeros_like(data, dtype=bool)
        internal_mask[0, 0] = True
        ccd = CCDData(data, unit="adu", mask=internal_mask)

        result = sky_fit(ccd, annulus=None, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert result["nsky"][0] == data.size - 1

    def test_sky_fit_no_annulus_combines_masks(self, uniform_100x100):
        """Test sky_fit with annulus=None combines CCDData and external masks."""
        data = uniform_100x100.copy()
        data[0, 0] = 1000.0
        data[0, 1] = 1000.0
        internal_mask = np.zeros_like(data, dtype=bool)
        internal_mask[0, 0] = True
        external_mask = np.zeros_like(data, dtype=bool)
        external_mask[0, 1] = True
        ccd = CCDData(data, unit="adu", mask=internal_mask)

        result = sky_fit(ccd, annulus=None, mask=external_mask, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert result["nsky"][0] == data.size - 2

    def test_sky_fit_iraf_method(self, uniform_with_noise):
        """
        Test sky_fit with method='iraf'.

        IRAF: if mean < median, use mean; else use 3*median - 2*mean
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method="iraf")

        # Should be close to 100
        assert_allclose(result["msky"][0], 100.0, atol=5.0)

    def test_sky_fit_mmm_method(self, uniform_with_noise):
        """
        Test sky_fit with method='mmm'.

        MMM: 3*median - 2*mean
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method="mmm")

        # Should be close to 100
        assert_allclose(result["msky"][0], 100.0, atol=5.0)

    def test_sky_fit_callable_method(self, uniform_100x100):
        """
        Test sky_fit with callable method.

        Custom method: return max of sky array.
        """

        def custom_method(skyarr, ssky):
            return np.max(skyarr)

        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method=custom_method)

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)

    def test_sky_fit_return_dict(self, uniform_100x100):
        """Test sky_fit returns dict when to_table=False."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method="mean", to_table=False)

        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert "msky" in result[0]

    def test_sky_fit_return_skyarr(self, uniform_100x100):
        """Test sky_fit returns sky array when return_skyarr=True."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result, skys = sky_fit(uniform_100x100, an, method="mean", return_skyarr=True)

        assert isinstance(skys, list)
        assert len(skys) == 1
        assert_allclose(skys[0], 10.0, rtol=1e-10)

    def test_sky_fit_return_dict_and_skyarr(self, uniform_100x100):
        """Test sky_fit with to_table=False and return_skyarr=True."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result, skys = sky_fit(
            uniform_100x100, an, method="mean", to_table=False, return_skyarr=True
        )

        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert isinstance(skys, list)
        assert_allclose(skys[0], 10.0, rtol=1e-10)

    def test_sky_fit_sky_clipper_none(self, uniform_100x100):
        """
        Test sky_fit with sky_clipper=None (no clipping applied).

        All pixels should be used; nrej should be 0.
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method="mean", sky_clipper=None)

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert result["nrej"][0] == 0

    def test_sky_fit_std_ddof(self, uniform_with_noise):
        """
        Test std_ddof parameter affects ssky.

        ddof=0 gives population std, ddof=1 gives sample std.
        They should differ for finite samples.
        """
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=30)
        result_ddof0 = sky_fit(uniform_with_noise, an, method="mean", std_ddof=0)
        result_ddof1 = sky_fit(uniform_with_noise, an, method="mean", std_ddof=1)

        # ddof=1 gives slightly larger std than ddof=0
        assert result_ddof1["ssky"][0] > result_ddof0["ssky"][0]

    def test_sky_fit_sex_skewed_uses_formula(self):
        """
        Test sky_fit 'sex' method branch logic.

        'sex' uses: median if (mean-med)/std > 0.3, else 2.5*med - 1.5*mean.
        This mirrors SExtractor: for symmetric data (small ratio) use the
        formula; for skewed data (large ratio) fall back to median.
        """
        rng = np.random.default_rng(0)
        sky = np.concatenate(
            [
                rng.normal(0.0, 1.0, 900),
                rng.normal(20.0, 1.0, 100),
            ]
        )
        sky_clipped = sigma_clipper(sky)
        std = np.std(sky_clipped, ddof=1)
        mean = np.mean(sky_clipped)
        med = np.median(sky_clipped)

        result = sky_fit(sky, annulus=None, method="sex")

        # Replicate _sky_fit branch logic exactly (note: condition selects median)
        if std > 0 and (mean - med) / std > 0.3:
            expected = med
        else:
            expected = 2.5 * med - 1.5 * mean
        assert_allclose(result["msky"][0], expected, rtol=1e-10)

    def test_sky_fit_invalid_method(self, uniform_100x100):
        """Test sky_fit raises ValueError for unknown method string."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        with pytest.raises(ValueError):
            sky_fit(uniform_100x100, an, method="unknown_method")

    def test_sky_fit_method_case_insensitive(self, uniform_100x100):
        """Test sky_fit method strings are case-insensitive."""
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result_lower = sky_fit(uniform_100x100, an, method="iraf")
        result_upper = sky_fit(uniform_100x100, an, method="IRAF")

        assert_allclose(result_lower["msky"][0], result_upper["msky"][0], rtol=1e-10)

    def test_sky_fit_ccddata_input(self, uniform_100x100):
        """Test sky_fit accepts CCDData input."""
        ccd = CCDData(uniform_100x100, unit="adu")
        an = aap.CircAn(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(ccd, an, method="mean")

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)

    def test_sky_fit_no_annulus_dict(self, uniform_100x100):
        """Test sky_fit with annulus=None and to_table=False."""
        result = sky_fit(uniform_100x100, annulus=None, method="mean", to_table=False)

        assert isinstance(result, list)
        assert_allclose(result[0]["msky"], 10.0, rtol=1e-10)

    def test_sky_fit_multiple_positions(self, uniform_100x100):
        """
        Test sky_fit with multi-position annulus returns one row per position.
        """
        positions = [(30, 30), (50, 50), (70, 70)]
        an = aap.CircAn(positions=positions, r_in=5, r_out=10)
        result = sky_fit(uniform_100x100, an, method="mean")

        assert len(result) == 3
        assert_allclose(result["msky"], 10.0, rtol=1e-10)


# =============================================================================
# Tests for quick_sky_circ
# =============================================================================
class TestQuickSkyCirc:
    """Tests for quick_sky_circ convenience function."""

    def test_quick_sky_circ_uniform(self, uniform_100x100):
        """Test quick_sky_circ on uniform array."""
        result = quick_sky_circ(uniform_100x100, pos=(50, 50), r_in=10, r_out=20)

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)

    def test_quick_sky_circ_with_mask(self, uniform_100x100):
        """Test quick_sky_circ passes mask through to sky_fit."""
        mask = np.zeros_like(uniform_100x100, dtype=bool)
        mask[50, 60] = True

        result_nomask = quick_sky_circ(uniform_100x100, pos=(50, 50), r_in=10, r_out=20)
        result_masked = quick_sky_circ(
            uniform_100x100, pos=(50, 50), r_in=10, r_out=20, mask=mask
        )

        # Both should give same msky (uniform array), but nsky may differ
        assert_allclose(result_masked["msky"][0], 10.0, rtol=1e-10)
        assert result_masked["nsky"][0] <= result_nomask["nsky"][0]

    def test_quick_sky_circ_kwargs_passthrough(self, uniform_100x100):
        """Test quick_sky_circ passes kwargs (method) to sky_fit."""
        result_mean = quick_sky_circ(
            uniform_100x100, pos=(50, 50), r_in=10, r_out=20, method="mean"
        )
        result_median = quick_sky_circ(
            uniform_100x100, pos=(50, 50), r_in=10, r_out=20, method="median"
        )

        # Both should give 10.0 for uniform array
        assert_allclose(result_mean["msky"][0], 10.0, rtol=1e-10)
        assert_allclose(result_median["msky"][0], 10.0, rtol=1e-10)

    def test_quick_sky_circ_custom_radii(self, uniform_100x100):
        """Test quick_sky_circ with non-default r_in and r_out."""
        result = quick_sky_circ(uniform_100x100, pos=(50, 50), r_in=5, r_out=8)

        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)


# =============================================================================
# Tests for mmm_dao
# =============================================================================
class TestMmmDao:
    """Tests for mmm_dao function (DAOPHOT MMM algorithm)."""

    def test_mmm_dao_uniform(self):
        """
        Test mmm_dao on uniform array.

        For uniform data, mmm_dao should return that value.
        """
        sky = np.full(1000, 100.0)
        result = mmm_dao(sky)
        assert_allclose(result, 100.0, rtol=1e-5)

    def test_mmm_dao_gaussian_noise(self):
        """
        Test mmm_dao on Gaussian noise.

        N(100, 10) should give mmm estimate close to 100.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        result = mmm_dao(sky)

        # Should be close to 100
        assert_allclose(result, 100.0, atol=2.0)

    def test_mmm_dao_with_outliers(self):
        """
        Test mmm_dao is robust to outliers.

        Add some high outliers to Gaussian background.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        # Add outliers
        sky[:50] = 500.0

        result = mmm_dao(sky)

        # Should still be close to 100 (robust estimator)
        assert_allclose(result, 100.0, atol=5.0)

    def test_mmm_dao_too_few_pixels(self):
        """Test mmm_dao raises error with too few pixels."""
        sky = np.array([100.0, 100.0, 100.0])  # only 3 pixels

        with pytest.raises(ValueError, match="must contain at least"):
            mmm_dao(sky, min_nsky=20)

    def test_mmm_dao_highbad(self):
        """
        Test mmm_dao with highbad parameter.

        Pixels above highbad should be rejected.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        sky[:100] = 200.0  # These should be rejected if highbad=150

        result = mmm_dao(sky, highbad=150)

        # Should be close to 100
        assert_allclose(result, 100.0, atol=3.0)

    def test_mmm_dao_integer_sky(self):
        """
        Test mmm_dao with integer sky array.

        Integer arrays trigger the cut >= 1.5 floor in the rejection loop.
        """
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=100.0, scale=10.0, size=5000).astype(int)
        result = mmm_dao(sky)

        assert_allclose(result, 100.0, atol=3.0)

    def test_mmm_dao_readnoise(self):
        """
        Test mmm_dao with readnoise > 0.

        Should still converge and return a reasonable sky estimate.
        """
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=100.0, scale=10.0, size=5000)
        result = mmm_dao(sky, readnoise=5.0)

        assert_allclose(result, 100.0, atol=3.0)

    def test_mmm_dao_min_nsky_boundary(self):
        """
        Test mmm_dao with exactly min_nsky elements passes.

        Exactly min_nsky elements should not raise.
        """
        sky = np.full(20, 100.0)
        # Should not raise with exactly min_nsky=20
        result = mmm_dao(sky, min_nsky=20)
        assert_allclose(result, 100.0, atol=1e-5)

    def test_mmm_dao_maxiter_exceeded(self):
        """
        Test mmm_dao raises ValueError when maxiter is exceeded.

        Use maxiter=1 with data that requires multiple iterations.
        """
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=100.0, scale=10.0, size=5000)
        # maxiter=1 should be too few for convergence on noisy data
        with pytest.raises(ValueError, match="Too many"):
            mmm_dao(sky, maxiter=1)


# =============================================================================
# Analytical sky estimation tests
# =============================================================================
class TestSkyFitAnalytical:
    """Analytical tests for sky estimation methods."""

    def test_sex_estimator_formula(self):
        """
        Test SExtractor sky estimator formula.

        For |mean - median|/std < 0.3: use median
        Otherwise: use 2.5*median - 1.5*mean
        """
        # Create slightly skewed distribution
        np.random.seed(42)
        # Uniform data: mean = median, so should use median path
        sky = np.full(1000, 50.0)
        result = sky_fit(sky, annulus=None, method="sex")

        assert_allclose(result["msky"][0], 50.0, rtol=1e-10)

    def test_iraf_estimator_formula(self):
        """
        Test IRAF sky estimator formula (after sigma clipping).

        if mean < median: msky = mean
        else: msky = 3*median - 2*mean
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=5.0, size=1000)

        result = sky_fit(sky, annulus=None, method="iraf")

        sky_clipped = sigma_clipper(sky)
        mean = np.mean(sky_clipped)
        median = np.median(sky_clipped)
        expected = mean if mean < median else 3 * median - 2 * mean

        assert_allclose(result["msky"][0], expected, rtol=1e-5)

    def test_mmm_estimator_formula(self):
        """
        Test MMM sky estimator formula: 3*median - 2*mean (after sigma clipping).
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=1000)

        result = sky_fit(sky, annulus=None, method="mmm")

        sky_clipped = sigma_clipper(sky)
        mean = np.mean(sky_clipped)
        median = np.median(sky_clipped)
        expected = 3 * median - 2 * mean

        assert_allclose(result["msky"][0], expected, rtol=1e-5)


class TestSkyFitEllipticalAnnulus:
    """Tests for sky fitting with elliptical annuli."""

    def test_sky_fit_with_elliptical_annulus(self, uniform_100x100):
        """sky_fit works end-to-end with EllipticalAnnulus fast path."""
        import astropy.units as u

        an = _ellip_an(positions=(50, 50), a_in=6, a_out=12, b_out=8, theta=0.0 * u.rad)
        result = sky_fit(uniform_100x100, an, method="mean")
        assert_allclose(result["msky"][0], 10.0, rtol=1e-10)
        assert_allclose(result["ssky"][0], 0.0, atol=1e-10)
