"""
Tests for astroimred.phot.apphot module.

All expected values are analytically derived.
"""

import astroapers as aap
import numpy as np
from astropy.nddata import CCDData
from numpy.testing import assert_allclose

from astroimred.phot.apphot import apphot_annulus


# =============================================================================
# Tests for apphot_annulus
# =============================================================================
class TestApphotAnnulus:
    """Tests for apphot_annulus function."""

    def test_apphot_uniform_source_uniform_sky(self, ccd_uniform):
        """
        Test aperture photometry on uniform image.

        Image value = 100 everywhere
        Aperture r = 5 -> area = pi * 25 = 78.54
        Annulus r_in = 10, r_out = 15 -> sky = 100

        apsum = 100 * pi * 25 = 7853.98
        srcsum = apsum - area * sky = 0
        """
        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd_uniform, ap, an, pandas=False)

        # Sky should be 100
        assert_allclose(result["msky"][0], 100.0, rtol=1e-5)

        # Source sum should be ~0 (source - sky = 0)
        assert_allclose(result["srcsum"][0], 0.0, atol=1.0)

    def test_apphot_source_above_sky(
        self, gaussian_source_centered, gaussian_params_centered
    ):
        """
        Test aperture photometry on Gaussian source.

        Gaussian at (50, 50) with amplitude=1000, sigma=3, background=100
        Small aperture should capture most of the source flux.
        """
        ccd = CCDData(
            gaussian_source_centered,
            unit="adu",
            header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 60},
        )

        ap = aap.CircAp((50, 50), r=10)  # r ~ 3*sigma captures >99%
        an = aap.CircAn((50, 50), r_in=20, r_out=30)  # Far from source

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # Sky should be close to background (100)
        assert_allclose(result["msky"][0], 100.0, atol=5.0)

        # Source sum should be positive and significant
        assert result["srcsum"][0] > 1000  # Gaussian peak is 1000

    def test_apphot_aperture_area(self, ccd_uniform):
        """
        Test apsum_npix is correctly calculated.

        CircAp area = pi * r^2
        For r = 7: area = pi * 49 = 153.938...
        """
        ap = aap.CircAp((50, 50), r=7)
        an = aap.CircAn((50, 50), r_in=15, r_out=20)

        result = apphot_annulus(ccd_uniform, ap, an, pandas=False)

        expected_area = np.pi * 49
        assert_allclose(result["apsum_npix"][0], expected_area, rtol=1e-3)

    def test_apphot_magnitude(self, gaussian_source_centered):
        """
        Test magnitude calculation.

        mag = -2.5 * log10(srcsum / t_exposure)

        For positive srcsum, magnitude should be finite.
        """
        ccd = CCDData(
            gaussian_source_centered,
            unit="adu",
            header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 1},
        )

        ap = aap.CircAp((50, 50), r=10)
        an = aap.CircAn((50, 50), r_in=20, r_out=30)

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # Magnitude should be finite and negative (bright source)
        assert np.isfinite(result["mag"][0])

    def test_apphot_snr(self, gaussian_source_centered):
        """
        Test SNR calculation.

        SNR = srcsum / srcsum_err
        """
        ccd = CCDData(
            gaussian_source_centered,
            unit="adu",
            header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 60},
        )

        ap = aap.CircAp((50, 50), r=10)
        an = aap.CircAn((50, 50), r_in=20, r_out=30)

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # SNR should be positive for bright source
        assert result["snr"][0] > 0

        # Verify SNR = srcsum / srcsum_err
        expected_snr = result["srcsum"][0] / result["srcsum_err"][0]
        assert_allclose(result["snr"][0], expected_snr, rtol=1e-5)

    def test_apphot_error_propagation(self, ccd_uniform):
        """
        Test error propagation formula.

        srcsum_err^2 = apsum_err^2 + apsum_npix * ssky^2

        For uniform image with ssky=0, srcsum_err ≈ apsum_err
        """
        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd_uniform, ap, an, pandas=False)

        # For uniform sky, ssky ≈ 0
        assert_allclose(result["ssky"][0], 0.0, atol=1e-5)

        # srcsum_err should equal apsum_err when ssky=0
        assert_allclose(result["srcsum_err"][0], result["apsum_err"][0], rtol=1e-5)

    def test_apphot_pandas_output(self, ccd_uniform):
        """Test apphot_annulus returns pandas DataFrame when pandas=True."""
        import pandas as pd

        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd_uniform, ap, an, pandas=True)

        assert isinstance(result, pd.DataFrame)
        assert "srcsum" in result.columns
        assert "aperture_sum" not in result.columns
        assert "source_sum" not in result.columns
        assert "aparea" not in result.columns
        assert "npix" not in result.columns
        assert "mag" in result.columns

    def test_apphot_no_annulus(self, ccd_uniform):
        """
        Test apphot_annulus with annulus=None.

        When no annulus is provided, sky should be 0.
        """
        ap = aap.CircAp((50, 50), r=5)

        result = apphot_annulus(ccd_uniform, ap, annulus=None, pandas=False)

        # Sky should be 0
        assert_allclose(result["msky"][0], 0.0, rtol=1e-10)

        # srcsum should equal apsum
        assert_allclose(result["srcsum"][0], result["apsum"][0], rtol=1e-10)


# =============================================================================
# Analytical photometry tests
# =============================================================================
class TestPhotometryAnalytical:
    """Analytical tests for photometry calculations."""

    def test_apsum_uniform(self):
        """
        Aperture sum on uniform image equals value * area.

        Image value = V, apsum_npix = A
        apsum = V * A
        """
        value = 50.0
        radius = 10.0
        data = np.full((100, 100), value)
        ap = aap.CircAp((50, 50), r=radius)

        apsum = ap.apsum(data)[0]

        expected = value * np.pi * radius**2
        assert_allclose(apsum, expected, rtol=0.01)

    def test_srcsum_formula(self):
        """
        srcsum = apsum - apsum_npix * msky

        Test this formula directly.
        """
        # Create image with source (value=200) in center, sky (value=100) outside
        data = np.full((100, 100), 100.0)  # sky = 100
        data[45:55, 45:55] = 200.0  # source region = 200

        ccd = CCDData(data, unit="adu", header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 60})

        # Small aperture to capture source
        ap = aap.CircAp((50, 50), r=4)
        # Annulus in sky region
        an = aap.CircAn((50, 50), r_in=15, r_out=25)

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # Verify formula: srcsum = apsum - apsum_npix * msky
        calculated = result["apsum"][0] - result["apsum_npix"][0] * result["msky"][0]
        assert_allclose(result["srcsum"][0], calculated, rtol=1e-5)

    def test_magnitude_formula(self):
        """
        mag = -2.5 * log10(srcsum / t_exposure)

        Test this formula.
        """
        # Create bright source
        data = np.full((100, 100), 10.0)  # background
        data[48:52, 48:52] = 1000.0  # bright source

        ccd = CCDData(
            data, unit="adu", header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 10.0}
        )

        ap = aap.CircAp((50, 50), r=3)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # Verify magnitude formula
        t_exp = 10.0
        expected_mag = -2.5 * np.log10(result["srcsum"][0] / t_exp)
        assert_allclose(result["mag"][0], expected_mag, rtol=1e-5)

    def test_magnitude_error_formula(self):
        """
        merr = 2.5 / ln(10) * (1 / snr)

        Test this formula.
        """
        data = np.full((100, 100), 10.0)
        data[48:52, 48:52] = 1000.0

        ccd = CCDData(data, unit="adu", header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 60})

        ap = aap.CircAp((50, 50), r=3)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd, ap, an, pandas=False)

        # Verify magnitude error formula
        expected_merr = 2.5 / np.log(10) * (1 / result["snr"][0])
        assert_allclose(result["merr"][0], expected_merr, rtol=1e-5)


# =============================================================================
# Error handling tests
# =============================================================================
class TestApphotErrorHandling:
    """Tests for error handling in apphot_annulus."""

    def test_apphot_with_mask(self, ccd_uniform):
        """Test apphot_annulus handles mask correctly."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True  # Mask center pixel

        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd_uniform, ap, an, mask=mask, pandas=False)
        _, total_npix = ap.apsum(ccd_uniform.data)

        # Should still compute result
        assert np.isfinite(result["srcsum"][0])
        assert_allclose(result["apsum_npix"][0] + result["nbadpix"][0], total_npix)
        assert result["nbadpix"][0] > 0

    def test_apphot_nbadpix_uses_exact_fractional_weights(self):
        data = np.ones((30, 30), dtype=float)
        ap = aap.CircAp((12.2, 13.3), r=5)
        weights = ap.get_apmask("exact").to_image(data.shape)
        fractional = np.argwhere((weights > 0) & (weights < 1))
        y, x = fractional[0]
        mask = np.zeros_like(data, dtype=bool)
        mask[y, x] = True
        ccd = CCDData(data, unit="adu", header={"GAIN": 1, "RDNOISE": 0}, mask=mask)

        result = apphot_annulus(ccd, ap, annulus=None, pandas=False)
        _, total_npix = ap.apsum(data)

        assert_allclose(result["nbadpix"][0], weights[y, x])
        assert_allclose(result["apsum_npix"][0], total_npix - weights[y, x])
        assert_allclose(result["apsum_npix"][0] + result["nbadpix"][0], total_npix)

    def test_apphot_apsum_npix_excludes_out_of_image_support(self):
        data = np.ones((20, 20), dtype=float)
        ccd = CCDData(data, unit="adu", header={"GAIN": 1, "RDNOISE": 0})
        ap = aap.CircAp((-1.5, 10.0), r=6)

        result = apphot_annulus(ccd, ap, annulus=None, pandas=False)
        expected_apsum, expected_npix = ap.apsum(data)

        assert_allclose(result["apsum"][0], expected_apsum)
        assert_allclose(result["apsum_npix"][0], expected_npix)
        assert_allclose(result["nbadpix"][0], 0.0)

    def test_apphot_bad_pixel_flag(self):
        """Test bad pixel flagging."""
        data = np.full((100, 100), 100.0)
        mask = np.zeros((100, 100), dtype=bool)
        # Mask several pixels in aperture
        mask[48:52, 48:52] = True

        ccd = CCDData(
            data, unit="adu", header={"GAIN": 2, "RDNOISE": 5, "EXPTIME": 60}, mask=mask
        )

        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        result = apphot_annulus(ccd, ap, an, npix_mask_ap=2, pandas=False)

        # Should flag as bad (>2 masked pixels in aperture)
        assert result["bad"][0] == 1

    def test_apphot_ndarray_input(self, uniform_100x100):
        """Test apphot_annulus works with ndarray input."""
        ap = aap.CircAp((50, 50), r=5)
        an = aap.CircAn((50, 50), r_in=10, r_out=15)

        # Should work with plain ndarray
        result = apphot_annulus(uniform_100x100, ap, an, pandas=False)

        assert np.isfinite(result["srcsum"][0])

    def test_apphot_multiple_apertures(self, ccd_uniform):
        """Test apphot_annulus with multiple aperture radii."""
        # List of apertures with different radii
        apertures = [
            aap.CircAp((50, 50), r=3),
            aap.CircAp((50, 50), r=5),
            aap.CircAp((50, 50), r=7),
        ]
        an = aap.CircAn((50, 50), r_in=15, r_out=20)

        result = apphot_annulus(ccd_uniform, apertures, an, pandas=False)

        # Should have 3 rows
        assert len(result) == 3

        # Areas should be different
        areas = result["apsum_npix"]
        assert areas[0] < areas[1] < areas[2]
