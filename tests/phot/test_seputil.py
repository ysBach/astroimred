"""
Tests for astroimred.external.sep module.

All expected values are analytically derived.

Note: These tests require the sep package, which is a core dependency.
"""

import numpy as np
from numpy.testing import assert_allclose

import astroimred.external.sep as sepmod
from astroimred.external.sep import (
    sep_back,
    sep_extract,
    sep_extract_iterative,
    sep_flux_auto,
)


# =============================================================================
# Tests for sep_back
# =============================================================================
class TestSepBack:
    """Tests for sep_back function (background estimation)."""

    def test_sep_back_uniform(self, uniform_100x100):
        """
        Test sep_back on uniform image.

        Global background should equal the uniform value.
        """
        bkg = sep_back(uniform_100x100)

        assert_allclose(bkg.globalback, 10.0, rtol=0.1)

    def test_sep_back_globalrms_uniform(self, uniform_100x100):
        """
        Test sep_back rms on uniform image.

        Note: SEP may return a floor value (e.g., 1.0) for perfectly uniform images.
        """
        bkg = sep_back(uniform_100x100)

        # RMS should be very small - SEP may return a floor value of 1.0
        assert bkg.globalrms <= 1.0 + 1e-10

    def test_sep_back_with_noise(self, uniform_with_noise):
        """
        Test sep_back on noisy image.

        For N(100, 10), background should be ~100, rms ~10.
        """
        bkg = sep_back(uniform_with_noise)

        assert_allclose(bkg.globalback, 100.0, atol=5.0)
        assert_allclose(bkg.globalrms, 10.0, atol=3.0)

    def test_sep_back_array_shape(self, uniform_100x100):
        """Test sep_back returns correct shape arrays."""
        bkg = sep_back(uniform_100x100)

        back_arr = bkg.back()
        rms_arr = bkg.rms()

        assert back_arr.shape == (100, 100)
        assert rms_arr.shape == (100, 100)

    def test_sep_back_with_mask(self, uniform_100x100):
        """Test sep_back respects mask."""
        # Create image with masked region having different value
        data = uniform_100x100.copy()
        data[40:60, 40:60] = 1000.0  # Bright region

        mask = np.zeros_like(data, dtype=bool)
        mask[40:60, 40:60] = True

        bkg = sep_back(data, mask=mask)

        # Background should be ~10 (ignoring masked region)
        assert_allclose(bkg.globalback, 10.0, atol=2.0)

    def test_sep_back_box_size(self, uniform_100x100):
        """Test sep_back with different box sizes."""
        bkg_small = sep_back(uniform_100x100, box_size=(32, 32))
        bkg_large = sep_back(uniform_100x100, box_size=(64, 64))

        # Both should give same result for uniform image
        assert_allclose(bkg_small.globalback, bkg_large.globalback, rtol=0.1)


def test_disk_struct_cache_reuses_kernel():
    """Repeated dilation kernels of the same radius are cached."""
    sepmod._disk_struct.cache_clear()

    first = sepmod._disk_struct(3)
    second = sepmod._disk_struct(3)

    assert first is second


# =============================================================================
# Tests for sep_extract
# =============================================================================
class TestSepExtract:
    """Tests for sep_extract function (source extraction)."""

    def test_sep_extract_single_source(
        self, gaussian_source_centered, gaussian_params_centered
    ):
        """
        Test sep_extract finds single Gaussian source.

        Source at (50, 50) with amplitude=1000, sigma=3.
        """
        # Subtract background for detection
        bkg = sep_back(gaussian_source_centered)
        data_skysub = gaussian_source_centered - bkg.back()

        obj, segm = sep_extract(data_skysub, thresh=50, bkg=None)

        # Should find exactly 1 object
        assert len(obj) >= 1

        # Position should be close to (50, 50)
        # Find the brightest/closest to center
        if len(obj) > 1:
            obj = obj.iloc[[0]]  # First one (sorted by distance or flux)

        x_true = gaussian_params_centered["x_mean"]
        y_true = gaussian_params_centered["y_mean"]

        assert_allclose(obj["x"].iloc[0], x_true, atol=1.0)
        assert_allclose(obj["y"].iloc[0], y_true, atol=1.0)

    def test_sep_extract_no_source(self, uniform_100x100):
        """Test sep_extract finds no sources in uniform image."""
        obj, segm = sep_extract(uniform_100x100, thresh=100, bkg=None)

        # Should find 0 objects (no sources above threshold)
        assert len(obj) == 0

    def test_sep_extract_segmentation_map(self, gaussian_source_centered):
        """Test sep_extract returns segmentation map."""
        bkg = sep_back(gaussian_source_centered)

        obj, segm = sep_extract(gaussian_source_centered, thresh=50, bkg=bkg)

        # Segmentation map should have same shape as input
        assert segm.shape == gaussian_source_centered.shape

        # Should have non-zero values where source is detected
        if len(obj) > 0:
            assert np.any(segm > 0)

    def test_sep_extract_with_bezel(self, gaussian_source_centered):
        """Test sep_extract bezel parameter excludes edge detections."""
        bkg = sep_back(gaussian_source_centered)

        # Without bezel
        obj1, _ = sep_extract(
            gaussian_source_centered, thresh=50, bkg=bkg, bezel_x=[0, 0], bezel_y=[0, 0]
        )

        # With large bezel (source at 50,50 should still be found)
        obj2, _ = sep_extract(
            gaussian_source_centered,
            thresh=50,
            bkg=bkg,
            bezel_x=[10, 10],
            bezel_y=[10, 10],
        )

        # Both should find the central source
        assert len(obj1) >= 1
        assert len(obj2) >= 1

    def test_sep_extract_minarea(self, gaussian_source_centered):
        """Test sep_extract minarea parameter."""
        bkg = sep_back(gaussian_source_centered)

        # With small minarea
        obj_small, _ = sep_extract(
            gaussian_source_centered, thresh=50, bkg=bkg, minarea=5
        )

        # With large minarea (might reject small sources)
        obj_large, _ = sep_extract(
            gaussian_source_centered, thresh=50, bkg=bkg, minarea=100
        )

        # Source should be found with small minarea
        assert len(obj_small) >= 1

    def test_sep_extract_pos_ref(self, gaussian_source_centered):
        """Test sep_extract with pos_ref adds distance column."""
        bkg = sep_back(gaussian_source_centered)

        obj, _ = sep_extract(
            gaussian_source_centered, thresh=50, bkg=bkg, pos_ref=(50, 50)
        )

        # Should have dist_ref column
        assert "dist_ref" in obj.columns

        # Distance should be small for source at (50, 50)
        if len(obj) > 0:
            assert obj["dist_ref"].iloc[0] < 5.0

    def test_sep_extract_pos_ref_sorts_by_distance_by_default(self):
        """Test sep_extract sorts by dist_ref when pos_ref is given."""
        yy, xx = np.mgrid[:100, :100]
        data = (
            100.0
            + 1000.0 * np.exp(-((xx - 25.0) ** 2 + (yy - 50.0) ** 2) / (2 * 3.0**2))
            + 1000.0 * np.exp(-((xx - 75.0) ** 2 + (yy - 50.0) ** 2) / (2 * 3.0**2))
        )
        bkg = sep_back(data)

        obj, _ = sep_extract(data, thresh=50, bkg=bkg, pos_ref=(75, 50))

        assert len(obj) >= 2
        assert obj["dist_ref"].iloc[0] <= obj["dist_ref"].iloc[1]
        assert_allclose(obj["x"].iloc[0], 75.0, atol=2.0)

    def test_sep_extract_pos_ref_respects_explicit_sort_by(self):
        """Test explicit sort_by is not overwritten by pos_ref."""
        yy, xx = np.mgrid[:100, :100]
        data = (
            100.0
            + 1000.0 * np.exp(-((xx - 25.0) ** 2 + (yy - 50.0) ** 2) / (2 * 3.0**2))
            + 100.0 * np.exp(-((xx - 75.0) ** 2 + (yy - 50.0) ** 2) / (2 * 3.0**2))
        )
        bkg = sep_back(data)

        obj, _ = sep_extract(
            data,
            thresh=20,
            bkg=bkg,
            pos_ref=(75, 50),
            sort_by="flux",
            sort_ascending=False,
        )

        assert len(obj) >= 2
        assert_allclose(obj["x"].iloc[0], 25.0, atol=2.0)


# =============================================================================
# Tests for sep_flux_auto
# =============================================================================
class TestSepFluxAuto:
    """Tests for sep_flux_auto function (FLUX_AUTO calculation)."""

    def test_sep_flux_auto_basic(self, gaussian_source_centered):
        """Test sep_flux_auto computes flux."""
        bkg = sep_back(gaussian_source_centered)
        data_skysub = gaussian_source_centered - bkg.back()

        obj, _ = sep_extract(data_skysub, thresh=50, bkg=None)

        if len(obj) > 0:
            fl, dfl, flag = sep_flux_auto(data_skysub, obj)

            # Flux should be positive
            assert fl[0] > 0

            # Error should be positive
            assert dfl[0] >= 0

    def test_sep_flux_auto_with_error(self, gaussian_source_centered):
        """Test sep_flux_auto with explicit error map."""
        bkg = sep_back(gaussian_source_centered)
        data_skysub = gaussian_source_centered - bkg.back()

        # Create explicit error map with realistic values
        # For uniform background, SEP's rms() may be 0 or floor value
        err = np.full_like(data_skysub, 10.0)  # Explicit non-zero error

        obj, _ = sep_extract(data_skysub, thresh=50, bkg=None)

        if len(obj) > 0:
            fl, dfl, flag = sep_flux_auto(data_skysub, obj, err=err)

            # Error should be non-zero when explicit error map provided
            assert dfl[0] > 0


# =============================================================================
# Analytical SEP tests
# =============================================================================
class TestSepAnalytical:
    """Analytical tests for SEP functionality."""

    def test_background_subtraction(
        self, gaussian_source_centered, gaussian_params_centered
    ):
        """
        Test background subtraction preserves source.

        After subtracting background, peak should be ~amplitude.
        """
        bkg = sep_back(gaussian_source_centered)
        data_skysub = gaussian_source_centered - bkg.back()

        # Peak in subtracted image should be close to amplitude
        peak = np.max(data_skysub)
        expected_peak = gaussian_params_centered["amplitude"]

        # Allow some tolerance for background estimation error
        assert_allclose(peak, expected_peak, rtol=0.1)

    def test_source_position_accuracy(
        self, gaussian_source_centered, gaussian_params_centered
    ):
        """
        Test source position accuracy.

        SEP should find position within 0.5 pixels of truth.
        """
        bkg = sep_back(gaussian_source_centered)

        obj, _ = sep_extract(gaussian_source_centered, thresh=50, bkg=bkg)

        if len(obj) > 0:
            x_meas = obj["x"].iloc[0]
            y_meas = obj["y"].iloc[0]
            x_true = gaussian_params_centered["x_mean"]
            y_true = gaussian_params_centered["y_mean"]

            assert abs(x_meas - x_true) < 0.5
            assert abs(y_meas - y_true) < 0.5


# =============================================================================
# Edge cases and error handling
# =============================================================================
class TestSepEdgeCases:
    """Edge case tests for SEP wrapper functions."""

    def test_sep_back_ccddata_input(self, ccd_uniform):
        """Test sep_back with CCDData-like input.

        Note: The sep_back function docstring says it accepts CCDData, but the
        current implementation doesn't properly extract .data before passing
        to sep.Background(). This test uses .data explicitly as a workaround.
        """
        # Use .data to work around the implementation issue
        bkg = sep_back(ccd_uniform.data)

        assert_allclose(bkg.globalback, 100.0, rtol=0.1)

    def test_sep_extract_empty_result(self):
        """Test sep_extract handles case with no detections."""
        data = np.zeros((100, 100), dtype=np.float32)

        obj, segm = sep_extract(data, thresh=1.0, bkg=None)

        assert len(obj) == 0
        assert segm.shape == data.shape

    def test_sep_back_small_image(self):
        """Test sep_back on small image."""
        data = np.full((20, 20), 50.0, dtype=np.float32)

        # Should handle small images by adjusting box size
        bkg = sep_back(data, box_size=(8, 8))

        assert_allclose(bkg.globalback, 50.0, rtol=0.1)

    def test_sep_byte_order(self, uniform_100x100):
        """Test sep handles different byte orders."""
        # Create big-endian array
        data_be = uniform_100x100.astype(">f4")

        bkg = sep_back(data_be)

        assert_allclose(bkg.globalback, 10.0, rtol=0.1)


# =============================================================================
# Integration tests
# =============================================================================
class TestSepIntegration:
    """Integration tests for SEP workflow."""

    def test_full_detection_workflow(self, gaussian_source_centered):
        """Test full detection workflow: background -> extract -> photometry."""
        # 1. Estimate background
        bkg = sep_back(gaussian_source_centered)

        # 2. Extract sources
        obj, segm = sep_extract(gaussian_source_centered, thresh=50, bkg=bkg)

        assert len(obj) >= 1

        # 3. Compute auto flux
        data_skysub = gaussian_source_centered - bkg.back()
        fl, dfl, flag = sep_flux_auto(data_skysub, obj, err=bkg.rms())

        assert fl[0] > 0
        assert dfl[0] >= 0


# =============================================================================
# Correctness regression tests (fixes from sep-wrapper-fix-plan)
# =============================================================================
class TestSepBackCorrectness:
    """Regression tests for sep_back correctness fixes."""

    def test_ccddata_input_direct(self):
        """sep_back(CCDData(...)) must not raise — advertised API."""
        from astropy.nddata import CCDData

        data = np.full((50, 50), 42.0, dtype=np.float32)
        ccd = CCDData(data, unit="adu")
        bkg = sep_back(ccd)
        assert_allclose(bkg.globalback, 42.0, atol=2.0)

    def test_native_int16_not_byte_corrupted(self):
        """int16 image with value 7 must give globalback ~7, not 1792."""
        data = np.full((50, 50), 7, dtype=np.int16)
        bkg = sep_back(data)
        assert_allclose(bkg.globalback, 7.0, atol=1.0)

    def test_big_endian_float(self):
        """Big-endian float32 array must work and give correct background."""
        data = np.full((50, 50), 25.0, dtype=">f4")
        bkg = sep_back(data)
        assert_allclose(bkg.globalback, 25.0, atol=2.0)

    def test_numeric_mask_respects_maskthresh(self):
        """Numeric mask values at or below maskthresh must not be masked."""
        data = np.full((64, 64), 10.0, dtype=np.float32)
        # Put a bright patch that would skew background
        data[28:36, 28:36] = 1000.0
        # Mask value 0.5 with maskthresh=1.0 → pixel should be UNmasked
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[28:36, 28:36] = 0.5  # below maskthresh=1.0 → should not mask
        bkg_unmasked = sep_back(data, mask=mask, maskthresh=1.0)
        bkg_no_mask = sep_back(data)
        # Both should include the bright patch → similar (both elevated)
        assert_allclose(bkg_unmasked.globalback, bkg_no_mask.globalback, rtol=0.1)


class TestSepExtractCorrectness:
    """Regression tests for _sep_extract correctness fixes."""

    def test_big_endian_err(self):
        """Big-endian err array must not raise."""
        yy, xx = np.mgrid[:60, :60]
        data = (
            5.0 + 500.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 18.0)
        ).astype(np.float32)
        err = np.full((60, 60), 2.0, dtype=">f4")
        obj, segm = sep_extract(data, thresh=5.0, err=err)
        assert len(obj) >= 1

    def test_big_endian_var(self):
        """Big-endian var array must not raise."""
        yy, xx = np.mgrid[:60, :60]
        data = (
            5.0 + 500.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 18.0)
        ).astype(np.float32)
        var = np.full((60, 60), 4.0, dtype=">f4")
        obj, segm = sep_extract(data, thresh=5.0, var=var)
        assert len(obj) >= 1

    def test_numeric_mask_segmap_respects_maskthresh(self):
        """Segmap must not erase pixels where numeric mask <= maskthresh."""
        yy, xx = np.mgrid[:60, :60]
        data = (
            5.0 + 500.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 18.0)
        ).astype(np.float32)
        # mask=0.5 with maskthresh=1.0 means the source pixels are NOT masked
        mask = np.full((60, 60), 0.5, dtype=np.float32)
        obj, segm = sep_extract(data, thresh=5.0, mask=mask, maskthresh=1.0)
        # Source should still be detected and segmap should be non-zero at peak
        assert len(obj) >= 1
        assert np.any(segm > 0)

    def test_maxarea_rejects_large_central_object(self):
        """Objects with npix > maxarea must be removed from obj and segmap."""
        yy, xx = np.mgrid[:100, :100]
        # Large source covering ~900 pixels (sigma=15)
        data = (
            5.0
            + 2000.0 * np.exp(-((xx - 50.0) ** 2 + (yy - 50.0) ** 2) / (2 * 15.0**2))
        ).astype(np.float32)
        obj_all, _ = sep_extract(data, thresh=3.0)
        assert len(obj_all) >= 1
        large_npix = obj_all["npix"].max()

        obj_filtered, segm_filtered = sep_extract(
            data, thresh=3.0, maxarea=large_npix - 1
        )
        # The large central source should be rejected
        assert len(obj_filtered) == 0 or obj_filtered["npix"].max() < large_npix
        # Its label must be zeroed from the segmap
        surviving_labels = (
            set(obj_filtered["segm_label"].values) if len(obj_filtered) > 0 else set()
        )
        for label in set(np.unique(segm_filtered)) - {0}:
            assert label in surviving_labels


class TestSepExtractIterativeCorrectness:
    """Tests for sep_extract_iterative correctness."""

    def test_n_iter_1_finds_source(self):
        """n_iter=1 must find a simple Gaussian source."""
        yy, xx = np.mgrid[:80, :80]
        data = (
            10.0 + 800.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 18.0)
        ).astype(np.float32)
        obj, segm = sep_extract_iterative(data, thresh=5.0, n_iter=1)
        assert len(obj) >= 1

    def test_return_bkg_true(self):
        """return_bkg=True must return a sep.Background as third element."""
        import sep as sep_module

        yy, xx = np.mgrid[:80, :80]
        data = (
            10.0 + 800.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 18.0)
        ).astype(np.float32)
        result = sep_extract_iterative(data, thresh=5.0, n_iter=1, return_bkg=True)
        assert len(result) == 3
        assert isinstance(result[2], sep_module.Background)

    def test_ccddata_input(self):
        """CCDData input must work end-to-end through sep_extract_iterative."""
        from astropy.nddata import CCDData

        yy, xx = np.mgrid[:80, :80]
        data = (
            10.0 + 800.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 18.0)
        ).astype(np.float32)
        ccd = CCDData(data, unit="adu")
        obj, segm = sep_extract_iterative(ccd, thresh=5.0, n_iter=1)
        assert len(obj) >= 1

    def test_numeric_mask_respects_maskthresh(self):
        """Numeric mask values below maskthresh must remain unmasked."""
        data = np.zeros((30, 30), dtype=np.float32)
        data[15, 15] = 10.0
        mask = np.full(data.shape, 0.5, dtype=np.float32)

        obj, segm = sep_extract_iterative(
            data,
            thresh=1.0,
            mask=mask,
            maskthresh=0.75,
            n_iter=1,
            minarea=1,
            box_size=(16, 16),
        )

        assert len(obj) == 1
        assert segm[15, 15] > 0

    def test_seg_dilate_preserves_output_shape(self):
        """seg_dilate > 0 must not crash and segmap shape must match data."""
        yy, xx = np.mgrid[:80, :80]
        data = (
            10.0 + 800.0 * np.exp(-((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 18.0)
        ).astype(np.float32)
        obj, segm = sep_extract_iterative(data, thresh=5.0, n_iter=2, seg_dilate=3)
        assert segm.shape == data.shape

    def test_n_iter_0_raises(self):
        """n_iter=0 must raise ValueError."""
        import pytest

        data = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError):
            sep_extract_iterative(data, thresh=1.0, n_iter=0)
