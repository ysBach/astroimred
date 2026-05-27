"""Tests for IRAF-like image statistics helpers."""

import numpy as np

import astroimred as air
from astroimred.imutil import imstat

RTOL = 1e-6
ATOL = 1e-8


class TestGiveStats:
    """Tests for give_stats function."""

    def test_give_stats_lives_in_imstat(self):
        """give_stats is an imstat helper."""
        assert air.give_stats is imstat.give_stats

    def test_stats_basic(self, sample_data_2d):
        """Test basic statistics calculation."""
        result = imstat.give_stats(sample_data_2d)
        assert isinstance(result, dict)
        # Check required keys exist (implementation uses avg, madstd, med)
        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert "med" in result
        assert "std" in result

    def test_stats_known_values(self):
        """Test statistics with known values."""
        # Create array with known statistics
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = imstat.give_stats(arr)

        assert result["min"] == 1.0
        assert result["max"] == 5.0
        np.testing.assert_allclose(result["avg"], 3.0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(result["med"], 3.0, rtol=RTOL, atol=ATOL)
        # std of [1,2,3,4,5] with ddof=0 is sqrt(2), implementation uses ddof=1 for std
        # std of sample [1,2,3,4,5] ddof=1 is sqrt(2.5) ~ 1.5811388
        np.testing.assert_allclose(
            result["std"], np.std(arr, ddof=1), rtol=RTOL, atol=ATOL
        )

    def test_stats_extrema_match_sorted_values(self):
        """Extrema output preserves sorted-low/high semantics."""
        arr = np.array([5.0, 1.0, 3.0, 2.0, 100.0, -4.0, 7.0])

        result = imstat.give_stats(arr, N_extrema=3)

        np.testing.assert_allclose(result["ext_lo"], [-4.0, 1.0, 2.0])
        np.testing.assert_allclose(result["ext_hi"], [5.0, 7.0, 100.0])

    def test_stats_mask_does_not_mutate_input(self):
        """Applying a mask for stats does not alter the caller's array."""
        arr = np.arange(9.0).reshape(3, 3)
        original = arr.copy()
        mask = np.zeros_like(arr, dtype=bool)
        mask[1, 1] = True

        imstat.give_stats(arr, mask=mask)

        np.testing.assert_array_equal(arr, original)

    def test_stats_path_input(self, temp_fits_file):
        """Test statistics on a path-like FITS input."""
        result = imstat.give_stats(temp_fits_file)
        assert result["num"] == 10000
        assert np.isfinite(result["avg"])

    def test_stats_rejects_ccddata_input(self, sample_ccddata):
        """Stats helpers intentionally accept arrays or paths, not CCDData."""
        with np.testing.assert_raises(TypeError):
            imstat.give_stats(sample_ccddata)
