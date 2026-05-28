"""Tests for IRAF-like image statistics helpers."""

import numpy as np
import pytest

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
        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert "med" in result
        assert "std" in result
        assert "madstd" in result
        assert "pct" in result
        assert "zmin" in result
        assert "ext_lo" in result
        assert "ext_hi" in result

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

    def test_stats_uses_imcombiners_variance_mean(self, monkeypatch):
        """Basic mean/std statistics should use imcombiners' one-pass kernel."""
        import imcombiners.kernels as imck

        calls = {}

        def fake_var_1d(values, *, ddof=0, return_mean=False, validate=True):
            calls["values"] = values.copy()
            calls["ddof"] = ddof
            calls["return_mean"] = return_mean
            return 4.0, 10.0

        monkeypatch.setattr(imck, "var_1d", fake_var_1d)

        result = imstat.give_stats(np.array([1.0, 2.0, 3.0]), num_extrema=None)

        assert result["avg"] == 10.0
        assert result["std"] == 2.0
        assert calls["ddof"] == 1
        assert calls["return_mean"] is True
        np.testing.assert_array_equal(calls["values"], [1.0, 2.0, 3.0])

    def test_stats_percentiles_none_skips_percentile_work(self, monkeypatch):
        """`percentiles=None` omits percentile output and percentile work."""

        def fail_percentile(*args, **kwargs):
            raise AssertionError("percentile should not be called")

        monkeypatch.setattr(imstat.np, "percentile", fail_percentile)

        result = imstat.give_stats(
            np.arange(10.0),
            percentiles=None,
            num_extrema=None,
        )

        assert "pct" not in result
        assert "percentiles" not in result
        assert "zmin" in result
        assert "zmax" in result
        assert "madstd" in result

    def test_stats_extrema_match_sorted_values(self):
        """Extrema output preserves sorted-low/high semantics."""
        arr = np.array([5.0, 1.0, 3.0, 2.0, 100.0, -4.0, 7.0])

        result = imstat.give_stats(arr, num_extrema=(3, 3))

        np.testing.assert_allclose(result["ext_lo"], [-4.0, 1.0, 2.0])
        np.testing.assert_allclose(result["ext_hi"], [5.0, 7.0, 100.0])

    def test_stats_default_extrema_are_one_low_and_one_high(self):
        """Default extrema output should include one low and one high value."""
        arr = np.array([5.0, 1.0, 3.0, 2.0, 100.0, -4.0, 7.0])

        result = imstat.give_stats(arr)

        np.testing.assert_allclose(result["ext_lo"], [-4.0])
        np.testing.assert_allclose(result["ext_hi"], [100.0])

    def test_stats_asymmetric_extrema_counts(self):
        """`num_extrema=(n_lo, n_hi)` should allow asymmetric extrema output."""
        arr = np.array([5.0, 1.0, 3.0, 2.0, 100.0, -4.0, 7.0])

        result = imstat.give_stats(arr, num_extrema=(2, 1))

        np.testing.assert_allclose(result["ext_lo"], [-4.0, 1.0])
        np.testing.assert_allclose(result["ext_hi"], [100.0])

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

    def test_stats_header_preserves_median_and_zscale(self, temp_fits_file):
        """STATMED should not be overwritten by zscale metadata."""
        result, hdr = imstat.give_stats(temp_fits_file, return_header=True)

        assert hdr["STATMED"] == pytest.approx(result["med"])
        assert hdr["STATZMIN"] == pytest.approx(result["zmin"])

    def test_stats_rejects_ccddata_input(self, sample_ccddata):
        """Stats helpers intentionally accept arrays or paths, not CCDData."""
        with np.testing.assert_raises(TypeError):
            imstat.give_stats(sample_ccddata)
