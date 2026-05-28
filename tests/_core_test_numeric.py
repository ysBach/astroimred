"""Tests for standalone numeric helpers."""

import numpy as np
import pytest

from astroimred._core import numeric

RTOL = 1e-6
ATOL = 1e-8


class TestWvg:
    """Tests for wvg function."""

    def test_known_values(self):
        """Test weighted average with known values."""
        val = np.array([1.0, 2.0, 3.0])
        err = np.array([0.1, 0.2, 0.1])  # weights = 1/err^2

        # Manual calculation:
        # w = 1/err^2 = [100, 25, 100]
        # wvg = (1*100 + 2*25 + 3*100) / (100+25+100)
        #              = (100 + 50 + 300) / 225 = 450/225 = 2.0
        result = numeric.wvg(val, err=err)
        np.testing.assert_allclose(result, 2.0, rtol=RTOL, atol=ATOL)

    def test_equal_weights(self):
        """Test that equal weights give simple mean."""
        val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = numeric.wvg(val, err=err)
        np.testing.assert_allclose(result, 3.0, rtol=RTOL, atol=ATOL)

    def test_inverse_variance_input(self):
        """Already-computed inverse variance should be usable directly."""
        val = np.array([1.0, 2.0, 3.0])
        ivar = np.array([100.0, 25.0, 100.0])
        result = numeric.wvg(val, ivar=ivar)
        np.testing.assert_allclose(result, 2.0, rtol=RTOL, atol=ATOL)

    def test_variance_input(self):
        """Variance input should avoid square-rooting when errors are unavailable."""
        val = np.array([1.0, 2.0, 3.0])
        var = np.array([0.01, 0.04, 0.01])
        result = numeric.wvg(val, var=var)
        np.testing.assert_allclose(result, 2.0, rtol=RTOL, atol=ATOL)

    def test_axis_combines_stack(self):
        """Axis selection should support image-stack style weighted averages."""
        val = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
        err = np.ones_like(val)

        avg, stderr = numeric.wvg(val, err=err, axis=0, return_se=True)

        np.testing.assert_allclose(avg, [[1.5, 3.0], [4.5, 6.0]])
        np.testing.assert_allclose(stderr, np.sqrt(0.5) * np.ones((2, 2)))

    def test_one_uncertainty_representation_required(self):
        """Exactly one of err, var, or ivar must be supplied."""
        val = np.array([1.0, 2.0])
        err = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            numeric.wvg(val)
        with pytest.raises(ValueError):
            numeric.wvg(val, err=err, ivar=err)


class TestBinning:
    """Tests for n-D array binning."""

    def test_default_factors_are_noop(self):
        """Default factors should bin each axis by one."""
        arr = np.arange(6).reshape(2, 3)

        out = numeric.binning(arr)

        np.testing.assert_allclose(out, arr)

    def test_accepts_array_like_input(self):
        """Array-like inputs should match ndarray inputs."""
        arr = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

        out = numeric.binning(arr, factors=(2, 2))

        np.testing.assert_allclose(out, np.array([[2.5, 4.5], [10.5, 12.5]]))

    def test_factors_python_axis_order(self):
        """order_xyz=False means factors are already in NumPy axis order."""
        arr = np.arange(6 * 8).reshape(6, 8)

        out = numeric.binning(
            arr,
            factors=(3, 2),
            order_xyz=False,
            binfunc=np.sum,
        )

        expected = arr.reshape(2, 3, 4, 2).sum(axis=(1, 3))
        np.testing.assert_array_equal(out, expected)

    def test_factors_xyz_order(self):
        """order_xyz=True reverses xyz-style factors into NumPy axis order."""
        arr = np.arange(6 * 8).reshape(6, 8)

        out = numeric.binning(
            arr,
            factors=(2, 3),
            order_xyz=True,
            binfunc=np.sum,
        )

        expected = arr.reshape(2, 3, 4, 2).sum(axis=(1, 3))
        np.testing.assert_array_equal(out, expected)

    def test_nd_binning_preserves_leading_axis(self):
        """n-D binning should work when leading axes have factor one."""
        arr = np.arange(4 * 6 * 8).reshape(4, 6, 8)

        out = numeric.binning(
            arr,
            factors=(1, 3, 2),
            order_xyz=False,
            binfunc=np.sum,
        )

        expected = arr.reshape(4, 1, 2, 3, 4, 2).sum(axis=(1, 3, 5))
        assert out.shape == (4, 2, 4)
        np.testing.assert_array_equal(out, expected)

    def test_none_factor_collapses_axis(self):
        """None in factors should collapse the corresponding axis."""
        arr = np.arange(4 * 6 * 8).reshape(4, 6, 8)

        out = numeric.binning(
            arr,
            factors=(None, 3, 2),
            order_xyz=False,
            binfunc=np.mean,
        )

        expected = arr.reshape(1, 4, 2, 3, 4, 2).mean(axis=(1, 3, 5))
        assert out.shape == (1, 2, 4)
        np.testing.assert_allclose(out, expected)

    def test_trim_end_discards_trailing_elements(self):
        """trim_end=True should drop partial trailing bins on each axis."""
        arr = np.arange(5 * 7).reshape(5, 7)

        out = numeric.binning(
            arr,
            factors=(3, 2),
            binfunc=np.sum,
            trim_end=True,
        )

        expected = arr[:4, :6].reshape(2, 2, 2, 3).sum(axis=(1, 3))
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out, expected)

    def test_non_divisible_shape_requires_trim(self):
        """Non-divisible axes should raise a clear error unless trimming."""
        arr = np.arange(5 * 7).reshape(5, 7)

        with pytest.raises(ValueError, match="not divisible"):
            numeric.binning(arr, factors=(3, 2))

    @pytest.mark.parametrize(
        "factors",
        [(0, 2), (-1, 2), (1.5, 2), (True, 2)],
    )
    def test_invalid_factors_raise_value_error(self, factors):
        """Factors must be positive integers."""
        arr = np.arange(16).reshape(4, 4)

        with pytest.raises(ValueError, match="positive integer"):
            numeric.binning(arr, factors=factors, order_xyz=False)

    def test_factor_count_must_match_ndim(self):
        """The factors length must match arr.ndim for explicit factors."""
        arr = np.arange(2 * 4 * 4).reshape(2, 4, 4)

        with pytest.raises(ValueError, match="arr.ndim"):
            numeric.binning(arr, factors=(2, 2), order_xyz=False)

    def test_factor_larger_than_axis_raises(self):
        """Oversized factors should not silently create empty bins."""
        arr = np.arange(3 * 4).reshape(3, 4)

        with pytest.raises(ValueError, match="larger than"):
            numeric.binning(arr, factors=(5, 2), order_xyz=False)


class TestGainConversion:
    """Tests for gain conversion helpers."""

    def test_roundtrip(self):
        """dB and electron/ADU conversions should round-trip."""
        gain = 2.5
        np.testing.assert_allclose(
            numeric.dB2epadu(numeric.epadu2dB(gain)),
            gain,
            rtol=RTOL,
            atol=ATOL,
        )
