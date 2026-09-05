"""Tests for standalone numeric helpers."""

import numpy as np
import pytest
import reducers as rd

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

    @pytest.mark.parametrize(
        "uncertainty", [{"err": 2.0}, {"var": 4.0}, {"ivar": 0.25}]
    )
    def test_scalar_uncertainty_counts_every_measurement(self, uncertainty):
        mean, se = numeric.wvg([1.0, 3.0], return_se=True, **uncertainty)
        np.testing.assert_allclose(mean, 2.0)
        np.testing.assert_allclose(se, np.sqrt(2.0))

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1), ()])
    def test_scalar_uncertainty_matches_repeated_errors(self, axis):
        values = np.arange(6.0).reshape(2, 3)
        mean, se = numeric.wvg(values, err=2.0, axis=axis, return_se=True)
        repeated = numeric.wvg(
            values, err=np.full(values.shape, 2.0), axis=axis, return_se=True
        )
        np.testing.assert_allclose(mean, np.mean(values, axis=axis))
        np.testing.assert_allclose(
            se, 2.0 / np.sqrt(np.sum(np.ones_like(values), axis=axis))
        )
        np.testing.assert_allclose(mean, repeated[0])
        np.testing.assert_allclose(se, repeated[1])

    @pytest.mark.parametrize("axis", [None, 0, -1])
    @pytest.mark.parametrize("layout", ["contiguous", "strided", "big-endian"])
    def test_weighted_results_across_array_layouts(self, axis, layout):
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        weights = np.array([[1.0, 3.0], [2.0, 4.0]])
        if layout == "strided":
            values = np.repeat(values, 2, axis=1)[:, ::2]
            weights = np.repeat(weights, 2, axis=1)[:, ::2]
        elif layout == "big-endian":
            values = values.astype(">f8")
            weights = weights.astype(">f8")
        expected = {
            None: (4.8, 1 / np.sqrt(10)),
            0: ([11 / 3, 37 / 7], 1 / np.sqrt([3.0, 7.0])),
            -1: ([2.5, 19 / 3], 1 / np.sqrt([4.0, 6.0])),
        }
        mean, se = numeric.wvg(values, ivar=weights, axis=axis, return_se=True)
        np.testing.assert_allclose(mean, expected[axis][0], rtol=1e-14)
        np.testing.assert_allclose(se, expected[axis][1], rtol=1e-14)

    def test_broadcast_weights_count_repeated_rows(self):
        mean, se = numeric.wvg(
            [[1.0, 3.0], [5.0, 7.0]], ivar=[[1.0, 3.0]], return_se=True
        )
        np.testing.assert_allclose(mean, 4.5)
        np.testing.assert_allclose(se, 1 / np.sqrt(8))

    @pytest.mark.parametrize("axis", [0, -2, 1, -1])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, ">f8"])
    def test_axis_vector_weights(self, axis: int, dtype: object) -> None:
        values = np.array([[1, 3, 5], [7, 9, 11]], dtype=dtype)
        if axis in (1, -1):
            values = values.T
        mean, se = numeric.wvg(
            values, ivar=np.array([1, 3], dtype=dtype), axis=axis, return_se=True
        )
        np.testing.assert_allclose(mean, [5.5, 7.5, 9.5])
        np.testing.assert_allclose(se, [0.5, 0.5, 0.5])

    def test_middle_axis_vector_weights(self) -> None:
        values = np.array([[[1, 3], [5, 7], [9, 11]]], dtype=float)
        mean, se = numeric.wvg(values, ivar=[1, 2, 1], axis=1, return_se=True)
        np.testing.assert_allclose(mean, [[5, 7]])
        np.testing.assert_allclose(se, [[0.5, 0.5]])

    def test_integer_errors_are_squared_without_integer_overflow(self):
        mean, se = numeric.wvg(
            [1.0, 3.0], err=np.array([50000, 50000], dtype=np.int32), return_se=True
        )
        np.testing.assert_allclose(mean, 2.0)
        np.testing.assert_allclose(se, 50000 / np.sqrt(2))

    def test_complex_values_with_real_errors(self):
        mean, se = numeric.wvg([1 + 2j, 3 + 4j], err=2.0, return_se=True)
        np.testing.assert_allclose(mean, 2 + 3j)
        np.testing.assert_allclose(se, np.sqrt(2))

    def test_complex_values_with_axis_weights(self) -> None:
        mean, se = numeric.wvg(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            ivar=[1, 3],
            axis=0,
            return_se=True,
        )
        np.testing.assert_allclose(mean, [4 + 5j, 6 + 7j])
        np.testing.assert_allclose(se, [0.5, 0.5])

    @pytest.mark.parametrize("error", [TypeError, ValueError, RuntimeError])
    def test_unexpected_backend_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise error("backend failure")

        monkeypatch.setattr(rd, "sum", fail)
        with pytest.raises(error, match="backend failure"):
            numeric.wvg([1.0, 3.0], err=2.0)


class TestQuantile:
    def test_nan_quantile_axis_none_uses_reducers(self, monkeypatch):
        """NaN-aware flattened quantiles should delegate to reducers."""
        calls = {}

        def fake_nanquantile(values, q, axis=None, *, ignore_inf=False, validate=True):
            calls["values"] = values.copy()
            calls["q"] = q
            calls["axis"] = axis
            calls["ignore_inf"] = ignore_inf
            calls["validate"] = validate
            return np.array([1.25, 4.75])

        monkeypatch.setattr(rd, "nanquantile", fake_nanquantile)

        result = numeric.quantile_lh(
            np.array([1.0, np.nan, 3.0, 5.0]),
            0.25,
            0.75,
            nanfunc=True,
        )

        np.testing.assert_array_equal(result, [1.25, 4.75])
        np.testing.assert_array_equal(calls["values"], [1.0, np.nan, 3.0, 5.0])
        assert calls == {
            "values": calls["values"],
            "q": (0.25, 0.75),
            "axis": None,
            "ignore_inf": False,
            "validate": True,
        }

    def test_nan_quantile_empty_input_returns_pair_of_nan(self):
        """Empty flattened input follows reducers' two-quantile shape."""
        result = numeric.quantile_lh(np.array([]), 0.25, 0.75, nanfunc=True)

        np.testing.assert_array_equal(np.shape(result), (2,))
        np.testing.assert_array_equal(np.isnan(result), [True, True])

    def test_plain_quantile_axis_none_uses_reducers(self, monkeypatch):
        """Plain flattened quantiles should delegate to reducers."""
        calls = {}

        def fake_quantile(values, q, axis=None, *, validate=True):
            calls["values"] = values.copy()
            calls["q"] = q
            calls["axis"] = axis
            calls["validate"] = validate
            return np.array([1.5, 4.5])

        monkeypatch.setattr(rd, "quantile", fake_quantile)

        data = np.array([1.0, 3.0, 5.0])
        result = numeric.quantile_lh(data, 0.25, 0.75)

        np.testing.assert_array_equal(result, [1.5, 4.5])
        np.testing.assert_array_equal(calls["values"], [1.0, 3.0, 5.0])
        assert calls["q"] == (0.25, 0.75)
        assert calls["axis"] is None
        assert calls["validate"] is True
        np.testing.assert_array_equal(data, [1.0, 3.0, 5.0])

    def test_plain_quantile_axis_none_delegates_int64_to_reducers(self, monkeypatch):
        """Integer dtype support should come from reducers, not local whitelists."""
        calls = {}

        def fake_quantile(values, q, axis=None, *, validate=True):
            calls["dtype"] = values.dtype
            return np.array([2.0, 4.0])

        monkeypatch.setattr(rd, "quantile", fake_quantile)

        data = np.array([2**60, 2**60 + 2, 2**60 + 4], dtype=np.int64)
        result = numeric.quantile_lh(data, 0.25, 0.75)

        np.testing.assert_array_equal(result, [2.0, 4.0])
        assert calls["dtype"] == np.dtype(np.int64)

    def test_nan_quantile_axis_last_uses_reducers(self, monkeypatch):
        """NaN-aware last-axis quantiles should delegate to reducers."""
        calls = {}

        def fake_nanquantile(values, q, axis=None, *, ignore_inf=False, validate=True):
            calls["shape"] = values.shape
            calls["q"] = q
            calls["axis"] = axis
            calls["ignore_inf"] = ignore_inf
            calls["validate"] = validate
            return np.full((2, values.shape[0], values.shape[1]), 7.0)

        monkeypatch.setattr(rd, "nanquantile", fake_nanquantile)

        arr = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)
        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=-1, nanfunc=True)

        assert calls == {
            "shape": (2, 3, 4),
            "q": (0.25, 0.75),
            "axis": -1,
            "ignore_inf": False,
            "validate": True,
        }
        np.testing.assert_array_equal(result, np.full((2, 2, 3), 7.0))

    def test_plain_quantile_axis0_uses_reducers(self, monkeypatch):
        """Plain axis-0 quantiles should delegate to reducers."""
        calls = {}

        def fake_quantile(values, q, axis=None, *, validate=True):
            calls["shape"] = values.shape
            calls["q"] = q
            calls["axis"] = axis
            calls["validate"] = validate
            return np.full((2, values.shape[1]), 8.0)

        monkeypatch.setattr(rd, "quantile", fake_quantile)

        arr = np.arange(3 * 4.0).reshape(3, 4)
        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=0, nanfunc=False)

        assert calls == {
            "shape": (3, 4),
            "q": (0.25, 0.75),
            "axis": 0,
            "validate": True,
        }
        np.testing.assert_array_equal(result, np.full((2, 4), 8.0))

    def test_plain_quantile_axis0_int64_uses_reducers(self, monkeypatch):
        """Axis quantiles should not use local dtype gates."""
        calls = {}

        def fake_quantile(values, q, axis=None, *, validate=True):
            calls["dtype"] = values.dtype
            return np.array([[1.0, 2.0], [3.0, 4.0]])

        monkeypatch.setattr(rd, "quantile", fake_quantile)

        arr = np.array(
            [[2**60, 2**60 + 2], [2**60 + 4, 2**60 + 8]],
            dtype=np.int64,
        )
        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=0, nanfunc=False)

        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
        assert calls["dtype"] == np.dtype(np.int64)

    def test_nan_quantile_axis_with_inf_follows_reducers_semantics(self):
        """Reducers semantics, not NumPy fallback, define infinity handling."""
        arr = np.array([[1.0, np.inf], [3.0, 5.0]])

        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=0, nanfunc=True)
        expected = rd.nanquantile(arr, (0.25, 0.75), axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_plain_quantile_axis_with_inf_follows_reducers_semantics(self):
        """Plain quantiles should keep reducers infinity semantics."""
        arr = np.array([[1.0, np.inf], [3.0, 5.0]])

        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=0, nanfunc=False)
        expected = rd.quantile(arr, (0.25, 0.75), axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_complex_quantile_uses_reducers_error(self):
        """Complex inputs should surface reducers' real-numeric validation."""
        with pytest.raises(TypeError, match="real numeric dtypes"):
            numeric.quantile_lh(np.array([1 + 2j, 2 + 1j]), 0.25, 0.75)

    def test_nan_quantile_axis_all_nan_slice_matches_reducers(self):
        """All-NaN slices should follow reducers semantics."""
        arr = np.array([[np.nan, 1.0], [np.nan, 3.0]])

        result = numeric.quantile_lh(arr, 0.25, 0.75, axis=0, nanfunc=True)
        expected = rd.nanquantile(arr, (0.25, 0.75), axis=0)

        np.testing.assert_allclose(result, expected, equal_nan=True)


def test_quantile_lh_rejects_removed_interpolation_api():
    """Removed interpolation keyword arguments should no longer be accepted."""
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError):
        numeric.quantile_lh(data, 0.25, 0.75, interpolation="nearest")
    with pytest.raises(TypeError):
        numeric.quantile_lh(data, 0.25, 0.75, linterp="higher")
    with pytest.raises(TypeError):
        numeric.quantile_lh(data, 0.25, 0.75, hinterp="lower")


def test_quantile_lh_middle_axis_follows_reducers_contract():
    """Only reducers-supported axes should be accepted."""
    data = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)

    with pytest.raises(NotImplementedError, match="axis"):
        numeric.quantile_lh(data, 0.25, 0.75, axis=1)


def test_quantile_sigma_rejects_removed_interpolation_api():
    """Quantile-sigma should expose the same simplified quantile API."""
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError):
        numeric.quantile_sigma(data, interpolation="nearest")


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

    def test_median_binning_uses_reducers(self, monkeypatch):
        """Median binning should dispatch through reducers' stack median."""
        import reducers as rd

        calls = {}

        def fake_median(stack, axis=None, *, validate=True):
            calls["shape"] = stack.shape
            calls["axis"] = axis
            calls["validate"] = validate
            return np.full(stack.shape[1:], 7.0)

        monkeypatch.setattr(rd, "median", fake_median)

        arr = np.arange(16.0).reshape(4, 4)
        out = numeric.binning(arr, factors=(2, 2), order_xyz=False, binfunc=np.median)

        assert calls["shape"] == (4, 2, 2)
        assert calls["axis"] == 0
        assert calls["validate"] is False
        np.testing.assert_array_equal(out, np.full((2, 2), 7.0))

    @pytest.mark.parametrize(
        ("binfunc_name", "reducer_name"),
        [
            ("mean", "mean"),
            ("nanmean", "nanmean"),
            ("median", "median"),
            ("nanmedian", "nanmedian"),
            ("sum", "sum"),
            ("nansum", "nansum"),
        ],
    )
    def test_float_binning_known_reducers_use_reducers(
        self,
        monkeypatch,
        binfunc_name,
        reducer_name,
    ):
        """Known NumPy reducers should dispatch through reducers for float data."""
        import reducers as rd

        calls = {}
        reducer = getattr(rd, reducer_name)

        def fake_reducer(stack, axis=None, *, validate=True):
            calls["shape"] = stack.shape
            calls["axis"] = axis
            calls["validate"] = validate
            calls["dtype"] = stack.dtype
            return np.full(stack.shape[1:], 7.0, dtype=stack.dtype)

        monkeypatch.setattr(rd, reducer_name, fake_reducer)

        arr = np.arange(16.0, dtype=np.float32).reshape(4, 4)
        out = numeric.binning(
            arr,
            factors=(2, 2),
            order_xyz=False,
            binfunc=getattr(np, binfunc_name),
        )

        assert reducer is not getattr(rd, reducer_name)
        assert calls == {
            "shape": (4, 2, 2),
            "axis": 0,
            "validate": False,
            "dtype": np.dtype("float32"),
        }
        np.testing.assert_array_equal(out, np.full((2, 2), 7.0, dtype=np.float32))

    def test_integer_sum_binning_preserves_numpy_dtype(self, monkeypatch):
        """Integer sum keeps NumPy's integer accumulator semantics."""
        import reducers as rd

        def fail_sum(*args, **kwargs):
            raise AssertionError("integer sum should use NumPy")

        monkeypatch.setattr(rd, "sum", fail_sum)

        arr = np.arange(16, dtype=np.int16).reshape(4, 4)
        out = numeric.binning(arr, factors=(2, 2), order_xyz=False, binfunc=np.sum)
        expected = arr.reshape(2, 2, 2, 2).sum(axis=(1, 3))

        assert out.dtype == expected.dtype
        np.testing.assert_array_equal(out, expected)

    def test_reducer_binning_accepts_big_endian_data(self):
        """Reducer-backed binning should handle FITS-style big-endian arrays."""
        arr = np.arange(16.0, dtype=">f8").reshape(4, 4)

        out = numeric.binning(arr, factors=(2, 2), order_xyz=False, binfunc=np.mean)
        expected = arr.reshape(2, 2, 2, 2).mean(axis=(1, 3))

        np.testing.assert_allclose(out, expected)

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
