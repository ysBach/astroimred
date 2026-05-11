import numpy as np
import pytest

from astroimred.imutil._util_lmed import (
    lmedian,
    median,
    nanlmedian,
    nanmedian,
)


class TestLowerMedian:
    """Tests for lower-median helpers."""

    def test_lmedian_even_axis(self):
        arr = np.array([[1, 4, 2, 3], [10, 40, 20, 30]])

        np.testing.assert_array_equal(lmedian(arr, axis=1), [2, 20])
        np.testing.assert_array_equal(median(arr, axis=1, choose="upper"), [3, 30])
        np.testing.assert_allclose(median(arr, axis=1), np.median(arr, axis=1))

    def test_lmedian_keepdims_tuple_axis(self):
        arr = np.arange(24).reshape(2, 3, 4)

        result = lmedian(arr, axis=(1, 2), keepdims=True)

        assert result.shape == (2, 1, 1)
        np.testing.assert_array_equal(result.ravel(), [5, 17])

    def test_nanlmedian_ignores_nan(self):
        arr = np.array([[1.0, np.nan, 2.0, 3.0], [4.0, np.nan, 6.0, 8.0]])

        np.testing.assert_array_equal(nanlmedian(arr, axis=1), [2.0, 6.0])
        np.testing.assert_array_equal(
            nanmedian(arr, axis=1, choose="upper"), [2.0, 6.0]
        )

    def test_nanlmedian_warns_for_all_nan_slice(self):
        arr = np.array([[1.0, 2.0], [np.nan, np.nan]])

        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            result = nanlmedian(arr, axis=1)

        assert result[0] == 1.0
        assert np.isnan(result[1])

    def test_out_argument(self):
        arr = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        out = np.empty(2)

        result = lmedian(arr, axis=0, out=out)

        assert result is out
        np.testing.assert_array_equal(out, [2.0, 5.0])
