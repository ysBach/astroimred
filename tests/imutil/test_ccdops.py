"""Tests for CCDData manipulation helpers."""

import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.nddata.utils import PartialOverlapError

import astroimred as air
from astroimred.imutil import ccdops


def _header_value(header, key):
    value = header[key]
    if isinstance(value, tuple):
        return value[0]
    return value


class TestCcdUtils:
    """Tests for CCD helper module exports."""

    def test_ccd_helpers_have_canonical_modules(self):
        """CCD operations are exposed from ccdops."""
        assert air.imslice is ccdops.imslice
        assert air.imcut is ccdops.imcut
        assert air.cut_ccd is ccdops.cut_ccd
        assert air.bin_ccd is ccdops.bin_ccd
        assert air.set_ccd_attribute is ccdops.set_ccd_attribute

    def test_imcut_trim_centered_cutout(self):
        """imcut returns a centered ndarray cutout without metadata."""
        data = np.arange(100).reshape(10, 10)

        out = ccdops.imcut(data, position=(5, 5), size=(4, 4), mode="trim")

        np.testing.assert_array_equal(out, data[3:7, 3:7])

    def test_imcut_trim_partial_overlap(self):
        """trim mode returns only the overlapping pixels."""
        data = np.arange(100).reshape(10, 10)

        out = ccdops.imcut(data, position=(0, 0), size=(4, 4), mode="trim")

        np.testing.assert_array_equal(out, data[0:2, 0:2])

    def test_imcut_partial_fill(self):
        """partial mode preserves requested shape and fills missing pixels."""
        data = np.arange(100).reshape(10, 10)

        out = ccdops.imcut(
            data, position=(0, 0), size=(4, 4), mode="partial", fill_value=-1
        )

        expected = np.full((4, 4), -1)
        expected[2:4, 2:4] = data[0:2, 0:2]
        np.testing.assert_array_equal(out, expected)

    def test_imcut_strict_rejects_partial_overlap(self):
        """strict mode requires the requested box to be fully in frame."""
        data = np.arange(100).reshape(10, 10)

        with pytest.raises(PartialOverlapError):
            ccdops.imcut(data, position=(0, 0), size=(4, 4), mode="strict")

    def test_imcut_copy_false_returns_view(self):
        """copy=False lets trim/strict callers avoid allocation."""
        data = np.arange(100).reshape(10, 10)

        out = ccdops.imcut(data, position=(5, 5), size=(4, 4), copy=False)

        assert np.shares_memory(out, data)

    def test_imcut_accepts_ccddata_data_only(self):
        """CCDData inputs are accepted, but only the data array is cut."""
        ccd = CCDData(np.arange(100).reshape(10, 10), unit="adu")

        out = ccdops.imcut(ccd, position=(5, 5), size=4)

        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, ccd.data[3:7, 3:7])

    def test_bin_ccd_uses_xyz_header_keys_for_2d(self):
        """2-D binning should write X/Y binning header cards."""
        ccd = CCDData(np.arange(6 * 8).reshape(6, 8), unit="adu")

        out = ccdops.bin_ccd(ccd, factors=(2, 3), binfunc=np.sum)

        assert out.shape == (2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert "BINNING1" not in out.header

    def test_bin_ccd_uses_xyz_header_keys_for_3d(self):
        """3-D binning should write X/Y/Z binning header cards."""
        ccd = CCDData(np.arange(4 * 6 * 8).reshape(4, 6, 8), unit="adu")

        out = ccdops.bin_ccd(ccd, factors=(2, 3, 2), binfunc=np.sum)

        assert out.shape == (2, 2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert _header_value(out.header, "ZBINNING") == 2
        assert "BINNING1" not in out.header

    def test_bin_ccd_uses_numbered_header_keys_for_4d(self):
        """Higher-dimensional binning should write generic numbered cards."""
        ccd = CCDData(np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6), unit="adu")

        out = ccdops.bin_ccd(ccd, factors=(2, 2, 3, 1), binfunc=np.sum)

        assert out.shape == (2, 1, 2, 3)
        assert _header_value(out.header, "BINNING1") == 2
        assert _header_value(out.header, "BINNING2") == 2
        assert _header_value(out.header, "BINNING3") == 3
        assert _header_value(out.header, "BINNING4") == 1
        assert "XBINNING" not in out.header

    def test_bin_ccd_default_is_noop_for_nd_data(self):
        """Default factors should be a no-op for any data dimensionality."""
        ccd = CCDData(np.arange(2 * 3 * 4).reshape(2, 3, 4), unit="adu")

        out = ccdops.bin_ccd(ccd)

        assert out is ccd

    def test_bin_ccd_header_records_effective_none_factor(self):
        """None factors should be recorded as their effective collapse factor."""
        ccd = CCDData(np.arange(4 * 6 * 8).reshape(4, 6, 8), unit="adu")

        out = ccdops.bin_ccd(ccd, factors=(2, 3, None), binfunc=np.sum)

        assert out.shape == (1, 2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert _header_value(out.header, "ZBINNING") == 4
