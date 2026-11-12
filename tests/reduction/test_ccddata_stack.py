"""Tests for imcombine with in-memory CCDData and missing NAXIS headers."""

import numpy as np
import pytest
from astropy.nddata import CCDData

from astroimred.imutil.imcombine import imcombine


def test_imcombine_in_memory_ccddata_without_naxis():
    """In-memory CCDData lacking NAXIS headers must combine successfully."""
    arr1 = np.full((12, 16), 10.0, dtype=np.float32)
    arr2 = np.full((12, 16), 20.0, dtype=np.float32)
    arr3 = np.full((12, 16), 30.0, dtype=np.float32)

    ccd1 = CCDData(arr1, unit="adu")
    ccd2 = CCDData(arr2, unit="adu")
    ccd3 = CCDData(arr3, unit="adu")

    # Header of newly created CCDData has no NAXIS keyword
    assert "NAXIS" not in ccd1.header

    combined = imcombine([ccd1, ccd2, ccd3], combine="median")
    np.testing.assert_allclose(combined, 20.0)


def test_imcombine_stale_naxis_header_uses_data_shape():
    """Stale NAXIS in in-memory CCDData header must not override actual data shape."""
    arr1 = np.full((10, 10), 1.0, dtype=np.float32)
    arr2 = np.full((10, 10), 3.0, dtype=np.float32)

    ccd1 = CCDData(arr1, unit="adu")
    ccd2 = CCDData(arr2, unit="adu")

    # Insert stale / incorrect NAXIS header cards
    ccd1.header["NAXIS"] = 3
    ccd1.header["NAXIS1"] = 99
    ccd1.header["NAXIS2"] = 99
    ccd1.header["NAXIS3"] = 99
    ccd2.header["NAXIS"] = 4

    combined = imcombine([ccd1, ccd2], combine="mean")
    assert combined.shape == (10, 10)
    np.testing.assert_allclose(combined, 2.0)


def test_imcombine_exposure_scale_header_branch():
    """Exposure scaling should read header exposure times from in-memory CCDData."""
    arr1 = np.full((8, 8), 10.0, dtype=np.float32)
    arr2 = np.full((8, 8), 20.0, dtype=np.float32)

    ccd1 = CCDData(arr1, unit="adu")
    ccd2 = CCDData(arr2, unit="adu")

    ccd1.header["EXPTIME"] = 1.0
    ccd2.header["EXPTIME"] = 2.0

    # Scaled to 0th: scale factor for ccd1 is 1.0/1.0 = 1, for ccd2 is 1.0/2.0 = 0.5
    # ccd1: 10 * 1 = 10; ccd2: 20 * 0.5 = 10 -> average is 10
    combined = imcombine(
        [ccd1, ccd2],
        combine="mean",
        scale="exposure",
        exposure_key="EXPTIME",
        scale_to_0th=True,
    )
    np.testing.assert_allclose(combined, 10.0)


def test_imcombine_mixed_ndim_rejected():
    """Combining inputs with mismatched ndim must raise ValueError."""
    ccd_2d = CCDData(np.ones((10, 10), dtype=np.float32), unit="adu")
    ccd_3d = CCDData(np.ones((2, 10, 10), dtype=np.float32), unit="adu")

    with pytest.raises(ValueError, match="identical ndim"):
        imcombine([ccd_2d, ccd_3d], combine="mean")


def test_imcombine_inputs_not_mutated():
    """Input CCDData arrays and headers must not be mutated by imcombine."""
    arr1 = np.full((8, 8), 5.0, dtype=np.float32)
    arr2 = np.full((8, 8), 15.0, dtype=np.float32)

    ccd1 = CCDData(arr1, unit="adu")
    ccd2 = CCDData(arr2, unit="adu")
    ccd1.header["TESTKEY"] = "orig1"
    ccd2.header["TESTKEY"] = "orig2"

    data1_orig = ccd1.data.copy()
    data2_orig = ccd2.data.copy()

    _ = imcombine([ccd1, ccd2], combine="mean")

    np.testing.assert_array_equal(ccd1.data, data1_orig)
    np.testing.assert_array_equal(ccd2.data, data2_orig)
    assert ccd1.header["TESTKEY"] == "orig1"
    assert ccd2.header["TESTKEY"] == "orig2"
