"""Tests for mask-aware zero/scale/weight normalization during image combination."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData

from astroimred.imutil.imcombine import imcombine


@pytest.mark.parametrize("memlimit", [None, 800])
@pytest.mark.parametrize("norm_mode", ["zero_only", "scale_only"])
@pytest.mark.parametrize("mask_source", ["ccddata", "extension", "param_mask", "union"])
def test_mask_aware_zero_and_scale_statistical_normalization(
    tmp_path: Path, memlimit: float | None, norm_mode: str, mask_source: str
) -> None:
    """Masked contamination pixels are excluded when computing zero or scale statistics.

    Separates zero-only from scale-only to avoid degenerate zero outcomes, and
    verifies full arrays including locations where only one sample survives.
    """
    shape = (10, 12)
    clean_val0 = 100.0
    clean_val1 = 200.0

    # Image 0 contaminated at (0, 0)
    arr0 = np.full(shape, clean_val0, dtype=np.float32)
    arr0[0, 0] = 9999.0

    # Image 1 contaminated at (1, 1) and (2, 2)
    arr1 = np.full(shape, clean_val1, dtype=np.float32)
    arr1[1, 1] = 8888.0
    arr1[2, 2] = -7777.0

    mask0 = np.zeros(shape, dtype=bool)
    mask0[0, 0] = True

    mask1_ext = np.zeros(shape, dtype=bool)
    mask1_ext[1, 1] = True

    mask1_param = np.zeros(shape, dtype=bool)
    mask1_param[2, 2] = True

    mask_param_all = np.zeros((2, *shape), dtype=bool)

    if mask_source == "ccddata":
        ccd0 = CCDData(arr0.copy(), unit="adu", mask=mask0)
        ccd1 = CCDData(arr1.copy(), unit="adu", mask=mask1_ext | mask1_param)
        inputs = [ccd0, ccd1]
        combine_kwargs: dict[str, Any] = {}
    elif mask_source == "extension":
        path0 = tmp_path / f"img0_ext_{norm_mode}.fits"
        path1 = tmp_path / f"img1_ext_{norm_mode}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(arr0, name="SCI"),
                fits.ImageHDU(mask0.astype(np.uint8), name="DQ"),
            ]
        ).writeto(path0)
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(arr1, name="SCI"),
                fits.ImageHDU((mask1_ext | mask1_param).astype(np.uint8), name="DQ"),
            ]
        ).writeto(path1)
        inputs = [path0, path1]
        combine_kwargs = {"extension": "SCI", "extension_mask": "DQ"}
    elif mask_source == "param_mask":
        mask_param_all[0] = mask0
        mask_param_all[1] = mask1_ext | mask1_param
        inputs = [CCDData(arr0.copy(), unit="adu"), CCDData(arr1.copy(), unit="adu")]
        combine_kwargs = {"mask": mask_param_all}
    elif mask_source == "union":
        path0 = tmp_path / f"img0_union_{norm_mode}.fits"
        path1 = tmp_path / f"img1_union_{norm_mode}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(arr0, name="SCI"),
                fits.ImageHDU(mask0.astype(np.uint8), name="DQ"),
            ]
        ).writeto(path0)
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(arr1, name="SCI"),
                fits.ImageHDU(mask1_ext.astype(np.uint8), name="DQ"),
            ]
        ).writeto(path1)
        mask_param_all[1] = mask1_param
        inputs = [path0, path1]
        combine_kwargs = {
            "extension": "SCI",
            "extension_mask": "DQ",
            "mask": mask_param_all,
        }
    else:
        raise ValueError(f"Unknown mask_source: {mask_source}")

    if norm_mode == "zero_only":
        # zero="mean", scale=None:
        # img0 zero = 100.0 -> normalized img0 = clean_val0 - 100.0 = 0.0
        # img1 zero = 200.0 -> normalized img1 = clean_val1 - 200.0 = 0.0
        # Clean pixels average = (0.0 + 0.0) / 2 = 0.0
        # (0, 0): only img1 survives -> normalized img1[0, 0] = 200.0 - 200.0 = 0.0
        # (1, 1): only img0 survives -> normalized img0[1, 1] = 100.0 - 100.0 = 0.0
        # (2, 2): only img0 survives -> normalized img0[2, 2] = 100.0 - 100.0 = 0.0
        result = imcombine(
            inputs,
            combine="average",
            zero="mean",
            zero_to_0th=False,
            scale_to_0th=False,
            memlimit=memlimit,
            **combine_kwargs,
        )
        expected = np.zeros(shape, dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-5)
    else:
        # scale_only: zero=None, scale="mean"
        # img0 scale = 100.0 -> normalized img0 = clean_val0 / 100.0 = 1.0
        # img1 scale = 200.0 -> normalized img1 = clean_val1 / 200.0 = 1.0
        # Clean pixels average = (1.0 + 1.0) / 2 = 1.0
        # (0, 0): only img1 survives -> normalized img1[0, 0] = 200.0 / 200.0 = 1.0
        # (1, 1): only img0 survives -> normalized img0[1, 1] = 100.0 / 100.0 = 1.0
        # (2, 2): only img0 survives -> normalized img0[2, 2] = 100.0 / 100.0 = 1.0
        result = imcombine(
            inputs,
            combine="average",
            scale="mean",
            zero_to_0th=False,
            scale_to_0th=False,
            memlimit=memlimit,
            **combine_kwargs,
        )
        expected = np.ones(shape, dtype=np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-5)


@pytest.mark.parametrize("memlimit", [None, 800])
def test_mask_aware_weights_with_stat_section(
    memlimit: float | None,
) -> None:
    """Statistical weights computed in scale_section ignore masked pixels."""
    shape = (10, 10)
    statsec = "[3:8, 3:8]"

    arr0 = np.full(shape, 10.0, dtype=np.float32)
    arr0[0, :] = 500.0  # Outside the statistic section.
    mask0 = np.zeros(shape, dtype=bool)

    arr1 = np.full(shape, 30.0, dtype=np.float32)
    arr1[4, 4] = 99999.0
    mask1 = np.zeros(shape, dtype=bool)
    mask1[4, 4] = True

    ccd0 = CCDData(arr0, unit="adu", mask=mask0)
    ccd1 = CCDData(arr1, unit="adu", mask=mask1)

    result = imcombine(
        [ccd0, ccd1],
        combine="average",
        weight="mean",
        scale_section=statsec,
        memlimit=memlimit,
    )
    # img0 weight = 10.0, img1 weight = 30.0 (masked 99999.0 ignored)
    # clean pixel average = (10*10 + 30*30) / (10+30) = 25.0
    # at (4, 4): img1 is masked, only img0 survives -> combined value = 10.0
    expected = np.full(shape, 25.0, dtype=np.float32)
    expected[0, :] = (500 * 10 + 30 * 30) / 40
    expected[4, 4] = 10.0
    np.testing.assert_allclose(result.data, expected, rtol=1e-5)


@pytest.mark.parametrize("memlimit", [None, 800])
def test_trimsec_and_statsec_with_explicit_post_trim_mask(
    tmp_path: Path, memlimit: float | None
) -> None:
    """Stat-section and explicit parameter mask operate in post-trim coordinates."""
    raw_shape = (12, 14)
    # trimsec "[2:11, 2:11]" gives trimmed shape (10, 10)
    trimsec = "[2:11, 2:11]"
    trimmed_shape = (10, 10)
    statsec = "[2:9, 2:9]"

    arr0 = np.full(raw_shape, 10.0, dtype=np.float32)
    arr1 = np.full(raw_shape, 30.0, dtype=np.float32)
    # Contamination in raw data that will be inside the trimmed statsec
    arr1[5, 5] = 99999.0

    # Explicit parameter mask uses post-trim coordinates: trimmed (4, 4) corresponds to raw (5, 5)
    param_mask = np.zeros((2, *trimmed_shape), dtype=bool)
    param_mask[1, 4, 4] = True

    path0 = tmp_path / "raw_img0.fits"
    path1 = tmp_path / "raw_img1.fits"
    fits.PrimaryHDU(arr0).writeto(path0)
    fits.PrimaryHDU(arr1).writeto(path1)

    result = imcombine(
        [path0, path1],
        combine="average",
        weight="mean",
        trimsec=trimsec,
        scale_section=statsec,
        mask=param_mask,
        memlimit=memlimit,
    )
    assert result.data.shape == trimmed_shape
    expected = np.full(trimmed_shape, 25.0, dtype=np.float32)
    expected[4, 4] = 10.0
    np.testing.assert_allclose(result.data, expected, rtol=1e-5)


@pytest.mark.parametrize("memlimit", [None, 600])
def test_integer_inputs_and_inputs_nonmutation(memlimit: float | None) -> None:
    """Integer arrays preserve precision and input objects/arrays are not mutated."""
    shape = (6, 8)
    arr0 = np.tile(np.array([10, 11], dtype=np.int16), (6, 4))
    arr0[1, :2] = 30000
    mask0 = np.zeros(shape, dtype=bool)
    mask0[1, :2] = True

    arr1 = np.tile(np.array([20, 21], dtype=np.int16), (6, 4))
    mask1 = np.zeros(shape, dtype=bool)

    ccd0 = CCDData(arr0, unit="adu", mask=mask0)
    ccd1 = CCDData(arr1, unit="adu", mask=mask1)

    orig_arr0 = arr0.copy()
    orig_mask0 = mask0.copy()
    orig_arr1 = arr1.copy()
    orig_mask1 = mask1.copy()
    for array in (ccd0.data, ccd0.mask, ccd1.data, ccd1.mask):
        array.setflags(write=False)

    result = imcombine(
        [ccd0, ccd1],
        combine="average",
        zero="mean",
        scale_to_0th=False,
        zero_to_0th=False,
        memlimit=memlimit,
    )

    expected = np.tile([-0.5, 0.5], (6, 4))
    np.testing.assert_allclose(result.data, expected, atol=1e-5)

    np.testing.assert_array_equal(ccd0.data, orig_arr0)
    np.testing.assert_array_equal(ccd0.mask, orig_mask0)
    np.testing.assert_array_equal(ccd1.data, orig_arr1)
    np.testing.assert_array_equal(ccd1.mask, orig_mask1)


@pytest.mark.parametrize("memlimit", [None, 800])
def test_numeric_specifications_independent_of_mask(memlimit: float | None) -> None:
    """Explicit numeric zero/scale/weights succeed on all-masked inputs and yield all NaN."""
    shape = (8, 8)
    arr0 = np.full(shape, 5.0, dtype=np.float32)
    mask0 = np.ones(shape, dtype=bool)

    arr1 = np.full(shape, 15.0, dtype=np.float32)
    mask1 = np.ones(shape, dtype=bool)

    ccd0 = CCDData(arr0, unit="adu", mask=mask0)
    ccd1 = CCDData(arr1, unit="adu", mask=mask1)

    result = imcombine(
        [ccd0, ccd1],
        combine="average",
        zero=[1.0, 3.0],
        scale=[2.0, 4.0],
        weight=[1.0, 3.0],
        zero_to_0th=False,
        scale_to_0th=False,
        memlimit=memlimit,
    )
    # All samples are masked, so combined image data must be entirely NaN
    assert np.isnan(result.data).all()


@pytest.mark.parametrize("memlimit", [None, 200])
@pytest.mark.parametrize("param_name", ["zero", "scale", "weight"])
def test_empty_selection_raises_with_parameter_name_and_index(
    memlimit: float | None, param_name: str
) -> None:
    """Empty finite unmasked selection raises ValueError with param name and image index."""
    shape = (5, 5)
    arr0 = np.full(shape, 10.0, dtype=np.float32)
    mask0 = np.zeros(shape, dtype=bool)

    arr1 = np.full(shape, 20.0, dtype=np.float32)
    mask1 = np.ones(shape, dtype=bool)

    ccd0 = CCDData(arr0, unit="adu", mask=mask0)
    ccd1 = CCDData(arr1, unit="adu", mask=mask1)

    kwargs: dict[str, Any] = {
        "zero_to_0th": False,
        "scale_to_0th": False,
        "memlimit": memlimit,
    }
    kwargs[param_name] = "mean"

    # Strict match: parameter name, 'no finite unmasked', and 'input 1'
    regex = rf"(?i){param_name}.*no finite unmasked.*input 1"
    with pytest.raises(ValueError, match=regex):
        imcombine([ccd0, ccd1], combine="average", **kwargs)


@pytest.mark.parametrize("memlimit", [None, 800])
def test_callable_stat_receives_nan_sentinel_and_preserves_shape(
    memlimit: float | None,
) -> None:
    """Custom callable returns scalar float, receives (100, 1), called exactly once."""
    shape = (10, 10)
    arr0 = np.full(shape, 50.0, dtype=np.float32)
    arr0[0, :5] = 999.0
    mask0 = np.zeros(shape, dtype=bool)
    mask0[0, :5] = True

    ccd0 = CCDData(arr0, unit="adu", mask=mask0)

    received_workspaces: list[np.ndarray] = []

    def custom_zero_stat(workspace: np.ndarray) -> float:
        received_workspaces.append(workspace.copy())
        assert workspace.shape == (100, 1)
        np.testing.assert_array_equal(np.isnan(workspace[:, 0]), mask0.ravel())
        value = float(np.nanmean(workspace))
        workspace[:] = -1000  # User code cannot overwrite combination pixels.
        return value

    result = imcombine(
        [ccd0],
        combine="average",
        zero=custom_zero_stat,
        zero_to_0th=False,
        scale_to_0th=False,
        memlimit=memlimit,
    )
    # The nanmean of unmasked samples is 50.0. (50.0 - 50.0) = 0.0
    expected = np.zeros(shape, dtype=np.float32)
    expected[mask0] = np.nan
    np.testing.assert_allclose(result.data, expected, atol=1e-5, equal_nan=True)
    assert len(received_workspaces) == 1


@pytest.mark.parametrize("memlimit", [None, 300])
@pytest.mark.parametrize(
    ("statistic", "expected_zero"),
    [("mean", 3.5), ("median", 3.5), ("mean_sc", 3.5), ("median_sc", 3.5), ("sum", 21)],
)
def test_statistic_families_exclude_masks_and_nonfinite_pixels(
    memlimit: float | None, statistic: str, expected_zero: float
) -> None:
    """Each named statistic uses the same six valid samples, including clipped stats."""
    data = np.full((6, 8), 1000.0)
    data[0, :6] = np.arange(1, 7)
    data[0, 6:] = [np.nan, np.inf]
    data[1, 0] = -np.inf
    mask = np.ones(data.shape, dtype=bool)
    mask[0, :] = False
    mask[1, 0] = False
    result = imcombine(
        [CCDData(data, mask=mask, unit="adu")],
        zero=statistic,
        zero_to_0th=False,
        memlimit=memlimit,
    )
    expected = np.full(data.shape, np.nan)
    expected[0, :6] = np.arange(1, 7) - expected_zero
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


@pytest.mark.parametrize("memlimit", [None, 300])
def test_sparse_offsets_keep_original_statistical_weights(
    memlimit: float | None,
) -> None:
    """Weights use each masked input footprint before spatial chunk compaction."""
    images = []
    for level in (10, 20, 30):
        data = np.full((3, 4), level, dtype=np.float32)
        data[0, 0] = 10000
        mask = np.zeros(data.shape, dtype=bool)
        mask[0, 0] = True
        images.append(CCDData(data, mask=mask, unit="adu"))
    result = imcombine(
        images,
        weight="mean",
        offsets=[[0, 0], [0, 2], [0, 12]],
        memlimit=memlimit,
    )
    expected = np.full((3, 16), np.nan)
    expected[:, :2] = 10
    expected[:, 2:4] = 50 / 3
    expected[:, 4:6] = 20
    expected[:, 12:] = 30
    expected[0, [0, 12]] = np.nan
    expected[0, 2] = 10
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


@pytest.mark.parametrize("memlimit", [None, 300])
def test_thresholds_do_not_select_normalization_samples(
    memlimit: float | None,
) -> None:
    """Threshold rejection keeps its separate role after statistic estimation."""
    data = np.full((6, 8), 10.0)
    data[0, 0] = 490
    result = imcombine(
        [CCDData(data, unit="adu")],
        zero="mean",
        zero_to_0th=False,
        thresholds=(0, 100),
        memlimit=memlimit,
    )
    # The raw mean is 20; the value 490 is excluded only from combination.
    expected = np.full(data.shape, -10.0)
    expected[0, 0] = np.nan
    np.testing.assert_allclose(result.data, expected, equal_nan=True)
