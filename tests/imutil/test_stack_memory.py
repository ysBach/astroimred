"""Memory-budget and uncertainty contracts for FITS combination."""

import numpy as np
import pytest
from astropy.nddata import CCDData, StdDevUncertainty

from astroimred.imutil._util_fits import (
    check_stack_memory,
    load_imcombine_item,
    load_imcombine_item_region,
)
from astroimred.imutil.imcombine import imcombine


@pytest.mark.parametrize("extension", [0, "UNCERT", ("ERR", 1)])
def test_uncertainty_request_rejected_before_input_io(extension: object) -> None:
    """Unsupported propagation must not silently return an unweighted result."""
    with pytest.raises(NotImplementedError, match="extension_uncertainty"):
        imcombine(["missing.fits"], extension_uncertainty=extension, verbose=False)


@pytest.mark.parametrize("region", [False, True])
def test_disabled_uncertainty_is_not_copied(region: bool) -> None:
    """Data-only loading ignores attached uncertainties and preserves the input."""
    ccd = CCDData(
        np.arange(12.0).reshape(3, 4),
        unit="adu",
        uncertainty=StdDevUncertainty(np.full((3, 4), 2.0)),
    )
    kwargs = {
        "trimsec": None,
        "extension": 0,
        "extension_mask": None,
        "extension_uncertainty": None,
    }
    if region:
        data, variance, mask = load_imcombine_item_region(
            ccd, (slice(1, 3), slice(1, 4)), ccd.shape, **kwargs
        )
        expected = ccd.data[1:3, 1:4]
    else:
        data, variance, mask = load_imcombine_item(ccd, **kwargs)
        expected = ccd.data
    np.testing.assert_array_equal(data, expected)
    assert variance is None
    assert not mask.any()
    np.testing.assert_array_equal(ccd.uncertainty.array, 2.0)


def test_memory_budget_cannot_shrink_persistent_diagnostics() -> None:
    """Even single-pixel chunks cannot fit below the retained output size."""
    with pytest.raises(ValueError, match="persistent"):
        check_stack_memory(
            4,
            (100, 100),
            "float32",
            "average",
            40_000,
            full=True,
            reject="sigclip",
            thresholds=True,
        )


def test_memory_budget_reserves_combined_image_without_diagnostics() -> None:
    """The stitched image remains resident even with diagnostics disabled."""
    with pytest.raises(ValueError, match="persistent"):
        check_stack_memory(4, (100, 100), "float32", "average", 39_999)


def test_nondefault_uncertainty_type_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="uncertainty_type"):
        imcombine(["missing.fits"], uncertainty_type="variance", verbose=False)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16"])
def test_full_stack_preserves_nan_gaps_and_values(dtype: str) -> None:
    """NaN workspaces preserve integer output compatibility without sentinel casts."""
    from astroimred.imutil._util_fits import load_full_stack

    images = [CCDData(np.full((2, 2), v), unit="adu") for v in (3, 7)]
    data, mask, variance, *_ = load_full_stack(
        images,
        offsets=np.array([[0, 0], [0, 3]]),
        shapes=np.array([[2, 2], [2, 2]]),
        sh_comb=(2, 5),
        dtype=dtype,
        mask=None,
        trimsec=None,
        extension=0,
        extension_mask=None,
        extension_uncertainty=None,
        extract_exptime=False,
        scale=None,
        zero=None,
        weight=None,
        zero_kw=None,
        scale_kw=None,
        zero_section=None,
        scale_section=None,
        scales=np.ones(2),
    )
    assert data.dtype == np.result_type(np.dtype(dtype), np.nan)
    assert np.isnan(data[:, :, 2]).all()
    np.testing.assert_array_equal(data[0, :, :2], 3)
    np.testing.assert_array_equal(data[1, :, 3:], 7)
    assert not mask.any()
    assert variance is None


@pytest.mark.parametrize("offset_aware", [False, True])
def test_chunks_cover_output_once_with_budget_for_one_pixel(offset_aware: bool) -> None:
    """A tight valid budget can split more than one spatial dimension."""
    kwargs = {}
    if offset_aware:
        kwargs = {
            "offsets": np.zeros((2, 2), dtype=int),
            "shapes": np.array([[3, 5], [3, 5]]),
        }
    # 60 retained bytes plus the legacy 3x estimate for 2 float32 samples.
    _, count, chunks = check_stack_memory(2, (3, 5), "float32", "average", 84, **kwargs)
    covered = np.zeros((3, 5), dtype=int)
    for chunk in chunks:
        covered[chunk] += 1
    np.testing.assert_array_equal(covered, 1)
    assert count == 15


def test_diagnostics_force_chunking_with_same_byte_budget() -> None:
    basic = check_stack_memory(4, (10, 10), "float32", "average", 6_000)
    diagnostic = check_stack_memory(
        4,
        (10, 10),
        "float32",
        "average",
        6_000,
        full=True,
        reject="sigclip",
        thresholds=True,
    )
    assert basic[1] == 1
    assert diagnostic[1] > 1


def test_requested_diagnostic_reserves_outputs_before_loading(tmp_path) -> None:
    """An output path enables persistent diagnostics even if full=False."""
    path = tmp_path / "input.fits"
    CCDData(np.ones((10, 10), dtype="float32"), unit="adu").write(path)
    with pytest.raises(ValueError, match="persistent"):
        imcombine(
            [path, path],
            full=False,
            output_mask=tmp_path / "mask.fits",
            memlimit=500,
            verbose=False,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16"])
def test_chunked_threshold_and_mask_results_match_full(tmp_path, dtype: str) -> None:
    """Chunking preserves threshold masks, caller masks, and output dtypes."""
    paths = []
    for i in range(3):
        path = tmp_path / f"frame_{i}.fits"
        data = np.arange(20, dtype=np.float32).reshape(4, 5) + 10 * i
        CCDData(data, unit="adu").write(path)
        paths.append(path)
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 1, 2] = True
    # Persistent: combined image plus total and threshold masks, 20 pixels.
    persistent = 20 * (np.dtype(dtype).itemsize + 6)
    common = {
        "dtype": dtype,
        "mask": mask,
        "thresholds": (0, 25),
        "full": True,
        "return_dict": True,
        "verbose": False,
    }
    full = imcombine(paths, memlimit=None, **common)
    chunked = imcombine(paths, memlimit=persistent + 200, **common)
    assert chunked["comb"].data.dtype == np.dtype(dtype)
    for key in full:
        if key == "comb":
            np.testing.assert_array_equal(chunked[key].data, full[key].data)
        elif full[key] is None:
            assert chunked[key] is None
        else:
            np.testing.assert_array_equal(chunked[key], full[key])
    assert mask.sum() == 1
