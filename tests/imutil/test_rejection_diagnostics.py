"""Rejection provenance and spatial growth through full and chunked FITS I/O."""

from math import prod
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData

from astroimred.imutil._util_fits import check_stack_memory
from astroimred.imutil.imcombine import imcombine


@pytest.mark.parametrize("memlimit", [None, 2_100])
def test_detailed_flags_and_existing_returns(memlimit: float | None) -> None:
    """Detailed flags explain each input sample without changing its mean."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    mask = np.zeros((3, 10, 12), dtype=bool)
    mask[0, 0, 0] = True
    images[1].data[0, 1] = np.nan
    images[2].data[0, 2] = 100
    kwargs = {"mask": mask, "thresholds": (0, 50), "memlimit": memlimit}
    plain = imcombine(images, **kwargs)
    simple = imcombine(images, full=True, **kwargs)
    detailed = imcombine(images, diagnostics="full", return_dict=True, **kwargs)
    assert isinstance(plain, CCDData)
    assert len(simple) == 8
    np.testing.assert_array_equal(detailed["comb"].data, plain.data)
    expected = np.zeros((3, 10, 12), dtype=np.uint8)
    expected[0, 0, 0] = 1
    expected[1, 0, 1] = 2
    expected[2, 0, 2] = 4
    np.testing.assert_array_equal(detailed["sample_flags"], expected)
    assert detailed["sample_flags"].dtype == np.uint8
    assert len(imcombine(images, diagnostics="simple", **kwargs)) == 8
    assert len(imcombine(images, diagnostics="full", full=True, **kwargs)) == 9
    assert mask.sum() == 1


@pytest.mark.parametrize("memlimit", [None, 2_500])
@pytest.mark.parametrize("diagnostics", [None, "simple", "full"])
@pytest.mark.parametrize("grow", [0.0, 0.9, 1.0, 1.5])
def test_growth_geometry(
    memlimit: float | None, diagnostics: str | None, grow: float
) -> None:
    """Only the outlier's plane grows; radius one excludes the diagonals."""
    images = [
        CCDData(np.full((7, 9), level, dtype=np.float32), unit="adu")
        for level in range(4)
    ]
    images[3].data[3, 4] = 1_000
    result = imcombine(
        images,
        reject="sigclip",
        sigma=1.5,
        grow=grow,
        diagnostics=diagnostics,
        return_dict=True,
        memlimit=memlimit,
    )
    yy, xx = np.ogrid[:7, :9]
    grown = (yy - 3) ** 2 + (xx - 4) ** 2 <= grow**2
    expected = np.full((7, 9), 1.5)
    expected[grown] = 1.0
    actual = result.data if diagnostics is None else result["comb"].data
    np.testing.assert_array_equal(actual, expected)
    if diagnostics is not None:
        np.testing.assert_array_equal(result["mask_rej"][3], grown)
        assert not result["mask_rej"][:3].any()
        added = grown.copy()
        added[3, 4] = False
        np.testing.assert_array_equal((result["output_flags"] & 16) != 0, added)
    if diagnostics == "full":
        flags = np.zeros((4, 7, 9), dtype=np.uint8)
        flags[3, grown] = 16
        flags[3, 3, 4] = 8
        np.testing.assert_array_equal(result["sample_flags"], flags)
    assert images[3].data[3, 4] == 1_000


@pytest.mark.parametrize("diagnostics", [None, "full"])
@pytest.mark.parametrize("memlimit", [None, 12_000])
def test_growth_preserves_all_spatial_axes(
    diagnostics: str | None, memlimit: float | None
) -> None:
    """A radius-one rejection in a volume reaches all six spatial neighbors."""
    images = [
        CCDData(np.full((5, 7, 9), level, dtype=np.float32), unit="adu")
        for level in range(4)
    ]
    images[3].data[2, 3, 4] = 1_000
    result = imcombine(
        images,
        reject="sigclip",
        sigma=1.5,
        grow=1,
        diagnostics=diagnostics,
        return_dict=True,
        memlimit=memlimit,
    )
    expected = np.full((5, 7, 9), 1.5)
    expected[1:4, 3, 4] = 1
    expected[2, 2:5, 4] = 1
    expected[2, 3, 3:6] = 1
    np.testing.assert_array_equal(
        result.data if diagnostics is None else result["comb"].data, expected
    )


@pytest.mark.parametrize("reject", ["sigclip", "ccdclip", "minmax", "pclip"])
def test_chunked_growth_matches_full_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reject: str
) -> None:
    """Partial reads include neighboring rejections across chunk boundaries."""
    paths = []
    for i in range(4):
        data = np.full((12, 14), i + 10, dtype=np.float32)
        if i == 3:
            data[5:7, 6] = 1_000
        path = tmp_path / f"image_{i}.fits"
        CCDData(data, unit="adu").write(path)
        paths.append(path)
    kwargs = {
        "reject": reject,
        "sigma": 1.5,
        "grow": 1.5,
        "diagnostics": "full",
        "return_dict": True,
        "offsets": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "weight": [1, 2, 3, 4],
    }
    full = imcombine(paths, memlimit=None, **kwargs)
    reads = []
    original = fits.PrimaryHDU._get_scaled_image_data

    def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        reads.append(shape)
        assert prod(shape) < 12 * 14
        return original(self, offset, shape)

    monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", record_read)
    chunked = imcombine(paths, memlimit=8_000, **kwargs)
    for key, expected in full.items():
        if key == "comb":
            np.testing.assert_allclose(chunked[key].data, expected.data, equal_nan=True)
        elif expected is None:
            assert chunked[key] is None
        else:
            np.testing.assert_array_equal(chunked[key], expected)
    assert len(reads) > len(paths)


@pytest.mark.parametrize("memlimit", [None, 700])
def test_sample_flags_keep_uncovered_mosaic_planes(memlimit: float | None) -> None:
    """Absent input planes and entirely uncovered chunks carry NONFINITE flags."""
    images = [CCDData(np.ones((3, 4)), unit="adu") for _ in range(2)]
    result = imcombine(
        images,
        offsets=[[0, 0], [0, 12]],
        diagnostics="full",
        return_dict=True,
        memlimit=memlimit,
    )
    flags = np.full((2, 3, 16), 2, dtype=np.uint8)
    flags[0, :, :4] = 0
    flags[1, :, 12:] = 0
    np.testing.assert_array_equal(result["sample_flags"], flags)


@pytest.mark.parametrize("memlimit", [None, 1_000])
def test_sample_flags_fits_output(tmp_path: Path, memlimit: float | None) -> None:
    """Requesting the flag file enables detailed returns and preserves FITS flags."""
    path = tmp_path / "sample_flags.fits"
    images = [CCDData(np.ones((10, 12)), unit="adu") for _ in range(2)]
    images[0].data[0, 0] = np.nan
    result = imcombine(
        images, output_sample_flags=path, checksum=True, memlimit=memlimit
    )
    assert len(result) == 9
    with fits.open(path, checksum=True) as hdul:
        np.testing.assert_array_equal(hdul[0].data, result[-1])
        assert hdul[0].data.dtype == np.uint8
        assert hdul[0].verify_checksum() == 1
    with pytest.raises(OSError):
        imcombine(images, output_sample_flags=path, memlimit=memlimit)
    imcombine(images, output_sample_flags=path, overwrite=True, memlimit=memlimit)


@pytest.mark.parametrize("grow", [-1, np.nan, np.inf])
def test_invalid_growth_fails_before_input_io(grow: float) -> None:
    with pytest.raises(ValueError, match="grow"):
        imcombine(["missing.fits"], grow=grow)


def test_invalid_diagnostics_fails_before_input_io() -> None:
    with pytest.raises(ValueError, match="diagnostics"):
        imcombine(["missing.fits"], diagnostics="all")


def test_detailed_flags_reserve_persistent_memory() -> None:
    """A flag volume cannot be made smaller by spatial chunking."""
    images = [CCDData(np.ones((10, 12)), unit="adu") for _ in range(3)]
    # Simple: 480-byte image + 360-byte total mask. Detailed adds 360 bytes.
    imcombine(images, full=True, memlimit=1_000)
    with pytest.raises(ValueError, match="persistent"):
        imcombine(images, diagnostics="full", memlimit=1_000)


def test_growth_halos_fit_chunk_memory_estimate() -> None:
    """Budget every expanded input region, not just the pixels stitched back."""
    _, count, chunks = check_stack_memory(
        4, (7, 9), "float32", "average", 1_020, halo=1
    )
    assert count > 1
    coverage = np.zeros((7, 9), dtype=int)
    for chunk in chunks:
        coverage[chunk] += 1
        expanded_pixels = prod(
            min(size, sl.stop + 1) - max(0, sl.start - 1)
            for sl, size in zip(chunk, (7, 9), strict=True)
        )
        assert 7 * 9 * 4 + expanded_pixels * 4 * (4 * 3 + 1) <= 1_020
    np.testing.assert_array_equal(coverage, 1)
    # An interior core pixel needs 3x3 samples: 252 + 9*4*(12+1) = 720 bytes.
    with pytest.raises(ValueError, match="chunk"):
        check_stack_memory(4, (7, 9), "float32", "average", 719, halo=1)


@pytest.mark.parametrize("memlimit", [None, 2_700])
@pytest.mark.parametrize("excluded", ["mask", "threshold", "nan"])
def test_growth_does_not_spread_preexisting_exclusions(
    memlimit: float | None, excluded: str
) -> None:
    """Already unavailable samples are not seeds for algorithmic growth."""
    images = [
        CCDData(np.full((7, 9), level, dtype=np.float32), unit="adu")
        for level in range(4)
    ]
    mask = np.zeros((4, 7, 9), dtype=bool)
    if excluded == "mask":
        mask[3, 3, 4] = True
    elif excluded == "threshold":
        images[3].data[3, 4] = 1_000
    else:
        images[3].data[3, 4] = np.nan
    result = imcombine(
        images,
        mask=mask,
        thresholds=(0, 10),
        reject="sigclip",
        sigma=1.5,
        grow=1,
        diagnostics="full",
        return_dict=True,
        memlimit=memlimit,
    )
    expected = np.full((7, 9), 1.5)
    expected[3, 4] = 1
    np.testing.assert_array_equal(result["comb"].data, expected)
    assert not (result["sample_flags"] & 16).any()


@pytest.mark.parametrize("memlimit", [None, 2_500])
@pytest.mark.parametrize(("limits", "flag"), [({"nkeep": 4}, 64), ({"maxrej": 0}, 128)])
def test_restored_samples_are_flagged_but_not_grown(
    memlimit: float | None, limits: dict[str, int], flag: int
) -> None:
    """Restored outliers stay in the mean and do not reject their neighbors."""
    images = [
        CCDData(np.full((7, 9), level, dtype=np.float32), unit="adu")
        for level in range(4)
    ]
    images[3].data[3, 4] = 1_000
    result = imcombine(
        images,
        reject="sigclip",
        sigma=1.5,
        grow=1,
        diagnostics="full",
        return_dict=True,
        memlimit=memlimit,
        **limits,
    )
    expected = np.full((7, 9), 1.5)
    expected[3, 4] = 250.75
    np.testing.assert_array_equal(result["comb"].data, expected)
    flags = np.zeros((4, 7, 9), dtype=np.uint8)
    flags[3, 3, 4] = flag
    np.testing.assert_array_equal(result["sample_flags"], flags)


@pytest.mark.parametrize("memlimit", [None, 2_400])
def test_weights_are_normalized_only_after_growth_on_the_chunk_core(
    memlimit: float | None,
) -> None:
    """Growth removes a cancelling weight; incomplete halos must not be reduced."""
    images = [
        CCDData(np.full((7, 9), level, dtype=np.float32), unit="adu")
        for level in range(4)
    ]
    images[2].data[3, 5] = 1_000
    mask = np.zeros((4, 7, 9), dtype=bool)
    mask[[1, 3], 3, 4] = True
    result = imcombine(
        images,
        mask=mask,
        weight=[1, 1, -1, 1],
        reject="sigclip",
        sigma=1.5,
        grow=1,
        diagnostics="full",
        return_dict=True,
        memlimit=memlimit,
    )
    expected = np.ones((7, 9))
    expected[2:5, 5] = 4 / 3
    expected[3, 4:7] = 4 / 3
    expected[3, 4] = 0
    np.testing.assert_allclose(result["comb"].data, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16"])
@pytest.mark.parametrize("memlimit", [None, 6_000])
def test_growth_preserves_calibration_and_output_dtype(
    dtype: str, memlimit: float | None
) -> None:
    """Both passes use the same fractional normalization, exactly once each."""
    images = []
    for level, zero, scale in zip(
        range(4), [0.5, 1.5, 2.5, 3.5], [1, 2, 4, 8], strict=True
    ):
        data = np.full((7, 9), level, dtype=np.float32)
        if level == 3:
            data[3, 4] = 1_000
        images.append(CCDData(data * scale + zero, unit="adu"))
    result = imcombine(
        images,
        zero=[0.5, 1.5, 2.5, 3.5],
        scale=[1, 2, 4, 8],
        zero_to_0th=False,
        scale_to_0th=False,
        weight=[1, 1, 1, 3],
        reject="sigclip",
        sigma=1.5,
        grow=1,
        diagnostics="full",
        return_dict=True,
        dtype=dtype,
        memlimit=memlimit,
    )
    expected = np.full((7, 9), 2, dtype=dtype)
    expected[2:5, 4] = 1
    expected[3, 3:6] = 1
    np.testing.assert_array_equal(result["comb"].data, expected)
    assert result["comb"].data.dtype == np.dtype(dtype)


def test_growth_without_rejection_is_inactive() -> None:
    """A radius alone must not enable rejection or grow an input mask."""
    images = [CCDData(np.ones((3, 4)), unit="adu") for _ in range(2)]
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 1, 1] = True
    result = imcombine(images, mask=mask, grow=100, memlimit=80)
    np.testing.assert_array_equal(result.data, 1)
