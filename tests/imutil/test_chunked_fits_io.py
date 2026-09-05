"""Exercise automatic chunking through real FITS section reads."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData

from astroimred.imutil.imcombine import imcombine


@pytest.mark.parametrize("metadata", [{"scale": "exposure"}, {"imcmb_key": "OBJECT"}])
def test_metadata_does_not_load_whole_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, str]
) -> None:
    """Header-based normalization and provenance must not bypass chunking."""
    paths = []
    sky = np.arange(120, dtype=np.float32).reshape(10, 12)
    for i, exposure in enumerate((1.0, 2.0, 3.0)):
        path = tmp_path / f"image_{i}.fits"
        CCDData(
            sky * exposure,
            unit="adu",
            header={"EXPTIME": exposure, "OBJECT": "test target"},
        ).write(path)
        paths.append(path)

    full = imcombine(paths, memlimit=None, verbose=False, **metadata)
    reads = []
    original = fits.PrimaryHDU._get_scaled_image_data

    def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        reads.append(shape)
        assert tuple(shape) != sky.shape, "unexpected full-image read"
        return original(self, offset, shape)

    monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", record_read)
    chunked = imcombine(paths, memlimit=1_000, verbose=False, **metadata)

    np.testing.assert_allclose(chunked.data, full.data)
    assert len(reads) > len(paths)


def test_chunked_extension_images_preserve_trim_and_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Named image/mask extensions share the same trimmed section coordinates."""
    paths = []
    base = np.arange(120, dtype=np.float32).reshape(10, 12)
    for i in range(3):
        mask = np.zeros(base.shape, dtype=np.uint8)
        mask[3, 4] = i == 2
        path = tmp_path / f"extensions_{i}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(base + 10 * i, name="SCI"),
                fits.ImageHDU(mask, name="MASK"),
            ]
        ).writeto(path)
        paths.append(path)
    common = {
        "extension": "SCI",
        "extension_mask": "MASK",
        "trimsec": "[2:10,2:9]",
        "full": True,
        "return_dict": True,
        "verbose": False,
    }
    full = imcombine(paths, memlimit=None, **common)
    reads = []
    original = fits.ImageHDU._get_scaled_image_data

    def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        reads.append(shape)
        # Count pixels, since even a flattened full-image read would defeat chunking.
        assert np.prod(shape) < 8 * 9
        return original(self, offset, shape)

    monkeypatch.setattr(fits.ImageHDU, "_get_scaled_image_data", record_read)
    chunked = imcombine(paths, memlimit=900, **common)

    np.testing.assert_allclose(chunked["comb"].data, full["comb"].data)
    np.testing.assert_array_equal(chunked["mask_total"], full["mask_total"])
    assert chunked["comb"].data[2, 3] == base[3, 4] + 5
    assert len(reads) > 2 * len(paths)


@pytest.mark.parametrize("encoding", ["unsigned", "scaled", "blank"])
def test_chunked_scaled_fits_matches_whole_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    """FITS scaling and blank-value conversion apply to sections, not full images."""
    paths = []
    for i in range(3):
        data = np.arange(120, dtype=np.int16).reshape(10, 12) + i
        if encoding == "unsigned":
            hdu = fits.PrimaryHDU(data.astype(np.uint16) + 40_000)
        else:
            hdu = fits.PrimaryHDU(data)
            if encoding == "scaled":
                hdu.header["BSCALE"] = 2.5
                hdu.header["BZERO"] = 10.0
            else:
                hdu.header["BLANK"] = -99
                hdu.data[2, 3] = -99
        hdu.header["BUNIT"] = "adu"
        path = tmp_path / f"{encoding}_{i}.fits"
        hdu.writeto(path)
        paths.append(path)

    full = imcombine(paths, memlimit=None, verbose=False)
    if encoding == "blank":
        assert np.isnan(full.data[2, 3])
    reads = []
    original = fits.PrimaryHDU._get_scaled_image_data

    def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        reads.append(shape)
        assert tuple(shape) != (10, 12), "unexpected full-image read"
        return original(self, offset, shape)

    monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", record_read)
    chunked = imcombine(paths, memlimit=1_000, verbose=False)

    np.testing.assert_allclose(chunked.data, full.data, equal_nan=True)
    assert len(reads) > len(paths)


@pytest.mark.parametrize(
    ("combine", "reject"),
    [("average", None), ("median", None), ("average", "minmax"), ("median", "sigclip")],
)
def test_partial_fits_reads_preserve_combination_and_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, combine: str, reject: str | None
) -> None:
    """Each spatial chunk combines all input samples before writing its result."""
    paths = []
    base = np.arange(120, dtype=np.float32).reshape(10, 12)
    for i, level in enumerate((0, 10, 20, 1_000)):
        path = tmp_path / f"image_{i}.fits"
        CCDData(base + level, unit="adu").write(path)
        paths.append(path)
    common = {
        "combine": combine,
        "reject": reject,
        "sigma": (1.0, 1.0),
        "full": True,
        "return_dict": True,
        "verbose": False,
    }
    full = imcombine(paths, memlimit=None, **common)
    reads = []
    original = fits.PrimaryHDU._get_scaled_image_data

    def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        reads.append(shape)
        assert tuple(shape) != base.shape, "unexpected full-image read"
        return original(self, offset, shape)

    monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", record_read)
    chunked = imcombine(paths, memlimit=4_000, **common)

    np.testing.assert_allclose(chunked["comb"].data, full["comb"].data, equal_nan=True)
    for key in (
        "mask_total",
        "mask_rej",
        "mask_thresh",
        "std",
        "low",
        "upp",
        "nit",
        "output_flags",
    ):
        if full[key] is None:
            assert chunked[key] is None
        else:
            np.testing.assert_allclose(chunked[key], full[key], equal_nan=True)
    assert len(reads) > len(paths)


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize("per_image", [False, True])
def test_numeric_normalization_is_independent_of_memory_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memlimit: float | None,
    per_image: bool,
) -> None:
    """Scalar and per-image calibration must produce the same calibrated sky."""
    sky = np.arange(120, dtype=np.float32).reshape(10, 12)
    zeros = [3.0, 7.0, 11.0] if per_image else 3.0
    scales = [1.0, 2.0, 4.0] if per_image else 2.0
    paths = []
    for i in range(3):
        zero_i = zeros[i] if per_image else zeros
        scale_i = scales[i] if per_image else scales
        path = tmp_path / f"calibrated_{i}.fits"
        CCDData(sky * scale_i + zero_i, unit="adu").write(path)
        paths.append(path)
    if memlimit is not None:
        original = fits.PrimaryHDU._get_scaled_image_data

        def record_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
            assert tuple(shape) != sky.shape, "numeric calibration read whole image"
            return original(self, offset, shape)

        monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", record_read)

    result = imcombine(
        paths,
        zero=zeros,
        scale=scales,
        zero_to_0th=False,
        scale_to_0th=False,
        memlimit=memlimit,
    )
    np.testing.assert_allclose(result.data, sky)


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize("stale_shape", [False, True])
def test_ccddata_dimensions_come_from_data(
    memlimit: float | None, stale_shape: bool
) -> None:
    """CCDData needs no FITS dimensions; stale header dimensions cannot crop it."""
    sky = np.arange(120, dtype=np.float32).reshape(10, 12)
    header = {"NAXIS": 1, "NAXIS1": 3} if stale_shape else {}
    images = [CCDData(sky + 10 * i, unit="adu", header=header.copy()) for i in range(3)]
    result = imcombine(images, memlimit=memlimit)
    np.testing.assert_array_equal(result.data, sky + 10)
    assert dict(images[0].header) == header


@pytest.mark.parametrize("memlimit", [None, 100])
@pytest.mark.parametrize("zero", [0.5, "mean"])
def test_integer_inputs_keep_fractional_normalization(
    tmp_path: Path, memlimit: float | None, zero: float | str
) -> None:
    """Integer FITS storage must not round calibration values or statistics."""
    base = np.arange(12, dtype=np.int16).reshape(3, 4)
    paths = []
    for i in range(3):
        path = tmp_path / f"integer_{i}.fits"
        CCDData(base + i, unit="adu").write(path)
        paths.append(path)
    result = imcombine(
        paths,
        zero=zero,
        scale=1.5,
        zero_to_0th=False,
        scale_to_0th=False,
        memlimit=memlimit,
    )
    expected = (base - 5.5) / 1.5 if zero == "mean" else (base + 0.5) / 1.5
    np.testing.assert_allclose(result.data, expected, rtol=1e-6)


@pytest.mark.parametrize("reject", [None, "sigclip", "ccdclip", "minmax", "pclip"])
def test_uncovered_mosaic_diagnostics_match_full_stack(
    tmp_path: Path, reject: str | None
) -> None:
    """Uncovered samples must have the same rejection flags at any chunk size."""
    paths = []
    for i in range(4):
        path = tmp_path / f"mosaic_{i}.fits"
        CCDData(np.full((3, 4), i + 1, dtype=np.float32), unit="adu").write(path)
        paths.append(path)
    common = {
        "offsets": [[0, 0], [0, 1], [0, 20], [0, 21]],
        "reject": reject,
        "full": True,
        "return_dict": True,
    }
    full = imcombine(paths, memlimit=None, **common)
    # 26 retained bytes per output pixel covers all sigma-clipping diagnostics.
    chunked = imcombine(paths, memlimit=3 * 25 * 26 + 100, **common)
    for key in full:
        if key == "comb":
            np.testing.assert_allclose(
                chunked[key].data, full[key].data, equal_nan=True
            )
        elif full[key] is None:
            assert chunked[key] is None
        else:
            np.testing.assert_allclose(chunked[key], full[key], equal_nan=True)


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint16"])
def test_uncovered_integer_mosaic_matches_full_stack(
    tmp_path: Path, dtype: str
) -> None:
    """An entirely uncovered chunk uses the full path's final NaN conversion."""
    paths = []
    for i in range(2):
        path = tmp_path / f"gap_{i}.fits"
        CCDData(np.full((3, 4), i + 1, dtype=np.float32), unit="adu").write(path)
        paths.append(path)
    common = {"offsets": [[0, 0], [0, 20]], "dtype": dtype}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = imcombine(paths, memlimit=None, **common)
        chunked = imcombine(
            paths, memlimit=3 * 24 * np.dtype(dtype).itemsize + 100, **common
        )
    np.testing.assert_array_equal(chunked.data, full.data)


def test_uncovered_sum_mosaic_uses_zero(tmp_path: Path) -> None:
    """Summing zero contributing samples returns the additive identity."""
    paths = []
    for i in range(2):
        path = tmp_path / f"sum_{i}.fits"
        CCDData(np.full((3, 4), i + 1, dtype=np.float32), unit="adu").write(path)
        paths.append(path)
    result = imcombine(paths, offsets=[[0, 0], [0, 20]], combine="sum", memlimit=400)
    np.testing.assert_array_equal(result.data[:, :4], 1)
    np.testing.assert_array_equal(result.data[:, 4:20], 0)
    np.testing.assert_array_equal(result.data[:, 20:], 2)


@pytest.mark.parametrize("memlimit", [None, 160])
@pytest.mark.parametrize("diagnostics", [None, "full"])
@pytest.mark.parametrize(
    ("calibration", "expected"),
    [({"zero": 3.0}, 7.0), ({"scale": 2.0}, 5.0), ({"zero": 3.0, "scale": 2.0}, 3.5)],
)
def test_single_image_keeps_scalar_calibration(
    memlimit: float | None, diagnostics: str | None, calibration: dict, expected: float
) -> None:
    """A single image's scalar calibration is not rebased to zero and one."""
    image = CCDData(np.full((4, 5), 10, dtype=np.float32), unit="adu")
    result = imcombine(
        [image],
        memlimit=memlimit,
        diagnostics=diagnostics,
        return_dict=True,
        **calibration,
    )
    combined = result if diagnostics is None else result["comb"]
    np.testing.assert_array_equal(combined.data, expected)


@pytest.mark.parametrize("memlimit", [None, 250])
@pytest.mark.parametrize("diagnostics", [None, "full"])
def test_zero_offsets_rebase_before_float32_rounding(
    memlimit: float | None, diagnostics: str | None
) -> None:
    """The reference cancels before calibration is cast to the pixel dtype."""
    images = [
        CCDData(np.full((4, 5), 10, dtype=np.float32), unit="adu") for _ in range(3)
    ]
    result = imcombine(
        images,
        zero=[100_000_000, 100_000_001, 100_000_002],
        weight=[1, 2, 1],
        memlimit=memlimit,
        diagnostics=diagnostics,
        return_dict=True,
    )
    combined = result if diagnostics is None else result["comb"]
    # Relative zeros are [0, 1, 2]: (10 + 2*9 + 8) / 4 = 9.
    np.testing.assert_array_equal(combined.data, 9)


@pytest.mark.parametrize("memlimit", [None, 250])
def test_scale_ratios_rebase_before_float32_conversion(memlimit: float | None) -> None:
    """Representable scale ratios do not require representable raw scales."""
    images = [
        CCDData(np.full((4, 5), value, dtype=np.float32), unit="adu")
        for value in [10, 20, 40]
    ]
    result = imcombine(images, scale=[1e40, 2e40, 4e40], memlimit=memlimit)
    np.testing.assert_array_equal(result.data, 10)


@pytest.mark.parametrize("from_fits", [False, True])
@pytest.mark.parametrize("memlimit", [None, 550])
@pytest.mark.parametrize(
    "trimsec",
    [
        (slice(None, None, -1), slice(None, None, -2)),
        (slice(8, 1, -2), slice(10, 2, -2)),
    ],
)
def test_reversed_trim_preserves_data_and_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    from_fits: bool,
    memlimit: float | None,
    trimsec: tuple[slice, ...],
) -> None:
    """Negative-step slices use the same pixels and masks at any budget."""
    base = np.arange(120, dtype=np.float32).reshape(10, 12)
    inputs = []
    for i in range(3):
        mask = np.zeros(base.shape, dtype=bool)
        mask[8, 10] = i == 2
        ccd = CCDData(base + 10 * i, mask=mask, unit="adu")
        if from_fits:
            path = tmp_path / f"reverse_{i}.fits"
            ccd.write(path)
            inputs.append(path)
        else:
            inputs.append(ccd)
    if from_fits and memlimit is not None:
        original = fits.PrimaryHDU._get_scaled_image_data

        def partial_read(self, offset: int, shape: tuple[int, ...]) -> np.ndarray:
            assert np.prod(shape) < base.size
            return original(self, offset, shape)

        monkeypatch.setattr(fits.PrimaryHDU, "_get_scaled_image_data", partial_read)
    result = imcombine(
        inputs,
        trimsec=trimsec,
        extension_mask="MASK" if from_fits else None,
        memlimit=memlimit,
        full=True,
        return_dict=True,
    )
    expected = base + 10
    expected[8, 10] = base[8, 10] + 5
    np.testing.assert_array_equal(result["comb"].data, expected[trimsec])
    expected_mask = np.zeros((3, *base.shape), dtype=bool)
    expected_mask[2, 8, 10] = True
    np.testing.assert_array_equal(
        result["mask_total"], expected_mask[(slice(None), *trimsec)]
    )
