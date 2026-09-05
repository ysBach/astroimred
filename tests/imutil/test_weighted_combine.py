"""Weighted FITS combination agrees with per-pixel arithmetic."""

from pathlib import Path

import numpy as np
import pytest
from astropy.nddata import CCDData

from astroimred.imutil.imcombine import imcombine


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize(
    "combine", ["average", "mean", "avg", "weighted_average", "wvg"]
)
def test_weighted_mean_with_normalization(
    tmp_path: Path, memlimit: float | None, combine: str
) -> None:
    """Weights apply to calibrated values, with all image weights retained."""
    base = np.arange(120, dtype=np.float32).reshape(10, 12)
    paths = []
    for i, (level, scale, zero) in enumerate(
        zip([10, 20, 30], [1, 2, 4], [3, 7, 11], strict=True)
    ):
        path = tmp_path / f"frame_{i}.fits"
        CCDData((base + level) * scale + zero, unit="adu").write(path)
        paths.append(path)
    weights = np.array([1.0, 2.0, 5.0])
    weights.setflags(write=False)
    result = imcombine(
        paths,
        weight=weights,
        combine=combine,
        zero=[3, 7, 11],
        scale=[1, 2, 4],
        zero_to_0th=False,
        scale_to_0th=False,
        memlimit=memlimit,
    )
    # (10*1 + 20*2 + 30*5) / 8 = 25.
    np.testing.assert_allclose(result.data, base + 25)
    assert result.data.dtype == np.dtype("float32")
    for i, weight in enumerate(weights, start=1):
        assert result.header[f"WEIGH{i:03d}"] == weight
    np.testing.assert_array_equal(weights, [1, 2, 5])


@pytest.mark.parametrize("memlimit", [None, 1_800])
def test_weighted_mean_renormalizes_valid_samples(memlimit: float | None) -> None:
    """Input masks, thresholds, and NaNs exclude both values and their weights."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    mask = np.zeros((3, 10, 12), dtype=bool)
    mask[2, 0, 0] = True
    images[2].data[0, 1] = np.nan
    images[2].data[0, 2] = 100
    mask[:, 0, 3] = True
    images[2].data[0, 4] = np.inf
    result = imcombine(
        images,
        weight=[1, 2, 5],
        mask=mask,
        thresholds=(0, 50),
        memlimit=memlimit,
        full=True,
        return_dict=True,
    )
    expected = np.full((10, 12), 25.0)
    expected[0, :3] = 50 / 3
    expected[0, 3] = np.nan
    expected[0, 4] = 50 / 3
    np.testing.assert_allclose(result["comb"].data, expected, equal_nan=True)
    assert result["mask_total"][2, 0, 0]
    assert result["mask_thresh"][2, 0, 2]
    assert mask.sum() == 4
    assert images[2].data[0, 2] == 100


@pytest.mark.parametrize("memlimit", [None, 3_400])
@pytest.mark.parametrize(("reject", "expected"), [("sigclip", 80 / 6), ("minmax", 16)])
def test_weighted_mean_after_rejection(
    memlimit: float | None, reject: str, expected: float
) -> None:
    """A large weight cannot restore a rejected outlier."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (0, 10, 20, 1_000)
    ]
    result = imcombine(
        images,
        weight=[1, 2, 3, 100],
        reject=reject,
        sigma=(1, 1),
        full=True,
        return_dict=True,
        memlimit=memlimit,
    )
    np.testing.assert_allclose(result["comb"].data, expected)
    assert result["mask_rej"][3].all()


@pytest.mark.parametrize("memlimit", [None, 300])
def test_sparse_chunks_use_original_image_weights(memlimit: float | None) -> None:
    """A compact chunk uses the original weights of its contributing images."""
    images = [
        CCDData(np.full((3, 4), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    result = imcombine(
        images, weight=[1, 3, 9], offsets=[[0, 0], [0, 2], [0, 12]], memlimit=memlimit
    )
    expected = np.full((3, 16), np.nan)
    expected[:, :2] = 10
    expected[:, 2:4] = 17.5
    expected[:, 4:6] = 20
    expected[:, 12:] = 30
    np.testing.assert_allclose(result.data, expected, equal_nan=True)


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize(("weight", "expected"), [(2.0, 20.0), ("mean", 140 / 6)])
def test_scalar_and_statistical_weights(
    memlimit: float | None, weight: float | str, expected: float
) -> None:
    """Scalar weights are shared; named weights are resolved once per image."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    result = imcombine(images, weight=weight, memlimit=memlimit)
    np.testing.assert_allclose(result.data, expected)


@pytest.mark.parametrize("combine", ["median", "sum", "min", "variance"])
def test_weights_require_mean_combination(combine: str) -> None:
    """An incompatible combination method fails before FITS input I/O."""
    with pytest.raises(ValueError, match="weight.*mean"):
        imcombine(["missing.fits"], combine=combine, weight=[1])


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize("weight", [[1, 2], [1, np.nan, 3], [1, np.inf, 3], [1, 0, 3]])
def test_invalid_weights_raise(memlimit: float | None, weight: list[float]) -> None:
    """Weight length, finiteness, and the existing nonzero contract are enforced."""
    images = [
        CCDData(np.ones((10, 12), dtype=np.float32), unit="adu") for _ in range(3)
    ]
    with pytest.raises(ValueError, match="weight"):
        imcombine(images, weight=weight, memlimit=memlimit)


@pytest.mark.parametrize("memlimit", [None, 1_000])
@pytest.mark.parametrize(
    ("weight", "expected"), [([-1, 2, 3], 30.0), ([-1, -2, -5], 25.0)]
)
def test_signed_weights_with_single_valid_sample(
    memlimit: float | None, weight: list[float], expected: float
) -> None:
    """Signed weights retain their signs and cancel for a lone valid sample."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    mask = np.zeros((3, 10, 12), dtype=bool)
    mask[1:, 0, 0] = True
    result = imcombine(images, weight=weight, mask=mask, memlimit=memlimit)
    assert result.data[0, 0] == 10
    np.testing.assert_allclose(result.data[1:], expected)


@pytest.mark.parametrize("memlimit", [None, 1_000])
def test_valid_sample_weights_cancelling_to_zero_raise(memlimit: float | None) -> None:
    """A mask can leave a zero denominator even when the full vector sums to one."""
    images = [
        CCDData(np.full((10, 12), level, dtype=np.float32), unit="adu")
        for level in (10, 20, 30)
    ]
    mask = np.zeros((3, 10, 12), dtype=bool)
    mask[2, 0, 0] = True
    with pytest.raises(ZeroDivisionError):
        imcombine(images, weight=[1, -1, 1], mask=mask, memlimit=memlimit)
