"""Summary exports contain the selected rows, values, and missing-value masks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

from astroimred.fitsmgmt.table import fits_summary


@pytest.fixture
def summary_inputs(tmp_path: Path) -> list[Path]:
    rows = [
        {"OBJECT": "M1", "EXPTIME": 10.5, "COUNT": 7, "GOOD": True},
        {"OBJECT": "", "EXPTIME": 20.5, "COUNT": 8, "GOOD": False},
        {},
    ]
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f"image{index}.fits"
        fits.writeto(path, np.zeros((2, 3)), header=fits.Header(row))
        paths.append(path)
    return paths


def test_fits_binary_table_preserves_values_and_masks(
    summary_inputs: list[Path], tmp_path: Path
) -> None:
    output = tmp_path / "summary.fits"
    keywords = ["OBJECT", "EXPTIME", "COUNT", "GOOD", "ABSENT"]
    expected = fits_summary(summary_inputs, keywords=keywords)
    actual = fits_summary(
        summary_inputs, keywords=keywords, output=output, output_format="fits"
    )

    # Export must leave the returned summary's values and dtypes unchanged.
    pd.testing.assert_frame_equal(actual, expected)
    with fits.open(output) as hdul:
        assert isinstance(hdul[1], fits.BinTableHDU)
        assert len(hdul[1].data) == 3
    restored = Table.read(output)
    assert restored.colnames == list(actual.columns)
    assert list(restored["file"]) == [str(path) for path in summary_inputs]
    np.testing.assert_array_equal(
        restored["filesize"], [path.stat().st_size for path in summary_inputs]
    )
    assert list(restored["OBJECT"][:2]) == ["M1", ""]
    np.testing.assert_array_equal(restored["EXPTIME"][:2], [10.5, 20.5])
    np.testing.assert_array_equal(restored["COUNT"][:2], [7, 8])
    np.testing.assert_array_equal(restored["GOOD"][:2], [True, False])
    assert restored["GOOD"].dtype.kind == "b"
    for key in keywords[:-1]:
        np.testing.assert_array_equal(restored[key].mask, [False, False, True])
    np.testing.assert_array_equal(restored["ABSENT"].mask, [True, True, True])


@pytest.mark.parametrize("suffix", [".csv", ".fits", ".parq", ".parquet"])
def test_default_format_keeps_suffix_behavior(
    summary_inputs: list[Path], tmp_path: Path, suffix: str
) -> None:
    output = tmp_path / f"summary{suffix}"
    expected = fits_summary(summary_inputs[:1], keywords="COUNT", output=output)
    read = pd.read_parquet if suffix in {".parq", ".parquet"} else pd.read_csv
    pd.testing.assert_frame_equal(read(output), expected)


@pytest.mark.parametrize(
    ("output_format", "suffix"),
    [("csv", ".parquet"), ("parquet", ".csv"), ("fits", ".csv")],
)
def test_explicit_format_overrides_suffix_and_replaces_existing_output(
    summary_inputs: list[Path], tmp_path: Path, output_format: str, suffix: str
) -> None:
    output = tmp_path / f"summary{suffix}"
    output.write_text("previous summary")
    expected = fits_summary(
        summary_inputs[:1], keywords="COUNT", output=output, output_format=output_format
    )
    if output_format == "fits":
        restored = Table.read(output, format="fits")
        assert list(restored["COUNT"]) == [7]
        assert restored.colnames == list(expected.columns)
    else:
        read = pd.read_parquet if output_format == "parquet" else pd.read_csv
        pd.testing.assert_frame_equal(read(output), expected)


def test_fits_export_uses_filtered_sorted_rows_and_nonunique_columns(
    summary_inputs: list[Path], tmp_path: Path
) -> None:
    output = tmp_path / "selected.fits"
    result = fits_summary(
        summary_inputs,
        keywords=["COUNT"],
        querystr="COUNT >= 7",
        sort_by="file",
        sort_map={str(summary_inputs[0]): 1, str(summary_inputs[1]): 0},
        nonunique_keys=True,
        output=output,
        output_format="fits",
    )
    restored = Table.read(output)
    assert restored.colnames == ["file", "COUNT"]
    assert restored.colnames == list(result.columns)
    assert list(restored["COUNT"]) == [8, 7]


def test_fits_export_supports_empty_selection(
    summary_inputs: list[Path], tmp_path: Path
) -> None:
    output = tmp_path / "empty.fits"
    result = fits_summary(
        summary_inputs,
        keywords=["COUNT", "OBJECT"],
        querystr="COUNT < 0",
        output=output,
        output_format="fits",
    )
    restored = Table.read(output)
    assert len(restored) == 0
    assert restored.colnames == list(result.columns)


def test_unknown_output_format_fails_before_input_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_format"):
        fits_summary(
            [tmp_path / "missing.fits"],
            output=tmp_path / "summary.csv",
            output_format="unknown",
        )


def test_fits_export_rejects_mixed_text_numeric_column_without_overwriting(
    tmp_path: Path,
) -> None:
    paths = []
    for index, value in enumerate([1.5, "unknown"]):
        path = tmp_path / f"input{index}.fits"
        fits.writeto(path, np.zeros((2, 2)), header=fits.Header({"GAIN": value}))
        paths.append(path)
    output = tmp_path / "summary.fits"
    output.write_bytes(b"previous summary")

    with pytest.raises(TypeError, match="GAIN"):
        fits_summary(paths, keywords="GAIN", output=output, output_format="fits")
    assert output.read_bytes() == b"previous summary"
