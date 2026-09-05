"""Pandas summary exports preserve selection and output-format behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from astroimred.fitsmgmt.table import fits_summary


@pytest.fixture
def summary_inputs(tmp_path: Path) -> list[Path]:
    paths = []
    for index, row in enumerate([{"COUNT": 7}, {"COUNT": 8}, {}]):
        path = tmp_path / f"image{index}.fits"
        fits.writeto(path, np.zeros((2, 3)), header=fits.Header(row))
        paths.append(path)
    return paths


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
    [("csv", ".parquet"), ("parquet", ".csv")],
)
def test_explicit_format_overrides_suffix_and_replaces_existing_output(
    summary_inputs: list[Path], tmp_path: Path, output_format: str, suffix: str
) -> None:
    output = tmp_path / f"summary{suffix}"
    output.write_text("previous summary")
    expected = fits_summary(
        summary_inputs[:1], keywords="COUNT", output=output, output_format=output_format
    )
    read = pd.read_parquet if output_format == "parquet" else pd.read_csv
    pd.testing.assert_frame_equal(read(output), expected)


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_export_uses_filtered_sorted_rows_and_nonunique_columns(
    summary_inputs: list[Path], tmp_path: Path, output_format: str
) -> None:
    output = tmp_path / f"selected.{output_format}"
    result = fits_summary(
        summary_inputs,
        keywords=["COUNT"],
        querystr="COUNT >= 7",
        sort_by="file",
        sort_map={str(summary_inputs[0]): 1, str(summary_inputs[1]): 0},
        nonunique_keys=True,
        output=output,
        output_format=output_format,
    )
    read = pd.read_parquet if output_format == "parquet" else pd.read_csv
    restored = read(output)
    assert list(restored.columns) == ["file", "COUNT"]
    assert list(restored.columns) == list(result.columns)
    assert restored["COUNT"].tolist() == [8, 7]
    assert restored["file"].tolist() == [str(summary_inputs[1]), str(summary_inputs[0])]


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_export_supports_empty_selection(
    summary_inputs: list[Path], tmp_path: Path, output_format: str
) -> None:
    output = tmp_path / f"empty.{output_format}"
    result = fits_summary(
        summary_inputs,
        keywords=["COUNT"],
        querystr="COUNT < 0",
        output=output,
        output_format=output_format,
    )
    read = pd.read_parquet if output_format == "parquet" else pd.read_csv
    restored = read(output)
    assert len(restored) == 0
    assert list(restored.columns) == list(result.columns)


def test_unknown_output_format_fails_before_input_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_format"):
        fits_summary(
            [tmp_path / "missing.fits"],
            output=tmp_path / "summary.csv",
            output_format="unknown",
        )


def test_missing_header_does_not_round_large_integers(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate([2**53 + 1, None, 2**53 + 3]):
        header = fits.Header()
        if value is not None:
            header["BIGINT"] = value
        path = tmp_path / f"large_{index}.fits"
        fits.writeto(path, np.zeros((2, 2)), header=header)
        paths.append(path)
    result = fits_summary(paths, keywords="BIGINT")
    assert result["BIGINT"].tolist() == [2**53 + 1, None, 2**53 + 3]
