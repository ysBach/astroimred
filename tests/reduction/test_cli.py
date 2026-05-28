import numpy as np
from astropy.io import fits
from click.testing import CliRunner

from astroimred.reduction.cli import arith_command, comb_command, copy_command, main


def _write_fits(path, data):
    hdr = fits.Header()
    hdr["BUNIT"] = "adu"
    fits.PrimaryHDU(data=np.asarray(data, dtype="float32"), header=hdr).writeto(path)


def test_comb_command_help():
    runner = CliRunner()

    result = runner.invoke(comb_command, ["--help"])

    assert result.exit_code == 0
    assert "comb [OPTIONS] input output" in result.output
    assert "astroimred.reduction" in result.output
    assert "--combine" in result.output
    assert "--reject" in result.output
    assert "--logfile" not in result.output


def test_main_help_uses_astroimred_reduction_and_imred_subcommands():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"], prog_name="imred")

    assert result.exit_code == 0
    assert "Usage: imred [OPTIONS] COMMAND" in result.output
    assert (
        "astroimred.reduction FITS image-in/image-out reduction tools" in result.output
    )
    assert "comb" in result.output
    assert "copy" in result.output
    assert "arith" in result.output
    assert "fitscombine" not in result.output
    assert "fitscopy" not in result.output
    assert "fitsarith" not in result.output
    assert "imcombine" not in result.output


def test_comb_group_help_shows_imred_usage():
    runner = CliRunner()

    result = runner.invoke(main, ["comb", "--help"], prog_name="imred")

    assert result.exit_code == 0
    assert "Usage: imred comb [OPTIONS] input output" in result.output
    assert "imred comb INPUT OUTPUT" in result.output.replace("\n  ", " ")


def test_comb_command_average_comma_inputs(tmp_path):
    paths = []
    for i, value in enumerate([1.0, 2.0, 3.0]):
        path = tmp_path / f"avg_{i}.fits"
        _write_fits(path, np.full((3, 4), value))
        paths.append(path)

    output = tmp_path / "avg_out.fits"
    runner = CliRunner()
    result = runner.invoke(
        comb_command,
        [
            ",".join(str(path) for path in paths),
            str(output),
            "--combine",
            "average",
            "--outtype",
            "double",
        ],
    )

    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(fits.getdata(output), 2.0)
    assert fits.getdata(output).dtype == np.dtype(">f8")
    assert fits.getheader(output)["NCOMBINE"] == 3


def test_copy_command(tmp_path):
    input_path = tmp_path / "copy_in.fits"
    output = tmp_path / "copy_out.fits"
    _write_fits(input_path, np.arange(12, dtype=float).reshape(3, 4))

    runner = CliRunner()
    result = runner.invoke(
        copy_command,
        [
            str(input_path),
            str(output),
            "--trimsec",
            "[2:3,1:2]",
            "--outtype",
            "double",
        ],
    )

    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(
        fits.getdata(output), np.asarray([[1.0, 2.0], [5.0, 6.0]])
    )
    assert fits.getdata(output).dtype == np.dtype(">f8")


def test_arith_command(tmp_path):
    input_path = tmp_path / "arith_in.fits"
    output = tmp_path / "arith_out.fits"
    _write_fits(input_path, np.full((2, 3), 5.0))

    runner = CliRunner()
    result = runner.invoke(
        arith_command,
        [
            str(input_path),
            "+",
            "2",
            str(output),
            "--outtype",
            "double",
        ],
    )

    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(fits.getdata(output), 7.0)
    assert fits.getdata(output).dtype == np.dtype(">f8")


def test_group_comb_minmax_with_aux_outputs(tmp_path):
    paths = []
    for i, value in enumerate([0.0, 10.0, 10.0, 100.0]):
        path = tmp_path / f"mm_{i}.fits"
        _write_fits(path, np.full((2, 3), value))
        paths.append(path)

    input_list = tmp_path / "inputs.list"
    input_list.write_text("\n".join(str(path) for path in paths))
    output = tmp_path / "mm_out.fits"
    nrej = tmp_path / "mm_nrej.fits"
    mask = tmp_path / "mm_mask.fits"
    low = tmp_path / "mm_low.fits"
    upp = tmp_path / "mm_upp.fits"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "comb",
            f"@{input_list}",
            str(output),
            "--reject",
            "minmax",
            "--nlow",
            "1",
            "--nhigh",
            "1",
            "--output-nrej",
            str(nrej),
            "--rejmask",
            str(mask),
            "--output-low",
            str(low),
            "--output-upp",
            str(upp),
        ],
    )

    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(fits.getdata(output), 10.0)
    np.testing.assert_array_equal(fits.getdata(nrej), np.full((2, 3), 2))
    assert fits.getdata(mask).shape == (4, 2, 3)
    np.testing.assert_allclose(fits.getdata(low), 10.0)
    np.testing.assert_allclose(fits.getdata(upp), 10.0)


def test_comb_command_numeric_offsets_and_memlimit(tmp_path):
    base = np.arange(6, dtype=float).reshape(2, 3)
    paths = []
    for i, offset in enumerate([0.0, 10.0]):
        path = tmp_path / f"offset_{i}.fits"
        _write_fits(path, base + offset)
        paths.append(path)

    output = tmp_path / "offset_out.fits"
    runner = CliRunner()
    result = runner.invoke(
        comb_command,
        [
            ",".join(str(path) for path in paths),
            str(output),
            "--combine",
            "average",
            "--offsets",
            "0,0; 1,1",
            "--memlimit",
            "1MiB",
        ],
    )

    assert result.exit_code == 0, result.output
    expected = np.full((3, 4), np.nan)
    expected[0:2, 0:3] = base
    expected[1:3, 1:4] = np.nanmean(
        np.stack(
            [
                expected[1:3, 1:4],
                base + 10.0,
            ]
        ),
        axis=0,
    )
    np.testing.assert_allclose(fits.getdata(output), expected, equal_nan=True)
