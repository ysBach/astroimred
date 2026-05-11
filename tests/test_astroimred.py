import logging
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

import astroimred as air
from astroimred import logging as airlogging
from astroimred._core import geometry, numeric, time, units
from astroimred.fitsmgmt import header, io, table
from astroimred.imutil import ccdops


def test_top_level_lightweight_exports():
    """Lightweight functions land in the astroimred namespace."""
    assert callable(air.load_ccd)
    assert callable(air.write2fits)
    assert callable(air.imslice)
    assert callable(air.give_stats)
    assert callable(air.binning)
    assert callable(air.fixpix)
    assert callable(air.cmt2hdr)
    assert callable(air.fits_summary)


def test_top_level_does_not_export_heavy_tools():
    """Heavy image task functions are NOT in the top-level namespace."""
    assert not hasattr(air, "imcombine")
    assert not hasattr(air, "imarith")
    assert not hasattr(air, "imcopy")
    assert not hasattr(air, "smooth_med")


def test_heavy_tools_importable_directly():
    """Heavy image tools are importable from their own modules."""
    from astroimred.imutil.imarith import imarith
    from astroimred.imutil.imcombine import imcombine, ndcombine
    from astroimred.imutil.imcopy import imcopy
    from astroimred.imutil.imsmooth import smooth_med

    assert callable(imcombine)
    assert callable(ndcombine)
    assert callable(imarith)
    assert callable(imcopy)
    assert callable(smooth_med)


def test_subpackage_attributes():
    """air.imutil and air.fitsmgmt are accessible as subpackage attributes."""
    assert air.imutil is import_module("astroimred.imutil")
    assert air.fitsmgmt is import_module("astroimred.fitsmgmt")
    assert air.fitsmgmt.io is import_module("astroimred.fitsmgmt.io")
    assert air.imutil.ccdops is import_module("astroimred.imutil.ccdops")


def test_functions_match_submodules():
    """Top-level star-imported names point to the same objects as submodule attrs."""
    assert air.load_ccd is air.fitsmgmt.io.load_ccd
    assert air.imslice is air.imutil.ccdops.imslice
    assert air.give_stats is air.imutil.imstat.give_stats
    assert air.fixpix is air.imutil.pixels.fixpix
    assert air.binning is air._core.numeric.binning


def test_no_legacy_flat_module_aliases():
    """Legacy shortcuts like import astroimred.io are not registered."""
    for name in ("io", "ccdops", "header", "numeric", "pixels", "imstat"):
        assert f"astroimred.{name}" not in sys.modules


def test_logger_accessible():
    """logger is accessible at top level and matches fitsmgmt.logger."""
    assert air.logger is air.fitsmgmt.logger


def test_imutil_numba_flag():
    """set_use_numba updates _config and the module-level IMOPS_USE_NUMBA."""
    import astroimred.imutil as imutil
    from astroimred.imutil import _config

    original = _config.IMOPS_USE_NUMBA
    try:
        imutil.set_use_numba(False)
        assert imutil.IMOPS_USE_NUMBA is False
        assert _config.IMOPS_USE_NUMBA is False

        imutil.set_use_numba(True)
        assert imutil.IMOPS_USE_NUMBA is True
        assert _config.IMOPS_USE_NUMBA is True
    finally:
        imutil.set_use_numba(original)


def test_import_is_lightweight():
    """import astroimred does not load numba, ccdproc, or astroscrappy."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import astroimred; import sys; "
            "heavy = [m for m in sys.modules if m == 'numba' or "
            "m.startswith('numba.') or m in ('ccdproc', 'astroscrappy')]; "
            "print(heavy[:3] if heavy else 'clean')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "clean", f"Heavy deps loaded: {result.stdout}"


@pytest.fixture
def temp_env():
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def dummy_fits(temp_env):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["EXPTIME"] = 10.0
    data = np.zeros((10, 10))
    data[2:5, 2:5] = 100
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    fpath = temp_env / "test.fits"
    hdu.writeto(fpath)
    return fpath


def test_logging():
    airlogging.set_log_level("DEBUG")
    airlogging.enable_console_logging(level=10)
    assert airlogging.logger.level == logging.DEBUG
    for handler in airlogging.logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            airlogging.logger.removeHandler(handler)


def test_listify():
    assert geometry.listify(1) == [1]
    assert geometry.listify([1, 2]) == [1, 2]
    assert geometry.listify("abc") == ["abc"]


def test_str_now():
    assert len(time.str_now()) > 0


def test_as_quantity():
    q1 = units.as_quantity(10, "km")
    assert q1.value == 10.0 and q1.unit == u.km
    q2 = units.as_quantity(10 * u.m, "km")
    assert q2.value == 0.01 and q2.unit == u.km


def test_binning():
    arr = np.arange(16).reshape(4, 4)
    binned = numeric.binning(arr, factors=(2, 2))
    expected_bin = np.array([[2.5, 4.5], [10.5, 12.5]])
    assert np.allclose(binned, expected_bin)


def test_header_utils(dummy_fits):
    hdr = fits.getheader(dummy_fits)
    header.cmt2hdr(hdr, "h", "Test history")
    assert "Test history" in str(hdr.get("HISTORY"))
    header.update_process(hdr, "BiasSub")
    assert "BiasSub" in str(hdr.get("PROCESS"))
    header.update_tlm(hdr)
    assert "FITS-TLM" in hdr


def test_images_io(dummy_fits):
    ccd = io.load_ccd(dummy_fits)
    assert ccd.shape == (10, 10)
    inputs = io.inputs2list(str(dummy_fits.parent / "*.fits"))
    assert [Path(p).name for p in inputs] == ["test.fits"]
    outpath = dummy_fits.parent / "out.fits"
    io.write2fits(ccd.data, ccd.header, outpath)
    assert outpath.exists()


def test_image_process(dummy_fits):
    ccd = io.load_ccd(dummy_fits)
    sl_ccd = ccdops.imslice(ccd, "[2:5, 2:5]")
    assert sl_ccd.shape == (4, 4)
    cut, _ = ccdops.cut_ccd(ccd, (5, 5), (4, 4))
    assert cut.shape == (4, 4)
    binccd = ccdops.bin_ccd(ccd, factors=(2, 2))
    assert binccd.shape == (5, 5)
    assert "XBINNING" in binccd.header
    assert "YBINNING" in binccd.header


def test_header_edits(dummy_fits):
    header.hedit(
        dummy_fits, "OBJECT", "TestObj", overwrite=True, add=True, output=dummy_fits
    )
    assert fits.getval(dummy_fits, "OBJECT") == "TestObj"
    hdr = fits.getheader(dummy_fits)
    hdr["TEMP"] = 123
    hdr = header.key_remover(hdr, ["TEMP"])
    assert "TEMP" not in hdr


def test_ccd_attributes(dummy_fits):
    ccd = io.load_ccd(dummy_fits)
    ccdops.set_ccd_attribute(ccd, "gain", 2.0, unit="electron/adu")
    assert ccd.gain.value == 2.0
    assert ccd.gain.unit == u.electron / u.adu


def test_files_summary(dummy_fits):
    outpath = dummy_fits.parent / "out.fits"
    header.hedit(
        dummy_fits, "OBJECT", "TestObj", overwrite=True, add=True, output=dummy_fits
    )
    io.write2fits(np.zeros((10, 10)), fits.Header(), outpath)
    df = table.fits_summary([dummy_fits, outpath], keywords=["OBJECT", "NAXIS"])
    df = df.sort_values("file").reset_index(drop=True)
    assert df.iloc[0]["OBJECT"] is None
    assert df.iloc[1]["OBJECT"] == "TestObj"
