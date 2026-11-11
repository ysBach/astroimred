import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astroimred.fitsmgmt import naming


class TestFitsRenamer:
    """Tests for fitsrenamer."""

    def test_stepped_wcs_trim_is_rejected(self, tmp_path) -> None:
        hdr = fits.Header({"OBJECT": "X", "CRPIX1": 4.0, "CRPIX2": 4.0})
        path = tmp_path / "stepped.fits"
        fits.writeto(path, np.ones((10, 10)), hdr)
        with pytest.raises(NotImplementedError, match="unit steps"):
            naming.fitsrenamer(path, trimsec="[1:8:2,1:8]", newtop=tmp_path / "out")

    def test_fitsrenamer_with_trimsec_preserves_wcs_and_trims_correctly(self, tmp_path):
        """fitsrenamer with trimsec should trim correctly and update WCS without tuple error."""
        fpath = tmp_path / "orig.fits"
        data = np.arange(100, dtype=float).reshape(10, 10)
        hdr = fits.Header()
        hdr["OBJECT"] = "TARGET"
        hdr["CRPIX1"] = 5.0
        hdr["CRPIX2"] = 5.0
        hdr["CDELT1"] = 0.1
        hdr["CDELT2"] = 0.1
        hdr["CRVAL1"] = 120.0
        hdr["CRVAL2"] = 30.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        fits.writeto(fpath, data, header=hdr)

        # FITS section [2:5, 3:7]: x from 2 to 5 (1-based), y from 3 to 7 (1-based)
        # Python 0-based: x in [1:5] (len 4), y in [2:7] (len 5)
        out_path = naming.fitsrenamer(
            fpath=fpath,
            rename_by=["OBJECT"],
            trimsec="[2:5, 3:7]",
            newtop=tmp_path / "renamed",
        )

        assert out_path.exists()
        with fits.open(out_path) as hdul:
            trimmed = hdul[0].data
            out_hdr = hdul[0].header
            assert trimmed.shape == (5, 4)
            # Check that data slice matches exact expected region
            np.testing.assert_array_equal(trimmed, data[2:7, 1:5])
            # Verify exact world-coordinate equality across multiple points
            w_orig = WCS(hdr)
            w_trim = WCS(out_hdr)
            # Point 1: (x=3, y=4) -> (2, 2)
            # Point 2: (x=4, y=6) -> (3, 4)
            for x_orig, y_orig in [(3, 4), (4, 6)]:
                x_trim = x_orig - 1
                y_trim = y_orig - 2
                ra_orig, dec_orig = w_orig.pixel_to_world_values(x_orig, y_orig)
                ra_trim, dec_trim = w_trim.pixel_to_world_values(x_trim, y_trim)
                np.testing.assert_allclose(ra_trim, ra_orig, atol=1e-12)
                np.testing.assert_allclose(dec_trim, dec_orig, atol=1e-12)

    def test_fitsrenamer_trimsec_without_wcs_preserves_data(self, tmp_path):
        """fitsrenamer with trimsec should work correctly when no WCS is present."""
        fpath = tmp_path / "dark.fits"
        data = np.arange(100, dtype=float).reshape(10, 10)
        hdr = fits.Header({"OBJECT": "DARK"})
        fits.writeto(fpath, data, header=hdr)

        out_path = naming.fitsrenamer(
            fpath=fpath,
            rename_by=["OBJECT"],
            trimsec="[2:5, 3:7]",
            newtop=tmp_path / "renamed_dark",
        )
        assert out_path.exists()
        with fits.open(out_path) as hdul:
            assert hdul[0].data.shape == (5, 4)
            np.testing.assert_array_equal(hdul[0].data, data[2:7, 1:5])

    def test_fitsrenamer_trimsec_linear_wcs_equality(self, tmp_path):
        """Linear non-celestial WCS without CTYPE should also preserve coordinate mapping."""
        fpath = tmp_path / "linear.fits"
        data = np.arange(100, dtype=float).reshape(10, 10)
        hdr = fits.Header(
            {
                "OBJECT": "LIN",
                "CRPIX1": 5.0,
                "CRPIX2": 5.0,
                "CDELT1": 2.0,
                "CDELT2": 3.0,
                "CRVAL1": 10.0,
                "CRVAL2": 20.0,
            }
        )
        fits.writeto(fpath, data, header=hdr)

        out_path = naming.fitsrenamer(
            fpath=fpath,
            rename_by=["OBJECT"],
            trimsec="[2:5, 3:7]",
            newtop=tmp_path / "renamed_lin",
        )
        assert out_path.exists()
        with fits.open(out_path) as hdul:
            w_orig = WCS(hdr)
            w_trim = WCS(hdul[0].header)
            # (x=3, y=4) untrimmed -> (2, 2) trimmed
            orig_coords = w_orig.pixel_to_world_values(3, 4)
            trim_coords = w_trim.pixel_to_world_values(2, 2)
            np.testing.assert_allclose(trim_coords, orig_coords, atol=1e-12)
