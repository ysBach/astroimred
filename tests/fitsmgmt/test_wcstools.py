"""Tests for WCS helper utilities."""

import astroimred as air
from astroimred.fitsmgmt import wcs


class TestWcsTools:
    """Tests for WCS helper exports."""

    def test_wcs_exported_from_package_root(self):
        """WCS helpers live in wcs and package root."""
        assert air.wcsremove is wcs.wcsremove

    def test_wcsremove_header(self, sample_header):
        """Test WCS keyword removal from an in-memory header."""
        hdr = sample_header.copy()
        hdr["CRVAL1"] = 1.0
        hdr["CRVAL2"] = 2.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"

        out = wcs.wcsremove(hdr, verbose=False)

        assert "CRVAL1" not in out
        assert "CRVAL2" not in out
        assert "CTYPE1" not in out
        assert "CTYPE2" not in out
        assert "OBJECT" in out
