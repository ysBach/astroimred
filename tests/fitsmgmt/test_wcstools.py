"""Tests for WCS helper utilities."""

import numpy as np
from astropy.wcs import WCS

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


class TestLinearWcsHelpers:
    """Tests for local_cd_matrix / make_linear_wcs / make_zoomed_wcs."""

    def test_helpers_exported_from_package_root(self):
        assert air.local_cd_matrix is wcs.local_cd_matrix
        assert air.make_linear_wcs is wcs.make_linear_wcs
        assert air.make_zoomed_wcs is wcs.make_zoomed_wcs

    def test_local_cd_matrix_matches_cd_for_linear_wcs(self):
        """local_cd_matrix should recover the exact CD matrix of a linear WCS.

        Uses CRVAL = (0, 0) deliberately: away from dec=0, a degree of RA
        corresponds to less sky distance (the cos(dec) factor), so d(RA)/dx
        measured directly (as `local_cd_matrix` does) differs from the WCS's
        own CD matrix, which is defined in the cos(dec)-corrected
        tangent-plane frame. At dec=0 that factor is 1, removing the
        ambiguity so this test checks `local_cd_matrix` itself rather than
        the projection's geometry.
        """
        cd_true = np.array([[-0.01, 0.0], [0.0, 0.01]])
        w = WCS(naxis=2)
        w.wcs.crpix = [50.5, 50.5]
        w.wcs.crval = [0.0, 0.0]
        w.wcs.cd = cd_true
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        cd_est = wcs.local_cd_matrix(w, 49.5, 49.5)  # near CRPIX, 0-indexed
        np.testing.assert_allclose(cd_est, cd_true, atol=1e-7)

    def test_make_linear_wcs_roundtrip(self):
        cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
        shape = (100, 200)
        w = wcs.make_linear_wcs(cd, shape, crval=(10.0, 20.0))
        assert w.wcs.crpix.tolist() == [100.0, 50.0]
        assert w.wcs.crval.tolist() == [10.0, 20.0]
        np.testing.assert_allclose(w.wcs.cd, cd)
        ra, dec = w.all_pix2world(w.wcs.crpix[0] - 1, w.wcs.crpix[1] - 1, 0)
        np.testing.assert_allclose([ra, dec], [10.0, 20.0], atol=1e-9)

    def test_make_zoomed_wcs_preserves_footprint(self):
        cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
        shape = (100, 200)
        wcs_ref = wcs.make_linear_wcs(cd, shape, crval=(10.0, 20.0))
        wcs_zoom, shape_zoom = wcs.make_zoomed_wcs(wcs_ref, shape, zoom=4)
        assert shape_zoom == (400, 800)
        # Corners in FITS 1-indexed pixel convention (pixel edges at
        # half-integers, e.g. the left edge of pixel 1 is at x=0.5) must map
        # to the same sky position via both WCSs -- this is what
        # "edge-aligned" means. The `x' = zoom*(x-0.5)+0.5` affine map is
        # defined in this same 1-indexed convention (it is applied directly
        # to CRPIX, which is always 1-indexed in FITS/astropy.wcs).
        corners_ref = [(0.5, 0.5), (0.5, 100.5), (200.5, 0.5), (200.5, 100.5)]
        for x_ref, y_ref in corners_ref:
            x_zoom = 4 * (x_ref - 0.5) + 0.5
            y_zoom = 4 * (y_ref - 0.5) + 0.5
            sky_ref = wcs_ref.all_pix2world(x_ref, y_ref, 1)
            sky_zoom = wcs_zoom.all_pix2world(x_zoom, y_zoom, 1)
            np.testing.assert_allclose(sky_ref, sky_zoom, atol=1e-9)
