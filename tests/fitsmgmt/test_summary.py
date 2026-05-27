import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData

from astroimred.fitsmgmt import table


class TestSummary:
    """Tests for table module."""

    def test_fits_summary(self, tmp_path):
        """Test creating table table from FITS files."""
        # Create files with header
        keys = ["OBJECT", "FILTER", "EXPTIME"]
        data = [("M1", "V", 10.0), ("M1", "B", 20.0), ("M2", "V", 10.0)]

        paths = []
        for i, (obj, filt, exp) in enumerate(data):
            p = tmp_path / f"img{i}.fits"
            hdr = fits.Header()
            hdr["OBJECT"] = obj
            hdr["FILTER"] = filt
            hdr["EXPTIME"] = exp
            fits.writeto(p, np.zeros((10, 10)), header=hdr)
            paths.append(str(p))

        # Run fits_summary
        df = table.fits_summary(paths, keywords=keys)

        assert len(df) == 3
        assert "file" in df.columns
        assert list(df["OBJECT"]) == ["M1", "M1", "M2"]
        assert list(df["FILTER"]) == ["V", "B", "V"]
        np.testing.assert_allclose(df["EXPTIME"], [10.0, 20.0, 10.0])

    def test_fits_summary_parq_output(self, tmp_path, monkeypatch):
        """Test parquet output selection for .parq table files."""
        p = tmp_path / "img.fits"
        hdr = fits.Header()
        hdr["OBJECT"] = "M1"
        fits.writeto(p, np.zeros((10, 10)), header=hdr)

        calls = []

        def fake_to_parquet(self, output, index=False):
            calls.append((output, index, list(self.columns)))

        monkeypatch.setattr("pandas.DataFrame.to_parquet", fake_to_parquet)

        output = tmp_path / "table.parq"
        df = table.fits_summary(
            [p],
            keywords=["OBJECT"],
            output=output,
        )

        assert list(df["OBJECT"]) == ["M1"]
        assert calls == [(output, False, ["file", "filesize", "OBJECT"])]

    def test_fits_summary_string_keyword(self, tmp_path):
        """A single keyword string is treated as one column name."""
        p = tmp_path / "img.fits"
        hdr = fits.Header()
        hdr["OBJECT"] = "M1"
        fits.writeto(p, np.zeros((10, 10)), header=hdr)

        df = table.fits_summary([p], keywords="OBJECT")

        assert "OBJECT" in df.columns
        assert "O" not in df.columns
        assert list(df["OBJECT"]) == ["M1"]

    def test_fits_summary_glob_fname_name(self, tmp_path):
        """Glob inputs support the documented fname_option='name' path."""
        for name in ["b.fits", "a.fits"]:
            hdr = fits.Header()
            hdr["OBJECT"] = name
            fits.writeto(tmp_path / name, np.zeros((10, 10)), header=hdr)

        df = table.fits_summary(
            str(tmp_path / "*.fits"),
            keywords=["OBJECT"],
            fname_option="name",
        )

        assert list(df["file"]) == ["a.fits", "b.fits"]

    def test_fits_summary_hdu_inputs(self):
        """HDU-like inputs are summarized without being treated as paths."""
        hdr = fits.Header()
        hdr["OBJECT"] = "M1"
        hdu = fits.PrimaryHDU(data=np.zeros((10, 10)), header=hdr)

        df = table.fits_summary([hdu], keywords=["OBJECT"])

        assert list(df["file"]) == ["PrimaryHDU in fitslist[0]"]
        assert df.iloc[0]["filesize"] is None
        assert list(df["OBJECT"]) == ["M1"]

    def test_fits_summary_ccddata_inputs(self):
        """CCDData inputs keep their synthetic names and no filesystem size."""
        hdr = fits.Header()
        hdr["OBJECT"] = "M1"
        ccd = CCDData(np.zeros((10, 10)), unit="adu", header=hdr)

        df = table.fits_summary([ccd], keywords=["OBJECT"])

        assert list(df["file"]) == ["CCDData in fitslist[0]"]
        assert df.iloc[0]["filesize"] is None
        assert list(df["OBJECT"]) == ["M1"]
