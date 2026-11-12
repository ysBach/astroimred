import numpy as np
import pytest
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

    def test_fits_summary_fitsio_header_backend_matches_astropy(self, tmp_path):
        """The optional fitsio header backend preserves summary values."""
        pytest.importorskip("fitsio")

        paths = []
        for i, (obj, filt, exp) in enumerate([("M1", "V", 10.0), ("M2", "B", 20.0)]):
            p = tmp_path / f"img{i}.fits"
            hdr = fits.Header()
            hdr["OBJECT"] = obj
            hdr["FILTER"] = filt
            hdr["EXPTIME"] = exp
            fits.writeto(p, np.zeros((10, 10)), header=hdr)
            paths.append(p)

        keys = ["OBJECT", "FILTER", "EXPTIME"]
        expected = table.fits_summary(paths, keywords=keys, header_backend="astropy")
        actual = table.fits_summary(paths, keywords=keys, header_backend="fitsio")

        assert actual.equals(expected)

    def test_fits_summary_rejects_unknown_header_backend(self, tmp_path):
        """Header backend selection is explicit, not an arbitrary fits.open kwarg."""
        p = tmp_path / "img.fits"
        fits.writeto(p, np.zeros((10, 10)))

        with pytest.raises(ValueError, match="header_backend"):
            table.fits_summary([p], keywords=["NAXIS"], header_backend="bogus")

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

    def test_select_fits_missing_strings_and_caller_index_immutability(self):
        """select_fits should handle missing string values without mask error and not mutate caller index."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "file": ["img1.fits", "img2.fits", "img3.fits"],
                "OBJECT": ["NGC1234", None, "NGC5678"],
                "EXPTIME": [100.0, 100.0, 200.0],
            },
            index=[10, 20, 30],
        )
        original_index = df.index.copy()

        matched = table.select_fits(
            df,
            type_key="OBJECT",
            type_val="NGC1.*",
            path_to_text=True,
        )
        assert matched == ["img1.fits"]
        pd.testing.assert_index_equal(df.index, original_index)

        # Broad regex '.*' must not select missing values (None / pd.NA / NaN)
        matched_all = table.select_fits(
            df,
            type_key="OBJECT",
            type_val=".*",
            path_to_text=True,
        )
        assert matched_all == ["img1.fits", "img3.fits"]

        # Regex on all-numeric column
        matched_num = table.select_fits(
            df,
            type_key="EXPTIME",
            type_val="100.*",
            path_to_text=True,
        )
        assert matched_num == ["img1.fits", "img2.fits"]

    def test_select_fits_in_memory_ccddata_list(self):
        """select_fits should support list of CCDData objects without array coercion issues."""
        ccd1 = CCDData(
            np.ones((2, 2)), unit="adu", header={"OBJECT": "M1", "EXPTIME": 10}
        )
        ccd2 = CCDData(
            np.ones((2, 2)), unit="adu", header={"OBJECT": "M2", "EXPTIME": 20}
        )
        ccd_list = [ccd1, ccd2]

        matched = table.select_fits(
            ccd_list,
            type_key="OBJECT",
            type_val="M1",
            prefer_ccddata=True,
        )
        assert len(matched) == 1
        assert matched[0] is ccd1
