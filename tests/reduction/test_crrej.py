import importlib

import numpy as np
import pytest

import astroimred.reduction as imred
from astroimred.reduction._presets import MEDCOMB_KEYS_INT
from astroimred.reduction.crrej import (
    LACOSMIC_CRREJ,
    LACOSMIC_KEYS,
    _sigma_clipped_std,
    parse_crrej_psf,
)


def test_lacosmic_defaults_are_package_exports():
    assert imred.LACOSMIC_KEYS is LACOSMIC_KEYS
    assert imred.LACOSMIC_CRREJ is LACOSMIC_CRREJ
    assert imred.parse_crrej_psf is parse_crrej_psf

    assert LACOSMIC_KEYS["fsmode"] == "median"
    assert LACOSMIC_CRREJ["fs"] == "median"


def test_reduction_presets_are_package_exports():
    assert imred.MEDCOMB_KEYS_INT is MEDCOMB_KEYS_INT
    assert imred.MEDCOMB_KEYS_INT["combine_method"] == "median"


def test_parse_crrej_psf_scalar_modes():
    assert parse_crrej_psf() == {"fsmode": "median"}
    assert parse_crrej_psf("gauss", psffwhm=2, psfsize=3, psfbeta=1) == {
        "fsmode": "convolve",
        "psfmodel": "gauss",
        "psffwhm": 2,
        "psfsize": 3,
    }
    assert parse_crrej_psf("moffat", psffwhm=2, psfsize=3, psfbeta=1) == {
        "fsmode": "convolve",
        "psfmodel": "moffat",
        "psffwhm": 2,
        "psfsize": 3,
        "psfbeta": 1,
    }


def test_parse_crrej_psf_list_modes():
    assert parse_crrej_psf("moffat", psffwhm=2, psfsize=3, psfbeta=[1, 2]) == {
        "fsmode": ["convolve", "convolve"],
        "psfmodel": ["moffat", "moffat"],
        "psfk": [None, None],
        "psffwhm": [2, 2],
        "psfsize": [3, 3],
        "psfbeta": [1, 2],
    }


def test_parse_crrej_psf_kernel_mode():
    kernel = np.eye(3)
    parsed = parse_crrej_psf(kernel)
    assert parsed["fsmode"] == "convolve"
    np.testing.assert_array_equal(parsed["psfk"], kernel)


def test_parse_crrej_psf_rejects_incompatible_lengths():
    with pytest.raises(ValueError, match="must all be length 1 or the same length"):
        parse_crrej_psf("moffat", psffwhm=2, psfsize=[3, 3, 3], psfbeta=[1, 2])


def test_sigma_clipped_std_returns_scalar_sample_std_after_clipping():
    data = np.array([1.0, 2.0, 3.0, 100.0, np.nan])
    kwargs = {"sigma": 2.0, "maxiters": 5, "std_ddof": 1, "stdfunc": "mad_std"}

    result = _sigma_clipped_std(data, kwargs)

    np.testing.assert_allclose(result, 1.0)


def test_sigma_clipped_std_uses_lowlevel_skip_nonfinite_std(monkeypatch):
    crrej_module = importlib.import_module("astroimred.reduction.crrej")
    calls = {}

    def fake_sigma_clipper(data, **kwargs):
        calls["sigma_clipper_kwargs"] = kwargs
        return np.array([1.0, np.nan, 2.0, 3.0])

    def fake_std_skip_nonfinite(data, ddof=0, *, copy=False):
        calls["std_data"] = np.asarray(data).copy()
        calls["std_ddof"] = ddof
        calls["std_copy"] = copy
        return 1.25

    monkeypatch.setattr(crrej_module, "sigma_clipper", fake_sigma_clipper)
    monkeypatch.setattr(
        crrej_module.rdl,
        "std_skip_nonfinite",
        fake_std_skip_nonfinite,
    )

    result = _sigma_clipped_std(np.ones((2, 3)), {"sigma": 2.0, "std_ddof": 1})

    assert result == 1.25
    assert calls["sigma_clipper_kwargs"] == {"sigma": 2.0}
    np.testing.assert_array_equal(calls["std_data"], [1.0, np.nan, 2.0, 3.0])
    assert calls["std_ddof"] == 1
    assert calls["std_copy"] is False


def test_sigma_clipped_std_rejects_astropy_only_kwargs():
    with pytest.raises(ValueError, match="Unsupported sigclip_kw"):
        _sigma_clipped_std(np.ones((2, 3)), {"axis": 0})
