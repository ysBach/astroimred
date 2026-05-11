import numpy as np
import pytest

import astroimred.reduction as imred
from astroimred.reduction._presets import MEDCOMB_KEYS_INT
from astroimred.reduction.crrej import LACOSMIC_CRREJ, LACOSMIC_KEYS, parse_crrej_psf


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
