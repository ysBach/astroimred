"""Default parameter presets for reduction workflows."""

import numpy as np
from astropy import units as u

__all__ = [
    "MEDCOMB_KEYS_INT",
    "SUMCOMB_KEYS_INT",
    "MEDCOMB_KEYS_FLT32",
    "LACOSMIC_KEYS",
    "LACOSMIC_CRREJ",
]


MEDCOMB_KEYS_INT = {
    "dtype": "int16",
    "combine_method": "median",
    "reject_method": None,
    "unit": u.adu,
    "combine_uncertainty_function": None,
}

SUMCOMB_KEYS_INT = {
    "dtype": "int16",
    "combine_method": "sum",
    "reject_method": None,
    "unit": u.adu,
    "combine_uncertainty_function": None,
}

MEDCOMB_KEYS_FLT32 = {
    "dtype": "float32",
    "combine_method": "median",
    "reject_method": None,
    "unit": u.adu,
    "combine_uncertainty_function": None,
}


# I skipped two params in IRAF LACOSMIC: gain=2.0, readnoise=6.
LACOSMIC_KEYS = {
    "sigclip": 4.5,
    "sigfrac": 0.5,
    "objlim": 1.0,
    "satlevel": np.inf,
    "invar": None,
    "inbkg": None,
    "niter": 4,
    "sepmed": False,
    "cleantype": "medmask",
    "fsmode": "median",
    "psfmodel": "gauss",
    "psffwhm": 2.5,
    "psfsize": 7,
    "psfk": None,
    "psfbeta": 4.765,
}

# same as above, but simplify `fsmode`, `psfmodel`, and `psfk` into `fs`
LACOSMIC_CRREJ = {
    "sigclip": 4.5,
    "sigfrac": 0.5,
    "objlim": 1.0,
    "satlevel": np.inf,
    "invar": None,
    "inbkg": None,
    "niter": 4,
    "sepmed": False,
    "cleantype": "medmask",
    "fs": "median",
    "psffwhm": 2.5,
    "psfsize": 7,
    "psfbeta": 4.765,
}
