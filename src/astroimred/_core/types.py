"""Shared type aliases for the astroimred package."""

from __future__ import annotations

import os
from typing import TypeAlias, get_args

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData

StrPathLike: TypeAlias = str | os.PathLike[str]
"""String or any path-like object (e.g. pathlib.Path, PosixPath, WindowsPath)."""

HDUExt: TypeAlias = int | str | tuple[str, int] | None
"""FITS extension specifier: index, EXTNAME, (EXTNAME, EXTVER), or None (auto)."""

HDULike: TypeAlias = CCDData | fits.PrimaryHDU | fits.ImageHDU
"""Image object carrying FITS-style data and header metadata."""
HDULike_types = get_args(HDULike)

FQArr: TypeAlias = float | np.ndarray | u.Quantity
"""Float, numpy array, or astropy Quantity."""

FITSLike: TypeAlias = HDULike | np.ndarray | StrPathLike
"""Image data input accepted by core FITS/image utilities."""

SectionLike: TypeAlias = str | slice | tuple[slice, ...] | None
"""Image/FITS section specification."""
