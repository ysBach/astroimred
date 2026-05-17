"""Aperture helpers backed by astroaap."""

from __future__ import annotations

from typing import Any

import astroapers as aap
import numpy as np
from astropy.nddata import CCDData, Cutout2D

from ._aper_backend import as_radians

__all__ = [
    "circ_ap_an",
    "cutout_from_ap",
    "ellip_ap_an",
    "pill_ap_an",
]


def cutout_from_ap(
    ap: Any,
    ccd: CCDData | np.ndarray,
    method: str = "bbox",
    fill_value: float = np.nan,
) -> Cutout2D | list[Cutout2D]:
    """Return `~astropy.nddata.Cutout2D` objects from aperture bounding boxes."""
    data = ccd.data if isinstance(ccd, CCDData) else np.asarray(ccd)
    positions = np.asarray(ap.positions, dtype=np.float64).reshape(-1, 2)
    masks = ap.get_apmask(method="center" if method == "bbox" else method)
    masks = masks if isinstance(masks, list) else [masks]
    cuts = []
    for pos, apmask in zip(positions, masks, strict=True):
        cut = Cutout2D(data, position=pos, size=apmask.weights.shape)
        if method != "bbox":
            cut.data = apmask.weighted_cutout(data, fill_value=fill_value)
        cuts.append(cut)
    return cuts[0] if len(cuts) == 1 else cuts


def circ_ap_an(
    positions,
    r_ap: float | None = None,
    r_in: float | None = None,
    r_out: float | None = None,
    fwhm: float | None = None,
    f_ap: float = 1.5,
    f_in: float = 4.0,
    f_out: float = 6.0,
) -> tuple[aap.CircAp, aap.CircAn]:
    """Return circular aperture and annulus objects."""
    r_ap = _sanitize_apsize(r_ap, fwhm=fwhm, factor=f_ap, name="r_ap")
    r_in = _sanitize_apsize(r_in, fwhm=fwhm, factor=f_in, name="r_in")
    r_out = _sanitize_apsize(r_out, fwhm=fwhm, factor=f_out, name="r_out")
    return aap.CircAp(positions, r=r_ap), aap.CircAn(positions, r_in=r_in, r_out=r_out)


def ellip_ap_an(
    positions,
    r_ap: float | tuple[float, float] | None = None,
    r_in: float | tuple[float, float] | None = None,
    r_out: float | tuple[float, float] | None = None,
    fwhm: float | None = None,
    theta: float = 0.0,
    f_ap: float | tuple[float, float] = (1.5, 1.5),
    f_in: float | tuple[float, float] = (4.0, 4.0),
    f_out: float | tuple[float, float] = (6.0, 6.0),
) -> tuple[aap.EllipAp, aap.EllipAn]:
    """Return elliptical aperture and annulus objects."""
    a_ap, b_ap = _sanitize_apsize(r_ap, fwhm, factor=f_ap, name="r_ap", repeat=True)
    a_in, b_in = _sanitize_apsize(r_in, fwhm, factor=f_in, name="r_in", repeat=True)
    a_out, b_out = _sanitize_apsize(
        r_out, fwhm, factor=f_out, name="r_out", repeat=True
    )
    theta_rad = as_radians(theta)
    ap = aap.EllipAp(positions, a=a_ap, b=b_ap, theta=theta_rad)
    an = aap.EllipAn(
        positions,
        a_in=a_in,
        b_in=b_in,
        a_out=a_out,
        b_out=b_out,
        theta_in=theta_rad,
    )
    return ap, an


def pill_ap_an(
    positions,
    fwhm,
    trail,
    theta=0.0,
    f_ap=(1.5, 1.5),
    f_in=(4.0, 4.0),
    f_out=(6.0, 6.0),
    f_w=1.0,
) -> tuple[aap.PillAp, aap.PillAn]:
    """Return pill aperture and annulus objects."""
    fwhm = np.repeat(fwhm, 2) if np.isscalar(fwhm) else np.asarray(fwhm)
    f_ap = np.repeat(f_ap, 2) if np.isscalar(f_ap) else np.asarray(f_ap)
    f_in = np.repeat(f_in, 2) if np.isscalar(f_in) else np.asarray(f_in)
    f_out = np.repeat(f_out, 2) if np.isscalar(f_out) else np.asarray(f_out)

    a_ap = float(f_ap[0] * fwhm[0])
    b_ap = float(f_ap[1] * fwhm[1])
    a_in = float(f_in[0] * fwhm[0])
    b_in = float(f_in[1] * fwhm[1])
    a_out = float(f_out[0] * fwhm[0])
    b_out = float(f_out[1] * fwhm[1])
    w = float(f_w * trail)
    theta_rad = as_radians(theta)

    ap = aap.PillAp(positions, w=w, a=a_ap, b=b_ap, theta=theta_rad)
    an = aap.PillAn(
        positions,
        w_in=w,
        a_in=a_in,
        b_in=b_in,
        w_out=w,
        a_out=a_out,
        b_out=b_out,
        theta_in=theta_rad,
        validate=False,
    )
    return ap, an


def _sanitize_apsize(
    size=None,
    fwhm=None,
    factor=None,
    name: str = "size",
    repeat: bool = False,
):
    def _repeat(item, *, rep: int = 2):
        return (
            np.repeat(item, rep)
            if repeat and np.isscalar(item)
            else np.atleast_1d(item)
        )

    if size is None:
        if fwhm is None:
            raise ValueError(f"{name} is None; fwhm must be given.")
        values = _repeat(factor) * _repeat(fwhm)
        return values if repeat else float(values[0])
    values = _repeat(size)
    return values if repeat else float(values[0])
