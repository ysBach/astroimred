"""Aperture helpers backed by astroapers."""

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
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[aap.CircAp, aap.CircAn]:
    """Return circular aperture and annulus objects."""
    return aap.CircAp(positions, r=r_ap), aap.CircAn(positions, r_in=r_in, r_out=r_out)


def ellip_ap_an(
    positions,
    a_ap: float,
    b_ap: float,
    a_in: float,
    b_in: float,
    a_out: float,
    b_out: float,
    theta: float = 0.0,
) -> tuple[aap.EllipAp, aap.EllipAn]:
    """Return elliptical aperture and annulus objects."""
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
    w_ap: float,
    a_ap: float,
    b_ap: float,
    w_in: float,
    a_in: float,
    b_in: float,
    w_out: float,
    a_out: float,
    b_out: float,
    theta: float = 0.0,
) -> tuple[aap.PillAp, aap.PillAn]:
    """Return pill aperture and annulus objects."""
    theta_rad = as_radians(theta)

    ap = aap.PillAp(positions, w=w_ap, a=a_ap, b=b_ap, theta=theta_rad)
    an = aap.PillAn(
        positions,
        w_in=w_in,
        a_in=a_in,
        b_in=b_in,
        w_out=w_out,
        a_out=a_out,
        b_out=b_out,
        theta_in=theta_rad,
    )
    return ap, an
