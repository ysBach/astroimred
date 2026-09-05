"""Astropy cutout helpers for astroapers apertures."""

from __future__ import annotations

from typing import Any

import astroapers as aap
import numpy as np
from astropy.nddata import CCDData, Cutout2D

__all__ = [
    "circ_ap_an",
    "cutout_from_ap",
    "ellip_ap_an",
    "pill_ap_an",
    "rect_ap_an",
]


def _as_radians(theta: Any) -> float:
    """Return ``theta`` as a float in radians."""
    if hasattr(theta, "to_value"):
        from astropy import units as u

        return float(theta.to_value(u.rad))
    return float(theta)


def cutout_from_ap(
    ap: Any,
    ccd: CCDData | np.ndarray,
    method: str = "bbox",
    fill_value: float = np.nan,
) -> Cutout2D | list[Cutout2D]:
    """Return `~astropy.nddata.Cutout2D` objects from aperture bounding boxes.

    Parameters
    ----------
    ap : astroapers aperture object
        The aperture object, such as `~astroapers.CircAp` or `~astroapers.EllipAp`.
    ccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        The CCD data.
    method : str, optional
        The method to use for cutout generation. Default is "bbox".
    fill_value : float, optional
        The value to use for filling the cutout. Default is `numpy.nan`.

    Returns
    -------
    `~astropy.nddata.Cutout2D` or list of `~astropy.nddata.Cutout2D`
        Cutouts trimmed to the image boundary for every method, with matching
        data, coordinates, and WCS when supplied by `ccd`.
    """
    data = ccd.data if isinstance(ccd, CCDData) else np.asarray(ccd)
    wcs = getattr(ccd, "wcs", None) if isinstance(ccd, CCDData) else None
    if method not in {"bbox", "center", "exact"}:
        raise ValueError(f"Unsupported aperture method: {method!r}")
    boxes = ap.bboxes()
    cutout_data = None
    if method == "center":
        cutout_data = ap.sampled_cutout(data, fill_value=fill_value)
    elif method == "exact":
        cutout_data = ap.weighted_cutout(data, fill_value=fill_value)
    cuts = []
    for idx, (pos, box) in enumerate(zip(ap.positions, boxes, strict=True)):
        cut = Cutout2D(data, position=pos, size=box.shape, mode="trim", wcs=wcs)
        if method != "bbox":
            sl_cut = box.overlap_slices(data.shape)[1]
            cut.data = cutout_data[idx][sl_cut]
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
    theta_rad = _as_radians(theta)
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


def rect_ap_an(
    positions,
    w_ap: float,
    h_ap: float,
    w_in: float,
    h_in: float,
    w_out: float,
    h_out: float,
    theta: float = 0.0,
) -> tuple[aap.RectAp, aap.RectAn]:
    """Return rectangular aperture and annulus objects."""
    theta_rad = _as_radians(theta)
    ap = aap.RectAp(positions, w=w_ap, h=h_ap, theta=theta_rad)
    an = aap.RectAn(
        positions,
        w_in=w_in,
        h_in=h_in,
        w_out=w_out,
        h_out=h_out,
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
    theta_rad = _as_radians(theta)

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
