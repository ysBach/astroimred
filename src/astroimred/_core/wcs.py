"""WCS orientation helpers shared by astroimred subpackages."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

__all__ = ["is_eofn_ccw", "pa2pixtheta"]


def is_eofn_ccw(
    wcs: WCS,
    full: bool = False,
    tol: float = 5.0,
) -> bool | tuple[bool, float, float]:
    """Checks whether the East of North is counter-clockwise in the image.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS object.
    full : bool, optional
        If `True`, return the PA of x- and y-axes.
    tol : float, optional
        The tolerance in degrees for the difference of the two PA.
    """
    center = np.array(wcs._naxis) / 2
    coo = SkyCoord(*wcs.wcs_pix2world(*center, 0), unit="deg")
    plusx = wcs.wcs_pix2world(
        *(center + np.array((1, 0))), 0
    )  # basically (CD1_1, CD1_2)
    plusy = wcs.wcs_pix2world(
        *(center + np.array((0, 1))), 0
    )  # basically (CD2_1, CD2_2)
    pa_x = coo.position_angle(SkyCoord(plusx[0], plusx[1], unit="deg")).to_value(u.deg)
    pa_y = coo.position_angle(SkyCoord(plusy[0], plusy[1], unit="deg")).to_value(u.deg)
    dpa = pa_y - pa_x
    if (-270 - tol <= dpa <= -270 + tol) or (90 - tol <= dpa <= 90 + tol):
        # PA (East of North) is CCW in XY coordinate
        if full:
            return True, pa_x, pa_y
        return True
    elif (270 - tol <= dpa <= 270 + tol) or (-90 - tol <= dpa <= -90 + tol):
        # PA (East of North) is CW in XY coordinate
        if full:
            return False, pa_x, pa_y
        return False
    else:
        raise ValueError("PA calculation is problematic.")


def pa2pixtheta(
    pa: float,
    wcs: WCS,
    location: str | tuple[float, float] = "crpix",
    step_pix: float = 0.1,
) -> float:
    """
    pa : float
        The position angle in degrees, East of North.
    wcs : `~astropy.wcs.WCS`
        The WCS object.
    location : tuple or str, optional
        The location to convert the position angle. If ``"crpix"``, the
        location is the CRPIX of the WCS. If ``"center"``, the position angle
        is converted at the center of the image. Otherwise, it should be a
        tuple of ``(x, y)`` pixel coordinates.
    step_pix : float, optional
        The step in pixel unit to calculate the Jacobian of the WCS. It should
        be small enough to approximate the local linearity of the WCS, but not
        too small to cause numerical issues. Default is ``0.1`` pixel.

    Return
    ------
    theta: float
        The rotation angle in degrees from the positive ``x`` axis.  The
        angle increases counterclockwise.
    """
    if location == "crpix":
        try:
            location = np.array((wcs.wcs.crpix[0] - 1, wcs.wcs.crpix[1] - 1))
            # coo = SkyCoord(*wcs.wcs.crval, unit="deg")
        except AttributeError as err:
            raise AttributeError(
                "The WCS object does not have CRPIX and/or CRVAL. "
                + "Try with, e.g., `location`='center'."
            ) from err
    elif location == "center":
        location = np.array(wcs._naxis) / 2
        # coo = SkyCoord(*wcs.wcs_pix2world(*location, 0), unit="deg")
    else:
        location = np.array(location)
        # coo = SkyCoord(*wcs.wcs_pix2world(*location, 0), unit="deg")

    x, y = location

    # base world coord
    ra0, dec0 = wcs.all_pix2world(x, y, 0)

    # move slightly in pixel space
    ra_dx, dec_dx = wcs.all_pix2world(x + step_pix, y, 0)
    ra_dy, dec_dy = wcs.all_pix2world(x, y + step_pix, 0)

    # build Jacobian (world per pixel)
    dra_dx = (ra_dx - ra0) / step_pix
    ddec_dx = (dec_dx - dec0) / step_pix
    dra_dy = (ra_dy - ra0) / step_pix
    ddec_dy = (dec_dy - dec0) / step_pix

    # desired sky direction (unit vector in RA/Dec coords)
    pa_rad = np.deg2rad(pa)
    v_sky = np.array(
        [
            np.sin(pa_rad) / np.cos(np.deg2rad(dec0)),  # dRA corrected
            np.cos(pa_rad),  # dDec
        ]
    )

    # invert Jacobian: world -> pixel
    jacob = np.array([[dra_dx, dra_dy], [ddec_dx, ddec_dy]])
    v_pix = np.linalg.solve(jacob, v_sky)
    return np.degrees(np.arctan2(v_pix[1], v_pix[0]))
