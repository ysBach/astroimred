import numpy as np
from astropy.nddata import CCDData
from scipy.ndimage import median_filter

__all__ = ["smooth_med"]


def smooth_med(
    ccd: CCDData,
    cadd: float | np.ndarray = 1.0e-10,
    size: int | tuple[int, ...] = 5,
    footprint: np.ndarray | None = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> np.ndarray:
    """Smooth a CCD image with `scipy.ndimage.median_filter`.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Input CCD.

    cadd : `float`, `~numpy.ndarray`, optional.
        A very small const to be added to the input array to avoid 0-valued
        pixel after median filtering. This is to avoid the problem when doing
        ``image/|median_filtered|``.

    size, footprint, mode, cval, origin : optional.
        The parameters to obtain the median-filtered map. See
        `~scipy.ndimage.median_filter`.

    Returns
    -------
    ndarray
        Median-filtered data array.
    """
    return median_filter(
        ccd.data.copy() + cadd,
        size=size,
        footprint=footprint,
        mode=mode,
        cval=cval,
        origin=origin,
    )
