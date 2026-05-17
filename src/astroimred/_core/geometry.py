"""
Objects that are
(1) too fundamental, so used in various places,
(2) completely INDEPENDENT of all other modules of this package.
"""

import numpy as np
from astro_ndslice import listify as listify  # noqa: F401

_numba_available = False
try:
    import importlib.util as _ilu

    _numba_available = _ilu.find_spec("numba") is not None
except Exception:
    pass

__all__ = [
    "sigclip_dataerr",
    "circular_mask",
    "enclosing_circle_radius",
    "mask_enclosing_circle_radius",
    "bezel_mask",
]


# !FIXME: not finished
# TODO: add err_lower, err_upper, sigma_lower, sigma_upper
def sigclip_dataerr(
    val: np.ndarray,
    err: np.ndarray,
    cenfunc: str = "wvg",
    sigma: float = 3,
    maxiters: int = 3,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Sigma-clip values using per-point error estimates.

    Parameters
    ----------
    val, err : array-like
        Values and corresponding 1-sigma errors.

    cenfunc : {"wvg", "avg", "average", "mean"}, optional
        Center estimator. ``"wvg"`` uses weighted average.

    sigma : float, optional
        Rejection threshold in units of `err`.

    maxiters : int, optional
        Maximum clipping iterations.
    """
    if cenfunc == "wvg":
        from .._core.numeric import wvg

        def cenfunc(val, err):
            return wvg(val, err=err)
    elif cenfunc in ["avg", "average", "mean"]:

        def cenfunc(val, err):
            return np.mean(val)[0]  # err is dummy
    else:
        raise ValueError(f"{cenfunc=} is not implemented yet.")

    val = np.ma.array(val)
    val_clipped = val.compressed()
    err_clipped = err[val.mask]
    cen = cenfunc(val_clipped, err_clipped)

    for _i in range(maxiters):
        # calculate deviation for all (even masked) elements:
        deviation = np.abs(val.data - cen)
        mask = deviation > sigma * err

    return val, mask


def circular_mask(
    shape: tuple[int, ...],
    center: tuple | None = None,
    radius: float | None = None,
    center_xyz: bool = True,
) -> np.ndarray:
    """Creates an N-D circular (circular, spherical, ...) mask.

    Parameters
    ----------
    shape : `tuple`
        The pythonic shape, i.e., `arr.shape` (not xyz order).

    center : `tuple`, `None`, optional.
        The center of the circular mask. If `None` (default), the central
        position is used.
        Default: `None`.

    radius : `float`, `None`, optional.
        The radius of the mask. If `None`, the distance to the closest edge of
        the image is used.
        Default: `None`.

    center_xyz : `bool`, optional.
        Whether the center is in xyz order.
        Default: `True`.

    Notes
    -----
    Idea copied from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    Note that this is slow due to the "general" N-D nature of the mask.
    If you need an aperture-style 2-D mask, use ``astroapers.CircAp``.
    """
    if center is None:  # use the middle of the image
        center = [npix / 2 for npix in shape[::-1]]

    if center_xyz:
        center = center[::-1]

    shape = np.array(shape)
    center = np.array(center)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = np.min([center, shape - center])

    slices = tuple([slice(None, npix, None) for npix in shape])

    zyx = np.ogrid[slices]
    dist_sq = [((zyx[i] - center[i]) ** 2) for i in range(len(shape))]
    dist_from_center = np.sqrt(np.sum(np.array(dist_sq, dtype=object)))

    mask = dist_from_center <= radius
    return mask


def _enclosing_circle_radius_numpy(segm, center, segm_id, output):
    for i in range(len(segm_id)):
        _segm_id = segm_id[i]
        mask = segm == _segm_id
        y, x = np.nonzero(mask)
        rsq_max = np.max((x - center[i][0]) ** 2 + (y - center[i][1]) ** 2)
        output[i] = np.sqrt(rsq_max)


_enclosing_circle_radius_nb = None


def _get_enclosing_circle_radius_nb():
    global _enclosing_circle_radius_nb
    if _enclosing_circle_radius_nb is None:
        import numba as nb

        @nb.njit(fastmath=False, parallel=True)
        def _kernel(segm, center, segm_id, output):
            for i in nb.prange(len(segm_id)):
                _segm_id = segm_id[i]
                mask = segm == _segm_id
                y, x = np.nonzero(mask)
                rsq_max = np.max((x - center[i][0]) ** 2 + (y - center[i][1]) ** 2)
                output[i] = np.sqrt(rsq_max)

        _enclosing_circle_radius_nb = _kernel
    return _enclosing_circle_radius_nb


def enclosing_circle_radius(
    segm: np.ndarray,
    center: np.ndarray,
    segm_id: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the radius of the smallest enclosing circle for a given mask.

    Parameters
    ----------
    segm : 2D array-like
        The input segmentation map (binary image) where non-zero values are
        considered as the region of interest.

    center : 2-D array, optional
        The (x, y) coordinates of the center of the circles. If not provided,
        the center will be calculated as the centroid of the masked region.

    segm_id : `list` of `int`, optional
        The `list` of segmentation IDs to calculate the radius for. If not provided,
        it defaults to `[1]`, which is equivalent to `True` for binary masks.
        Default: `None`.

    Returns
    -------
    `~numpy.ndarray`
        The radius of the smallest enclosing circle.

    Notes
    -----
    Since it calculates distances from the center to the pixel center, one may
    want to add ~0.5 (or sqrt(2)*0.5) to enclose the full pixel area.

    By using numba, single segmentation radius finding is ~5 times faster than
    pure numpy, and it is boosted further if `parallel=True` is used.
    """

    if segm_id is None:
        segm_id = np.array([1], dtype=segm.dtype)  # same as `True`

    center = np.atleast_2d(center)
    if center.shape[1] != 2:
        raise ValueError("Center must be a 2D array with shape (N, 2)")

    radii = np.empty(len(segm_id), dtype=np.float64)

    if _numba_available:
        _get_enclosing_circle_radius_nb()(segm, center, segm_id, radii)
    else:
        _enclosing_circle_radius_numpy(segm, center, segm_id, radii)

    return radii


def mask_enclosing_circle_radius(
    mask: np.ndarray,
    center: tuple[float, float] | None = None,
) -> float:
    """Return the radius enclosing all non-zero pixels in a binary mask."""
    y, x = np.nonzero(mask)

    if len(x) == 0 or len(y) == 0:
        return 0.0

    if center is None:
        center = (float(np.mean(x)), float(np.mean(y)))

    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return float(np.max(distances))


def bezel_mask(
    xvals,
    yvals,
    nx,
    ny,
    bezel=(0, 0),
    bezel_x=None,
    bezel_y=None,
) -> np.ndarray:
    """Return mask for positions inside the image border bezel."""
    bezel = np.array(bezel)
    if len(bezel) == 1:
        bezel = np.repeat(bezel, 2)

    if bezel_x is None:
        bezel_x = bezel.copy()
    else:
        bezel_x = np.atleast_1d(bezel_x)
        if len(bezel_x) == 1:
            bezel_x = np.repeat(bezel_x, 2)

    if bezel_y is None:
        bezel_y = bezel.copy()
    else:
        bezel_y = np.atleast_1d(bezel_y)
        if len(bezel_y) == 1:
            bezel_y = np.repeat(bezel_y, 2)

    return (
        (xvals < bezel_x[0] + 0.5)
        | (yvals < bezel_y[0] + 0.5)
        | (xvals > (nx - bezel_x[1]) - 0.5)
        | (yvals > (ny - bezel_y[1]) - 0.5)
    )
