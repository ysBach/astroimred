"""Radial profiles"""

from itertools import repeat

import astroapers as aap
import astroapers.kernels as aapk
import numpy as np
import pandas as pd
import reducers as rd
from astropy.nddata import CCDData

from .background import sky_fit
from .center import circular_bbox_cut, circular_slice

__all__ = [
    "moffat_r",
    "gauss_r",
    "bivt_r",
    "radial_profile",
    "radcum_profile",
    "ee_radius",
    "radprof_pix",
]


# def psfmeasure(img, pos, level=0.5, radius=5, maxiters=3, sky_buffer=5, sky_width=5,
#                saturation=None, **kwargs):
#     if isinstance(img, CCDData):
#         img = img.data
#     elif not isinstance(img, np.ndarray):
#         raise TypeError(f'img must be a CCDData or ndarray (now {type(img) = })')

#     pos_old = pos
#     crad = radius + sky_buffer + sky_width
#     for i in range(maxiters):
#         cut, dists, pos_cut = circular_cut(img, pos, radius=crad)


def gauss_r(
    r: float | np.ndarray,
    amp: float = 1,
    sig: float = 1,
    const: float = 0,
) -> float | np.ndarray:
    return amp * np.exp(-0.5 * (r / sig) ** 2) + const


def moffat_r(
    r: float | np.ndarray,
    amp: float = 1,
    core: float = 1,
    power: float = 2.5,
    const: float = 0,
) -> float | np.ndarray:
    return amp * (1 + (r / core) ** 2) ** (-power) + const


def bivt_r(r, amp=1, num2=1, sig=1, const=0):
    """Bivariate t distribution (generalized Moffat).

    Parameters
    ----------
    r : array-like
        Radial distance.
    amp : float, optional
        Amplitude.
        Default is ``1``.
    num2 : float, optional
        The degrees of freedom (nu) minus 2.
        Default is ``1``.
    sig : float, optional
        Scale parameter (sigma).
        Default is ``1``.
    const : float, optional
        Constant background.
        Default is ``0``.

    Returns
    -------
    float or array-like
        The bivariate t profile values.
    """
    return amp * (1 + (r**2 / (num2 * sig**2))) ** (-(num2 + 4) / 2) + const


def fwhm_r(popt, fun):
    if fun == "gauss":
        return 2 * np.sqrt(2 * np.log(2)) * popt[1]
    elif fun == "moffat":
        return 2 * popt[1] * np.sqrt(2 ** (1 / popt[2]) - 1)
    elif fun == "bivt":
        # 2*sig*np.sqrt((nu-2)*(2**(2/(nu+2)) - 1))
        return 2 * popt[2] * np.sqrt(popt[1] * (2 ** (2 / (popt[1] + 4)) - 1))
    else:
        raise ValueError(f"Unknown function: {fun}")


def radial_profile(
    im: CCDData | np.ndarray,
    center: tuple[float, float],
    radii: float | np.ndarray = 1,
    thickness: int | np.ndarray = 1,
    mask: np.ndarray | None = None,
    norm_by_center: bool = False,
    add_center: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, float]:
    """Calculate an annular radial profile of an image.

    Parameters
    ----------
    im : `~astropy.nddata.CCDData` or ndarray
        Image data.
    center : tuple of float
        Profile center in ``(x, y)`` order.
    radii : 1-D array
        Annulus centers in pixels. Each value is paired with the corresponding
        `thickness`.
    thickness : int or array of int, optional
        Annulus width(s) in pixels. Overlapping annuli are allowed. Default is
        `1`.
    mask : 2-D array, optional
        Boolean mask; `True` pixels are ignored. Default is `None`.
    norm_by_center : bool, optional
        Divide profile values and spreads by the absolute center pixel value
        when `True`. Default is `False`.
    add_center : bool, optional
        Prepend the center pixel as
        ``r=0, mpix=center value, spix=0, npix=1`` when `True`. Default is
        `False`.
    **kwargs :
        Additional keyword arguments for `sky_fit`.

    Returns
    -------
    profs : pandas.DataFrame
        Radial profile with columns ``r``, ``mpix``, ``spix``, ``dpix``,
        ``npix``, ``nrej``, and ``spix_n``. The first five are the radius,
        estimated value, sample spread, standard error of ``mpix`` (`sky_fit`
        ``dsky``), and used pixel support; ``nrej`` is the number rejected by
        `sky_fit`, and ``spix_n`` is ``spix / sqrt(npix)``, which equals
        ``dpix`` only for the mean estimator.
    center_val : float
        The value of the pixel at the center position.

    Notes
    -----
    Each annulus delegates its statistics to `sky_fit`; this function has no
    separate mode estimator. Visualization's `annular_normalize` instead takes
    nonoverlapping bin edges and transforms the original image.

    Examples
    -------
    >>> profs, center_val = radial_profile(im, center=(50, 50), radii=np.arange(1, 11), thickness=2)
    >>> profs
         r      mpix      spix  npix    spix_n
    0   0.0  1500.0000   0.0000     1   0.000000
    1   1.0   1450.1234  10.5678   12   3.045678
    2   2.0   1400.5678  15.6789   28   2.962345
    ...
    >>> profs["spix_n"] = profs["spix"] / np.sqrt(profs["npix"])
    >>> plt.errorbar(profs['r'], profs['mpix'], yerr=profs['spix_n'], fmt='o', color="k", capsize=3)
    >>> plt.errorbar(profs['r'], profs['mpix'], yerr=profs['spix'], fmt='', color="k", capsize=0, elinewidth=0.5)
    """
    radii = np.asarray(radii).ravel()
    if np.isscalar(thickness):
        thickness = repeat(thickness, len(radii))
    else:
        thickness = np.asarray(thickness).ravel()

    center_val = im[
        *(np.round(center).astype(int)[::-1])
    ]  # reverse for (y, x) indexing

    if add_center:
        # use original names (names for sky_fit before renaming)
        profs = [
            {"r": 0, "msky": center_val, "ssky": 0, "dsky": 0, "nsky": 1, "nrej": 0}
        ]
    else:
        profs = []
    for r, _thick in zip(radii, thickness, strict=False):
        an = aap.CircAn(center, r_in=max(0.01, r - _thick / 2), r_out=r + _thick / 2)
        _skyfit = sky_fit(im, an, mask=mask, to_table=False, **kwargs)[0]
        _skyfit["r"] = r
        profs.append(_skyfit)

    profs = pd.DataFrame.from_dict(profs)
    profs = profs.rename(
        columns={"msky": "mpix", "ssky": "spix", "dsky": "dpix", "nsky": "npix"}
    )
    profs["spix_n"] = profs["spix"] / np.sqrt(profs["npix"])

    if norm_by_center:
        _cval = np.abs(center_val)
        profs["mpix"] /= _cval
        profs["spix"] /= _cval
        profs["spix_n"] /= _cval
    return profs, center_val


def radcum_profile(
    im,
    center,
    radii=1,
    mask=None,
    var=None,
    err=None,
    return_var=False,
    add_npix=True,
    norm_by_last=False,
):
    """Calculate a cumulative radial profile with circular apertures.

    Parameters
    ----------
    im : 2-D array
        Image data.
    center : tuple of float
        Aperture center in ``(x, y)`` order.
    radii : 1-D array
        Radii of the circular apertures.
    mask : 2-D array, optional
        Boolean mask; `True` pixels are ignored. Default is `None`.
    var : 2D array, optional
        Variance map. Its aperture sum is also computed. Mutually exclusive
        with `err`. Default is `None`.
    err : 2D array, optional
        Error map, converted internally to variance (``variance = error**2``).
        Mutually exclusive with `var`. Default is `None`.
    return_var : bool, optional
        If `True`, return the aperture sum as ``var`` without taking a square
        root. If `False` (default), return its square root as ``err``. Requires
        either `var` or `err`.
    add_npix : bool, optional
        Include ``npix`` (weighted unmasked aperture support) when `True`
        (default). Set `False` to skip the `to_mask` computation when counts
        are not needed.
    norm_by_last : bool, optional
        Normalize all flux columns by the value at the largest radius when
        `True`. Default is `False`.

    Returns
    -------
    profs : pandas.DataFrame
        DataFrame with these columns:

        - ``r``    : aperture radius
        - ``apsum``: aperture sum within radius r
        - ``npix`` : weighted unmasked aperture support (if ``add_npix=True``)
        - ``var``  : variance of ``apsum``, ``sum(w**2 * var)`` with exact
          overlap weights ``w`` (if ``return_var=True`` and `var`/`err` is given)
        - ``err``  : square root of ``var`` (if ``return_var=False`` and
          `var`/`err` is given)
    """
    if var is not None and err is not None:
        raise ValueError("Provide either `var` or `err`, not both.")

    var_map = None
    if err is not None:
        var_map = np.asarray(err) ** 2
    elif var is not None:
        var_map = np.asarray(var)

    radii = np.asarray(radii).ravel()
    x, y = center
    _mask_arr = np.asarray(mask, dtype=bool) if mask is not None else None

    if add_npix:
        ap_results = [aapk.apsum_circ_exact(im, x, y, r) for r in radii]
        if _mask_arr is not None:
            ap_results = [
                aapk.apsum_circ_exact(im, x, y, r, mask=_mask_arr) for r in radii
            ]
        apsums = np.array(
            [np.asarray(result[0]).reshape(-1)[0] for result in ap_results]
        )
        npixs = np.array(
            [np.asarray(result[1]).reshape(-1)[0] for result in ap_results]
        )
    else:
        apsums = np.array(
            [
                aapk.apsum_circ_exact(
                    im, x, y, r, mask=_mask_arr, return_npix=False
                ).reshape(-1)[0]
                for r in radii
            ]
        )
        npixs = None

    prof = {"r": radii, "apsum": apsums}

    if add_npix:
        prof["npix"] = npixs

    unc_vals = None
    unc_col = None
    if var_map is not None:
        # Var(sum(w * im)) = sum(w**2 * var) for independent pixel noise.
        var_vals = np.empty(radii.size, dtype=np.float64)
        for idx, r in enumerate(radii):
            ap = aap.CircAp((x, y), r=r)
            (w,) = ap.weights_exact()
            (bbox,) = ap.bboxes()
            var_vals[idx] = bbox.apsum(
                np.square(w), var_map, mask=_mask_arr, return_npix=False
            )
        if return_var:
            unc_vals, unc_col = var_vals, "var"
        else:
            unc_vals, unc_col = np.sqrt(var_vals), "err"

    if norm_by_last:
        _norm = np.abs(apsums[-1])
        apsums /= _norm
        if unc_vals is not None:
            unc_vals /= _norm**2 if return_var else _norm

    if unc_vals is not None:
        prof[unc_col] = unc_vals

    return pd.DataFrame(prof)


def ee_radius(im, center, fraction=0.5, r_min=0.5, r_max=None, sum_is_unity=False):
    """Find the radius enclosing a fraction of the total image flux.

    Parameters
    ----------
    im : 2-D array
        PSF or flux image. Total flux must be positive and contain no NaNs.
    center : tuple of float
        Center position in ``(x, y)`` order.
    fraction : float, optional
        The enclosed flux fraction to find the radius for.
        Default is ``0.5``.
    r_min : float, optional
        Lower bound for the search radius in pixels.
        Default is ``0.5``.
    r_max : float, optional
        Upper bound for the search radius. If `None`, defaults to half
        the minimum image dimension.
        Default is `None`.
    sum_is_unity : bool, optional
        If `True`, treat `fraction` as the target flux and assume `im.sum() == 1`.
        If `False` (default), use `fraction * np.sum(im)` as the target.

    Returns
    -------
    r : float
        The radius (in pixels) encircling the given flux fraction.
    """
    target = fraction if sum_is_unity else fraction * np.sum(im)

    if r_max is None:
        r_max = min(im.shape) / 2.0

    x, y = center

    def _residual(r):
        apsum = aapk.apsum_circ_exact(im, x, y, r, return_npix=False)
        return float(apsum.reshape(-1)[0]) - target

    from scipy.optimize import brentq

    return brentq(_residual, r_min, r_max)


def radprof_pix(img, pos, mask=None, rmax=10, sort_dist=False, fitfunc=None, refit=1):
    """Get pixel values and optional fits for an object's radial profile.

    Parameters
    ----------
    img : `~astropy.nddata.CCDData` or ndarray
        Image to profile.
    pos : array-like
        Object center in zero-based ``(x, y)`` coordinates.
    mask : array-like or bool, optional
        Pixels to exclude. It may match the full image or broadcast to the
        local cutout. Pass ``mask=ccd.mask`` explicitly for a CCDData mask.
    rmax : int, optional
        Maximum profile radius. Default is `10`.
    refit : float, None, optional
        If not `None`, refit within ``max(refit * FWHM, 3)``. Ignore the
        refit when this radius exceeds `rmax`. Default is `1`.

    Example
    -------
    >>> import astroimred.phot as fip
    >>> _r, _i, _f, popt, fwhm = fip.radprof_pix(ccd, pos=ap_t.positions, fit="gauss")
    >>> rr = np.arange(0, _r.max()+1, 0.1)
    >>> axs.plot(_r, _i, "s", mfc="none")
    >>> axs.plot(rr, _f(rr, *popt), "-")
    >>> axs.axvline(fwhm/2)
    >>> axs.axhline(popt[-1])
    >>> axs.axhline(popt[0] + popt[-1])
    """

    if isinstance(img, CCDData):
        img = img.data
    elif not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a CCDData or ndarray (now {type(img) = })")

    cut, _, _, dists = circular_bbox_cut(img, pos, radius=rmax, return_dists=True)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == img.shape:
            mask = mask[circular_slice(img.shape, pos, rmax)]
        mask = (dists > rmax) | np.broadcast_to(mask, cut.shape)
    else:
        mask = dists > rmax

    # cut = Cutout2D(img, pos, 2*rmax + 1).data
    # pos_cut = cut.
    # mask_c, dists = circular_mask(img.shape, pos, radius=rmax)
    # mask = (mask_c) if mask is None else (mask_c | mask)
    # mask = Cutout2D(mask, pos, 2*rmax + 1).data
    # dists = Cutout2D(dists, pos, 2*rmax + 1).data
    # cut = img[mask]

    if sort_dist:
        sort_idx = np.argsort(dists[~mask])
        return dists[~mask][sort_idx], cut[~mask][sort_idx]

    if fitfunc is not None:
        from scipy.optimize import curve_fit

        _r = dists[~mask]
        _i = cut[~mask]
        _imin, _imax = rd.minmax(_i)
        _rmin = rd.min(_r)
        if fitfunc == "gauss":
            fitter = gauss_r
            p0 = [_i[_r == _rmin][0] - _imin, max(1, rmax / 6), _imin]
            bounds = np.array([[0, 0, _imin - 1], [np.inf, rmax / 2, _imax + 1]])
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        elif fitfunc == "moffat":
            fitter = moffat_r
            p0 = [_i[_r == _rmin][0] - _imin, 1, 2.5, _imin]
            bounds = np.array(
                [[0, 0.1, 1.0, _imin - 1], [np.inf, rmax, np.inf, _imax + 1]]
            )
        elif fitfunc == "bivt":
            fitter = bivt_r
            p0 = [_i[_r == _rmin][0] - _imin, 1, max(1, rmax / 6), _imin]
            bounds = np.array(
                [[0, -2, 1.0e-10, _imin - 1], [np.inf, 15, 2 * rmax, _imax + 1]]
            )
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        else:
            raise ValueError(f"Unknown function: {fitfunc}")

        popt, _ = curve_fit(fitter, _r, _i, p0=p0, bounds=bounds)
        if refit is not None:
            fitrad = max(refit * fwhm_r(popt, fun=fitfunc), 3)
            if fitrad < rmax:
                mask = _r > fitrad
                # upper bound of sky as the profile at fitrad
                bounds[1][-1] = fitter(fitrad, *popt)
                # tighten others (amp, sigma, etc)
                bounds[0][:-1] = popt[:-1] * 0.5
                bounds[1][:-1] = popt[:-1] * 2

                popt, _ = curve_fit(
                    fitter, _r[~mask], _i[~mask], p0=popt, bounds=bounds
                )

        return _r, _i, fitter, popt, fwhm_r(popt, fun=fitfunc)

    return dists[~mask], cut[~mask]

    # # --- 1. mask out only the central region for quick initial fitting
    # _small_mask = (_r < max(rmax/4, 5))
    # # assuming the user gave ~ 2x FWHM, and we want ~ 0.5xFWHM
    # _popt, _ = curve_fit(fitter, _r[_small_mask], _i[_small_mask], p0=p0)
    # # --- 2. Use that as the initial input for the full fitting
