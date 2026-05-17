"""Small astroapers helpers used by photometry routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astroapers as aap
import numpy as np


@dataclass(frozen=True)
class PhotometryResult:
    """Aperture photometry arrays."""

    positions: np.ndarray
    apsum: np.ndarray
    apsum_err: np.ndarray | None
    apsum_npix: np.ndarray
    nbadpix: np.ndarray


def as_radians(theta: Any) -> float:
    """Return ``theta`` as a float in radians."""
    if hasattr(theta, "to_value"):
        from astropy import units as u

        return float(theta.to_value(u.rad))
    return float(theta)


def normalize_apertures(apertures: Any) -> list[Any]:
    """Return one or more astroapers aperture objects."""
    if isinstance(apertures, np.ndarray):
        return list(apertures.ravel())
    if isinstance(apertures, (list, tuple)):
        return list(np.asarray(apertures, dtype=object).ravel())
    return [apertures]


def normalize_positions(aperture: Any) -> np.ndarray:
    """Return aperture positions as an ``(N, 2)`` float array."""
    positions = np.asarray(aperture.positions, dtype=np.float64)
    return positions.reshape(1, 2) if positions.ndim == 1 else positions


def mask_list(aperture: Any, method: str = "exact") -> list[aap.ApMask]:
    """Return aperture masks as a list regardless of scalar/vector geometry."""
    masks = aperture.get_apmask(method=method)
    return masks if isinstance(masks, list) else [masks]


def photometer(
    data: np.ndarray,
    apertures: Any,
    *,
    error: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    method: str = "exact",
) -> PhotometryResult:
    """Measure aperture sums, errors, used aperture support, and masked support."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("data must be a 2-D array.")
    err = None if error is None else np.asarray(error)
    if err is not None and err.shape != arr.shape:
        raise ValueError("error must have the same shape as data.")
    bad = None if mask is None else np.asarray(mask, dtype=bool)
    if bad is not None and bad.shape != arr.shape:
        raise ValueError("mask must have the same shape as data.")

    positions: list[np.ndarray] = []
    sums: list[float] = []
    errs: list[float] = []
    apsum_npixs: list[float] = []
    nbadpix: list[float] = []

    for aperture in normalize_apertures(apertures):
        ap_positions = normalize_positions(aperture)
        ap_masks = mask_list(aperture, method=method)
        if len(ap_positions) != len(ap_masks):
            raise ValueError("aperture positions and masks have inconsistent lengths.")
        for pos, apmask in zip(ap_positions, ap_masks, strict=True):
            apsum, apsum_npix = apmask.apsum(arr, mask=bad)
            sums.append(float(apsum))
            apsum_npixs.append(float(apsum_npix))
            positions.append(pos)
            nbadpix.append(_weighted_bad_pixels(apmask, bad, arr.shape))
            if err is not None:
                errs.append(_weighted_error(apmask, err, mask=bad))

    return PhotometryResult(
        positions=np.asarray(positions, dtype=np.float64),
        apsum=np.asarray(sums, dtype=np.float64),
        apsum_err=None if err is None else np.asarray(errs, dtype=np.float64),
        apsum_npix=np.asarray(apsum_npixs, dtype=np.float64),
        nbadpix=np.asarray(nbadpix, dtype=np.float64),
    )


def center_values(
    data: np.ndarray,
    aperture: Any,
    *,
    mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return unweighted data values whose pixel centers fall in an aperture."""
    arr = np.asarray(data)
    bad = None if mask is None else np.asarray(mask, dtype=bool)
    values = []
    for apmask in mask_list(aperture, method="center"):
        values.append(apmask.weighted_values(arr, mask=bad))
    return values


def _weighted_error(
    apmask: aap.ApMask,
    error: np.ndarray,
    *,
    mask: np.ndarray | None,
) -> float:
    overlap = apmask.bbox.overlap_slices(error.shape)
    if overlap is None:
        return 0.0
    image_slices, mask_slices = overlap
    weights = np.array(apmask.weights[mask_slices], dtype=np.float64, copy=True)
    if mask is not None:
        weights[np.asarray(mask, dtype=bool)[image_slices]] = 0.0
    return float(np.sqrt(np.sum(error[image_slices] ** 2 * weights)))


def _weighted_bad_pixels(
    apmask: aap.ApMask,
    mask: np.ndarray | None,
    data_shape: tuple[int, int],
) -> float:
    if mask is None:
        return 0.0
    overlap = apmask.bbox.overlap_slices(data_shape)
    if overlap is None:
        return 0.0
    image_slices, mask_slices = overlap
    weights = apmask.weights[mask_slices]
    bad = np.asarray(mask, dtype=bool)[image_slices]
    return float(np.sum(weights[bad]))
