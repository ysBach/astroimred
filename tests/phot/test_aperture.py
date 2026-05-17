"""Tests for astroapers-backed aperture helpers."""

import astroapers as aap
import numpy as np
import pytest
from astropy.nddata import CCDData
from numpy.testing import assert_allclose

import astroimred.phot.aperture as phot_aperture
from astroimred.phot.aperture import (
    circ_ap_an,
    cutout_from_ap,
    ellip_ap_an,
    pill_ap_an,
)


def test_aperture_module_does_not_reexport_aap_classes():
    assert not hasattr(phot_aperture, "CircAp")
    assert not hasattr(phot_aperture, "PillBoxAperture")


def test_circ_ap_an_returns_astroapers_objects():
    ap, an = circ_ap_an((50, 50), fwhm=10, f_ap=1.5, f_in=4.0, f_out=6.0)

    assert isinstance(ap, aap.CircAp)
    assert isinstance(an, aap.CircAn)
    assert_allclose(ap.r, 15.0)
    assert_allclose(an.r_in, 40.0)
    assert_allclose(an.r_out, 60.0)
    assert_allclose(ap.area, np.pi * 15.0**2)
    assert_allclose(an.area, np.pi * (60.0**2 - 40.0**2))


def test_ellip_ap_an_returns_astroapers_objects():
    ap, an = ellip_ap_an(
        (50, 50),
        r_ap=(10, 5),
        r_in=(20, 10),
        r_out=(30, 15),
        theta=np.pi / 4,
    )

    assert isinstance(ap, aap.EllipAp)
    assert isinstance(an, aap.EllipAn)
    assert_allclose((ap.a, ap.b, ap.theta), (10.0, 5.0, np.pi / 4))
    assert_allclose((an.a_in, an.b_in), (20.0, 10.0))
    assert_allclose((an.a_out, an.b_out), (30.0, 15.0))


def test_pill_ap_an_returns_astroapers_objects():
    ap, an = pill_ap_an((50, 50), fwhm=4, trail=12, theta=0.25)

    assert isinstance(ap, aap.PillAp)
    assert isinstance(an, aap.PillAn)
    assert_allclose(ap.w, 12.0)
    assert_allclose(ap.theta, 0.25)


def test_cutout_from_ap_bbox_and_exact():
    data = CCDData(np.arange(10000, dtype=float).reshape(100, 100), unit="adu")
    ap = aap.CircAp((50, 50), r=5)

    bbox_cut = cutout_from_ap(ap, data, method="bbox")
    exact_cut = cutout_from_ap(ap, data, method="exact")

    assert bbox_cut.data.shape == ap.get_apmask("center").weights.shape
    assert exact_cut.data.shape == ap.get_apmask("exact").weights.shape
    assert np.nanmax(exact_cut.data) > 0


def test_cutout_from_ap_rejects_unknown_method():
    ap = aap.CircAp((50, 50), r=5)
    data = CCDData(np.ones((100, 100)), unit="adu")

    with pytest.raises(ValueError):
        cutout_from_ap(ap, data, method="not-a-method")
