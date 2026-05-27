"""Tests for astroapers-backed aperture helpers."""

import astroapers as aap
import numpy as np
import pytest
from astropy.nddata import CCDData
from numpy.testing import assert_allclose

from astroimred.phot.aperture import (
    circ_ap_an,
    cutout_from_ap,
    ellip_ap_an,
    pill_ap_an,
    rect_ap_an,
)


def test_circ_ap_an_returns_astroapers_objects():
    ap, an = circ_ap_an((50, 50), r_ap=15, r_in=40, r_out=60)

    assert isinstance(ap, aap.CircAp)
    assert isinstance(an, aap.CircAn)
    assert_allclose(ap.r, 15.0)
    assert_allclose(an.r_in, 40.0)
    assert_allclose(an.r_out, 60.0)


def test_ellip_ap_an_returns_astroapers_objects():
    ap, an = ellip_ap_an(
        (50, 50),
        a_ap=10,
        b_ap=5,
        a_in=20,
        b_in=10,
        a_out=30,
        b_out=15,
        theta=np.pi / 4,
    )

    assert isinstance(ap, aap.EllipAp)
    assert isinstance(an, aap.EllipAn)
    assert_allclose((ap.a, ap.b, ap.theta), (10.0, 5.0, np.pi / 4))
    assert_allclose((an.a_in, an.b_in, an.a_out, an.b_out), (20.0, 10.0, 30.0, 15.0))
    assert_allclose((an.theta_in, an.theta_out), (np.pi / 4, np.pi / 4))


def test_rect_ap_an_returns_astroapers_objects():
    ap, an = rect_ap_an(
        (50, 50),
        w_ap=8,
        h_ap=4,
        w_in=12,
        h_in=6,
        w_out=20,
        h_out=10,
        theta=0.25,
    )

    assert isinstance(ap, aap.RectAp)
    assert isinstance(an, aap.RectAn)
    assert_allclose((ap.w, ap.h, ap.theta), (8.0, 4.0, 0.25))
    assert_allclose((an.w_in, an.h_in, an.w_out, an.h_out), (12.0, 6.0, 20.0, 10.0))
    assert_allclose((an.theta_in, an.theta_out), (0.25, 0.25))


def test_pill_ap_an_returns_astroapers_objects():
    ap, an = pill_ap_an(
        (50, 50),
        w_ap=12,
        a_ap=6,
        b_ap=3,
        w_in=12,
        a_in=8,
        b_in=4,
        w_out=18,
        a_out=12,
        b_out=6,
        theta=0.25,
    )

    assert isinstance(ap, aap.PillAp)
    assert isinstance(an, aap.PillAn)
    assert_allclose((ap.w, ap.a, ap.b, ap.theta), (12.0, 6.0, 3.0, 0.25))
    assert_allclose((an.w_in, an.a_in, an.b_in), (12.0, 8.0, 4.0))
    assert_allclose((an.w_out, an.a_out, an.b_out), (18.0, 12.0, 6.0))
    assert_allclose((an.theta_in, an.theta_out), (0.25, 0.25))


def test_cutout_from_ap_bbox_and_exact():
    data = CCDData(np.arange(10000, dtype=float).reshape(100, 100), unit="adu")
    ap = aap.CircAp((50, 50), r=5)

    bbox_cut = cutout_from_ap(ap, data, method="bbox")
    exact_cut = cutout_from_ap(ap, data, method="exact")

    assert bbox_cut.data.shape == ap.weights_center()[0].shape
    assert exact_cut.data.shape == ap.weights_exact()[0].shape
    assert np.nanmax(exact_cut.data) > 0


def test_cutout_from_ap_uses_astroapers_object_api(monkeypatch):
    data = CCDData(np.arange(100, dtype=float).reshape(10, 10), unit="adu")
    ap = aap.CircAp((5, 5), r=2)
    box = aap.BoundingBox(3, 7, 4, 7)

    def fail_cutout(*args, **kwargs):
        raise AssertionError("bbox cutouts should not materialize aperture cutouts")

    monkeypatch.setattr(ap, "sampled_cutout", fail_cutout)
    monkeypatch.setattr(ap, "weighted_cutout", fail_cutout)
    monkeypatch.setattr(ap, "bboxes", lambda: [box])

    cutout = cutout_from_ap(ap, data, method="bbox")

    assert cutout.data.shape == box.shape


def test_cutout_from_ap_rejects_unknown_method():
    ap = aap.CircAp((50, 50), r=5)
    data = CCDData(np.ones((100, 100)), unit="adu")

    with pytest.raises(ValueError):
        cutout_from_ap(ap, data, method="not-a-method")
