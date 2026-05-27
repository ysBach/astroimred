"""Numerical parity checks for the astroapers aperture migration."""

import astroapers as aap
import numpy as np
import pytest
from astropy.nddata import CCDData
from numpy.testing import assert_allclose, assert_array_equal

from astroimred.phot.apphot import apphot_annulus, photometer

photutils_aperture = pytest.importorskip("photutils.aperture")


@pytest.fixture
def parity_image():
    y, x = np.indices((128, 128), dtype=np.float64)
    return 100.0 + 0.2 * x + 0.3 * y + 8.0 * np.sin(x / 7.0)


@pytest.mark.parametrize(
    ("astro_ap", "phot_ap", "rtol", "atol"),
    [
        (
            aap.CircAp([(32.4, 48.2), (80.1, 40.7)], r=5.3),
            photutils_aperture.CircularAperture([(32.4, 48.2), (80.1, 40.7)], r=5.3),
            2.0e-12,
            2.0e-9,
        ),
        (
            aap.EllipAp([(32.4, 48.2), (80.1, 40.7)], a=7.2, b=3.4, theta=0.31),
            photutils_aperture.EllipticalAperture(
                [(32.4, 48.2), (80.1, 40.7)], a=7.2, b=3.4, theta=0.31
            ),
            2.0e-12,
            2.0e-9,
        ),
        (
            aap.RectAp([(32.4, 48.2), (80.1, 40.7)], w=9.0, h=4.0, theta=0.23),
            photutils_aperture.RectangularAperture(
                [(32.4, 48.2), (80.1, 40.7)], w=9.0, h=4.0, theta=0.23
            ),
            2.0e-4,
            3.0,
        ),
    ],
)
def test_astroapers_apsum_matches_photutils(
    parity_image, astro_ap, phot_ap, rtol, atol
):
    measured = photometer(parity_image, astro_ap, method="exact")
    expected = photutils_aperture.aperture_photometry(
        parity_image, phot_ap, method="exact"
    )["aperture_sum"]

    assert_allclose(measured.apsum, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("astro_ap", "phot_ap", "rtol", "atol"),
    [
        (
            aap.CircAp([(32.4, 48.2), (80.1, 40.7)], r=5.3),
            photutils_aperture.CircularAperture([(32.4, 48.2), (80.1, 40.7)], r=5.3),
            2.0e-12,
            2.0e-9,
        ),
        (
            aap.EllipAp([(32.4, 48.2), (80.1, 40.7)], a=7.2, b=3.4, theta=0.31),
            photutils_aperture.EllipticalAperture(
                [(32.4, 48.2), (80.1, 40.7)], a=7.2, b=3.4, theta=0.31
            ),
            2.0e-12,
            2.0e-9,
        ),
        (
            aap.RectAp([(32.4, 48.2), (80.1, 40.7)], w=9.0, h=4.0, theta=0.23),
            photutils_aperture.RectangularAperture(
                [(32.4, 48.2), (80.1, 40.7)], w=9.0, h=4.0, theta=0.23
            ),
            2.0e-4,
            3.0,
        ),
    ],
)
def test_masked_photometer_matches_photutils(
    parity_image, astro_ap, phot_ap, rtol, atol
):
    bad = np.zeros_like(parity_image, dtype=bool)
    bad[46:53, 29:36] = True
    bad[38:45, 77:84] = True

    measured = photometer(parity_image, astro_ap, mask=bad, method="exact")
    expected = photutils_aperture.aperture_photometry(
        parity_image, phot_ap, mask=bad, method="exact"
    )["aperture_sum"]
    expected_npix = photutils_aperture.aperture_photometry(
        np.ones_like(parity_image), phot_ap, mask=bad, method="exact"
    )["aperture_sum"]
    total_npix = photutils_aperture.aperture_photometry(
        np.ones_like(parity_image), phot_ap, method="exact"
    )["aperture_sum"]

    assert_allclose(measured.apsum, expected, rtol=rtol, atol=atol)
    assert_allclose(measured.apsum_npix, expected_npix, rtol=rtol, atol=atol)
    assert_allclose(
        measured.apsum_npix + measured.nbadpix,
        total_npix,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "aperture",
    [
        aap.CircAn([(32.4, 48.2), (80.1, 40.7)], r_in=2.2, r_out=5.3),
        aap.EllipAn(
            [(32.4, 48.2), (80.1, 40.7)],
            a_in=2.1,
            b_in=1.0,
            a_out=7.2,
            b_out=3.4,
            theta_in=0.31,
        ),
        aap.EllipAn(
            [(32.4, 48.2), (80.1, 40.7)],
            a_in=2.1,
            b_in=1.0,
            a_out=7.2,
            b_out=3.4,
            theta_in=0.1,
            theta_out=0.4,
        ),
        aap.RectAn(
            [(32.4, 48.2), (80.1, 40.7)],
            w_in=2.1,
            h_in=1.0,
            w_out=7.2,
            h_out=3.4,
            theta_in=0.31,
        ),
        aap.PillAp([(32.4, 48.2), (80.1, 40.7)], w=5.0, a=1.8, b=1.1, theta=0.2),
        aap.PillAn(
            [(32.4, 48.2), (80.1, 40.7)],
            w_in=2.0,
            a_in=0.8,
            b_in=0.5,
            w_out=5.0,
            a_out=1.8,
            b_out=1.1,
            theta_in=0.2,
        ),
        aap.WedgeAp(
            [(32.4, 48.2), (80.1, 40.7)],
            r_in=2.0,
            r_out=7.0,
            theta_in=0.2,
            dtheta_in=0.7,
        ),
    ],
)
@pytest.mark.parametrize("method", ["exact", "center"])
def test_photometer_matches_astroapers_object_api(parity_image, aperture, method):
    bad = np.zeros_like(parity_image, dtype=bool)
    bad[46:53, 29:36] = True
    bad[38:45, 77:84] = True

    measured = photometer(parity_image, aperture, mask=bad, method=method)
    apsum = aperture.apsum_exact if method == "exact" else aperture.apsum_center
    expected_apsum, expected_npix = apsum(parity_image, mask=bad)
    _, total_npix = apsum(parity_image)

    assert_allclose(measured.apsum, expected_apsum)
    assert_allclose(measured.apsum_npix, expected_npix)
    assert_allclose(measured.apsum_npix + measured.nbadpix, total_npix)


def test_photometer_false_mask_has_zero_nbadpix(parity_image):
    bad = np.zeros_like(parity_image, dtype=bool)
    aperture = aap.CircAp([(32.4, 48.2), (80.1, 40.7)], r=5.3)

    measured = photometer(parity_image, aperture, mask=bad, method="exact")

    assert np.all(measured.nbadpix >= 0.0)
    assert_allclose(measured.nbadpix, 0.0, atol=1.0e-12)


def test_photometer_uses_astroapers_object_api(monkeypatch):
    data = np.ones((8, 8), dtype=float)
    error = np.full_like(data, 2.0)
    bad = np.zeros_like(data, dtype=bool)
    aperture = aap.CircAp([(2.0, 2.0), (5.0, 5.0)], r=1.5)
    calls = []

    def apsum_center(image, mask=None, *, return_npix=True):
        calls.append(("apsum_center", image.copy(), mask, return_npix))
        if np.all(image == 4.0):
            if not return_npix:
                return np.array([9.0, 16.0])
            return np.array([9.0, 16.0]), np.array([1.0, 2.0])
        return np.array([11.0, 22.0]), np.array([3.0, 4.0])

    def npix_center(shape, *, mask=None):
        calls.append(("npix_center", shape, mask))
        return np.array([5.0, 7.0])

    monkeypatch.setattr(aperture, "apsum_center", apsum_center)
    monkeypatch.setattr(aperture, "npix_center", npix_center)

    measured = photometer(data, aperture, error=error, mask=bad, method="center")

    assert_allclose(measured.apsum, [11.0, 22.0])
    assert_allclose(measured.apsum_npix, [3.0, 4.0])
    assert_allclose(measured.apsum_err, [3.0, 4.0])
    assert_allclose(measured.nbadpix, [2.0, 3.0])
    assert [call[0] for call in calls] == [
        "apsum_center",
        "npix_center",
        "apsum_center",
    ]
    assert calls[-1][3] is False


def test_sky_values_match_photutils_center_selection(parity_image):
    astro_an = aap.CircAn([(40.2, 41.5), (90.7, 70.1)], r_in=8, r_out=13)
    phot_an = photutils_aperture.CircularAnnulus(
        [(40.2, 41.5), (90.7, 70.1)], r_in=8, r_out=13
    )

    actual = astro_an.sampled_values(parity_image)
    expected_masks = phot_an.to_mask(method="center")
    expected = [mask.get_values(parity_image) for mask in expected_masks]

    for actual_values, expected_values in zip(actual, expected, strict=True):
        assert_array_equal(np.sort(actual_values), np.sort(expected_values))


@pytest.mark.parametrize("position", [(30.0, 31.0), (30.2, 31.3), (30.5, 31.5)])
@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_ellipan_sky_values_match_photutils_center_selection(
    parity_image, position, theta
):
    astro_an = aap.EllipAn(
        position,
        a_in=5,
        b_in=3,
        a_out=9,
        b_out=6,
        theta_in=theta,
    )
    phot_an = photutils_aperture.EllipticalAnnulus(
        position,
        a_in=5,
        b_in=3,
        a_out=9,
        b_out=6,
        theta=theta,
    )

    actual = np.sort(astro_an.sampled_values(parity_image)[0])
    expected = np.sort(phot_an.to_mask(method="center").get_values(parity_image))

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("position", [(10, 10), (10.5, 10.5), (10.2, 10.3)])
def test_circan_center_mask_matches_photutils(position):
    shape = (32, 32)
    astro_an = aap.CircAn(position, r_in=5, r_out=7)
    astro_mask = astro_an.bboxes()[0].to_image(astro_an.weights_center()[0], shape) > 0
    photutils_mask = (
        photutils_aperture.CircularAnnulus(position, r_in=5, r_out=7)
        .to_mask(method="center")
        .to_image(shape)
        .astype(bool)
    )

    assert_array_equal(astro_mask, photutils_mask)


def test_apphot_annulus_matches_photutils_reference(parity_image):
    ccd = CCDData(
        parity_image,
        unit="adu",
        header={"GAIN": 1.7, "RDNOISE": 4.2, "EXPTIME": 30.0},
    )
    astro_ap = aap.CircAp((64.3, 62.8), r=5.5)
    astro_an = aap.CircAn((64.3, 62.8), r_in=12, r_out=18)
    phot_ap = photutils_aperture.CircularAperture((64.3, 62.8), r=5.5)

    actual = apphot_annulus(ccd, astro_ap, astro_an, pandas=False)
    err = np.sqrt(parity_image / 1.7 + (4.2 / 1.7) ** 2)
    phot = photutils_aperture.aperture_photometry(
        parity_image, phot_ap, error=err, method="exact"
    )

    assert_allclose(actual["apsum"], phot["aperture_sum"], rtol=2e-12)
    assert_allclose(actual["apsum_err"], phot["aperture_sum_err"], rtol=2e-12)
    assert_allclose(
        actual["apsum_npix"], astro_ap.apsum_exact(parity_image)[1], rtol=2e-12
    )
