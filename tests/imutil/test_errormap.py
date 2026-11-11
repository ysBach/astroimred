"""Analytical detector-noise propagation tests."""

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from astroimred.imutil import imstat


@pytest.fixture(params=[False, True])
def error_backend(request, monkeypatch):
    """Exercise both actual implementations of the same variance model."""
    if request.param:
        pytest.importorskip("numexpr")
    monkeypatch.setattr(imstat, "HAS_NE", request.param)


@pytest.mark.parametrize("flat_err, expected", [(0.0, 25.0), (0.2, 50.0)])
def test_preflat_signal_propagates_through_nonunit_flat(
    error_backend, flat_err, expected
):
    variance = imstat.errormap(
        np.array([[100.0]]), flat=2.0, flat_err=flat_err, return_variance=True
    )
    np.testing.assert_allclose(variance, [[expected]])


def test_independent_noise_terms_have_consistent_output_units(error_backend):
    # (100+20)/(2*2**2) + (3/2)**2 + (100*.2/2**2)**2 + (4/(2*2))**2
    kw = {
        "gain_epadu": 2.0,
        "rdnoise_electron": 4.0,
        "subtracted_dark": 20.0,
        "flat": 2.0,
        "dark_std": 3.0,
        "flat_err": 0.2,
    }
    variance = imstat.errormap(np.array([[100.0]]), return_variance=True, **kw)
    error = imstat.errormap(np.array([[100.0]]), **kw)
    np.testing.assert_allclose(variance, [[43.25]])
    np.testing.assert_allclose(error, np.sqrt([[43.25]]))


def test_dark_error_floor_does_not_modify_callers_array(error_backend):
    dark_std = np.array([[-1.0, 5.0]])
    dark_std.setflags(write=False)
    variance = imstat.errormap(
        np.zeros((1, 2)), dark_std=dark_std, dark_std_min=2.0, return_variance=True
    )
    np.testing.assert_array_equal(dark_std, [[-1.0, 5.0]])
    np.testing.assert_allclose(variance, [[4.0, 25.0]])


def test_no_dark_calibration_error_does_not_add_second_read_noise(error_backend):
    variance = imstat.errormap(
        np.zeros((1, 1)), rdnoise_electron=4.0, gain_epadu=2.0, return_variance=True
    )
    np.testing.assert_allclose(variance, [[4.0]])


def test_dark_restoration_precedes_poisson_clipping(error_backend):
    # Negative dark-subtracted signal still had positive source+dark counts.
    data = np.array([[-10.0, -30.0]])
    variance = imstat.errormap(
        data, subtracted_dark=20.0, flat=2.0, flat_err=0.2, return_variance=True
    )
    np.testing.assert_allclose(variance, [[2.75, 2.25]])
    np.testing.assert_array_equal(data, [[-10.0, -30.0]])


def test_integer_signal_does_not_overflow_flat_variance(error_backend):
    variance = imstat.errormap(
        np.array([[50000]], dtype=np.int32),
        flat=2.0,
        flat_err=0.2,
        return_variance=True,
    )
    np.testing.assert_allclose(variance, [[6262500.0]])


def test_unit_flat_recovers_detector_variance(error_backend):
    variance = imstat.errormap(
        np.array([[100.0]]),
        gain_epadu=2 * u.electron / u.adu,
        rdnoise_electron=4 * u.electron,
        return_variance=True,
    )
    np.testing.assert_allclose(variance, [[54.0]])


def test_fits_input_and_broadcast_flat(error_backend, tmp_path):
    path = tmp_path / "signal.fits"
    fits.writeto(path, np.full((2, 2), 100, dtype=np.int32))
    variance = imstat.errormap(
        path, flat=np.array([1.0, 2.0]), flat_err=0.2, return_variance=True
    )
    np.testing.assert_allclose(variance, [[500.0, 50.0], [500.0, 50.0]])
