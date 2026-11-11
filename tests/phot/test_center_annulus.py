"""Check Gaussian centering with annuli that follow each fitted position."""

from typing import Any

import astroapers as aap
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astroimred.phot import center


@pytest.mark.parametrize(
    ("annulus_type", "geometry"),
    [
        pytest.param(aap.CircAn, {"r_in": 7.0, "r_out": 9.0}, id="circular"),
        pytest.param(
            aap.EllipAn,
            {
                "a_in": 7.0,
                "b_in": 6.0,
                "a_out": 10.0,
                "b_out": 9.0,
                "theta_in": 0.2,
                "theta_out": 0.45,
            },
            id="elliptical",
        ),
        pytest.param(
            aap.RectAn,
            {
                "w_in": 14.0,
                "h_in": 12.0,
                "w_out": 20.0,
                "h_out": 18.0,
                "theta_in": 0.2,
                "theta_out": 0.45,
            },
            id="rectangular",
        ),
        pytest.param(
            aap.PillAn,
            {
                "w_in": 3.0,
                "a_in": 6.0,
                "b_in": 5.0,
                "w_out": 5.0,
                "a_out": 8.0,
                "b_out": 7.0,
                "theta_in": 0.2,
                "theta_out": 0.45,
                "plot_samples": 48,
            },
            id="pill",
        ),
    ],
)
def test_gaussian_center_repositions_annulus_without_changing_input(
    annulus_type: type[aap.CircAn | aap.EllipAn | aap.RectAn | aap.PillAn],
    geometry: dict[str, float | int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow the fitted center while preserving the caller's annulus geometry."""
    y, x = np.mgrid[:31, :31]
    expected_center = np.array([15.2, 14.7])
    image = 10.0 + 1000.0 * np.exp(
        -((x - expected_center[0]) ** 2 + (y - expected_center[1]) ** 2)
        / (2.0 * 1.5**2)
    )
    annulus = annulus_type((4.0, 5.0), **geometry)
    original_positions = annulus.positions.copy()
    measured_centers = []
    original_sky_fit = center.sky_fit

    def record_sky_fit(*args: Any, **kwargs: Any) -> Any:
        measured_annulus = kwargs["annulus"]
        assert type(measured_annulus) is annulus_type
        measured_centers.append(measured_annulus.positions[0].copy())
        for name, value in geometry.items():
            assert getattr(measured_annulus, name) == value
        return original_sky_fit(*args, **kwargs)

    monkeypatch.setattr(center, "sky_fit", record_sky_fit)

    result, shift, history = center.find_center_2dg(
        image,
        (15.0, 15.0),
        cbox_size=9,
        sky_annulus=annulus,
        atol_shift=1.0e-6,
        full=True,
        full_history=True,
    )

    assert_allclose(result, expected_center, rtol=0, atol=1.0e-5)
    assert_allclose(shift, np.linalg.norm(expected_center - (15.0, 15.0)), atol=1.0e-5)
    assert len(measured_centers) >= 2
    assert_array_equal(measured_centers, history["positions"][: len(measured_centers)])
    assert_array_equal(annulus.positions, original_positions)
    for name, value in geometry.items():
        assert getattr(annulus, name) == value
