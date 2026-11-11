import numpy as np
from scipy.integrate import quad

from astroimred.phot.growth import (
    _profile_moffat,
    _profile_moffat_i,
    dmag,
    profile,
)


class TestMoffatGrowth:
    """Tests for Moffat integrated growth profile and width dependence."""

    def test_analytical_flux_at_r5_width10_power25(self):
        """Independently verified analytical flux at r=5, width=10, power=2.5."""
        r = 5.0
        flux = 1.0
        width = 10.0
        power = 2.5

        # F(R) = flux * [1 - (1 + (R / width)^2)^(-(power - 1))]
        # (5 / 10)^2 = 0.25 -> 1 + 0.25 = 1.25
        # power - 1 = 1.5
        # 1.25^1.5 = 1.25 * sqrt(1.25)
        expected = flux * (1.0 - 1.0 / (1.25**1.5))
        # expected is approximately 0.28445824720006733
        actual = _profile_moffat_i(r, flux=flux, width=width, power=power)

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)

    def test_integral_vs_quadrature(self):
        """Verify 2*pi*r integral of differential profile matches integrated profile."""
        flux = 120.0
        width = 4.25
        power = 3.2
        r_target = 8.5

        def integrand(r):
            return (
                2.0
                * np.pi
                * r
                * _profile_moffat(r, flux=flux, width=width, power=power)
            )

        numerical_integral, abserr = quad(
            integrand, 0.0, r_target, epsabs=1e-13, epsrel=1e-13
        )
        analytic = _profile_moffat_i(r_target, flux=flux, width=width, power=power)

        np.testing.assert_allclose(analytic, numerical_integral, rtol=1e-12, atol=1e-12)

    def test_derivative_matches_differential_profile(self):
        """Central difference of integrated profile dF/dr matches 2*pi*r*I(r)."""
        flux = 50.0
        width = 3.5
        power = 2.8
        r = 6.0
        h = 1e-6

        f_plus = _profile_moffat_i(r + h, flux=flux, width=width, power=power)
        f_minus = _profile_moffat_i(r - h, flux=flux, width=width, power=power)
        df_dr = (f_plus - f_minus) / (2.0 * h)

        expected_derivative = (
            2.0 * np.pi * r * _profile_moffat(r, flux=flux, width=width, power=power)
        )
        np.testing.assert_allclose(df_dr, expected_derivative, rtol=1e-9, atol=1e-9)

    def test_coordinate_scale_invariance(self):
        """Scaling both r and width by scale factor s leaves enclosed flux invariant."""
        r = 3.7
        width = 2.4
        power = 2.6
        flux = 1.0
        scale = 3.5

        base = _profile_moffat_i(r, flux=flux, width=width, power=power)
        scaled = _profile_moffat_i(
            scale * r, flux=flux, width=scale * width, power=power
        )

        np.testing.assert_allclose(scaled, base, rtol=1e-15, atol=1e-15)

    def test_profile_and_dmag_propagation(self):
        """profile() and dmag() correctly propagate width changes with valid parameters."""
        r1 = 3.0
        r2 = 6.0
        airmass = 1.15
        a = 0.6
        b = 0.05
        c = 0.4
        power = 2.5
        sigma = 2.0
        r0 = 5.0

        # Test with width=2.0 vs width=8.0
        prof_narrow_r1 = profile(
            r1, airmass, a=a, b=b, c=c, width=2.0, power=power, sigma=sigma, r0=r0
        )
        prof_wide_r1 = profile(
            r1, airmass, a=a, b=b, c=c, width=8.0, power=power, sigma=sigma, r0=r0
        )
        # Enclosed flux at r1 should be smaller for a wider profile
        assert prof_wide_r1 < prof_narrow_r1

        # dmag between r1 and r2
        delta_m = dmag(
            r1, r2, airmass, a=a, b=b, c=c, width=4.0, power=power, sigma=sigma, r0=r0
        )
        prof_r1 = profile(
            r1, airmass, a=a, b=b, c=c, width=4.0, power=power, sigma=sigma, r0=r0
        )
        prof_r2 = profile(
            r2, airmass, a=a, b=b, c=c, width=4.0, power=power, sigma=sigma, r0=r0
        )
        expected_dmag = -2.5 * np.log10(prof_r2 / prof_r1)

        np.testing.assert_allclose(delta_m, expected_dmag, rtol=1e-14, atol=1e-14)

        # Array broadcasting support
        r_arr = np.array([2.0, 4.0, 8.0], dtype=np.float64)
        prof_arr = profile(
            r_arr, airmass, a=a, b=b, c=c, width=4.0, power=power, sigma=sigma, r0=r0
        )
        assert prof_arr.shape == (3,)
        assert np.all(np.diff(prof_arr) > 0)
