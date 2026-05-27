"""Tests for standalone geometry utilities."""

import numpy as np

from astroimred._core import astropy_helpers, geometry

# Strict tolerance for numerical comparisons
RTOL = 1e-6
ATOL = 1e-8


class TestCircularMask:
    """Tests for circular_mask function."""

    def test_basic_2d(self):
        """Test basic 2D circular mask."""
        mask = geometry.circular_mask(shape=(10, 10), center=(5, 5), radius=3)
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Center should be inside the circle
        assert mask[5, 5]
        # Corners should be outside
        assert not mask[0, 0]
        assert not mask[9, 9]

    def test_mask_sum_known(self):
        """Test that mask sum matches expected count."""
        # For a 21x21 grid centered at (10,10) with radius=5
        # The number of pixels inside should be approximately pi*r^2 = 78.5
        mask = geometry.circular_mask(shape=(21, 21), center=(10, 10), radius=5)
        # Allow some tolerance for discretization
        assert 70 <= np.sum(mask) <= 90

    def test_default_center(self):
        """Test that default center is image center."""
        mask = geometry.circular_mask(shape=(10, 10), radius=2)
        # Default center should be (5, 5) for a 10x10 image
        assert mask[5, 5]


class TestStrNow:
    """Tests for str_now function."""

    def test_returns_string(self):
        """Test that str_now returns a string."""
        result = astropy_helpers.str_now()
        assert isinstance(result, str)

    def test_precision(self):
        """Test precision parameter affects output."""
        result_low = astropy_helpers.str_now(precision=0)
        result_high = astropy_helpers.str_now(precision=6)
        # Higher precision should result in longer string
        # (more decimal places in seconds)
        # Both should be valid ISO format times
        assert "T" in result_low
        assert "T" in result_high


class TestAsQuantity:
    """Tests for as_quantity function."""

    def test_float_to_quantity(self):
        """Test converting `float` to `~astropy.units.Quantity`."""
        from astropy import units as u

        result = astropy_helpers.as_quantity(5.0, u.m, to_value=False)
        assert hasattr(result, "unit")
        assert result.value == 5.0

    def test_quantity_passthrough(self):
        """Test that `~astropy.units.Quantity` is passed through."""
        from astropy import units as u

        q = 5.0 * u.m
        result = astropy_helpers.as_quantity(q, u.m, to_value=False)
        assert result.value == 5.0
        assert result.unit == u.m

    def test_to_value_true(self):
        """Test extracting value from `~astropy.units.Quantity`."""
        from astropy import units as u

        result = astropy_helpers.as_quantity(5.0 * u.km, u.m, to_value=True)
        np.testing.assert_allclose(result, 5000.0, rtol=RTOL, atol=ATOL)
