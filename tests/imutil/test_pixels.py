"""Tests for pixel mask and saturation helpers."""

import numpy as np

import astroimred as air
from astroimred.imutil import pixels


class TestPixelTools:
    """Tests for pixel helper module exports."""

    def test_pixel_helpers_have_canonical_modules(self):
        """Pixel operations are exposed from pixels."""
        assert air.fixpix is pixels.fixpix
        assert air.find_extpix is pixels.find_extpix
        assert air.find_satpix is pixels.find_satpix

    def test_fixpix_single_pixel_uses_axis_priority(self):
        """A bad pixel should interpolate along the default x priority."""
        data = np.zeros((5, 5), dtype=float)
        data[2, 1] = 10.0
        data[2, 3] = 14.0
        data[1, 2] = 100.0
        data[3, 2] = 200.0
        mask = np.zeros_like(data, dtype=bool)
        mask[2, 2] = True

        out = pixels.fixpix(data, mask, update_header=False)

        assert out.data[2, 2] == 12.0

    def test_fixpix_run_interpolates_without_scipy_label(self, monkeypatch):
        """Masked runs should not label the full image before interpolation."""
        import scipy.ndimage

        data = np.zeros((6, 6), dtype=float)
        data[1, 2] = 20.0
        data[3, 2] = 20.0
        data[1, 3] = 30.0
        data[3, 3] = 30.0
        mask = np.zeros_like(data, dtype=bool)
        mask[2, 2:4] = True

        def fail_label(*args, **kwargs):
            raise AssertionError("scipy.ndimage.label should not be called")

        monkeypatch.setattr(scipy.ndimage, "label", fail_label)

        out = pixels.fixpix(data, mask, update_header=False)

        np.testing.assert_allclose(out.data[2, 2:4], [20.0, 30.0])

    def test_fixpix_large_mask_uses_single_run_span_algorithm(self, monkeypatch):
        """Large masks should follow the same run-span algorithm without labeling."""
        import scipy.ndimage

        data = np.tile(np.arange(100, dtype=float), (100, 1))
        mask = np.zeros_like(data, dtype=bool)
        mask[:, 40:90] = True

        def fail_label(*args, **kwargs):
            raise AssertionError("scipy.ndimage.label should not be called")

        monkeypatch.setattr(scipy.ndimage, "label", fail_label)

        out = pixels.fixpix(data, mask, update_header=False)

        expected = np.linspace(40, 89, 50)
        np.testing.assert_allclose(out.data[0, 40:90], expected)
        np.testing.assert_allclose(out.data[-1, 40:90], expected)
