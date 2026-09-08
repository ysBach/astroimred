# Changelog

## Unreleased

- Fix comet parameter validation: `subpixels=np.int8(12)` could overflow its squared denominator and reverse the result's sign; NumPy integer counts now match Python integers. Boolean kernel coefficients such as `a=True` previously acted as numbers; they now raise `ValueError` consistently with other scalar options.
- Reuse radially variable filter stencils: `b=0` previously repeated an identical stencil 100 times with default sampling; it now evaluates once. Variable-width filtering reuses each pixel's stencil while its rounded width is unchanged, preserving mask and boundary behavior.
- Expand ring statistics: `statistic="mode"` previously raised an error; it now uses the documented `3*median - 2*mean` estimate. Mean, median, mode, and custom callables support asymmetric sigma clipping, configurable iterations, and mean/median plus standard-deviation/MAD clipping choices.
- Add direct annular normalization: radial intervals such as `radii=[0, 2, 5, 10]` previously required custom pixel selection; `annular_normalize` now groups pixel centers once and divides original pixels by each interval's statistic, without a polar round trip.
- Reduce repeated polar-coordinate work and stabilize averages/statistics: very bright finite constant fields could overflow during midpoint summation; running averages now preserve constant fields, while centered reductions retain narrow spreads on bright backgrounds.

## 0.3.0 - 2026-06-18

- Migrated aperture photometry and image combination to the `astroapers` and `imcombiners` backends; deprecated `combutil`.
- Added linear WCS helper functions (`local_cd_matrix`, `make_linear_wcs`, `make_zoomed_wcs`).
- Performance improvements: cached regex, `fitsio`-preferred I/O, and perf for `bin_ccd` and `sep`-based extraction.
- Unified logging across the package.
- Large-scale internal reorganization and type-hint cleanup.
- [breaking] Removed FWHM-based aperture initialization.

## 0.2.0 - 2026-05-07

- Bumped to v0.2.0.
- Python >=3.11 and Astropy >=7.0. Some type hints and Astropy-affiliated dependency floors were updated accordingly.
- Renamed `make_summary` to `fits_summary` and added `.parq`/`.parquet` summary output.

## 0.1.1 - 2026-05-07

- Bumped to v0.1.1.
- Fixed minor typos in comments and docstrings.
- Replaced diagnostic `print` calls and non-deprecation warnings with package logger calls.
- Added small neutral type hints and formatting/import cleanup.

## 0.1.0 - 2026-05-07

- Initial port of FITS management utilities from `ysfitsutilpy` and lightweight visualization helpers from `ysvisutilpy`.
