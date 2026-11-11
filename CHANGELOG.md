# Changelog

## Unreleased

### Breaking changes

- Aperture-sum errors overestimated the noise of partially covered pixels;
  `apsum_err` (and so `srcsum_err`, `merr`) and `radcum_profile` `err`/`var` now
  use `sqrt(sum(w**2 * error**2))` instead of `sqrt(sum(w * error**2))`, where
  `w` is the exact pixel-overlap weight. This is the variance of the computed
  sum `sum(w * data)` for independent pixel noise. Errors become smaller; for
  a uniform error map, by about `0.17/r` for radius `r` in pixels (14% at
  `r=1.5`, 3% at `r=5.5`, 0.8% at `r=20`). `method="center"` (`w` is 0 or 1)
  is unchanged. photutils made the same fix in 3.1.0 (PR #2398).
- `apphot_annulus` `srcsum_err` added `apsum_npix * ssky**2`, which counts sky
  noise twice (the error map already holds it), and dropped the sky-estimate
  error. It now uses `srcsum_err**2 = apsum_err**2 + (apsum_npix * dsky)**2`.
  In a 2000-image simulation, the reported error changed from 1.32× to 0.99×
  the measured `srcsum` scatter at `r=2` pixels and from 0.62× to 0.98× at
  `r=14` (annulus 20–25 pixels). Large apertures with small annuli now get
  correctly large errors. `merr` and `snr` change accordingly.
- `sky_fit` (and `apphot_annulus`) return `dsky = ssky * sqrt(c / nsky)`, the
  standard error of `msky`, with `c = 1` (mean), `pi/2` (median), `4.567`
  (`"sex"` mode), or `6.137` (`"IRAF"`/`"MMM"` mode) for Gaussian sky noise.
  `c` follows the per-annulus `"sex"` median and `"IRAF"` mean fallbacks.
  Callable methods give `NaN`; `srcsum_err` then uses `c = 1`, a lower bound.
  `radial_profile` returns it as `dpix`.

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
