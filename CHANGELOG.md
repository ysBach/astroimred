# Changelog

## Unreleased

### Breaking changes

- Aperture-sum errors used `sqrt(sum(w * error**2))` and overestimated small
  apertures; `apsum_err` and `radcum_profile` errors now use
  `sqrt(sum(w**2 * error**2))` (`w`: pixel-overlap weight), as photutils 3.1 does.
- `srcsum_err` counted sky noise twice and omitted the sky-estimate error; it
  now uses `sqrt(apsum_err**2 + (apsum_npix * dsky)**2)`.
- `sky_fit` and `apphot_annulus` add `dsky`, the standard error of `msky`
  (`radial_profile`: `dpix`).

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
