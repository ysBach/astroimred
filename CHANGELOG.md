# Changelog

## 0.4.0 - 2026-11-12

- `enhance_rho` only corrected `1/rho`; `power=-2` now uses mean `rho**2`,
  while default `power=-1` preserves finite results. Overflow now gives `NaN`.

- Larson–Sekanina filtering required custom code; `larson_sekanina(angle=10)`
  now sums opposite rotational differences, with radial shifts, interpolation, and masks.

- Sky fitting and comet enhancement duplicated MMM mode formulas; they now
  share one policy. Equal mean/median of `8e307` previously overflowed in
  sky fitting; mode now stays `8e307`. Sampling and clipping defaults are unchanged.

- `return_profile=True` / `.profile` (`RadialProfile`) are now
  `return_stats=True` / `.stats` (`NormStats`); values and image returns are unchanged.
- Masks removed stars from the output; `statistic_mask` now excludes them
  only from estimates (e.g., a `100` pixel divided by a masked mean of `2` stays `50`).
- Mean/std used separate reductions and MAD clipping repeated medians;
  reducers 0.4 now fuses mean/std and coordinate extrema, and unchanged
  clipping statistics are reused.

### Breaking changes

- Aperture-sum errors used `sqrt(sum(w * error**2))` and overestimated small
  apertures; `apsum_err` and `radcum_profile` errors now use
  `sqrt(sum(w**2 * error**2))` (`w`: pixel-overlap weight), as photutils 3.1 does.
- `srcsum_err` counted sky noise twice and omitted the sky-estimate error; it
  now uses `sqrt(apsum_err**2 + (apsum_npix * dsky)**2)`.
- `sky_fit` and `apphot_annulus` add `dsky`, the standard error of `msky`
  (`radial_profile`: `dpix`).
- Development installs selected `astro-ndslice 0.2` and `astroapers 0.1.2`;
  now require `astro-ndslice>=0.4` and `astroapers>=0.2.0`. All four internal
  dependencies install from PyPI; no local checkout paths are configured.
- `imcombiners>=0.1.1` allowed CCD clipping without the gain/scale and stable
  variance fixes in 0.1.2; now require `imcombiners>=0.1.2`.
- Require `reducers>=0.4.0`; environments with 0.3.x must upgrade.
- CCDData/FITS masks, `mask`, and nonfinite pixels now exclude samples from
  `imcombine` zero/scale/weight statistics. Empty sections raise `ValueError`;
  callables receive `NaN`. Numeric calibration and threshold behavior are unchanged.
- Explicit uncertainty extensions and non-default uncertainty types now raise
  `NotImplementedError`; uncertainty propagation remains unsupported.

### Comet visualization

- CometCIEF required external programs; `astroimred.viz` now provides array-based
  rho, azimuthal, and variable-filter enhancements with fractional, zero-based
  centers and masks. Undefined divisions/neighborhoods return `NaN`.
- `statistic="mode"` raised an error; it now estimates `3*median - 2*mean`.
  Mean, median, mode, and callables accept asymmetric, iterative sigma clipping
  with mean/median centers and std/MAD spread.
- Radial bins required custom code; `annular_normalize(radii=[0, 2, 5, 10])`
  now normalizes original pixel-center annuli without polar resampling.
- `subpixels=np.int8(12)` could reverse output signs; it now matches Python
  integers. Boolean coefficients such as `a=True` now raise `ValueError`.
- Bright constant fields could overflow; normalization now preserves unity
  and small background spreads. Zero-width filters preserve a flat field's
  zero response.

### FITS and combination

- Explicit weights now affect mean FITS combinations, including masked,
  rejected, offset, and chunked inputs. Weighted non-mean modes raise `ValueError`.
- Changing memory budgets could change calibrated results; full and chunked
  runs now agree for single-image/scalar/vector calibration, fractional integer
  calibration, reversed trims, and rejection flags. Uncovered integer mosaics are handled.
- Metadata reads loaded whole images; they now avoid pixel reads. Partial
  reads support scaled/unsigned FITS; both paths treat integer `BLANK` as
  missing. CCDData shapes now come from its data.
- Add opt-in rejection flags, FITS flag output, and Euclidean growth across
  chunks. Only algorithm rejections grow; input masks and threshold exclusions do not.
- Summary tables now preserve large integers beside missing values.
  `fits_summary(output_format=...)` also supports explicit CSV/Parquet export.

### Photometry

- `find_center_2dg(..., sky_annulus=...)` assigned immutable positions and
  failed with astroapers 0.2.0; now rebuilds circular, elliptical, rectangular,
  and pill annuli at each centroid while preserving geometry and the input object.

### Performance

- `b=0` repeated 100 identical stencils by default; it now evaluates once.
  Variable filters reuse unchanged widths; polar methods reuse coordinates
  and use reducers for statistics/counts. See [benchmarks](benchmarks/COMET_ENHANCEMENT.md).
- Photometry reuses variance/bad-pixel calculations; combination uses fewer
  temporary arrays. Budgets below retained outputs now raise an error instead
  of underestimating memory needs.

### Documentation and releases

- Development dependency changes were hidden by ignoring `uv.lock`; it can
  now be committed and records PyPI versions only. Duplicate core and
  `full` requirements are removed without changing the installed feature set.
- `return_stats=True` details were a dense paragraph; they now use a field
  table. Guides and docstrings group options while retaining examples and limits.
- The external CometCIEF sample is now bundled losslessly (2.57 → 1.51 MB),
  with provenance and a runnable tutorial.
- API outlines now include runnable FITS, combination, photometry, calibration,
  and CLI guides. Correct obsolete examples such as `ccdred(calc_err=...)`.
- PyPI publishing was manual; matching `vX.Y.Z` tags now publish after CI checks.
  Manual runs only build. Pages deploys from `main`; packages include BSD-3-Clause metadata/text.

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
