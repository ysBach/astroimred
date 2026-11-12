# Changelog

## 0.4.1 - 2026-11-12

Includes every 0.4.0 change and the comet enhancement work. Module names are relative to `astroimred`.

### Breaking changes

#### `phot`

- `apphot`, `radprof`: aperture-sum errors used `sqrt(sum(w * error**2))` and overestimated small apertures; `apsum_err` and `radcum_profile` errors now use `sqrt(sum(w**2 * error**2))` (`w`: pixel-overlap weight), as photutils 3.1 does. ([#6])
- `apphot`: `srcsum_err` counted sky noise twice and omitted the sky-estimate error; it now uses `sqrt(apsum_err**2 + (apsum_npix * dsky)**2)`. ([#6])
- `background`, `apphot`, `radprof`: `sky_fit` and `apphot_annulus` add `dsky`, the standard error of `msky` (`radial_profile`: `dpix`). ([#6])

#### `imutil`

- `imcombine`: zero/scale/weight statistics now exclude masked pixels (CCDData/FITS masks, `mask`) and non-finite pixels. Empty sections raise `ValueError`; callables receive `NaN`. Calibration and thresholds are unchanged. ([#3])
- `imcombine`: explicit uncertainty extensions and non-default uncertainty types raise `NotImplementedError`; uncertainty propagation remains unsupported. ([#2])

#### Dependencies

- Require `astro-ndslice>=0.4`, `astroapers>=0.2.0`, `imcombiners>=0.1.2` (gain/scale and variance fixes in CCD clipping), and `reducers>=0.4.0`. ([#3])

### Comet enhancement in `viz` ([#3])

- `enhance`, `rvsf`, `larson`: new functions replace the external CometCIEF programs: `enhance_rho`, `azimuthal_normalize`, `azimuthal_renormalize`, `annular_normalize`, `radially_variable_filter`, and `larson_sekanina`. Centers are fractional and zero-based; masks are supported; undefined values are `NaN`.
- `enhance`: `enhance_rho(power=...)` accepts any radial power (default `-1`).
- `enhance`: statistics are mean, median, mode (`3*median - 2*mean`), or a callable, with optional asymmetric iterative sigma clipping (mean/median center, std/MAD spread). `statistic_mask` excludes pixels from the estimates only; they stay in the output.
- `enhance`: the normalize functions accept `return_stats=True` to return the statistics used (`NormStats`).
- Timings and the reasons for implementation choices (e.g., `reducers` vs. NumPy): [benchmarks](https://github.com/ysBach/astroimred/blob/main/benchmarks/COMET_ENHANCEMENT.md).

### Fixes

#### `phot`

- `polarimetry`: `correct_off` subtracted the wrong instrumental offset from Stokes `u`; `correct_eff` and `correct_pa` propagated errors incorrectly.
- `growth`: in `profile`, the integrated Moffat profile ignored `width`.
- `center`: `find_center_2dg` raises `ValueError` when the cutout has no source. With `sky_annulus=...`, it failed with astroapers 0.2.0; it now rebuilds circular, elliptical, rectangular, and pill annuli at each centroid. ([#1], [#3])
- `background`: in `sky_fit`, when mean and median were both `8e307`, the mode overflowed; it now stays `8e307`. ([#3])

#### `imutil`

- `imstat`: in `errormap`, the flat-error term was `(D*s_F/F)**2`; it is now `(D*s_F/F**2)**2` (`D`: pre-flat signal, `F`: flat, `s_F`: flat error). Negative signal is clipped to zero only in the Poisson term.
- `imstat`: `give_stats` accepts masked integer data and sections of different sizes; it raises `ValueError` when no finite unmasked pixel remains. ([#1])
- `imcombine`: `full=True` now requests diagnostics from imcombiners.

#### `_core`

- `numeric`: in `wvg` weighted means (also `cenfunc="wvg"`), scalar uncertainties apply to all values, arrays broadcast, and a 1-D uncertainty along `axis` works as in `numpy.average`.

### FITS and combination

#### `imutil`

- `imcombine`: explicit weights apply to mean FITS combinations, including masked, rejected, offset, and chunked inputs. Weighted non-mean modes raise `ValueError`. ([#2])
- `imcombine`: full and chunked runs now agree, so the memory budget no longer changes calibrated results (single-image, scalar, and vector calibration; fractional integer calibration; reversed trims; rejection flags). Integer mosaics with uncovered pixels are handled. ([#2])
- `_util_fits`: metadata reads no longer load pixels. Partial reads support scaled and unsigned FITS; integer `BLANK` is treated as missing. CCDData shape comes from its data. ([#1], [#2])
- `imcombine`: opt-in rejection flags, FITS flag output, and Euclidean growth across chunks. Only algorithm rejections grow, not input masks or threshold exclusions. ([#2])

#### `fitsmgmt`

- `table`: `fits_summary(output_format=...)` exports CSV or Parquet; large integers survive next to missing values. ([#2])

### Performance

- `phot.apphot`: photometry computes the variance once per call. ([#2])
- `imutil.imcombine`: combination uses fewer temporary arrays; a memory budget below the retained outputs raises an error instead of underestimating memory needs. ([#2])

### Documentation and packaging

- `viz.data`: bundled CometCIEF sample (lossless, 1.51 MB, with provenance), with a runnable tutorial. ([#3])
- Docs: tutorials for aperture photometry, image combination, calibration, and the command line. ([#3])
- A workflow builds the documentation and can deploy GitHub Pages from `main`. ([#3])
- `vX.Y.Z` tags publish to PyPI after CI checks. Packages include the BSD-3-Clause license. ([#7])
- Allow photutils 3 (`photutils>=2.0,<4.0`).

[#1]: https://github.com/ysBach/astroimred/pull/1
[#2]: https://github.com/ysBach/astroimred/pull/2
[#3]: https://github.com/ysBach/astroimred/pull/3
[#6]: https://github.com/ysBach/astroimred/pull/6
[#7]: https://github.com/ysBach/astroimred/pull/7

## 0.4.0 - 2026-11-12

Released by mistake before the comet enhancement work was merged. See 0.4.1,
which includes every 0.4.0 change. Sorry for the confusion.

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
