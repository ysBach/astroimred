# Changelog

## 0.2.0 - 2026-05-07

- Bumped to v0.2.0.
- Python >=3.11 and Astropy >=7.0. (Some type hints/old astropy-affiliated package supports dropped/updated accordingly.)
- Moved CR-rejection tools to `crrej.py` (`crrej`, `medfilt_bpm`, `LACOSMIC_*`, `parse_crrej_psf`).
- Reworked lower-median helper for current NumPy.
- Fixed `imcombine` diagnostic upper output and refactored full-stack combine stages.
- Added `imred` Click CLI with `comb`, `copy`, and `arith` subcommands.
- Fixed `run_reduc_plan(return_ccd=True)` return path.
- Fixed `imarith` header/history preservation.
- Removed `bdf_process`; reduction plans now use `ccdred`.
- Added/updated CR-rejection, lower-median, preprocessing, and image-combination regression tests.

## 0.1.1 - 2026-05-07

- Bumped to v0.1.1.
- Fixed minor typos in comments and docstrings.
- Removed shared mutable defaults in preprocessing helpers.
- Replaced diagnostic `print` calls and non-deprecation warnings with package logger calls.

## 0.1.0 - 2026-05-07

- Initial port of image reduction utilities from `ysfitsutilpy` and `ysphotutilpy`.
- Added `preproc.py`, `combutil.py`, and the `imutil` package.
- Added focused preprocessing tests and image-combination regression tests with fixtures.
