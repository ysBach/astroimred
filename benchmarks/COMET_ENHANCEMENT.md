# Comet enhancement benchmarks

Why `air` uses its current implementations.

- Machine: MBP 14" [2024, macOS 26.5, M4Pro(8P+4E/G20c/N16c/48G)], Python 3.13.11, NumPy 2.4.4, SciPy 1.17.1 (September 2026).
- Time: median of warm runs; imports and input creation excluded.
- Input: synthetic inverse-radius coma, seeded Gaussian noise, fractional center.
- Speedup: (time of the compared option) / (time of what `air` uses).
  Above 1 means `air`'s choice is faster.
- Memory: peak `tracemalloc` allocation, not process RSS.
- Every comparison gave the same outputs (`rtol=atol=1e-10`) and the same NaN masks.

## 1. Larson–Sekanina: NumPy vs. SciPy

- **`air`** (`viz.larson`): `larson_sekanina` samples the image at the rotated and shifted pixel positions with vectorized NumPy indexing.
- **vs.:** `scipy.ndimage.map_coordinates` on a polar grid, written in the benchmark script.
- Each row: range over 4 cases (`radial_shift=0` or `2`, with or without missing pixels); `angle=10`.

| Image | Interpolation | NumPy (used) | SciPy | Speedup |
| --- | --- | ---: | ---: | ---: |
| 300 × 300 | `nearest` | 1.2–1.8 ms | 4.5–4.8 ms | 2.58–3.83× |
| 300 × 300 | `linear` | 4.4–5.0 ms | 7.0–7.1 ms | 1.43–1.62× |
| 801 × 801 | `nearest` | 9.6–14.3 ms | 35.1–35.8 ms | 2.46–3.66× |
| 801 × 801 | `linear` | 33.9–38.0 ms | 52.2–52.7 ms | 1.38–1.55× |

## 2. `reducers` (`rd`) vs. NumPy (`viz.enhance`, `viz._enhance_stats`)

| Operation | `air` uses | vs. | Input | Speedup |
| --- | --- | --- | --- | ---: |
| Min and max | `rd.minmax(x, axis=0)`, one pass | `np.min(x, axis=0)` and `np.max(x, axis=0)`, two passes | 10 × 300 and 10 × 801, finite | 1.27–1.46× |
| Finite count per ring | `rd.count_finite(x, axis=1)` | `np.count_nonzero(np.isfinite(x), axis=1)` | (1, 1) to (1500, 720), with NaN and inf | 1.3–2.1× |

## 3. Ring statistics: fewer passes over the data (`viz._enhance_stats`)

- Input: rings of 148 × 360 and 398 × 360 values, with NaN and ±1e6 outliers.

| Statistic | `air` uses | vs. | Speedup |
| --- | --- | --- | ---: |
| Mean and std | one call: `rd.nanstd(x, axis=1, ddof=1, return_mean=True)` | two calls: `rd.nanmean(x, axis=1)` and `rd.nanstd(x, axis=1, ddof=1)` | 1.15–1.28× |
| Mode, median/MAD clipping | one median per iteration, reused for the center, the MAD, and the final value | separate medians for the center and the MAD in each iteration, plus one for the final value | 1.32–1.38× |

## 4. The speedup for total timing
In sect 2/3 above (using `rd` v0.4.0),
- **`air` uses:** `rd.minmax` (one call for mean & std, and one median per clipping iteration).
- **vs.:** `np.min` and `np.max`, two calls for mean and std, and repeated medians.
Even though statistics calculation is faster using `rd`, the total time is dominated by other parts (sampling, normalization, etc). The speedup is smaller than for the individual operations:
- **Result:** 0.98–1.02× for all methods at 300 × 300 and 801 × 801.
Still, there's no reason not to use `rd` for the statistics, since it is faster and more memory-efficient, and the speedup is larger for larger images and more iterations.

## 5. Cost of `return_stats=True`

- **vs.:** the same call with `return_stats=True` and without it.
- Input: 801 × 801, `statistic="mode"`, `sigma=(3, 3)`, `maxiters=5`.

| Method | Masks | `return_stats=False` | `return_stats=True` | Added time |
| --- | --- | ---: | ---: | ---: |
| Polar | `mask` | 861.6 ms | 867.6 ms | 0.69% |
| Polar | `mask` + `statistic_mask` | 899.9 ms | 912.0 ms | 1.34% |
| Annular | `mask` | 44.6 ms | 50.6 ms | 13.3% |
| Annular | `mask` + `statistic_mask` | 46.3 ms | 49.2 ms | 6.4% |

## 6. Enhancement algorithms: optimized vs. naive implementation (`viz.enhance`, `viz.rvsf`)

- **`air` uses:** constant-width RVSF evaluates one stencil; variable-width RVSF caches the stencil for each width; polar methods reuse 1-D coordinates.
- **vs.:** the "naive" implementation of the same functions, without these three optimizations.
- Both runs use `reducers` 0.3.0. Input: 801 × 801, `subpixels=10`; polar rows `nrad=398, ntheta=360`; clipped rows `sigma=3`.

| Operation | naive version | Optimized (used) | Speedup | Memory, naive → optimized |
| --- | ---: | ---: | ---: | ---: |
| Multiply by radius | 0.137 s | 0.117 s | 1.16× | 34.3 → 15.4 MiB |
| Polar median | 1.073 s | 0.867 s | 1.24× | 72.6 → 68.9 MiB |
| Polar sigma-clipped mean | 1.078 s | 0.875 s | 1.23× | 72.6 → 68.9 MiB |
| Polar renormalization | 1.088 s | 0.886 s | 1.23× | 72.6 → 68.9 MiB |
| RVSF, `a=1, b=0, n=1` | 3.012 s | 0.014 s | 216× | 69.8 → 77.5 MiB |
| RVSF, `a=1, b=1, n=0.3` | 3.444 s | 0.715 s | 4.82× | 69.8 → 85.9 MiB |
| RVSF, `a=0.5, b=0.8, n=1` | 3.416 s | 0.479 s | 7.13× | 69.8 → 54.6 MiB |
