# Comet enhancement performance

Run `uv run --no-sync python benchmarks/benchmark_comet_enhancement.py --baseline-ref 2fc418a --sizes 300 801 --repeats 3` from the repository root.

The script checks each unchanged method against the pre-optimization commit before timing it. It compares complete finite/NaN masks and float64 outputs with `rtol=atol=1e-10`. The tolerance allows changed accumulation order; production code does not use that tolerance or dispatch on benchmark settings.

## Measured results

September 8, 2026, macOS ARM64, Python 3.13.11, NumPy 2.4.4, reducers 0.3.0. Median of three warm runs; imports and input creation excluded. The 801 × 801 input is a seeded synthetic inverse-radius coma with Gaussian noise and a fractional center. These are local measurements, not universal speed guarantees.

| Operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Multiply by radius | 0.137 s | 0.117 s | 1.16× |
| Polar median | 1.073 s | 0.867 s | 1.24× |
| Polar sigma-clipped mean | 1.078 s | 0.875 s | 1.23× |
| Polar renormalization | 1.088 s | 0.886 s | 1.23× |
| RVSF, `a=1, b=0, n=1` | 3.012 s | 0.014 s | 216× |
| RVSF, `a=1, b=1, n=0.3` | 3.444 s | 0.715 s | 4.82× |
| RVSF, `a=0.5, b=0.8, n=1` | 3.416 s | 0.479 s | 7.13× |

All rows use `subpixels=10`. Polar rows use `nrad=398, ntheta=360`; clipped rows use `sigma=3`. The new direct annular median takes 0.035 s for unit-width intervals from radius 0.5 through 398.5. It has different sampling semantics, so it is not a parity comparison with polar normalization.

Raw measurements, including 300 × 300 cases and maximum output differences, are in [comet_enhancement_results.json](comet_enhancement_results.json).

## Why it changes

1. Constant-width RVSF evaluates one stencil. Variable-width RVSF caches the last rounded width and stencil value per pixel; unchanged widths need no neighbor gathers.
2. Radius and polar operations reuse one-dimensional coordinates. Reconstruction skips pixels whose midpoint radii cannot all have valid rings.
3. Ring statistics use reducers across rows. The current minimum `reducers>=0.3.0` supports these operations; newer clipping APIs are not required.
4. Direct annular normalization computes a radius map once and groups pixels once. It avoids repeatedly constructing an aperture bounding box for every ring.

## Tradeoffs

Caching uses memory. For gentle-growth RVSF, peak traced allocations rise from 69.8 to 85.9 MiB; the constant-width case rises to 77.5 MiB. The steep case falls to 54.6 MiB because fewer pixels remain valid. Polar operations fall from 72.6 to 68.9 MiB, and radius multiplication falls from 34.3 to 15.4 MiB. These are separate `tracemalloc` samples, not process peak RSS.

`annular_normalize` gives every finite pixel center equal weight in explicit radial intervals. CometCIEF-style normalization averages equal-angle polar samples and reconstructs them. Exact `astroapers` overlap weights would define a third estimator; they are not silently substituted. This implementation uses the already-available NumPy radius map because a requested outer radius can extend far beyond the image without requiring a huge aperture mask.

`statistic="mode"` is the MMM approximation `3*median - 2*mean`, not an exact density mode. Strong outliers can make it negative or move it outside the observed range. Use explicit clipping or a custom statistic when appropriate. Clipping modifies only the estimated divisor/location/spread; rejected bright features remain in the output.
