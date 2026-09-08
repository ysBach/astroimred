# Comet enhancement performance

## Reducers 0.4 and profile diagnostics

The API now calls these records `NormStats` (`return_stats=True`, `.stats`).
The benchmark's `--profiles` flag and JSON labels retain their historical names.

September 8, 2026: Python 3.13.11, NumPy 2.4.4, reducers 0.4.0,
macOS ARM64; median of three warm runs against commit `5d29543`.
All compared baseline images matched exactly, including finite masks.
Whole-image speedups ranged from 0.98–1.02×; this run shows no material
end-to-end speed gain. Profiles and estimation masks are opt-in.

| 801 × 801 clipped-mode input | Image only | With profile | Added time |
| --- | ---: | ---: | ---: |
| Polar, hard mask | 861.6 ms | 867.6 ms | 0.69% |
| Polar, hard + estimation masks | 899.9 ms | 912.0 ms | 1.34% |
| Direct annuli, hard mask | 44.6 ms | 50.6 ms | 13.3% |
| Direct annuli, hard + estimation masks | 46.3 ms | 49.2 ms | 6.4% |

All rows use `statistic="mode", sigma=(3, 3), maxiters=5,
clip_center="median", clip_scale="mad_std"`. Polar rows use `nrad=398`,
`ntheta=360`, and `subpixels=10`. Direct annuli use unit-width pixel-center
bins from radius 0.5 through 398.5. The image is a seeded inverse-radius
coma with Gaussian noise and a fractional center; masks exclude small patches.
Profiles matched image-only results at `rtol=atol=1e-10` with identical finite
masks. They added 0.006 MiB (polar) or 0.016 MiB (annular) in peak traced
allocations; these are not process RSS measurements.

Reproduce, including 300 × 300 cases, with:

```bash
uv run --no-sync python benchmarks/benchmark_comet_enhancement.py \
    --baseline-ref 5d29543 --sizes 300 801 --repeats 3 --profiles \
    --output benchmarks/comet_profile_results.json
```

Raw measurements: [comet_profile_results.json](comet_profile_results.json).
Small timing differences are local noise, not universal speed guarantees.

With missing values and injected outliers, isolated ring reductions improved
1.15–1.28× for mean/std and 1.32–1.38× for mode with MAD clipping. These use
148 × 360 and 398 × 360 arrays, three warmups and eleven timed calls.
Reducers 0.4 `minmax` also beat separate NumPy extrema by 1.27–1.46× on
10 × 300 and 10 × 801 arrays. All compared results were exactly equal.
The coordinate arrays are finite, so reconstruction now uses `reducers.minmax`.
These operations are a small fraction of whole-image runtime.

```bash
uv run --no-sync python benchmarks/benchmark_comet_statistics.py \
    --baseline-ref 5d29543 --output benchmarks/comet_statistics_results.json
```

Raw kernel measurements: [comet_statistics_results.json](comet_statistics_results.json).

## Earlier optimization baseline

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
3. Ring statistics use reducers across rows. These timings used reducers 0.3.0;
   the package now requires `reducers>=0.4.0`. Remeasure before attributing these
   results to the newer backend.
4. Direct annular normalization computes a radius map once and groups pixels once. It avoids repeatedly constructing an aperture bounding box for every ring.

## Tradeoffs

Caching uses memory. For gentle-growth RVSF, peak traced allocations rise from 69.8 to 85.9 MiB; the constant-width case rises to 77.5 MiB. The steep case falls to 54.6 MiB because fewer pixels remain valid. Polar operations fall from 72.6 to 68.9 MiB, and radius multiplication falls from 34.3 to 15.4 MiB. These are separate `tracemalloc` samples, not process peak RSS.

`annular_normalize` gives every finite pixel center equal weight in explicit radial intervals. CometCIEF-style normalization averages equal-angle polar samples and reconstructs them. Exact `astroapers` overlap weights would define a third estimator; they are not silently substituted. This implementation uses the already-available NumPy radius map because a requested outer radius can extend far beyond the image without requiring a huge aperture mask.

`statistic="mode"` is the MMM approximation `3*median - 2*mean`, not an exact density mode. Strong outliers can make it negative or move it outside the observed range. Use explicit clipping or a custom statistic when appropriate. Clipping modifies only the estimated divisor/location/spread; rejected bright features remain in the output.

## Finite-count backend check

Prefer reducers where parity and timings support it. On the same environment,
`reducers.count_finite(values, axis=1)` took 0.82–114 microseconds versus
1.11–237 microseconds for `np.count_nonzero(np.isfinite(values), axis=1)`.
The tested ring arrays ranged from `(1, 1)` through `(1500, 720)` and included
NaN and infinity. Speedups were 1.3–2.1× for this counting operation; this is
not an end-to-end enhancement speedup. The reducers call also avoids the
full-size boolean mask. Singleton and empty-ring behavior is unchanged.

Reproduce the comparison in the existing project environment:

```python
import statistics
import timeit

import numpy as np
import reducers as rd

rng = np.random.default_rng(320981)
for shape in [(1, 1), (1, 16), (1, 360), (140, 360), (398, 360), (1500, 720)]:
    values = rng.normal(size=shape)
    values.ravel()[::11] = np.nan
    values.ravel()[5::37] = np.inf
    before = lambda: np.count_nonzero(np.isfinite(values), axis=1)
    after = lambda: rd.count_finite(values, axis=1)
    np.testing.assert_array_equal(after(), before())
    seconds = [
        statistics.median(timeit.repeat(call, number=100, repeat=5)) / 100
        for call in (before, after)
    ]
    print(shape, seconds, seconds[0] / seconds[1])
```

The earlier reducers 0.3.0 benchmark retained NumPy for coordinate extrema:
that version used separate min and max reductions and was slower on most
tested short-axis arrays. The reducers 0.4 measurements above supersede that
choice. Backend preference does not imply a universal performance guarantee.

`ndfiltrs.convolve` uses a fixed kernel, so it cannot directly express the
spatially variable eight-tap RVSF stencil. A fixed-width replacement would
also need to preserve the current weighted-difference arithmetic, which
cancels constant fields before scaling. No ndfiltrs dependency is added for
an unverified substitution.
