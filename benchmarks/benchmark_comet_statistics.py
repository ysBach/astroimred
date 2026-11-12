"""Benchmark comet statistic reductions against the committed baseline.

The statistic inputs are float64 rings with shapes ``(148, 360)`` and
``(398, 360)``. Each uses seeded normal values centered on 1000, NaNs at the
``(19*row + 7*column) % 113 == 0`` positions, and positive/negative ``1e6``
outliers at two separate modular masks. These constants are benchmark inputs;
they do not select production code paths.

The extrema inputs are finite float64 coordinate-like arrays with shapes
``(10, 300)`` and ``(10, 801)``. They compare NumPy's separate min/max scans
with reducers' fused ``minmax`` and NaN-aware ``nanminmax``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import reducers as rd
from benchmark_comet_enhancement import load_baseline

SAMPLE_SHAPES = ((148, 360), (398, 360))
SAMPLE_SEEDS = (148360, 398360)
EXTREMA_SHAPES = ((10, 300), (10, 801))
REPEATS = 11
WARMUPS = 3
FloatArray = npt.NDArray[np.float64]


def measure(
    function: Callable[[], object], repeats: int, warmups: int
) -> dict[str, float]:
    """Return warm median call time in microseconds."""
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return {
        "median_microseconds": statistics.median(samples) / 1_000.0,
        "minimum_microseconds": min(samples) / 1_000.0,
        "maximum_microseconds": max(samples) / 1_000.0,
    }


def assert_parity(
    expected: tuple[FloatArray, ...], actual: tuple[FloatArray, ...]
) -> dict[str, Any]:
    """Assert finite masks and values agree before any timing starts."""
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    finite_mask_equal = True
    for old, new in zip(expected, actual, strict=True):
        old_finite = np.isfinite(old)
        new_finite = np.isfinite(new)
        np.testing.assert_array_equal(old_finite, new_finite)
        finite_mask_equal &= np.array_equal(old_finite, new_finite)
        finite = old_finite & new_finite
        if np.any(finite):
            difference = np.abs(old[finite] - new[finite])
            scale = np.maximum(np.abs(old[finite]), np.finfo(np.float64).tiny)
            max_absolute_difference = max(
                max_absolute_difference, float(np.max(difference))
            )
            max_relative_difference = max(
                max_relative_difference, float(np.max(difference / scale))
            )
    np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12, equal_nan=True)
    return {
        "finite_mask_equal": finite_mask_equal,
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": max_relative_difference,
    }


def make_samples(shape: tuple[int, int], seed: int) -> FloatArray:
    """Build one deterministic ring matrix with missing values and outliers."""
    rng = np.random.default_rng(seed)
    samples = 1000.0 + 10.0 * rng.normal(size=shape)
    rows, columns = np.indices(shape)
    samples[(rows * 19 + columns * 7) % 113 == 0] = np.nan
    samples[(rows * 11 + columns * 5) % 127 == 0] = 1.0e6
    samples[(rows * 13 + columns * 3) % 149 == 0] = -1.0e6
    return samples


def benchmark_statistics(
    current: ModuleType,
    baseline: ModuleType,
    samples: FloatArray,
    shape: tuple[int, int],
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    """Check parity and benchmark the requested statistic paths."""
    cases: dict[str, dict[str, object]] = {
        "mean_std": {
            "statistic": "mean",
            "sigma": None,
            "maxiters": 1,
            "clip_center": "mean",
            "clip_scale": "std",
            "return_std": True,
        },
        "mode_sigma": {
            "statistic": "mode",
            "sigma": (3.0, 3.0),
            "maxiters": 5,
            "clip_center": "median",
            "clip_scale": "mad_std",
            "return_std": True,
        },
    }
    records: dict[str, Any] = {"shape": list(shape), "cases": {}}
    for label, options in cases.items():
        old_call = partial(baseline._ring_statistics, samples, **options)
        new_call = partial(current._ring_statistics, samples, **options)
        parity = assert_parity(old_call(), new_call())
        old_time = measure(old_call, repeats, warmups)
        new_time = measure(new_call, repeats, warmups)
        records["cases"][label] = {
            "parity": parity,
            "baseline": old_time,
            "current": new_time,
            "speedup": old_time["median_microseconds"]
            / new_time["median_microseconds"],
        }

    old_call = partial(baseline._mean_std, samples, return_std=True)
    new_call = partial(current._mean_std, samples, return_std=True)
    records["mean_std_helper_parity"] = assert_parity(old_call(), new_call())
    old_time = measure(old_call, repeats, warmups)
    new_time = measure(new_call, repeats, warmups)
    records["mean_std_helper_baseline"] = old_time
    records["mean_std_helper_current"] = new_time
    records["mean_std_helper_speedup"] = (
        old_time["median_microseconds"] / new_time["median_microseconds"]
    )
    return records


def numpy_extrema(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the two NumPy extrema scans used as the reference."""
    return np.min(values, axis=0), np.max(values, axis=0)


def benchmark_extrema(repeats: int, warmups: int) -> list[dict[str, Any]]:
    """Compare NumPy extrema with reducers 0.4 fused extrema."""
    rng = np.random.default_rng(20260908)
    records = []
    for shape in EXTREMA_SHAPES:
        values = rng.normal(size=shape).astype(np.float64)
        expected = numpy_extrema(values)
        candidates = {
            "reducers_minmax": partial(rd.minmax, values, axis=0),
            "reducers_nanminmax": partial(rd.nanminmax, values, axis=0),
        }
        for name, candidate in candidates.items():
            actual = tuple(candidate())
            assert_parity(expected, actual)
            numpy_time = measure(partial(numpy_extrema, values), repeats, warmups)
            reducer_time = measure(candidate, repeats, warmups)
            records.append(
                {
                    "shape": list(shape),
                    "candidate": name,
                    "equality": True,
                    "numpy": numpy_time,
                    "reducers": reducer_time,
                    "numpy_over_reducers": numpy_time["median_microseconds"]
                    / reducer_time["median_microseconds"],
                }
            )
    return records


def main() -> None:
    """Load the baseline, assert parity, and emit JSON timing records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be nonnegative")

    load_baseline(args.baseline_ref)
    baseline = sys.modules["_comet_benchmark_baseline.viz._enhance_stats"]
    current = importlib.import_module("astroimred.viz._enhance_stats")
    statistics_records = [
        benchmark_statistics(
            current,
            baseline,
            make_samples(shape, seed),
            shape,
            args.repeats,
            args.warmups,
        )
        for shape, seed in zip(SAMPLE_SHAPES, SAMPLE_SEEDS, strict=True)
    ]
    result = {
        "baseline_ref": args.baseline_ref,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "numpy": importlib.metadata.version("numpy"),
            "reducers": importlib.metadata.version("reducers"),
        },
        "repeats": args.repeats,
        "warmups": args.warmups,
        "statistics": statistics_records,
        "extrema": benchmark_extrema(args.repeats, args.warmups),
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output)
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
