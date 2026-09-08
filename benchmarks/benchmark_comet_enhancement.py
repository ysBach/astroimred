"""Compare comet enhancements with a Git baseline after checking parity.

Run from the repository root with the existing project environment:

    uv run --no-sync python benchmarks/benchmark_comet_enhancement.py \\
        --baseline-ref 2fc418a --sizes 300 801 --repeats 3

Timings exclude imports and data creation. Peak traced allocations are
measured separately from timing; they are not process peak RSS. Annular
normalization is timed separately because it uses different sampling.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from astroimred import viz


def load_baseline(ref: str) -> dict[str, Callable]:
    """Load enhancement modules and their helpers from a Git snapshot."""
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("_comet_benchmark_baseline")
    package.__path__ = []
    sys.modules[package.__name__] = package
    functions = {}
    for name in (
        "_enhance_common",
        "_enhance_profile",
        "_enhance_stats",
        "enhance",
        "rvsf",
    ):
        path = f"src/astroimred/viz/{name}.py"
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{path}"],
            cwd=root,
            capture_output=True,
        )
        if exists.returncode and name.startswith("_"):
            continue
        source = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        module = types.ModuleType(f"{package.__name__}.{name}")
        sys.modules[module.__name__] = module
        exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
        functions.update(
            {key: getattr(module, key) for key in getattr(module, "__all__", ())}
        )
    return functions


def measure(function: Callable[[], object], repeats: int) -> dict[str, float]:
    """Measure warm median runtime and a separate peak allocation sample."""
    function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    gc.collect()
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "median_seconds": statistics.median(elapsed),
        "peak_traced_mib": peak / 2**20,
    }


def main() -> None:
    """Validate unchanged methods before reporting their measured speedups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 801])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Also compare clipped mode and measure profile/statistic-mask overhead.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 9 for size in args.sizes):
        parser.error("repeats must be positive and sizes must be at least 9")
    baseline = load_baseline(args.baseline_ref)
    records = []
    rng = np.random.default_rng(20260908)
    for size in args.sizes:
        yy, xx = np.indices((size, size), dtype=float)
        center = (size / 2 - 0.25, size / 2 - 0.5)
        radius = np.hypot(xx - center[0], yy - center[1])
        data = 1000.0 / np.sqrt(1.0 + radius**2) + rng.normal(0, 0.02, (size, size))
        nrad = size // 2 - 2
        cases = [
            ("rho", "enhance_rho", {}),
            ("polar_median", "azimuthal_normalize", {"nrad": nrad, "ntheta": 360}),
            (
                "polar_clipped_mean",
                "azimuthal_normalize",
                {
                    "nrad": nrad,
                    "ntheta": 360,
                    "statistic": "mean",
                    "sigma": 3,
                },
            ),
            (
                "polar_renormalize",
                "azimuthal_renormalize",
                {
                    "nrad": nrad,
                    "ntheta": 360,
                    "nsigma": 3,
                    "sigma": 3,
                },
            ),
            ("rvsf_constant", "radially_variable_filter", {"a": 1, "b": 0, "n": 1}),
            ("rvsf_growing", "radially_variable_filter", {"a": 1, "b": 1, "n": 0.3}),
            ("rvsf_steep", "radially_variable_filter", {"a": 0.5, "b": 0.8, "n": 1}),
        ]
        if args.profiles:
            cases.append(
                (
                    "polar_clipped_mode",
                    "azimuthal_normalize",
                    {
                        "nrad": nrad,
                        "ntheta": 360,
                        "statistic": "mode",
                        "sigma": (3, 3),
                        "maxiters": 5,
                        "clip_center": "median",
                        "clip_scale": "mad_std",
                    },
                )
            )
        for label, name, kwargs in cases:
            old = partial(baseline[name], data, center, **kwargs)
            new = partial(getattr(viz, name), data, center, **kwargs)
            expected, actual = old(), new()
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            # Float64 summation order can change; this tolerance is a benchmark
            # assertion, never a production dispatch or rounding rule.
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            before = measure(old, args.repeats)
            after = measure(new, args.repeats)
            records.append(
                {
                    "case": label,
                    "shape": [size, size],
                    "subpixels": 10,
                    "parameters": kwargs,
                    "baseline": before,
                    "current": after,
                    "speedup": before["median_seconds"] / after["median_seconds"],
                    "finite_mask_equal": True,
                    "max_absolute_difference": float(
                        np.nanmax(np.abs(actual - expected))
                    ),
                }
            )
        annular = measure(
            partial(
                viz.annular_normalize, data, center, radii=np.arange(0.5, nrad + 1)
            ),
            args.repeats,
        )
        records.append(
            {"case": "annular_median", "shape": [size, size], "current": annular}
        )
        if args.profiles:
            # Deterministic exclusions exercise support near masks. They are
            # benchmark inputs, never production thresholds or dispatch rules.
            statistic_mask = np.zeros(data.shape, dtype=bool)
            statistic_mask[size // 3 : size // 3 + 3, size // 2 : size // 2 + 3] = True
            hard_mask = np.zeros(data.shape, dtype=bool)
            hard_mask[size // 4 : size // 4 + 2, size // 3 : size // 3 + 2] = True
            pipelines = [
                ("polar", viz.azimuthal_normalize, {"nrad": nrad, "ntheta": 360}),
                ("annular", viz.annular_normalize, {"radii": np.arange(0.5, nrad + 1)}),
            ]
            for label, function, geometry in pipelines:
                options = dict(
                    geometry,
                    statistic="mode",
                    sigma=(3, 3),
                    maxiters=5,
                    clip_center="median",
                    clip_scale="mad_std",
                    mask=hard_mask,
                )
                for masked in (False, True):
                    call = partial(
                        function,
                        data,
                        center,
                        **options,
                        statistic_mask=statistic_mask if masked else None,
                    )
                    with_profile = partial(call, return_profile=True)
                    expected, profile = call(), with_profile()
                    np.testing.assert_allclose(
                        profile.image, expected, rtol=1e-10, atol=1e-10
                    )
                    np.testing.assert_array_equal(
                        np.isfinite(profile.image), np.isfinite(expected)
                    )
                    before = measure(call, args.repeats)
                    after = measure(with_profile, args.repeats)
                    records.append(
                        {
                            "case": f"{label}_profile",
                            "shape": [size, size],
                            "statistic_mask": masked,
                            "without_profile": before,
                            "with_profile": after,
                            "profile_time_ratio": after["median_seconds"]
                            / before["median_seconds"],
                        }
                    )
    payload = {
        "baseline_ref": args.baseline_ref,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "reducers", "astroapers")
        },
        "repeats": args.repeats,
        "input": "Synthetic inverse-radius coma plus seeded Gaussian noise; fractional center.",
        "records": records,
    }
    output = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output)
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
