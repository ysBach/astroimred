"""Compare Larson–Sekanina sampling with a SciPy reference after parity checks.

Run ``uv run --no-sync python benchmarks/benchmark_larson_sekanina.py``.
The reference uses independent polar coordinates and two SciPy passes per
rotation: one for intensity, one for missing support. Timings exclude imports
and input creation. Peak traced allocations are measured separately, not RSS.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from astroimred.viz import larson_sekanina

logger = logging.getLogger(__name__)


def scipy_reference(
    image: np.ndarray,
    center: tuple[float, float],
    *,
    angle: float,
    radial_shift: float,
    interpolation: str,
) -> np.ndarray:
    """Evaluate the symmetric difference via polar source coordinates."""
    yy, xx = np.indices(image.shape, dtype=float)
    dx, dy = xx - center[0], yy - center[1]
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    shifted_radius = radius - radial_shift
    order = 1 if interpolation == "linear" else 0
    missing = ~np.isfinite(image)
    filled = np.where(missing, 0.0, image)
    valid = ~missing & (shifted_radius >= 0)
    if radial_shift != 0:
        valid &= radius != 0
    result = np.zeros_like(image)
    for sign in (1, -1):
        shifted_theta = theta + sign * np.deg2rad(angle)
        coordinates = np.array(
            [
                center[1] + shifted_radius * np.sin(shifted_theta),
                center[0] + shifted_radius * np.cos(shifted_theta),
            ]
        )
        sample = map_coordinates(
            filled,
            coordinates,
            order=order,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        invalid = (
            map_coordinates(
                missing.astype(float),
                coordinates,
                order=order,
                mode="constant",
                cval=1.0,
                prefilter=False,
            )
            > 0
        )
        valid &= ~invalid
        result += image - sample
    result[~valid] = np.nan
    return result


def measure(call: Callable[[], np.ndarray], repeats: int) -> dict[str, float]:
    """Measure warm median runtime and a separate peak allocation sample."""
    call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    gc.collect()
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "median_seconds": statistics.median(elapsed),
        "peak_traced_mib": peak / 2**20,
    }


def main() -> None:
    """Check complete outputs before recording timing and allocation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 801])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 9 for size in args.sizes):
        parser.error("repeats must be positive and sizes must be at least 9")
    rng = np.random.default_rng(20260909)
    records = []
    for size in args.sizes:
        center = (size / 2 - 0.25, size / 2 - 0.5)
        yy, xx = np.indices((size, size), dtype=float)
        radius = np.hypot(xx - center[0], yy - center[1])
        original = 1000 / np.sqrt(1 + radius**2) + rng.normal(0, 0.02, radius.shape)
        for masked in (False, True):
            data = original.copy()
            if masked:
                data[size // 3 : size // 3 + 3, size // 2 : size // 2 + 3] = np.nan
            for interpolation in ("nearest", "linear"):
                for radial_shift in (0.0, 2.0):
                    options = {
                        "angle": 10.0,
                        "radial_shift": radial_shift,
                        "interpolation": interpolation,
                    }
                    baseline = partial(scipy_reference, data, center, **options)
                    current = partial(larson_sekanina, data, center, **options)
                    expected, actual = baseline(), current()
                    np.testing.assert_array_equal(
                        np.isfinite(actual), np.isfinite(expected)
                    )
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
                    before, after = (
                        measure(baseline, args.repeats),
                        measure(current, args.repeats),
                    )
                    records.append(
                        {
                            "shape": [size, size],
                            "masked": masked,
                            "parameters": options,
                            "scipy_reference": before,
                            "current": after,
                            "speedup": before["median_seconds"]
                            / after["median_seconds"],
                            "finite_mask_equal": True,
                            "max_absolute_difference": float(
                                np.nanmax(np.abs(actual - expected))
                            ),
                        }
                    )
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "reducers")
        },
        "repeats": args.repeats,
        "records": records,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        logger.info("%s", payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
