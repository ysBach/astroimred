"""Benchmark astroapers migration paths against photutils.

The script validates numerical agreement before reporting timings. It is meant
to be run from the astroimred repo root, for example:

    PYTHONPATH=../astroapers/python uv run --with pytest python benchmarks/benchmark_astroapers_migration.py --format table
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import astroapers as aap
import astroapers.kernels as aapk
import numpy as np
from numpy.testing import assert_allclose
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)

from astroimred.phot._aper_backend import photometer
from astroimred.phot.background import _sky_fit, annul2values, sky_fit


@dataclass(frozen=True)
class Timing:
    task: str
    library: str
    n_sources: int
    seconds: float
    speedup_vs_photutils: float | None


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    image = make_image(args.image_size)
    timings: list[Timing] = []

    for n_sources in args.counts:
        x = rng.uniform(32, args.image_size - 32, n_sources)
        y = rng.uniform(32, args.image_size - 32, n_sources)
        timings.extend(benchmark_apsum(image, x, y, args))
        timings.extend(benchmark_masked_apsum(image, x, y, args))
        timings.extend(benchmark_sky_values(image, x, y, args))
        timings.extend(benchmark_sky_fit(image, x, y, args))

    payload = {
        "metadata": metadata(args),
        "timings": [asdict(timing) for timing in timings],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=Timing.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(payload["timings"])
    else:
        print_table(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 100, 10_000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    return parser.parse_args()


def make_image(image_size: int) -> np.ndarray:
    y, x = np.indices((image_size, image_size), dtype=np.float64)
    return 100.0 + 0.01 * x + 0.02 * y + 4.0 * np.sin(x / 31.0)


def benchmark_apsum(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, args: argparse.Namespace
) -> list[Timing]:
    astro_ap = aap.CircAp(np.column_stack([x, y]), r=5.5)
    phot_ap = CircularAperture(np.column_stack([x, y]), r=5.5)

    expected = aperture_photometry(image, phot_ap, method="exact")["aperture_sum"]
    actual = photometer(image, astro_ap).apsum
    kernel_actual = aapk.apsum_circ_exact(image, x, y, r=5.5, return_npix=False)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-9)
    assert_allclose(kernel_actual, expected, rtol=2e-12, atol=2e-9)

    photutils_time = median_seconds(
        lambda: aperture_photometry(image, phot_ap, method="exact"),
        repeats=args.repeats,
    )
    astroapers_time = median_seconds(
        lambda: photometer(image, astro_ap),
        repeats=args.repeats,
    )
    optimal_time = median_seconds(
        lambda: aapk.apsum_circ_exact(image, x, y, r=5.5, return_npix=False),
        repeats=args.repeats,
    )
    return pair_timings("apsum", len(x), photutils_time, astroapers_time) + [
        Timing(
            "apsum_aapk",
            "astroapers",
            len(x),
            optimal_time,
            photutils_time / optimal_time if optimal_time > 0 else None,
        )
    ]


def benchmark_masked_apsum(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, args: argparse.Namespace
) -> list[Timing]:
    positions = np.column_stack([x, y])
    astro_ap = aap.CircAp(positions, r=5.5)
    phot_ap = CircularAperture(positions, r=5.5)
    mask = np.zeros_like(image, dtype=bool)
    mask[::23, ::19] = True
    mask[24:40, 24:40] = True

    expected = aperture_photometry(image, phot_ap, mask=mask, method="exact")[
        "aperture_sum"
    ]
    actual = photometer(image, astro_ap, mask=mask).apsum
    kernel_actual = aapk.apsum_circ_exact(
        image, x, y, r=5.5, mask=mask, return_npix=False
    )
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-9)
    assert_allclose(kernel_actual, expected, rtol=2e-12, atol=2e-9)

    photutils_time = median_seconds(
        lambda: aperture_photometry(image, phot_ap, mask=mask, method="exact"),
        repeats=args.repeats,
    )
    astroapers_time = median_seconds(
        lambda: photometer(image, astro_ap, mask=mask),
        repeats=args.repeats,
    )
    optimal_time = median_seconds(
        lambda: aapk.apsum_circ_exact(image, x, y, r=5.5, mask=mask, return_npix=False),
        repeats=args.repeats,
    )
    return pair_timings("apsum_masked", len(x), photutils_time, astroapers_time) + [
        Timing(
            "apsum_masked_aapk",
            "astroapers",
            len(x),
            optimal_time,
            photutils_time / optimal_time if optimal_time > 0 else None,
        )
    ]


def benchmark_sky_values(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, args: argparse.Namespace
) -> list[Timing]:
    positions = np.column_stack([x, y])
    astro_an = aap.CircAn(positions, r_in=8.0, r_out=13.0)
    phot_an = CircularAnnulus(positions, r_in=8.0, r_out=13.0)

    expected = [mask.get_values(image) for mask in phot_an.to_mask(method="center")]
    actual = annul2values(image, astro_an)
    for actual_values, expected_values in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.sort(actual_values), np.sort(expected_values))

    photutils_time = median_seconds(
        lambda: [mask.get_values(image) for mask in phot_an.to_mask(method="center")],
        repeats=args.repeats,
    )
    astroapers_time = median_seconds(
        lambda: annul2values(image, astro_an),
        repeats=args.repeats,
    )
    return pair_timings("sky_values", len(x), photutils_time, astroapers_time)


def benchmark_sky_fit(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, args: argparse.Namespace
) -> list[Timing]:
    positions = np.column_stack([x, y])
    astro_an = aap.CircAn(positions, r_in=8.0, r_out=13.0)
    phot_an = CircularAnnulus(positions, r_in=8.0, r_out=13.0)

    photutils_time = median_seconds(
        lambda: sky_fit_photutils_reference(image, phot_an),
        repeats=args.repeats,
    )
    astroapers_time = median_seconds(
        lambda: sky_fit(image, astro_an),
        repeats=args.repeats,
    )
    return pair_timings("sky_fit", len(x), photutils_time, astroapers_time)


def sky_fit_photutils_reference(image: np.ndarray, annulus: CircularAnnulus):
    values = [mask.get_values(image) for mask in annulus.to_mask(method="center")]
    return [
        _sky_fit(value, method="sex", sigma=3, maxiters=5, std_ddof=1)
        for value in values
    ]


def pair_timings(
    task: str, n_sources: int, photutils_time: float, astroapers_time: float
) -> list[Timing]:
    return [
        Timing(task, "photutils", n_sources, photutils_time, None),
        Timing(
            task,
            "astroapers",
            n_sources,
            astroapers_time,
            photutils_time / astroapers_time if astroapers_time > 0 else None,
        ),
    ]


def median_seconds(func: Callable[[], object], *, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "astroapers": package_version("astroapers"),
        "image_size": args.image_size,
        "counts": args.counts,
        "repeats": args.repeats,
        "seed": args.seed,
    }


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return getattr(aap, "__version__", "unknown")


def print_table(payload: dict[str, object]) -> None:
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    print()
    print(f"{'task':<16} {'library':<12} {'n':>8} {'seconds':>12} {'speedup':>10}")
    for row in payload["timings"]:
        speedup = row["speedup_vs_photutils"]
        speedup_text = "" if speedup is None else f"{speedup:0.2f}x"
        print(
            f"{row['task']:<16} {row['library']:<12} "
            f"{row['n_sources']:>8} {row['seconds']:>12.6g} {speedup_text:>10}"
        )


if __name__ == "__main__":
    main()
