"""Verify an installed astroimred artifact, its license, and a FITS round trip."""

from __future__ import annotations

import logging
import os
import sysconfig
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits

import astroimred as air


def main() -> None:
    """Check installed metadata and exercise the public FITS reader.

    Set `EXPECTED_VERSION` to the manifest version before running this script.
    Run with ``python -I scripts/smoke_install.py`` to ignore `PYTHONPATH`.

    Raises
    ------
    AssertionError
        The import comes from the checkout, metadata is wrong, or FITS data differs.
    """
    installed = distribution("astroimred")
    assert installed.version == os.environ["EXPECTED_VERSION"]
    package_path = Path(air.__file__).resolve()
    assert any(
        package_path.is_relative_to(Path(sysconfig.get_path(scheme)).resolve())
        for scheme in ("purelib", "platlib")
    ), f"Expected installed artifact, imported {package_path}"
    licenses = [
        file
        for file in installed.files or []
        if file.name == "LICENSE" and ".dist-info/licenses/" in str(file)
    ]
    assert licenses, "Wheel must contain the BSD license"
    assert "BSD 3-Clause License" in installed.locate_file(licenses[0]).read_text()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "image.fits"
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        fits.writeto(path, data, header=fits.Header({"BUNIT": "adu"}))
        ccd = air.load_ccd(path)
        np.testing.assert_array_equal(ccd.data, data)
    logging.info(
        "Installed astroimred %s passed FITS and license checks", installed.version
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
