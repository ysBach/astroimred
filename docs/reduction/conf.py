"""Sphinx configuration for astroimred.reduction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "astroimred.reduction"
author = "Yoonsoo P. Bach"
copyright = "2026, Yoonsoo P. Bach"

try:
    from importlib.metadata import version as package_version

    release = package_version("astroimred")
except Exception:
    release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

try:
    import pydata_sphinx_theme  # noqa: F401

    html_theme = "pydata_sphinx_theme"
except ImportError:
    html_theme = "alabaster"
html_title = f"{project} {release}"
html_static_path = []

# Avoid writing Astropy/Matplotlib caches into unwritable home directories on RTD.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
