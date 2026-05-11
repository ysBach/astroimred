"""Image and CCDData operations."""

from .._core.numeric import *
from . import _config
from ._config import IMOPS_USE_NUMBA
from .ccdops import *
from .imstat import *
from .pixels import *


def set_use_numba(value: bool) -> None:
    """Set IMOPS_USE_NUMBA; prefer this over direct attribute assignment."""
    _config.IMOPS_USE_NUMBA = bool(value)
    globals()["IMOPS_USE_NUMBA"] = _config.IMOPS_USE_NUMBA
