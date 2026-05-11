"""Small filesystem and runtime helpers."""

import sys
from pathlib import Path

import numpy as np

from .types import StrPathLike

__all__ = ["get_size", "mkdir"]


def get_size(obj: object, seen: set | None = None) -> int:
    """Recursively estimate an object's memory size in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        for values in (obj.keys(), obj.values()):
            for value in values:
                if not (isinstance(value, np.ndarray) and value.ndim == 0):
                    size += get_size(value, seen)
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(item, seen) for item in obj)
    return size


def mkdir(fpath: StrPathLike, mode: int = 0o777, exist_ok: bool = True) -> None:
    """Create a directory with `~pathlib.Path.mkdir` semantics."""
    Path(fpath).mkdir(mode=mode, exist_ok=exist_ok)
