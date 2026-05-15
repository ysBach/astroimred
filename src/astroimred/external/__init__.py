"""Adapters around optional external astronomy packages."""

try:
    from .sep import *
except ModuleNotFoundError as err:
    if err.name != "sep":
        raise
