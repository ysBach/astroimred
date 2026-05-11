"""Astropy unit conversion helpers."""

import copy
from warnings import warn

from astropy import units as u

__all__ = ["as_quantity", "change_to_quantity"]


def as_quantity(
    value: object, unit: str | u.Unit = "", to_value: bool = False
) -> u.Quantity | object:
    """Convert an object to `~astropy.units.Quantity`, or to a scalar value."""
    if value is None:
        return None

    try:
        return value.to(unit).value if to_value else value.to(unit)
    except AttributeError:
        if to_value:
            return _copy(value)
        if isinstance(unit, str):
            unit = u.Unit(unit)
        try:
            return value * unit
        except TypeError:
            return _copy(value)
    except TypeError:
        return _copy(value)
    except u.UnitConversionError as err:
        raise ValueError(
            "If you use astropy.Quantity, you should use unit convertible to `unit`."
            + f'\nYou gave "{value.unit}", unconvertible with "{unit}".'
        ) from err


def change_to_quantity(
    value: object, desired: str | u.Unit = "", to_value: bool = False
) -> u.Quantity | object:
    """Deprecated alias for `as_quantity`."""
    warn(
        "change_to_quantity is deprecated; use as_quantity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return as_quantity(value, desired, to_value=to_value)


def _copy(x: object) -> object:
    try:
        return x.copy()
    except AttributeError:
        return copy.deepcopy(x)
