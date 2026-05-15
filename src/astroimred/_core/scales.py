"""Linear display-scale converters for dimensionless values and angles."""

import numpy as np

__all__ = ["scale_values", "percent_scale", "degree_scale"]


def scale_values(
    *values: object,
    factor: float,
    input_scaled: bool = False,
    output_scaled: bool = False,
) -> list[object]:
    """Convert values between an internal scale and a linearly scaled display unit.

    Parameters
    ----------
    *values
        Values to convert.
    factor : float
        Multiplicative factor from internal values to display values.
    input_scaled : bool, optional
        Whether input values are already in the scaled display unit.
    output_scaled : bool, optional
        Whether returned values should be in the scaled display unit.
    """
    if not input_scaled and output_scaled:
        scale = factor
    elif input_scaled and not output_scaled:
        scale = 1 / factor
    else:
        scale = 1
    return [value * scale for value in values]


def percent_scale(
    *values: object,
    input_percent: bool = False,
    output_percent: bool = False,
) -> list[object]:
    """Convert values between fractions and percent."""
    return scale_values(
        *values,
        factor=100,
        input_scaled=input_percent,
        output_scaled=output_percent,
    )


def degree_scale(
    *values: object,
    input_degree: bool = False,
    output_degree: bool = False,
) -> list[object]:
    """Convert values between radians and degrees."""
    return scale_values(
        *values,
        factor=180 / np.pi,
        input_scaled=input_degree,
        output_scaled=output_degree,
    )
