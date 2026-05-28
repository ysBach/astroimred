"""Small helpers built around Astropy units, modeling, and visualization."""

import copy

import numpy as np
from astropy import units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from astropy.time import Time
from astropy.visualization import (
    AsinhStretch,
    AsymmetricPercentileInterval,
    BaseInterval,
    BaseStretch,
    LinearStretch,
    LogStretch,
    PercentileInterval,
    PowerStretch,
    SinhStretch,
    SqrtStretch,
    SquaredStretch,
    ZScaleInterval,
)

from .numeric import normalize

__all__ = [
    "as_quantity",
    "str_now",
    "parse_stretch",
    "parse_interval",
    "Gaussian2D_correct",
    "fit_astropy_model",
    "fit_Gaussian2D",
    "gaussian_kernel",
]

_STRETCH_CLASS_MAP: dict[str, type[BaseStretch]] = {
    "linear": LinearStretch,
    "sqrt": SqrtStretch,
    "asinh": AsinhStretch,
    "log": LogStretch,
    "power": PowerStretch,
    "sinh": SinhStretch,
    "square": SquaredStretch,
    "squared": SquaredStretch,
}

_STRETCH_DEFAULTS: dict[str, dict[str, float]] = {
    "asinh": {"a": 0.1},
    "log": {"a": 1000.0},
    "power": {"a": 1.0},
    "sinh": {"a": 0.3},
}


def as_quantity(
    value: object,
    unit: str | u.Unit = "",
    to_value: bool = False,
) -> u.Quantity | object:
    """Convert an object to an Astropy Quantity or scalar in the requested unit.

    Non-Quantity inputs are multiplied by ``unit`` unless ``to_value=True``.
    Objects that cannot be multiplied by a unit are copied and returned as-is.
    """
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


def _copy(x: object) -> object:
    try:
        return x.copy()
    except AttributeError:
        return copy.deepcopy(x)


def str_now(
    precision: int = 3,
    fmt: str = "{:.>72s}",
    t_ref: Time | None = None,
    dt_fmt: str = "(dt = {:.3f} s)",
    return_time: bool = False,
) -> str | tuple:
    """Get stringified time now in UT ISOT format.

    Parameters
    ----------
    precision : `int`, optional.
        The precision of the isot format time.
        Default: ``3``.

    fmt : `str`, optional.
        The Python 3 format string to format the time. Examples::

          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with _.
        Default: ``'{:.>72s}'``.

    t_ref : `~astropy.time.Time`, optional.
        The reference time. If not `None`, delta time is calculated.
        Default: `None`.

    dt_fmt : `str`, optional.
        The Python 3 format string to format the delta time.
        Default: ``'(dt = {:.3f} s)'``.

    return_time : `bool`, optional.
        Whether to return the time at the start of this function and the delta
        time (`dt`), as well as the time information string. If `t_ref` is
        `None`, `dt` is automatically set to `None`.
        Default: `False`.
    """
    now = Time(Time.now(), precision=precision)
    timestr = now.isot
    if t_ref is not None:
        dt = (now - Time(t_ref)).sec
        timestr = dt_fmt.format(dt) + " " + timestr
    else:
        dt = None

    if return_time:
        return fmt.format(timestr), now, dt
    return fmt.format(timestr)


def parse_stretch(
    stretch: str | BaseStretch,
    *,
    asinh_a: float | None = None,
    log_a: float | None = None,
    power: float | None = None,
    sinh_a: float | None = None,
    **kwargs: object,
) -> BaseStretch:
    """Resolve a stretch specification to an Astropy stretch instance.

    Accepts a `~astropy.visualization.BaseStretch` instance, bare names such as
    `"sqrt"`, and full class-style names such as `"SqrtStretch"`. Extra
    keyword arguments are forwarded to the stretch class constructor.
    Named tuning parameters (`asinh_a`, `log_a`, `power`, `sinh_a`) override
    the package defaults for the corresponding stretches.
    """
    if isinstance(stretch, BaseStretch):
        return stretch

    key = stretch.strip().lower()
    if key.endswith("stretch"):
        key = key[: -len("stretch")]
    if key not in _STRETCH_CLASS_MAP:
        supported = ", ".join(sorted(_STRETCH_CLASS_MAP))
        raise ValueError(f"Unknown stretch {stretch!r}. Supported names: {supported}")

    params = {**_STRETCH_DEFAULTS.get(key, {})}
    if key == "asinh" and asinh_a is not None:
        params["a"] = asinh_a
    elif key == "log" and log_a is not None:
        params["a"] = log_a
    elif key == "power" and power is not None:
        params["a"] = power
    elif key == "sinh" and sinh_a is not None:
        params["a"] = sinh_a
    params.update(kwargs)
    return _STRETCH_CLASS_MAP[key](**params)


def parse_interval(
    interval: str | BaseInterval | None = None,
    *,
    percent: float | None = None,
    min_percent: float | None = None,
    max_percent: float | None = None,
) -> BaseInterval | None:
    """Resolve an interval specification to an Astropy interval instance.

    Accepts a `~astropy.visualization.BaseInterval` instance, `"zscale"`,
    `percent`, or asymmetric `min_percent`/`max_percent` bounds. Returns
    `None` when no interval shortcut is requested.
    """
    if isinstance(interval, str):
        if interval.lower() == "zscale":
            return ZScaleInterval()
        raise ValueError(
            f"Unknown interval string {interval!r}. "
            "Use 'zscale' or a BaseInterval instance."
        )

    if interval is not None:
        return interval

    if percent is not None:
        return PercentileInterval(percent)

    if min_percent is not None or max_percent is not None:
        lo = min_percent if min_percent is not None else 0.0
        hi = max_percent if max_percent is not None else 100.0
        return AsymmetricPercentileInterval(lo, hi)

    return None


def Gaussian2D_correct(
    model, theta_lower: float = -np.pi / 2, theta_upper: float = np.pi / 2
):
    """Return a Gaussian2D-like model with x as semimajor axis.

    The returned copy has positive ``x_stddev``/``y_stddev``, ``x_stddev`` as
    the semimajor-axis sigma, and ``theta`` normalized into the requested
    interval.
    """
    new_model = model.copy()
    sig_x = np.abs(model.x_stddev.value)
    sig_y = np.abs(model.y_stddev.value)
    theta = model.theta.value

    if sig_x > sig_y:
        theta_norm = normalize(theta, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_x
        new_model.y_stddev.value = sig_y
        new_model.theta.value = theta_norm
    else:
        theta_norm = normalize(theta + np.pi / 2, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_y
        new_model.y_stddev.value = sig_x
        new_model.theta.value = theta_norm

    return new_model


def fit_astropy_model(data, model_init, sigma=None, fitter=None, **kwargs):
    """Fit an Astropy 2D model to image data.

    ``sigma`` is interpreted as per-pixel 1-sigma uncertainty and passed to
    Astropy as ``weights=1/sigma``.
    """
    yy, xx = np.mgrid[: data.shape[0], : data.shape[1]]
    weights = 1 / sigma if sigma is not None else None
    if fitter is None:
        fitter = LevMarLSQFitter()
    fitted = fitter(model_init, xx, yy, data, weights=weights, **kwargs)
    return fitted, fitter


def fit_Gaussian2D(data, model_init, correct=True, sigma=None, fitter=None, **kwargs):
    """Fit a Gaussian2D-like Astropy model and optionally normalize orientation."""
    fitted, fitter = fit_astropy_model(
        data=data, model_init=model_init, sigma=sigma, fitter=fitter, **kwargs
    )
    if correct:
        fitted = Gaussian2D_correct(fitted)
    return fitted, fitter


def gaussian_kernel(fwhm=None, sigma=None, theta=0, nsigma=5, normalize_area=False):
    """Generate a 2D Gaussian kernel array.

    Exactly one of ``fwhm`` and ``sigma`` must be supplied. Scalar widths are
    applied to both axes; two-element widths are interpreted as ``(x, y)``.
    """
    if ((fwhm is None) + (sigma is None)) != 1:
        raise ValueError("One and only one of `fwhm` and `sigma` should be given.")
    elif fwhm is not None:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = np.atleast_1d(sigma)
    if len(sigma) == 1:
        sigma = np.repeat(sigma, 2)
    amp = 1 / (2 * np.pi * sigma[0] * sigma[1]) if normalize_area else 1
    gauss = Gaussian2D(
        amplitude=amp,
        x_mean=0,
        y_mean=0,
        x_stddev=sigma[0],
        y_stddev=sigma[1],
        theta=theta,
    )
    shape = np.ceil(nsigma * sigma[::-1]).astype(int)
    return gauss(
        *np.ogrid[-shape[0] / 2 : shape[0] / 2 + 1, -shape[1] / 2 : shape[1] / 2 + 1]
    )
