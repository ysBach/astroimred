"""Time formatting helpers."""

from astropy.time import Time

__all__ = ["str_now"]


def str_now(
    precision: int = 3,
    fmt: str = "{:.>72s}",
    t_ref: Time | None = None,
    dt_fmt: str = "(dt = {:.3f} s)",
    return_time: bool = False,
) -> str | tuple:
    """Get stringified current UTC time in ISOT format."""
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
