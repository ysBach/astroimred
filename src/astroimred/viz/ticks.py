"""Tick-location and tick-label helpers for image displays."""

from collections.abc import Sequence

__all__ = ["symmetric_ticks", "format_offset_label", "apply_center_origin_ticks"]


def symmetric_ticks(half: int, n_ticks: int) -> list[int]:
    """Return symmetric integer offsets centered on zero."""
    import math

    if half == 0:
        return [0]
    n_side = max(1, (n_ticks - 1) // 2)
    raw_step = half / n_side
    magnitude = 10 ** math.floor(math.log10(max(raw_step, 1)))
    step = magnitude
    for factor in (1, 2, 5, 10):
        step = int(magnitude * factor)
        if step >= raw_step:
            break
    step = max(1, step)
    offsets = list(range(0, half + 1, step))
    return sorted({-offset for offset in offsets} | set(offsets))


def format_offset_label(
    value: int | float,
    *,
    plus: bool = False,
    math: bool = False,
) -> str:
    """Format an offset tick label."""
    label = f"{value:g}" if isinstance(value, float) else str(value)
    if plus and value > 0:
        label = f"+{label}"
    if math:
        label = f"${label}$"
    return label


def apply_center_origin_ticks(
    ax,
    shape: tuple[int, int],
    xticks: Sequence[int] | None = None,
    yticks: Sequence[int] | None = None,
    *,
    plus: bool = False,
    math: bool = False,
) -> None:
    """Relabel axes ticks so that coordinate zero sits at the image center."""
    from matplotlib.ticker import FixedLocator

    center_col = shape[1] // 2
    center_row = shape[0] // 2

    if xticks is None:
        n_x = len(ax.get_xticks())
        x_offsets = symmetric_ticks(center_col, n_x)
    else:
        x_offsets = list(xticks)
    x_pixel = [offset + center_col for offset in x_offsets]
    ax.xaxis.set_major_locator(FixedLocator(x_pixel))
    ax.set_xticklabels(
        [format_offset_label(offset, plus=plus, math=math) for offset in x_offsets]
    )

    if yticks is None:
        n_y = len(ax.get_yticks())
        y_offsets = symmetric_ticks(center_row, n_y)
    else:
        y_offsets = list(yticks)
    y_pixel = [offset + center_row for offset in y_offsets]
    ax.yaxis.set_major_locator(FixedLocator(y_pixel))
    ax.set_yticklabels(
        [format_offset_label(offset, plus=plus, math=math) for offset in y_offsets]
    )
