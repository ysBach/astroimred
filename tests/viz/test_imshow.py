import numpy as np
import pytest
from astropy.visualization import (
    AsinhStretch,
    PercentileInterval,
    SquaredStretch,
    ZScaleInterval,
)

from astroimred._core.astropy_helpers import parse_interval, parse_stretch
from astroimred.viz import imshow_norm
from astroimred.viz.ticks import format_offset_label


def test_parse_stretch_accepts_bare_and_class_names():
    assert isinstance(parse_stretch("asinh", asinh_a=0.2), AsinhStretch)
    assert isinstance(parse_stretch("SquaredStretch"), SquaredStretch)


def test_parse_interval_shortcuts():
    assert isinstance(parse_interval("zscale"), ZScaleInterval)
    assert isinstance(parse_interval(percent=90), PercentileInterval)


def test_parse_interval_rejects_unknown_string():
    with pytest.raises(ValueError, match="Unknown interval"):
        parse_interval("minmax")


@pytest.mark.parametrize(
    ("value", "plus", "math", "expected"),
    [
        (5, True, False, "+5"),
        (-5, True, False, "-5"),
        (0, True, False, "0"),
        (5, True, True, "$+5$"),
        (-5, True, True, "$-5$"),
    ],
)
def test_format_offset_label(value, plus, math, expected):
    assert format_offset_label(value, plus=plus, math=math) == expected


def test_imshow_norm_center_ticks_can_show_plus_and_math_labels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        imshow_norm(
            np.arange(25).reshape(5, 5),
            ax=ax,
            tickorigin2center=True,
            xticks=[-1, 0, 1],
            yticks=[-2, 0, 2],
            ticklabel_plus=True,
            ticklabel_math=True,
        )

        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "$-1$",
            "$0$",
            "$+1$",
        ]
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "$-2$",
            "$0$",
            "$+2$",
        ]
    finally:
        plt.close(fig)
