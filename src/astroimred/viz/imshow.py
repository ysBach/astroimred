"""Image display helpers with Astropy normalization."""

from collections.abc import Sequence

import numpy.typing as npt
from astropy.visualization import (
    BaseInterval,
    BaseStretch,
    ImageNormalize,
    LinearStretch,
    ZScaleInterval,
)

from astroimred._core.astropy_helpers import parse_interval, parse_stretch
from astroimred.viz.ticks import apply_center_origin_ticks

__all__ = [
    "znorm",
    "zimshow",
    "parse_stretch",
    "parse_interval",
    "imshow_norm",
]


def znorm(
    image: npt.ArrayLike,
    stretch: BaseStretch | None = None,
    **kwargs: object,
) -> ImageNormalize:
    """Create an image normalization using a ZScale interval.

    Parameters
    ----------
    image : array-like
        The image data to normalize.
    stretch : BaseStretch, optional
        Stretch function to apply. The default is `LinearStretch`.
    **kwargs
        Additional keyword arguments passed to `ZScaleInterval`.

    Returns
    -------
    ImageNormalize
        Normalization object suitable for `imshow`.
    """
    if stretch is None:
        stretch = LinearStretch()
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs), stretch=stretch)


def zimshow(
    ax: object,
    image: npt.ArrayLike,
    stretch: BaseStretch | None = None,
    cmap: object = None,
    origin: str = "lower",
    zscale_kw: dict[str, object] | None = None,
    **kwargs: object,
) -> object:
    """Display an image with ZScale normalization.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes on which to display the image.
    image : array-like
        The 2D image data to display.
    stretch : BaseStretch, optional
        Stretch function to apply. The default is `LinearStretch`.
    cmap : str or Colormap, optional
        Colormap to use.
    origin : {'upper', 'lower'}, optional
        Image origin convention. The default is 'lower'.
    zscale_kw : dict, optional
        Additional keyword arguments passed to `ZScaleInterval`.
    **kwargs
        Additional keyword arguments passed to `Axes.imshow`.

    Returns
    -------
    AxesImage
        Image object returned by `imshow`.
    """
    if zscale_kw is None:
        zscale_kw = {}
    im = ax.imshow(
        image,
        norm=znorm(image, stretch=stretch, **zscale_kw),
        origin=origin,
        cmap=cmap,
        **kwargs,
    )
    return im


def imshow_norm(
    data: npt.ArrayLike,
    ax: object = None,
    stretch: str | BaseStretch = "linear",
    interval: str | BaseInterval | None = None,
    origin: str = "lower",
    tickorigin2center: bool = False,
    xticks: Sequence[int] | None = None,
    yticks: Sequence[int] | None = None,
    ticklabel_plus: bool = False,
    ticklabel_math: bool = False,
    return_norm: bool = False,
    # stretch tuning
    asinh_a: float = 0.1,
    log_a: float = 1000.0,
    power: float = 1.0,
    sinh_a: float = 0.3,
    # range / clipping
    vmin: float | None = None,
    vmax: float | None = None,
    min_percent: float | None = None,
    max_percent: float | None = None,
    percent: float | None = None,
    clip: bool = False,
    invalid: float | None = -1.0,
    **kwargs: object,
) -> object | tuple[object, ImageNormalize]:
    """Display an image with Astropy normalization.

    A unified wrapper around `astropy.visualization.imshow_norm` that resolves
    stretch names from strings, supports common interval shortcuts, defaults
    `origin` to `"lower"`, and optionally relabels axes so that
    coordinate 0 sits at the image center.

    Parameters
    ----------
    data : array-like
        The 2-D image data to display.
    ax : matplotlib Axes or None, optional
        Target axes. If `None`, uses the current pyplot axes.
    stretch : str or BaseStretch, optional
        Stretch specification. A string is resolved case-insensitively and
        accepts both bare names (`"asinh"`) and full class names
        (`"AsinhStretch"`). A `BaseStretch` instance is passed through
        unchanged. The default is `"linear"`.
    interval : str, BaseInterval, or None, optional
        Interval controlling the data range mapping:

        - `None`: uses `vmin`/`vmax` if given, otherwise the data min/max.
        - `"zscale"`: `ZScaleInterval`.
        - `percent=<v>`: `PercentileInterval(v)`.
        - `min_percent`/`max_percent`: `AsymmetricPercentileInterval`.
        - Any `BaseInterval` instance: passed through unchanged.

        The default is `None`.
    origin : str, optional
        Image origin convention. The default is `"lower"`.
    tickorigin2center : bool, optional
        If `True`, relabel axes ticks so that coordinate 0 sits at the
        image center. The default is `False`.
    xticks, yticks : array-like of int or None, optional
        Only used when `tickorigin2center` is `True`. Tick positions as offsets
        from the image center along the x- and y-axis. `None` generates
        symmetric ticks automatically.
    ticklabel_plus : bool, optional
        If `True`, positive centered-offset tick labels include a leading `+`.
        The default is `False`.
    ticklabel_math : bool, optional
        If `True`, centered-offset tick labels are wrapped in `$...$`. The
        default is `False`.
    return_norm : bool, optional
        If `True`, return `(AxesImage, ImageNormalize)`. If `False`, return
        only the `AxesImage` for backward compatibility. The default is `False`.
    asinh_a : float, optional
        The `a` parameter for `AsinhStretch`. The default is `0.1`.
    log_a : float, optional
        The `a` parameter for `LogStretch`. The default is `1000.0`.
    power : float, optional
        The `a` parameter for `PowerStretch`. The default is `1.0`.
    sinh_a : float, optional
        The `a` parameter for `SinhStretch`. The default is `0.3`.
    vmin, vmax : float or None, optional
        Explicit minimum/maximum data values for the normalization. Ignored
        when `interval` is set to a `BaseInterval` instance or `"zscale"`.
    min_percent, max_percent : float or None, optional
        Percentile-based minimum/maximum. Constructs `AsymmetricPercentileInterval`
        when either is given and `interval` is `None`.
    percent : float or None, optional
        Symmetric percentile for both min and max. Constructs `PercentileInterval`
        when given and `interval` is `None`. Takes precedence over
        `min_percent`/`max_percent`.
    clip : bool, optional
        Whether to clip values outside the normalized range. The default is
        `False`.
    invalid : float or None, optional
        Value assigned to invalid (NaN/inf) pixels before normalization.
        The default is `-1.0`.
    **kwargs
        Additional keyword arguments forwarded to `Axes.imshow`.

    Returns
    -------
    AxesImage or tuple
        Image object. If `return_norm=True`, returns `(im, norm)`, where
        `norm` is the `ImageNormalize` instance.
    """
    import numpy as np

    stretch_obj = parse_stretch(
        stretch,
        asinh_a=asinh_a,
        log_a=log_a,
        power=power,
        sinh_a=sinh_a,
    )
    interval_obj = parse_interval(
        interval,
        percent=percent,
        min_percent=min_percent,
        max_percent=max_percent,
    )

    norm = ImageNormalize(
        np.asarray(data, dtype=float),
        interval=interval_obj,
        vmin=vmin,
        vmax=vmax,
        stretch=stretch_obj,
        clip=clip,
        invalid=invalid,
    )

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    im = ax.imshow(np.asarray(data, dtype=float), origin=origin, norm=norm, **kwargs)

    if tickorigin2center:
        shape = getattr(data, "shape", None)
        if shape is None:
            shape = np.asarray(data).shape
        apply_center_origin_ticks(
            ax,
            shape,
            xticks=xticks,
            yticks=yticks,
            plus=ticklabel_plus,
            math=ticklabel_math,
        )

    if return_norm:
        return im, norm
    return im
