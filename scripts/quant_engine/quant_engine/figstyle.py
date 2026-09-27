"""Chart style for the README figures, with a light and a dark variant of every figure.

The categorical colours follow a fixed order (blue, orange, aqua, yellow, magenta)
whose adjacent pairs were checked for colour-vision-deficiency separation on both
surfaces; the dark variant uses steps chosen for the dark surface rather than an
automatic inversion. Lines are 2 px, gridlines are solid hairlines, text is set
in neutral inks and never in the series colour, and series are identified by a
legend plus selective direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#f0efec",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "negative": "#e34948", "band": (0.12, 0.24),
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "neutral": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "negative": "#e66767", "band": (0.24, 0.45),
    },
}

WIDTH, DPI = 10.0, 150
# A neo-grotesque sans stack: the first family installed is used (Liberation Sans and FreeSans are metric clones of
# Arial and Helvetica available on most Linux systems); DejaVu Sans, bundled with matplotlib, is the last resort.
SANS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "FreeSans", "DejaVu Sans"]


def rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "font.family": "sans-serif", "font.sans-serif": SANS, "font.size": 10,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 6, "ytick.major.pad": 6, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": t["ink2"],
        "axes.prop_cycle": plt.cycler(color=t["series"]),
    }


def new_figure(t: dict, height=4.6, ncols=1, sharey=False, width=WIDTH):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=sharey)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.80, bottom=0.14, wspace=0.18)
    return fig, axes


def header(fig, t: dict, title: str, subtitle: str | None = None, source: str | None = None):
    fig.text(0.07, 0.945, title, fontsize=13, weight="bold", color=t["ink"], ha="left", va="top")
    if subtitle:
        fig.text(0.07, 0.885, subtitle, fontsize=10, color=t["ink2"], ha="left", va="top")
    if source:
        fig.text(0.07, 0.025, source, fontsize=8, color=t["muted"], ha="left", va="bottom")


def end_label(ax, t: dict, x, y, text: str, color: str, dy: float = 0.0, dx: float = 6.0):
    """Marker with a surface ring at (x, y) and a label in neutral ink to its right."""
    ax.plot([x], [y], "o", ms=7, color=color, mec=t["surface"], mew=2, zorder=5, clip_on=False)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=t["ink2"],
                va="center", ha="left", annotation_clip=False)


def point_label(ax, t: dict, x, y, text: str, color: str, dx=0.0, dy=10.0, ha="center"):
    ax.plot([x], [y], "o", ms=7, color=color, mec=t["surface"], mew=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=t["ink2"], ha=ha,
                va="bottom" if dy >= 0 else "top")


def band(ax, x, lo, hi, color: str, alpha: float):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def render(builder, name: str, out: Path, *args, **kwargs) -> list[Path]:
    """Draw `builder(t, *args)` in both themes and save <name>-light.png and <name>-dark.png."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode, t in THEMES.items():
        with plt.rc_context(rc(t)):
            fig = builder(t, *args, **kwargs)
            path = out / f"{name}-{mode}.png"
            fig.savefig(path, dpi=DPI, facecolor=t["surface"])
            plt.close(fig)
            paths.append(path)
    return paths


def percent_axis(ax, axis="y", decimals=0):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def nice_limits(values, pad=0.08):
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    span = hi - lo if hi > lo else 1.0
    return lo - pad * span, hi + pad * span


def label_offsets(ax, values, min_gap=13.0) -> list[float]:
    """Vertical offsets in points that keep end labels at least `min_gap` points apart (same order as `values`)."""
    fig = ax.figure
    fig.canvas.draw()
    y_pts = [ax.transData.transform((0, v))[1] * 72.0 / fig.dpi for v in values]
    order = np.argsort(y_pts)
    placed, offsets = [], [0.0] * len(values)
    for i in order:
        y = y_pts[i]
        if placed and y < placed[-1] + min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        offsets[i] = y - y_pts[i]
    return offsets


# --------------------------------------------------------------------------- notebooks

NOTEBOOK = THEMES["light"]
SERIES = NOTEBOOK["series"]            # categorical order: blue, orange, aqua, yellow, magenta (never cycled)
NEGATIVE = NOTEBOOK["negative"]        # losses, breaches and thresholds
MUTED = NOTEBOOK["muted"]              # context series drawn behind the ones that matter
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # one hue, light to dark (validated, --ordinal)


def notebook_style(dpi: int = 110) -> None:
    """Matplotlib defaults for the notebooks, matching the README charts.

    - validated categorical order, never cycled;
    - neo-grotesque sans type with a clear hierarchy: a bold headline (see `headline`), a lighter subtitle, panel
      titles and axis text in secondary ink;
    - solid hairline grid behind the marks, no top, right or left spine, no tick marks;
    - 1.5 pt lines, a thin surface-coloured edge on every bar and patch so that touching fills are separated by a
      gap rather than an outline;
    - frameless legends and constrained layout, so that legends placed outside the axes, for example with
      ``fig.legend(loc="outside lower center")``, and direct labels never overlap titles, labels or data.
    """
    import matplotlib as mpl

    params = rc(NOTEBOOK)
    params.update({
        "figure.dpi": dpi, "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
        "figure.titlesize": 13, "figure.titleweight": "bold",
        "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titlepad": 8, "axes.titlecolor": NOTEBOOK["ink2"],
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.handlelength": 1.4, "legend.columnspacing": 1.6,
        "lines.linewidth": 1.5, "patch.force_edgecolor": True, "patch.edgecolor": NOTEBOOK["surface"],
        "patch.linewidth": 1.0,
    })
    mpl.rcParams.update(params)


def headline(ax, title: str, subtitle: str | None = None) -> None:
    """Bold takeaway title and a lighter subtitle (what is plotted, in which units), left-aligned above `ax`.

    Both are artists of the axes, so constrained layout reserves room for them. Keep the subtitle to one line.
    """
    pad = 8.0
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5, color=NOTEBOOK["ink2"], annotation_clip=False)
        pad = 7 + 9.5 * 1.3 + 6
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=NOTEBOOK["ink"], pad=pad)


def figure_headline(fig, title: str) -> None:
    """Bold takeaway title over a multi-panel figure; the panel titles say what each panel shows."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=NOTEBOOK["ink"])


def label_ends(ax, series: dict, colors: dict, fmt=None, min_gap: float = 12.0) -> None:
    """Direct labels at the right end of line series.

    `series` maps a name to a pandas Series; each label is a dot with a surface ring on the last point and the name
    (and, with `fmt`, the last value) in neutral ink. Labels are spread vertically to at least `min_gap` points.
    """
    names = list(series)
    ends = [(series[n].index[-1], float(series[n].iloc[-1])) for n in names]
    offsets = label_offsets(ax, [y for _, y in ends], min_gap=min_gap)
    for name, (x, y), dy in zip(names, ends, offsets):
        text = name if fmt is None else f"{name}  {fmt(y)}"
        end_label(ax, NOTEBOOK, x, y, text, colors[name], dy=dy)


def label_bar_segments(ax, bars, values, fmt, min_value: float) -> None:
    """Value labels centred inside horizontal bar segments that are wide enough (`values` >= `min_value`), in
    white or ink according to the luminance of the fill; smaller segments are left to the legend and the table."""
    from matplotlib.colors import to_rgb

    for bar, value in zip(bars, values):
        if value < min_value:
            continue
        r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(bar.get_facecolor()))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ink = NOTEBOOK["ink"] if luminance > 0.18 else "white"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, fmt(value), ha="center",
                va="center", fontsize=8, color=ink)


def ordinal_colors(n: int) -> list:
    """n colours along the validated blue ramp, light to dark, for ordered categories such as dates."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ordinal", ORDINAL)
    return [cmap(v) for v in np.linspace(0.0, 1.0, n)]
