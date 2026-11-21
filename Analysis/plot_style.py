"""Shared look for every chart in the Analysis package.

Colors are a validated categorical palette (fixed order, colorblind-checked on the light
surface below) plus recessive chart chrome: hairline solid gridlines, muted axis text,
no top/right spines, and text in neutral ink, never in a series color.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# --- Surfaces & ink ---
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- Categorical palette: assign in this order, never cycle past it ---
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                           "#e87ba4", "#008300", "#6250d6", "#e34948")
ACCENT = SERIES[0]          # single-series charts
ACCENT_WASH = "#2a78d61a"   # ~10% fill of the accent (box bodies)
DE_EMPHASIS = BASELINE      # e.g. the "Missing" bar
OTHER = "#b4b2a9"           # folded "Other" group

# Most groups that get their own color; the rest fold into "Other"
MAX_COLORED_GROUPS: int = len(SERIES)


def compact(value: float, _position: object = None) -> str:
    """Axis/label number format: in full under 10,000, compact (12K, 3.4M) above."""
    magnitude = abs(value)
    if magnitude >= 999.5e6:
        return f"{value / 1e9:.1f}B".replace(".0B", "B")
    if magnitude >= 999.5e3:
        return f"{value / 1e6:.1f}M".replace(".0M", "M")
    if magnitude >= 1e4:
        return f"{value / 1e3:.0f}K"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.4g}"


def new_figure(width: float = 8.0, height: float = 5.0) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def style_axes(ax: Axes, grid: str | None = "y", value_axis: str | None = "y") -> None:
    """Recessive chrome: hairline grid on `grid` ('x', 'y' or None), compact numbers on `value_axis`."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_SECONDARY, labelsize=9, length=0, pad=6)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    if grid:
        ax.grid(axis=grid, color=GRIDLINE, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
    if value_axis == "y":
        ax.yaxis.set_major_formatter(FuncFormatter(compact))
    elif value_axis == "x":
        ax.xaxis.set_major_formatter(FuncFormatter(compact))


def set_titles(ax: Axes, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title in primary ink, optional one-line subtitle in secondary ink."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY,
                 pad=26 if subtitle else 10)
    if subtitle:
        # Offset in points (not axes fraction) so the gap is the same in short and tall figures
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, ha="left", va="bottom")


def style_legend(ax: Axes, title: str | None = None) -> None:
    """Legend outside the plot on the right, text in neutral ink."""
    legend = ax.legend(title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                       fontsize=9, title_fontsize=9, labelcolor=TEXT_SECONDARY, handlelength=1.2)
    if legend.get_title() is not None:
        legend.get_title().set_color(TEXT_SECONDARY)
        legend._legend_box.align = "left"


def no_data(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            color=TEXT_MUTED, fontsize=11)
