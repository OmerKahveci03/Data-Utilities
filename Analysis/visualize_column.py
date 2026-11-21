"""Create a Matplotlib figure for each supported column type.

Every chart shares the look in :mod:`plot_style`: left-aligned title, a subtitle with the
counts that matter (rows, missing %, ...), recessive axes, and compact number formats.
Box plots draw whiskers at the 1st and 99th percentiles and state the true min and max in
the subtitle, so a handful of extreme values cannot flatten the box.
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from . import plot_style as style
from .summarize_column import (
    ColumnType, classify_column, summarize_binary, summarize_categorical,
)

_MAX_LABEL = 40  # characters of a category label shown on an axis


def _figure_and_axes(ax: Axes | None, width: float = 8.0, height: float = 5.0) -> tuple[Figure, Axes, bool]:
    """Return the figure and axes to draw on, and whether a new figure was created."""
    if ax is None:
        fig, ax = style.new_figure(width, height)
        return fig, ax, True
    return ax.figure, ax, False


def _finish(fig: Figure, created: bool) -> Figure:
    if created:
        fig.tight_layout()
    return fig


def _short(label: object) -> str:
    text = str(label)
    return text if len(text) <= _MAX_LABEL else text[:_MAX_LABEL - 1] + "…"


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "0%"


# ---------------------------------------------------------------------------- binary

def visualize_binary(values: pd.Series | pd.DataFrame, column_name: str, ax: Axes | None = None) -> Figure:
    """Plot negative, positive, and missing counts for a binary column."""
    summary = (summarize_binary(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    fig, ax, created = _figure_and_axes(ax, 7, 4.5)
    style.style_axes(ax, grid="y", value_axis="y")
    subtitle = None
    if summary.empty:
        style.no_data(ax, "No values")
    else:
        row = summary.iloc[0]
        positive, non_missing, missing = int(row["positive_count"]), int(row["non_missing_count"]), int(row["missing_count"])
        total = non_missing + missing
        names = ["Negative", "Positive", "Missing"]
        counts = [non_missing - positive, positive, missing]
        bars = ax.bar(names, counts, width=0.5, color=[style.ACCENT, style.ACCENT, style.DE_EMPHASIS])
        for bar, count in zip(bars, counts):
            ax.annotate(f"{style.compact(count)} ({_pct(count, total)})", (bar.get_x() + bar.get_width() / 2, count),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=style.TEXT_SECONDARY)
        subtitle = f"{style.compact(total)} rows · {_pct(positive, non_missing)} positive of non-missing"
    ax.set_ylabel("Rows")
    style.set_titles(ax, column_name, subtitle)
    return _finish(fig, created)


# ----------------------------------------------------------------------- categorical

def visualize_categorical(values: pd.Series | pd.DataFrame, column_name: str,
                          max_categories: int = 20, ax: Axes | None = None) -> Figure:
    """Plot the most frequent values of a categorical column as horizontal bars."""
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    summary = (summarize_categorical(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    categories = (summary.dropna(subset=["value"]).nlargest(max_categories, "value_count")
                  if not summary.empty else summary)
    shown = len(categories)
    fig, ax, created = _figure_and_axes(ax, 9, max(3.5, 0.32 * shown + 1.8))
    style.style_axes(ax, grid="x", value_axis="x")
    subtitle = None
    if categories.empty:
        style.no_data(ax, "No values")
    else:
        first = summary.iloc[0]
        rows, unique = int(first["row_count"]), int(first["unique_count"])
        labels = [_short(value) for value in categories["value"]][::-1]
        counts = categories["value_count"].to_numpy()[::-1]
        bars = ax.barh(labels, counts, height=0.6, color=style.ACCENT)
        for bar, count in zip(bars, counts):
            ax.annotate(f"{style.compact(count)}  {_pct(count, rows)}", (count, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=style.TEXT_SECONDARY)
        ax.margins(x=0.15)
        shown_note = f"top {shown} of {unique:,} values" if unique > shown else f"{unique:,} values"
        subtitle = f"{style.compact(rows)} rows · {first['missing_pct']:.1f}% missing · {shown_note}"
    ax.set_xlabel("Rows")
    style.set_titles(ax, column_name, subtitle)
    return _finish(fig, created)


# ------------------------------------------------------------------------------ date

def visualize_date(values: pd.Series | pd.DataFrame, column_name: str,
                   summaries: dict[str, pd.DataFrame] | None = None,
                   group: str | None = None, ax: Axes | None = None) -> Figure:
    """Plot the number of observations in each calendar year."""
    subtitle = None
    if isinstance(values, pd.Series):
        dates = pd.to_datetime(values, errors="coerce")
        counts = dates.dt.year.value_counts()
        histogram = pd.DataFrame({"year": counts.index, "date_count": counts.values})
        if dates.notna().any():
            subtitle = (f"{style.compact(len(dates))} rows · {_pct(dates.isna().sum(), len(dates))} missing · "
                        f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    else:
        histogram = pd.DataFrame() if summaries is None else summaries.get("date_histogram", pd.DataFrame())
        if not histogram.empty:
            histogram = histogram.loc[histogram["column"].eq(column_name)]
            if group is not None and "group" in histogram.columns:
                histogram = histogram.loc[histogram["group"].astype("string").eq(group)]
    fig, ax, created = _figure_and_axes(ax, 9, 4.5)
    style.style_axes(ax, grid="y", value_axis="y")
    if histogram.empty:
        style.no_data(ax, "No dates")
    else:
        bins = histogram.sort_values("year")
        ax.bar(bins["year"].astype(int), bins["date_count"].astype(int), width=0.7, color=style.ACCENT)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows")
    style.set_titles(ax, column_name, subtitle)
    return _finish(fig, created)


# -------------------------------------------------------------------------------- id

def visualize_id(values: pd.Series | pd.DataFrame, column_name: str,
                 summaries: dict[str, pd.DataFrame] | None = None,
                 group: str | None = None, ax: Axes | None = None) -> Figure:
    """Plot how many identifiers have each number of associated rows."""
    subtitle = None
    if isinstance(values, pd.Series):
        per_id = values.dropna().astype("string").value_counts()
        counts = per_id.value_counts()
        histogram = pd.DataFrame({"row_count": counts.index, "id_count": counts.values})
        if not per_id.empty:
            subtitle = (f"{style.compact(len(per_id))} unique IDs · median {per_id.median():,.0f} rows per ID · "
                        f"max {style.compact(per_id.max())}")
    else:
        histogram = pd.DataFrame() if summaries is None else summaries.get("id_histogram", pd.DataFrame())
        if not histogram.empty:
            histogram = histogram.loc[histogram["column"].eq(column_name)]
            if group is not None and "group" in histogram.columns:
                histogram = histogram.loc[histogram["group"].astype("string").eq(group)]
    fig, ax, created = _figure_and_axes(ax, 9, 4.5)
    style.style_axes(ax, grid="y", value_axis="y")
    if histogram.empty:
        style.no_data(ax, "No IDs")
    else:
        bins = histogram.sort_values("row_count")
        if bins["row_count"].max() > 100:
            ax.plot(bins["row_count"].astype(int), bins["id_count"].astype(int), color=style.ACCENT, linewidth=2)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(style.compact))
        else:
            ax.bar(bins["row_count"].astype(int), bins["id_count"].astype(int), width=0.7, color=style.ACCENT)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Rows per ID")
    ax.set_ylabel("IDs")
    style.set_titles(ax, column_name, subtitle)
    return _finish(fig, created)


# ------------------------------------------------------------------------- numerical

def _numeric_stats(values: pd.Series | pd.DataFrame) -> dict | None:
    """Box statistics from raw values or a `summarize_numerical` row; None if no numbers."""
    if isinstance(values, pd.Series):
        numbers = pd.to_numeric(values, errors="coerce")
        present = numbers.dropna()
        if present.empty:
            return None
        p01, p25, p50, p75, p99 = present.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        return {"n": len(present), "rows": len(numbers), "missing": int(numbers.isna().sum()),
                "min": present.min(), "max": present.max(),
                "p01": p01, "p25": p25, "p50": p50, "p75": p75, "p99": p99}
    if values.empty or pd.isna(values.iloc[0]["min"]):
        return None
    row = values.iloc[0]
    return {"n": int(row["non_missing_count"]), "rows": int(row["row_count"]), "missing": int(row["missing_count"]),
            "min": row["min"], "max": row["max"], "p01": row["p01"], "p25": row["p25"],
            "p50": row["median"], "p75": row["p75"], "p99": row["p99"]}


def _draw_boxes(ax: Axes, stats: list[dict], labels: list[str]) -> None:
    """Boxes (quartiles, median) with p1–p99 whiskers; y-limits fit the whiskers."""
    ax.bxp([{"med": s["p50"], "q1": s["p25"], "q3": s["p75"], "whislo": s["p01"], "whishi": s["p99"],
             "fliers": [], "label": label} for s, label in zip(stats, labels)],
           showfliers=False, widths=0.45, patch_artist=True,
           boxprops={"facecolor": style.ACCENT_WASH, "edgecolor": style.ACCENT, "linewidth": 1.5},
           medianprops={"color": style.ACCENT, "linewidth": 2.5},
           whiskerprops={"color": style.TEXT_MUTED, "linewidth": 1.2},
           capprops={"color": style.TEXT_MUTED, "linewidth": 1.2})
    low, high = min(s["p01"] for s in stats), max(s["p99"] for s in stats)
    pad = (high - low) * 0.06 or abs(high) * 0.05 or 1
    ax.set_ylim(low - pad, high + pad)


def visualize_numerical(values: pd.Series | pd.DataFrame, column_name: str, ax: Axes | None = None) -> Figure:
    """Plot a box summary for a numerical column (whiskers at p1 and p99)."""
    stats = _numeric_stats(values)
    fig, ax, created = _figure_and_axes(ax, 6, 5)
    style.style_axes(ax, grid="y", value_axis="y")
    subtitle = None
    if stats is None:
        style.no_data(ax, "No numeric values")
    else:
        _draw_boxes(ax, [stats], [f"n = {style.compact(stats['n'])}"])
        subtitle = (f"{_pct(stats['missing'], stats['rows'])} missing · median {style.compact(stats['p50'])} · "
                    f"min {style.compact(stats['min'])}, max {style.compact(stats['max'])} · whiskers p1–p99")
    ax.set_ylabel("Value")
    style.set_titles(ax, column_name, subtitle)
    return _finish(fig, created)


# -------------------------------------------------------------------------- dispatch

def visualize_column(values: pd.Series | pd.DataFrame,
                     column_type: ColumnType | None = None,
                     column_name: str | None = None,
                     max_categories: int = 20,
                     summaries: dict[str, pd.DataFrame] | None = None,
                     group: str | None = None,
                     ax: Axes | None = None) -> Figure:
    """Dispatch a column to its type-specific visualization function.

    Arguments
    ---------
    values: pd.Series | pd.DataFrame
        Raw values or summary rows for one column.
    column_type: ColumnType | None (default None)
        Column type, inferred from a raw Series when omitted.
    column_name: str | None (default None)
        Name displayed in the figure title.
    max_categories: int (default 20)
        Maximum categorical values to plot.
    summaries: dict[str, pd.DataFrame] | None (default None)
        Related summary tables containing date and ID histograms.
    group: str | None (default None)
        Group label for selecting its histogram rows.
    ax: Axes | None (default None)
        Axes to draw on. A new figure is created when omitted.

    Returns
    -------
    Figure
        Matplotlib figure owned by the caller.
    """
    name = column_name or (str(values.name or "column") if isinstance(values, pd.Series)
                           else str(values["column"].iloc[0]) if not values.empty else "column")
    kind = column_type or (classify_column(name, values) if isinstance(values, pd.Series) else None)
    if kind is None:
        raise ValueError("column_type is required for summary rows")
    if kind == "binary":
        return visualize_binary(values, name, ax=ax)
    if kind == "categorical":
        return visualize_categorical(values, name, max_categories=max_categories, ax=ax)
    if kind == "date":
        return visualize_date(values, name, summaries=summaries, group=group, ax=ax)
    if kind == "id":
        return visualize_id(values, name, summaries=summaries, group=group, ax=ax)
    if kind == "numerical":
        return visualize_numerical(values, name, ax=ax)
    raise ValueError(f"Unsupported column type: {kind}")


# ------------------------------------------------------------------------ stratified

def _fold_groups(order: list[str], labels: pd.Series) -> tuple[list[str], pd.Series]:
    """Keep the first MAX_COLORED_GROUPS - 1 groups and fold the rest into one "Other"
    group, so every colored group has its own palette slot (colors are never cycled)."""
    if len(order) <= style.MAX_COLORED_GROUPS:
        return order, labels
    kept = order[:style.MAX_COLORED_GROUPS - 1]
    other = f"Other ({len(order) - len(kept)} groups)"
    return [*kept, other], labels.where(labels.isin(kept), other)


def _group_colors(groups: list[str]) -> list[str]:
    return [style.OTHER if group.startswith("Other (") else style.SERIES[i] for i, group in enumerate(groups)]


def visualize_stratified(values: pd.Series,
                         groups: pd.Series,
                         column_type: ColumnType,
                         column_name: str,
                         group_order: list[str] | None = None,
                         group_name: str | None = None,
                         max_categories: int = 20) -> Figure:
    """Plot one column stratified by group: ONE plot with every group side by side.

    - numerical: one box per group on a shared axis (quartiles, median, p1–p99 whiskers),
      each labeled with its n
    - categorical: grouped horizontal bars, one colored bar per group for each of the
      most frequent values overall
    - binary: grouped bars of negative / positive / missing per group
    - date: rows per year, one line per group
    - id: IDs by rows per ID, one line per group

    Color identifies a group only where position doesn't (bars, lines). Groups past the
    palette's eight colors are folded into "Other" there; box plots show every group.

    Arguments
    ---------
    values: pd.Series
        Raw values of the column.
    groups: pd.Series
        Group label of every row, aligned with ``values``.
    column_type: ColumnType
        Type of the column.
    column_name: str
        Name displayed in the title.
    group_order: list[str] | None (default None)
        Groups to plot, in order. Defaults to every group, most frequent first.
    group_name: str | None (default None)
        Name of the grouping, shown in the title as "<column> by <group_name>".
    max_categories: int (default 20)
        Maximum categorical values plotted (the most frequent over all groups).

    Returns
    -------
    Figure
        Matplotlib figure owned by the caller.
    """
    if not values.index.equals(groups.index):
        raise ValueError("values and groups must share the same index")
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    labels: pd.Series = groups.astype("string")
    order: list[str] = ([str(group) for group in group_order] if group_order is not None
                        else labels.value_counts(sort=True).index.tolist())
    labels = labels.where(labels.isin(order))
    title: str = column_name if group_name is None else f"{column_name} by {group_name}"

    if column_type == "numerical":
        stats, names = [], []
        for group in order:
            group_stats = _numeric_stats(values.loc[labels.eq(group)])
            if group_stats is not None:
                stats.append(group_stats)
                names.append(f"{_short(group)}\nn = {style.compact(group_stats['n'])}\n"
                             f"min {style.compact(group_stats['min'])} · max {style.compact(group_stats['max'])}")
        fig, ax = style.new_figure(max(7.0, 1.15 * len(names) + 2.5), 5.5)
        style.style_axes(ax, grid="y", value_axis="y")
        if stats:
            _draw_boxes(ax, stats, names)
            if len(names) > 5:
                ax.tick_params(axis="x", labelrotation=35)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment("right")
        else:
            style.no_data(ax, "No numeric values")
        ax.set_ylabel("Value")
        style.set_titles(ax, title, f"{len(stats)} groups · box = quartiles, line = median, whiskers p1–p99")

    elif column_type in ("categorical", "binary"):
        # One row per group; a 100% bar split into the group's values. A group with an
        # unexpected value (e.g. a second unit) shows a second color in its row.
        if column_type == "binary":
            def binary_counts(series: pd.Series) -> pd.Series:
                row = summarize_binary(pd.DataFrame({"v": series}), ["v"]).iloc[0]
                positive = int(row["positive_count"])
                return pd.Series({"Negative": int(row["non_missing_count"]) - positive,
                                  "Positive": positive, "Missing": int(row["missing_count"])})
            counts = pd.DataFrame({group: binary_counts(values.loc[labels.eq(group)]) for group in order}).T
            segments = ["Negative", "Positive", "Missing"]
            totals = counts.sum(axis=1)
        else:
            counts = pd.DataFrame({group: (values.loc[labels.eq(group)].astype("string")
                                           .fillna("Missing").value_counts())
                                   for group in order}).fillna(0).T  # rows = groups, columns = values
            totals = counts.sum(axis=1)  # before "Other" is added, so it isn't counted twice
            ranked = counts.sum().sort_values(ascending=False).index.tolist()
            values_kept = [v for v in ranked if v != "Missing"][:min(max_categories, style.MAX_COLORED_GROUPS - 1)]
            rest = [v for v in ranked if v not in values_kept and v != "Missing"]
            if rest:
                counts[f"Other ({len(rest)} values)"] = counts[rest].sum(axis=1)
            segments = values_kept + ([f"Other ({len(rest)} values)"] if rest else []) \
                + (["Missing"] if "Missing" in counts.columns else [])
        shares = counts[segments].div(totals.where(totals > 0, 1), axis=0) * 100
        colors = [style.DE_EMPHASIS if s == "Missing" else style.OTHER if s.startswith("Other (")
                  else style.SERIES[i] for i, s in enumerate(segments)]
        fig, ax = style.new_figure(10, max(3.2, 0.42 * len(order) + 2.0))
        style.style_axes(ax, grid="x", value_axis=None)
        if counts.empty:
            style.no_data(ax, "No values")
        else:
            rows = [f"{_short(group)}  (n = {style.compact(totals[group])})" for group in order][::-1]
            left = np.zeros(len(order))
            for segment, color in zip(segments, colors):
                widths = shares.loc[order[::-1], segment].to_numpy()
                ax.barh(rows, widths, left=left, height=0.6, color=color, edgecolor=style.SURFACE,
                        linewidth=1.5, label=_short(segment))
                left += widths
            ax.set_xlim(0, 100)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
            style.style_legend(ax, column_name)
        ax.set_xlabel("Share of the group's rows")
        style.set_titles(ax, title, f"{len(order)} groups · each bar is 100% of its group")

    elif column_type in ("date", "id"):
        shown, labels = _fold_groups(order, labels)
        colors = _group_colors(shown)
        fig, ax = style.new_figure(10, 5)
        style.style_axes(ax, grid="y", value_axis="y")
        largest = 0
        for i, group in enumerate(shown):
            series = values.loc[labels.eq(group)]
            if column_type == "date":
                points = pd.to_datetime(series, errors="coerce").dt.year.value_counts().sort_index()
            else:
                points = series.dropna().astype("string").value_counts().value_counts().sort_index()
            if points.empty:
                continue
            largest = max(largest, int(points.index.max()))
            ax.plot(points.index.astype(int), points.to_numpy(), color=colors[i], linewidth=2,
                    marker="o", markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=2,
                    label=_short(group))
        if column_type == "id" and largest > 100:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(style.compact))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year" if column_type == "date" else "Rows per ID")
        ax.set_ylabel("Rows" if column_type == "date" else "IDs")
        if ax.get_lines():
            style.style_legend(ax, group_name)
        else:
            style.no_data(ax, "No values")
        style.set_titles(ax, title, f"{len(order)} groups")

    else:
        raise ValueError(f"Unsupported column type: {column_type}")

    fig.tight_layout()
    return fig
