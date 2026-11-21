"""Create a Matplotlib figure for each supported column type."""

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .summarize_column import (
    ColumnType, classify_column, summarize_binary, summarize_categorical,
    summarize_dates, summarize_ids, summarize_numerical,
)


def _figure_and_axes(ax: Axes | None) -> tuple[Figure, Axes, bool]:
    """Return the figure and axes to draw on, and whether a new figure was created."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        return fig, ax, True
    return ax.figure, ax, False


def _finish(fig: Figure, created: bool) -> Figure:
    if created:
        fig.tight_layout()
    return fig


def visualize_binary(values: pd.Series | pd.DataFrame, column_name: str, ax: Axes | None = None) -> Figure:
    """Plot negative, positive, and missing counts for a binary column."""
    summary = (summarize_binary(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    fig, ax, created = _figure_and_axes(ax)
    if not summary.empty:
        row = summary.iloc[0]
        positive = int(row["positive_count"])
        ax.bar(["Negative", "Positive", "Missing"],
               [int(row["non_missing_count"]) - positive, positive, int(row["missing_count"])])
    ax.set_ylabel("Count")
    ax.set_title(column_name)
    return _finish(fig, created)


def visualize_categorical(values: pd.Series | pd.DataFrame, column_name: str,
                          max_categories: int = 20, ax: Axes | None = None) -> Figure:
    """Plot the most frequent values of a categorical column."""
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    summary = (summarize_categorical(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    categories = (summary.dropna(subset=["value"]).nlargest(max_categories, "value_count")
                  if not summary.empty else summary)
    fig, ax, created = _figure_and_axes(ax)
    if not categories.empty:
        ax.barh(categories["value"].astype(str).iloc[::-1], categories["value_count"].iloc[::-1])
    ax.set_xlabel("Count")
    ax.set_title(column_name)
    return _finish(fig, created)


def visualize_date(values: pd.Series | pd.DataFrame, column_name: str,
                   summaries: dict[str, pd.DataFrame] | None = None,
                   group: str | None = None, ax: Axes | None = None) -> Figure:
    """Plot the number of observations in each calendar year."""
    if isinstance(values, pd.Series):
        counts = pd.to_datetime(values, errors="coerce").dt.year.value_counts()
        histogram = pd.DataFrame({"year": counts.index, "date_count": counts.values})
    else:
        histogram = pd.DataFrame() if summaries is None else summaries.get("date_histogram", pd.DataFrame())
        if not histogram.empty:
            histogram = histogram.loc[histogram["column"].eq(column_name)]
            if group is not None and "group" in histogram.columns:
                histogram = histogram.loc[histogram["group"].astype("string").eq(group)]
    fig, ax, created = _figure_and_axes(ax)
    if histogram.empty:
        ax.text(0.5, 0.5, "No date histogram available", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = histogram.sort_values("year")
        ax.bar(bins["year"].astype(int), bins["date_count"].astype(int), width=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of rows")
    ax.set_title(column_name)
    return _finish(fig, created)


def visualize_id(values: pd.Series | pd.DataFrame, column_name: str,
                 summaries: dict[str, pd.DataFrame] | None = None,
                 group: str | None = None, ax: Axes | None = None) -> Figure:
    """Plot how many identifiers have each number of associated rows."""
    if isinstance(values, pd.Series):
        counts = values.dropna().astype("string").value_counts().value_counts()
        histogram = pd.DataFrame({"row_count": counts.index, "id_count": counts.values})
    else:
        histogram = pd.DataFrame() if summaries is None else summaries.get("id_histogram", pd.DataFrame())
        if not histogram.empty:
            histogram = histogram.loc[histogram["column"].eq(column_name)]
            if group is not None and "group" in histogram.columns:
                histogram = histogram.loc[histogram["group"].astype("string").eq(group)]
    fig, ax, created = _figure_and_axes(ax)
    if histogram.empty:
        ax.text(0.5, 0.5, "No ID histogram available", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = histogram.sort_values("row_count")
        ax.bar(bins["row_count"].astype(int), bins["id_count"].astype(int), width=0.9)
        if bins["row_count"].max() > 100:
            ax.set_xscale("log")
    ax.set_xlabel("Rows per ID")
    ax.set_ylabel("Number of IDs")
    ax.set_title(column_name)
    return _finish(fig, created)


def visualize_numerical(values: pd.Series | pd.DataFrame, column_name: str, ax: Axes | None = None) -> Figure:
    """Plot a box summary for a numerical column."""
    summary = (summarize_numerical(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    fig, ax, created = _figure_and_axes(ax)
    if not summary.empty and pd.notna(summary.iloc[0]["min"]) and pd.notna(summary.iloc[0]["max"]):
        row = summary.iloc[0]
        ax.bxp([{"med": row["median"], "q1": row["p25"], "q3": row["p75"],
                 "whislo": row["min"], "whishi": row["max"], "fliers": []}],
               showfliers=False)
        ax.set_xticks([])
    else:
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Value")
    ax.set_title(column_name)
    return _finish(fig, created)


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
        Axes to draw on, e.g. one panel of a larger figure. A new figure is created
        when omitted.

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


def visualize_stratified(values: pd.Series,
                         groups: pd.Series,
                         column_type: ColumnType,
                         column_name: str,
                         group_order: list[str] | None = None,
                         group_name: str | None = None,
                         max_categories: int = 20,
                         max_columns: int = 3) -> Figure:
    """Plot one column stratified by group: one figure, one panel per group.

    Each panel is the column's normal plot for that group's rows, titled with the group
    label and drawn on its own axes, so groups measured in different units (e.g. heart
    rate and temperature sharing a `valuenum` column) stay readable.

    Arguments
    ---------
    values: pd.Series
        Raw values of the column.
    groups: pd.Series
        Group label of every row, aligned with ``values``.
    column_type: ColumnType
        Type of the column.
    column_name: str
        Name displayed in the figure title.
    group_order: list[str] | None (default None)
        Groups to plot, in panel order. Defaults to every group, most frequent first.
    group_name: str | None (default None)
        Name of the grouping, shown in the figure title as "<column> by <group_name>".
    max_categories: int (default 20)
        Maximum categorical values per panel.
    max_columns: int (default 3)
        Maximum panels per row.

    Returns
    -------
    Figure
        Matplotlib figure owned by the caller.
    """
    if not values.index.equals(groups.index):
        raise ValueError("values and groups must share the same index")
    order: list[str] = (list(group_order) if group_order is not None
                        else groups.value_counts(sort=True).index.astype(str).tolist())
    panels: int = max(1, len(order))
    columns: int = min(max_columns, panels)
    rows: int = math.ceil(panels / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.8 * rows), squeeze=False)
    for ax, group in zip(axes.flat, order):
        visualize_column(values.loc[groups.eq(group)], column_type=column_type, column_name=str(group),
                         max_categories=max_categories, ax=ax)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    if not order:
        axes.flat[0].set_visible(True)
        axes.flat[0].text(0.5, 0.5, "No groups", ha="center", va="center", transform=axes.flat[0].transAxes)
    fig.suptitle(column_name if group_name is None else f"{column_name} by {group_name}")
    fig.tight_layout()
    return fig
