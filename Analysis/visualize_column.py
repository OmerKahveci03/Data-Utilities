"""Create a Matplotlib figure for each supported column type."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .summarize_column import (
    ColumnType, classify_column, summarize_binary, summarize_categorical,
    summarize_dates, summarize_ids, summarize_numerical,
)


def visualize_binary(values: pd.Series | pd.DataFrame, column_name: str) -> Figure:
    """Plot negative, positive, and missing counts for a binary column."""
    summary = (summarize_binary(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not summary.empty:
        row = summary.iloc[0]
        positive = int(row["positive_count"])
        ax.bar(["Negative", "Positive", "Missing"],
               [int(row["non_missing_count"]) - positive, positive, int(row["missing_count"])])
    ax.set_ylabel("Count")
    ax.set_title(column_name)
    fig.tight_layout()
    return fig


def visualize_categorical(values: pd.Series | pd.DataFrame, column_name: str,
                          max_categories: int = 20) -> Figure:
    """Plot the most frequent values of a categorical column."""
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    summary = (summarize_categorical(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    categories = (summary.dropna(subset=["value"]).nlargest(max_categories, "value_count")
                  if not summary.empty else summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not categories.empty:
        ax.barh(categories["value"].astype(str).iloc[::-1], categories["value_count"].iloc[::-1])
    ax.set_xlabel("Count")
    ax.set_title(column_name)
    fig.tight_layout()
    return fig


def visualize_date(values: pd.Series | pd.DataFrame, column_name: str,
                   summaries: dict[str, pd.DataFrame] | None = None,
                   group: str | None = None) -> Figure:
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
    fig, ax = plt.subplots(figsize=(8, 5))
    if histogram.empty:
        ax.text(0.5, 0.5, "No date histogram available", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = histogram.sort_values("year")
        ax.bar(bins["year"].astype(int), bins["date_count"].astype(int), width=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of rows")
    ax.set_title(column_name)
    fig.tight_layout()
    return fig


def visualize_id(values: pd.Series | pd.DataFrame, column_name: str,
                 summaries: dict[str, pd.DataFrame] | None = None,
                 group: str | None = None) -> Figure:
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
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig.tight_layout()
    return fig


def visualize_numerical(values: pd.Series | pd.DataFrame, column_name: str) -> Figure:
    """Plot a box summary for a numerical column."""
    summary = (summarize_numerical(pd.DataFrame({column_name: values}), [column_name])
               if isinstance(values, pd.Series) else values)
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig.tight_layout()
    return fig


def visualize_column(values: pd.Series | pd.DataFrame,
                     column_type: ColumnType | None = None,
                     column_name: str | None = None,
                     max_categories: int = 20,
                     summaries: dict[str, pd.DataFrame] | None = None,
                     group: str | None = None) -> Figure:
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
        return visualize_binary(values, name)
    if kind == "categorical":
        return visualize_categorical(values, name, max_categories=max_categories)
    if kind == "date":
        return visualize_date(values, name, summaries=summaries, group=group)
    if kind == "id":
        return visualize_id(values, name, summaries=summaries, group=group)
    if kind == "numerical":
        return visualize_numerical(values, name)
    raise ValueError(f"Unsupported column type: {kind}")
