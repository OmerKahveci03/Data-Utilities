"""Plot a column from its summary statistics."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .summarize_column import (
    ColumnType, classify_column, summarize_binary, summarize_categorical,
    summarize_dates, summarize_ids, summarize_numerical,
)


def visualize_column(values: pd.Series | pd.DataFrame,
                     column_type: ColumnType | None = None,
                     column_name: str | None = None,
                     max_categories: int = 20) -> Figure:
    """Return a plot appropriate for one column.

    Arguments
    ---------
    values: pd.Series | pd.DataFrame
        Raw column values or summary rows for one column.
    column_type: ColumnType | None (default None)
        Type of plot to create. Required for summary rows; inferred for a Series.
    column_name: str | None (default None)
        Column name to show in the plot title.
    max_categories: int (default 20)
        Maximum number of categorical values shown.

    Returns
    -------
    Figure
        Matplotlib figure owned by the caller.
    """
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")

    if isinstance(values, pd.Series):
        name = column_name or str(values.name or "column")
        kind = column_type or classify_column(name, values)
        source = pd.DataFrame({name: values})
        summarize = {
            "binary": summarize_binary, "categorical": summarize_categorical,
            "date": summarize_dates, "id": summarize_ids,
            "numerical": summarize_numerical,
        }[kind]
        summary = summarize(source, [name])
    else:
        if column_type is None:
            raise ValueError("column_type is required for summary rows")
        kind = column_type
        summary = values
        name = column_name or (str(summary["column"].iloc[0]) if not summary.empty else "column")

    fig, ax = plt.subplots(figsize=(8, 5))
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif kind == "categorical":
        categories = summary.dropna(subset=["value"]).nlargest(max_categories, "value_count")
        ax.barh(categories["value"].astype(str).iloc[::-1], categories["value_count"].iloc[::-1])
        ax.set_xlabel("Count")
    elif kind == "binary":
        row = summary.iloc[0]
        positive = int(row["positive_count"])
        non_missing = int(row["non_missing_count"])
        ax.bar(["Negative", "Positive", "Missing"],
               [non_missing - positive, positive, int(row["missing_count"])])
        ax.set_ylabel("Count")
    elif kind == "numerical":
        row = summary.iloc[0]
        if pd.notna(row["min"]) and pd.notna(row["max"]):
            ax.bxp([{"med": row["median"], "q1": row["p25"], "q3": row["p75"],
                     "whislo": row["min"], "whishi": row["max"], "fliers": []}],
                   showfliers=False)
            ax.set_xticks([])
            ax.set_ylabel("Value")
        else:
            ax.text(0.5, 0.5, "No numeric values", ha="center", va="center", transform=ax.transAxes)
    elif kind == "date":
        row = summary.iloc[0]
        start, end = pd.to_datetime(row["min"]), pd.to_datetime(row["max"])
        if pd.notna(start) and pd.notna(end):
            ax.plot([start, end], [0, 0], marker="|")
            ax.set_yticks([])
            fig.autofmt_xdate()
        else:
            ax.text(0.5, 0.5, "No valid dates", ha="center", va="center", transform=ax.transAxes)
    elif kind == "id":
        row = summary.iloc[0]
        ax.bar(["Unique", "Missing"], [int(row["unique_count"]), int(row["missing_count"])])
        ax.set_ylabel("Count")
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported column type: {kind}")

    ax.set_title(name)
    fig.tight_layout()
    return fig
