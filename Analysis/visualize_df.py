"""Create one Matplotlib figure per summarized DataFrame column and group."""

from typing import Collection, Iterable, Iterator, Literal

import pandas as pd
from matplotlib.figure import Figure

from .summarize_column import ColumnType, _COLUMN_TYPES
from .summarize_df import summarize_df
from .visualize_column import visualize_column


def visualize_df(df: pd.DataFrame | dict[str, pd.DataFrame],
                 configured_col_types: dict[ColumnType, Iterable[str]] | None = None,
                 group_by: str | Collection[str] | None = None,
                 group_types: list[ColumnType] | Literal["all"] = "all",
                 max_groups: int = 1000,
                 max_categories: int = 20) -> dict[tuple[str, str | None], Figure]:
    """Return figures keyed by column name and optional group label.

    Arguments
    ---------
    df: pd.DataFrame | dict[str, pd.DataFrame]
        Raw DataFrame or the tables returned by ``summarize_df`` or
        ``summarize_dataset``.
    configured_col_types: dict[ColumnType, Iterable[str]] | None (default None)
        Column type overrides when ``df`` contains raw data.
    group_by: str | Collection[str] | None (default None)
        Grouping column or columns when ``df`` contains raw data.
    group_types: list[ColumnType] | Literal["all"] (default "all")
        Types to summarize separately by group.
    max_groups: int (default 1000)
        Maximum number of groups to plot from raw data.
    max_categories: int (default 20)
        Maximum number of values in each categorical plot.

    Returns
    -------
    dict[tuple[str, str | None], Figure]
        Figures keyed by ``(column, group)``. The caller closes each figure.
    """
    return dict(iter_visualizations(df=df, configured_col_types=configured_col_types,
                                    group_by=group_by, group_types=group_types,
                                    max_groups=max_groups, max_categories=max_categories))


def iter_visualizations(df: pd.DataFrame | dict[str, pd.DataFrame],
                        configured_col_types: dict[ColumnType, Iterable[str]] | None = None,
                        group_by: str | Collection[str] | None = None,
                        group_types: list[ColumnType] | Literal["all"] = "all",
                        max_groups: int = 1000,
                        max_categories: int = 20) -> Iterator[tuple[tuple[str, str | None], Figure]]:
    """Yield column figures one at a time so callers can save and close them."""
    summaries = (summarize_df(df=df, configured_col_types=configured_col_types,
                              group_by=group_by, group_types=group_types,
                              max_groups=max_groups)
                 if isinstance(df, pd.DataFrame) else df)
    for column_type in _COLUMN_TYPES:
        summary = summaries.get(column_type)
        if summary is None or summary.empty:
            continue
        keys = ["column", "group"] if "group" in summary.columns else ["column"]
        for key, rows in summary.groupby(keys, dropna=False, sort=False):
            column = str(key[0] if isinstance(key, tuple) else key)
            group = ("<missing>" if pd.isna(key[1]) else str(key[1])) if isinstance(key, tuple) and len(key) > 1 else None
            yield (column, group), visualize_column(
                rows, column_type=column_type, column_name=column,
                max_categories=max_categories,
            )
