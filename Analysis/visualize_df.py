"""Create one Matplotlib figure per summarized DataFrame column and group."""

from typing import Collection, Iterable, Iterator, Literal

import pandas as pd
from matplotlib.figure import Figure

from .summarize_column import ColumnType, _COLUMN_TYPES
from .summarize_df import classify_df
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
    if isinstance(df, pd.DataFrame):
        column_types = classify_df(df=df, configured_col_types=configured_col_types)
        group_columns = ([group_by] if isinstance(group_by, str)
                         else list(group_by) if group_by is not None else None)
        grouped_types = set(_COLUMN_TYPES if group_types == "all" else group_types)
        if max_groups < 1:
            raise ValueError("max_groups must be at least 1")
        groups = None
        top_groups = []
        if group_columns is not None:
            if not group_columns:
                raise ValueError("group_by must contain at least one column")
            groups = df.groupby(group_columns, dropna=False, sort=False, observed=True)
            top_groups = groups.size().sort_values(ascending=False, kind="stable").head(max_groups).index
        for column, column_type in column_types.items():
            if groups is None or column_type not in grouped_types:
                yield (column, None), visualize_column(df[column], column_type=column_type,
                                                       column_name=column, max_categories=max_categories)
                continue
            for key in top_groups:
                values = groups.get_group((key,) if len(group_columns) == 1 else key)[column]
                label = key if len(group_columns) == 1 else tuple(key)
                group = "<missing>" if len(group_columns) == 1 and pd.isna(label) else str(label)
                yield (column, group), visualize_column(values, column_type=column_type,
                                                        column_name=column, max_categories=max_categories)
        return

    summaries = df
    for column_type in _COLUMN_TYPES:
        summary = summaries.get(column_type)
        if summary is None or summary.empty:
            continue
        keys = ["column", "group"] if "group" in summary.columns else ["column"]
        for key, rows in summary.groupby(keys, dropna=False, sort=False):
            column = str(key[0] if isinstance(key, tuple) else key)
            group = None
            if isinstance(key, tuple) and len(key) > 1:
                group = "<missing>" if pd.isna(key[1]) else str(key[1])
            yield (column, group), visualize_column(
                rows, column_type=column_type, column_name=column,
                max_categories=max_categories, summaries=summaries, group=group,
            )
