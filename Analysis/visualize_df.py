"""Create one Matplotlib figure per summarized DataFrame column and group."""

from typing import Collection, Iterable, Iterator, Literal

import pandas as pd
from matplotlib.figure import Figure

from .summarize_column import ColumnType, _COLUMN_TYPES
from .summarize_df import ALL_GROUPS, _configured_subset, _split_columns, classify_df
from .visualize_column import visualize_column, visualize_stratified


def visualize_df(df: pd.DataFrame | dict[str, pd.DataFrame],
                 configured_col_types: dict[ColumnType, Iterable[str]] | None = None,
                 group_by: str | Collection[str] | None = None,
                 group_types: list[ColumnType] | Literal["all"] = "all",
                 max_groups: int = 1000,
                 max_categories: int = 20,
                 group_columns: Collection[str] | None = None) -> dict[tuple[str, str | None], Figure]:
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
    group_columns: Collection[str] | None (default None)
        Raw data only: stratify these columns by ``group_by``. Each gets ONE figure with
        a panel per group (see ``visualize_stratified``); every other column, including
        the grouping columns, gets one figure over all rows. All figures are keyed with
        group ``None``. Requires ``group_by`` and cannot be combined with
        ``group_types``. See ``summarize_df`` for when to use it.

    Returns
    -------
    dict[tuple[str, str | None], Figure]
        Figures keyed by ``(column, group)``. The caller closes each figure.
    """
    return dict(iter_visualizations(df=df, configured_col_types=configured_col_types,
                                    group_by=group_by, group_types=group_types,
                                    max_groups=max_groups, max_categories=max_categories,
                                    group_columns=group_columns))


def iter_visualizations(df: pd.DataFrame | dict[str, pd.DataFrame],
                        configured_col_types: dict[ColumnType, Iterable[str]] | None = None,
                        group_by: str | Collection[str] | None = None,
                        group_types: list[ColumnType] | Literal["all"] = "all",
                        max_groups: int = 1000,
                        max_categories: int = 20,
                        group_columns: Collection[str] | None = None,
                        ) -> Iterator[tuple[tuple[str, str | None], Figure]]:
    """Yield column figures one at a time so callers can save and close them.

    Arguments are as in :func:`visualize_df`."""
    if group_columns is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("group_columns applies to raw DataFrames, not summary tables")
        by_columns = [group_by] if isinstance(group_by, str) else list(group_by) if group_by is not None else None
        split = _split_columns(df, by_columns, group_columns, group_types)
        if max_groups < 1:
            raise ValueError("max_groups must be at least 1")
        # Unsplit columns (and the grouping columns): one figure each over all rows
        whole = df.drop(columns=split)
        yield from iter_visualizations(whole, configured_col_types=_configured_subset(configured_col_types, whole.columns),
                                       max_categories=max_categories)
        if not split:
            return
        # Split columns: ONE stratified figure each, with a panel per group (most frequent first)
        column_types = classify_df(df=df[split], configured_col_types=_configured_subset(configured_col_types, split))
        labels = (df[by_columns[0]] if len(by_columns) == 1 else df[by_columns].agg(tuple, axis=1))\
            .astype("string").fillna("<missing>")
        order = labels.value_counts(sort=True).head(max_groups).index.tolist()
        group_name = " | ".join(by_columns)
        for column, column_type in column_types.items():
            yield (column, None), visualize_stratified(df[column], labels, column_type=column_type,
                                                       column_name=column, group_order=order,
                                                       group_name=group_name, max_categories=max_categories)
        return

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
                group = "<missing>" if pd.isna(key[1]) else None if key[1] == ALL_GROUPS else str(key[1])
            yield (column, group), visualize_column(
                rows, column_type=column_type, column_name=column,
                max_categories=max_categories, summaries=summaries, group=group,
            )
