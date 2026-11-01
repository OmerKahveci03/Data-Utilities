"""Classify and summarize every column in a pandas DataFrame."""

from typing import Collection, Iterable, Literal, Optional

import pandas as pd

from .summarize_column import (
    ColumnType,
    _COLUMN_TYPES,
    classify_column,
    summarize_binary,
    summarize_categorical,
    summarize_dates,
    summarize_ids,
    summarize_numerical,
)

def classify_df(
    df: pd.DataFrame,
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] = None,
) -> dict[str, ColumnType]:
    """
    Classify every DataFrame column, honoring caller-configured type overrides.

    Caller-configured types take precedence over inferred types. Every column not
    included in ``configured_col_types`` is classified from its name, dtype, and
    observed values with :func:`classify_column`.

    Parameters
    ----------
    df: pd.DataFrame
        Data whose columns will be classified and summarized. Column names must be
        strings.
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] (default None)
        Optional mapping from a column type to columns that must use that type. For
        example, ``{"id": ["person_key"], "categorical": ["numeric_code"]}``
        prevents those columns from being inferred incorrectly. A column may be
        configured under only one type, and every configured column must exist in
        ``df``.

    Returns
    -------
    column_types: dict[str, ColumnType]
        Mapping from every DataFrame column to its configured or inferred type.

    Raises
    ------
    KeyError
        If a configured column does not exist in ``df``.
    TypeError
        If a DataFrame column name is not a string.
    ValueError
        If an unsupported type is configured or a column is configured under more
        than one type.
    """

    # --- Columns must be strings ---
    non_string_columns: list[object] = [column for column in df.columns if not isinstance(column, str)]
    if non_string_columns:
        raise TypeError(f"All DataFrame column names must be strings; invalid columns: {non_string_columns}")
    column_names: list[str] = [column for column in df.columns if isinstance(column, str)]

    # --- Build column to type mapping ---
    configured_types_by_column: dict[str, ColumnType] = {}
    if configured_col_types is not None:
        for column_type, columns in configured_col_types.items():
            if column_type not in _COLUMN_TYPES:
                raise ValueError(f"Unsupported configured column type: {column_type}")
            for column in columns:
                if column not in df.columns:
                    raise KeyError(f"Configured column is missing from the DataFrame: {column}")
                if column in configured_types_by_column:
                    previous_type: ColumnType = configured_types_by_column[column]
                    raise ValueError(f"Column '{column}' is configured as both '{previous_type}' and '{column_type}'")
                configured_types_by_column[column] = column_type

    return {
        column: configured_types_by_column[column] if column in configured_types_by_column else classify_column(column=column, values=df[column])
        for column in column_names
    }


def summarize_df(
    df: pd.DataFrame,
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] = None,
    entire_dataset: bool = False,
    group_by: str | Collection[str] | None = None,
    group_types: list[ColumnType] | Literal["all"] = "all",
    max_groups: int = 1000,
) -> dict[str, pd.DataFrame]:
    """
    Classify every DataFrame column and run each type-specific summarizer.

    Caller-configured types take precedence over inferred types. Columns not listed
    in ``configured_col_types`` are classified by :func:`classify_df`.

    Parameters
    ----------
    df: pd.DataFrame
        Data whose columns will be classified and summarized.
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] (default None)
        Optional mapping from a column type to columns that must use that type.
    entire_dataset: bool (default False)
        If True, also include dataset dimensions and a column inventory so the result
        can be passed directly to ``write_dataset_summary``.
    group_by: str | Collection[str] | None (default None)
        One column or a collection of columns used to calculate separate summaries.
        Missing values form groups. For multiple columns, ``group`` contains a tuple
        of values in the order supplied.
    group_types: list[ColumnType] | Literal["all"] (default "all")
        Column types to summarize by group when ``group_by`` is supplied. Other types
        receive one summary for the full DataFrame.
    max_groups: int (default 1000)
        Maximum number of most frequent groups to summarize.

    Returns
    -------
    summaries: dict[str, pd.DataFrame]
        Mapping containing one summary DataFrame for each supported column type.
        When entire_dataset is True, also contains dataset and columns summaries.

    """

    column_types: dict[str, ColumnType] = classify_df(df=df, configured_col_types=configured_col_types)
    columns_by_type: dict[ColumnType, list[str]] = {
        column_type: [column for column, assigned_type in column_types.items() if assigned_type == column_type]
        for column_type in _COLUMN_TYPES
    }

    if max_groups < 1:
        raise ValueError("max_groups must be at least 1")
    by_columns = [group_by] if isinstance(group_by, str) else list(group_by) if group_by is not None else None
    if group_types != "all":
        if not isinstance(group_types, list) or any(column_type not in _COLUMN_TYPES for column_type in group_types):
            raise ValueError(f"group_types must be 'all' or a list of supported column types: {group_types}")
    grouped_types = set(_COLUMN_TYPES if group_types == "all" else group_types)
    if by_columns is not None:
        if not by_columns:
            raise ValueError("group_by must contain at least one column")
        missing_group_columns = [column for column in by_columns if column not in df.columns]
        if missing_group_columns:
            raise KeyError(f"Grouping columns are missing from the DataFrame: {missing_group_columns}")

    groups = [(None, df)] if by_columns is None else df.groupby(by_columns, dropna=False, sort=False, observed=True)
    if by_columns is not None:
        top_groups = groups.size().sort_values(ascending=False, kind="stable").head(max_groups).index
        grouped_df = groups
        groups = ((group, grouped_df.get_group((group,) if len(by_columns) == 1 else group))
                  for group in top_groups)
    grouped_summaries: dict[str, list[pd.DataFrame]] = {}
    for group, group_df in groups:
        summaries: dict[str, pd.DataFrame] = {
            "binary": summarize_binary(df=group_df, columns=columns_by_type["binary"]),
            "categorical": summarize_categorical(df=group_df, columns=columns_by_type["categorical"]),
            "date": summarize_dates(df=group_df, columns=columns_by_type["date"]),
            "id": summarize_ids(df=group_df, columns=columns_by_type["id"]),
            "numerical": summarize_numerical(df=group_df, columns=columns_by_type["numerical"]),
        }
        if entire_dataset:
            summaries = {
                "dataset": pd.DataFrame([{"row_count": len(group_df), "column_count": len(df.columns)}]),
                "columns": summarize_columns(df=group_df, column_types=column_types),
                **summaries,
            }
        for name, summary in summaries.items():
            if by_columns is not None and name in _COLUMN_TYPES and name not in grouped_types:
                continue
            if by_columns is not None and name != "columns":
                group_value = group if len(by_columns) == 1 else tuple(
                    value.item() if hasattr(value, "item") else value for value in group
                )
                summary.insert(0, "group", [group_value] * len(summary))
            grouped_summaries.setdefault(name, []).append(summary)

    if entire_dataset:
        grouped_summaries["columns"] = [summarize_columns(df=df, column_types=column_types)]

    if by_columns is not None:
        whole_summaries = summarize_df(df=df, configured_col_types=configured_col_types)
        for column_type in _COLUMN_TYPES:
            if column_type not in grouped_types:
                grouped_summaries[column_type] = [whole_summaries[column_type]]
        if df.empty:
            for column_type in grouped_types:
                if column_type not in grouped_summaries:
                    empty_summary = whole_summaries[column_type].iloc[0:0].copy()
                    empty_summary.insert(0, "group", pd.Series(dtype="object"))
                    grouped_summaries[column_type] = [empty_summary]
            if entire_dataset:
                grouped_summaries["dataset"] = [pd.DataFrame(columns=["group", "row_count", "column_count"])]

    if grouped_summaries:
        return {name: pd.concat(frames, ignore_index=True) for name, frames in grouped_summaries.items()}
    # An empty grouped input has no observed groups, but retains the expected tables.
    empty_summaries = summarize_df(df=df, configured_col_types=configured_col_types, entire_dataset=entire_dataset)
    for name, summary in empty_summaries.items():
        if name in _COLUMN_TYPES and name not in grouped_types:
            continue
        if name != "columns":
            summary.insert(0, "group", pd.Series(dtype="object"))
        summary.drop(index=summary.index, inplace=True)
    return empty_summaries


def summarize_columns(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType],
    example_count: int = 3,
) -> pd.DataFrame:
    """
    Describe each DataFrame column with its assigned type and provide example values.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every column in ``column_types``.
    column_types: dict[str, ColumnType]
        Assigned type for each column to include, in the desired output order.
    example_count: int (default 3)
        Maximum number of distinct, non-missing string examples to retain.

    Returns
    -------
    column_summary: pd.DataFrame
        One row per column with ``column``, ``column_type``, and numbered example
        columns from ``example_1`` through ``example_N``.
    """

    if example_count < 0:
        raise ValueError(f"example_count must be non-negative, but received {example_count}")
    missing_columns: list[str] = [column for column in column_types if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns are missing from the DataFrame: {missing_columns}")

    output_columns: list[str] = ["column", "column_type", *[f"example_{i + 1}" for i in range(example_count)]]
    records: list[dict[str, object]] = []
    for column, column_type in column_types.items():
        # --- Include n unique example values ---
        examples: list[object] = df[column].dropna().astype("string").drop_duplicates().head(example_count).tolist()
        record: dict[str, object] = {"column": column, "column_type": column_type}
        for i, example in enumerate(examples):
            record[f"example_{i + 1}"] = example
        records.append(record)
    return pd.DataFrame(records, columns=output_columns)
