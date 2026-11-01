"""Combine DataFrame summaries and coordinate ordered batch processing."""

import logging
import os
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Collection, Iterable, Literal, Optional

import pandas as pd

from ..FileHandling.io import get_files, load_df, write_df
from .summarize_column import ColumnType, _COLUMN_TYPES, summarize_numerical
from .summarize_df import classify_df, summarize_columns, summarize_df


DatasetSummary = dict[str, pd.DataFrame]
DATASET_SUMMARY_NAMES: tuple[str, ...] = (
    "dataset", "columns", "binary", "categorical", "date", "id", "numerical"
)


def read_dataset_summary(path: str) -> DatasetSummary:
    """Read and validate a dataset summary workbook."""

    summaries: dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    missing_sheets: list[str] = sorted(set(DATASET_SUMMARY_NAMES) - set(summaries))
    if missing_sheets:
        raise KeyError(f"Dataset summary workbook is missing sheets: {missing_sheets}")
    return {summary_name: summaries[summary_name] for summary_name in DATASET_SUMMARY_NAMES}


def write_dataset_summary(summary: DatasetSummary, path: str) -> str:
    """Write the provided summary tables to an Excel workbook and return its path."""

    if not summary:
        raise ValueError("At least one summary table is required")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ordered_summary_names: list[str] = [
            *[name for name in DATASET_SUMMARY_NAMES if name in summary],
            *sorted(set(summary) - set(DATASET_SUMMARY_NAMES)),
        ]
        for summary_name in ordered_summary_names:
            summary[summary_name].to_excel(writer, sheet_name=summary_name, index=False)
    return path


def _summarize_dataset_batch(
    batch: int,
    data_dir: str,
    write_dir: str,
    file_name: str,
    configured_col_types: dict[ColumnType, list[str]],
    disjoint_id_columns: list[str],
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
    overwrite: bool,
    group_by: str | Collection[str] | None,
    group_types: list[ColumnType] | Literal["all"],
    selected_groups: Optional[set[str]],
) -> DatasetSummary:
    """Load, summarize, and cache one batch for :func:`summarize_dataset`."""

    logger: logging.Logger = logging.getLogger(f"summarize-dataset-{file_name}-{batch}")
    # --- Skip Duplicate Work ---
    result_names: list[str] = ["dataset", "columns", "binary", "categorical", "date", "id", "id_values", "numerical_values"]
    cache_extension: str = ".csv" if batch == 0 else ".parquet"
    cache_paths: dict[str, str] = {
        result_name: os.path.join(write_dir, "_cache", f"{file_name}_{result_name}_{batch}{cache_extension}")
        for result_name in result_names
    }
    cache_exists: bool = all(os.path.exists(cache_path) for cache_path in cache_paths.values())
    if group_by is None and not overwrite and cache_exists:
        return {result_name: load_df(files=cache_path, logger=logger) for result_name, cache_path in cache_paths.items()}


    batch_files: list[str] = get_files(data_dir=data_dir, file_type=file_name, batch=batch, logger=logger)
    if not batch_files:
        raise FileNotFoundError(f"No CSV or Parquet files found for '{file_name}' batch {batch} in {data_dir}")
    df: pd.DataFrame = load_df(files=batch_files, logger=logger)
    if filter_func is not None:
        df = filter_func(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"filter_func must return a pandas DataFrame, but returned {type(df).__name__}")

    # --- Clean Blank Text ---
    text_columns: pd.Index = df.select_dtypes(include=["object", "string"]).columns
    df.loc[:, text_columns] = df.loc[:, text_columns].replace(r"^\s*$", pd.NA, regex=True)

    group_columns: Optional[list[str]] = [group_by] if isinstance(group_by, str) else list(group_by) if group_by is not None else None
    if group_columns is not None and selected_groups is not None:
        labels = df[group_columns].astype("string").fillna("<missing>").agg(" | ".join, axis=1)
        df = df.loc[labels.isin(selected_groups)].copy()

    # --- Classify Column Types ---
    present_configured_types: dict[ColumnType, list[str]] = {
        column_type: [column for column in columns if column in df.columns]
        for column_type, columns in configured_col_types.items()
    }
    column_types: dict[str, ColumnType] = classify_df(df=df, configured_col_types=present_configured_types)

    # --- Obtain Summaries for each Column Type ---
    summaries: DatasetSummary = summarize_df(df=df, configured_col_types=present_configured_types,
                                             group_by=group_by, group_types=group_types,
                                             entire_dataset=group_by is not None)

    if group_columns is not None:
        group_labels = pd.Series(index=df.index, dtype="string")
        for group, positions in df.groupby(group_columns, dropna=False, sort=False, observed=True).indices.items():
            group_value = group[0] if len(group_columns) == 1 else tuple(
                value.item() if hasattr(value, "item") else value for value in group
            )
            group_labels.iloc[positions] = pd.Series([group_value]).astype("string").iloc[0]
        for summary in summaries.values():
            if "group" in summary.columns:
                summary["group"] = summary["group"].astype("string")

    # NOTE: Returning non-disjoint ids because they cannot be aggregated via sum to produce
    # unique counts.
    id_columns: list[str] = [column for column, column_type in column_types.items() if column_type == "id"]
    exact_id_columns: list[str] = [column for column in id_columns if column not in disjoint_id_columns]
    id_source = df[exact_id_columns].astype("string").copy()
    if group_columns is not None and (group_types == "all" or "id" in group_types):
        id_source.insert(0, "group", group_labels)
    id_values: pd.DataFrame = id_source.melt(id_vars=["group"] if "group" in id_source.columns else None, var_name="column", value_name="value")\
        .dropna(subset=["value"])\
        .drop_duplicates()

    # NOTE: Return the numerical values instead of running summaries because numerical summaries
    # cannot be aggregated correctly. 
    numerical_columns: list[str] = [column for column, column_type in column_types.items() if column_type == "numerical"]

    numerical_values = df[numerical_columns].copy()
    if group_columns is not None and (group_types == "all" or "numerical" in group_types):
        numerical_values.insert(0, "group", group_labels)
    if group_columns is not None:
        dataset = summaries["dataset"].copy()
        dataset.insert(0, "batch", batch)
        columns_summary = summaries["columns"]
    else:
        dataset = pd.DataFrame([{"batch": batch, "row_count": len(df), "column_count": len(df.columns)}])
        columns_summary = summarize_columns(df=df, column_types=column_types)
    results: DatasetSummary = {
        "dataset": dataset,
        "columns": columns_summary,
        "binary": summaries["binary"],
        "categorical": summaries["categorical"],
        "date": summaries["date"],
        "id": summaries["id"],
        "id_values": id_values,
        "numerical_values": numerical_values,
    }
    if group_by is None:
        for result_name, result in results.items():
            write_df(df=result, write_fp=cache_paths[result_name], logger=logger)
    return results


def combine_df_summaries(
    summaries: Iterable[dict[ColumnType, pd.DataFrame]],
    id_values: Optional[Iterable[pd.DataFrame]] = None,
    numerical_values: Optional[Iterable[pd.DataFrame]] = None,
    max_categorical_values: Optional[int] = None,
    group_by: str | Collection[str] | None = None,
    group_types: list[ColumnType] | Literal["all"] = "all",
) -> dict[ColumnType, pd.DataFrame]:
    """
    Combine per-batch DataFrame summaries into one dataset-level summary.

    Some summary statistics cannot be aggregated by applying the same summarizer to
    the batch summary rows. Counts can be summed, but percentages must be recalculated
    from the combined counts; categorical values must be regrouped across batches;
    date bounds must use the global minimum and maximum; and overlapping identifiers
    must be deduplicated before calculating global uniqueness. Numerical means,
    standard deviations, and percentiles are recalculated from the retained numerical
    values because batch-level distribution statistics cannot be combined directly.

    This function handles those type-specific aggregation rules so multiple results
    from :func:`summarize_df` represent the complete dataset correctly.

    Parameters
    ----------
    summaries: Iterable[dict[ColumnType, pd.DataFrame]]
        Batch results produced by ``summarize_df``.
    id_values: Optional[Iterable[pd.DataFrame]] (default None)
        Optional frames with ``column`` and ``value`` fields used to calculate global
        distinct counts. Columns not represented retain their summed batch counts.
    numerical_values: Optional[Iterable[pd.DataFrame]] (default None)
        Optional raw numerical frames used to calculate exact global distributions.
    max_categorical_values: Optional[int] (default None)
        Maximum number of globally most frequent values retained for each
        categorical column. Remaining values are combined into one
        ``<Other (remaining)>`` row. All values are retained when omitted.
    group_by: str | list[str] | None (default None)
        Grouping columns used to produce the batch summaries.
    group_types: list[ColumnType] | Literal["all"] (default "all")
        Column types whose summaries are grouped.

    Returns
    -------
    combined: dict[ColumnType, pd.DataFrame]
        Combined binary, categorical, date, identifier, and numerical summaries.
        The numerical summary is empty when raw numerical values are not supplied.
    """

    if max_categorical_values is not None and max_categorical_values < 1:
        raise ValueError(f"max_categorical_values must be at least 1, but received {max_categorical_values}")
    if group_types != "all" and (not isinstance(group_types, list) or any(value not in _COLUMN_TYPES for value in group_types)):
        raise ValueError(f"group_types must be 'all' or a list of supported column types: {group_types}")

    batch_summaries: list[dict[ColumnType, pd.DataFrame]] = list(summaries)
    if not batch_summaries:
        return {column_type: pd.DataFrame() for column_type in ("binary", "categorical", "date", "id", "numerical")}

    if group_by is not None:
        selected_types = set(_COLUMN_TYPES if group_types == "all" else group_types)
        id_frames = [] if id_values is None else list(id_values)
        numerical_frames = [] if numerical_values is None else list(numerical_values)
        ungrouped_summaries = [{column_type: frame if column_type not in selected_types else pd.DataFrame()
                                for column_type, frame in summary.items()} for summary in batch_summaries]
        combined_results = combine_df_summaries(
            summaries=ungrouped_summaries,
            id_values=id_frames if "id" not in selected_types else None,
            numerical_values=numerical_frames if "numerical" not in selected_types else None,
            max_categorical_values=max_categorical_values,
        )
        group_series = [frame["group"] for summary in batch_summaries for column_type, frame in summary.items()
                        if column_type in selected_types and "group" in frame.columns]
        if "numerical" in selected_types:
            group_series.extend(frame["group"] for frame in numerical_frames if "group" in frame.columns)
        if not group_series:
            return combined_results
        group_values = pd.concat(group_series, ignore_index=True).drop_duplicates()
        grouped_results: dict[ColumnType, list[pd.DataFrame]] = {column_type: [] for column_type in selected_types}
        for group in group_values:
            def belongs_to_group(frame: pd.DataFrame) -> pd.Series:
                """Select rows for the current group, including missing group labels."""
                return frame["group"].isna() if pd.isna(group) else frame["group"].eq(group)

            group_summaries = [{column_type: frame.loc[belongs_to_group(frame)].drop(columns="group")
                                if column_type in selected_types and "group" in frame.columns else pd.DataFrame()
                                for column_type, frame in summary.items()} for summary in batch_summaries]
            group_ids = [frame.loc[belongs_to_group(frame)].drop(columns="group") for frame in id_frames] if "id" in selected_types else None
            group_numerical = [frame.loc[belongs_to_group(frame)].drop(columns="group") for frame in numerical_frames] if "numerical" in selected_types else None
            combined_group = combine_df_summaries(summaries=group_summaries, id_values=group_ids,
                                                  numerical_values=group_numerical,
                                                  max_categorical_values=max_categorical_values)
            for column_type, frame in combined_group.items():
                if column_type not in selected_types:
                    continue
                frame.insert(0, "group", [group] * len(frame))
                grouped_results[column_type].append(frame)
        combined_results.update({column_type: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                                 for column_type, frames in grouped_results.items()})
        return combined_results

    # --- Binary results can be directly concatenated and summed ---
    binary_batches: list[pd.DataFrame] = [summary["binary"] for summary in batch_summaries if not summary["binary"].empty]
    binary = pd.DataFrame()
    if binary_batches:
        binary = pd.concat(binary_batches, ignore_index=True).groupby("column", as_index=False, sort=False).agg(
            row_count=("row_count", "sum"), missing_count=("missing_count", "sum"),
            non_missing_count=("non_missing_count", "sum"), positive_count=("positive_count", "sum"),
        )
        binary["missing_pct"] = binary["missing_count"] / binary["row_count"] * 100
        binary["prevalence_pct"] = binary["positive_count"] / binary["non_missing_count"] * 100

    # --- Combine category counts across batches and recalculate dataset-level percentages and rankings ---
    categorical_batches: list[pd.DataFrame] = [summary["categorical"] for summary in batch_summaries if not summary["categorical"].empty]
    categorical = pd.DataFrame()
    if categorical_batches:
        categorical_batch_metrics: list[pd.DataFrame] = [frame.drop_duplicates(subset=["column"]) for frame in categorical_batches]
        categorical_metrics: pd.DataFrame = pd.concat(categorical_batch_metrics, ignore_index=True)\
            .groupby("column", as_index=False, sort=False)\
            .agg(row_count=("row_count", "sum"), missing_count=("missing_count", "sum"), non_missing_count=("non_missing_count", "sum"))
        categorical_metrics["missing_pct"] = categorical_metrics["missing_count"] / categorical_metrics["row_count"] * 100
        categorical_values: pd.DataFrame = pd.concat(categorical_batches, ignore_index=True)
        categorical_counts: pd.DataFrame = categorical_values.loc[categorical_values["value"].notna()]\
            .groupby(["column", "value"], as_index=False, sort=False).agg(value_count=("value_count", "sum"))
        categorical = categorical_counts.merge(categorical_metrics, on="column", how="left", validate="many_to_one")
        categorical["unique_count"] = categorical.groupby("column", sort=False)["value"].transform("count")
        categorical["value_pct"] = categorical["value_count"] / categorical["row_count"] * 100
        columns_without_values: pd.DataFrame = categorical_metrics.loc[~categorical_metrics["column"].isin(categorical["column"])]\
            .assign(value=pd.NA, value_count=0, unique_count=0, value_pct=0)
        categorical = pd.concat([categorical, columns_without_values], ignore_index=True)
        has_category: pd.Series = categorical["value"].notna()
        categorical.loc[has_category, "ranking"] = categorical.loc[has_category].groupby("column", sort=False)["value_pct"]\
            .rank(method="first", ascending=False)
        categorical["ranking"] = categorical["ranking"].astype("Int64")
        if max_categorical_values is not None:
            is_remainder: pd.Series = categorical["ranking"].gt(max_categorical_values).fillna(False)
            remainder: pd.DataFrame = categorical.loc[is_remainder]
            retained_categories: pd.DataFrame = categorical.loc[~is_remainder].copy()
            if not remainder.empty:
                other_categories: pd.DataFrame = remainder.groupby("column", as_index=False, sort=False).agg(
                    value_count=("value_count", "sum"), row_count=("row_count", "first"),
                    missing_count=("missing_count", "first"), missing_pct=("missing_pct", "first"),
                    non_missing_count=("non_missing_count", "first"), unique_count=("unique_count", "first"),
                )
                other_categories["value"] = "<Other (remaining)>"
                other_categories["value_pct"] = other_categories["value_count"] / other_categories["row_count"] * 100
                other_categories["ranking"] = max_categorical_values + 1
                other_categories = other_categories.reindex(columns=categorical.columns)
                retained_categories = pd.concat([retained_categories, other_categories], ignore_index=True)
                retained_categories["ranking"] = retained_categories["ranking"].astype("Int64")
            categorical = retained_categories

    # --- Recalculating Date Results ---
    date_batches: list[pd.DataFrame] = [summary["date"] for summary in batch_summaries if not summary["date"].empty]
    dates = pd.DataFrame()
    if date_batches:
        combined_dates: pd.DataFrame = pd.concat(date_batches, ignore_index=True)
        # NOTE: CSV-backed batch-zero caches reload bounds as strings while Parquet caches retain datetime dtypes.
        combined_dates[["min", "max"]] = combined_dates[["min", "max"]].apply(pd.to_datetime, errors="coerce")
        dates = combined_dates.groupby("column", as_index=False, sort=False).agg(
            row_count=("row_count", "sum"), missing_count=("missing_count", "sum"),
            non_missing_count=("non_missing_count", "sum"), min=("min", "min"), max=("max", "max"),
        )
        dates["missing_pct"] = dates["missing_count"] / dates["row_count"] * 100

    # --- Sum disjoint ID counts and deduplicate overlapping IDs for exact dataset-level uniqueness --- 
    id_batches: list[pd.DataFrame] = [summary["id"] for summary in batch_summaries if not summary["id"].empty]
    ids = pd.DataFrame()
    if id_batches:
        ids = pd.concat(id_batches, ignore_index=True).groupby("column", as_index=False, sort=False).agg(
            row_count=("row_count", "sum"), missing_count=("missing_count", "sum"),
            non_missing_count=("non_missing_count", "sum"), unique_count=("unique_count", "sum"),
        )
        ids["missing_pct"] = ids["missing_count"] / ids["row_count"] * 100
        id_value_frames: list[pd.DataFrame] = [] if id_values is None else list(id_values)
        if id_value_frames:
            distinct_values: pd.DataFrame = pd.concat(id_value_frames, ignore_index=True).drop_duplicates()
            union_counts: pd.DataFrame = distinct_values.groupby("column", as_index=False, sort=False).agg(unique_count_union=("value", "nunique"))
            ids = ids.merge(union_counts, on="column", how="left", validate="one_to_one")
            ids["unique_count"] = ids["unique_count_union"].fillna(ids["unique_count"]).astype(int)
            ids = ids.drop(columns="unique_count_union")

    # --- Numerical Results Must be Built From Scratch Using all Values ---
    numerical_value_frames: list[pd.DataFrame] = [] if numerical_values is None else list(numerical_values)
    numerical = pd.DataFrame()
    if numerical_value_frames:
        all_numerical_values: pd.DataFrame = pd.concat(numerical_value_frames, ignore_index=True)
        numerical = summarize_numerical(df=all_numerical_values, columns=all_numerical_values.columns.to_list())

    return {"binary": binary, "categorical": categorical, "date": dates, "id": ids, "numerical": numerical}


def summarize_dataset(
    data_dir: str,
    write_dir: str,
    file_name: str,
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] = None,
    disjoint_id_columns: Optional[Iterable[str]] = None,
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    max_categorical_values: Optional[int] = 1000,
    max_workers: int = 1,
    overwrite: bool = False,
    group_by: str | Collection[str] | None = None,
    group_types: list[ColumnType] | Literal["all"] = "all",
    max_groups: int = 1000,
) -> DatasetSummary:
    """
    Summarize and aggregate every batch of a consistently structured dataset.

    Parameters
    ----------
    data_dir: str
        Directory containing ``<file_name>_<batch>.csv`` and/or
        ``<file_name>_<batch>.parquet`` files.
    write_dir: str
        Directory used for compact per-batch caches.
    file_name: str
        Base filename used to discover and load batches.
    configured_col_types: Optional[dict[ColumnType, Iterable[str]]] (default None)
        Optional authoritative column-type assignments. Configured columns absent
        from a particular batch are ignored for that batch.
    disjoint_id_columns: Optional[Iterable[str]] (default None)
        Identifier columns whose values are known to be disjoint across batches, so
        their batch unique counts can be summed. All other identifier values are
        unioned to calculate exact global distinct counts.
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] (default None)
        Optional function applied to each loaded batch before columns are classified
        and summarized. It must accept and return a pandas DataFrame and must be
        pickleable when multiple worker processes are used. Cached summaries are
        recomputed when a filter is supplied.
    max_categorical_values: Optional[int] (default 1000)
        Maximum number of globally most frequent values retained for each
        categorical column. Remaining values are combined into one
        ``<Other (remaining)>`` row. All values are retained when set to ``None``.
    max_workers: int (default 1)
        Maximum number of worker processes used to summarize batches.
    overwrite: bool (default False)
        Whether to recompute existing per-batch caches.
    group_by: str | Collection[str] | None (default None)
        One column or a collection of columns used to calculate separate summaries.
    group_types: list[ColumnType] | Literal["all"] (default "all")
        Column types to group; other types receive one dataset-wide summary.
    max_groups: int (default 1000)
        Maximum number of most frequent groups to summarize across all batches.

    Returns
    -------
    results: DatasetSummary
        Dataset metrics, first-batch column inventory, and globally aggregated
        binary, categorical, date, identifier, and numerical summaries.
    """

    # --- Determine Batch Count & Prepare Workers ---
    if group_types != "all" and (not isinstance(group_types, list) or any(value not in _COLUMN_TYPES for value in group_types)):
        raise ValueError(f"group_types must be 'all' or a list of supported column types: {group_types}")
    if max_groups < 1:
        raise ValueError("max_groups must be at least 1")
    batch_pattern = re.compile(rf"^{re.escape(file_name)}_(\d+)\.(?:csv|csv\.gz|parquet)$")
    batches: list[int] = sorted({int(match.group(1)) for name in os.listdir(data_dir)
                                 if (match := batch_pattern.match(name))})
    if not batches:
        raise FileNotFoundError(f"No batches for '{file_name}' were found in {data_dir}")
    normalized_config: dict[ColumnType, list[str]] = {
        column_type: list(columns) for column_type, columns in (configured_col_types or {}).items()
    }
    normalized_disjoint_ids: list[str] = [] if disjoint_id_columns is None else list(disjoint_id_columns)
    group_columns = [group_by] if isinstance(group_by, str) else list(group_by) if group_by is not None else None
    selected_groups: Optional[set[str]] = None
    if group_columns is not None:
        if not group_columns:
            raise ValueError("group_by must contain at least one column")
        group_counts: pd.Series = pd.Series(dtype="int64")
        for batch in batches:
            files = get_files(data_dir=data_dir, file_type=file_name, batch=batch)
            for chunk in load_df(files=files, usecols=None if filter_func else group_columns, chunk_size=50_000):
                if filter_func is not None:
                    chunk = filter_func(chunk)
                labels = chunk[group_columns].astype("string").fillna("<missing>").agg(" | ".join, axis=1)
                group_counts = group_counts.add(labels.value_counts(), fill_value=0)
        selected_groups = set(group_counts.sort_values(ascending=False, kind="stable").head(max_groups).index)
    worker = partial(
        _summarize_dataset_batch,
        data_dir=data_dir,
        write_dir=write_dir,
        file_name=file_name,
        configured_col_types=normalized_config,
        disjoint_id_columns=normalized_disjoint_ids,
        filter_func=filter_func,
        overwrite=overwrite,
        group_by=group_by,
        group_types=group_types,
        selected_groups=selected_groups,
    )

    # --- Pull all Batch Results ---
    if len(batches) <= 1 or max_workers <= 1:
        batch_results: list[DatasetSummary] = [worker(batch=batch) for batch in batches]
    else:
        results_by_batch: dict[int, DatasetSummary] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future[DatasetSummary], int] = {
                executor.submit(worker, batch=batch): batch for batch in batches
            }
            for future in as_completed(futures):
                batch: int = futures[future]
                results_by_batch[batch] = future.result()
        batch_results = [results_by_batch[batch] for batch in batches]

    # --- Combine Results ---
    type_summaries: list[dict[ColumnType, pd.DataFrame]] = [{
        "binary": result["binary"], "categorical": result["categorical"], "date": result["date"],
        "id": result["id"], "numerical": pd.DataFrame(),
    } for result in batch_results]
    combined: dict[ColumnType, pd.DataFrame] = combine_df_summaries(
        summaries=type_summaries,
        id_values=[result["id_values"] for result in batch_results],
        numerical_values=[result["numerical_values"] for result in batch_results],
        max_categorical_values=max_categorical_values,
        group_by=group_by,
        group_types=group_types,
    )

    # --- Custom Dataet Metrics ---
    batch_metrics: pd.DataFrame = pd.concat([result["dataset"] for result in batch_results], ignore_index=True)
    if group_by is None:
        dataset_metrics: pd.DataFrame = pd.DataFrame([{
            "batch_count": len(batch_results), "row_count": batch_metrics["row_count"].sum(),
            "column_count": batch_metrics["column_count"].iloc[0],
        }])
    else:
        dataset_metrics = batch_metrics.groupby("group", dropna=False, sort=False, as_index=False).agg(
            batch_count=("batch", "nunique"), row_count=("row_count", "sum"),
            column_count=("column_count", "first"),
        )
    columns_summary = batch_results[0]["columns"]
    if group_by is not None:
        columns_summary = pd.concat([result["columns"] for result in batch_results], ignore_index=True)\
            .drop_duplicates(subset=["column"])
    return {
        "dataset": dataset_metrics,
        "columns": columns_summary,
        "binary": combined["binary"],
        "categorical": combined["categorical"],
        "date": combined["date"],
        "id": combined["id"],
        "numerical": combined["numerical"],
    }
