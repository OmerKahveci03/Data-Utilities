"""
Classify DataFrame columns and produce validation summaries by data type.

The module provides reusable helpers for summarizing binary, categorical, date,
identifier, and numerical columns. Each summary is returned as a DataFrame and
includes missingness metrics alongside statistics appropriate for the column type.
Values that cannot be converted by a type-specific summarizer are treated as missing.
"""

import pandas as pd

from typing import Literal, Optional, TypeAlias, TypedDict


_PERCENTILES: list[float] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

ColumnType: TypeAlias = Literal["binary", "categorical", "date", "id", "numerical"]
_COLUMN_TYPES: tuple[ColumnType, ...] = ("binary", "categorical", "date", "id", "numerical")

class MissingnessMetrics(TypedDict):
    """Fixed schema for the metrics returned by :func:`missingness`."""

    row_count: int
    missing_count: int
    missing_pct: float
    non_missing_count: int


def missingness(values: pd.Series) -> MissingnessMetrics:
    """
    Calculate row counts and missingness metrics for a Series.

    Parameters
    ----------
    values: pd.Series
        Values to assess. Missing values are identified with ``Series.isna()``.

    Returns
    -------
    metrics: MissingnessMetrics
        Row count, missing count, percentage missing, and non-missing count. The
        missing percentage is ``NaN`` when the Series is empty.
    """

    row_count = len(values)
    missing_count = int(values.isna().sum())
    return {
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_pct": missing_count / row_count * 100 if row_count else float("nan"),
        "non_missing_count": row_count - missing_count,
    }


def classify_column(column: str, values: pd.Series) -> ColumnType:
    """
    Infer the summary type for a column from its name, dtype, and observed values.

    Classification is applied in priority order. Names containing an ``id`` token
    are classified as identifiers. Datetime dtypes and names containing ``date`` or
    ``datetime`` are classified as dates. Boolean dtypes and columns whose non-missing
    values are numeric and limited to zero and one are binary; other numeric columns
    are numerical, and all remaining columns are categorical.

    Parameters
    ----------
    column: str
        Name of the column being classified.
    values: pd.Series
        Values used to infer the column type. Numeric strings are considered numeric,
        and missing values are ignored when identifying binary columns.
    Returns
    -------
    column_type: ColumnType
        One of ``"binary"``, ``"categorical"``, ``"date"``, ``"id"``, or
        ``"numerical"``.
    """

    column: str = column.lower()

    # NOTE: Concept ids are categorical
    if "concept_id" in column:
        return "categorical"

    # --- ID columns contain 'id' as a token ---
    column_tokens: set[str] = set(column.split("_"))
    if "id" in column_tokens:
        return "id"

    # --- Datetime columns contain 'date'
    if pd.api.types.is_datetime64_any_dtype(values) or "date" in column:
        return "date"

    # --- Binary columns are either bool, or purely contain 0/1 ---
    if pd.api.types.is_bool_dtype(values.dtype):
        return "binary"
    non_missing: pd.Series = values.dropna()
    numeric_values: pd.Series = pd.to_numeric(non_missing, errors="coerce")
    is_numeric: bool = len(non_missing) > 0 and numeric_values.notna().all()
    if is_numeric and set(numeric_values.unique()).issubset({0, 1}):
        return "binary"

    # --- Numeric columns only contain numbers ---
    # NOTE: It is important to cast certain categorical columns to string because they may
    # contains numeric-like values (CPT codes, or_case_num, concept_ids for example)
    if is_numeric or pd.api.types.is_numeric_dtype(values):
        return "numerical"

    # --- Otherwise, this is categorical ---
    return "categorical"


def summarize_binary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Summarize prevalence and missingness for binary columns.

    Values are converted to numeric with unconvertible values treated as missing.
    A positive value is exactly ``1``, and prevalence uses the non-missing count as
    its denominator.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every requested column.
    columns: list[str]
        Names of binary columns to summarize.

    Returns
    -------
    summary: pd.DataFrame
        One row per column with missingness metrics, positive count, and prevalence
        percentage. Prevalence is ``NaN`` when a column has no non-missing values.
    """

    records: list[dict] = []
    for column in columns:
        values: pd.Series = pd.to_numeric(df[column], errors="coerce")
        metrics: MissingnessMetrics = missingness(values=values)
        positive_count = int(values.eq(1).sum())
        non_missing_count = int(metrics["non_missing_count"])
        records.append({
            "column": column,
            **metrics,
            "positive_count": positive_count,
            "prevalence_pct": positive_count / non_missing_count * 100 if non_missing_count else float("nan"),
        })
    return pd.DataFrame(records)


def summarize_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Summarize observed values and missingness for categorical columns.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every requested column.
    columns: list[str]
        Names of categorical columns to summarize. Values are converted to pandas'
        nullable string dtype before counting.

    Returns
    -------
    summary: pd.DataFrame
        One row per observed non-missing value, with its count and percentage of all
        rows, plus column-level cardinality and missingness metrics. An entirely missing
        column is represented by one row with a missing value and counts of zero.
    """

    records: list[dict] = []
    for column in columns:
        values: pd.Series = df[column].astype("string")
        metrics: MissingnessMetrics = missingness(values=values)
        counts: pd.Series = values.value_counts(dropna=True, sort=True)
        records.extend({
            "column": column,
            "value": value,
            "value_count": int(count),
            "unique_count": len(counts),
            "value_pct": count / len(values) * 100 if len(values) else float("nan"),
            **metrics,
        } for value, count in counts.items())
        if counts.empty:
            records.append({"column": column, "value": pd.NA, "value_count": 0, "unique_count": 0, "value_pct": 0, **metrics})
    return pd.DataFrame(records)


def summarize_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Summarize ranges and missingness for date and datetime columns.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every requested column.
    columns: list[str]
        Names of date or datetime columns to summarize. Values are converted with
        ``pandas.to_datetime``; unconvertible values are treated as missing.

    Returns
    -------
    summary: pd.DataFrame
        One row per column with missingness metrics and the earliest and latest valid
        timestamps. Bounds are ``NaT`` when no valid timestamps are present.
    """

    records: list[dict] = []
    for column in columns:
        values: pd.Series = pd.to_datetime(df[column], errors="coerce")
        records.append({"column": column, **missingness(values=values), "min": values.min(), "max": values.max()})
    return pd.DataFrame(records)


def summarize_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Summarize cardinality and missingness for identifier columns.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every requested column.
    columns: list[str]
        Names of identifier columns to summarize.

    Returns
    -------
    summary: pd.DataFrame
        One row per column with missingness metrics and the count of distinct
        non-missing identifiers.
    """

    records: list[dict] = []
    for column in columns:
        values: pd.Series = df[column]
        records.append({"column": column, **missingness(values=values), "unique_count": values.nunique(dropna=True)})
    return pd.DataFrame(records)


def summarize_numerical(df: pd.DataFrame, columns: list[str], percentiles: Optional[list[float]] = None) -> pd.DataFrame:
    """
    Summarize distributions and missingness for numerical columns.

    Parameters
    ----------
    df: pd.DataFrame
        Data containing every requested column.
    columns: list[str]
        Names of numerical columns to summarize. Values are converted to numeric,
        with unconvertible values treated as missing.
    percentiles: Optional[list[float]] (default None)
        Quantiles to calculate, expressed as values from zero through one. When
        omitted, :data:`_PERCENTILES` is used. Output names are formatted as ``pNN``.

    Returns
    -------
    summary: pd.DataFrame
        One row per column with missingness metrics, minimum, mean, median, maximum,
        sample standard deviation, and the requested percentiles.
    """

    if percentiles is None:
        percentiles = _PERCENTILES
    records: list[dict] = []
    for column in columns:
        values: pd.Series = pd.to_numeric(df[column], errors="coerce")
        metrics: MissingnessMetrics = missingness(values=values)
        quantile_values: pd.Series = values.quantile(percentiles)
        records.append({
            "column": column,
            **metrics,
            "min": values.min(),
            "mean": values.mean(),
            "median": values.median(),
            "max": values.max(),
            "std": values.std(),
            **{f"p{int(percentile * 100):02d}": quantile_values.loc[percentile] for percentile in percentiles},
        })
    return pd.DataFrame(records)