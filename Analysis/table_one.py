"""
Build a "Table 1": characteristics of a cohort, overall and by group, formatted for a report.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

OVERALL = "Overall"


def _number(value: float) -> str:
    """ Two decimals below 10 (pH 7.39, lactate 1.70), one decimal above (heart rate 85.0). """
    if pd.isna(value):
        return ""
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"


def _median_iqr(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return f"{_number(median)} [{_number(q1)}, {_number(q3)}]"


def _count_pct(hits: int, total: int) -> str:
    return f"{hits:,} ({100 * hits / total:.1f}%)" if total else ""


def table_one(
    df: pd.DataFrame,
    group_column: str,
    numerical: Iterable[str] = (),
    binary: Iterable[str] = (),
    categorical: Iterable[str] = (),
    group_order: Iterable[object] | None = None,
) -> pd.DataFrame:
    """
    Describe each variable overall and within each group of ``group_column``.

    Parameters
    ----------
    df: pd.DataFrame
        One row per observation.
    group_column: str
        Column whose values define the groups (e.g. race). Rows where it is missing count
        toward ``Overall`` only.
    numerical: Iterable[str]
        Reported as median [Q1, Q3] of the non-missing values.
    binary: Iterable[str]
        0/1 columns, reported as the count (%) of 1s among non-missing values.
    categorical: Iterable[str]
        Reported as one row per level: count (%) among non-missing values.
    group_order: Iterable[object] | None (default None)
        Order of the group columns; groups not listed follow in order of size. Listed
        groups that do not occur are left out.

    Returns
    -------
    table: pd.DataFrame
        Columns ``variable``, ``level``, ``statistic``, ``Overall``, one column per group,
        and ``missing_pct`` (of all rows). The first row is the row count.
    """

    numerical, binary, categorical = list(numerical), list(binary), list(categorical)
    requested = [group_column, *numerical, *binary, *categorical]
    absent = [column for column in requested if column not in df.columns]
    if absent:
        raise KeyError(f"Columns are missing from the DataFrame: {absent}")
    overlap = (set(numerical) & set(binary)) | (set(numerical) & set(categorical)) | (set(binary) & set(categorical))
    if overlap:
        raise ValueError(f"Columns listed under more than one type: {sorted(overlap)}")

    # --- Groups: requested order first, then the rest by size ---
    by_size = df[group_column].value_counts().index.tolist()
    listed = [group for group in (group_order or []) if group in by_size]
    groups = [*listed, *[group for group in by_size if group not in listed]]
    subsets: dict[object, pd.DataFrame] = {OVERALL: df, **{group: df.loc[df[group_column].eq(group)] for group in groups}}

    def cells(describe) -> dict[str, str]:
        """ `describe(subset)` for Overall and every group. """
        return {str(name): describe(subset) for name, subset in subsets.items()}

    def missing_pct(column: str) -> float:
        return round(100 * df[column].isna().mean(), 2) if len(df) else 0.0

    def share(subset: pd.DataFrame, column: str, value: object) -> str:
        """ Count (%) of `value` among the non-missing values of `column`. """
        return _count_pct(int(subset[column].eq(value).sum()), int(subset[column].notna().sum()))

    # --- Rows: the row count, then each variable ---
    rows: list[dict[str, object]] = [
        {"variable": "n", "level": "", "statistic": "count", **cells(lambda s: f"{len(s):,}"), "missing_pct": None}
    ]
    for column in numerical:
        rows.append({"variable": column, "level": "", "statistic": "median [Q1, Q3]",
                     **cells(lambda s: _median_iqr(s[column])), "missing_pct": missing_pct(column)})
    for column in binary:
        rows.append({"variable": column, "level": "", "statistic": "n (%)",
                     **cells(lambda s: share(s, column, 1)), "missing_pct": missing_pct(column)})
    for column in categorical:
        for level in df[column].dropna().value_counts().index:
            rows.append({"variable": column, "level": str(level), "statistic": "n (%)",
                         **cells(lambda s: share(s, column, level)), "missing_pct": missing_pct(column)})

    return pd.DataFrame(rows, columns=["variable", "level", "statistic", *map(str, subsets), "missing_pct"])
