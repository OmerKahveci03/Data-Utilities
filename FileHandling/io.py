import os
import glob
import re
import logging
import pandas as pd
import pyarrow.parquet as pq

from typing import Optional, Collection, Callable, Iterator
from ..Logging.convenience import _get_logger

def get_file_types(data_dir: str, 
                   ext: Optional[str] = "parquet", 
                   require_batch_number: bool = True,
                   logger: Optional[logging.Logger] = None,
                   recursive: bool = False) -> list[str]:
    """Return sorted file type prefixes from source files in ``data_dir``.

    A prefix ends before the first underscore followed by digits. For example,
    ``visits_0_1.parquet`` has the prefix ``visits``.

    Arguments
    ---------
    data_dir: str
        Directory to scan.
    ext: Optional[str] (default 'parquet')
        Filename extension without a dot, or ``None`` for any extension.
    require_batch_number: bool (default True)
        Ignore names without an underscore-number segment
        when true.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.
    recursive: bool (default False)
        Search subdirectories and prefix nested types with their
        immediate parent directory name.

    Returns
    -------
        Sorted, unique prefixes found in matching filenames.
    """

    if ext is None:
        ext = "*"

    groups = set()
    pattern = os.path.join(data_dir, "**", f"*.{ext}") if recursive else os.path.join(data_dir, f"*.{ext}")
    files = glob.glob(pattern, recursive=recursive)

    for file in files:
        if not file.endswith((".csv", ".csv.gz", ".parquet")):
            continue
        name = re.sub(r"\.(csv(\.gz)?|parquet)$", "", os.path.basename(file))
        if recursive and os.path.dirname(file) != data_dir:
            name = f"{os.path.basename(os.path.dirname(file))}_{name}"
        # Split before the first numeric segment

        has_batch = bool(re.search(r"_\d+", name))
        if require_batch_number and not has_batch:
            continue

        prefix = re.split(r"_\d+", name, maxsplit=1)[0]
        if prefix:
            groups.add(prefix)
    result = sorted(groups)
    _get_logger(logger).debug(f"Found {len(result)} file types in {data_dir}")
    return result

def peak_columns(fp: str, sheet_name: Optional[str] = None,
                 logger: Optional[logging.Logger] = None) -> list[str]:
    """Return column names from a file.

    Arguments
    ---------
    fp: str
        Path to the file.
    sheet_name: Optional[str] (default None)
        Sheet to inspect for spreadsheet files.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.

    Notes
    -----
        This function is not implemented yet and currently returns ``None``.
    """

def get_files(data_dir: str, file_type: str, batch: Optional[int] = None,
              logger: Optional[logging.Logger] = None,
              recursive: bool = False) -> list[str]:
    """Find CSV and Parquet files for a file type.

    Matches ``<type>_<number>`` and ``<type>_<number>_<number>`` names.
    When ``batch`` is ``None`` or zero, bare ``<type>`` files also match.

    Arguments
    ---------
    data_dir: str
        Directory to scan.
    file_type: str
        Filename prefix to match.
    batch: Optional[int] (default None)
        Optional first numeric segment to select.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.
    recursive: bool (default False)
        Search subdirectories and prefix nested types with their
        immediate parent directory name.

    Returns
    -------
        Sorted paths to matching files.
    """

    if batch is not None:
        batch_extension = str(batch)
    else:
        batch_extension = "[0-9]+"
    
    escaped_type = re.escape(file_type)
    patterns = [
        re.compile(rf"^{escaped_type}_{batch_extension}\.parquet$"),
        re.compile(rf"^{escaped_type}_{batch_extension}_[0-9]+\.parquet$"),
        re.compile(rf"^{escaped_type}_{batch_extension}\.csv(\.gz)?$"),
        re.compile(rf"^{escaped_type}_{batch_extension}_[0-9]+\.csv(\.gz)?$"),
    ]
    if not batch:
        patterns.append(re.compile(rf"^{escaped_type}\.parquet$"))
        patterns.append(re.compile(rf"^{escaped_type}\.csv(\.gz)?$"))

    found_files: list[str] = []
    paths = glob.glob(os.path.join(data_dir, "**", "*"), recursive=True) if recursive else [
        os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
    ]
    for path in paths:
        if not os.path.isfile(path):
            continue
        fname = os.path.basename(path)
        if recursive and os.path.dirname(path) != data_dir:
            fname = f"{os.path.basename(os.path.dirname(path))}_{fname}"
        for pattern in patterns:
            if pattern.match(fname):
                found_files.append(path)
                break  # prevents double matching

    result = sorted(found_files)
    _get_logger(logger).debug(f"Found {len(result)} {file_type} files in {data_dir}")
    return result

def load_df(files: str | Collection[str], 
            usecols: Optional[list[str]] = None,
            parse_dates: Optional[list[str]] = None,
            filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            logger: Optional[logging.Logger] = None,
            chunk_size: Optional[int] = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a CSV or Parquet file into a DataFrame.

    Arguments
    ---------
    files: str | Collection[str]
        A path or collection of paths.
    usecols: Optional[list[str]] (default None)
        Columns to read, or ``None`` for all columns.
    parse_dates: Optional[list[str]] (default None)
        Columns to convert to UTC datetimes after reading.
    filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] (default None)
        Optional transformation applied to the loaded DataFrame.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.
    chunk_size: Optional[int] (default None)
        Return an iterator of DataFrames of this size when set.

    Returns
    -------
        The loaded DataFrame, or an iterator of DataFrames when ``chunk_size``
        is set.

    Raises
    ------
        ValueError: If a filename has an unsupported extension.
    """

    if isinstance(files, str):
        files = [files]

    log = _get_logger(logger)

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        def read_chunks() -> Iterator[pd.DataFrame]:
            for file in files:
                log.debug(f"Reading {file}")
                if file.endswith((".csv", ".csv.gz")):
                    chunks = pd.read_csv(file, usecols=usecols, chunksize=chunk_size,
                                         low_memory=False)
                elif file.endswith(".parquet"):
                    parquet = pq.ParquetFile(file)
                    chunks = (batch.to_pandas() for batch in
                              parquet.iter_batches(batch_size=chunk_size, columns=usecols))
                else:
                    raise ValueError(f"Unsupported file: {file}")

                for df in chunks:
                    if parse_dates:
                        for dt_col in parse_dates:
                            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)

                    if callable(filter_func):
                        df = filter_func(df)
                    yield df

        return read_chunks()

    dfs: list[pd.DataFrame] = []
    for file in files:
        log.debug(f"Reading {file}")
        if file.endswith((".csv", ".csv.gz")):
            df = pd.read_csv(file, usecols=usecols)
        elif file.endswith(".parquet"):
            df = pd.read_parquet(file, columns=usecols)
        else:
            raise ValueError(f"Unsupported file: {file}")

        if callable(filter_func):
            df = filter_func(df)
        dfs.append(df)

    if not dfs:
        raise ValueError("No files were supplied")

    df = pd.concat(dfs, ignore_index=True)

    # NOTE: Parsing after concat because pd.concat may unparse the column.
    if parse_dates:
        for dt_col in parse_dates:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    log.debug(f"Loaded {len(df):,} rows")
    return df


def write_df(df: pd.DataFrame, write_fp: str,
             logger: Optional[logging.Logger] = None) -> None:
    """Write a DataFrame as CSV or Parquet without its index.

    Creates the parent directory if needed.

    Arguments
    ---------
    df: pd.DataFrame
        DataFrame to write.
    write_fp: str
        Destination path ending in ``.csv`` or ``.parquet``.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.

    Raises
    ------
        ValueError: If the destination has an unsupported extension.
    """

    # NOTE: May potentially make this a parameter that can be passed.
    index: bool = False

    parent_dir: str = os.path.dirname(write_fp)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if write_fp.endswith(".csv"):
        df.to_csv(write_fp, index=index)

    elif write_fp.endswith(".parquet"):
        df.to_parquet(write_fp, index=index)

    else:
        raise ValueError(f"Unsupported write_fp: {write_fp}")
    _get_logger(logger).debug(f"Wrote {len(df)} rows to {write_fp}")
