import os
import glob
import re
import logging
import pandas as pd

from typing import Optional, Collection, Callable

_NULL_LOGGER = logging.Logger("data_utilities.null", level=logging.CRITICAL)


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Return the supplied logger or a logger that discards debug messages."""
    return logger if logger is not None else _NULL_LOGGER

def get_file_types(data_dir: str, 
                   ext: Optional[str] = "parquet", 
                   require_batch_number: bool = True,
                   logger: Optional[logging.Logger] = None) -> list[str]:
    """Return sorted file type prefixes from files directly in ``data_dir``.

    A prefix ends before the first underscore followed by digits. For example,
    ``visits_0_1.parquet`` has the prefix ``visits``.

    Args:
        data_dir: Directory to scan; subdirectories are not searched.
        ext: Filename extension without a dot, or ``None`` for any extension.
        require_batch_number: Ignore names without an underscore-number segment
            when true.
        logger: Optional logger for debug messages.

    Returns:
        Sorted, unique prefixes found in matching filenames.
    """

    if ext is None:
        ext = "*"

    groups = set()
    files = glob.glob(os.path.join(data_dir, f"*.{ext}"))

    for file in files:
        name = os.path.splitext(os.path.basename(file))[0]
        # Split before the first numeric segment

        has_batch = bool(re.search(r"_\d+", name))
        if require_batch_number and not has_batch:
            continue

        prefix = re.split(r"_\d+", name, maxsplit=1)[0]
        if prefix:
            groups.add(prefix)
    result = sorted(groups)
    _get_logger(logger).debug("Found %d file types in %s", len(result), data_dir)
    return result

def peak_columns(fp: str, sheet_name: Optional[str] = None,
                 logger: Optional[logging.Logger] = None) -> list[str]:
    """Return column names from a file.

    Args:
        fp: Path to the file.
        sheet_name: Sheet to inspect for spreadsheet files.
        logger: Optional logger for debug messages.

    Note:
        This function is not implemented yet and currently returns ``None``.
    """

def get_files(data_dir: str, file_type: str, batch: Optional[int] = None,
              logger: Optional[logging.Logger] = None) -> list[str]:
    """Find CSV and Parquet files for a file type in one directory.

    Matches ``<type>_<number>`` and ``<type>_<number>_<number>`` names.
    When ``batch`` is ``None`` or zero, bare ``<type>`` files also match.

    Args:
        data_dir: Directory to scan; subdirectories are not searched.
        file_type: Filename prefix to match.
        batch: Optional first numeric segment to select.
        logger: Optional logger for debug messages.

    Returns:
        Sorted paths to matching files.
    """

    if batch is not None:
        batch_extension = str(batch)
    else:
        batch_extension = "[0-9]+"
    
    patterns = [
        re.compile(rf"^{file_type}_{batch_extension}\.parquet$"),
        re.compile(rf"^{file_type}_{batch_extension}_[0-9]+\.parquet$"),
        re.compile(rf"^{file_type}_{batch_extension}\.csv$"),
        re.compile(rf"^{file_type}_{batch_extension}_[0-9]+\.csv$"),
    ]
    if not batch:
        patterns.append(re.compile(rf"^{file_type}\.parquet$"))
        patterns.append(re.compile(rf"^{file_type}\.csv$"))

    found_files: list[str] = []
    for fname in os.listdir(data_dir):
        for pattern in patterns:
            if pattern.match(fname):
                found_files.append(os.path.join(data_dir, fname))
                break  # prevents double matching

    result = sorted(found_files)
    _get_logger(logger).debug("Found %d %s files in %s", len(result), file_type, data_dir)
    return result

def load_df(files: str | Collection[str], 
            usecols: Optional[list[str]] = None,
            parse_dates: Optional[list[str]] = None,
            filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
            logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Args:
        files: A path or collection of paths. If several paths are supplied,
            the current implementation returns only the last file's data.
        usecols: Columns to read, or ``None`` for all columns.
        parse_dates: Columns to convert to UTC datetimes after reading.
        filter_func: Optional transformation applied to the loaded DataFrame.
        logger: Optional logger for debug messages.

    Returns:
        The loaded and optionally transformed DataFrame.

    Raises:
        ValueError: If a filename has an unsupported extension.
    """

    if isinstance(files, str):
        files = [files]

    log = _get_logger(logger)
    for file in files:
        log.debug("Reading %s", file)
        if file.endswith(".csv"):
            df = pd.read_csv(file, usecols=usecols)
        elif file.endswith(".parquet"):
            df = pd.read_parquet(file, columns=usecols)
        else:
            raise ValueError(f"Unsupported file: {file}")

    if parse_dates:
        for dt_col in parse_dates:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)

    if callable(filter_func):
        df = filter_func(df)

    log.debug("Loaded %d rows", len(df))
    return df


def write_df(df: pd.DataFrame, write_fp: str,
             logger: Optional[logging.Logger] = None) -> None:
    """Write a DataFrame as CSV or Parquet without its index.

    Creates the parent directory if needed.

    Args:
        df: DataFrame to write.
        write_fp: Destination path ending in ``.csv`` or ``.parquet``.
        logger: Optional logger for debug messages.

    Raises:
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
    _get_logger(logger).debug("Wrote %d rows to %s", len(df), write_fp)
