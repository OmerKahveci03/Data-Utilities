import os
import glob
import re
import pandas as pd

from typing import Optional, Collection, Callable

def get_file_types(data_dir: str, 
                   ext: Optional[str] = "parquet", 
                   require_batch_number: bool = True) -> list[str]:
    """
    Returns a list of unique file type prefixes found in data_dir.
    Works with patterns like:
        - file_name_0.parquet
        - file_name_0_0.parquet

    It keeps everything before the first numeric segment as the group name.

    Example:
        clinical_encounters_0.parquet
        clinical_encounters_1.parquet
        dialysis_0_0.parquet
    -> returns ['clinical_encounters', 'dialysis']
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
    return sorted(groups)

def peak_columns(fp: str, sheet_name: Optional[str] = None) -> list[str]:
    """ Return the columns belonging to the file."""

def get_files(data_dir: str, file_type: str, batch: Optional[int] = None) -> list[str]:
    """ """

def load_df(files: str | Collection[str], 
            usecols: Optional[list[str]] = None,
            filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    """ Load tabular data from one or more files. """


def write_df(df: pd.DataFrame, write_fp: str) -> None:
    """ Write the dataframe to disk. """

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
