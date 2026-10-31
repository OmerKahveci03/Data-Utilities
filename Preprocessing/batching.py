"""Build identifier lookup tables and write datasets in Parquet batches."""

import os
import logging
import pandas as pd
import numpy as np

from typing import Optional, Collection
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from ..FileHandling.io import get_files, load_df, write_df, get_file_types
from ..Logging.convenience import _get_logger
from .handlers.batch_dataset import process_file_type


def create_batch_lookup_table(data_dir: str, 
                              lookup_table_fp: str,
                              batch_by_id: str,
                              batch_size: int = 5_000,
                              master_file_type: Optional[str] = None,
                              overwrite: bool = False,
                              max_workers: Optional[int] = None,
                              logger: Optional[logging.Logger] = None,
                              recursive: bool = False) -> pd.DataFrame:
    """Map each distinct identifier to a zero-based batch and save the lookup.

    Identifiers are sorted before assignment. An existing lookup is loaded when
    ``overwrite`` is false.

    Arguments
    ---------
    data_dir: str
        Directory containing source files.
    lookup_table_fp: str
        Destination CSV or Parquet path for the lookup.
    batch_by_id: str
        Column containing the identifiers to batch.
    batch_size: int (default 5000)
        Maximum number of distinct identifiers per batch.
    master_file_type: Optional[str] (default None)
        File type containing all identifiers. Required for
        lookup creation; omitting it is not implemented.
    overwrite: bool (default False)
        Rebuild an existing lookup when true.
    max_workers: Optional[int] (default None)
        Reserved for future use; currently ignored.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages.
    recursive: bool (default False)
        Search nested source directories when true.

    Returns
    -------
        DataFrame with ``batch_by_id`` and ``batch`` columns.

    Raises
    ------
        NotImplementedError: If ``master_file_type`` is not supplied.
        ValueError: If ``batch_size`` is not positive when creating a lookup.
    """

    log = _get_logger(logger)
    # -- Return Early if Output Already Exists --
    if not overwrite and os.path.exists(lookup_table_fp):
        log.debug(f"Loading existing batch lookup from {lookup_table_fp}")
        return load_df(files=lookup_table_fp, usecols=[batch_by_id, "batch"], logger=log)

    # --- 1. Find All Identifiers ---
    if isinstance(master_file_type, str):
        all_files: list[str] = get_files(data_dir=data_dir, file_type=master_file_type,
                                         logger=log, recursive=recursive)
        if not all_files:
            raise FileNotFoundError(f"Master file type not found: {master_file_type}")

        all_ids: set = set()
        for file in all_files:
            for chunk in load_df(files=file, usecols=[batch_by_id],
                                 chunk_size=50_000, logger=log):
                all_ids.update(chunk[batch_by_id].dropna().unique().tolist())
        ids: np.ndarray = np.array(sorted(all_ids))

    else:
        raise NotImplementedError("Find all patients from all file types.")

    # --- 2. Map Each Identifier to a Batch ---
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    lookup_df = pd.DataFrame({batch_by_id: ids,
                              "batch": np.arange(len(ids), dtype=np.int64) // batch_size})

    # --- 3. Write to Disk and Return the Mapping DataFrame ---
    write_df(df=lookup_df, write_fp=lookup_table_fp, logger=log)
    log.debug(f"Assigned {len(lookup_df)} identifiers to {lookup_df['batch'].nunique()} batches")

    return lookup_df



def batch_dataset(data_dir: str, 
                  write_dir: str, 
                  batch_by_id: str,
                  batch_size: int = 5_000,
                  master_file_type: Optional[str] = None, 
                  lookup_table_fp: Optional[str] = None,
                  keep_file_types: Optional[Collection[str]] = None,
                  overwrite: bool = False,
                  max_workers: Optional[int] = None,
                  logger: Optional[logging.Logger] = None,
                  recursive: bool = False) -> list[int]:
    """Write each selected file type into batches of Parquet files.

    Builds or loads an identifier lookup, then processes file types in separate
    worker processes. Files without the identifier column are written as one
    unbatched Parquet file per file type.

    Arguments
    ---------
    data_dir: str
        Directory containing source files.
    write_dir: str
        Directory for output batches and the default lookup path.
    batch_by_id: str
        Column used to assign rows to batches.
    batch_size: int (default 5000)
        Maximum number of distinct identifiers per batch.
    master_file_type: Optional[str] (default None)
        File type containing all identifiers for the lookup.
    lookup_table_fp: Optional[str] (default None)
        Optional existing or destination lookup path.
    keep_file_types: Optional[Collection[str]] (default None)
        File types to process; ``None`` discovers all types.
    overwrite: bool (default False)
        Rebuild an existing lookup when true.
    max_workers: Optional[int] (default None)
        Maximum number of worker processes, or ``None`` for the
        executor default.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages in the parent process.
    recursive: bool (default False)
        Search nested source directories when true.

    Returns
    -------
        Batch numbers present in the identifier lookup.
    """

    log = _get_logger(logger)
    # -- Save Intermediate Data and Progress at the Cache Directory --
    cache_dir = os.path.join(write_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)


    # --- 1. Determine Batch Lookup ---
    if not isinstance(lookup_table_fp, str):
        lookup_table_fp = os.path.join(write_dir, "lookup_tables", f"{batch_by_id}_batch_lookup.parquet")

    # NOTE: One row per batch_by_id. Two columns; batch_by_id, 'batch'
    batch_lookup: pd.DataFrame = create_batch_lookup_table(data_dir=data_dir, 
                                                           lookup_table_fp=lookup_table_fp,
                                                           batch_by_id=batch_by_id, 
                                                           batch_size=batch_size, 
                                                           master_file_type=master_file_type,
                                                           overwrite=overwrite,
                                                           max_workers=max_workers,
                                                           logger=log,
                                                           recursive=recursive)

    # --- 2. Find All File Types ---
    # NOTE: If not specified, we process all files.
    if keep_file_types is None:
        keep_file_types: list[str] = get_file_types(data_dir=data_dir,
                                                    ext=None,
                                                    require_batch_number=False,
                                                    logger=log,
                                                    recursive=recursive)

    # --- 3. Process Each File Type ---
    id_to_batch: dict[str, int] = dict(zip(batch_lookup[batch_by_id], 
                                           batch_lookup["batch"].astype("int32")))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future, str] = {
            executor.submit(process_file_type,
                            file_type=file_type,
                            data_dir=data_dir,
                            write_dir=write_dir,
                            batch_by_id=batch_by_id,
                            id_to_batch=id_to_batch,
                            overwrite=overwrite,
                            recursive=recursive): file_type
            for file_type in keep_file_types
        }

        for future in as_completed(futures):
            future.result()
            log.debug(f"Completed file type {futures[future]}")

    # --- 4. Return the Batches of the Dataset ---
    return list(batch_lookup["batch"].unique())
