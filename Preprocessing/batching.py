"""
Convert your dataset into smaller batches.
"""

import os
import pandas as pd
import numpy as np

from typing import Optional, Collection
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from FileHandling.io import get_files, load_df, write_df, get_file_types
from Preprocessing.handlers.batch_dataset import process_file_type

def create_batch_lookup_table(data_dir: str, 
                              lookup_table_fp: str,
                              batch_by_id: str,
                              batch_size: int = 5_000,
                              master_file_type: Optional[str] = None,
                              overwrite: bool = False,
                              max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Create and write a table mapping the `batch_by_id` column to the `batch` the identifier belongs to.
    """

    # -- Return Early if Output Already Exists --
    if not overwrite and os.path.exists(lookup_table_fp):
        return load_df(files=lookup_table_fp, usecols=[batch_by_id, "batch"])

    # --- 1. Find All Identifiers ---
    if isinstance(master_file_type, str):
        all_files: list[str] = get_files(data_dir=data_dir, file_type=master_file_type)

        df: pd.DataFrame = load_df(files=all_files, 
                                   usecols=[batch_by_id], 
                                   filter_func=lambda x: x.drop_duplicates(subset=[batch_by_id]))\
                                    .drop_duplicates(subset=[batch_by_id])\
                                    .sort_values(by=[batch_by_id])

    else:
        raise NotImplementedError("Find all patients from all file types.")

    ids: np.ndarray = df[batch_by_id].unique()
    del df

    # --- 2. Map Each Identifier to a Batch ---
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    lookup_df = pd.DataFrame({batch_by_id: ids,
                              "batch": np.arange(len(ids), dtype=np.int64) // batch_size})

    # --- 3. Write to Disk and Return the Mapping DataFrame ---
    write_df(df=lookup_df, write_fp=lookup_table_fp)

    return lookup_df



def batch_dataset(data_dir: str, 
                  write_dir: str, 
                  batch_by_id: str,
                  batch_size: int = 5_000,
                  master_file_type: Optional[str] = None, 
                  lookup_table_fp: Optional[str] = None,
                  keep_file_types: Optional[Collection[str]] = None,
                  overwrite: bool = False,
                  max_workers: Optional[int] = None) -> list[int]:
    """ Break down your dataset into multiple parts, known as `batches`. Tables are written
    in `parquet` format.
    
    Parameters
    ----------
    batch_by_id: str
        The column to split data by. Intended to be the patient identifier for the dataset.
    batch_size: int (default 5_000)
        The number of unique patients to include in each batch.
    master_file_type: Optional[str] (default None)
        The file type intended to own all patients in the dataset. Providing this value will greatly improve
        performance.
    
    Returns
    -------
    batches: list[int]
        The batches created.
    """

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
                                                           max_workers=max_workers)

    # --- 2. Find All File Types ---
    # NOTE: If not specified, we process all files.
    if keep_file_types is None:
        keep_file_types: list[str] = get_file_types(data_dir=data_dir, 
                                                    ext=None,
                                                    require_batch_number=False)

    # --- 3. Process Each File Type ---
    id_to_batch: dict[str, int] = dict(zip(batch_lookup[batch_by_id], 
                                           batch_lookup["batch"].astype("int32")))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future] = [
            executor.submit(process_file_type,
                            file_type=file_type,
                            data_dir=data_dir,
                            write_dir=write_dir,
                            batch_by_id=batch_by_id,
                            id_to_batch=id_to_batch,
                            overwrite=overwrite)
            for file_type in keep_file_types
        ]

        for future in as_completed(futures):
            future.result()

    # --- 4. Return the Batches of the Dataset ---
    return list(batch_lookup["batch"].unique())
