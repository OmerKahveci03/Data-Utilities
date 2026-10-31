# import logging
import os
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from typing import Optional
# from io import StringIO

from FileHandling.io import get_files, load_df

def process_file_type(file_type: str,
                      data_dir: str, 
                      write_dir: str,
                      batch_by_id: str,
                      id_to_batch: dict[str, int],
                      overwrite: bool) -> None:
    """ """

    # logger = logging.getLogger(__name__)
    # final_success_fp = get_final_success_fp(file_type=file_type, write_dir=write_dir)
    # with cached_step(success_fp=final_success_fp, disable=overwrite) as should_run:
    #     if not should_run:
    #         logger.info(f"Skipping completed batching for {file_type}")
    #         return

    # --- Find all Input Files ---
    files: list[str] = get_files(data_dir=data_dir, file_type=file_type)
    if not files:
        raise FileNotFoundError(f"No input files found for requested file type '{file_type}' in {data_dir}")

    # --- Delete Incomplete Outputs ---
    # NOTE: This is done to ensure that previous incomplete attempts do not result in
    # corrupted parquet files.
    os.makedirs(write_dir, exist_ok=True)
    incomplete_outputs: list[str] = get_files(data_dir=write_dir, file_type=file_type)
    if incomplete_outputs:
        # logger.info(f"Deleting {len(incomplete_outputs):,} incomplete outputs for {file_type}")
        for output_fp in incomplete_outputs:
            os.remove(output_fp)

    # buffer = StringIO()
    expected_batches = set(id_to_batch.values())
    # prefix = f" - [{os.getpid()}]"

    # logger.info(f"{prefix} Converting {file_type}...")

    writers: dict[int, pq.ParquetWriter] = {}
    rows_written: dict[int, int] = {}
    schema: pa.Schema | None = None

    has_pid: bool = True
    try:
        for i, file in enumerate(files):
            df = load_df(file)

            if not has_pid or batch_by_id not in df.columns:
                out_fp = os.path.join(write_dir, f"{file_type}_{i}.parquet")
                # buffer.write(f"File type {file_type} does not have batch_by_id column. Writing to {out_fp} instead...")
                df.to_parquet(out_fp, index=False)

                has_pid = False
                rows_written[i] = len(df)
                continue

            df["_batch"] = df[batch_by_id].map(id_to_batch)
            df = df.dropna(subset=["_batch"])
            df["_batch"] = df["_batch"].astype("int32")

            # buffer.write(f"File {os.path.basename(file)} has {len(df):,} rows in patient lookup tables\n")
            # if before != after:
            #     buffer.write(f" - File dropped {before - after:,} rows ({before:,} -> {after:,}) because patients didn't map to a batch.\n")

            # Group by _batch then write each batch to the write_dir with the {file_type}_{batch}.parquet naming convention
            for batch, batched_df in df.groupby("_batch", sort=False):
                batch = int(batch)
                out_fp = os.path.join(write_dir, f"{file_type}_{batch}.parquet")
                batched_df = batched_df.sort_values(batch_by_id)

                batched_df = batched_df.astype("string", copy=False)
                batched_df["_batch"] = batched_df["_batch"].astype("int32")

                table = pa.Table.from_pandas(batched_df, preserve_index=False)
                if schema is None:
                    schema = table.schema

                if batch not in writers:
                    writers[batch] = pq.ParquetWriter(out_fp, table.schema, compression="snappy")
                    rows_written[batch] = 0
                rows_written[batch] += len(batched_df)
                writers[batch].write_table(table)

            if schema is None:
                empty_df = df.head(0).astype("string", copy=False)
                empty_df["_batch"] = empty_df["_batch"].astype("int32")
                table = pa.Table.from_pandas(empty_df, preserve_index=False)
                schema = table.schema
            del df

            # logger.info(f"{prefix} Finished reading {os.path.basename(file)}")
        gc.collect()

        # buffer.write(f"\nFinal row counts for {file_type}:\n")
        # for batch in sorted(rows_written):
        #     buffer.write(f"  batch {batch}: {rows_written[batch]:,} rows\n")

        if not has_pid:
            return

        # Create empty files to ensure a file exists for all batches
        written_batches = set(rows_written.keys())

        missing_batches = expected_batches - written_batches

        for batch in sorted(missing_batches):
            out_fp = os.path.join(write_dir, f"{file_type}_{batch}.parquet")

            if schema is None:
                raise RuntimeError(f"Cannot create empty parquet for {file_type}: no schema inferred")

            empty_table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema,)

            pq.write_table(empty_table, out_fp, compression="snappy")

            rows_written[batch] = 0
            # buffer.write(f"  batch {batch}: 0 rows (empty file created)\n")

    finally:
        for writer in writers.values():
            writer.close()
        pa.default_memory_pool().release_unused()

    # buffer.write(f"Done creating batched {file_type} files")
    # logger.info(buffer.getvalue())
