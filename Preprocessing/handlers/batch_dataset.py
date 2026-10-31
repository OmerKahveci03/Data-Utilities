import logging
import os
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from typing import Optional

from FileHandling.io import get_files, load_df, _get_logger

def process_file_type(file_type: str,
                      data_dir: str, 
                      write_dir: str,
                      batch_by_id: str,
                      id_to_batch: dict[str, int],
                      overwrite: bool,
                      logger: Optional[logging.Logger] = None) -> None:
    """Write one file type to Parquet files grouped by identifier batch.

    Existing matching outputs are removed before processing. Rows whose
    identifiers are absent from ``id_to_batch`` are dropped. If an input lacks
    ``batch_by_id``, it is written as a separate numbered Parquet file.

    Args:
        file_type: Source filename prefix and output filename prefix.
        data_dir: Directory containing source files.
        write_dir: Directory for output Parquet files.
        batch_by_id: Identifier column used for row assignment.
        id_to_batch: Mapping from identifier values to batch numbers.
        overwrite: Reserved for future use; currently ignored.
        logger: Optional logger for debug messages when called directly.

    Raises:
        FileNotFoundError: If no source files match ``file_type``.
        RuntimeError: If an empty batch is needed but no schema was inferred.
    """

    log = _get_logger(logger)
    # --- Find all Input Files ---
    files: list[str] = get_files(data_dir=data_dir, file_type=file_type, logger=log)
    if not files:
        raise FileNotFoundError(f"No input files found for requested file type '{file_type}' in {data_dir}")

    # --- Delete Incomplete Outputs ---
    # NOTE: This is done to ensure that previous incomplete attempts do not result in
    # corrupted parquet files.
    os.makedirs(write_dir, exist_ok=True)
    incomplete_outputs: list[str] = get_files(data_dir=write_dir, file_type=file_type, logger=log)
    if incomplete_outputs:
        log.debug("Removing %d existing outputs for %s", len(incomplete_outputs), file_type)
        for output_fp in incomplete_outputs:
            os.remove(output_fp)

    expected_batches = set(id_to_batch.values())

    log.debug("Converting %s from %d files", file_type, len(files))

    writers: dict[int, pq.ParquetWriter] = {}
    rows_written: dict[int, int] = {}
    schema: pa.Schema | None = None

    has_pid: bool = True
    try:
        for i, file in enumerate(files):
            df = load_df(file, logger=log)

            if not has_pid or batch_by_id not in df.columns:
                out_fp = os.path.join(write_dir, f"{file_type}_{i}.parquet")
                df.to_parquet(out_fp, index=False)
                log.debug("Wrote %d rows to %s without batch identifiers", len(df), out_fp)

                has_pid = False
                rows_written[i] = len(df)
                continue

            df["_batch"] = df[batch_by_id].map(id_to_batch)
            df = df.dropna(subset=["_batch"])
            df["_batch"] = df["_batch"].astype("int32")
            log.debug("Mapped %d rows in %s", len(df), file)


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

            log.debug("Finished reading %s", file)
        gc.collect()

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
            log.debug("Created empty batch file %s", out_fp)

            rows_written[batch] = 0

    finally:
        for writer in writers.values():
            writer.close()
        pa.default_memory_pool().release_unused()

    log.debug("Finished converting %s", file_type)
