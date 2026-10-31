import logging
import os
import gc

import pyarrow as pa
import pyarrow.parquet as pq

from typing import Optional

from ...FileHandling.io import get_files, load_df
from ...Logging.convenience import _get_logger


def process_file_type(file_type: str,
                      data_dir: str, 
                      write_dir: str,
                      batch_by_id: str,
                      id_to_batch: dict[str, int],
                      overwrite: bool,
                      logger: Optional[logging.Logger] = None,
                      recursive: bool = False) -> None:
    """Write one file type to Parquet files grouped by identifier batch.

    Existing matching outputs are removed before processing. Rows whose
    identifiers absent from ``id_to_batch`` raise an error. If an input lacks
    ``batch_by_id``, all its rows are written to one unbatched Parquet file.

    Arguments
    ---------
    file_type: str
        Source filename prefix and output filename prefix.
    data_dir: str
        Directory containing source files.
    write_dir: str
        Directory for output Parquet files.
    batch_by_id: str
        Identifier column used for row assignment.
    id_to_batch: dict[str, int]
        Mapping from identifier values to batch numbers.
    overwrite: bool
        Reserved for future use; currently ignored.
    logger: Optional[logging.Logger] (default None)
        Optional logger for debug messages when called directly.
    recursive: bool (default False)
        Search nested source directories when true.

    Raises
    ------
        FileNotFoundError: If no source files match ``file_type``.
        ValueError: If a source identifier is absent from the lookup.
        RuntimeError: If an empty batch is needed but no schema was inferred.
    """

    log = _get_logger(logger)
    # --- Find all Input Files ---
    files: list[str] = get_files(data_dir=data_dir, file_type=file_type,
                                 logger=log, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No input files found for requested file type '{file_type}' in {data_dir}")

    # --- Delete Incomplete Outputs ---
    # NOTE: This is done to ensure that previous incomplete attempts do not result in
    # corrupted parquet files.
    os.makedirs(write_dir, exist_ok=True)
    incomplete_outputs: list[str] = get_files(data_dir=write_dir, file_type=file_type, logger=log)
    if incomplete_outputs:
        log.debug(f"Removing {len(incomplete_outputs)} existing outputs for {file_type}")
        for output_fp in incomplete_outputs:
            os.remove(output_fp)

    expected_batches = set(id_to_batch.values())

    log.debug(f"Converting {file_type} from {len(files)} files")

    writers: dict[int, pq.ParquetWriter] = {}
    rows_written: dict[int, int] = {}
    schema: pa.Schema | None = None

    unbatched_writer: Optional[pq.ParquetWriter] = None
    try:
        for file in files:
            for df in load_df(files=file, chunk_size=50_000, logger=log):

                if batch_by_id not in df.columns:
                    if writers:
                        raise ValueError(f"Mixed identifier columns for {file_type}")
                    out_fp = os.path.join(write_dir, f"{file_type}.parquet")
                    table = pa.Table.from_pandas(df.astype("string"), preserve_index=False)
                    if unbatched_writer is None:
                        unbatched_writer = pq.ParquetWriter(out_fp, table.schema, compression="snappy")
                    unbatched_writer.write_table(table)
                    log.debug(f"Wrote {len(df)} rows to {out_fp} without batch identifiers")

                    continue

                if unbatched_writer is not None:
                    raise ValueError(f"Mixed identifier columns for {file_type}")

                df["_batch"] = df[batch_by_id].map(id_to_batch)
                if df["_batch"].isna().any():
                    raise ValueError(f"{file} contains identifiers absent from the batch lookup")
                df["_batch"] = df["_batch"].astype("int32")
                log.debug(f"Mapped {len(df)} rows in {file}")


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

                log.debug(f"Finished reading {file}")
        gc.collect()

        if unbatched_writer is not None:
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
            log.debug(f"Created empty batch file {out_fp}")

            rows_written[batch] = 0

    finally:
        if unbatched_writer is not None:
            unbatched_writer.close()
        for writer in writers.values():
            writer.close()
        pa.default_memory_pool().release_unused()

    log.debug(f"Finished converting {file_type}")
