import os
import shutil
import sqlite3
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple

import click
import duckdb

from .io.util import check_sqlite_table


def process_run(run_info: Tuple):
    """
    Split an OpenSWATH results file into multiple files, one for each run.

    Args:
        run_info: Tuple containing information about the run to process.
            Tuple format: (index, run_file, run_id, infile, outdir)
    """
    index, run_file, run_id, infile, outdir = run_info
    click.echo(f"Info: Splitting run {index + 1} - {run_file}.")

    # Create a new OSW file for each run
    outfile = os.path.join(outdir, f"{run_file}.osw")
    shutil.copy(infile, outfile)

    # Connect to the new OSW file
    conn_out = sqlite3.connect(outfile)
    c_out = conn_out.cursor()

    # Delete entries from RUN table for runs other than the current one
    c_out.execute(f"DELETE FROM RUN WHERE ID != {run_id}")

    # Delete entries from FEATURE table for runs other than the current one
    c_out.execute(f"DELETE FROM FEATURE WHERE RUN_ID != {run_id}")

    # Delete entries from FEATURE_MS1, FEATURE_MS2, and FEATURE_TRANSITION tables
    # for runs other than the current one
    tables_to_clear = ["FEATURE_MS1", "FEATURE_MS2", "FEATURE_TRANSITION"]
    for table in tables_to_clear:
        c_out.execute(
            f"DELETE FROM {table} WHERE FEATURE_ID NOT IN (SELECT ID FROM FEATURE WHERE RUN_ID = {run_id})"
        )

    # Delete entries from SCORE tables for runs other than the current one
    score_tables_to_clear = ["SCORE_MS1", "SCORE_MS2", "SCORE_TRANSITION", "SCORE_IPF"]
    for table in score_tables_to_clear:
        if check_sqlite_table(conn_out, table):
            c_out.execute(
                f"DELETE FROM {table} WHERE FEATURE_ID NOT IN (SELECT ID FROM FEATURE WHERE RUN_ID = {run_id})"
            )

    # Delete entries from context scores 'SCORE_PEPTIDE', 'SCORE_PROTEIN'
    score_tables_to_clear = ["SCORE_PEPTIDE", "SCORE_PROTEIN"]
    for table in score_tables_to_clear:
        if check_sqlite_table(conn_out, table):
            c_out.execute(f"DELETE FROM {table} WHERE RUN_ID != {run_id}")

    # Vacuum the database
    c_out.executescript("VACUUM")

    conn_out.commit()
    conn_out.close()


def split_osw(infile: str, threads: int = cpu_count() - 1):
    """Split an OpenSWATH results file into multiple files, one for each run."""
    outdir = os.path.dirname(infile)

    # Connect to the merged OSW file
    conn = sqlite3.connect(infile)
    cursor = conn.cursor()

    # Get unique run IDs from the RUN table
    cursor.execute("SELECT * FROM RUN")
    columns = [description[0] for description in cursor.description]
    run_rows = cursor.fetchall()
    conn.close()

    if len(run_rows) == 1:
        click.echo(f"Info: Only one run found in {infile}. No splitting necessary.")
        return

    click.echo(f"Info: Splitting {infile} into {len(run_rows)} files.")

    # Find column indices
    id_idx = columns.index("ID")
    filename_idx = columns.index("FILENAME")

    run_info_list = []
    for index, row in enumerate(run_rows):
        run_file = os.path.basename(row[filename_idx]).split(".")[0]
        run_id = row[id_idx]
        run_info_list.append((index, run_file, run_id, infile, outdir))

    if threads == -1:
        threads = cpu_count()

    with Pool(processes=threads) as pool:
        pool.map(process_run, run_info_list)

    click.echo("Info: Splitting complete.")


def split_merged_parquet(input_dir, output_dir):
    """Split merged Parquet files into individual files per run."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)

    precursor_file = input_dir / "precursors_features.parquet"
    transition_file = input_dir / "transition_features.parquet"
    alignment_file = input_dir / "feature_alignment.parquet"

    if not precursor_file.exists() or not transition_file.exists():
        raise click.ClickException(
            "Both 'precursors_features.parquet' and 'transition_features.parquet' are required."
        )

    con = duckdb.connect()

    # Step 1: Load precursor table and determine unique runs
    con.execute(
        f"SELECT RUN_ID, FILENAME FROM read_parquet('{precursor_file}') GROUP BY RUN_ID, FILENAME"
    )
    run_info = con.fetchall()

    click.echo(f"Detected {len(run_info)} runs:")
    for run_id, filename in run_info:
        click.echo(f"  - RUN_ID={run_id}, FILENAME='{filename}'")

    for run_id, filename in run_info:
        run_basename = Path(filename).stem
        run_dir = output_dir / f"{run_basename}.oswpq"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Save precursor subset
        precursor_out = run_dir / "precursors_features.parquet"
        click.echo(f"Writing: {precursor_out}")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{precursor_file}')
                WHERE RUN_ID = {run_id}
            ) TO '{precursor_out}' (FORMAT 'parquet', COMPRESSION 'ZSTD');
        """
        )

        # Step 3: Extract FEATURE_IDs for the run
        con.execute(
            f"""
            SELECT FEATURE_ID FROM read_parquet('{precursor_file}')
            WHERE RUN_ID = {run_id}
        """
        )
        feature_ids = [row[0] for row in con.fetchall()]
        if not feature_ids:
            click.echo(f"Warning: No FEATURE_IDs found for RUN_ID={run_id}")
            continue

        # Step 4: Write corresponding transition rows
        transition_out = run_dir / "transition_features.parquet"
        click.echo(f"Writing: {transition_out}")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{transition_file}')
                WHERE FEATURE_ID IN ({",".join(map(str, feature_ids))})
            ) TO '{transition_out}' (FORMAT 'parquet', COMPRESSION 'ZSTD');
        """
        )

    # Step 5: Copy alignment file (optional)
    if alignment_file.exists():
        output_alignment = output_dir / "feature_alignment.parquet"
        if not output_alignment.exists():
            click.echo(f"Copying alignment file: {output_alignment}")
            con.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet('{alignment_file}')
                ) TO '{output_alignment}' (FORMAT 'parquet', COMPRESSION 'ZSTD');
            """
            )
        else:
            click.echo(f"Alignment file already exists: {output_alignment}")

    con.close()
    click.echo("Info: Done splitting merged Parquet folder.")
