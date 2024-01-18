import sqlite3
import os
import shutil
import click
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Tuple

from .data_handling import check_sqlite_table

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
    tables_to_clear = ['FEATURE_MS1', 'FEATURE_MS2', 'FEATURE_TRANSITION']
    for table in tables_to_clear:
        c_out.execute(f"DELETE FROM {table} WHERE FEATURE_ID NOT IN (SELECT ID FROM FEATURE WHERE RUN_ID = {run_id})")

    # Delete entries from SCORE tables for runs other than the current one
    score_tables_to_clear = ['SCORE_MS1', 'SCORE_MS2', 'SCORE_TRANSITION', 'SCORE_IPF']
    for table in score_tables_to_clear:
        if check_sqlite_table(conn_out, table):
            c_out.execute(f"DELETE FROM {table} WHERE FEATURE_ID NOT IN (SELECT ID FROM FEATURE WHERE RUN_ID = {run_id})")

    # Delete entries from context scores 'SCORE_PEPTIDE', 'SCORE_PROTEIN'
    score_tables_to_clear = ['SCORE_PEPTIDE', 'SCORE_PROTEIN']
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

    # Get unique run IDs from the RUN table
    run_ids = pd.read_sql("SELECT * FROM RUN", conn)
    conn.close()
    
    if run_ids.shape[0] == 1:
        click.echo(f"Info: Only one run found in {infile}. No splitting necessary.")
        return
    
    click.echo(f"Info: Splitting {infile} into {run_ids.shape[0]} files.")
    
    run_info_list = []
    for index, row in run_ids.iterrows():
        run_file = os.path.basename(row['FILENAME']).split('.')[0]
        run_id = row['ID']
        run_info_list.append((index, run_file, run_id, infile, outdir))

    if threads == -1:
        threads = cpu_count()

    with Pool(processes=threads) as pool:
        pool.map(process_run, run_info_list)

    click.echo("Info: Splitting complete.")