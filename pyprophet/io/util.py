import os
import sys
from pathlib import Path
from importlib.metadata import version
from datetime import datetime
from collections import defaultdict
import platform
import psutil
import click
from loguru import logger
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid, ArrowIOError


def is_sqlite_file(filename):
    # https://stackoverflow.com/questions/12932607/how-to-check-with-python-and-sqlite3-if-one-sqlite-database-file-exists
    from os.path import isfile, getsize

    if not isfile(filename):
        return False
    if getsize(filename) < 100:  # SQLite database file header is 100 bytes
        return False

    with open(filename, "rb") as fd:
        header = fd.read(100)

    if "SQLite format 3" in str(header):
        return True
    else:
        return False


def check_sqlite_table(con, table):
    table_present = False
    c = con.cursor()
    c.execute(
        'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="%s"' % table
    )
    if c.fetchone()[0] == 1:
        table_present = True
    else:
        table_present = False
    c.fetchall()

    return table_present


# extracts the scores and writes it into an SQL command
# in some cases some post processing has to be performed depending on which
# position the statement should be inserted (e.g. export_compounds.py)
def write_scores_sql_command(con, score_sql, feature_name, var_replacement):
    feature = pd.read_sql_query("""PRAGMA table_info(%s)""" % feature_name, con)
    score_names_sql = [
        name for name in feature["name"].tolist() if name.startswith("VAR")
    ]
    score_names_lower = [
        name.lower().replace("var_", var_replacement) for name in score_names_sql
    ]
    for i in range(0, len(score_names_sql)):
        score_sql = score_sql + str(
            feature_name
            + "."
            + score_names_sql[i]
            + " AS "
            + score_names_lower[i]
            + ", "
        )
    return score_sql


def check_duckdb_table(con, schema: str, table: str) -> bool:
    """
    Check if a table exists in a DuckDB-attached SQLite schema (case-insensitive).

    Args:
        con: DuckDB connection.
        schema (str): The schema name (e.g., 'osw').
        table (str): The table name to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    query = f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE LOWER(table_schema) = LOWER('{schema}') 
          AND LOWER(table_name) = LOWER('{table}')
    """
    result = con.execute(query).fetchone()[0]
    return result == 1


def create_index_if_not_exists(con, index_name, table_name, column_name):
    """
    Create an index on a table if it does not already exist. For duckdb connections to sqlite files
    """
    res = con.execute(
        f"""
        SELECT count(*) 
        FROM duckdb_indexes() 
        WHERE index_name = '{index_name}' 
        AND table_name = '{table_name}'
    """
    ).fetchone()

    if res[0] == 0:
        con.execute(f"CREATE INDEX {index_name} ON {table_name} ({column_name})")


def is_parquet_file(file_path):
    """
    Check if the file is a valid Parquet file.
    """

    # First check extension
    if not os.path.splitext(file_path)[1].lower() in (".parquet", ".pq"):
        return False

    # Then verify it's actually a parquet file
    try:
        pq.read_schema(file_path)
        return True
    except (ArrowInvalid, ArrowIOError, OSError):
        return False


def is_valid_single_split_parquet_dir(path):
    """Check if directory contains single-run split parquet structure."""
    required_files = ["precursors_features.parquet", "transition_features.parquet"]
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, f)) and is_parquet_file(os.path.join(path, f))
        for f in required_files
    )


def is_valid_multi_split_parquet_dir(path):
    """Check if directory contains multiple subdirectories with split parquet files."""
    if not os.path.isdir(path):
        return False

    required_files = ["precursors_features.parquet", "transition_features.parquet"]
    subdirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.endswith(".oswpq") and os.path.isdir(os.path.join(path, d))
    ]
    if not subdirs:
        return False

    for subdir in subdirs:
        if not all(
            os.path.isfile(os.path.join(subdir, f))
            and is_parquet_file(os.path.join(subdir, f))
            for f in required_files
        ):
            return False

    return True


def get_parquet_column_names(file_path):
    """
    Retrieves column names from a Parquet file without reading the entire file.
    """
    try:
        table_schema = pq.read_schema(file_path)
        return table_schema.names
    except Exception as e:
        print(f"An error occurred while reading schema from '{file_path}': {e}")
        return None


def print_parquet_tree(root_dir, precursors, transitions, alignment=None, max_runs=10):

    def group_by_run(files):
        grouped = defaultdict(list)
        for f in files:
            parts = f.strip("/").split(os.sep)
            if len(parts) >= 2:
                grouped[parts[-2]].append(parts[-1])
            else:
                grouped["_root"].append(parts[-1])
        return grouped

    precursor_runs = group_by_run(precursors)
    transition_runs = group_by_run(transitions)

    all_runs = sorted(set(precursor_runs.keys()) | set(transition_runs.keys()))
    logger.info(f"Detected {len(all_runs)} split_parquet run files")
    logger.info("Input Parquet Structure:")
    click.echo(f"└── 📁 {root_dir}")

    runs_to_print = all_runs[:max_runs]
    skipped = len(all_runs) - max_runs

    for run in runs_to_print:
        click.echo(f"    ├── 📁 {run}")
        printed = set()
        if run in precursor_runs:
            for f in sorted(precursor_runs[run]):
                click.echo(f"    │   ├── 📄 {f}")
                printed.add(f)
        if run in transition_runs:
            for f in sorted(transition_runs[run]):
                if f not in printed:
                    click.echo(f"    │   └── 📄 {f}")

    if skipped > 0:
        click.echo(f"    │   ... ({skipped} more run(s) collapsed)")

    if alignment:
        click.echo(f"    └── 📄 {os.path.basename(alignment)}")
