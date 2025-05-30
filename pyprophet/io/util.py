"""
This module provides utility functions for handling SQLite databases, Parquet files,
and directory structures. It includes functions for file validation, schema inspection,
and logging of file structures, which are commonly used in workflows involving
data processing and analysis.

Functions:
    - is_sqlite_file(filename): Checks if a file is a valid SQLite database.
    - check_sqlite_table(con, table): Verifies if a table exists in a SQLite database.
    - write_scores_sql_command(con, score_sql, feature_name, var_replacement):
      Constructs an SQL command to select specific scores from a feature table.
    - check_duckdb_table(con, schema, table): Checks if a table exists in a DuckDB-attached SQLite schema.
    - create_index_if_not_exists(con, index_name, table_name, column_name):
      Creates an index on a table if it does not already exist.
    - is_parquet_file(file_path): Validates if a file is a Parquet file.
    - is_valid_single_split_parquet_dir(path): Checks if a directory contains a valid single-run split Parquet structure.
    - is_valid_multi_split_parquet_dir(path): Checks if a directory contains multiple subdirectories with split Parquet files.
    - get_parquet_column_names(file_path): Retrieves column names from a Parquet file without reading the entire file.
    - print_parquet_tree(root_dir, precursors, transitions, alignment=None, max_runs=10):
      Prints the structure of Parquet files in a tree-like format.

Key Features:
    - SQLite Utilities: Functions for validating SQLite files, checking table existence, and creating indexes.
    - Parquet Utilities: Functions for validating Parquet files, retrieving schema information, and inspecting directory structures.
    - Logging: Provides detailed logging of file structures and validation results using the `loguru` logger.

Dependencies:
    - os
    - collections (defaultdict)
    - click
    - pandas
    - pyarrow.parquet
    - loguru

Usage:
    This module is designed to be used as a helper library for workflows involving
    SQLite and Parquet file processing. It can be imported and its functions called
    directly in scripts or pipelines.
"""

import os
from collections import defaultdict

import click
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from pyarrow.lib import ArrowInvalid, ArrowIOError


def is_sqlite_file(filename):
    """
    Check if the given file is a SQLite database file by examining the file header.
    :param filename: The path to the file to be checked.
    :return: True if the file is a SQLite database file, False otherwise.
    """
    # https://stackoverflow.com/questions/12932607/how-to-check-with-python-and-sqlite3-if-one-sqlite-database-file-exists
    from os.path import getsize, isfile

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
    """
    Check if a table exists in a SQLite database.

    Parameters:
    - con: SQLite connection object
    - table: Name of the table to check for existence

    Returns:
    - table_present: Boolean indicating if the table exists in the database
    """
    table_present = False
    c = con.cursor()
    c.execute(
        f'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="{table}"'
    )
    if c.fetchone()[0] == 1:
        table_present = True
    else:
        table_present = False
    c.fetchall()

    return table_present


def write_scores_sql_command(con, score_sql, feature_name, var_replacement):
    """
    Write SQL command to select specific scores from a given feature table.

    extracts the scores and writes it into an SQL command
    in some cases some post processing has to be performed depending on which
    position the statement should be inserted (e.g. export_compounds.py)

    Parameters:
    - con: Connection object to the database.
    - score_sql: SQL command to select scores.
    - feature_name: Name of the feature table.
    - var_replacement: Replacement string for "VAR" in score names.

    Returns:
    - Updated SQL command with selected scores.
    """
    feature = pd.read_sql_query(f"""PRAGMA table_info({feature_name})""", con)
    score_names_sql = [
        name for name in feature["name"].tolist() if name.startswith("VAR")
    ]
    score_names_lower = [
        name.lower().replace("var_", var_replacement) for name in score_names_sql
    ]
    for i, score_name_sql in enumerate(score_names_sql):
        score_sql = score_sql + str(
            feature_name + "." + score_name_sql + " AS " + score_names_lower[i] + ", "
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
    if os.path.splitext(file_path)[1].lower() not in (".parquet", ".pq"):
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
    """
    Prints the structure of Parquet files in a tree-like format based on the provided root directory, precursor files, transition files, alignment file, and maximum number of runs to display.
    The function groups precursor and transition files by run, sorts the runs, and prints the file structure for each run up to the specified maximum number of runs.
    If there are more runs than the maximum allowed, it indicates the number of collapsed runs.
    If an alignment file is provided, it is also displayed in the structure.
    """

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
