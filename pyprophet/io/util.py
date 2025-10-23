"""
This module provides utility functions for handling tsv files, SQLite databases, Parquet files,
and directory structures. It includes functions for file validation, schema inspection,
and logging of file structures, which are commonly used in workflows involving
data processing and analysis.

Functions:
    - is_tsv_file(file_path): Checks if a file is likely a TSV file based on its extension and content.
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
    - unimod_to_codename(seq): Converts a sequence with unimod modifications to a codename.

Key Features:
    - SQLite Utilities: Functions for validating SQLite files, checking table existence, and creating indexes.
    - Parquet Utilities: Functions for validating Parquet files, retrieving schema information, and inspecting directory structures.
    - Logging: Provides detailed logging of file structures and validation results using the `loguru` logger.

Dependencies:
    - os
    - collections (defaultdict)
    - click
    - pandas
    - pyopenms
    - pyarrow.parquet
    - loguru

Usage:
    This module is designed to be used as a helper library for workflows involving
    SQLite and Parquet file processing. It can be imported and its functions called
    directly in scripts or pipelines.
"""

import os
from collections import defaultdict
import sqlite3
import importlib
from typing import Type
import duckdb
import click
import pandas as pd
import pyopenms as poms
from loguru import logger


def _ensure_pyarrow():
    """
    Avoid importing pyarrow at module import time; import lazily in functions that need it.
    """
    try:
        import pyarrow as pa  # type: ignore
        from pyarrow.lib import ArrowInvalid, ArrowIOError  # type: ignore

        return pa, ArrowInvalid, ArrowIOError
    except ImportError as exc:
        import click

        raise click.ClickException(
            "Parquet support requires 'pyarrow'. Install with 'pip install pyarrow' or 'pip install pyprophet[parquet]'."
        ) from exc


def is_tsv_file(file_path):
    """
    Checks if a file is likely a TSV file based on extension and content.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is likely a TSV file, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False

    if not file_path.lower().endswith(".tsv") and not file_path.lower().endswith(
        ".txt"
    ):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if "\t" in line:
                    return True  # Found tab character, likely a TSV
        return False  # No tab character found
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


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


def get_table_columns(sqlite_file: str, table: str) -> list:
    """
    Retrieve the column names of a table in a SQLite database file.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        table (str): Name of the table to retrieve column names from.

    Returns:
        list: List of column names in the specified table.
    """
    with sqlite3.connect(sqlite_file) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def get_table_columns_with_types(sqlite_file: str, table: str) -> list:
    """
    Get the columns and their types for a given table in a SQLite database file.

    Args:
        sqlite_file (str): The path to the SQLite database file.
        table (str): The name of the table to retrieve columns and types from.

    Returns:
        list: A list of tuples where each tuple contains the column name and its data type.
    """
    with sqlite3.connect(sqlite_file) as conn:
        return [(row[1], row[2]) for row in conn.execute(f"PRAGMA table_info({table})")]


def check_table_column_exists(sqlite_file: str, table: str, column: str) -> bool:
    """
    Check if a specific column exists in a table of a SQLite database file.

    Args:
        sqlite_file (str): Path to the SQLite database file.
        table (str): Name of the table to check.
        column (str): Name of the column to check for existence.

    Returns:
        bool: True if the column exists, False otherwise.
    """
    with sqlite3.connect(sqlite_file) as conn:
        columns = get_table_columns(sqlite_file, table)
        return column in columns


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


def _lazy_parquet_class(module_path: str, class_name: str) -> Type:
    """
    Import the given module (relative to this package) and return the class.
    Raises a ClickException with a friendly message if the import fails (e.g. missing pyarrow).
    """
    try:
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    except ModuleNotFoundError as exc:
        # Likely pyarrow or the module itself is missing; user should install the parquet extra.
        raise click.ClickException(
            "Parquet support requires the 'pyarrow' package. "
            "Install it with 'pip install pyarrow' or 'pip install pyprophet[parquet]'."
        ) from exc
    except Exception:
        # Propagate other exceptions (syntax errors, attribute errors) to surface the real problem.
        raise


def _area_from_config(config) -> str:
    """
    Map a config instance to its package area name used in the io package.
    """
    # Avoid importing config classes here to prevent circular imports.
    cname = type(config).__name__
    if cname == "RunnerIOConfig":
        return "scoring"
    if cname == "IPFIOConfig":
        return "ipf"
    if cname == "LevelContextIOConfig":
        return "levels_context"
    if cname == "ExportIOConfig":
        return "export"
    raise ValueError(f"Unsupported config context: {type(config).__name__}")


def _get_parquet_reader_class_for_config(config, split: bool = False) -> Type:
    _, _, _ = _ensure_pyarrow()
    area = _area_from_config(config)
    module = f".{area}.split_parquet" if split else f".{area}.parquet"
    return _lazy_parquet_class(
        module, "SplitParquetReader" if split else "ParquetReader"
    )


def _get_parquet_writer_class_for_config(config, split: bool = False) -> Type:
    _, _, _ = _ensure_pyarrow()
    area = _area_from_config(config)
    module = f".{area}.split_parquet" if split else f".{area}.parquet"
    return _lazy_parquet_class(
        module, "SplitParquetWriter" if split else "ParquetWriter"
    )


def is_parquet_file(file_path):
    """
    Check if the file is a valid Parquet file.
    """

    # First check extension
    if os.path.splitext(file_path)[1].lower() not in (".parquet", ".pq"):
        return False

    # Then verify it's actually a parquet file
    try:
        pa, ArrowInvalid, ArrowIOError = _ensure_pyarrow()
        pa.parquet.read_schema(file_path)
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


def load_sqlite_scanner(conn: duckdb.DuckDBPyConnection):
    """
    Ensures the `sqlite_scanner` extension is installed and loaded in DuckDB.
    """
    try:
        conn.execute("LOAD sqlite_scanner")
    except Exception as e:
        if "Extension 'sqlite_scanner' not found" in str(e):
            try:
                conn.execute("INSTALL sqlite_scanner")
                conn.execute("LOAD sqlite_scanner")
            except Exception as install_error:
                if "already installed but the origin is different" in str(
                    install_error
                ):
                    conn.execute("FORCE INSTALL sqlite_scanner")
                    conn.execute("LOAD sqlite_scanner")
                else:
                    raise install_error
        else:
            raise e


def get_parquet_column_names(file_path):
    """
    Retrieves column names from a Parquet file without reading the entire file.
    """
    try:
        pa, _, _ = _ensure_pyarrow()
        table_schema = pa.parquet.read_schema(file_path)
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


def unimod_to_codename(seq):
    """
    Convert a sequence with unimod modifications to a codename.
    """
    seq_poms = poms.AASequence.fromString(seq)
    codename = seq_poms.toString()
    return codename
