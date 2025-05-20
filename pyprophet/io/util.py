import os
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


def is_valid_split_parquet_dir(path):
    """
    Checks if the directory contains both required parquet files
    and that each is a valid Parquet file.
    """
    if not os.path.isdir(path):
        return False

    required_files = ["precursors_features.parquet", "transition_features.parquet"]

    for filename in required_files:
        full_path = os.path.join(path, filename)
        if not os.path.isfile(full_path):
            return False
        if not is_parquet_file(full_path):
            return False

    return True


def get_parquet_column_names(file_path):
    """
    Retrieves column names from a Parquet file without reading the entire file.

    Args:
        file_path (str): The path to the Parquet file.

    Returns:
        list: A list of column names in the Parquet file.
    """

    try:
        table_schema = pq.read_schema(file_path)
        column_names = table_schema.names
        return column_names
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
