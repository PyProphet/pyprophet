"""
The `io` module provides tools and utilities for handling input and output operations
in PyProphet. It supports various file formats, including SQLite (OSW),
Parquet, Split Parquet, and TSV, and provides functionality for reading, writing,
and validating data.

Submodules:
-----------
- `util`: Contains utility functions for file validation, schema inspection, and logging.
- `dispatcher`: Provides dispatcher classes for routing I/O configurations to the appropriate
  reader and writer implementations based on file type and context.
- `_base`: Defines abstract base classes and utility methods for implementing custom readers
  and writers for different data formats.

Dependencies:
-------------
- `pandas`
- `pyarrow`
- `duckdb`
- `sqlite3`
- `loguru`
- `click`

"""
