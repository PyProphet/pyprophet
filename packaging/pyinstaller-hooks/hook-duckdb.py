"""
PyInstaller hook for duckdb package.
Collects package metadata and native libraries.
"""
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# Collect all duckdb submodules
hiddenimports = collect_submodules('duckdb')

# Collect package metadata (needed for importlib.metadata.version())
datas = copy_metadata('duckdb')
datas += copy_metadata('duckdb-extensions', recursive=True)
datas += copy_metadata('duckdb-extension-sqlite-scanner', recursive=True)
