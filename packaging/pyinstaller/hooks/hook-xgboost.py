# PyInstaller hook: include xgboost shared libs and package data.
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import os

binaries = collect_dynamic_libs("xgboost")
datas = collect_data_files("xgboost")

# Try to collect all submodules, but don't fail if xgboost imports test-only deps.
hiddenimports = []
try:
    from PyInstaller.utils.hooks import collect_submodules
    hiddenimports = collect_submodules("xgboost")
except Exception:
    # Fallback: include the core runtime modules that are normally needed.
    hiddenimports = [
        "xgboost.core",
        "xgboost.libpath",
        "xgboost.tracker",
        "xgboost.compat",
    ]

# Ensure VERSION file included if present
try:
    import xgboost
    pkg_dir = os.path.dirname(xgboost.__file__)
    version_file = os.path.join(pkg_dir, "VERSION")
    if os.path.exists(version_file):
        datas.append((version_file, "xgboost"))
except Exception:
    pass
