"""
PyInstaller hook for pyprophet package.
Ensures all pyprophet submodules and data files are collected.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect everything from pyprophet
datas, binaries, hiddenimports = collect_all("pyprophet")

# Ensure all submodules are included
hiddenimports += collect_submodules("pyprophet")
hiddenimports += collect_submodules("pyprophet.cli")
hiddenimports += collect_submodules("pyprophet.scoring")
hiddenimports += collect_submodules("pyprophet.io")
hiddenimports += collect_submodules("pyprophet.export")

# Explicitly include key modules that might be missed
hiddenimports += [
    "pyprophet.main",
    "pyprophet._version",
    "pyprophet._config",
    "pyprophet._base",
]
