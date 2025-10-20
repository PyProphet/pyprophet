#!/usr/bin/env python3
"""
Launcher for PyInstaller bundles. Forces threading-based parallelism when frozen
to avoid worker process serialization issues with bundled dependencies.
"""
import os
import sys

# When running frozen, configure joblib to use threading backend instead of loky processes
if getattr(sys, "frozen", False):
    # Tell joblib to prefer threading (not a multiprocessing context, but joblib's own backend)
    os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # disable multiprocessing in joblib
    # Fallback: if code checks LOKY env, limit it
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Prefer fork on POSIX when running in dev mode (not frozen)
if not getattr(sys, "frozen", False):
    try:
        import multiprocessing
        multiprocessing.set_start_method("fork", force=False)
    except Exception:
        pass

# Import and run the CLI
from pyprophet.main import cli


def main(argv=None):
    cli()


if __name__ == "__main__":
    main()
