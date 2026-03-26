from importlib.metadata import PackageNotFoundError, version

def _detect_version() -> str:
    """Return the installed package version"""
    try:
        return version("pyprophet")
    except PackageNotFoundError:
        return "0+unknown"

__version__ = _detect_version()
