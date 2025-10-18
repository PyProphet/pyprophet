"""
Version checking utilities for PyProphet.

This module provides functionality to check if a newer version of PyProphet
is available on PyPI.
"""

import json
import urllib.request
import urllib.error
from importlib.metadata import version, PackageNotFoundError
from packaging.version import parse as parse_version
from typing import Optional, Tuple


def get_current_version() -> str:
    """
    Get the current installed version of pyprophet.
    
    Returns:
        str: The current version string (e.g., "3.0.2")
    """
    try:
        return version("pyprophet")
    except PackageNotFoundError:
        # Fallback to unknown version if package metadata is not available
        return "Unknown"


def get_latest_pypi_version(timeout: float = 2.0) -> Optional[str]:
    """
    Fetch the latest version of pyprophet from PyPI.
    
    Args:
        timeout: Timeout in seconds for the HTTP request
        
    Returns:
        str: The latest version string, or None if the request fails
    """
    try:
        url = "https://pypi.org/pypi/pyprophet/json"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'pyprophet-version-checker'}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data['info']['version']
            
    except (urllib.error.URLError, urllib.error.HTTPError, 
            json.JSONDecodeError, KeyError, TimeoutError):
        # Silently fail if we can't reach PyPI or parse the response
        return None


def check_for_updates() -> Optional[Tuple[str, str]]:
    """
    Check if a newer version of pyprophet is available on PyPI.
    
    Returns:
        Tuple[str, str]: A tuple of (current_version, latest_version) if an update
                        is available, or None if no update is available or if
                        the check fails.
    """
    current = get_current_version()
    latest = get_latest_pypi_version()
    
    if latest is None:
        return None
    
    try:
        # Use packaging.version for proper version comparison
        if parse_version(latest) > parse_version(current):
            return (current, latest)
    except Exception:
        # If version parsing fails, don't show any message
        return None
    
    return None


def format_update_message(current: str, latest: str) -> str:
    """
    Format the update notification message.
    
    Args:
        current: The current installed version
        latest: The latest available version
        
    Returns:
        str: A formatted message string
    """
    return (
        f"Info: A new version of pyprophet is available on PyPI: "
        f"v{latest} (current version: v{current})"
    )
