"""
Tests for version checking functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from pyprophet.version_check import (
    get_current_version,
    get_latest_pypi_version,
    check_for_updates,
    format_update_message,
)


def test_get_current_version():
    """Test that we can get the current version."""
    version = get_current_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_format_update_message():
    """Test the update message formatting."""
    message = format_update_message("3.0.0", "3.0.5")
    assert "3.0.0" in message
    assert "3.0.5" in message
    assert "Info:" in message
    assert "pypi" in message.lower()


@patch('pyprophet.version_check.urllib.request.urlopen')
def test_get_latest_pypi_version_success(mock_urlopen):
    """Test successful PyPI version fetch."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"info": {"version": "3.0.5"}}'
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    version = get_latest_pypi_version()
    assert version == "3.0.5"


@patch('pyprophet.version_check.urllib.request.urlopen')
def test_get_latest_pypi_version_network_error(mock_urlopen):
    """Test that network errors are handled gracefully."""
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("Network error")
    
    version = get_latest_pypi_version()
    assert version is None


@patch('pyprophet.version_check.urllib.request.urlopen')
def test_get_latest_pypi_version_timeout(mock_urlopen):
    """Test that timeouts are handled gracefully."""
    mock_urlopen.side_effect = TimeoutError("Timeout")
    
    version = get_latest_pypi_version()
    assert version is None


@patch('pyprophet.version_check.get_current_version')
@patch('pyprophet.version_check.get_latest_pypi_version')
def test_check_for_updates_newer_available(mock_latest, mock_current):
    """Test when a newer version is available."""
    mock_current.return_value = "3.0.0"
    mock_latest.return_value = "3.0.5"
    
    result = check_for_updates()
    assert result is not None
    assert result[0] == "3.0.0"
    assert result[1] == "3.0.5"


@patch('pyprophet.version_check.get_current_version')
@patch('pyprophet.version_check.get_latest_pypi_version')
def test_check_for_updates_current_is_latest(mock_latest, mock_current):
    """Test when current version is the latest."""
    mock_current.return_value = "3.0.5"
    mock_latest.return_value = "3.0.5"
    
    result = check_for_updates()
    assert result is None


@patch('pyprophet.version_check.get_current_version')
@patch('pyprophet.version_check.get_latest_pypi_version')
def test_check_for_updates_current_is_newer(mock_latest, mock_current):
    """Test when current version is newer than PyPI (dev version)."""
    mock_current.return_value = "3.1.0"
    mock_latest.return_value = "3.0.5"
    
    result = check_for_updates()
    assert result is None


@patch('pyprophet.version_check.get_current_version')
@patch('pyprophet.version_check.get_latest_pypi_version')
def test_check_for_updates_network_failure(mock_latest, mock_current):
    """Test when network request fails."""
    mock_current.return_value = "3.0.0"
    mock_latest.return_value = None
    
    result = check_for_updates()
    assert result is None


@patch('pyprophet.version_check.get_current_version')
@patch('pyprophet.version_check.get_latest_pypi_version')
def test_check_for_updates_version_parsing_error(mock_latest, mock_current):
    """Test when version parsing fails."""
    mock_current.return_value = "invalid"
    mock_latest.return_value = "3.0.5"
    
    result = check_for_updates()
    assert result is None
