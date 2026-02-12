"""HTTP utilities for fetching evaluation data from remote sources."""

import csv
import io
from typing import Any, Dict, List, Optional

import requests


DEFAULT_TIMEOUT = 60  # seconds

class FetchError(Exception):
    """Raised when fetching data from a remote source fails."""
    pass

def fetch_json(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Fetch JSON data from a URL.

    Args:
        url: The URL to fetch from
        timeout: Request timeout in seconds
        headers: Optional HTTP headers

    Returns:
        Parsed JSON data (dict or list)

    Raises:
        FetchError: If the request fails or returns non-200 status
    """
    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise FetchError(f"Failed to fetch {url}: {e}") from e
    except ValueError as e:
        raise FetchError(f"Failed to parse JSON from {url}: {e}") from e


def fetch_csv(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """
    Fetch CSV data from a URL and parse it into a list of dicts.

    Args:
        url: The URL to fetch from
        timeout: Request timeout in seconds
        headers: Optional HTTP headers

    Returns:
        List of dicts, one per CSV row, keyed by column headers

    Raises:
        FetchError: If the request fails or returns non-200 status
    """
    try:
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        reader = csv.DictReader(io.StringIO(response.text))
        return list(reader)
    except requests.exceptions.RequestException as e:
        raise FetchError(f"Failed to fetch {url}: {e}") from e
    except csv.Error as e:
        raise FetchError(f"Failed to parse CSV from {url}: {e}") from e
