"""HTTP utilities for fetching evaluation data from remote sources."""

import csv
import io
from typing import Any, Dict, List, Optional

import requests

DEFAULT_TIMEOUT = 60  # seconds


class FetchError(Exception):
    """Raised when fetching data from a remote source fails."""

    pass


def fetch_http_revision(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Return a cheap HTTP revision token without downloading the body.

    The endpoint must support HEAD and return ETag or Last-Modified. Missing
    revision metadata is an error because incremental ingestion cannot safely
    infer that a payload is unchanged.
    """
    try:
        response = requests.head(
            url,
            timeout=timeout,
            headers=headers,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetchError(f'Failed to probe revision for {url}: {exc}') from exc

    if 'ETag' in response.headers and response.headers['ETag'].strip():
        return f'etag:{response.headers["ETag"].strip()}'
    if (
        'Last-Modified' in response.headers
        and response.headers['Last-Modified'].strip()
    ):
        return f'last-modified:{response.headers["Last-Modified"].strip()}'
    raise FetchError(
        f'Cannot safely identify revision for {url}: '
        'HEAD response has neither ETag nor Last-Modified'
    )


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
        raise FetchError(f'Failed to fetch {url}: {e}') from e
    except ValueError as e:
        raise FetchError(f'Failed to parse JSON from {url}: {e}') from e


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
        response = requests.get(
            url, timeout=timeout, headers=headers, allow_redirects=True
        )
        response.raise_for_status()
        reader = csv.DictReader(io.StringIO(response.text))
        return list(reader)
    except requests.exceptions.RequestException as e:
        raise FetchError(f'Failed to fetch {url}: {e}') from e
    except csv.Error as e:
        raise FetchError(f'Failed to parse CSV from {url}: {e}') from e
