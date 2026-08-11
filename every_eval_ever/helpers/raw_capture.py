"""Opt-in on-disk capture of raw source payloads.

Adapters fetch through :mod:`every_eval_ever.helpers.fetch`. When
``EEE_RAW_CAPTURE_DIR`` names a directory, every response body those helpers
receive is also written there verbatim, next to a ``manifest.jsonl`` recording
the URL, byte count, and SHA-256 of each payload. Nothing is captured when the
variable is unset, so interactive adapter runs are unaffected.

Capture is best-effort: a capture failure is logged and the fetch still
returns. The caller decides whether missing raw data matters —
:func:`fingerprint` and :func:`read_manifest` describe what was actually
written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

RAW_CAPTURE_DIR_ENV = 'EEE_RAW_CAPTURE_DIR'
RAW_CAPTURE_MAX_BYTES_ENV = 'EEE_RAW_CAPTURE_MAX_BYTES'
MANIFEST_NAME = 'manifest.jsonl'

#: A response body stored exactly as the server sent it.
VERBATIM_SOURCE = 'fetch_helpers'
#: A file an adapter's own --save-raw-* flag wrote. Archived, but derived: it
#: may wrap the payload or stamp it with a retrieval time, so it cannot be
#: compared across runs to decide whether the source moved.
ADAPTER_FLAG_SOURCE = 'adapter_flag'

DEFAULT_MAX_BYTES = 64 * 1024 * 1024

_UNSAFE_NAME = re.compile(r'[^A-Za-z0-9._-]+')
_EXTENSION_BY_CONTENT_TYPE = {
    'application/json': '.json',
    'text/csv': '.csv',
    'text/html': '.html',
    'text/plain': '.txt',
}

_logger = logging.getLogger(__name__)
_lock = threading.Lock()


def capture_dir() -> Path | None:
    """Return the configured capture directory, or ``None`` when disabled."""
    value = os.environ.get(RAW_CAPTURE_DIR_ENV, '').strip()
    return Path(value) if value else None


def max_capture_bytes() -> int:
    """Return the per-payload capture ceiling in bytes."""
    value = os.environ.get(RAW_CAPTURE_MAX_BYTES_ENV, '').strip()
    if not value:
        return DEFAULT_MAX_BYTES
    try:
        parsed = int(value)
    except ValueError:
        _logger.warning(
            '%s is not an integer (%r); using the default ceiling',
            RAW_CAPTURE_MAX_BYTES_ENV,
            value,
        )
        return DEFAULT_MAX_BYTES
    return parsed if parsed > 0 else DEFAULT_MAX_BYTES


def payload_filename(
    url: str,
    content_type: str | None = None,
    checksum: str | None = None,
) -> str:
    """Return a stable, filesystem-safe capture filename for a payload.

    The name is a function of the URL *and* the body: re-fetching identical
    bytes maps to the same file (idempotent), while the same URL serving
    different bytes gets a second file rather than overwriting the first —
    each manifest row must keep pointing at exactly the bytes it hashed, or
    the content-addressed archive would store one body under another's hash.
    """
    digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]
    if checksum:
        digest = f'{digest}-{checksum[:12]}'
    parsed = urlparse(url)
    stem, suffix = _split_last_segment(parsed.path)
    label = _UNSAFE_NAME.sub('-', stem or parsed.netloc).strip('-')
    extension = suffix or _extension_for(content_type)
    return f'{digest}-{label[:60] or "payload"}{extension}'


def _split_last_segment(path: str) -> tuple[str, str]:
    """Split a URL path's last segment into a stem and a file extension."""
    segment = path.rstrip('/').rsplit('/', 1)[-1]
    stem, dot, suffix = segment.rpartition('.')
    if dot and 1 <= len(suffix) <= 5 and suffix.isalnum():
        return stem, f'.{suffix.lower()}'
    return segment, ''


def _extension_for(content_type: str | None) -> str:
    if not content_type:
        return '.bin'
    base = content_type.split(';', 1)[0].strip().lower()
    return _EXTENSION_BY_CONTENT_TYPE.get(base, '.bin')


def capture_response(
    url: str,
    body: bytes,
    *,
    content_type: str | None = None,
) -> Path | None:
    """Write ``body`` to the capture directory and record it in the manifest.

    Returns the payload path, or ``None`` when capture is disabled, skipped, or
    failed. Never raises: a fetch must not fail because archiving did.
    """
    directory = capture_dir()
    if directory is None:
        return None
    try:
        return _capture(directory, url, body, content_type)
    except Exception as error:  # pragma: no cover - defensive
        _logger.warning('could not capture raw payload for %s: %s', url, error)
        return None


def _capture(
    directory: Path,
    url: str,
    body: bytes,
    content_type: str | None,
) -> Path | None:
    checksum = hashlib.sha256(body).hexdigest()
    filename = payload_filename(url, content_type, checksum)
    ceiling = max_capture_bytes()
    entry: dict[str, Any] = {
        'url': url,
        'file': filename,
        'bytes': len(body),
        'sha256': checksum,
        'content_type': content_type,
        'source': VERBATIM_SOURCE,
        'retrieved_at': datetime.now(timezone.utc).isoformat(
            timespec='seconds'
        ),
    }

    if len(body) > ceiling:
        # Record the gap rather than dropping the payload silently.
        entry['file'] = None
        entry['skipped'] = f'payload exceeds {ceiling} byte capture ceiling'
        _logger.warning(
            'not capturing %s: %d bytes exceeds the %d byte ceiling',
            url,
            len(body),
            ceiling,
        )
        _record(directory, filename, checksum, entry)
        return None

    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / filename
    temporary_path = payload_path.with_name(f'{filename}.partial')
    temporary_path.write_bytes(body)
    temporary_path.replace(payload_path)
    _record(directory, filename, checksum, entry)
    return payload_path


def _record(
    directory: Path,
    filename: str,
    checksum: str,
    entry: dict[str, Any],
) -> None:
    """Append ``entry`` to the manifest unless it is already recorded.

    Deduplication reads the manifest itself rather than any in-process memo, so
    the record always agrees with the directory it describes — a cleared and
    reused capture directory starts from a genuinely empty manifest.
    """
    with _lock:
        already = any(
            item.get('file') == filename and item.get('sha256') == checksum
            for item in read_manifest(directory)
        )
        if already:
            return
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / MANIFEST_NAME).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


def read_manifest(directory: str | Path) -> list[dict[str, Any]]:
    """Return the manifest entries written under ``directory``."""
    manifest_path = Path(directory) / MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    entries = []
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def index_unlisted_payloads(directory: str | Path) -> list[str]:
    """Add manifest entries for payloads written outside the capture hook.

    An adapter's own ``--save-raw-*`` flag writes into the same directory
    without going through :func:`capture_response`. Indexing those files brings
    them under :func:`fingerprint`, so raw archived by either route counts.

    Returns the filenames newly recorded.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    listed = {
        entry['file'] for entry in read_manifest(root) if entry.get('file')
    }
    added = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name in listed or path.name == MANIFEST_NAME:
            continue
        if path.name.endswith('.partial'):
            continue
        with path.open('rb') as handle:
            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
        _record(
            root,
            path.name,
            checksum,
            {
                'url': None,
                'file': path.name,
                'bytes': path.stat().st_size,
                'sha256': checksum,
                'source': ADAPTER_FLAG_SOURCE,
            },
        )
        added.append(path.name)
    return added


def fingerprint(
    directory: str | Path, *, verbatim_only: bool = False
) -> str | None:
    """Return a digest of the captured payloads, or ``None`` if none exist.

    The digest covers each payload's name and content hash and nothing else, so
    two runs that fetched byte-identical data agree even though their manifests
    carry different retrieval timestamps. That equality is what lets a caller
    conclude the upstream source has not moved.

    Pass ``verbatim_only`` when the answer must support that conclusion. Only
    bodies stored exactly as the server sent them qualify: a file written by an
    adapter's own ``--save-raw-*`` flag is a derived artifact and may embed its
    own fetch time, which would make every run look different from the last.
    """
    entries = read_manifest(directory)
    if verbatim_only:
        entries = [
            entry
            for entry in entries
            if entry.get('source', VERBATIM_SOURCE) == VERBATIM_SOURCE
        ]
    lines = sorted(
        f'{entry.get("file") or entry["url"]} {entry["sha256"]}'
        for entry in entries
    )
    if not lines:
        return None
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()
