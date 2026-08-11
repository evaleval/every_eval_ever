"""Deciding whether a refresh actually found anything new.

A daily refresh that republishes identical records is how a datastore
accumulates thousands of near-duplicate files. The cron avoids that at the
level of a whole run rather than per record: if a run's inputs and outputs are
unchanged from the previous run, it publishes nothing.

Two fingerprints support that, in order of preference:

- the **raw** fingerprint, over the archived source payloads
  (:mod:`every_eval_ever.helpers.raw_capture`). Closest to the source: equality
  means the upstream data itself has not moved.
- the **output** fingerprint, over the generated records with per-run values
  removed. The fallback for adapters whose raw payloads are not archived.

Neither is record-level de-duplication: a run either publishes everything it
produced or nothing at all, so no individual record is ever dropped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from every_eval_ever.cron.stamp import (
    ADAPTER_KEY,
    RUN_DATE_KEY,
    RUN_URL_KEY,
    TYPE_OF_ADDITION_KEY,
    UNKNOWN_FIELDS_KEY,
    aggregate_records,
)

#: Record fields that differ on every run without the data differing.
VOLATILE_RECORD_FIELDS = (
    # Regenerated per run.
    'retrieved_timestamp',
    # Convention embeds retrieved_timestamp in the id.
    'evaluation_id',
)

#: Stamp keys the cron itself adds; they must not make a run look changed.
VOLATILE_STAMP_KEYS = (
    TYPE_OF_ADDITION_KEY,
    RUN_DATE_KEY,
    ADAPTER_KEY,
    RUN_URL_KEY,
    UNKNOWN_FIELDS_KEY,
)


def canonical_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` without the values that change on every run."""
    canonical = json.loads(json.dumps(payload))
    for name in VOLATILE_RECORD_FIELDS:
        canonical.pop(name, None)

    details = canonical.get('source_metadata', {}).get('additional_details')
    if isinstance(details, dict):
        for key in VOLATILE_STAMP_KEYS:
            details.pop(key, None)
        if not details:
            canonical['source_metadata'].pop('additional_details', None)

    detailed = canonical.get('detailed_evaluation_results')
    if isinstance(detailed, dict):
        # Contains the freshly generated companion UUID; the checksum beside it
        # is content-derived and stays.
        detailed.pop('file_path', None)

    return canonical


def record_digest(payload: dict[str, Any], location: str) -> str:
    """Digest one record's content together with where it will be stored."""
    canonical = canonical_record(payload)
    body = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f'{location}\n{body}'.encode('utf-8')).hexdigest()


def output_fingerprint(data_root: str | Path) -> str | None:
    """Digest every generated record under ``data_root``, or ``None`` if empty.

    The record's UUID filename is excluded and its datastore directory is
    included, so a re-run that only regenerated UUIDs matches while a record
    that moved collection does not.
    """
    root = Path(data_root)
    digests = []
    for path in aggregate_records(root):
        payload = json.loads(path.read_text(encoding='utf-8'))
        location = path.parent.relative_to(root).as_posix()
        digests.append(record_digest(payload, location))
    if not digests:
        return None
    return hashlib.sha256(
        '\n'.join(sorted(digests)).encode('utf-8')
    ).hexdigest()


def read_fingerprint(path: str | Path) -> str | None:
    """Read a previously stored fingerprint, tolerating a missing file."""
    fingerprint_path = Path(path)
    if not fingerprint_path.is_file():
        return None
    value = fingerprint_path.read_text(encoding='utf-8').strip()
    return value or None


def write_fingerprint(path: str | Path, value: str) -> None:
    """Store ``value`` for the next run to compare against."""
    fingerprint_path = Path(path)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(f'{value}\n', encoding='utf-8')
