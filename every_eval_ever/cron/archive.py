"""Permanent storage for the raw payloads a refresh fetched.

Raw data lives in its own private Hugging Face dataset, separate from the
records it produced. Two things go there:

- **blobs** — each payload stored under its own content hash, so a payload that
  has not changed since the last run costs nothing to keep. This is what makes a
  daily archive affordable.
- **a ledger** — one JSONL row per payload per run, naming the adapter, the run,
  the source URL, and the blob it landed in. The ledger is the queryable index:
  ``load_dataset('json', data_files='ledger/**/*.jsonl')`` over the repo answers
  "what did this source look like on that date".

Each run writes its own ledger file, so parallel adapter jobs never contend for
the same path and nothing is ever rewritten.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from every_eval_ever.helpers import raw_capture

DEFAULT_RAW_REPO_ID = 'evaleval/EEE_raw'

BLOB_PREFIX = 'blobs'
LEDGER_PREFIX = 'ledger'

_logger = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Raised when raw payloads could not be archived."""


@dataclass
class ArchiveResult:
    """What archiving a run's raw payloads did."""

    repo_id: str
    ledger_path: str
    uploaded: int = 0
    reused: int = 0
    uploaded_bytes: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def payloads(self) -> int:
        return self.uploaded + self.reused


def blob_path(checksum: str, filename: str) -> str:
    """Return the content-addressed path for a payload.

    Sharded by the first two hex characters so no single directory grows
    unbounded, and given the payload's extension so the Hub previews it as the
    right type.
    """
    return f'{BLOB_PREFIX}/{checksum[:2]}/{checksum}{Path(filename).suffix}'


def ledger_path(adapter: str, run_date: str, run_id: str) -> str:
    """Return the ledger file for one run of one adapter."""
    return f'{LEDGER_PREFIX}/{adapter}/{run_date}-{run_id}.jsonl'


def ledger_rows(
    raw_dir: str | Path,
    *,
    adapter: str,
    run_date: str,
    run_id: str,
    run_url: str | None = None,
    raw_fingerprint: str | None = None,
    output_fingerprint: str | None = None,
    gating_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Build one ledger row per payload the run captured."""
    rows = []
    for entry in raw_capture.read_manifest(raw_dir):
        filename = entry.get('file')
        rows.append(
            {
                'adapter': adapter,
                'run_date': run_date,
                'run_id': run_id,
                'run_url': run_url,
                'source_url': entry.get('url'),
                'capture_source': entry.get('source'),
                'content_type': entry.get('content_type'),
                'retrieved_at': entry.get('retrieved_at'),
                'file_name': filename,
                'sha256': entry.get('sha256'),
                'bytes': entry.get('bytes'),
                'raw_fingerprint': raw_fingerprint,
                'output_fingerprint': output_fingerprint,
                # The one compared against the next run to decide whether the
                # source moved. Read back by last_gating_fingerprint().
                'gating_fingerprint': gating_fingerprint,
                'blob_path': (
                    blob_path(entry['sha256'], filename) if filename else None
                ),
                # Set when a payload was fetched but deliberately not stored.
                'skipped': entry.get('skipped'),
            }
        )
    return rows


def archive(
    raw_dir: str | Path,
    *,
    adapter: str,
    run_date: str,
    run_id: str,
    run_url: str | None = None,
    raw_fingerprint: str | None = None,
    output_fingerprint: str | None = None,
    gating_fingerprint: str | None = None,
    repo_id: str = DEFAULT_RAW_REPO_ID,
    token: str | None = None,
    api: HfApi | None = None,
    create_if_missing: bool = True,
) -> ArchiveResult:
    """Store a run's raw payloads and its ledger row permanently.

    A payload already present under its content hash is not re-uploaded, so a
    source that has not changed costs one small ledger file per run.

    Raises:
        ArchiveError: if the repository is unreachable or the commit fails.
            Callers treat this as fatal: records should not reach the datastore
            without their raw provenance stored somewhere permanent.
    """
    raw_dir = Path(raw_dir)
    api = api or HfApi(token=token)

    rows = ledger_rows(
        raw_dir,
        adapter=adapter,
        run_date=run_date,
        run_id=run_id,
        run_url=run_url,
        raw_fingerprint=raw_fingerprint,
        output_fingerprint=output_fingerprint,
        gating_fingerprint=gating_fingerprint,
    )
    if not rows:
        raise ArchiveError(f'no raw payloads to archive under {raw_dir}')

    if create_if_missing:
        try:
            # Private: raw source data is kept for provenance, not republished.
            api.create_repo(
                repo_id=repo_id,
                repo_type='dataset',
                private=True,
                exist_ok=True,
            )
        except Exception as error:
            raise ArchiveError(
                f'could not reach or create the raw dataset {repo_id}: {error}'
            ) from error

    destination = ledger_path(adapter, run_date, run_id)
    result = ArchiveResult(repo_id=repo_id, ledger_path=destination, rows=rows)
    operations: list[CommitOperationAdd] = []
    planned: set[str] = set()

    for row in rows:
        path_in_repo = row['blob_path']
        if path_in_repo is None:
            # Fetched but not stored (over the capture ceiling). The ledger row
            # still records that it happened, and says so.
            continue
        if path_in_repo in planned or _already_stored(
            api, repo_id, path_in_repo
        ):
            result.reused += 1
            continue
        payload = raw_dir / row['file_name']
        if not payload.is_file():
            raise ArchiveError(
                f'payload {payload} named in the manifest is gone'
            )
        planned.add(path_in_repo)
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo, path_or_fileobj=str(payload)
            )
        )
        result.uploaded += 1
        result.uploaded_bytes += row['bytes'] or 0

    body = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n'
    operations.append(
        CommitOperationAdd(
            path_in_repo=destination, path_or_fileobj=body.encode('utf-8')
        )
    )

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type='dataset',
            operations=operations,
            commit_message=(
                f'{adapter}: archive {result.uploaded} new payload(s) '
                f'({run_date})'
            ),
            commit_description=run_url or '',
        )
    except Exception as error:
        raise ArchiveError(
            f'could not commit raw payloads to {repo_id}: {error}'
        ) from error

    _logger.info(
        'archived %d new and %d already-stored payload(s) to %s',
        result.uploaded,
        result.reused,
        repo_id,
    )
    return result


def _already_stored(api: HfApi, repo_id: str, path_in_repo: str) -> bool:
    try:
        return api.file_exists(
            repo_id=repo_id, filename=path_in_repo, repo_type='dataset'
        )
    except Exception as error:
        raise ArchiveError(
            f'could not check {path_in_repo} in {repo_id}: {error}'
        ) from error


def last_gating_fingerprint(
    adapter: str,
    *,
    repo_id: str = DEFAULT_RAW_REPO_ID,
    token: str | None = None,
    api: HfApi | None = None,
) -> str | None:
    """Return the fingerprint this adapter's most recent run recorded.

    The ledger is the durable memory of what the source looked like last time.
    Reading it here rather than a build cache means a run cannot forget and
    republish an adapter's whole set because a cache entry was evicted.

    Returns ``None`` when there is nothing to compare against — no ledger, no
    repository, or no fingerprint recorded — which makes the run publish. That
    is the safe direction: it can add a duplicate, never lose a record.
    """
    api = api or HfApi(token=token)
    prefix = f'{LEDGER_PREFIX}/{adapter}/'
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
    except Exception as error:
        _logger.warning(
            'could not read the ledger in %s (%s); treating this run as new',
            repo_id,
            error,
        )
        return None

    # Ledger names start with the run date, so the last one sorted is the most
    # recent.
    ledgers = sorted(
        name
        for name in files
        if name.startswith(prefix) and name.endswith('.jsonl')
    )
    for name in reversed(ledgers):
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type='dataset',
                token=token,
            )
            rows = [
                json.loads(line)
                for line in Path(local).read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]
        except Exception as error:
            _logger.warning('could not read ledger %s: %s', name, error)
            return None
        for row in rows:
            fingerprint = row.get('gating_fingerprint')
            if fingerprint:
                return fingerprint
    return None
