"""Permanent storage for the raw payloads a refresh fetched.

Raw data lives in its own private Hugging Face dataset, separate from the
records it produced. Three things go there:

- **blobs** — each payload stored under its own content hash, so a payload that
  has not changed since the last run costs nothing to keep. This is what makes a
  daily archive affordable.
- **a ledger** — one JSONL row per payload per run, naming the adapter, the run,
  the source URL, and the blob it landed in. The ledger is the queryable index:
  ``load_dataset('json', data_files='ledger/**/*.jsonl')`` over the repo answers
  "what did this source look like on that date".

- **state** — one small JSON file per adapter, ``state/<adapter>.json``,
  overwritten only after a successful publish. It is the durable answer to
  "what did this adapter last publish": the gating fingerprint the next run
  compares against, and the adapter's pull request number. The ledger cannot
  play this role — it is written *before* publishing (so a run that fails to
  publish must not update the gate) and by every run (so a run would read back
  its own fingerprint and conclude nothing changed).

Each run writes its own ledger file, so parallel adapter jobs never contend for
the same path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from every_eval_ever.helpers import raw_capture

DEFAULT_RAW_REPO_ID = 'evaleval/EEE_raw'

BLOB_PREFIX = 'blobs'
LEDGER_PREFIX = 'ledger'
STATE_PREFIX = 'state'

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
    _require_private(api, repo_id)

    destination = ledger_path(adapter, run_date, run_id)
    result = ArchiveResult(repo_id=repo_id, ledger_path=destination, rows=rows)
    operations: list[CommitOperationAdd] = []
    planned: set[str] = set()
    stored = _already_stored(
        api,
        repo_id,
        [row['blob_path'] for row in rows if row['blob_path']],
    )

    for row in rows:
        path_in_repo = row['blob_path']
        if path_in_repo is None:
            # Fetched but not stored (over the capture ceiling). The ledger row
            # still records that it happened, and says so.
            continue
        if path_in_repo in planned or path_in_repo in stored:
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


def _require_private(api: HfApi, repo_id: str) -> None:
    """Refuse to archive into a dataset the world can read.

    Checked immediately before every commit, not only at preflight: visibility
    can change between the two, and raw source payloads are stored here on the
    promise of privacy.
    """
    try:
        info = api.repo_info(repo_id=repo_id, repo_type='dataset')
    except Exception as error:
        raise ArchiveError(
            f'could not verify that {repo_id} is private: {error}'
        ) from error
    if not getattr(info, 'private', False):
        raise ArchiveError(
            f'{repo_id} is PUBLIC; refusing to archive raw payloads into it. '
            'Make it private or point --raw-repo-id at a private dataset.'
        )


def _already_stored(api: HfApi, repo_id: str, paths: list[str]) -> set[str]:
    """Return which of ``paths`` the raw dataset already holds, in one call."""
    if not paths:
        return set()
    try:
        found = api.get_paths_info(
            repo_id=repo_id, paths=paths, repo_type='dataset'
        )
    except Exception as error:
        raise ArchiveError(
            f'could not check existing blobs in {repo_id}: {error}'
        ) from error
    return {entry.path for entry in found}


def state_path(adapter: str) -> str:
    """Return the per-adapter state file path in the raw dataset."""
    return f'{STATE_PREFIX}/{adapter}.json'


def read_state(
    adapter: str,
    *,
    repo_id: str = DEFAULT_RAW_REPO_ID,
    token: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any] | None:
    """Return what this adapter's last successful publish recorded.

    The state file is the durable memory of the last publish — the gating
    fingerprint and the pull request number. Returns ``None`` when there is
    nothing to compare against: no state yet, no repository, or an unreadable
    file. That makes the run publish, the safe direction — it can add a
    duplicate, never lose a record.
    """
    del api  # hf_hub_download manages its own client.
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=state_path(adapter),
            repo_type='dataset',
            token=token,
            force_download=True,
        )
        return json.loads(Path(local).read_text(encoding='utf-8'))
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    except Exception as error:
        _logger.warning(
            'could not read %s from %s (%s); treating this run as new',
            state_path(adapter),
            repo_id,
            error,
        )
        return None


def write_state(
    adapter: str,
    state: dict[str, Any],
    *,
    repo_id: str = DEFAULT_RAW_REPO_ID,
    token: str | None = None,
    api: HfApi | None = None,
) -> None:
    """Record a successful publish, for the next run to compare against.

    Called only after the datastore commit succeeded: a run that failed to
    publish must leave the previous state in place, or its records would be
    skipped as "unchanged" tomorrow and silently never reach the datastore.

    Raises:
        ArchiveError: if the state cannot be written. The publish already
            happened, so the caller reports this loudly rather than unwinding —
            the worst outcome of stale state is a duplicate publish tomorrow.
    """
    api = api or HfApi(token=token)
    body = json.dumps(state, indent=2, sort_keys=True) + '\n'
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type='dataset',
            operations=[
                CommitOperationAdd(
                    path_in_repo=state_path(adapter),
                    path_or_fileobj=body.encode('utf-8'),
                )
            ],
            commit_message=f'{adapter}: record publish state',
        )
    except Exception as error:
        raise ArchiveError(
            f'could not record publish state for {adapter} in {repo_id}: '
            f'{error}'
        ) from error
