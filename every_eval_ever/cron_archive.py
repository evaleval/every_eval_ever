"""Private raw-input archive and append-only cron lineage ledger."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import CommitOperationAdd

INGESTION_REPO_TYPE = 'dataset'
LEDGER_SCHEMA_VERSION = '1'
_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


@dataclass(frozen=True)
class RawArtifact:
    """One replayable input snapshot used by a cron adapter."""

    adapter: str
    logical_name: str
    local_path: Path
    media_type: str


@dataclass(frozen=True)
class ArchivedArtifact:
    """Content-addressed location and identity of an archived input body."""

    adapter: str
    logical_name: str
    media_type: str
    sha256: str
    size_bytes: int
    archive_path: str


def new_run_id() -> str:
    """Return a collision-resistant local run identifier."""
    timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    return f'{timestamp}-{uuid.uuid4().hex[:12]}'


def github_run_id(
    github_run_id_value: str | None,
    github_run_attempt: str | None,
) -> str:
    """Use GitHub's stable run identity when available."""
    run_id = (github_run_id_value or '').strip()
    attempt = (github_run_attempt or '').strip()
    if run_id:
        suffix = f'-attempt-{attempt}' if attempt else ''
        return f'github-{run_id}{suffix}'
    return new_run_id()


def sha256_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _safe_component(value: str, *, field: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f'{field} must be a safe path component: {value!r}')
    return value


def _event_path(
    *,
    run_id: str,
    phase: str,
    timestamp: datetime,
) -> str:
    safe_run_id = _safe_component(run_id, field='run_id')
    safe_phase = _safe_component(phase, field='phase')
    return f'ledger/events/{timestamp:%Y/%m/%d}/{safe_run_id}/{safe_phase}.json'


def _event_bytes(
    *,
    run_id: str,
    phase: str,
    timestamp: datetime,
    payload: dict[str, Any],
) -> bytes:
    event = {
        **payload,
        'schema_version': LEDGER_SCHEMA_VERSION,
        'run_id': run_id,
        'origin': 'cron',
        'phase': phase,
        'timestamp': timestamp.isoformat(),
    }
    return (
        json.dumps(event, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    ).encode()


def _remote_exists(api: Any, repo_id: str, path: str) -> bool:
    return bool(
        api.file_exists(
            repo_id=repo_id,
            filename=path,
            repo_type=INGESTION_REPO_TYPE,
            revision='main',
        )
    )


def _gzip_file(source: Path, destination: Path) -> None:
    with (
        source.open('rb') as source_handle,
        destination.open('wb') as destination_handle,
        gzip.GzipFile(
            filename='',
            mode='wb',
            fileobj=destination_handle,
            mtime=0,
        ) as gzip_handle,
    ):
        shutil.copyfileobj(source_handle, gzip_handle)


def archive_raw_artifacts(
    api: Any,
    *,
    repo_id: str,
    run_id: str,
    artifacts: Iterable[RawArtifact],
    run_metadata: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> list[ArchivedArtifact]:
    """Archive raw bodies and atomically append the run's archive event."""
    now = timestamp or datetime.now(UTC)
    archived: list[ArchivedArtifact] = []
    operations: list[CommitOperationAdd] = []

    with tempfile.TemporaryDirectory(prefix='eee-cron-archive-') as temp_dir:
        temp_root = Path(temp_dir)
        for index, artifact in enumerate(artifacts):
            adapter = _safe_component(artifact.adapter, field='adapter')
            logical_name = _safe_component(
                artifact.logical_name,
                field='logical_name',
            )
            if not artifact.local_path.is_file():
                raise ValueError(
                    f'{adapter}: raw artifact does not exist: '
                    f'{artifact.local_path}'
                )
            sha256, size_bytes = sha256_file(artifact.local_path)
            archive_path = (
                f'raw/{adapter}/{sha256[:2]}/{sha256}/{logical_name}.gz'
            )
            archived_artifact = ArchivedArtifact(
                adapter=adapter,
                logical_name=logical_name,
                media_type=artifact.media_type,
                sha256=sha256,
                size_bytes=size_bytes,
                archive_path=archive_path,
            )
            archived.append(archived_artifact)

            if not _remote_exists(api, repo_id, archive_path):
                compressed = temp_root / f'{index}-{logical_name}.gz'
                _gzip_file(artifact.local_path, compressed)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=archive_path,
                        path_or_fileobj=str(compressed),
                    )
                )

        event_path = _event_path(
            run_id=run_id,
            phase='raw_archived',
            timestamp=now,
        )
        if _remote_exists(api, repo_id, event_path):
            raise ValueError(f'ledger event already exists: {event_path}')
        payload = {
            **(run_metadata or {}),
            'artifacts': [asdict(item) for item in archived],
        }
        operations.append(
            CommitOperationAdd(
                path_in_repo=event_path,
                path_or_fileobj=io.BytesIO(
                    _event_bytes(
                        run_id=run_id,
                        phase='raw_archived',
                        timestamp=now,
                        payload=payload,
                    )
                ),
            )
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type=INGESTION_REPO_TYPE,
            operations=operations,
            commit_message=f'Archive cron inputs for {run_id}',
        )
    return archived


def append_ledger_event(
    api: Any,
    *,
    repo_id: str,
    run_id: str,
    phase: str,
    payload: dict[str, Any],
    timestamp: datetime | None = None,
) -> str:
    """Append one immutable event describing a later run transition."""
    now = timestamp or datetime.now(UTC)
    event_path = _event_path(
        run_id=run_id,
        phase=phase,
        timestamp=now,
    )
    if _remote_exists(api, repo_id, event_path):
        raise ValueError(f'ledger event already exists: {event_path}')
    api.create_commit(
        repo_id=repo_id,
        repo_type=INGESTION_REPO_TYPE,
        operations=[
            CommitOperationAdd(
                path_in_repo=event_path,
                path_or_fileobj=io.BytesIO(
                    _event_bytes(
                        run_id=run_id,
                        phase=phase,
                        timestamp=now,
                        payload=payload,
                    )
                ),
            )
        ],
        commit_message=f'Record cron {phase} for {run_id}',
    )
    return event_path
