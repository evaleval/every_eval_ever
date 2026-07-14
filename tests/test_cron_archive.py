from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from every_eval_ever.cron_archive import (
    RawArtifact,
    append_ledger_event,
    archive_raw_artifacts,
    github_run_id,
)


class FakeApi:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commits: list[list[str]] = []

    def file_exists(self, *, filename, **kwargs):
        return filename in self.files

    def create_commit(self, *, operations, **kwargs):
        paths = []
        for operation in operations:
            source = operation.path_or_fileobj
            if hasattr(source, 'read'):
                source.seek(0)
                content = source.read()
            else:
                content = Path(source).read_bytes()
            self.files[operation.path_in_repo] = content
            paths.append(operation.path_in_repo)
        self.commits.append(paths)


def test_archive_is_content_addressed_and_appends_event(tmp_path):
    raw = tmp_path / 'payload.json'
    raw.write_bytes(b'{"score": 1}\n')
    api = FakeApi()
    timestamp = datetime(2026, 7, 25, 12, 30, tzinfo=UTC)

    archived = archive_raw_artifacts(
        api,
        repo_id='evaleval/eee-cron-ingestion',
        run_id='github-123-attempt-1',
        artifacts=[
            RawArtifact(
                adapter='example',
                logical_name='payload.json',
                local_path=raw,
                media_type='application/json',
            )
        ],
        run_metadata={
            'adapters': ['example'],
            'artifacts': ['cannot override archive identities'],
        },
        timestamp=timestamp,
    )

    assert len(archived) == 1
    artifact = archived[0]
    assert artifact.archive_path.startswith(
        f'raw/example/{artifact.sha256[:2]}/{artifact.sha256}/'
    )
    assert gzip.decompress(api.files[artifact.archive_path]) == raw.read_bytes()

    event_path = (
        'ledger/events/2026/07/25/github-123-attempt-1/raw_archived.json'
    )
    event = json.loads(api.files[event_path])
    assert event['origin'] == 'cron'
    assert event['phase'] == 'raw_archived'
    assert event['artifacts'][0]['sha256'] == artifact.sha256
    assert event['adapters'] == ['example']


def test_existing_blob_is_reused_but_each_run_gets_an_event(tmp_path):
    raw = tmp_path / 'payload.json'
    raw.write_text('{}')
    api = FakeApi()
    artifact = RawArtifact(
        adapter='example',
        logical_name='payload.json',
        local_path=raw,
        media_type='application/json',
    )

    first = archive_raw_artifacts(
        api,
        repo_id='evaleval/eee-cron-ingestion',
        run_id='run-1',
        artifacts=[artifact],
    )
    second = archive_raw_artifacts(
        api,
        repo_id='evaleval/eee-cron-ingestion',
        run_id='run-2',
        artifacts=[artifact],
    )

    assert first[0].archive_path == second[0].archive_path
    assert len(api.commits[0]) == 2
    assert len(api.commits[1]) == 1
    assert api.commits[1][0].endswith('/run-2/raw_archived.json')


def test_ledger_events_are_append_only():
    api = FakeApi()
    timestamp = datetime(2026, 7, 25, tzinfo=UTC)

    path = append_ledger_event(
        api,
        repo_id='evaleval/eee-cron-ingestion',
        run_id='run-1',
        phase='completed',
        payload={'status': 'no_changes'},
        timestamp=timestamp,
    )

    assert json.loads(api.files[path])['status'] == 'no_changes'
    with pytest.raises(ValueError, match='already exists'):
        append_ledger_event(
            api,
            repo_id='evaleval/eee-cron-ingestion',
            run_id='run-1',
            phase='completed',
            payload={'status': 'uploaded'},
            timestamp=timestamp,
        )


def test_ledger_identity_fields_cannot_be_overridden():
    api = FakeApi()
    timestamp = datetime(2026, 7, 25, tzinfo=UTC)

    path = append_ledger_event(
        api,
        repo_id='evaleval/eee-cron-ingestion',
        run_id='real-run',
        phase='completed',
        payload={
            'run_id': 'spoofed-run',
            'phase': 'spoofed-phase',
            'origin': 'spoofed-origin',
        },
        timestamp=timestamp,
    )

    event = json.loads(api.files[path])
    assert event['run_id'] == 'real-run'
    assert event['phase'] == 'completed'
    assert event['origin'] == 'cron'


def test_github_run_id_uses_attempt_or_local_fallback():
    assert github_run_id('123', '2') == 'github-123-attempt-2'
    assert github_run_id('123', None) == 'github-123'
    assert github_run_id(None, None)
