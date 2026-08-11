"""Raw payloads are kept permanently, cheaply, and with a queryable ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.cron import archive
from every_eval_ever.helpers import raw_capture

REPO = 'evaleval/EEE_raw'


@pytest.fixture(autouse=True)
def _forget_recorded_payloads():
    raw_capture.reset_recorded_state()
    yield
    raw_capture.reset_recorded_state()


class _FakeApi:
    """Stands in for the Hub, recording what a run would store."""

    def __init__(self, existing: set[str] | None = None):
        self.existing = existing or set()
        self.commits: list[dict] = []
        self.created: list[dict] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def file_exists(self, *, repo_id, filename, repo_type):
        return filename in self.existing

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        for operation in kwargs['operations']:
            self.existing.add(operation.path_in_repo)
        return object()


def _capture(monkeypatch, raw_dir: Path, url: str, body: bytes) -> None:
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir))
    raw_capture.capture_response(url, body, content_type='application/json')


def _archive(raw_dir: Path, api: _FakeApi, **overrides):
    arguments = {
        'adapter': 'hle',
        'run_date': '2026-08-11',
        'run_id': '1234-1',
        'run_url': 'https://github.invalid/run/1234',
        'gating_fingerprint': 'abc123',
        'raw_fingerprint': 'raw123',
        'repo_id': REPO,
        'api': api,
    }
    arguments.update(overrides)
    return archive.archive(raw_dir, **arguments)


def _operations(api: _FakeApi, index: int = 0) -> dict[str, object]:
    return {
        operation.path_in_repo: operation
        for operation in api.commits[index]['operations']
    }


def test_payloads_are_stored_under_their_content_hash(
    monkeypatch, tmp_path: Path
):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{"a": 1}')
    api = _FakeApi()

    result = _archive(tmp_path, api)

    entry = raw_capture.read_manifest(tmp_path)[0]
    expected = archive.blob_path(entry['sha256'], entry['file'])
    assert expected.startswith(f'blobs/{entry["sha256"][:2]}/')
    assert expected.endswith('.json')
    assert expected in _operations(api)
    assert result.uploaded == 1
    assert result.reused == 0


def test_the_raw_dataset_is_created_private(monkeypatch, tmp_path: Path):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')
    api = _FakeApi()

    _archive(tmp_path, api)

    assert api.created == [
        {
            'repo_id': REPO,
            'repo_type': 'dataset',
            'private': True,
            'exist_ok': True,
        }
    ]


def test_a_payload_already_stored_is_not_uploaded_again(
    monkeypatch, tmp_path: Path
):
    # The point of content addressing: an unchanged source costs one ledger row.
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    api = _FakeApi()
    for directory in (first, second):
        _capture(
            monkeypatch, directory, 'https://x.invalid/a.json', b'{"a": 1}'
        )

    _archive(first, api)
    result = _archive(second, api, run_id='5678-1')

    assert result.uploaded == 0
    assert result.reused == 1
    assert result.uploaded_bytes == 0
    # Second commit carries the ledger row and nothing else.
    assert list(_operations(api, 1)) == [
        archive.ledger_path('hle', '2026-08-11', '5678-1')
    ]


def test_a_changed_payload_is_stored_beside_the_old_one(
    monkeypatch, tmp_path: Path
):
    api = _FakeApi()
    for index, body in enumerate((b'{"a": 1}', b'{"a": 2}')):
        directory = tmp_path / str(index)
        _capture(monkeypatch, directory, 'https://x.invalid/a.json', body)
        _archive(directory, api, run_id=f'{index}-1')

    blobs = [path for path in api.existing if path.startswith('blobs/')]
    assert len(blobs) == 2


def test_each_run_writes_its_own_ledger_file(monkeypatch, tmp_path: Path):
    # Parallel adapter jobs must never contend for the same ledger path.
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')
    api = _FakeApi()

    result = _archive(tmp_path, api)

    assert result.ledger_path == 'ledger/hle/2026-08-11-1234-1.jsonl'
    assert result.ledger_path in _operations(api)


def test_the_ledger_row_traces_a_payload_back_to_its_source_and_run(
    monkeypatch, tmp_path: Path
):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/board.json', b'{"a": 1}')
    api = _FakeApi()

    result = _archive(tmp_path, api)

    operation = _operations(api)[result.ledger_path]
    rows = [
        json.loads(line)
        for line in bytes(operation.path_or_fileobj).decode().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row['adapter'] == 'hle'
    assert row['run_date'] == '2026-08-11'
    assert row['run_id'] == '1234-1'
    assert row['run_url'] == 'https://github.invalid/run/1234'
    assert row['source_url'] == 'https://x.invalid/board.json'
    assert row['capture_source'] == raw_capture.VERBATIM_SOURCE
    assert row['gating_fingerprint'] == 'abc123'
    assert row['raw_fingerprint'] == 'raw123'
    assert row['blob_path'] == archive.blob_path(
        row['sha256'], row['file_name']
    )
    assert row['bytes'] == 8


def test_the_ledger_is_newline_delimited_json(monkeypatch, tmp_path: Path):
    # So `load_dataset('json', data_files='ledger/**/*.jsonl')` reads the repo.
    for index in range(2):
        _capture(
            monkeypatch,
            tmp_path,
            f'https://x.invalid/{index}.json',
            f'{{"n": {index}}}'.encode(),
        )
    api = _FakeApi()

    result = _archive(tmp_path, api)

    body = bytes(_operations(api)[result.ledger_path].path_or_fileobj).decode()
    assert body.endswith('\n')
    assert [json.loads(line)['bytes'] for line in body.splitlines()] == [8, 8]


def test_a_payload_too_large_to_store_still_gets_a_ledger_row(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_MAX_BYTES_ENV, '4')
    _capture(monkeypatch, tmp_path, 'https://x.invalid/big.json', b'123456')
    api = _FakeApi()

    result = _archive(tmp_path, api)

    row = result.rows[0]
    assert row['blob_path'] is None
    assert 'ceiling' in row['skipped']
    # Only the ledger is committed: there is no payload to store.
    assert list(_operations(api)) == [result.ledger_path]


def test_archiving_nothing_is_an_error(tmp_path: Path):
    with pytest.raises(archive.ArchiveError, match='no raw payloads'):
        _archive(tmp_path, _FakeApi())


def test_an_unreachable_raw_dataset_is_an_error(monkeypatch, tmp_path: Path):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')

    class _Broken(_FakeApi):
        def create_repo(self, **kwargs):
            raise RuntimeError('403 forbidden')

    with pytest.raises(archive.ArchiveError, match='403 forbidden'):
        _archive(tmp_path, _Broken())


def test_a_failed_commit_is_an_error(monkeypatch, tmp_path: Path):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')

    class _Broken(_FakeApi):
        def create_commit(self, **kwargs):
            raise RuntimeError('504 gateway timeout')

    with pytest.raises(archive.ArchiveError, match='504'):
        _archive(tmp_path, _Broken())


def test_a_missing_local_payload_is_an_error(monkeypatch, tmp_path: Path):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')
    entry = raw_capture.read_manifest(tmp_path)[0]
    (tmp_path / entry['file']).unlink()

    with pytest.raises(archive.ArchiveError, match='named in the manifest'):
        _archive(tmp_path, _FakeApi())


def test_adapter_written_dumps_are_archived_too(tmp_path: Path):
    # Not usable as a fingerprint, but still raw data worth keeping.
    (tmp_path / 'hle.json').write_bytes(b'{"fetched_at": "1", "rows": []}')
    raw_capture.index_unlisted_payloads(tmp_path)
    api = _FakeApi()

    result = _archive(tmp_path, api)

    assert result.uploaded == 1
    assert result.rows[0]['capture_source'] == raw_capture.ADAPTER_FLAG_SOURCE


def test_the_ledger_remembers_the_fingerprint_for_the_next_run(
    monkeypatch, tmp_path: Path
):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{"a": 1}')
    api = _FakeApi()
    _archive(tmp_path, api, gating_fingerprint='today')

    class _ReadBack(_FakeApi):
        def list_repo_files(self, *, repo_id, repo_type):
            return sorted(self.existing)

    read_back = _ReadBack(existing=api.existing)
    ledger = next(
        operation
        for operation in api.commits[0]['operations']
        if operation.path_in_repo.startswith('ledger/')
    )
    stored = tmp_path / 'stored.jsonl'
    stored.write_bytes(bytes(ledger.path_or_fileobj))
    monkeypatch.setattr(
        archive, 'hf_hub_download', lambda **kwargs: str(stored)
    )

    assert (
        archive.last_gating_fingerprint('hle', repo_id=REPO, api=read_back)
        == 'today'
    )


def test_the_most_recent_ledger_wins(monkeypatch, tmp_path: Path):
    class _Api(_FakeApi):
        def list_repo_files(self, *, repo_id, repo_type):
            return [
                'ledger/hle/2026-08-09-1-1.jsonl',
                'ledger/hle/2026-08-11-9-1.jsonl',
                'ledger/hle/2026-08-10-5-1.jsonl',
                'ledger/vals_ai/2026-08-12-1-1.jsonl',
            ]

    requested = []

    def fake_download(**kwargs):
        requested.append(kwargs['filename'])
        path = tmp_path / 'row.jsonl'
        path.write_text(
            json.dumps({'gating_fingerprint': 'latest'}) + '\n',
            encoding='utf-8',
        )
        return str(path)

    monkeypatch.setattr(archive, 'hf_hub_download', fake_download)

    result = archive.last_gating_fingerprint('hle', repo_id=REPO, api=_Api())

    assert result == 'latest'
    assert requested == ['ledger/hle/2026-08-11-9-1.jsonl']


def test_no_ledger_yet_means_no_previous_fingerprint(tmp_path: Path):
    class _Empty(_FakeApi):
        def list_repo_files(self, *, repo_id, repo_type):
            return []

    assert (
        archive.last_gating_fingerprint('hle', repo_id=REPO, api=_Empty())
        is None
    )


def test_an_unreadable_ledger_does_not_stop_a_run(tmp_path: Path):
    # Failing closed here would mean never publishing; failing open can only
    # add a duplicate.
    class _Broken(_FakeApi):
        def list_repo_files(self, *, repo_id, repo_type):
            raise RuntimeError('404 not found')

    assert (
        archive.last_gating_fingerprint('hle', repo_id=REPO, api=_Broken())
        is None
    )
