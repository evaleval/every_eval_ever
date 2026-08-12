"""Raw payloads are kept permanently, cheaply, and with a queryable ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.cron import archive
from every_eval_ever.helpers import raw_capture

REPO = 'evaleval/EEE_raw'


class _Info:
    def __init__(self, private: bool):
        self.private = private


class _FakeApi:
    """Stands in for the Hub, recording what a run would store."""

    def __init__(self, existing: set[str] | None = None, private: bool = True):
        self.existing = existing or set()
        self.commits: list[dict] = []
        self.created: list[dict] = []
        self.private = private

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def repo_info(self, *, repo_id, repo_type):
        return _Info(private=self.private)

    def get_paths_info(self, *, repo_id, paths, repo_type):
        return [_PathInfo(path) for path in paths if path in self.existing]

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        for operation in kwargs['operations']:
            self.existing.add(operation.path_in_repo)
        return object()


class _PathInfo:
    def __init__(self, path: str):
        self.path = path


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
    with pytest.raises(archive.ArchiveError, match='nothing to archive'):
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


def test_publish_state_round_trips(monkeypatch, tmp_path: Path):
    written = {}

    class _StateApi(_FakeApi):
        def create_commit(self, **kwargs):
            super().create_commit(**kwargs)
            operation = kwargs['operations'][0]
            written[operation.path_in_repo] = bytes(operation.path_or_fileobj)
            return object()

    api = _StateApi()
    archive.write_state(
        'hle',
        {'gating_fingerprint': 'abc', 'pr_number': 7, 'partial': False},
        repo_id=REPO,
        api=api,
    )

    stored = tmp_path / 'state.json'
    stored.write_bytes(written[archive.state_path('hle')])
    monkeypatch.setattr(
        archive, 'hf_hub_download', lambda **kwargs: str(stored)
    )

    state = archive.read_state('hle', repo_id=REPO)
    assert state == {
        'gating_fingerprint': 'abc',
        'pr_number': 7,
        'partial': False,
    }


def test_missing_state_reads_as_none(monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    def missing(**kwargs):
        raise EntryNotFoundError('no state yet')

    monkeypatch.setattr(archive, 'hf_hub_download', missing)

    assert archive.read_state('hle', repo_id=REPO) is None


def test_a_transient_state_read_error_raises(monkeypatch):
    # Guessing "first run" on a 504 would republish an entire unchanged record
    # set; a failed run just retries tomorrow.
    def broken(**kwargs):
        raise RuntimeError('504 gateway timeout')

    monkeypatch.setattr(archive, 'hf_hub_download', broken)

    with pytest.raises(archive.ArchiveError, match='504'):
        archive.read_state('hle', repo_id=REPO)


def test_a_malformed_state_file_raises(monkeypatch, tmp_path: Path):
    stored = tmp_path / 'state.json'
    stored.write_text('not json', encoding='utf-8')
    monkeypatch.setattr(
        archive, 'hf_hub_download', lambda **kwargs: str(stored)
    )

    with pytest.raises(archive.ArchiveError, match='unreadable'):
        archive.read_state('hle', repo_id=REPO)


def test_a_public_raw_dataset_is_refused(monkeypatch, tmp_path: Path):
    # Raw source payloads are stored on the promise of privacy; visibility is
    # re-checked before every commit, not only at preflight.
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')
    api = _FakeApi(private=False)

    with pytest.raises(archive.ArchiveError, match='PUBLIC'):
        _archive(tmp_path, api)

    assert api.commits == []


def test_one_url_serving_two_bodies_archives_both_correctly(
    monkeypatch, tmp_path: Path
):
    # Regression: a URL-keyed capture filename let the second body overwrite
    # the first while both manifest rows survived, so the archive stored one
    # body under the other's hash.
    import hashlib

    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 1}')
    raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 2}')
    api = _FakeApi()

    result = _archive(tmp_path, api)

    assert result.uploaded == 2
    blobs = {
        operation.path_in_repo: operation
        for operation in api.commits[0]['operations']
        if operation.path_in_repo.startswith('blobs/')
    }
    assert len(blobs) == 2
    for path_in_repo, operation in blobs.items():
        body = Path(operation.path_or_fileobj).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        # The blob really contains the bytes its address promises.
        assert Path(path_in_repo).stem == digest


def test_the_attempt_record_round_trips(monkeypatch, tmp_path: Path):
    written = {}

    class _StateApi(_FakeApi):
        def create_commit(self, **kwargs):
            super().create_commit(**kwargs)
            operation = kwargs['operations'][0]
            written[operation.path_in_repo] = bytes(operation.path_or_fileobj)
            return object()

    archive.write_attempt(
        'hle',
        {'run_id': '9-1', 'paths': ['data/hle/dev/m/a.json']},
        repo_id=REPO,
        api=_StateApi(),
    )
    stored = tmp_path / 'attempt.json'
    stored.write_bytes(written[archive.attempt_path('hle')])
    monkeypatch.setattr(
        archive, 'hf_hub_download', lambda **kwargs: str(stored)
    )

    attempt = archive.read_attempt('hle', repo_id=REPO)
    assert attempt == {'run_id': '9-1', 'paths': ['data/hle/dev/m/a.json']}


def test_a_missing_attempt_reads_as_none(monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    def missing(**kwargs):
        raise EntryNotFoundError('none')

    monkeypatch.setattr(archive, 'hf_hub_download', missing)
    assert archive.read_attempt('hle', repo_id=REPO) is None


def test_writing_state_clears_the_attempt_in_the_same_commit(tmp_path: Path):
    from huggingface_hub import CommitOperationDelete

    api = _FakeApi()
    archive.write_state(
        'hle',
        {'gating_fingerprint': 'abc'},
        repo_id=REPO,
        api=api,
        clear_attempt=True,
    )

    operations = api.commits[0]['operations']
    assert any(
        isinstance(op, CommitOperationDelete)
        and op.path_in_repo == archive.attempt_path('hle')
        for op in operations
    ), 'the state and the attempt must change atomically'


def test_failure_reports_are_archived_under_the_run(
    monkeypatch, tmp_path: Path
):
    _capture(monkeypatch, tmp_path, 'https://x.invalid/a.json', b'{}')
    report = tmp_path / 'hle_failures.json'
    report.write_text('{"failed_records": []}', encoding='utf-8')
    api = _FakeApi()

    _archive(tmp_path, api, reports=[report])

    assert 'reports/hle/2026-08-11-1234-1/hle_failures.json' in _operations(api)


def test_reports_alone_are_enough_to_archive(monkeypatch, tmp_path: Path):
    # A NOT_CAPTURED adapter has no payloads, but its failure report still
    # embeds raw source rows and needs the private dataset.
    report = tmp_path / 'report.json'
    report.write_text('{}', encoding='utf-8')
    api = _FakeApi()

    result = _archive(tmp_path, api, reports=[report])

    assert result.uploaded == 0
    paths = list(_operations(api))
    assert paths == ['reports/hle/2026-08-11-1234-1/report.json']


def test_ledger_rows_carry_capture_errors(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))

    def broken(directory, url, body, content_type):
        raise OSError('disk full')

    monkeypatch.setattr(raw_capture, '_capture', broken)
    raw_capture.capture_response('https://x.invalid/a.json', b'{}')

    rows = archive.ledger_rows(
        tmp_path, adapter='hle', run_date='2026-08-11', run_id='1'
    )

    assert len(rows) == 1
    assert 'disk full' in rows[0]['error']
    assert rows[0]['blob_path'] is None
