"""Raw source payloads are archived when asked, and never break a fetch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.helpers import fetch, raw_capture


@pytest.fixture(autouse=True)
def _forget_recorded_payloads():
    raw_capture.reset_recorded_state()
    yield
    raw_capture.reset_recorded_state()


class _Response:
    """The parts of a requests response the fetch helpers use."""

    def __init__(self, body: bytes, content_type: str = 'application/json'):
        self.content = body
        self.text = body.decode('utf-8')
        self.headers = {'Content-Type': content_type}

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return json.loads(self.text)


def _serve(monkeypatch, body: bytes, content_type='application/json'):
    def fake_get(url, **kwargs):
        return _Response(body, content_type)

    monkeypatch.setattr(fetch.requests, 'get', fake_get)


def test_nothing_is_captured_when_the_directory_is_not_configured(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv(raw_capture.RAW_CAPTURE_DIR_ENV, raising=False)
    assert raw_capture.capture_dir() is None
    assert raw_capture.capture_response('https://x.invalid/a', b'{}') is None
    assert not list(tmp_path.iterdir())


def test_capture_writes_the_body_verbatim_and_records_it(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    body = b'{"models": [1, 2, 3]}'

    path = raw_capture.capture_response(
        'https://x.invalid/leaderboard.json',
        body,
        content_type='application/json',
    )

    assert path is not None
    assert path.read_bytes() == body
    entry = raw_capture.read_manifest(tmp_path)[0]
    assert entry['url'] == 'https://x.invalid/leaderboard.json'
    assert entry['bytes'] == len(body)
    assert entry['file'] == path.name
    assert entry['sha256']


def test_capture_filename_is_stable_for_a_url(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    url = 'https://x.invalid/api/v1/leaderboard.json'

    first = raw_capture.capture_response(url, b'a')
    raw_capture.reset_recorded_state()
    second = raw_capture.capture_response(url, b'bb')

    assert first == second
    assert second.read_bytes() == b'bb'


def test_two_urls_sharing_a_last_segment_do_not_collide(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    first = raw_capture.capture_response('https://a.invalid/data.json', b'1')
    second = raw_capture.capture_response('https://b.invalid/data.json', b'2')
    assert first != second


def test_oversized_payloads_are_reported_rather_than_written(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_MAX_BYTES_ENV, '4')

    path = raw_capture.capture_response('https://x.invalid/big', b'123456')

    assert path is None
    entry = raw_capture.read_manifest(tmp_path)[0]
    assert entry['file'] is None
    assert 'ceiling' in entry['skipped']
    assert entry['bytes'] == 6


def test_a_capture_failure_does_not_break_the_fetch(monkeypatch, tmp_path):
    # A file where the capture directory should be: writing there must fail.
    blocked = tmp_path / 'blocked'
    blocked.write_text('not a directory', encoding='utf-8')
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(blocked))
    _serve(monkeypatch, b'{"ok": true}')

    assert fetch.fetch_json('https://x.invalid/a.json') == {'ok': True}


def test_fetch_json_archives_the_response(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    _serve(monkeypatch, b'{"rows": []}')

    fetch.fetch_json('https://x.invalid/rows.json')

    entries = raw_capture.read_manifest(tmp_path)
    assert [entry['url'] for entry in entries] == [
        'https://x.invalid/rows.json'
    ]


def test_fetch_json_archives_a_body_it_cannot_parse(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    _serve(monkeypatch, b'not json at all')

    with pytest.raises(fetch.FetchError):
        fetch.fetch_json('https://x.invalid/broken.json')

    # The evidence for debugging the failure is exactly this payload.
    entries = raw_capture.read_manifest(tmp_path)
    assert (tmp_path / entries[0]['file']).read_bytes() == b'not json at all'


def test_fetch_csv_archives_the_response(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    _serve(monkeypatch, b'model,score\ngpt,0.5\n', content_type='text/csv')

    rows = fetch.fetch_csv('https://x.invalid/board.csv')

    assert rows == [{'model': 'gpt', 'score': '0.5'}]
    assert raw_capture.read_manifest(tmp_path)[0]['bytes'] == 20


def test_fingerprint_ignores_when_the_payload_was_retrieved(
    monkeypatch, tmp_path: Path
):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    for directory in (first, second):
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(directory))
        raw_capture.reset_recorded_state()
        raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 1}')

    assert raw_capture.fingerprint(first) == raw_capture.fingerprint(second)


def test_fingerprint_changes_when_the_payload_changes(
    monkeypatch, tmp_path: Path
):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    bodies = [b'{"a": 1}', b'{"a": 2}']
    for directory, body in zip((first, second), bodies):
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(directory))
        raw_capture.reset_recorded_state()
        raw_capture.capture_response('https://x.invalid/a.json', body)

    assert raw_capture.fingerprint(first) != raw_capture.fingerprint(second)


def test_fingerprint_is_none_when_nothing_was_captured(tmp_path: Path):
    assert raw_capture.fingerprint(tmp_path) is None


def test_payloads_written_by_an_adapter_flag_are_indexed(tmp_path: Path):
    # What --save-raw-json leaves behind: a file, no manifest entry.
    (tmp_path / 'vals-ai.json').write_bytes(b'{"rows": 2}')

    added = raw_capture.index_unlisted_payloads(tmp_path)

    assert added == ['vals-ai.json']
    assert raw_capture.read_manifest(tmp_path)[0]['source'] == 'adapter_flag'
    assert raw_capture.fingerprint(tmp_path) is not None


def test_indexing_is_idempotent(tmp_path: Path):
    (tmp_path / 'vals-ai.json').write_bytes(b'{"rows": 2}')
    raw_capture.index_unlisted_payloads(tmp_path)
    fingerprint = raw_capture.fingerprint(tmp_path)

    assert raw_capture.index_unlisted_payloads(tmp_path) == []
    assert raw_capture.fingerprint(tmp_path) == fingerprint


def test_indexing_skips_the_manifest_itself(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    raw_capture.capture_response('https://x.invalid/a.json', b'{}')

    raw_capture.reset_recorded_state()
    assert raw_capture.index_unlisted_payloads(tmp_path) == []


def test_an_unreadable_ceiling_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_MAX_BYTES_ENV, 'lots')
    assert raw_capture.max_capture_bytes() == raw_capture.DEFAULT_MAX_BYTES


def test_the_same_filename_in_two_directories_is_recorded_in_both(
    monkeypatch, tmp_path: Path
):
    # One process can capture into several directories in turn; a payload name
    # already seen elsewhere must not be skipped for the current directory.
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    for directory in (first, second):
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(directory))
        raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 1}')

    assert len(raw_capture.read_manifest(first)) == 1
    assert len(raw_capture.read_manifest(second)) == 1
    assert raw_capture.fingerprint(second) is not None


def test_an_adapter_written_dump_is_archived_but_not_fingerprinted(
    tmp_path: Path,
):
    # An adapter's own --save-raw-* dump may wrap the payload or stamp it with a
    # fetch time, so it must not decide whether the source moved.
    (tmp_path / 'hle.json').write_bytes(b'{"fetched_at": "1", "rows": []}')
    raw_capture.index_unlisted_payloads(tmp_path)

    assert raw_capture.fingerprint(tmp_path) is not None
    assert raw_capture.fingerprint(tmp_path, verbatim_only=True) is None


def test_verbatim_captures_still_fingerprint_alongside_adapter_dumps(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(tmp_path))
    raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 1}')
    (tmp_path / 'hle.json').write_bytes(b'{"fetched_at": "1"}')
    raw_capture.index_unlisted_payloads(tmp_path)

    verbatim = raw_capture.fingerprint(tmp_path, verbatim_only=True)

    assert verbatim is not None
    assert verbatim != raw_capture.fingerprint(tmp_path)


def test_a_changed_adapter_dump_does_not_change_the_verbatim_fingerprint(
    monkeypatch, tmp_path: Path
):
    fingerprints = []
    for stamp in (b'1', b'2'):
        directory = tmp_path / stamp.decode()
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(directory))
        raw_capture.capture_response('https://x.invalid/a.json', b'{"a": 1}')
        (directory / 'hle.json').write_bytes(
            b'{"fetched_at": "' + stamp + b'"}'
        )
        raw_capture.index_unlisted_payloads(directory)
        fingerprints.append(
            raw_capture.fingerprint(directory, verbatim_only=True)
        )

    assert fingerprints[0] == fingerprints[1]
