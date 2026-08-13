"""Raw source snapshots: on when automation asks, silent otherwise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from every_eval_ever.helpers import fetch, raw_capture


@pytest.fixture(autouse=True)
def no_leaked_sink(monkeypatch):
    """Capture is process-wide state; never let it leak between tests."""
    monkeypatch.delenv(raw_capture.CAPTURE_DIR_ENV, raising=False)
    raw_capture.deactivate()
    yield
    raw_capture.deactivate()


class FakeResponse:
    """The parts of ``requests.Response`` the fetch helpers actually use."""

    def __init__(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None = None,
    ) -> None:
        self.url = url
        self.content = content
        self.headers = (
            {} if content_type is None else {'Content-Type': content_type}
        )

    @property
    def text(self) -> str:
        return self.content.decode('utf-8')

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        return None


def entries(sink: raw_capture.RawSink) -> list[dict]:
    return sink.entries()


def test_capture_is_off_by_default(tmp_path: Path) -> None:
    assert raw_capture.active_sink() is None
    assert raw_capture.record(url='https://x', content=b'{}') is None
    assert not list(tmp_path.iterdir())


def test_sink_activates_from_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(raw_capture.CAPTURE_DIR_ENV, str(tmp_path / 'raw'))

    sink = raw_capture.active_sink()

    assert sink is not None
    assert sink.root == tmp_path / 'raw'
    assert raw_capture.active_sink() is sink


def test_environment_sink_follows_a_changed_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(raw_capture.CAPTURE_DIR_ENV, str(tmp_path / 'one'))
    first = raw_capture.active_sink()
    monkeypatch.setenv(raw_capture.CAPTURE_DIR_ENV, str(tmp_path / 'two'))

    second = raw_capture.active_sink()

    assert second is not first
    assert second.root == tmp_path / 'two'


def test_the_same_bytes_under_two_content_types_name_one_file(
    tmp_path,
) -> None:
    """The manifest must never name a file the sink did not write.

    A source served as JSON and again as HTML hashes identically, but the
    second sighting would have taken a .html name while only the .json file
    exists, leaving a manifest line pointing at nothing.
    """
    sink = raw_capture.activate(tmp_path / 'raw')

    sink.record(url='https://x', content=b'{}', content_type='application/json')
    sink.record(url='https://y', content=b'{}', content_type='text/html')

    entries = [
        json.loads(line)
        for line in (tmp_path / 'raw' / raw_capture.MANIFEST_NAME)
        .read_text(encoding='utf-8')
        .splitlines()
    ]
    paths = {entry['path'] for entry in entries}
    assert len(paths) == 1
    stored = paths.pop()
    assert (tmp_path / 'raw' / stored).is_file()
    assert entries[1]['duplicate'] is True


@pytest.mark.parametrize('value', ['not-a-number', '0', '-5', '1.5'])
def test_an_unusable_size_cap_falls_back_instead_of_raising(
    tmp_path, monkeypatch, capsys, value
) -> None:
    """A typo in a workflow variable must not crash the conversion.

    ``active_sink`` is called from the shared fetch helpers on every request,
    so raising here would fail a refresh over a setting that only governs how
    much of the source gets snapshotted.
    """
    monkeypatch.setenv(raw_capture.CAPTURE_DIR_ENV, str(tmp_path / 'raw'))
    monkeypatch.setenv(raw_capture.MAX_PAYLOAD_MB_ENV, value)

    sink = raw_capture.active_sink()

    assert sink is not None
    assert sink.max_payload_bytes == raw_capture.DEFAULT_MAX_PAYLOAD_BYTES
    assert raw_capture.MAX_PAYLOAD_MB_ENV in capsys.readouterr().err
    assert sink.record(url='https://x', content=b'{}') is not None


def test_an_explicit_sink_wins_over_the_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(raw_capture.CAPTURE_DIR_ENV, str(tmp_path / 'env'))
    explicit = raw_capture.activate(tmp_path / 'explicit')

    assert raw_capture.active_sink() is explicit

    raw_capture.deactivate()
    assert raw_capture.active_sink().root == tmp_path / 'env'


def test_record_stores_content_addressed_bytes(tmp_path: Path) -> None:
    sink = raw_capture.activate(tmp_path)

    digest = raw_capture.record(
        url='https://example.com/board.json',
        content=b'{"a": 1}',
        content_type='application/json; charset=utf-8',
    )

    assert digest
    stored = tmp_path / f'{digest}.json'
    assert stored.read_bytes() == b'{"a": 1}'
    assert entries(sink) == [
        {
            'bytes': 8,
            'content_type': 'application/json; charset=utf-8',
            'kind': 'payload',
            'path': f'{digest}.json',
            'sha256': digest,
            'url': 'https://example.com/board.json',
        }
    ]
    assert not sink.degraded


@pytest.mark.parametrize(
    ('content_type', 'suffix'),
    [
        ('application/json', '.json'),
        ('text/csv; charset=utf-8', '.csv'),
        ('text/html', '.html'),
        ('application/octet-stream', '.bin'),
        (None, '.bin'),
    ],
)
def test_extension_follows_the_media_type(content_type, suffix) -> None:
    assert raw_capture.extension_for(content_type) == suffix


def test_identical_bytes_are_stored_once_but_both_urls_are_kept(
    tmp_path: Path,
) -> None:
    sink = raw_capture.activate(tmp_path)

    raw_capture.record(url='https://a/1', content=b'same', label='first')
    raw_capture.record(url='https://b/2', content=b'same', label='second')

    stored = sorted(p.name for p in tmp_path.glob('*.bin'))
    assert len(stored) == 1
    recorded = entries(sink)
    assert [entry['url'] for entry in recorded] == [
        'https://a/1',
        'https://b/2',
    ]
    assert recorded[1]['duplicate'] is True
    assert sink.total_bytes == 4


def test_an_oversized_payload_is_dropped_and_reported(tmp_path: Path) -> None:
    sink = raw_capture.activate(tmp_path, max_payload_bytes=4)

    digest = raw_capture.record(url='https://big', content=b'123456')

    assert digest is None
    assert not list(tmp_path.glob('*.bin'))
    dropped = entries(sink)[0]
    assert dropped['kind'] == 'dropped'
    assert 'exceeds' in dropped['reason']
    assert sink.degraded


def test_a_run_stops_capturing_once_it_hits_the_total_cap(
    tmp_path: Path,
) -> None:
    sink = raw_capture.activate(tmp_path, max_total_bytes=8)

    assert raw_capture.record(url='https://a', content=b'12345')
    assert raw_capture.record(url='https://b', content=b'67890') is None

    kinds = [entry['kind'] for entry in entries(sink)]
    assert kinds == ['payload', 'dropped']
    assert sink.degraded


def test_bytes_already_stored_are_kept_even_at_the_total_cap(
    tmp_path: Path,
) -> None:
    """A second sighting of stored bytes costs no storage, so no cap applies.

    Dropping it would fail the run over a payload the run already has.
    """
    sink = raw_capture.activate(tmp_path, max_total_bytes=5)

    assert raw_capture.record(url='https://a', content=b'12345')
    assert raw_capture.record(url='https://b', content=b'12345')

    recorded = entries(sink)
    assert [entry['kind'] for entry in recorded] == ['payload', 'payload']
    assert recorded[1]['duplicate'] is True
    assert recorded[1]['path'] == recorded[0]['path']
    assert sink.total_bytes == 5
    assert not sink.degraded


def test_unseen_bytes_at_the_total_cap_are_still_dropped(
    tmp_path: Path,
) -> None:
    sink = raw_capture.activate(tmp_path, max_total_bytes=5)

    assert raw_capture.record(url='https://a', content=b'12345')
    assert raw_capture.record(url='https://b', content=b'different') is None

    assert [entry['kind'] for entry in entries(sink)] == ['payload', 'dropped']
    assert sink.degraded


def test_pointer_records_a_reference_without_bytes(tmp_path: Path) -> None:
    sink = raw_capture.activate(tmp_path)

    raw_capture.record_pointer(
        kind='hf_dataset',
        reference='vectara/results',
        revision='7c104699e98ade53dd719f79ae9f7eb281c8107d',
        url='https://huggingface.co/datasets/vectara/results',
    )

    assert entries(sink) == [
        {
            'kind': 'pointer',
            'pointer_kind': 'hf_dataset',
            'reference': 'vectara/results',
            'revision': '7c104699e98ade53dd719f79ae9f7eb281c8107d',
            'url': 'https://huggingface.co/datasets/vectara/results',
        }
    ]
    assert not list(tmp_path.glob('*.bin'))


def test_pointer_helpers_do_nothing_when_capture_is_off(monkeypatch) -> None:
    """No sink means no network call to resolve a revision, either."""

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError('resolved a revision with capture off')

    monkeypatch.setattr('huggingface_hub.HfApi.dataset_info', explode)

    raw_capture.record_hf_dataset('vectara/results')
    raw_capture.record_pointer(kind='git', reference='https://example.com')


def test_an_hf_pointer_whose_commit_will_not_resolve_is_dropped(
    tmp_path, monkeypatch
) -> None:
    """Recording the requested revision instead would name a moving target."""
    sink = raw_capture.activate(tmp_path)

    def explode(*args, **kwargs):
        raise RuntimeError('offline')

    monkeypatch.setattr('huggingface_hub.HfApi.dataset_info', explode)

    raw_capture.record_hf_dataset('some/dataset', revision='main')

    entry = entries(sink)[0]
    assert entry['kind'] == 'dropped'
    assert entry['reference'] == 'some/dataset'
    assert 'revision' not in entry
    assert 'offline' in entry['note']
    assert 'requested main' in entry['note']
    assert sink.degraded


def test_an_hf_pointer_the_hub_gives_no_commit_for_is_dropped(
    tmp_path, monkeypatch
) -> None:
    sink = raw_capture.activate(tmp_path)

    class _NoSha:
        sha = None

    monkeypatch.setattr(
        'huggingface_hub.HfApi.dataset_info',
        lambda *args, **kwargs: _NoSha(),
    )

    raw_capture.record_hf_dataset('some/dataset')

    entry = entries(sink)[0]
    assert entry['kind'] == 'dropped'
    assert 'no commit' in entry['note']
    assert sink.degraded


def test_a_resolved_hf_commit_is_recorded_as_a_pointer(
    tmp_path, monkeypatch
) -> None:
    sink = raw_capture.activate(tmp_path)

    class _Info:
        sha = 'a' * 40

    monkeypatch.setattr(
        'huggingface_hub.HfApi.dataset_info',
        lambda *args, **kwargs: _Info(),
    )

    raw_capture.record_hf_dataset('some/dataset', revision='main')

    entry = entries(sink)[0]
    assert entry['kind'] == 'pointer'
    assert entry['revision'] == 'a' * 40
    assert not sink.degraded


def test_git_pointer_records_the_checked_out_commit(tmp_path) -> None:
    import subprocess

    checkout = tmp_path / 'repo'
    checkout.mkdir()
    run = lambda *args: subprocess.run(  # noqa: E731
        ['git', '-C', str(checkout), *args], check=True, capture_output=True
    )
    run('init', '-q')
    run('config', 'user.email', 'test@example.com')
    run('config', 'user.name', 'test')
    (checkout / 'file.txt').write_text('hi', encoding='utf-8')
    run('add', 'file.txt')
    run('commit', '-qm', 'first')

    sink = raw_capture.activate(tmp_path / 'raw')
    raw_capture.record_git_checkout(
        'https://example.com/repo', checkout, ref='submission'
    )

    entry = entries(sink)[0]
    assert entry['pointer_kind'] == 'git'
    assert len(entry['revision']) == 40
    assert entry['note'] == 'ref=submission'


def test_a_git_pointer_outside_a_repository_is_dropped(tmp_path) -> None:
    sink = raw_capture.activate(tmp_path / 'raw')

    raw_capture.record_git_checkout(
        'https://example.com/repo', tmp_path / 'missing'
    )

    entry = entries(sink)[0]
    assert entry['kind'] == 'dropped'
    assert entry.get('revision') is None
    assert 'commit not resolved' in entry['note']
    assert sink.degraded


def test_an_unwritable_sink_degrades_instead_of_raising(tmp_path) -> None:
    # A file where the directory should be makes every write fail.
    blocked = tmp_path / 'raw'
    blocked.write_text('not a directory', encoding='utf-8')
    sink = raw_capture.activate(blocked)

    assert raw_capture.record(url='https://a', content=b'payload') is None
    assert sink.degraded


def test_fetch_json_captures_the_response_body(tmp_path, monkeypatch) -> None:
    sink = raw_capture.activate(tmp_path)
    monkeypatch.setattr(
        requests,
        'get',
        lambda *args, **kwargs: FakeResponse(
            url='https://example.com/api',
            content=b'{"rows": [1, 2]}',
            content_type='application/json',
        ),
    )

    assert fetch.fetch_json('https://example.com/api') == {'rows': [1, 2]}

    entry = entries(sink)[0]
    assert entry['url'] == 'https://example.com/api'
    assert (tmp_path / entry['path']).read_bytes() == b'{"rows": [1, 2]}'


def test_fetch_csv_captures_the_response_body(tmp_path, monkeypatch) -> None:
    sink = raw_capture.activate(tmp_path)
    monkeypatch.setattr(
        requests,
        'get',
        lambda *args, **kwargs: FakeResponse(
            url='https://example.com/board.csv',
            content=b'model,score\ngpt,1\n',
            content_type='text/csv',
        ),
    )

    assert fetch.fetch_csv('https://example.com/board.csv') == [
        {'model': 'gpt', 'score': '1'}
    ]

    entry = entries(sink)[0]
    assert entry['path'].endswith('.csv')


def test_fetch_helpers_write_nothing_when_capture_is_off(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        requests,
        'get',
        lambda *args, **kwargs: FakeResponse(
            url='https://example.com/api', content=b'{}'
        ),
    )

    assert fetch.fetch_json('https://example.com/api') == {}

    assert not list(tmp_path.iterdir())


def test_entries_is_empty_before_anything_is_recorded(tmp_path) -> None:
    assert raw_capture.RawSink(tmp_path).entries() == []
