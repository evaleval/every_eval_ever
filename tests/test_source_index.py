from __future__ import annotations

from types import SimpleNamespace

import pytest

from every_eval_ever.helpers.fetch import FetchError, fetch_http_revision
from every_eval_ever.source_index import (
    DownloadAction,
    SourceCandidate,
    SourceIndex,
    SourceIndexError,
    execute_download_plan,
    source_manifest_entry,
)


def _manifest(candidate: SourceCandidate, files: list[str]) -> dict:
    key, entry = source_manifest_entry(candidate, files)
    return {
        'files': {path: {'fingerprint': 'test'} for path in files},
        'sources': {key: entry},
    }


def test_source_index_skips_only_exact_revision_matches():
    accepted = SourceCandidate(
        adapter='alpaca_eval',
        source_id='leaderboard:v2',
        revision='etag:abc',
    )
    index = SourceIndex.from_manifest(
        _manifest(accepted, ['data/alpaca/dev/model/result.json'])
    )

    unchanged = index.decide(accepted)
    assert unchanged.action is DownloadAction.SKIP_UNCHANGED
    assert unchanged.existing_files == ('data/alpaca/dev/model/result.json',)

    changed = index.decide(
        SourceCandidate(
            adapter='alpaca_eval',
            source_id='leaderboard:v2',
            revision='etag:def',
        )
    )
    assert changed.action is DownloadAction.DOWNLOAD_CHANGED
    assert changed.previous_revision == 'etag:abc'

    new = index.decide(
        SourceCandidate(
            adapter='alpaca_eval',
            source_id='leaderboard:v1',
            revision='etag:xyz',
        )
    )
    assert new.action is DownloadAction.DOWNLOAD_NEW


def test_source_index_requires_source_metadata_in_manifest():
    with pytest.raises(
        SourceIndexError, match="missing required field 'sources'"
    ):
        SourceIndex.from_manifest({'files': {}})


def test_source_index_rejects_tampered_key():
    candidate = SourceCandidate('adapter', 'source', 'revision')
    manifest = _manifest(candidate, ['data/bench/dev/model/file.json'])
    entry = manifest['sources'].pop(candidate.key)
    manifest['sources']['0' * 64] = entry
    with pytest.raises(SourceIndexError, match='does not match'):
        SourceIndex.from_manifest(manifest)


def test_source_index_rejects_unknown_output_path():
    candidate = SourceCandidate('adapter', 'source', 'revision')
    key, entry = source_manifest_entry(
        candidate, ['data/bench/dev/model/missing.json']
    )
    with pytest.raises(SourceIndexError, match='unknown accepted path'):
        SourceIndex.from_manifest({'files': {}, 'sources': {key: entry}})


def test_source_index_rejects_duplicate_discovery():
    candidate = SourceCandidate('adapter', 'source', 'revision')
    index = SourceIndex.from_manifest({'files': {}, 'sources': {}})
    with pytest.raises(SourceIndexError, match='duplicate source identity'):
        index.plan([candidate, candidate])


def test_source_manifest_entry_requires_output_files():
    candidate = SourceCandidate('adapter', 'source', 'revision')
    with pytest.raises(SourceIndexError, match='at least one file'):
        source_manifest_entry(candidate, [])


def test_execute_download_plan_does_not_download_unchanged_sources():
    accepted = SourceCandidate('adapter', 'stable-id', 'revision-1')
    old_changed = SourceCandidate('adapter', 'changed-id', 'revision-1')
    manifest = _manifest(accepted, ['data/bench/dev/model/existing.json'])
    changed_key, changed_entry = source_manifest_entry(
        old_changed, ['data/bench/dev/model/old-changed.json']
    )
    manifest['sources'][changed_key] = changed_entry
    manifest['files']['data/bench/dev/model/old-changed.json'] = {
        'fingerprint': 'test'
    }
    index = SourceIndex.from_manifest(manifest)
    changed = SourceCandidate('adapter', 'changed-id', 'revision-2')
    new = SourceCandidate('adapter', 'new-id', 'revision-1')
    calls = []

    def download(candidate):
        calls.append(candidate)
        return [f'data/bench/dev/model/{candidate.source_id}.json']

    executions = execute_download_plan(
        index.plan([accepted, changed, new]), download
    )
    assert calls == [changed, new]
    assert executions[0].files == ('data/bench/dev/model/existing.json',)


def test_fetch_http_revision_prefers_etag(monkeypatch):
    response = SimpleNamespace(
        headers={'ETag': '"abc"', 'Last-Modified': 'yesterday'},
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr(
        'every_eval_ever.helpers.fetch.requests.head',
        lambda *args, **kwargs: response,
    )
    assert fetch_http_revision('https://example.test/data.json') == 'etag:"abc"'


def test_fetch_http_revision_rejects_unversioned_response(monkeypatch):
    response = SimpleNamespace(headers={}, raise_for_status=lambda: None)
    monkeypatch.setattr(
        'every_eval_ever.helpers.fetch.requests.head',
        lambda *args, **kwargs: response,
    )
    with pytest.raises(FetchError, match='neither ETag nor Last-Modified'):
        fetch_http_revision('https://example.test/data.json')
