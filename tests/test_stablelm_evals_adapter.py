"""Offline tests for the StableLM evals wrapper.

The wrapper's own job is small -- pin the source, repair the model identity, pin one
collection -- so that is what these cover. Conversion itself belongs to the lm_eval
converter and is covered by its own `ConverterCase`.
"""

from __future__ import annotations

import json

import pytest

from every_eval_ever.adapters.stablelm_evals import adapter

SHA = 'b' * 40


def _payload(pretrained: str, tasks: dict | None = None) -> dict:
    return {
        'results': tasks or {
            'sciq': {
                'acc': 0.891, 'acc_stderr': 0.009859828407037191,
                'acc_norm': 0.816, 'acc_norm_stderr': 0.012259457340938588,
            },
        },
        'versions': {'sciq': 0},
        'config': {
            'model': 'gpt2',
            'model_args': f'use_fast=True,pretrained={pretrained},dtype=auto',
            'num_fewshot': 0, 'batch_size': '8', 'bootstrap_iters': 100000,
            'limit': None,
        },
    }


# --------------------------------------------------------------------------- #
# model identity, which is the whole reason this wrapper exists
# --------------------------------------------------------------------------- #
def test_a_full_repo_id_passes_through():
    payload = _payload('bigscience/bloom-3b')
    assert adapter.normalize_model_args(payload, 'f.json') == 'bigscience/bloom-3b'
    # untouched, so the record keeps what the harness was actually given
    assert 'pretrained=bigscience/bloom-3b' in payload['config']['model_args']


def test_a_registered_orgless_id_gains_its_namespace():
    """One file was run from a local checkout, so `pretrained=` has no org."""
    payload = _payload('stablelm-3b-4e1t')
    assert adapter.normalize_model_args(payload, 'f.json') == 'stabilityai/stablelm-3b-4e1t'
    assert 'pretrained=stabilityai/stablelm-3b-4e1t' in payload['config']['model_args']


def test_an_unregistered_orgless_id_is_refused_not_guessed():
    """A placeholder developer would route unrelated models into one directory."""
    payload = _payload('some-local-checkout')
    with pytest.raises(ValueError, match='no publishing namespace'):
        adapter.normalize_model_args(payload, 'f.json')


@pytest.mark.parametrize('config', [None, 'not-a-dict', {'model_args': 42}, {}])
def test_a_malformed_config_is_a_failure_with_a_reason(config):
    payload = {'results': {}, 'config': config}
    with pytest.raises(ValueError):
        adapter.normalize_model_args(payload, 'f.json')


def test_pretrained_is_read_as_its_own_key():
    """`model_args` is a comma-joined key=value string, and other keys end in the
    same letters -- matching loosely would read the wrong value."""
    payload = _payload('bigscience/bloom-3b')
    payload['config']['model_args'] = (
        'peft_pretrained=someone/else,pretrained=bigscience/bloom-3b,dtype=auto'
    )
    assert adapter.normalize_model_args(payload, 'f.json') == 'bigscience/bloom-3b'


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
def test_staging_names_files_the_converter_will_discover(tmp_path, monkeypatch):
    """The converter finds logs by a `results_*.json` name."""
    calls = {}

    class FakeResp:
        status_code = 200
        content = b'{}'

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, timeout=None):
        calls[url] = calls.get(url, 0) + 1
        name = url.rsplit('/', 1)[-1]
        pretrained = 'stablelm-3b-4e1t' if name.startswith('stablelm') else 'bigscience/bloom-3b'
        return FakeResp(_payload(pretrained))

    monkeypatch.setattr(adapter.requests, 'get', fake_get)
    staging = tmp_path / 'in'
    staging.mkdir()
    staged, failures = adapter.stage_sources(
        SHA, ['evals/external/bigscience-bloom-3b.json', 'evals/stablelm-3b-4e1t.json'],
        staging,
    )
    assert failures == []
    assert set(staged) == {
        'results_bigscience-bloom-3b.json', 'results_stablelm-3b-4e1t.json'
    }
    assert set(staged.values()) == {'bigscience/bloom-3b', 'stabilityai/stablelm-3b-4e1t'}
    for name in staged:
        assert json.loads((staging / name).read_text())['config']['model_args']
    # every source file fetched from the pinned sha, not a branch
    assert all(SHA in url for url in calls)


def test_an_unreadable_source_file_is_a_failure_not_a_crash(tmp_path, monkeypatch):
    def fake_get(url, timeout=None):
        raise RuntimeError('404')

    monkeypatch.setattr(adapter.requests, 'get', fake_get)
    staging = tmp_path / 'in'
    staging.mkdir()
    staged, failures = adapter.stage_sources(SHA, ['evals/x.json'], staging)
    assert staged == {}
    assert len(failures) == 1
    assert 'could not read source file' in failures[0].reason


# --------------------------------------------------------------------------- #
# CLI guards
# --------------------------------------------------------------------------- #
def test_an_unresolvable_revision_stops_the_run_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: None)
    args = adapter.parse_args([
        '--revision', 'no-such-branch',
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    with pytest.raises(SystemExit, match='could not resolve'):
        adapter.run(args)


def test_output_dir_must_be_the_collection_directory(tmp_path, monkeypatch):
    """Left to the converter each task becomes its own bare collection, which is
    what `collection_override` prevents -- so the destination has to be explicit."""
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    args = adapter.parse_args([
        '--output-dir', str(tmp_path / 'data' / 'not-the-collection'),
    ])
    with pytest.raises(SystemExit, match=adapter.COLLECTION):
        adapter.run(args)
    assert not list(tmp_path.glob('data/**/*.json'))


def test_a_sha_revision_needs_no_network(monkeypatch):
    def fail(*a, **k):
        raise AssertionError('resolve_commit should not call out for a sha')

    monkeypatch.setattr(adapter.requests, 'get', fail)
    assert adapter.resolve_commit(SHA) == SHA


def test_emit_source_version_prints_the_commit_and_converts_nothing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    args = adapter.parse_args([
        '--emit-source-version',
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    assert adapter.run(args) == []
    assert capsys.readouterr().out.strip() == SHA
    assert not list(tmp_path.glob('data/**/*.json'))


def test_a_truncated_tree_listing_stops_the_run(monkeypatch):
    """A truncated listing would publish a silent subset of the source."""
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {'truncated': True, 'tree': []}

    monkeypatch.setattr(adapter.requests, 'get', lambda url, timeout=None: FakeResp())
    with pytest.raises(SystemExit, match='truncated'):
        adapter.list_result_files(SHA)


def test_per_task_files_are_skipped_and_reported(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {'truncated': False, 'tree': [
                {'type': 'blob', 'path': 'evals/external/a.json'},
                {'type': 'blob', 'path': f'{adapter.SKIP_PREFIX}b-hellaswag.json'},
                {'type': 'blob', 'path': 'README.md'},
            ]}

    monkeypatch.setattr(adapter.requests, 'get', lambda url, timeout=None: FakeResp())
    keep, skipped = adapter.list_result_files(SHA)
    assert keep == ['evals/external/a.json']
    assert skipped == [f'{adapter.SKIP_PREFIX}b-hellaswag.json']
