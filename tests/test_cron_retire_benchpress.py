"""Tests for retiring the BenchPress datastore collection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from every_eval_ever.cron import retire_benchpress


class FakeHub:
    def __init__(self, files: set[str]) -> None:
        self.files = files
        self.commits = []
        self.error = None
        self.land_then_error = False
        self.keep_files = False

    def list_repo_tree(self, *, path_in_repo=None, **kwargs):
        if path_in_repo is None:
            paths = {'data'} if any(p.startswith('data/') for p in self.files) else set()
        else:
            paths = {
                '/'.join(path.split('/')[:2])
                for path in self.files
                if path.startswith('data/')
            }
        return [SimpleNamespace(path=path) for path in paths]

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        if self.error and not self.land_then_error:
            raise self.error
        if not self.keep_files:
            self.files = {
                path for path in self.files
                if not path.startswith('data/benchpress/')
            }
        if self.error:
            raise self.error


def test_removes_only_benchpress_and_is_idempotent():
    hub = FakeHub({
        'data/benchpress/model/result.json',
        'data/benchpress/model/result_samples.jsonl',
        'data/benchpress-other/model/result.json',
        'data/other/model/result.json',
    })

    assert retire_benchpress.retire(hub, 'example/datastore') is True
    assert hub.files == {
        'data/benchpress-other/model/result.json',
        'data/other/model/result.json',
    }
    operation, = hub.commits[0]['operations']
    assert operation.path_in_repo == 'data/benchpress'
    assert operation.is_folder is True
    assert hub.commits[0]['repo_id'] == 'example/datastore'
    assert retire_benchpress.retire(hub, 'example/datastore') is False
    assert len(hub.commits) == 1


def test_empty_datastore_is_a_noop():
    hub = FakeHub(set())
    assert retire_benchpress.retire(hub, 'example/datastore') is False
    assert not hub.commits


def test_missing_token_fails_before_connecting(monkeypatch):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.delenv('HUGGING_FACE_HUB_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='token is required'):
        retire_benchpress.main()


def test_unreadable_datastore_fails():
    hub = FakeHub(set())

    def fail(**kwargs):
        raise ConnectionError('cannot list datastore')

    hub.list_repo_tree = fail
    with pytest.raises(ConnectionError):
        retire_benchpress.retire(hub, 'example/datastore')


def test_timeout_after_commit_is_verified():
    hub = FakeHub({'data/benchpress/model/result.json'})
    hub.error = TimeoutError('response lost')
    hub.land_then_error = True
    assert retire_benchpress.retire(hub, 'example/datastore') is True


def test_failure_does_not_report_success():
    hub = FakeHub({'data/benchpress/model/result.json'})
    hub.error = PermissionError('forbidden')
    with pytest.raises(PermissionError):
        retire_benchpress.retire(hub, 'example/datastore')


def test_success_response_without_deletion_fails():
    hub = FakeHub({'data/benchpress/model/result.json'})
    hub.keep_files = True
    with pytest.raises(RuntimeError, match='still exists'):
        retire_benchpress.retire(hub, 'example/datastore')


def test_commit_conflict_is_retried(monkeypatch):
    hub = FakeHub({'data/benchpress/model/result.json'})
    original = hub.create_commit
    calls = []

    def conflict_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError('409 commit operation is in progress')
        return original(**kwargs)

    hub.create_commit = conflict_once
    monkeypatch.setattr(retire_benchpress.store, 'wait_before_retry', lambda _: None)
    assert retire_benchpress.retire(hub, 'example/datastore') is True
    assert len(calls) == 2


def test_scheduled_cleanup_precedes_ingestion():
    path = Path(__file__).resolve().parents[1] / '.github/workflows/adapter_cron.yml'
    jobs = yaml.safe_load(path.read_text(encoding='utf-8'))['jobs']
    cleanup = jobs['retire-benchpress']
    assert "github.event_name == 'schedule'" in cleanup['if']
    assert "github.repository == 'evaleval/every_eval_ever'" in cleanup['if']
    assert 'retire-benchpress' in jobs['run']['needs']
    assert 'retire-benchpress' in jobs['summary']['needs']
    assert any('every_eval_ever.cron.retire_benchpress' in step.get('run', '')
               for step in cleanup['steps'])
