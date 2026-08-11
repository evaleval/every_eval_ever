"""Credential problems surface before a refresh does any work."""

from __future__ import annotations

from pathlib import Path

from every_eval_ever.cron import preflight

DATASTORE = 'evaleval/EEE_datastore'
RAW = 'evaleval/EEE_raw'


class _Info:
    def __init__(self, private: bool = True):
        self.private = private


class _FakeApi:
    def __init__(
        self,
        *,
        identity: dict | None = None,
        repos: dict[str, _Info] | None = None,
        can_create: bool = True,
    ):
        self._identity = identity
        self.repos = repos if repos is not None else {}
        self.can_create = can_create
        self.created: list[str] = []

    def whoami(self):
        if self._identity is None:
            raise RuntimeError('Token is required')
        return self._identity

    def repo_info(self, *, repo_id, repo_type):
        if repo_id not in self.repos:
            raise RuntimeError(f'404 {repo_id}')
        return self.repos[repo_id]

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        if not self.can_create:
            raise RuntimeError('403 you do not have permission')
        self.created.append(repo_id)
        self.repos[repo_id] = _Info(private=private)


def _named(checks, name):
    return next(check for check in checks if check.name == name)


def _environment(**extra) -> dict[str, str]:
    return {'HF_TOKEN': 'token', **extra}


def test_a_missing_token_fails_before_anything_else():
    checks = preflight.run_preflight(
        environment={}, api=_FakeApi(identity=None)
    )

    token = _named(checks, 'hugging face token')
    assert not token.ok
    assert 'HF_TOKEN' in token.detail
    # No point reporting on destinations we cannot reach.
    assert not any(check.name == 'datastore' for check in checks)
    assert preflight.failed(checks)


def test_a_working_token_reports_who_it_is():
    api = _FakeApi(
        identity={'name': 'eee-bot', 'auth': {'accessToken': {'role': 'write'}}},
        repos={DATASTORE: _Info(), RAW: _Info()},
    )

    checks = preflight.run_preflight(environment=_environment(), api=api)

    token = _named(checks, 'hugging face token')
    assert token.ok
    assert 'eee-bot' in token.detail
    assert 'write' in token.detail


def test_an_unreachable_datastore_is_a_failure():
    api = _FakeApi(identity={'name': 'bot'}, repos={RAW: _Info()})

    checks = preflight.run_preflight(environment=_environment(), api=api)

    datastore = _named(checks, 'datastore')
    assert not datastore.ok
    assert DATASTORE in datastore.detail
    assert datastore in preflight.failed(checks)


def test_a_missing_raw_dataset_is_created_private():
    api = _FakeApi(identity={'name': 'bot'}, repos={DATASTORE: _Info()})

    checks = preflight.run_preflight(environment=_environment(), api=api)

    raw = _named(checks, 'raw dataset')
    assert raw.ok
    assert api.created == [RAW]
    assert api.repos[RAW].private is True
    assert not preflight.failed(checks)


def test_a_token_that_cannot_create_the_raw_dataset_fails_early():
    # Better here than fifteen jobs deep, since the refresh fails closed.
    api = _FakeApi(
        identity={'name': 'bot'},
        repos={DATASTORE: _Info()},
        can_create=False,
    )

    checks = preflight.run_preflight(environment=_environment(), api=api)

    raw = _named(checks, 'raw dataset')
    assert not raw.ok
    assert 'Create it by hand' in raw.detail
    assert raw in preflight.failed(checks)


def test_a_public_raw_dataset_warns_but_does_not_block():
    api = _FakeApi(
        identity={'name': 'bot'},
        repos={DATASTORE: _Info(), RAW: _Info(private=False)},
    )

    checks = preflight.run_preflight(environment=_environment(), api=api)

    raw = _named(checks, 'raw dataset')
    assert raw.ok
    assert raw.required is False
    assert 'public' in raw.detail
    # Not flipped automatically: changing a repo's visibility is not ours to do.
    assert api.created == []
    assert not preflight.failed(checks)


def test_a_missing_raw_dataset_can_be_reported_without_creating_it():
    api = _FakeApi(identity={'name': 'bot'}, repos={DATASTORE: _Info()})

    checks = preflight.run_preflight(
        environment=_environment(), api=api, create_raw=False
    )

    assert not _named(checks, 'raw dataset').ok
    assert api.created == []


def test_adapters_held_back_by_a_credential_are_named():
    api = _FakeApi(
        identity={'name': 'bot'}, repos={DATASTORE: _Info(), RAW: _Info()}
    )

    checks = preflight.run_preflight(environment=_environment(), api=api)

    warnings = [
        check for check in checks if check.name.startswith('credential for')
    ]
    assert {check.name for check in warnings} == {
        'credential for artificial_analysis',
        'credential for llm_stats',
    }
    # A missing API key holds back one adapter; it does not stop the refresh.
    assert not any(check.required for check in warnings)
    assert not preflight.failed(checks)


def test_configured_credentials_leave_no_warning():
    api = _FakeApi(
        identity={'name': 'bot'}, repos={DATASTORE: _Info(), RAW: _Info()}
    )
    environment = _environment(
        ARTIFICIAL_ANALYSIS_API_KEY='a', LLM_STATS_API_KEY='b'
    )

    checks = preflight.run_preflight(environment=environment, api=api)

    assert not [
        check for check in checks if check.name.startswith('credential for')
    ]
    scheduled = _named(checks, 'scheduled adapters')
    assert 'artificial_analysis' in scheduled.detail
    assert 'llm_stats' in scheduled.detail
    assert scheduled.detail.startswith('17 will run')


def test_the_checklist_renders_for_a_step_summary(tmp_path: Path):
    api = _FakeApi(
        identity={'name': 'bot'}, repos={DATASTORE: _Info(), RAW: _Info()}
    )
    checks = preflight.run_preflight(environment=_environment(), api=api)
    destination = tmp_path / 'summary.md'

    preflight.write_markdown(checks, destination)

    body = destination.read_text(encoding='utf-8')
    assert body.startswith('## Preflight')
    assert '✅ **datastore**' in body
    assert '⚠️ **credential for llm_stats**' in body


def test_the_checklist_appends_rather_than_replaces(tmp_path: Path):
    destination = tmp_path / 'summary.md'
    destination.write_text('## Planned refresh\n', encoding='utf-8')
    api = _FakeApi(
        identity={'name': 'bot'}, repos={DATASTORE: _Info(), RAW: _Info()}
    )

    preflight.write_markdown(
        preflight.run_preflight(environment=_environment(), api=api),
        destination,
    )

    body = destination.read_text(encoding='utf-8')
    assert body.startswith('## Planned refresh')
    assert '## Preflight' in body
