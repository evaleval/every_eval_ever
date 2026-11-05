"""Tests for the cron production datastore safety guardrail."""

from __future__ import annotations

import argparse
import pytest

from every_eval_ever.cron import submit
from every_eval_ever.cron.__main__ import build_parser, cmd_run


@pytest.fixture
def enforce_production_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all bypass flags to test the production guardrail in its strict local state."""
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.delenv('EEE_ALLOW_PRODUCTION', raising=False)


def test_guard_blocks_default_production_repo_locally(enforce_production_guard) -> None:
    """Local calls targeting evaleval/EEE_datastore without bypass must raise."""
    with pytest.raises(submit.ProductionDatastoreGuardError) as exc_info:
        submit.check_production_guard(submit.DEFAULT_DATASTORE_REPO)
    assert 'Direct commits to the production datastore' in str(exc_info.value)
    assert 'restricted to GitHub Actions' in str(exc_info.value)


@pytest.mark.parametrize(
    'url',
    [
        'https://huggingface.co/datasets/evaleval/EEE_datastore',
        'https://huggingface.co/evaleval/EEE_datastore',
        ' evaleval/EEE_datastore ',
    ],
)
def test_guard_normalizes_repo_urls(enforce_production_guard, url: str) -> None:
    """Full URLs pointing to the production datastore must also be blocked."""
    with pytest.raises(submit.ProductionDatastoreGuardError):
        submit.check_production_guard(url)


def test_guard_permits_non_production_repo(enforce_production_guard) -> None:
    """Commits targeting fork or sandbox repositories are always permitted."""
    submit.check_production_guard('some-user/my-test-datastore')
    submit.check_production_guard('evaleval/EEE_staging')


def test_guard_permits_under_github_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execution within GitHub Actions is permitted."""
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.delenv('EEE_ALLOW_PRODUCTION', raising=False)
    submit.check_production_guard(submit.DEFAULT_DATASTORE_REPO)


def test_guard_permits_with_explicit_flag(enforce_production_guard) -> None:
    """Explicitly passing allow_production=True bypasses the guardrail."""
    submit.check_production_guard(submit.DEFAULT_DATASTORE_REPO, allow_production=True)


def test_guard_permits_with_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting EEE_ALLOW_PRODUCTION=true bypasses the guardrail."""
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.setenv('EEE_ALLOW_PRODUCTION', 'true')
    submit.check_production_guard(submit.DEFAULT_DATASTORE_REPO)


def test_submitter_checks_guard_on_ensure_writable(enforce_production_guard) -> None:
    """DatastoreSubmitter.ensure_writable must trigger the production guard."""
    submitter = submit.DatastoreSubmitter(api=None, repo_id=submit.DEFAULT_DATASTORE_REPO)
    with pytest.raises(submit.ProductionDatastoreGuardError):
        submitter.ensure_writable()


def test_submitter_checks_guard_on_publish(enforce_production_guard) -> None:
    """DatastoreSubmitter.publish must trigger the production guard."""
    submitter = submit.DatastoreSubmitter(api=None, repo_id=submit.DEFAULT_DATASTORE_REPO)
    with pytest.raises(submit.ProductionDatastoreGuardError):
        submitter.publish(operations=[], description='', message='')


def test_submitter_allows_with_flag(enforce_production_guard) -> None:
    """DatastoreSubmitter with allow_production=True bypasses the check."""
    submitter = submit.DatastoreSubmitter(
        api=None, repo_id=submit.DEFAULT_DATASTORE_REPO, allow_production=True
    )
    assert submitter.allow_production is True


def test_cli_parser_has_allow_production_flag() -> None:
    """The cron CLI parser must expose the --allow-production option."""
    parser = build_parser()
    args = parser.parse_args(['run', '--adapter', 'hle', '--allow-production'])
    assert args.allow_production is True

    default_args = parser.parse_args(['run', '--adapter', 'hle'])
    assert default_args.allow_production is False


def test_cli_blocks_local_run_against_production(
    enforce_production_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invoking cron run locally targeting production without --dry-run or flag must fail."""
    parser = build_parser()
    args = parser.parse_args(['run', '--adapter', 'hle'])
    monkeypatch.setenv('HF_TOKEN', 'fake_token')

    with pytest.raises(submit.ProductionDatastoreGuardError):
        cmd_run(args)


def test_cli_permits_local_run_with_dry_run(
    enforce_production_guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invoking cron run with --dry-run must not trigger the production guard."""
    parser = build_parser()
    args = parser.parse_args(['run', '--adapter', 'hle', '--dry-run'])
    monkeypatch.setattr('every_eval_ever.cron.runner.run', lambda *a, **kw: None)
    monkeypatch.setattr('every_eval_ever.cron.__main__._finish', lambda *a, **kw: 0)

    exit_code = cmd_run(args)
    assert exit_code == 0
