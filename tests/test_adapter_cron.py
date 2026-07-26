from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).parents[1] / 'utils' / 'scripts' / 'run_adapters.py'
SPEC = importlib.util.spec_from_file_location('eee_run_adapters', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_adapters = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_adapters
SPEC.loader.exec_module(run_adapters)


def _args(
    *, adapters: list[str] | None = None, force_all: bool = False
) -> argparse.Namespace:
    return argparse.Namespace(adapters=adapters, force_all=force_all)


def test_force_all_is_limited_to_explicit_allowlist():
    selected = run_adapters.selected_contracts(_args(force_all=True))

    assert {contract.name for contract in selected} == set(
        run_adapters.CRON_ALLOWLIST
    )


def test_dead_adapters_are_not_scheduled():
    assert {
        'arc_agi',
        'hfopenllm_v2',
        'livecodebenchpro',
        'rewardbench',
    }.isdisjoint(run_adapters.CRON_ALLOWLIST)


def test_manual_selection_can_run_ready_non_allowlisted_adapter():
    selected = run_adapters.selected_contracts(_args(adapters=['helm']))

    assert [contract.name for contract in selected] == ['helm']


def test_manual_selection_rejects_blocked_adapter():
    with pytest.raises(ValueError, match='not cron-ready'):
        run_adapters.selected_contracts(_args(adapters=['arc_agi']))


def test_allowlist_and_contract_table_pass_audit():
    assert run_adapters.audit_contracts() == []


def test_every_scheduled_adapter_captures_a_replayable_raw_input():
    scheduled = {
        contract.name: contract
        for contract in run_adapters.CONTRACTS
        if contract.name in run_adapters.CRON_ALLOWLIST
    }

    assert set(scheduled) == set(run_adapters.CRON_ALLOWLIST)
    assert all(
        contract.raw_capture is not None for contract in scheduled.values()
    )


def test_raw_capture_is_passed_outside_the_validation_tree(tmp_path):
    contract = next(
        contract
        for contract in run_adapters.CONTRACTS
        if contract.name == 'mmlu_pro'
    )

    command = run_adapters.command_for(
        contract,
        contract.commands[0],
        tmp_path,
    )

    raw_flag_index = command.index('--save-raw-csv')
    assert command[raw_flag_index + 1] == str(tmp_path / 'raw' / 'payload.csv')
    assert str(tmp_path / 'data' / 'MMLU-Pro') in command


def test_collect_raw_artifacts_uses_available_workspaces(tmp_path):
    contract = next(
        contract
        for contract in run_adapters.CONTRACTS
        if contract.name == 'mmlu_pro'
    )
    raw_path = tmp_path / 'raw' / 'payload.csv'
    raw_path.parent.mkdir()
    raw_path.write_text('model,score\nexample,1\n')

    artifacts = run_adapters.collect_raw_artifacts(
        [contract],
        {'mmlu_pro': tmp_path},
    )

    assert len(artifacts) == 1
    assert artifacts[0].adapter == 'mmlu_pro'
    assert artifacts[0].local_path == raw_path


def test_validation_failure_keeps_captured_raw_workspace(
    tmp_path,
    monkeypatch,
):
    contract = run_adapters.AdapterContract(
        'example',
        ('example',),
        raw_capture=run_adapters.RawCaptureSpec(
            '--save-raw-json',
            'payload.json',
            'application/json',
        ),
    )

    def fake_run(command, **kwargs):
        if 'validate' not in command:
            output_path = Path(command[command.index('--output-dir') + 1])
            output_path.mkdir(parents=True)
            raw_path = Path(command[command.index('--save-raw-json') + 1])
            raw_path.write_text('{"source": "captured"}\n')
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='',
                stderr='',
            )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout='[{"valid": false, "warnings": []}]',
            stderr='',
        )

    monkeypatch.setattr(run_adapters.subprocess, 'run', fake_run)
    monkeypatch.setenv('RUNNER_TEMP', str(tmp_path))

    workspace, result = run_adapters.run_contract(contract, {})

    assert result['status'] == 'failed'
    assert workspace is not None
    assert (workspace / 'raw' / 'payload.json').is_file()


@pytest.mark.parametrize(
    ('ingestion_repo', 'ingestion_token', 'expected_error'),
    [
        (
            'evaleval/eee-cron-ingestion',
            None,
            'EEE_INGESTION_HF_TOKEN is required',
        ),
        (
            None,
            'archive-token',
            'EEE_INGESTION_REPO_ID or --ingestion-repo is required',
        ),
    ],
)
def test_cron_archive_configuration_has_no_implicit_fallback(
    tmp_path,
    monkeypatch,
    ingestion_repo,
    ingestion_token,
    expected_error,
):
    args = argparse.Namespace(
        archive_only=False,
        audit=False,
        dedup_mode='deferred',
        dry_run=False,
        ingestion_repo=ingestion_repo,
        manifest=None,
        report=tmp_path / 'report.json',
    )
    monkeypatch.setattr(run_adapters, 'parse_args', lambda: args)
    monkeypatch.setattr(run_adapters, 'audit_contracts', lambda: [])
    monkeypatch.setattr(run_adapters, 'selected_contracts', lambda _: [])
    monkeypatch.setenv('HF_TOKEN', 'datastore-token')
    if ingestion_token is None:
        monkeypatch.delenv('EEE_INGESTION_HF_TOKEN', raising=False)
    else:
        monkeypatch.setenv('EEE_INGESTION_HF_TOKEN', ingestion_token)

    assert run_adapters.main() == 1
    assert expected_error in args.report.read_text()


def test_archive_only_never_reads_datastore_or_requires_public_token(
    tmp_path,
    monkeypatch,
):
    args = argparse.Namespace(
        archive_only=True,
        audit=False,
        dedup_mode='deferred',
        ingestion_repo='evaleval/eee-cron-ingestion',
        report=tmp_path / 'report.json',
    )
    completed_payloads = []
    monkeypatch.setattr(run_adapters, 'parse_args', lambda: args)
    monkeypatch.setattr(run_adapters, 'audit_contracts', lambda: [])
    monkeypatch.setattr(run_adapters, 'selected_contracts', lambda _: [])
    monkeypatch.setattr(run_adapters, 'HfApi', lambda token: object())
    monkeypatch.setattr(
        run_adapters,
        'archive_raw_artifacts',
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        run_adapters,
        'append_ledger_event',
        lambda *args, **kwargs: completed_payloads.append(kwargs['payload']),
    )
    monkeypatch.setattr(
        run_adapters,
        'open_collection_prs',
        lambda _: pytest.fail('archive-only read the datastore'),
    )
    monkeypatch.setenv('EEE_INGESTION_HF_TOKEN', 'archive-token')
    monkeypatch.delenv('HF_TOKEN', raising=False)

    assert run_adapters.main() == 0
    report = args.report.read_text()
    assert '"status": "archive_only"' in report
    assert completed_payloads == [
        {
            'status': 'archive_only',
            'dedup_mode': 'not_run',
            'adapter_results': {},
            'duplicates': [],
            'selected_files': {},
            'prs': {},
        }
    ]


def test_deferred_dedup_base_does_not_read_datastore_manifest(monkeypatch):
    monkeypatch.setattr(
        run_adapters,
        'load_manifest',
        lambda **_: pytest.fail('deferred dedup read the datastore manifest'),
    )

    manifest, scope = run_adapters.load_dedup_base(
        mock.Mock(),
        mode='deferred',
        manifest_path=None,
    )

    assert manifest == run_adapters.empty_manifest()
    assert scope == 'pending_prs_and_current_run'


def test_enforced_dedup_base_loads_main_manifest(monkeypatch):
    api = mock.Mock()
    expected = run_adapters.empty_manifest()
    load = mock.Mock(return_value=expected)
    monkeypatch.setattr(run_adapters, 'load_manifest', load)

    manifest, scope = run_adapters.load_dedup_base(
        api,
        mode='enforced',
        manifest_path=None,
    )

    assert manifest is expected
    assert scope == 'datastore_pending_prs_and_current_run'
    load.assert_called_once_with(
        api=api,
        dataset_repo_id=run_adapters.DATASET_REPO_ID,
        revision='main',
    )


def test_deferred_collection_pr_discloses_limited_dedup_scope():
    api = mock.Mock()
    api.create_pull_request.return_value = mock.Mock()

    run_adapters.ensure_collection_pr(
        api,
        'example',
        None,
        dedup_mode='deferred',
    )

    description = api.create_pull_request.call_args.kwargs['description']
    assert 'Datastore-wide deduplication is explicitly deferred' in description
    assert 'do not merge' in description


def test_scheduled_cron_publishes_with_explicit_deferred_dedup():
    workflow = (
        Path(__file__).parents[1] / '.github' / 'workflows' / 'adapter_cron.yml'
    ).read_text()

    assert 'EEE_CRON_DEDUP_MODE' in workflow
    assert "|| 'deferred'" in workflow
    assert 'EEE_CRON_PUBLISH_ENABLED' not in workflow
    assert '--archive-only' not in workflow


def test_deferred_publish_opens_collection_pr_without_main_manifest(
    tmp_path,
    monkeypatch,
):
    args = argparse.Namespace(
        archive_only=False,
        audit=False,
        dedup_mode='deferred',
        dry_run=False,
        ingestion_repo='evaleval/eee-cron-ingestion',
        manifest=None,
        report=tmp_path / 'report.json',
    )
    contract = run_adapters.AdapterContract('example', ('example',))
    output = tmp_path / 'result.json'
    output.write_text('{}')
    upload = mock.Mock(return_value={'example': 'https://example.test/pr/1'})

    apis = iter(
        (mock.Mock(name='ingestion_api'), mock.Mock(name='datastore_api'))
    )
    monkeypatch.setattr(run_adapters, 'parse_args', lambda: args)
    monkeypatch.setattr(run_adapters, 'audit_contracts', lambda: [])
    monkeypatch.setattr(
        run_adapters, 'selected_contracts', lambda _: [contract]
    )
    monkeypatch.setattr(
        run_adapters,
        'run_contract',
        lambda *_: (tmp_path, {'status': 'ready'}),
    )
    monkeypatch.setattr(run_adapters, 'collect_raw_artifacts', lambda *_: [])
    monkeypatch.setattr(
        run_adapters, 'archive_raw_artifacts', lambda *_, **__: []
    )
    monkeypatch.setattr(
        run_adapters, 'append_ledger_event', lambda *_, **__: None
    )
    monkeypatch.setattr(run_adapters, 'HfApi', lambda token: next(apis))
    monkeypatch.setattr(run_adapters, 'open_collection_prs', lambda _: {})
    monkeypatch.setattr(
        run_adapters,
        'load_manifest',
        lambda **_: pytest.fail('deferred publish loaded main manifest'),
    )
    monkeypatch.setattr(
        run_adapters,
        'augment_manifest_with_pending_prs',
        lambda _api, manifest, _prs: manifest,
    )
    monkeypatch.setattr(
        run_adapters,
        'select_new_files',
        lambda *_: ({'example': {'data/example/result.json': output}}, []),
    )
    monkeypatch.setattr(run_adapters, 'upload_collections', upload)
    monkeypatch.setenv('EEE_INGESTION_HF_TOKEN', 'archive-token')
    monkeypatch.setenv('HF_TOKEN', 'datastore-token')

    assert run_adapters.main() == 0
    assert upload.call_args.kwargs['dedup_mode'] == 'deferred'
    report = json.loads(args.report.read_text())
    assert report['status'] == 'uploaded'
    assert report['dedup_scope'] == 'pending_prs_and_current_run'
