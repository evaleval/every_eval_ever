"""The command line, and what it commits after a run.

The interesting behaviour is in ``_finish``: which fingerprints get recorded,
whether a quiet run still updates state, and whether a failure can ever look
like a success.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from every_eval_ever.adapters import registry
from every_eval_ever.cron import __main__ as cli
from every_eval_ever.cron import runner, store, submit
from tests.test_cron_store_and_submit import FakeDiscussion, FakeHub

RUN_DATE = date(2026, 8, 10)  # a Monday
SPEC = registry.get('hle')


@pytest.fixture(autouse=True)
def no_ci_environment(monkeypatch):
    for name in (
        'GITHUB_STEP_SUMMARY',
        'GITHUB_SERVER_URL',
        'GITHUB_REPOSITORY',
        'GITHUB_RUN_ID',
        'HF_TOKEN',
        'HUGGING_FACE_HUB_TOKEN',
    ):
        monkeypatch.delenv(name, raising=False)


def make_outcome(
    tmp_path: Path,
    *,
    status: runner.Status = 'completed',
    uploaded: int = 1,
    skipped: int = 0,
) -> runner.RunOutcome:
    """Build an outcome with a real upload tree on disk."""
    upload = tmp_path / 'upload'
    raw = tmp_path / 'raw'
    raw.mkdir(parents=True, exist_ok=True)

    def record(index: int) -> runner.StagedRecord:
        repo_path = f'data/hle/org/model{index}/{index:08d}-0000-4000-8000-000000000000.json'
        target = upload / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}', encoding='utf-8')
        return runner.StagedRecord(
            aggregate=target,
            repo_path=repo_path,
            collection='hle',
            model_id=f'org/model{index}',
            fingerprint=f'fingerprint-{index}',
        )

    return runner.RunOutcome(
        adapter='hle',
        run_date=RUN_DATE,
        status=status,
        staging_dir=tmp_path / 'staging',
        upload_dir=upload,
        raw_dir=raw,
        uploaded=[record(index) for index in range(uploaded)],
        skipped_unchanged=[
            runner.StagedRecord(
                aggregate=tmp_path / 'ignored.json',
                repo_path=f'data/hle/org/skipped{index}/x.json',
                collection='hle',
                model_id=f'org/skipped{index}',
                fingerprint=f'known-{index}',
            )
            for index in range(skipped)
        ],
    )


def write_capture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / 'abc.json').write_bytes(b'{"rows": []}')
    (raw_dir / store.MANIFEST_NAME).write_text(
        json.dumps(
            {
                'kind': 'payload',
                'sha256': 'abc',
                'path': 'abc.json',
                'url': 'https://example/board.json',
            }
        )
        + '\n',
        encoding='utf-8',
    )


def finish(outcome, hub, **kwargs):
    return cli._finish(
        outcome,
        spec=SPEC,
        state=kwargs.pop('state', store.RawStore(hub).read_state('hle')),
        raw_store=kwargs.pop('raw_store', store.RawStore(hub)),
        submitter=kwargs.pop('submitter', submit.DatastoreSubmitter(hub)),
        run_url=kwargs.pop('run_url', None),
        dry_run=kwargs.pop('dry_run', False),
    )


# --- list and plan -------------------------------------------------------


def test_list_shows_every_adapter_including_the_unrunnable(capsys) -> None:
    assert cli.main(['list']) == 0

    printed = capsys.readouterr().out
    assert 'hle' in printed
    assert 'bfcl' in printed
    assert 'no live fetch path' in printed


def test_list_json_is_machine_readable(capsys) -> None:
    assert cli.main(['list', '--format', 'json']) == 0

    rows = json.loads(capsys.readouterr().out)
    by_key = {row['adapter']: row for row in rows}
    assert by_key['exgentic']['packages'] == ['datasets']
    assert by_key['bfcl']['runnable'] is False


def test_plan_emits_a_matrix_for_the_date(capsys) -> None:
    assert cli.main(['plan', '--date', '2026-08-10']) == 0

    matrix = json.loads(capsys.readouterr().out)
    adapters = {entry['adapter'] for entry in matrix['include']}
    assert 'hle' in adapters  # daily
    assert 'helm_capabilities' in adapters  # weekly, Monday
    assert 'helm_lite' not in adapters  # weekly, Tuesday
    assert 'bfcl' not in adapters  # not schedulable


def test_plan_carries_the_timeout_and_extra_packages(capsys) -> None:
    cli.main(['plan', '--date', '2026-08-10'])

    matrix = json.loads(capsys.readouterr().out)
    entries = {entry['adapter']: entry for entry in matrix['include']}
    assert entries['openeval']['timeout_minutes'] == 45
    # swe_bench_verified is the Monday weekly that needs an optional package.
    assert entries['swe_bench_verified']['packages'] == 'datasets'


def test_an_adapter_without_credentials_stays_in_the_plan(capsys) -> None:
    """Reporting a skip is visible; dropping it from the matrix is not."""
    cli.main(['plan', '--date', '2026-08-10'])

    matrix = json.loads(capsys.readouterr().out)
    adapters = {entry['adapter'] for entry in matrix['include']}
    assert 'artificial_analysis' in adapters


# --- run guards ----------------------------------------------------------


def test_an_unknown_adapter_is_rejected(capsys) -> None:
    assert cli.main(['run', '--adapter', 'nope']) == 1
    assert 'unknown adapter' in capsys.readouterr().err


def test_an_unschedulable_adapter_is_rejected(capsys) -> None:
    assert cli.main(['run', '--adapter', 'bfcl']) == 1
    assert 'not schedulable' in capsys.readouterr().err


# --- what a finished run commits -----------------------------------------


def test_a_successful_run_opens_a_pull_request_and_records_it(
    tmp_path,
) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 0

    assert hub.commits[0]['create_pr'] is True
    state = json.loads(hub.files['state/hle.json'])
    assert state['pull_request_number'] == 42
    assert state['last_run_date'] == '2026-08-10'
    assert state['last_status'] == 'completed'
    assert hub.files['state/hle.fingerprints'] == 'fingerprint-0\n'


def test_the_raw_snapshot_and_run_report_land_in_one_commit(
    tmp_path,
) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub)

    store_commit = hub.commits[-1]
    paths = {operation.path_in_repo for operation in store_commit['operations']}
    assert paths == {
        'raw/hle/2026-08-10/abc.json',
        'raw/hle/2026-08-10/manifest.jsonl',
        'raw/hle/2026-08-10/run.json',
        'state/hle.json',
        'state/hle.fingerprints',
    }
    assert store_commit['parent_commit'] == 'headsha'
    report = json.loads(hub.files['raw/hle/2026-08-10/run.json'])
    assert report['status'] == 'completed'
    assert report['pull_request']['number'] == 42
    assert report['raw_reference'] == 'raw/hle/2026-08-10'


def test_a_second_run_reuses_the_remembered_pull_request(tmp_path) -> None:
    hub = FakeHub(
        {
            'state/hle.json': json.dumps(
                {
                    'pull_request_number': 12,
                    'last_run_date': '2026-08-09',
                    'last_raw_date': '2026-08-09',
                }
            ),
            'state/hle.fingerprints': 'known-0\n',
        },
        discussions=[
            FakeDiscussion(12, '[Submission] cron: hle — automated ingestion')
        ],
    )
    outcome = make_outcome(tmp_path, uploaded=1, skipped=1)
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 0

    upload_commit = hub.commits[0]
    assert upload_commit['revision'] == 'refs/pr/12'
    assert 'create_pr' not in upload_commit
    assert set(hub.files['state/hle.fingerprints'].split()) == {
        'known-0',
        'fingerprint-0',
    }


def test_a_run_with_nothing_new_still_records_that_it_ran(tmp_path) -> None:
    """Otherwise a quiet adapter looks like one that has never run."""
    hub = FakeHub()
    outcome = make_outcome(tmp_path, uploaded=0, skipped=3)
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 0

    assert not any(commit.get('create_pr') for commit in hub.commits)
    state = json.loads(hub.files['state/hle.json'])
    assert state['last_run_date'] == '2026-08-10'
    report = json.loads(hub.files['raw/hle/2026-08-10/run.json'])
    assert report['records_skipped_unchanged'] == 3


def test_a_failed_run_records_evidence_and_exits_non_zero(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path, status='failed', uploaded=0)
    outcome.messages.append('validation did not pass')
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 1

    assert not any(commit.get('create_pr') for commit in hub.commits)
    report = json.loads(hub.files['raw/hle/2026-08-10/run.json'])
    assert report['status'] == 'failed'
    assert json.loads(hub.files['state/hle.json'])['last_status'] == 'failed'


def test_a_skipped_run_is_healthy(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(
        tmp_path, status='skipped_missing_credential', uploaded=0
    )

    assert finish(outcome, hub) == 0


def test_a_dry_run_touches_neither_repository(tmp_path, capsys) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    assert (
        cli._finish(
            outcome,
            spec=SPEC,
            state=store.AdapterState(adapter='hle'),
            raw_store=None,
            submitter=None,
            run_url=None,
            dry_run=True,
        )
        == 0
    )

    assert hub.commits == []
    assert 'dry run' in capsys.readouterr().out


def test_an_upload_failure_leaves_the_fingerprints_unrecorded(
    tmp_path,
) -> None:
    """So the next run retries the records instead of forgetting them."""
    hub = FakeHub()
    hub.commit_error = RuntimeError('502 bad gateway')
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    with pytest.raises(submit.SubmissionError):
        finish(outcome, hub)

    assert 'state/hle.fingerprints' not in hub.files


def test_the_run_url_reaches_the_pull_request_body(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub, run_url='https://ci.example/run/9')

    assert 'https://ci.example/run/9' in hub.commits[0]['commit_description']


def test_a_step_summary_is_written_when_ci_asks_for_one(
    tmp_path, monkeypatch
) -> None:
    summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub)

    text = summary.read_text(encoding='utf-8')
    assert '`hle`' in text
    assert 'completed' in text


def test_a_store_error_becomes_a_reported_failure(monkeypatch, capsys) -> None:
    def explode(args):
        raise store.StoreError('could not read state/hle.json')

    monkeypatch.setattr(cli, 'cmd_run', explode)

    assert cli.main(['run', '--adapter', 'hle']) == 1
    assert 'could not read' in capsys.readouterr().err
