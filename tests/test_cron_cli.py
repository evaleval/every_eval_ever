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

from every_eval_ever.adapters import catalog
from every_eval_ever.cron import __main__ as cli
from every_eval_ever.cron import runner, store, submit
from tests.test_cron_store_and_submit import FakeHub, cron_pr

RUN_DATE = date(2026, 8, 10)  # a Monday
SPEC = catalog.get('hle')


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
        discussions=[cron_pr(12)],
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


def test_records_that_landed_before_a_failure_are_still_remembered(
    tmp_path,
) -> None:
    """The job fails, but not by forgetting what it published.

    The batches that landed are irreversible. Leaving their fingerprints out
    of the ledger would make the next run publish them again under fresh
    paths, so the pull request would hold each evaluation twice.
    """
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.files['state/hle.json'] = json.dumps({'pull_request_number': 12})
    outcome = make_outcome(tmp_path, uploaded=3)
    write_capture(outcome.raw_dir)
    submitter = submit.DatastoreSubmitter(hub, batch_size=1)
    real_create_commit = hub.create_commit

    def fail_after_the_first_batch(**kwargs):
        # Only the datastore batches fail. The raw-store commit that records
        # what landed has to go through, which is the point of the test.
        if kwargs.get('revision') == 'refs/pr/12' and any(
            commit.get('revision') == 'refs/pr/12' for commit in hub.commits
        ):
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = fail_after_the_first_batch

    assert finish(outcome, hub, submitter=submitter) == 1

    remembered = set(hub.files['state/hle.fingerprints'].split())
    assert remembered == {'fingerprint-0'}
    assert json.loads(hub.files['state/hle.json'])['pull_request_number'] == 12


def test_the_snapshot_pointer_only_moves_when_a_snapshot_was_written(
    tmp_path,
) -> None:
    """A pointer at a date with no manifest re-uploads everything next run."""
    hub = FakeHub(
        {
            'state/hle.json': json.dumps({'last_raw_date': '2026-08-09'}),
        }
    )
    outcome = make_outcome(tmp_path)  # no raw capture written

    finish(outcome, hub)

    assert json.loads(hub.files['state/hle.json'])['last_raw_date'] == (
        '2026-08-09'
    )


def test_the_snapshot_pointer_moves_when_one_was(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub)

    assert json.loads(hub.files['state/hle.json'])['last_raw_date'] == (
        '2026-08-10'
    )


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


def test_a_missing_package_is_healthy(tmp_path) -> None:
    """with_packages is installed from the matrix, so this is a build gap."""
    hub = FakeHub()
    outcome = make_outcome(
        tmp_path, status='skipped_missing_dependency', uploaded=0
    )

    assert finish(outcome, hub) == 0


def test_a_missing_credential_exits_non_zero(tmp_path) -> None:
    """A green job here is indistinguishable from an unchanged leaderboard."""
    hub = FakeHub()
    outcome = make_outcome(
        tmp_path, status='skipped_missing_credential', uploaded=0
    )

    assert finish(outcome, hub) == 1

    assert (
        json.loads(hub.files['state/hle.json'])['last_status']
        == 'skipped_missing_credential'
    )


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


def test_a_scheduled_run_without_a_token_fails_instead_of_publishing_nothing(
    capsys,
) -> None:
    """A missing secret must not look like an unchanged leaderboard.

    Falling back to a dry run left the nightly job green while it published
    nothing, so an expired token could go unnoticed for as long as nobody
    compared the datastore against the run log.
    """
    assert cli.main(['run', '--adapter', 'hle']) == 1

    error = capsys.readouterr().err
    assert 'HF_TOKEN' in error
    assert '--dry-run' in error


def test_an_explicit_dry_run_still_needs_no_token(monkeypatch, capsys) -> None:
    """Not publishing stays available; it just has to be asked for."""
    seen = {}

    def record(args):
        seen['dry_run'] = args.dry_run
        return 0

    monkeypatch.setattr(cli, 'cmd_run', record)

    assert cli.main(['run', '--adapter', 'hle', '--dry-run']) == 0
    assert seen['dry_run'] is True


def test_a_store_error_becomes_a_reported_failure(monkeypatch, capsys) -> None:
    def explode(args):
        raise store.StoreError('could not read state/hle.json')

    monkeypatch.setattr(cli, 'cmd_run', explode)

    assert cli.main(['run', '--adapter', 'hle']) == 1
    assert 'could not read' in capsys.readouterr().err
