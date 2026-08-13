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
#: Names this run within its date, so a second run that day writes its own
#: manifest and report instead of replacing the first's.
RUN_TOKEN = 'run-2-1'
RAW_PREFIX = store.raw_prefix('hle', RUN_DATE, RUN_TOKEN)


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


def datastore_commits(hub) -> list[dict]:
    """The commits that went to the datastore, not to the raw store.

    A run makes two raw-store commits around its publication, so a commit is
    no longer identifiable by its position in the list.
    """
    return [
        commit
        for commit in hub.commits
        if commit.get('repo_id') != store.DEFAULT_RAW_REPO
    ]


def raw_commits(hub) -> list[dict]:
    return [
        commit
        for commit in hub.commits
        if commit.get('repo_id') == store.DEFAULT_RAW_REPO
    ]


def finish(outcome, hub, **kwargs):
    return cli._finish(
        outcome,
        spec=SPEC,
        state=kwargs.pop('state', store.RawStore(hub).read_state('hle')),
        raw_store=kwargs.pop('raw_store', store.RawStore(hub)),
        submitter=kwargs.pop('submitter', submit.DatastoreSubmitter(hub)),
        run_url=kwargs.pop('run_url', None),
        run_token=kwargs.pop('run_token', RUN_TOKEN),
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
    assert 'swe_bench_verified' in adapters  # weekly, Monday
    assert 'swe_polybench' not in adapters  # weekly, Tuesday
    assert 'helm_capabilities' not in adapters  # parked: static upstream
    assert 'bfcl' not in adapters  # not schedulable


def test_plan_carries_the_timeout_and_extra_packages(capsys) -> None:
    cli.main(['plan', '--date', '2026-08-10'])

    matrix = json.loads(capsys.readouterr().out)
    entries = {entry['adapter']: entry for entry in matrix['include']}
    assert entries['openeval']['timeout_minutes'] == 45
    # swe_bench_verified is the Monday weekly that needs an optional package.
    assert entries['swe_bench_verified']['packages'] == 'datasets'


def test_plan_gives_the_job_more_time_than_the_adapter(capsys) -> None:
    """The job also checks out, installs and uploads.

    Cancelling it at the adapter's own budget is how a run ends with records
    on a pull request and nothing in the ledger saying so.
    """
    cli.main(['plan', '--date', '2026-08-10'])

    matrix = json.loads(capsys.readouterr().out)
    assert matrix['include']
    for entry in matrix['include']:
        assert entry['job_timeout_minutes'] == (
            entry['timeout_minutes'] + catalog.JOB_TIMEOUT_BUFFER_MINUTES
        )


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


def test_a_public_raw_store_stops_the_run_before_the_adapter(
    monkeypatch, tmp_path, capsys
) -> None:
    """An hour of scraping is a poor way to learn the snapshot has nowhere
    private to go, and no adapter output should exist to be published."""
    hub = FakeHub(private=False)
    monkeypatch.setattr('huggingface_hub.HfApi', lambda *a, **k: hub)
    monkeypatch.setenv('HF_TOKEN', 'a-token')
    started = []
    monkeypatch.setattr(cli.runner, 'run', lambda *a, **k: started.append(True))

    exit_code = cli.main(
        ['run', '--adapter', 'hle', '--workdir', str(tmp_path)]
    )

    assert exit_code == 1
    assert started == []
    assert 'is public' in capsys.readouterr().err
    assert hub.commits == []


@pytest.mark.parametrize(
    ('hub_kwargs', 'unreachable', 'expected'),
    [
        ({'token_role': 'read'}, None, 'read-only'),
        ({}, 'evaleval/EEE_datastore', 'could not reach'),
    ],
)
def test_a_token_that_cannot_publish_stops_the_run_before_the_adapter(
    monkeypatch, tmp_path, capsys, hub_kwargs, unreachable, expected
) -> None:
    """Both of these already fail at the publish step, an adapter run later."""
    hub = FakeHub(**hub_kwargs)
    if unreachable:
        hub.unreachable.add(unreachable)
    monkeypatch.setattr('huggingface_hub.HfApi', lambda *a, **k: hub)
    monkeypatch.setenv('HF_TOKEN', 'a-token')
    started = []
    monkeypatch.setattr(cli.runner, 'run', lambda *a, **k: started.append(True))

    exit_code = cli.main(
        ['run', '--adapter', 'hle', '--workdir', str(tmp_path)]
    )

    assert exit_code == 1
    assert started == []
    assert expected in capsys.readouterr().err
    # Nothing was created either, so a bad token cannot leave a repo behind.
    assert hub.created == []
    assert hub.commits == []


# --- what a finished run commits -----------------------------------------


def test_a_successful_run_opens_a_pull_request_and_records_it(
    tmp_path,
) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 0

    assert datastore_commits(hub)[0]['create_pr'] is True
    state = json.loads(hub.files['state/hle.json'])
    assert state['pull_request_number'] == 42
    assert state['last_run_date'] == '2026-08-10'
    assert state['last_status'] == 'completed'
    # Committed to a pull request is not merged into the datastore, so the
    # fingerprint waits in the pending ledger until the pull request does.
    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    assert hub.files['state/hle.fingerprints'] == ''


def test_the_snapshot_is_committed_before_the_records_are_published(
    tmp_path,
) -> None:
    """Two raw-store commits, one either side of the publication.

    The first carries the snapshot and what this run is about to publish; the
    second carries the report and the ledger. Between them is the only window
    where records can exist with nothing naming them, and the first commit is
    what lets the next run close it.
    """
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub)

    order = [
        'raw' if commit.get('repo_id') == store.DEFAULT_RAW_REPO else 'records'
        for commit in hub.commits
    ]
    assert order == ['raw', 'records', 'raw']

    before, after = raw_commits(hub)
    assert {operation.path_in_repo for operation in before['operations']} == {
        f'{RAW_PREFIX}/abc.json',
        f'{RAW_PREFIX}/manifest.jsonl',
        'state/hle.inflight',
    }
    assert {operation.path_in_repo for operation in after['operations']} == {
        f'{RAW_PREFIX}/run.json',
        'state/hle.json',
        'state/hle.fingerprints',
        'state/hle.pending',
        'state/hle.inflight',
    }
    assert before['parent_commit'] == 'headsha'
    # The second commit builds on the first rather than on the head the state
    # was read at, which the first one moved.
    assert after['parent_commit'] == 'newsha'
    # Nothing is left in flight once the ledger names it.
    assert json.loads(hub.files['state/hle.inflight'])['records'] == []
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
    assert report['status'] == 'completed'
    assert report['pull_request']['number'] == 42
    assert report['raw_reference'] == RAW_PREFIX


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

    upload_commit = datastore_commits(hub)[0]
    assert upload_commit['revision'] == 'refs/pr/12'
    assert 'create_pr' not in upload_commit
    # The merged ledger is untouched; the new record waits on the still-open
    # pull request it was just committed to.
    assert hub.files['state/hle.fingerprints'] == 'known-0\n'
    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    # The body now describes this run, not whichever one opened the request.
    assert [number for number, _ in hub.edited_comments] == [12]
    body = hub.discussions[0].body
    assert RAW_PREFIX in body
    assert '2026-08-10' in body


def test_a_stale_description_is_reported_and_not_fatal(tmp_path) -> None:
    """The records reached the pull request; only its body did not."""
    hub = FakeHub(
        {'state/hle.json': json.dumps({'pull_request_number': 12})},
        discussions=[cron_pr(12)],
    )
    hub.edit_comment_error = RuntimeError('403 Forbidden')
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 0

    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
    assert any('403 Forbidden' in message for message in report['messages'])


# --- records published but never recorded --------------------------------


def inflight(hub, **kwargs) -> str:
    """Put an in-flight batch in the store, as a crashed run would leave it."""
    batch = store.InflightBatch(
        adapter='hle',
        run_date='2026-08-09',
        run_token='run-1-1',
        **kwargs,
    )
    hub.files[store.inflight_path('hle')] = batch.to_json()
    return batch.to_json()


def record_entry(index: int) -> dict:
    return {
        'fingerprint': f'fingerprint-{index}',
        'paths': [f'data/hle/org/model{index}/{index}.json'],
    }


def test_records_that_reached_the_pull_request_are_recorded_late() -> None:
    """The run that uploaded them died before writing its ledger.

    Without this, the next run finds those fingerprints unknown, uploads the
    same evaluations again under fresh UUID paths, and the pull request ends
    up holding each of them twice.
    """
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.files[record_entry(0)['paths'][0]] = '{}'
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    note = cli._reconcile_inflight(
        state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
    )

    assert state.pending_fingerprints == {'fingerprint-0'}
    assert state.pull_request_number == 12
    assert 'published 1 of 1' in note


def test_records_that_never_arrived_are_published_again() -> None:
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.files[record_entry(0)['paths'][0]] = '{}'
    inflight(
        hub,
        pull_request_number=12,
        records=[record_entry(0), record_entry(1)],
    )
    state = store.AdapterState(adapter='hle')

    note = cli._reconcile_inflight(
        state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
    )

    assert state.pending_fingerprints == {'fingerprint-0'}
    assert 'published 1 of 2' in note


def test_a_batch_in_flight_to_no_pull_request_is_published_again() -> None:
    """A cold start whose opening commit never landed."""
    hub = FakeHub()
    inflight(hub, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    note = cli._reconcile_inflight(
        state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
    )

    assert state.pending_fingerprints == set()
    assert state.fingerprints == set()
    assert 'opened no pull request' in note


def test_a_batch_in_flight_to_a_merged_pull_request_becomes_durable() -> None:
    """Merged means those records are in the datastore, not on a branch."""
    hub = FakeHub(discussions=[cron_pr(12, status='merged')])
    hub.files[record_entry(0)['paths'][0]] = '{}'
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    cli._reconcile_inflight(
        state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
    )

    assert state.fingerprints == {'fingerprint-0'}
    assert state.pending_fingerprints == set()


def test_a_batch_in_flight_to_a_closed_pull_request_is_forgotten() -> None:
    hub = FakeHub(discussions=[cron_pr(12, status='closed')])
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    note = cli._reconcile_inflight(
        state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
    )

    assert state.pending_fingerprints == set()
    assert state.fingerprints == set()
    assert 'closed without merging' in note


def test_an_unreadable_pull_request_stops_the_run() -> None:
    """Both guesses lose: one buries records, the other duplicates them."""
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.list_files_error = ConnectionError('network is unreachable')
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    with pytest.raises(submit.SubmissionError, match='without recording'):
        cli._reconcile_inflight(
            state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
        )


def test_settling_the_same_batch_twice_settles_it_the_same_way() -> None:
    """A run that dies before its own commit leaves the file for the next."""
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.files[record_entry(0)['paths'][0]] = '{}'
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    first = store.AdapterState(adapter='hle')
    second = store.AdapterState(adapter='hle')

    for state in (first, second):
        cli._reconcile_inflight(
            state, store.RawStore(hub), submit.DatastoreSubmitter(hub)
        )

    assert first.pending_fingerprints == second.pending_fingerprints
    assert second.pending_fingerprints == {'fingerprint-0'}


def test_a_run_settles_what_was_in_flight_before_the_adapter_starts(
    monkeypatch, tmp_path
) -> None:
    """End to end: the records are known before the adapter is asked for
    anything, so it never restages them."""
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.files[record_entry(0)['paths'][0]] = '{}'
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    monkeypatch.setattr('huggingface_hub.HfApi', lambda *a, **k: hub)
    monkeypatch.setenv('HF_TOKEN', 'a-token')
    seen = {}

    def fake_run(spec, workdir, **kwargs):
        seen['known'] = set(kwargs['known_fingerprints'])
        outcome = make_outcome(tmp_path, uploaded=0)
        write_capture(outcome.raw_dir)
        return outcome

    monkeypatch.setattr(cli.runner, 'run', fake_run)

    exit_code = cli.main(
        [
            'run',
            '--adapter',
            'hle',
            '--workdir',
            str(tmp_path / 'work'),
            '--run-id',
            RUN_TOKEN,
        ]
    )

    assert exit_code == 0
    assert seen['known'] == {'fingerprint-0'}
    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    assert json.loads(hub.files['state/hle.inflight'])['records'] == []
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
    assert any('without recording' in m for m in report['messages'])


def test_a_dry_run_settles_nothing() -> None:
    hub = FakeHub(discussions=[cron_pr(12)])
    inflight(hub, pull_request_number=12, records=[record_entry(0)])
    state = store.AdapterState(adapter='hle')

    assert cli._reconcile_inflight(state, store.RawStore(hub), None) is None
    assert state.pending_fingerprints == set()


def test_a_merged_pull_request_promotes_its_pending_fingerprints() -> None:
    hub = FakeHub(discussions=[cron_pr(12, status='merged')])
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=12,
        fingerprints={'old-0'},
        pending_fingerprints={'known-0'},
    )

    note = cli._reconcile_pending(state, submit.DatastoreSubmitter(hub))

    assert state.fingerprints == {'old-0', 'known-0'}
    assert state.pending_fingerprints == set()
    assert state.pull_request_number is None
    assert 'merged' in note


def test_a_closed_pull_request_requeues_its_pending_fingerprints() -> None:
    """Fingerprints from a rejected pull request must be forgotten.

    The ledger holds records committed to a pull request, not records merged
    into the datastore. Kept after that pull request is closed unmerged, they
    would filter the same records out of every later run before publication
    is attempted, so the replacement pull request could never open.
    """
    hub = FakeHub(discussions=[cron_pr(12, status='closed')])
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=12,
        fingerprints={'old-0'},
        pending_fingerprints={'known-0'},
    )

    note = cli._reconcile_pending(state, submit.DatastoreSubmitter(hub))

    assert state.fingerprints == {'old-0'}
    assert state.pending_fingerprints == set()
    assert state.pull_request_number is None
    assert 'closed without merging' in note


def test_an_open_pull_request_keeps_its_pending_fingerprints() -> None:
    hub = FakeHub(discussions=[cron_pr(12)])
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=12,
        pending_fingerprints={'known-0'},
    )

    assert cli._reconcile_pending(state, submit.DatastoreSubmitter(hub)) is None
    assert state.pending_fingerprints == {'known-0'}
    assert state.known_fingerprints == {'known-0'}


def test_a_dry_run_does_not_ask_the_hub_about_pending_records() -> None:
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=12,
        pending_fingerprints={'known-0'},
    )

    assert cli._reconcile_pending(state, None) is None
    assert state.pending_fingerprints == {'known-0'}


def test_an_unanswerable_pull_request_fate_stops_the_run() -> None:
    hub = FakeHub(discussions=[cron_pr(12)])
    hub.details_error = ConnectionError('boom')
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=12,
        pending_fingerprints={'known-0'},
    )

    with pytest.raises(submit.SubmissionError):
        cli._reconcile_pending(state, submit.DatastoreSubmitter(hub))


def test_records_from_a_closed_pull_request_are_resubmitted(
    monkeypatch, tmp_path
) -> None:
    """The end-to-end shape of the requeue: the closed pull request's
    fingerprints are not handed to the runner as known, so the unchanged
    records upload again, and a fresh pull request opens to carry them."""
    hub = FakeHub(
        {
            'state/hle.json': json.dumps({'pull_request_number': 12}),
            'state/hle.pending': 'known-0\n',
        },
        discussions=[cron_pr(12, status='closed')],
    )
    monkeypatch.setattr('huggingface_hub.HfApi', lambda *a, **k: hub)
    monkeypatch.setenv('HF_TOKEN', 'a-token')
    seen = {}

    def fake_run(spec, workdir, **kwargs):
        seen['known'] = set(kwargs['known_fingerprints'])
        outcome = make_outcome(tmp_path)
        write_capture(outcome.raw_dir)
        return outcome

    monkeypatch.setattr(cli.runner, 'run', fake_run)

    exit_code = cli.main(
        [
            'run',
            '--adapter',
            'hle',
            '--workdir',
            str(tmp_path / 'work'),
            '--run-id',
            RUN_TOKEN,
        ]
    )

    assert exit_code == 0
    assert seen['known'] == set()
    assert any(commit.get('create_pr') for commit in hub.commits)
    state = json.loads(hub.files['state/hle.json'])
    assert state['pull_request_number'] == 42
    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
    assert any(
        'closed without merging' in message for message in report['messages']
    )


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

    remembered = set(hub.files['state/hle.pending'].split())
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

    state = json.loads(hub.files['state/hle.json'])
    assert state['last_raw_date'] == '2026-08-10'
    # The whole path, because a date names more than one directory now.
    assert state['last_raw_prefix'] == RAW_PREFIX


def test_a_second_run_on_one_day_keeps_the_first_runs_manifest(
    tmp_path,
) -> None:
    """A re-run is how the interesting days go: a cancelled job, a source
    that was down at 03:17. Overwriting the first attempt's manifest and
    report loses the account of what it fetched and what happened to it."""
    hub = FakeHub()
    first = make_outcome(tmp_path / 'first')
    write_capture(first.raw_dir)
    finish(first, hub, run_token='run-2-1')

    second = make_outcome(tmp_path / 'second')
    write_capture(second.raw_dir)
    finish(second, hub, run_token='run-2-2')

    first_prefix = store.raw_prefix('hle', RUN_DATE, 'run-2-1')
    second_prefix = store.raw_prefix('hle', RUN_DATE, 'run-2-2')
    assert f'{first_prefix}/manifest.jsonl' in hub.files
    assert f'{second_prefix}/manifest.jsonl' in hub.files
    assert (
        json.loads(hub.files[f'{first_prefix}/run.json'])['run_token']
        == 'run-2-1'
    )
    assert (
        json.loads(hub.files[f'{second_prefix}/run.json'])['run_token']
        == 'run-2-2'
    )
    # The second run reads the first's manifest, so the payload it already
    # holds is referenced rather than stored again.
    assert f'{second_prefix}/abc.json' not in hub.files
    manifest = hub.files[f'{second_prefix}/manifest.jsonl']
    assert f'{first_prefix}/abc.json' in manifest
    assert json.loads(hub.files['state/hle.json'])['last_raw_prefix'] == (
        second_prefix
    )


def test_a_state_written_before_run_scoped_snapshots_still_de_duplicates(
    tmp_path,
) -> None:
    """A ledger from the old layout names a date, which was the directory."""
    hub = FakeHub(
        {
            'state/hle.json': json.dumps({'last_raw_date': '2026-08-09'}),
            'raw/hle/2026-08-09/manifest.jsonl': json.dumps(
                {'kind': 'payload', 'sha256': 'abc', 'path': 'abc.json'}
            )
            + '\n',
        }
    )
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub)

    assert f'{RAW_PREFIX}/abc.json' not in hub.files
    assert (
        'raw/hle/2026-08-09/abc.json'
        in (hub.files[f'{RAW_PREFIX}/manifest.jsonl'])
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
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
    assert report['records_skipped_unchanged'] == 3


def test_a_failed_run_records_evidence_and_exits_non_zero(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path, status='failed', uploaded=0)
    outcome.messages.append('validation did not pass')
    write_capture(outcome.raw_dir)

    assert finish(outcome, hub) == 1

    assert not any(commit.get('create_pr') for commit in hub.commits)
    report = json.loads(hub.files[f'{RAW_PREFIX}/run.json'])
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
            run_token=RUN_TOKEN,
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
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)
    real_create_commit = hub.create_commit

    def fail_the_datastore(**kwargs):
        if kwargs.get('repo_id') != store.DEFAULT_RAW_REPO:
            raise RuntimeError('502 bad gateway')
        return real_create_commit(**kwargs)

    hub.create_commit = fail_the_datastore

    with pytest.raises(submit.SubmissionError):
        finish(outcome, hub)

    assert 'state/hle.fingerprints' not in hub.files
    # The snapshot commit went first, so the records this run meant to
    # publish are named even though the publication failed. Nothing landed,
    # so the next run's reconciliation finds nothing on the pull request and
    # publishes them again.
    assert json.loads(hub.files['state/hle.inflight'])['records'] == [
        {
            'fingerprint': 'fingerprint-0',
            'paths': [
                'data/hle/org/model0/00000000-0000-4000-8000-000000000000.json'
            ],
        }
    ]


def test_an_unanswerable_batch_stays_in_flight(tmp_path) -> None:
    """The commit errored, the ref could not be read, and the records may be
    on the pull request anyway. Clearing the in-flight file here is how the
    next run uploads them a second time; keeping them in it is what lets that
    run ask the pull request instead of guessing."""
    hub = FakeHub(
        {'state/hle.json': json.dumps({'pull_request_number': 12})},
        discussions=[cron_pr(12)],
    )
    outcome = make_outcome(tmp_path, uploaded=3)
    write_capture(outcome.raw_dir)
    real_create_commit = hub.create_commit

    def lose_the_second_batch(**kwargs):
        if (
            kwargs.get('repo_id') != store.DEFAULT_RAW_REPO
            and len(datastore_commits(hub)) >= 1
        ):
            # The reconciliation read fails too, so the batch's fate is
            # unknowable this run.
            hub.list_files_error = ConnectionError('network is unreachable')
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = lose_the_second_batch

    exit_code = finish(
        outcome, hub, submitter=submit.DatastoreSubmitter(hub, batch_size=1)
    )

    assert exit_code == 1
    # The batch that landed is in the ledger; the never-attempted third
    # record is safe to upload again, so neither stays in flight. Only the
    # unanswerable second record does, addressed to the pull request the
    # next run must ask about it.
    assert hub.files['state/hle.pending'] == 'fingerprint-0\n'
    batch = json.loads(hub.files['state/hle.inflight'])
    assert batch['pull_request_number'] == 12
    assert [record['fingerprint'] for record in batch['records']] == [
        'fingerprint-1'
    ]


def test_the_run_url_reaches_the_pull_request_body(tmp_path) -> None:
    hub = FakeHub()
    outcome = make_outcome(tmp_path)
    write_capture(outcome.raw_dir)

    finish(outcome, hub, run_url='https://ci.example/run/9')

    assert (
        'https://ci.example/run/9'
        in datastore_commits(hub)[0]['commit_description']
    )


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
