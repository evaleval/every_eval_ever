"""Command line for scheduled ingestion.

    uv run python -m every_eval_ever.cron list
    uv run python -m every_eval_ever.cron plan
    uv run python -m every_eval_ever.cron run --adapter hle --dry-run

``plan`` prints the job matrix so the schedule lives in the catalog rather
than in workflow YAML. ``run`` does one adapter end to end and exits non-zero
only when the run was actually unhealthy. An unchanged leaderboard is a clean
outcome; a crash, a validation failure, an unsnapshotted source or a missing
credential is not.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Sequence

from every_eval_ever.adapters import catalog
from every_eval_ever.cron import runner, store, submit

#: Overrides for the two Hub repositories, so no destination is hardcoded at
#: a call site. An earlier attempt shipped pointing at a personal fork.
DATASTORE_REPO_ENV = 'EEE_DATASTORE_REPO_ID'
RAW_REPO_ENV = 'EEE_RAW_REPO_ID'
TOKEN_ENVS = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')


def _today() -> date:
    return datetime.now(UTC).date()


def _run_url() -> str | None:
    server = os.environ.get('GITHUB_SERVER_URL')
    repository = os.environ.get('GITHUB_REPOSITORY')
    run_id = os.environ.get('GITHUB_RUN_ID')
    if server and repository and run_id:
        return f'{server}/{repository}/actions/runs/{run_id}'
    return None


def _run_token() -> str:
    """Return a name for this run, unique within its date.

    Two runs of one adapter on one day are ordinary: a cancelled job, a source
    that was down at 03:17, a manual run after a fix. They each write a
    manifest and a report, so the snapshot directory has to tell them apart or
    the second silently replaces the first's account of what it fetched.

    Actions supplies the identity; outside it the clock does, which is enough
    to separate two runs a human started minutes apart.
    """
    run_id = _safe_component(os.environ.get('GITHUB_RUN_ID'))
    if run_id:
        attempt = _safe_component(os.environ.get('GITHUB_RUN_ATTEMPT')) or '1'
        return f'run-{run_id}-{attempt}'
    return f'local-{datetime.now(UTC).strftime("%H%M%S")}'


def _safe_component(value: str | None) -> str:
    """Return ``value`` reduced to what may appear in a repository path."""
    if not value:
        return ''
    return re.sub(r'[^A-Za-z0-9._-]', '', value)[:64]


def _have_token() -> bool:
    return any(os.environ.get(name) for name in TOKEN_ENVS)


def _step_summary(text: str) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(text.rstrip() + '\n')


def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for spec in catalog.ADAPTERS:
        rows.append(
            {
                'adapter': spec.key,
                'collections': list(spec.collections),
                'cadence': (
                    spec.cadence
                    if spec.weekday is None
                    else f'{spec.cadence} (weekday {spec.weekday})'
                ),
                'timeout_minutes': spec.timeout_minutes,
                'requires': list(spec.required_env),
                'packages': list(spec.with_packages),
                'runnable': spec.runnable,
                'reason': spec.unrunnable_reason,
            }
        )
    if args.output_format == 'json':
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(row['adapter']) for row in rows)
    for row in rows:
        state = 'ok' if row['runnable'] else f'skip: {row["reason"]}'
        print(
            f'{row["adapter"]:<{width}}  {row["cadence"]:<20} '
            f'{row["timeout_minutes"]:>3}m  {state}'
        )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Print the matrix of adapters due on a date.

    Adapters whose credentials are absent stay in the matrix on purpose: the
    run reports ``skipped_missing_credential``, which is visible, where
    dropping them from the plan would be silent.
    """
    run_date = args.date or _today()
    if args.adapter:
        try:
            spec = catalog.get(args.adapter)
        except catalog.UnknownAdapterError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not spec.runnable:
            print(
                f'{spec.key} is not schedulable: {spec.unrunnable_reason}',
                file=sys.stderr,
            )
            return 1
        due = (spec,)
    else:
        due = catalog.scheduled_for(run_date)
    include = [
        {
            'adapter': spec.key,
            'timeout_minutes': spec.timeout_minutes,
            # What the job gets, as opposed to what the adapter gets. The
            # workflow reads this one; the adapter's own budget is here so a
            # reader of the matrix can see both.
            'job_timeout_minutes': spec.job_timeout_minutes,
            'packages': ' '.join(spec.with_packages),
        }
        for spec in due
    ]
    print(json.dumps({'include': include}))
    return 0


def _resolve_state(
    raw_store: store.RawStore | None, adapter: str
) -> store.AdapterState:
    if raw_store is None:
        return store.AdapterState(adapter=adapter)
    return raw_store.read_state(adapter)


def _reconcile_pending(
    state: store.AdapterState,
    submitter: submit.DatastoreSubmitter | None,
) -> str | None:
    """Settle what the last pull request's fingerprints mean, before running.

    Runs commit straight to the datastore now and write no new pending
    fingerprints; this settles what the retired pull-request flow left
    behind. Its pending fingerprints describe records committed to a pull
    request, not records merged into the datastore. While that pull request
    is open they mean "already queued" and keep unchanged records from being
    uploaded twice. Once it is merged they are durable. Once it is closed
    without merging they mean nothing, and treating them as published would
    filter the same records out of every later run before publication is
    ever attempted, so the resubmission could never happen.

    Mutates ``state`` and returns a line for the run report, or ``None`` when
    there was nothing to settle. The mutation is persisted by the same
    end-of-run state commit as everything else, so a run that dies before
    then simply settles again next time.
    """
    if not state.pending_fingerprints:
        return None
    if submitter is None:
        # A dry run publishes nothing either way, so the pending records are
        # left queued rather than asking the Hub about their pull request.
        return None
    if state.pull_request_number is None:
        # Written only alongside a pull request number, so this is a state
        # file someone edited by hand. Fingerprints that can never be
        # promoted or requeued would recreate the buried-forever failure,
        # so they are requeued now, at the cost of a possible duplicate if
        # they really were on some open pull request.
        count = len(state.pending_fingerprints)
        state.pending_fingerprints.clear()
        return (
            f'{count} pending fingerprint(s) named no pull request to wait '
            'on; their records will be resubmitted'
        )
    number = state.pull_request_number
    status = submitter.pull_request_status(number)
    if status == 'open':
        return None
    count = len(state.pending_fingerprints)
    if status == 'merged':
        state.fingerprints |= state.pending_fingerprints
        state.pending_fingerprints.clear()
        state.pull_request_number = None
        state.pull_request_url = None
        return (
            f'pull request {number} was merged; {count} record(s) are now in '
            'the datastore ledger'
        )
    # Closed without merging: those records never reached the datastore, so
    # forgetting their fingerprints is what lets the next run publish them.
    state.pending_fingerprints.clear()
    state.pull_request_number = None
    state.pull_request_url = None
    return (
        f'pull request {number} was closed without merging; its {count} '
        'record(s) are forgotten from the ledger and will be resubmitted'
    )


def _record_paths(record: runner.StagedRecord) -> list[str]:
    """Return every datastore path one record consists of."""
    paths = [record.repo_path]
    if record.samples_repo_path:
        paths.append(record.samples_repo_path)
    return paths


def _landed_fingerprints(
    outcome: runner.RunOutcome, committed_paths: Sequence[str]
) -> list[str]:
    """Return fingerprints for records whose every file reached the Hub.

    A record with an instance sidecar spans two files. ``submit`` commits both
    in the same batch, so in practice they land together or not at all; this
    still checks every file rather than trusting that, because remembering a
    record on the strength of half of it would leave the other half
    permanently unpublished and invisible.
    """
    landed = set(committed_paths)
    return [
        record.fingerprint
        for record in outcome.uploaded
        if all(path in landed for path in _record_paths(record))
    ]


def _still_inflight(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    run_token: str,
    failure: submit.PartialSubmissionError | None,
) -> store.InflightBatch:
    """Return what the run's closing commit leaves in flight.

    Usually nothing: what landed is in the ledger, and what never reached
    the datastore is safe to upload again. The exception is a batch whose
    commit errored while the datastore was unreadable. Its records are
    neither recorded nor known to be absent, and clearing them would make
    the next run publish them a second time on top of copies that may
    already be there. They stay in flight, and the next run settles them the
    way it settles any interrupted run's.
    """
    unresolved = set(failure.unresolved_paths) if failure is not None else set()
    records = [
        {
            'fingerprint': record.fingerprint,
            'paths': _record_paths(record),
        }
        for record in outcome.uploaded
        if any(path in unresolved for path in _record_paths(record))
    ]
    if not records:
        return store.InflightBatch(adapter=spec.key)
    return store.InflightBatch(
        adapter=spec.key,
        run_date=outcome.run_date.isoformat(),
        run_token=run_token,
        destination=store.DIRECT_DESTINATION,
        records=records,
    )


def _reconcile_inflight(
    state: store.AdapterState,
    raw_store: store.RawStore | None,
    submitter: submit.DatastoreSubmitter | None,
) -> str | None:
    """Settle records a previous run published but never got to record.

    Publication is written down before it happens, because it is the one step
    that cannot be undone by repeating it. A run that uploaded records and
    then failed to commit its ledger leaves them in the datastore with
    nothing naming them, and the run after that would send the same
    evaluations again under fresh UUID paths.

    So the in-flight file is read back here and checked against where the
    records were headed: the datastore itself, or the pull request the
    retired flow was publishing into when it wrote the file. The ones that
    arrived are recorded; the ones that did not are forgotten, so this run
    uploads them. A question the Hub cannot answer stops the run, because
    the two wrong answers are losing records and duplicating them.

    Mutates ``state`` and returns a line for the run report, or ``None`` when
    there was nothing in flight. Running it twice settles the same batch the
    same way, so a run that dies before its own commit costs nothing.
    """
    if raw_store is None or submitter is None:
        return None
    batch = raw_store.read_inflight(state.adapter)
    if not batch.records:
        return None

    count = len(batch.records)
    if batch.destination == store.DIRECT_DESTINATION:
        present = submitter.paths_present(batch.paths)
        if present is None:
            raise submit.SubmissionError(
                f'could not read {submitter.repo_id} to settle {count} '
                'record(s) an earlier run published without recording them; '
                'a re-run would publish them a second time, so inspect the '
                'datastore first'
            )
        landed = [
            record['fingerprint']
            for record in batch.records
            if all(path in present for path in record['paths'])
        ]
        state.fingerprints.update(landed)
        return (
            f'an earlier run published {len(landed)} of {count} record(s) to '
            'the datastore without recording them; they are recorded now and '
            'the rest are published again'
        )

    # What follows settles an in-flight file the retired pull-request flow
    # wrote, against the pull request its records were headed for.
    pull_request = None
    if batch.pull_request_number is None:
        # A cold start: the upload itself was to open the pull request, so
        # whether one exists is the whole answer to whether it happened.
        pull_request = submitter.find_by_marker(state.adapter)
        if pull_request is None:
            return (
                f'{count} record(s) were staged for publication by an earlier '
                'run that opened no pull request; nothing reached the '
                'datastore, so they are published again'
            )
    else:
        number = batch.pull_request_number
        status = submitter.pull_request_status(number)
        if status == 'closed':
            return (
                f'pull request {number} was closed without merging while '
                f'{count} record(s) were in flight to it; they are forgotten '
                'and will be resubmitted'
            )
        if status == 'open':
            pull_request = submitter.resolve_known(state.adapter, number)
            if pull_request is None:
                raise submit.SubmissionError(
                    f'pull request {number} is open but no longer identifies '
                    f'itself as {state.adapter}, and {count} record(s) were '
                    'in flight to it; inspect it before re-running'
                )

    # Merged means the records that landed are in the datastore itself, so
    # that is where to look for them, and what is found is durable rather
    # than pending.
    revision = pull_request.revision if pull_request is not None else None
    present = submitter.paths_present(batch.paths, revision=revision)
    if present is None:
        raise submit.SubmissionError(
            f'could not read {revision or "the datastore"} to settle '
            f'{count} record(s) an earlier run published without recording '
            'them; a re-run would publish them a second time, so inspect the '
            'pull request first'
        )
    landed = [
        record['fingerprint']
        for record in batch.records
        if all(path in present for path in record['paths'])
    ]
    if pull_request is None:
        state.fingerprints.update(landed)
        where = 'the datastore'
    else:
        state.pull_request_number = pull_request.number
        state.pull_request_url = pull_request.url
        state.pending_fingerprints.update(landed)
        where = f'pull request {pull_request.number}'
    return (
        f'an earlier run published {len(landed)} of {count} record(s) to '
        f'{where} without recording them; they are recorded now and the rest '
        'are published again'
    )


def _publish(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    submitter: submit.DatastoreSubmitter,
    run_url: str | None,
    raw_reference: str,
    notes: list[str],
) -> submit.Submission | None:
    """Commit this run's records to the datastore."""
    operations = submit.upload_operations(outcome.upload_dir)
    if not operations:
        return None

    description = submit.commit_description(
        spec.key,
        coverage_line=outcome.coverage_line(),
        run_date=outcome.run_date.isoformat(),
        status=outcome.status,
        run_url=run_url,
        raw_reference=raw_reference,
        notes=notes,
    )
    return submitter.publish(
        operations=operations,
        description=description,
        message=(
            f'cron: {spec.key} {outcome.run_date.isoformat()} '
            f'({len(outcome.uploaded)} record(s))'
        ),
    )


def cmd_run(args: argparse.Namespace) -> int:
    try:
        spec = catalog.get(args.adapter)
    except catalog.UnknownAdapterError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not spec.runnable:
        print(
            f'{spec.key} is not schedulable: {spec.unrunnable_reason}',
            file=sys.stderr,
        )
        return 1

    run_date = args.date or _today()
    run_url = args.run_url or _run_url()
    run_token = _safe_component(args.run_id) or _run_token()
    dry_run = args.dry_run
    if not dry_run and not _have_token():
        # Falling back to a dry run here was convenient locally and wrong in
        # the workflow: a missing or expired secret produced a green job that
        # published nothing, which reads as "the leaderboard was unchanged"
        # for as long as nobody checks. Not publishing has to be asked for.
        print(
            'no Hugging Face token configured. Set HF_TOKEN to publish, or '
            'pass --dry-run to run without publishing.',
            file=sys.stderr,
        )
        return 1

    raw_store = None
    submitter = None
    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        raw_store = store.RawStore(api, repo_id=args.raw_repo)
        submitter = submit.DatastoreSubmitter(api, repo_id=args.datastore_repo)
        # Everything answerable without running the adapter is answered here.
        # All of it is caught later anyway, at the publish step, but by then
        # the adapter has scraped a leaderboard for forty-five minutes for
        # nothing. The token is checked before the raw store, because
        # ensure_private may create a repository and a token that cannot
        # publish should not be creating anything.
        submitter.ensure_writable()
        raw_store.ensure_private()

    state = _resolve_state(raw_store, spec.key)
    # In flight first: it can add fingerprints to the pending set and name the
    # pull request they are waiting on, which is what _reconcile_pending then
    # settles against that pull request's fate.
    inflight_note = _reconcile_inflight(state, raw_store, submitter)
    pending_note = _reconcile_pending(state, submitter)

    with contextlib.ExitStack() as stack:
        if args.workdir:
            workdir = Path(args.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            workdir = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=f'eee-cron-{spec.key}-')
                )
            )
        outcome = runner.run(
            spec,
            workdir,
            run_date=run_date,
            known_fingerprints=state.known_fingerprints,
            force_full=args.force_full,
            run_url=run_url,
        )
        for note in (inflight_note, pending_note):
            if note:
                outcome.messages.append(note)
        return _finish(
            outcome,
            spec=spec,
            state=state,
            raw_store=raw_store,
            submitter=submitter,
            run_url=run_url,
            run_token=run_token,
            dry_run=dry_run,
        )


def _finish(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    state: store.AdapterState,
    raw_store: store.RawStore | None,
    submitter: submit.DatastoreSubmitter | None,
    run_url: str | None,
    run_token: str,
    dry_run: bool,
) -> int:
    notes: list[str] = []
    raw_reference = store.raw_prefix(spec.key, outcome.run_date, run_token)
    committed_paths: tuple[str, ...] = ()
    publish_failure: submit.PartialSubmissionError | None = None
    raw_manifest: list[dict] = []
    parent_commit = state.parent_commit

    if raw_store is not None:
        # The snapshot and the intention to publish, before the publishing.
        # Uploading records and then failing to write the ledger leaves them
        # in the datastore with nothing naming them, and the next run
        # sends the same evaluations again under fresh UUID paths. What this
        # commit writes is enough for that next run to find them instead.
        previous = (
            raw_store.read_manifest(state.last_raw_prefix)
            if state.last_raw_prefix
            else []
        )
        operations, raw_manifest = store.plan_raw_upload(
            outcome.raw_dir,
            prefix=raw_reference,
            previous_manifest=previous,
            previous_prefix=state.last_raw_prefix,
        )
        operations.append(
            store.inflight_operation(
                store.InflightBatch(
                    adapter=spec.key,
                    run_date=outcome.run_date.isoformat(),
                    run_token=run_token,
                    destination=store.DIRECT_DESTINATION,
                    records=(
                        []
                        if dry_run
                        else [
                            {
                                'fingerprint': record.fingerprint,
                                'paths': _record_paths(record),
                            }
                            for record in outcome.uploaded
                        ]
                    ),
                )
            )
        )
        result = raw_store.commit(
            operations,
            message=(
                f'cron: {spec.key} {outcome.run_date.isoformat()} (snapshot)'
            ),
            parent_commit=parent_commit,
        )
        parent_commit = getattr(result, 'oid', None) or parent_commit

    if dry_run:
        notes.append('dry run: nothing was published')
    elif outcome.has_upload and submitter is not None:
        try:
            submission = _publish(
                outcome,
                spec=spec,
                submitter=submitter,
                run_url=run_url,
                raw_reference=raw_reference,
                notes=notes,
            )
        except submit.PartialSubmissionError as exc:
            # Some batches landed. Fall through so the fingerprints that
            # reached the datastore are still written to the ledger, then
            # fail the job. Losing them would republish those records under
            # fresh paths on the next run.
            publish_failure = exc
            committed_paths = exc.committed_paths
            notes.append(str(exc))
        else:
            if submission is not None:
                committed_paths = submission.committed_paths

    landed = _landed_fingerprints(outcome, committed_paths)

    report = outcome.to_manifest()
    report['raw_reference'] = raw_reference
    report['run_token'] = run_token
    report['dry_run'] = dry_run
    report['records_committed'] = len(landed)
    if publish_failure is not None:
        report['publish_error'] = str(publish_failure)

    if raw_store is not None:
        operations = [store.run_report_operation(report, prefix=raw_reference)]

        state.last_run_date = outcome.run_date.isoformat()
        state.last_status = outcome.status
        # Only advance the snapshot pointer when a snapshot was actually
        # written. Pointing it at a directory holding nothing but a run report
        # would make the next run find no manifest and re-upload everything.
        if raw_manifest:
            state.last_raw_date = outcome.run_date.isoformat()
            state.last_raw_prefix = raw_reference
        # Only fingerprints that actually reached the datastore are
        # remembered, so a failed upload is retried rather than forgotten,
        # and a batch that landed before a later one failed is not
        # published a second time. They land on the default branch, so they
        # are durable the moment they land; there is no reviewer left to
        # close them out.
        state.fingerprints.update(landed)
        operations.extend(store.state_operations(state))
        # Nothing is in flight any more: either it is in the ledger above or
        # it never reached the datastore. Written empty rather than
        # deleted, so every run makes the same commit. The exception is a
        # batch whose commit errored while the datastore was unreadable:
        # those records may have landed anyway, so they stay in flight for
        # the next run to settle instead of being uploaded again on top of
        # copies that may already be there.
        operations.append(
            store.inflight_operation(
                _still_inflight(
                    outcome,
                    spec=spec,
                    run_token=run_token,
                    failure=publish_failure,
                )
            )
        )
        raw_store.commit(
            operations,
            message=(
                f'cron: {spec.key} {outcome.run_date.isoformat()} '
                f'({outcome.status})'
            ),
            parent_commit=parent_commit,
        )

    _report(outcome, spec=spec, published=len(landed), dry_run=dry_run)
    if publish_failure is not None:
        print(str(publish_failure), file=sys.stderr)
        return 1
    return 0 if outcome.ok else 1


def _report(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    published: int,
    dry_run: bool,
) -> None:
    lines = [
        f'### `{spec.key}`: {outcome.status}',
        '',
        f'- Coverage: {outcome.coverage_line()}',
    ]
    if published:
        lines.append(
            f'- Published: {published} record(s) committed to the datastore'
        )
    elif outcome.has_upload and dry_run:
        lines.append('- Published: nothing (dry run)')
    elif not outcome.has_upload and outcome.ok:
        lines.append('- Published: nothing, unchanged since the last run')
    for message in outcome.messages:
        lines.append(f'- {message.splitlines()[0]}')

    text = '\n'.join(lines)
    print(text)
    for message in outcome.messages:
        if '\n' in message:
            print(message)
    _step_summary(text + '\n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever.cron',
        description='Scheduled adapter ingestion for the EEE datastore.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser(
        'list', help='Show every registered adapter and its schedule.'
    )
    list_parser.add_argument(
        '--format',
        dest='output_format',
        choices=['text', 'json'],
        default='text',
    )
    list_parser.set_defaults(handler=cmd_list)

    plan_parser = subparsers.add_parser(
        'plan', help='Print the job matrix for a date, as JSON.'
    )
    plan_parser.add_argument('--date', type=date.fromisoformat, default=None)
    plan_parser.add_argument(
        '--adapter',
        default=None,
        help='Plan only this adapter, whatever its cadence says.',
    )
    plan_parser.set_defaults(handler=cmd_plan)

    run_parser = subparsers.add_parser(
        'run', help='Run one adapter and submit what it produced.'
    )
    run_parser.add_argument('--adapter', required=True)
    run_parser.add_argument('--date', type=date.fromisoformat, default=None)
    run_parser.add_argument(
        '--workdir',
        default=None,
        help='Keep staging output here instead of a temporary directory.',
    )
    run_parser.add_argument(
        '--dry-run',
        action='store_true',
        help=(
            'Run and validate without touching the Hub. Implied when no '
            'Hugging Face token is configured.'
        ),
    )
    run_parser.add_argument(
        '--force-full',
        action='store_true',
        help='Publish every record, ignoring the de-duplication ledger.',
    )
    run_parser.add_argument(
        '--datastore-repo',
        default=(
            os.environ.get(DATASTORE_REPO_ENV) or submit.DEFAULT_DATASTORE_REPO
        ),
        help=f'Datastore dataset repo (env: {DATASTORE_REPO_ENV}).',
    )
    run_parser.add_argument(
        '--raw-repo',
        default=os.environ.get(RAW_REPO_ENV) or store.DEFAULT_RAW_REPO,
        help=f'Raw snapshot and state dataset repo (env: {RAW_REPO_ENV}).',
    )
    run_parser.add_argument(
        '--run-url',
        default=None,
        help=(
            'Link recorded on every published record and in the datastore '
            'commit descriptions.'
        ),
    )
    run_parser.add_argument(
        '--run-id',
        default=None,
        help=(
            "Name this run's snapshot directory (default: the workflow run, "
            'or the local clock).'
        ),
    )
    run_parser.set_defaults(handler=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (store.StoreError, submit.SubmissionError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        _step_summary(f'### failed\n\n```\n{exc}\n```\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
