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
    fingerprints = []
    for record in outcome.uploaded:
        paths = [record.repo_path]
        if record.samples_repo_path:
            paths.append(record.samples_repo_path)
        if all(path in landed for path in paths):
            fingerprints.append(record.fingerprint)
    return fingerprints


def _publish(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    state: store.AdapterState,
    submitter: submit.DatastoreSubmitter,
    run_url: str | None,
    raw_reference: str,
    notes: list[str],
) -> submit.Submission | None:
    """Put this run's records into the adapter's own pull request."""
    operations = submit.upload_operations(outcome.upload_dir)
    if not operations:
        return None

    description = submit.pull_request_description(
        spec.key,
        coverage_line=outcome.coverage_line(),
        run_date=outcome.run_date.isoformat(),
        status=outcome.status,
        run_url=run_url,
        raw_reference=raw_reference,
        notes=notes,
    )

    pull_request = None
    if state.pull_request_number is not None:
        pull_request = submitter.resolve_known(
            spec.key, state.pull_request_number
        )
    if pull_request is None:
        pull_request = submitter.find_by_marker(spec.key)

    return submitter.publish(
        spec.key,
        pull_request=pull_request,
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
        # Before the adapter runs, not after: an hour of scraping is a poor
        # way to discover that its snapshot has nowhere private to go.
        raw_store.ensure_private()
        submitter = submit.DatastoreSubmitter(api, repo_id=args.datastore_repo)

    state = _resolve_state(raw_store, spec.key)

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
            known_fingerprints=state.fingerprints,
            force_full=args.force_full,
            run_url=run_url,
        )
        return _finish(
            outcome,
            spec=spec,
            state=state,
            raw_store=raw_store,
            submitter=submitter,
            run_url=run_url,
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
    dry_run: bool,
) -> int:
    notes: list[str] = []
    raw_reference = store.raw_prefix(spec.key, outcome.run_date)
    pull_request = None
    committed_paths: tuple[str, ...] = ()
    publish_failure: submit.PartialSubmissionError | None = None

    if dry_run:
        notes.append('dry run: nothing was published')
    elif outcome.has_upload and submitter is not None:
        try:
            submission = _publish(
                outcome,
                spec=spec,
                state=state,
                submitter=submitter,
                run_url=run_url,
                raw_reference=raw_reference,
                notes=notes,
            )
        except submit.PartialSubmissionError as exc:
            # Some batches landed. Fall through so the pull request number
            # and the fingerprints that reached it are still written to the
            # ledger, then fail the job. Losing them would republish those
            # records under fresh paths on the next run.
            publish_failure = exc
            pull_request = exc.pull_request
            committed_paths = exc.committed_paths
            notes.append(str(exc))
        else:
            if submission is not None:
                pull_request = submission.pull_request
                committed_paths = submission.committed_paths

    report = outcome.to_manifest()
    report['raw_reference'] = raw_reference
    report['dry_run'] = dry_run
    if publish_failure is not None:
        report['publish_error'] = str(publish_failure)
        report['records_committed'] = len(committed_paths)
    if pull_request is not None:
        report['pull_request'] = {
            'number': pull_request.number,
            'url': pull_request.url,
        }

    if raw_store is not None:
        previous = (
            raw_store.read_manifest(
                spec.key, date.fromisoformat(state.last_raw_date)
            )
            if state.last_raw_date
            else []
        )
        operations, raw_manifest = store.plan_raw_upload(
            outcome.raw_dir,
            adapter=spec.key,
            run_date=outcome.run_date,
            previous_manifest=previous,
            previous_date=state.last_raw_date,
        )
        operations.append(
            store.run_report_operation(
                report, adapter=spec.key, run_date=outcome.run_date
            )
        )

        state.last_run_date = outcome.run_date.isoformat()
        state.last_status = outcome.status
        # Only advance the snapshot pointer when a snapshot was actually
        # written. Pointing it at a date holding nothing but a run report
        # would make the next run find no manifest and re-upload everything.
        if raw_manifest:
            state.last_raw_date = outcome.run_date.isoformat()
        if pull_request is not None:
            state.pull_request_number = pull_request.number
            state.pull_request_url = pull_request.url
            # Only fingerprints that actually reached the datastore are
            # remembered, so a failed upload is retried rather than
            # forgotten, and a batch that landed before a later one failed
            # is not published a second time.
            state.fingerprints.update(
                _landed_fingerprints(outcome, committed_paths)
            )
        operations.extend(store.state_operations(state))
        raw_store.commit(
            operations,
            message=(
                f'cron: {spec.key} {outcome.run_date.isoformat()} '
                f'({outcome.status})'
            ),
            parent_commit=state.parent_commit,
        )

    _report(outcome, spec=spec, pull_request=pull_request, dry_run=dry_run)
    if publish_failure is not None:
        print(str(publish_failure), file=sys.stderr)
        return 1
    return 0 if outcome.ok else 1


def _report(
    outcome: runner.RunOutcome,
    *,
    spec: catalog.AdapterSpec,
    pull_request: submit.PullRequest | None,
    dry_run: bool,
) -> None:
    lines = [
        f'### `{spec.key}`: {outcome.status}',
        '',
        f'- Coverage: {outcome.coverage_line()}',
    ]
    if pull_request is not None:
        lines.append(f'- Pull request: {pull_request.url}')
    elif outcome.has_upload and dry_run:
        lines.append('- Pull request: not opened (dry run)')
    elif not outcome.has_upload and outcome.ok:
        lines.append('- Pull request: unchanged, nothing to submit')
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
        help='Link recorded on every published record and in the PR body.',
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
