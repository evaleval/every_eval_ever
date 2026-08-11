"""Run one adapter end to end for the daily refresh.

    uv run python -m every_eval_ever.cron run vals_ai --dry-run

One invocation handles one adapter, which is what makes the schedule one
workflow job and one datastore pull request per adapter. The stages are:

1. run the adapter into a scratch tree;
2. store what it fetched permanently, in the private raw dataset;
3. stop if the source has not moved since the fingerprint the ledger recorded;
4. stamp every record as cron-produced;
5. validate through the same CLI the datastore's pull request bot runs;
6. commit the records to the adapter's pull request.

Exit codes: ``0`` published, ``2`` nothing new to publish, ``1`` failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from every_eval_ever.cron import archive as archive_module
from every_eval_ever.cron import preflight as preflight_module
from every_eval_ever.cron import publish as publish_module
from every_eval_ever.cron.fingerprint import (
    output_fingerprint,
    read_fingerprint,
    write_fingerprint,
)
from every_eval_ever.cron.schedule import (
    CronAdapter,
    RawPolicy,
    get_adapter,
    scheduled_adapters,
)
from every_eval_ever.cron.stamp import stamp_tree
from every_eval_ever.helpers import raw_capture

EXIT_PUBLISHED = 0
EXIT_FAILED = 1
EXIT_NOTHING_NEW = 2

_logger = logging.getLogger('every_eval_ever.cron')


@dataclass
class RunSummary:
    """The record of one adapter refresh, written to disk for the artifact."""

    adapter: str
    run_date: str
    status: str = 'pending'
    detail: str = ''
    invocations: int = 0
    #: Invocations that exited non-zero. Their valid records are still kept and
    #: published; the reason each failed is in the run's adapter_reports/.
    failed_invocations: list[dict[str, object]] = field(default_factory=list)
    records: int = 0
    raw_payloads: int = 0
    raw_bytes: int = 0
    #: Payloads fetched but not stored, e.g. over the capture ceiling. They are
    #: still recorded in the ledger so the gap is visible rather than silent.
    raw_skipped: int = 0
    raw_policy: str = ''
    raw_archive: dict[str, object] = field(default_factory=dict)
    raw_fingerprint: str | None = None
    output_fingerprint: str | None = None
    previous_fingerprint: str | None = None
    fingerprint_source: str | None = None
    unknown_inferred_fields: dict[str, int] = field(default_factory=dict)
    collections: list[str] = field(default_factory=list)
    validation: dict[str, int] = field(default_factory=dict)
    pr_url: str | None = None
    pr_reused: bool | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )


def utc_run_date() -> str:
    """Return today's UTC date, the value stamped onto every record."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


@dataclass
class Invocation:
    """One adapter invocation and how it ended."""

    arguments: list[str]
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class AdapterOutcome:
    """Every invocation this refresh made for one adapter."""

    invocations: list[Invocation] = field(default_factory=list)

    @property
    def failed(self) -> list[Invocation]:
        return [item for item in self.invocations if not item.ok]

    @property
    def all_failed(self) -> bool:
        return bool(self.invocations) and not any(
            item.ok for item in self.invocations
        )


def run_adapter(
    adapter: CronAdapter,
    *,
    work_dir: Path,
    raw_dir: Path,
    environment: dict[str, str],
) -> AdapterOutcome:
    """Invoke the adapter once per configured run, and keep going on failure.

    Adapters default their output to a relative ``data/`` path, so the scratch
    tree is the working directory rather than a flag: the ones that do accept
    ``--output-dir`` disagree about whether it means the base or the collection
    directory.

    A non-zero exit is not treated as fatal here. An adapter that hits a source
    row it cannot represent writes every valid record, writes a report under
    ``adapter_reports/``, and *then* exits non-zero — so a run is reported,
    never discarded, and one bad leaderboard does not cost the others their
    records.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    child_environment = {
        **environment,
        raw_capture.RAW_CAPTURE_DIR_ENV: str(raw_dir),
        'PYTHONUNBUFFERED': '1',
    }
    outcome = AdapterOutcome()
    for run in adapter.runs:
        argv = [sys.executable, *adapter.argv_for(run, raw_dir)]
        _logger.info('running %s', ' '.join(argv[1:]))
        completed = subprocess.run(
            argv,
            cwd=work_dir,
            env=child_environment,
            check=False,
        )
        outcome.invocations.append(
            Invocation(arguments=list(run), returncode=completed.returncode)
        )
        if completed.returncode != 0:
            _logger.warning(
                '%s exited %d for %s; continuing with the remaining runs',
                adapter.name,
                completed.returncode,
                list(run) or '[no arguments]',
            )
    return outcome


def validate_records(data_root: Path) -> dict[str, int]:
    """Validate generated records with the CLI the datastore's bot uses.

    Raises:
        RuntimeError: if any record fails, or if warnings would block a merge.
    """
    pattern = str(data_root / '*' / '*' / '*' / '*.json*')
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'every_eval_ever',
            'validate',
            '--format',
            'json',
            pattern,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            'could not read validator output '
            f'(exit {completed.returncode}): {completed.stderr.strip()}'
        ) from error

    # The package CLI reports a bare list of file reports.
    reports = (
        payload if isinstance(payload, list) else payload.get('reports', [])
    )
    errors = sum(len(report.get('errors', [])) for report in reports)
    warnings = sum(len(report.get('warnings', [])) for report in reports)
    counts = {'files': len(reports), 'errors': errors, 'warnings': warnings}
    if completed.returncode != 0:
        # Exit 2 is warning-only, which does not make a record invalid but does
        # block a merge, so it is not something to publish and walk away from.
        raise RuntimeError(
            f'validation failed with {errors} error(s) and {warnings} '
            'warning(s); records were not published'
        )
    return counts


def collections_in(data_root: Path) -> list[str]:
    """Return the datastore collections a run wrote to."""
    if not data_root.is_dir():
        return []
    return sorted(entry.name for entry in data_root.iterdir() if entry.is_dir())


def _previous_fingerprint(
    adapter: str,
    *,
    fingerprint_path: Path | None,
    raw_repo_id: str,
    consult_ledger: bool,
    token: str | None,
) -> str | None:
    """Return the fingerprint to compare this run against.

    A local ``--fingerprint`` file wins when it has a value, which is what makes
    a local run reproducible. Otherwise the raw dataset's ledger is the durable
    memory: it survives the build cache, so an adapter cannot forget last night's
    state and republish its whole set.
    """
    stored = read_fingerprint(fingerprint_path) if fingerprint_path else None
    if stored:
        return stored
    if not consult_ledger:
        return None
    return archive_module.last_gating_fingerprint(
        adapter, repo_id=raw_repo_id, token=token
    )


def _archive_raw(
    raw_dir: Path,
    *,
    summary: RunSummary,
    captured: int,
    run_id: str,
    run_url: str | None,
    gating_fingerprint: str | None,
    raw_repo_id: str,
    archive_raw: bool,
    dry_run: bool,
    token: str | None,
) -> dict[str, object]:
    """Store this run's raw payloads permanently, or say why it did not.

    ``captured`` counts every manifest entry, not just the ones that landed on
    disk: a payload too large to store still gets a ledger row, so a run that
    only fetched an oversized payload must still be archived.
    """
    if not captured:
        return {
            'status': 'nothing_captured',
            'reason': (
                'the adapter archives no raw data; see its raw policy'
                if summary.raw_policy
                else ''
            ),
        }
    if not archive_raw:
        return {
            'status': 'disabled',
            'payloads': summary.raw_payloads,
            'skipped_payloads': summary.raw_skipped,
        }
    if dry_run:
        return {
            'status': 'skipped_dry_run',
            'repo_id': raw_repo_id,
            'payloads': summary.raw_payloads,
            'skipped_payloads': summary.raw_skipped,
        }

    result = archive_module.archive(
        raw_dir,
        adapter=summary.adapter,
        run_date=summary.run_date,
        run_id=run_id,
        run_url=run_url,
        raw_fingerprint=summary.raw_fingerprint,
        output_fingerprint=summary.output_fingerprint,
        gating_fingerprint=gating_fingerprint,
        repo_id=raw_repo_id,
        token=token,
    )
    return {
        'status': 'archived',
        'repo_id': result.repo_id,
        'ledger_path': result.ledger_path,
        'uploaded': result.uploaded,
        'reused': result.reused,
        'uploaded_bytes': result.uploaded_bytes,
        'ledger_rows': len(result.rows),
    }


def refresh(
    name: str,
    *,
    work_dir: Path,
    fingerprint_path: Path | None,
    summary_path: Path,
    repo_id: str,
    run_url: str | None,
    dry_run: bool,
    force: bool,
    run_id: str = 'local',
    raw_repo_id: str = archive_module.DEFAULT_RAW_REPO_ID,
    archive_raw: bool = True,
    environment: dict[str, str] | None = None,
) -> tuple[int, RunSummary]:
    """Refresh one adapter. Returns its exit code and summary."""
    environment = dict(os.environ if environment is None else environment)
    adapter = get_adapter(name)
    run_date = utc_run_date()
    summary = RunSummary(
        adapter=adapter.name,
        run_date=run_date,
        raw_policy=adapter.raw_policy.value,
    )

    missing = adapter.missing_env(environment)
    if missing:
        summary.status = 'skipped'
        summary.detail = f'missing environment: {", ".join(missing)}'
        summary.write(summary_path)
        _logger.warning('%s: %s', adapter.name, summary.detail)
        return EXIT_NOTHING_NEW, summary

    data_root = work_dir / 'data'
    raw_dir = work_dir / 'raw'

    outcome = run_adapter(
        adapter,
        work_dir=work_dir,
        raw_dir=raw_dir,
        environment=environment,
    )
    summary.invocations = len(outcome.invocations)
    summary.failed_invocations = [
        {
            'arguments': ' '.join(item.arguments) or '(no arguments)',
            'returncode': item.returncode,
        }
        for item in outcome.failed
    ]

    raw_capture.index_unlisted_payloads(raw_dir)
    manifest = raw_capture.read_manifest(raw_dir)
    stored = [entry for entry in manifest if entry.get('file')]
    summary.raw_payloads = len(stored)
    summary.raw_bytes = sum(entry.get('bytes') or 0 for entry in stored)
    # Fetched but deliberately not stored, e.g. over the capture ceiling. These
    # still belong in the ledger, so they must not read as "nothing captured".
    summary.raw_skipped = len(manifest) - len(stored)
    # Only verbatim wire captures can say whether the source moved; a dump an
    # adapter wrote itself is archived but may carry its own fetch timestamp.
    summary.raw_fingerprint = raw_capture.fingerprint(
        raw_dir, verbatim_only=True
    )

    if (
        adapter.raw_policy
        in {RawPolicy.VIA_FETCH_HELPERS, RawPolicy.VIA_ADAPTER_FLAG}
        and not manifest
    ):
        # Declared as archived but nothing was captured at all: report it rather
        # than let the run look like it saved raw data.
        _logger.warning(
            '%s declares raw policy %s but captured no payloads',
            adapter.name,
            adapter.raw_policy.value,
        )
    if summary.raw_skipped:
        _logger.warning(
            '%s fetched %d payload(s) that were not stored; the ledger records '
            'them but the bytes are gone',
            adapter.name,
            summary.raw_skipped,
        )

    summary.records = len(list(data_root.glob('*/*/*/*.json')))
    summary.collections = collections_in(data_root)

    # Computed before stamping deliberately: the digest strips the cron stamp,
    # so the value is the same either side of it, and archiving it here means
    # the ledger row carries the fingerprint even for a run that fails later.
    summary.output_fingerprint = output_fingerprint(data_root)
    summary.fingerprint_source = 'raw' if summary.raw_fingerprint else 'output'
    current = summary.raw_fingerprint or summary.output_fingerprint

    # Archive before anything is published, and on every path that captured
    # something — including a run that produced no records, whose payload is
    # often the evidence for why. Records must never reach the datastore without
    # their raw provenance stored somewhere permanent, so a failure here is
    # fatal rather than a warning.
    summary.raw_archive = _archive_raw(
        raw_dir,
        summary=summary,
        captured=len(manifest),
        run_id=run_id,
        run_url=run_url,
        gating_fingerprint=current,
        raw_repo_id=raw_repo_id,
        archive_raw=archive_raw,
        dry_run=dry_run,
        token=environment.get('HF_TOKEN'),
    )

    if not summary.records:
        # No records and a failing adapter is a broken source; no records and a
        # clean exit just means the source had nothing to give today.
        if outcome.failed:
            summary.status = 'failed'
            summary.detail = (
                'the adapter produced no records and '
                f'{len(outcome.failed)} of {summary.invocations} invocation(s) '
                'failed'
            )
            summary.write(summary_path)
            _logger.error('%s: %s', adapter.name, summary.detail)
            return EXIT_FAILED, summary
        summary.status = 'nothing_produced'
        summary.detail = 'the adapter produced no records'
        summary.write(summary_path)
        return EXIT_NOTHING_NEW, summary

    stamped = stamp_tree(
        data_root,
        adapter=adapter.name,
        run_date=run_date,
        run_url=run_url,
    )
    summary.unknown_inferred_fields = stamped.unknown_inferred

    previous = _previous_fingerprint(
        adapter.name,
        fingerprint_path=fingerprint_path,
        raw_repo_id=raw_repo_id,
        consult_ledger=archive_raw and not dry_run,
        token=environment.get('HF_TOKEN'),
    )
    summary.previous_fingerprint = previous

    if previous and current == previous and not force:
        summary.status = 'unchanged'
        summary.detail = (
            f'{summary.fingerprint_source} fingerprint matches the previous '
            'run; nothing published'
        )
        summary.write(summary_path)
        _logger.info('%s: %s', adapter.name, summary.detail)
        return EXIT_NOTHING_NEW, summary

    summary.validation = validate_records(data_root)

    if dry_run:
        preview = publish_module.plan(
            data_root, adapter=adapter.name, repo_id=repo_id
        )
        summary.status = 'dry_run'
        summary.detail = (
            f'would send {len(preview.files)} file(s) to {repo_id} as '
            f'{preview.title!r}'
        )
        summary.write(summary_path)
        _logger.info('%s: %s', adapter.name, summary.detail)
        return EXIT_PUBLISHED, summary

    if summary.failed_invocations:
        _logger.warning(
            '%s: publishing %d record(s) from a partial refresh; %d '
            'invocation(s) failed',
            adapter.name,
            summary.records,
            len(summary.failed_invocations),
        )

    result = publish_module.publish(
        data_root,
        adapter=adapter.name,
        repo_id=repo_id,
        token=environment.get('HF_TOKEN'),
        commit_description=_commit_description(summary, run_url),
    )
    summary.status = (
        'published_partial' if summary.failed_invocations else 'published'
    )
    summary.pr_url = result.pr_url
    summary.pr_reused = result.reused_existing_pr
    summary.detail = (
        f'{result.files} file(s) in {result.commits} commit(s) to '
        f'{result.pr_url}'
    )
    if fingerprint_path and current:
        write_fingerprint(fingerprint_path, current)
    summary.write(summary_path)
    _logger.info('%s: %s', adapter.name, summary.detail)
    return EXIT_PUBLISHED, summary


def _raw_archive_lines(summary: RunSummary) -> list[str]:
    """Point a reviewer at the permanently stored raw data for this run."""
    archive = summary.raw_archive
    if archive.get('status') != 'archived':
        return []
    return [
        f'- Raw data kept in `{archive["repo_id"]}` (private): '
        f'{archive["uploaded"]} new payload(s), {archive["reused"]} already '
        f'stored. Ledger row: `{archive["ledger_path"]}`.'
    ]


def _partial_refresh_lines(summary: RunSummary) -> list[str]:
    """Say up front that a refresh was partial, and which part is missing."""
    if not summary.failed_invocations:
        return []
    failures = '; '.join(
        f'`{item["arguments"]}` exited {item["returncode"]}'
        for item in summary.failed_invocations
    )
    return [
        f'- **Partial refresh**: {len(summary.failed_invocations)} of '
        f'{summary.invocations} invocation(s) failed ({failures}). The records '
        'here are the ones that converted; the rest of this source is not in '
        'this update. Reasons are in the run artifact under '
        '`adapter_reports/`.'
    ]


def _commit_description(summary: RunSummary, run_url: str | None) -> str:
    """Describe the run in the commit, so a reviewer can trace it."""
    lines = [
        f'Automated refresh of the `{summary.adapter}` adapter.',
        '',
        f'- Run date (UTC): {summary.run_date}',
        f'- Records: {summary.records} across {len(summary.collections)} '
        f'collection(s): {", ".join(summary.collections)}',
        f'- Adapter invocations: {summary.invocations}',
        f'- Raw payloads archived: {summary.raw_payloads} '
        f'({summary.raw_policy})',
        *_raw_archive_lines(summary),
        *_partial_refresh_lines(summary),
        f'- Every record carries `type_of_addition: cron` and '
        f'`cron_run_date: {summary.run_date}` in '
        '`source_metadata.additional_details`.',
    ]
    if summary.unknown_inferred_fields:
        unknown = ', '.join(
            f'{name} ({count})'
            for name, count in sorted(summary.unknown_inferred_fields.items())
        )
        lines.append(
            f'- Inferred axes still `unknown`: {unknown}. The source does not '
            'state these; each affected record names them in '
            '`cron_unknown_inferred_fields`.'
        )
    if run_url:
        lines.append(f'- Workflow run (raw data artifact): {run_url}')
    return '\n'.join(lines)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('adapter', help='Adapter to refresh')
    parser.add_argument(
        '--work-dir',
        type=Path,
        required=True,
        help='Scratch directory for generated records and raw payloads',
    )
    parser.add_argument(
        '--fingerprint',
        type=Path,
        help=(
            'File holding the previous run fingerprint. Updated on a '
            'successful publish; a missing file means the run always publishes.'
        ),
    )
    parser.add_argument(
        '--summary',
        type=Path,
        help='Where to write the run summary JSON (default: <work-dir>/summary.json)',
    )
    parser.add_argument(
        '--repo-id',
        default=publish_module.DEFAULT_REPO_ID,
        help=f'Datastore repo (default: {publish_module.DEFAULT_REPO_ID})',
    )
    parser.add_argument(
        '--raw-repo-id',
        default=archive_module.DEFAULT_RAW_REPO_ID,
        help=(
            'Private dataset that permanently holds raw payloads and the '
            f'ledger (default: {archive_module.DEFAULT_RAW_REPO_ID})'
        ),
    )
    parser.add_argument(
        '--no-archive-raw',
        dest='archive_raw',
        action='store_false',
        help=(
            'Do not store raw payloads permanently. They stay in --work-dir '
            'only, so use this for local runs, not for the schedule.'
        ),
    )
    parser.add_argument(
        '--run-id',
        default='local',
        help='Identifier for this run, used in the ledger path',
    )
    parser.add_argument(
        '--run-url',
        default=os.environ.get('EEE_CRON_RUN_URL'),
        help='Workflow run URL recorded on each record and in the commit',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Do everything except commit to the datastore',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Publish even when the fingerprint matches the previous run',
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever.cron',
        description='Daily adapter refresh for the EEE datastore.',
    )
    subcommands = parser.add_subparsers(dest='command', required=True)
    run_parser = subcommands.add_parser('run', help='Refresh one adapter')
    _add_run_arguments(run_parser)
    subcommands.add_parser(
        'list', help='Print the adapters the schedule would run, as JSON'
    )
    preflight_parser = subcommands.add_parser(
        'preflight',
        help='Check credentials and destinations before refreshing anything',
    )
    preflight_parser.add_argument(
        '--repo-id', default=publish_module.DEFAULT_REPO_ID
    )
    preflight_parser.add_argument(
        '--raw-repo-id', default=archive_module.DEFAULT_RAW_REPO_ID
    )
    preflight_parser.add_argument(
        '--no-create-raw',
        dest='create_raw',
        action='store_false',
        help='Report a missing raw dataset instead of creating it',
    )
    preflight_parser.add_argument(
        '--markdown',
        type=Path,
        help='Append a checklist here as well (e.g. $GITHUB_STEP_SUMMARY)',
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(name)s: %(message)s',
    )

    if args.command == 'list':
        runnable, skipped = scheduled_adapters(dict(os.environ))
        print(
            json.dumps(
                {
                    'adapters': [adapter.name for adapter in runnable],
                    'skipped': {
                        adapter.name: reason for adapter, reason in skipped
                    },
                },
                indent=2,
            )
        )
        return 0

    if args.command == 'preflight':
        checks = preflight_module.run_preflight(
            environment=dict(os.environ),
            repo_id=args.repo_id,
            raw_repo_id=args.raw_repo_id,
            create_raw=args.create_raw,
        )
        print(preflight_module.render(checks))
        if args.markdown:
            preflight_module.write_markdown(checks, args.markdown)
        return EXIT_FAILED if preflight_module.failed(checks) else 0

    summary_path = args.summary or (args.work_dir / 'summary.json')
    try:
        code, _ = refresh(
            args.adapter,
            work_dir=args.work_dir,
            fingerprint_path=args.fingerprint,
            summary_path=summary_path,
            repo_id=args.repo_id,
            run_url=args.run_url,
            dry_run=args.dry_run,
            force=args.force,
            run_id=args.run_id,
            raw_repo_id=args.raw_repo_id,
            archive_raw=args.archive_raw,
        )
        return code
    except Exception as error:
        _logger.error('%s failed: %s', args.adapter, error)
        RunSummary(
            adapter=args.adapter,
            run_date=utc_run_date(),
            status='failed',
            detail=str(error),
        ).write(summary_path)
        return EXIT_FAILED


if __name__ == '__main__':
    raise SystemExit(main())
