"""Run one adapter and decide, on evidence, what may be published.

Everything here exists to make a scheduled run's outcome honest:

- each run stages into its own temporary tree, so one adapter's failure can
  never end up inside another adapter's pull request;
- nothing is published that the real validator has not passed, and "the
  adapter wrote nothing" is a failure rather than a clean run;
- a partial conversion is its own outcome, distinct from both success and
  crash, because several adapters legitimately drop rows every day;
- records are fingerprinted *before* they are stamped, so re-publishing an
  unchanged leaderboard is detectable at all.

The runner never uploads. It produces a directory that is ready to upload and
a report of what happened; ``store`` and ``submit`` do the rest.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from every_eval_ever.adapters.registry import ADAPTERS as _ALL_ADAPTERS
from every_eval_ever.adapters.registry import AdapterSpec
from every_eval_ever.cron.provenance import stamp_cron_provenance
from every_eval_ever.helpers.raw_capture import CAPTURE_DIR_ENV
from every_eval_ever.validator.check_duplicate_entries import normalized_hash
from every_eval_ever.validator.json_utils import strict_json_loads

Status = Literal[
    'completed',
    'partial',
    'skipped_missing_credential',
    'skipped_missing_dependency',
    'failed',
]

#: Credentials any adapter might need. A run is given only its own.
ALL_CREDENTIAL_ENV = frozenset(
    name for spec in _ALL_ADAPTERS for name in spec.required_env
)

_UUID = r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
_AGGREGATE_NAME = re.compile(rf'^{_UUID}\.json$')
_SAMPLES_NAME = re.compile(rf'^{_UUID}_samples\.jsonl$')

#: Directories an adapter is allowed to create in its staging tree.
#: ``adapter_reports/`` is where ``default_failure_report_path`` puts
#: provenance, deliberately outside ``data/`` so no validator mistakes it for
#: a record.
ALLOWED_STAGING_DIRS = frozenset({'data', 'adapter_reports'})


class StagingError(RuntimeError):
    """Raised when an adapter wrote somewhere it is not allowed to."""


@dataclass(frozen=True)
class AdapterProcess:
    """What running the adapter actually did."""

    argv: list[str]
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, limit: int = 2000) -> str:
        """Return the end of the adapter's output, for a failure message."""
        combined = f'{self.stdout}\n{self.stderr}'.strip()
        return combined[-limit:]


@dataclass(frozen=True)
class StagedRecord:
    """One aggregate record, its optional sidecar, and its identity."""

    aggregate: Path
    repo_path: str
    collection: str
    model_id: str
    fingerprint: str
    samples: Path | None = None
    samples_repo_path: str | None = None

    @property
    def paths(self) -> list[Path]:
        return [self.aggregate] + ([self.samples] if self.samples else [])


@dataclass(frozen=True)
class ValidationSummary:
    """The validator's verdict on a staged tree."""

    returncode: int
    valid: int
    invalid: int
    warnings: int
    problems: list[str]
    stderr: str = ''

    @property
    def publishable(self) -> bool:
        """Whether these files may be published.

        Warnings block: the datastore review bot treats a warning-only record
        as not merge-ready, and a scheduled run should not be the thing that
        parks warnings in a pull request nobody reads.
        """
        return (
            self.returncode == 0
            and self.invalid == 0
            and self.warnings == 0
            and self.valid > 0
        )


@dataclass
class RunOutcome:
    """Everything a caller needs to report, upload, or fail the job on."""

    adapter: str
    run_date: date
    status: Status
    staging_dir: Path
    upload_dir: Path
    raw_dir: Path
    process: AdapterProcess | None = None
    validation: ValidationSummary | None = None
    records: list[StagedRecord] = field(default_factory=list)
    uploaded: list[StagedRecord] = field(default_factory=list)
    skipped_unchanged: list[StagedRecord] = field(default_factory=list)
    coverage: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the scheduled job should be considered healthy."""
        return self.status != 'failed'

    @property
    def has_upload(self) -> bool:
        return bool(self.uploaded)

    @property
    def new_fingerprints(self) -> list[str]:
        return [record.fingerprint for record in self.uploaded]

    def coverage_line(self) -> str:
        """One line a reviewer can read instead of counting files."""
        parts = [f'{len(self.records)} record(s) produced']
        if self.coverage:
            parts.insert(
                0, f'{self.coverage["total_source_records"]} source row(s)'
            )
            dropped = self.coverage['failed_record_count']
            excluded = self.coverage['excluded_record_count']
            if dropped:
                parts.append(f'{dropped} dropped')
            if excluded:
                parts.append(f'{excluded} excluded as non-evaluations')
        if self.skipped_unchanged:
            parts.append(
                f'{len(self.skipped_unchanged)} unchanged since the last run'
            )
        parts.append(f'{len(self.uploaded)} uploaded')
        return ' -> '.join([parts[0], ', '.join(parts[1:])])

    def to_manifest(self) -> dict[str, Any]:
        """A JSON record of the run, stored next to that run's raw data."""
        return {
            'adapter': self.adapter,
            'run_date': self.run_date.isoformat(),
            'status': self.status,
            'adapter_exit_code': (
                self.process.returncode if self.process else None
            ),
            'adapter_timed_out': (
                self.process.timed_out if self.process else None
            ),
            'records_produced': len(self.records),
            'records_uploaded': len(self.uploaded),
            'records_skipped_unchanged': len(self.skipped_unchanged),
            'coverage': self.coverage,
            'validation': (
                None
                if self.validation is None
                else {
                    'valid': self.validation.valid,
                    'invalid': self.validation.invalid,
                    'warnings': self.validation.warnings,
                    'problems': self.validation.problems[:50],
                }
            ),
            'uploaded': [
                {
                    'repo_path': record.repo_path,
                    'model_id': record.model_id,
                    'fingerprint': record.fingerprint,
                }
                for record in self.uploaded
            ],
            'skipped_unchanged': [
                {
                    'repo_path': record.repo_path,
                    'model_id': record.model_id,
                    'fingerprint': record.fingerprint,
                }
                for record in self.skipped_unchanged
            ],
            'messages': self.messages,
        }


def missing_credentials(spec: AdapterSpec, env: dict[str, str]) -> list[str]:
    """Return the adapter's required credentials that are not set."""
    return [name for name in spec.required_env if not env.get(name)]


def missing_dependencies(spec: AdapterSpec) -> list[str]:
    """Return packages the adapter needs that are not importable.

    Checked up front so a missing optional dependency is reported as a
    configuration gap rather than surfacing as an adapter crash.
    """
    missing = []
    for package in spec.with_packages:
        try:
            found = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing


def adapter_environment(
    spec: AdapterSpec,
    *,
    raw_dir: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the environment one adapter subprocess should see.

    Credentials belonging to *other* adapters are removed: a leaderboard
    scraper has no reason to be handed another service's API key.
    """
    env = dict(os.environ if base_env is None else base_env)
    for name in ALL_CREDENTIAL_ENV - set(spec.required_env):
        env.pop(name, None)
    env[CAPTURE_DIR_ENV] = str(raw_dir)
    return env


def run_adapter(
    spec: AdapterSpec,
    *,
    data_root: Path,
    raw_dir: Path,
    base_env: dict[str, str] | None = None,
    cwd: Path | None = None,
    executable: str | None = None,
) -> AdapterProcess:
    """Run the adapter in its own process, staged and time-boxed."""
    argv = [
        executable or sys.executable,
        '-m',
        spec.module,
        *spec.argv(data_root),
    ]
    timeout = spec.timeout_minutes * 60
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None if cwd is None else str(cwd),
            env=adapter_environment(spec, raw_dir=raw_dir, base_env=base_env),
        )
    except subprocess.TimeoutExpired as expired:
        return AdapterProcess(
            argv=argv,
            returncode=124,
            timed_out=True,
            stdout=_as_text(expired.stdout),
            stderr=(
                f'{_as_text(expired.stderr)}\n'
                f'timed out after {spec.timeout_minutes} minute(s)'
            ).strip(),
        )
    return AdapterProcess(
        argv=argv,
        returncode=completed.returncode,
        timed_out=False,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def discover_records(
    staging_dir: Path, spec: AdapterSpec
) -> list[StagedRecord]:
    """Return the records an adapter staged, refusing anything misplaced.

    This is what makes the per-adapter ``--output-dir`` contract enforceable
    instead of merely documented: whatever the adapter *meant* to do, only
    files at a canonical datastore path inside this run's own staging tree,
    in a collection the registry declared, are eligible to be published.
    """
    staging_dir = Path(staging_dir)
    stray = sorted(
        entry.name
        for entry in staging_dir.iterdir()
        if entry.name not in ALLOWED_STAGING_DIRS
    )
    if stray:
        raise StagingError(
            f'{spec.key}: adapter wrote outside its staging layout: '
            f'{", ".join(stray)}'
        )

    data_root = staging_dir / 'data'
    if not data_root.is_dir():
        return []

    aggregates: dict[str, Path] = {}
    samples: dict[str, Path] = {}
    problems: list[str] = []
    for path in sorted(data_root.rglob('*')):
        if path.is_dir():
            continue
        relative = path.relative_to(staging_dir)
        parts = relative.parts
        if len(parts) != 5:
            problems.append(
                f'{relative.as_posix()}: expected '
                'data/<collection>/<developer>/<model>/<file>'
            )
            continue
        collection, filename = parts[1], parts[4]
        if collection not in spec.collections:
            problems.append(
                f'{relative.as_posix()}: collection {collection!r} is not '
                f'declared by {spec.key} '
                f'(declared: {", ".join(spec.collections)})'
            )
            continue
        repo_path = PurePosixPath(*parts).as_posix()
        if _AGGREGATE_NAME.fullmatch(filename):
            aggregates[repo_path] = path
        elif _SAMPLES_NAME.fullmatch(filename):
            samples[repo_path] = path
        else:
            problems.append(
                f'{relative.as_posix()}: filename is neither '
                '{uuid4}.json nor {uuid4}_samples.jsonl'
            )

    if problems:
        raise StagingError(
            f'{spec.key}: staged output is not publishable:\n  '
            + '\n  '.join(problems)
        )

    records = []
    for repo_path, path in sorted(aggregates.items()):
        payload = strict_json_loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise StagingError(
                f'{spec.key}: {repo_path} does not contain a JSON object'
            )
        sample_repo_path = repo_path.removesuffix('.json') + '_samples.jsonl'
        sample_path = samples.pop(sample_repo_path, None)
        model_info = payload.get('model_info')
        records.append(
            StagedRecord(
                aggregate=path,
                repo_path=repo_path,
                collection=PurePosixPath(repo_path).parts[1],
                model_id=(
                    model_info.get('id', '')
                    if isinstance(model_info, dict)
                    else ''
                ),
                # Fingerprint the record as the adapter produced it. Stamping
                # adds the run date, which would make every record look new.
                fingerprint=normalized_hash(payload),
                samples=sample_path,
                samples_repo_path=(
                    sample_repo_path if sample_path is not None else None
                ),
            )
        )

    if samples:
        raise StagingError(
            f'{spec.key}: samples files with no sibling aggregate: '
            f'{", ".join(sorted(samples))}'
        )
    return records


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Call a packaged CLI entry point in-process and capture its output.

    The adapter runs in its own process because it needs isolation and a
    timeout. The validator does not: this calls the same ``main`` the
    ``every_eval_ever`` command dispatches to, so the gate is the documented
    one without paying for an interpreter start per check.
    """
    from every_eval_ever.cli import main as cli_main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli_main(argv)
        except SystemExit as exit_request:  # argparse usage errors
            code = (
                exit_request.code if isinstance(exit_request.code, int) else 1
            )
    return code, out.getvalue(), err.getvalue()


def validate_staging(staging_dir: Path) -> ValidationSummary:
    """Run the packaged validator over a staged tree, as a contributor would.

    The command is exactly ``every_eval_ever validate --format json
    '<staging>/data/*/*/*/*.json*'``. Passing the absolute staged path lets
    the validator map it back to its canonical ``data/...`` repository path,
    so the semantic merge-gate checks run and not only the schema.
    """
    data_root = Path(staging_dir).resolve() / 'data'
    pattern = str(data_root / '*' / '*' / '*' / '*.json*')
    code, stdout, stderr = _run_cli(['validate', '--format', 'json', pattern])
    try:
        reports = json.loads(stdout)
    except json.JSONDecodeError:
        return ValidationSummary(
            returncode=code or 1,
            valid=0,
            invalid=0,
            warnings=0,
            problems=['validator produced no machine-readable report'],
            stderr=stderr.strip(),
        )

    problems: list[str] = []
    valid = invalid = warnings = 0
    for report in reports:
        name = PurePosixPath(str(report['file']).replace('\\', '/')).name
        if report['valid']:
            valid += 1
        else:
            invalid += 1
        for error in report['errors']:
            problems.append(f'{name}: {error["loc"]}: {error["msg"]}')
        for warning in report['warnings']:
            warnings += 1
            problems.append(
                f'{name}: warning: {warning["loc"]}: {warning["msg"]}'
            )
    return ValidationSummary(
        returncode=code,
        valid=valid,
        invalid=invalid,
        warnings=warnings,
        problems=problems,
        stderr=stderr.strip(),
    )


def check_duplicates(staging_dir: Path) -> str | None:
    """Return a description of duplicate records in the batch, if any.

    Two records that differ only in ``evaluation_id`` and
    ``retrieved_timestamp`` are the same result published twice — a
    conversion bug, not something to open a pull request with.
    """
    data_root = Path(staging_dir).resolve() / 'data'
    if not data_root.is_dir():
        return None
    code, stdout, stderr = _run_cli(['check-duplicates', str(data_root)])
    if code == 0:
        return None
    return (stdout + stderr).strip()


def read_coverage(staging_dir: Path) -> dict[str, Any] | None:
    """Summarise the adapter's own provenance reports, if it wrote any."""
    reports_dir = Path(staging_dir) / 'adapter_reports'
    if not reports_dir.is_dir():
        return None
    totals = {
        'total_source_records': 0,
        'converted_records': 0,
        'failed_record_count': 0,
        'excluded_record_count': 0,
    }
    reasons: list[str] = []
    found = False
    for path in sorted(reports_dir.rglob('*_failures.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        found = True
        for key in totals:
            value = report.get(key)
            if isinstance(value, int):
                totals[key] += value
        for failure in report.get('failed_records', [])[:5]:
            reason = failure.get('reason')
            if reason and reason not in reasons:
                reasons.append(reason)
    if not found:
        return None
    totals['example_reasons'] = reasons
    return totals


def build_upload_tree(
    records: list[StagedRecord],
    upload_dir: Path,
    *,
    adapter: str,
    run_date: date,
    run_url: str | None = None,
) -> list[Path]:
    """Write stamped copies of ``records`` into a clean upload tree.

    Only records that reach here are published, so the upload tree is the
    single place that decides what a pull request receives.
    """
    upload_dir = Path(upload_dir)
    written: list[Path] = []
    for record in records:
        payload = strict_json_loads(
            record.aggregate.read_text(encoding='utf-8')
        )
        stamped = stamp_cron_provenance(
            payload, adapter=adapter, run_date=run_date, run_url=run_url
        )
        target = upload_dir / record.repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(stamped, indent=2, ensure_ascii=False, allow_nan=False)
            + '\n',
            encoding='utf-8',
        )
        written.append(target)
        if record.samples is not None and record.samples_repo_path:
            # Copied byte for byte: the aggregate's declared checksum is over
            # these bytes.
            sample_target = upload_dir / record.samples_repo_path
            shutil.copyfile(record.samples, sample_target)
            written.append(sample_target)
    return written


def run(
    spec: AdapterSpec,
    workdir: Path,
    *,
    run_date: date,
    known_fingerprints: set[str] | None = None,
    force_full: bool = False,
    run_url: str | None = None,
    base_env: dict[str, str] | None = None,
    cwd: Path | None = None,
    executable: str | None = None,
) -> RunOutcome:
    """Produce one adapter's upload tree, or explain why there isn't one."""
    workdir = Path(workdir)
    staging_dir = workdir / 'staging'
    upload_dir = workdir / 'upload'
    raw_dir = workdir / 'raw'
    for directory in (staging_dir, upload_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (staging_dir / 'data').mkdir(exist_ok=True)

    outcome = RunOutcome(
        adapter=spec.key,
        run_date=run_date,
        status='failed',
        staging_dir=staging_dir,
        upload_dir=upload_dir,
        raw_dir=raw_dir,
    )

    env = dict(os.environ if base_env is None else base_env)
    absent = missing_credentials(spec, env)
    if absent:
        outcome.status = 'skipped_missing_credential'
        outcome.messages.append(
            f'not run: {", ".join(absent)} is not configured'
        )
        return outcome

    unavailable = missing_dependencies(spec)
    if unavailable:
        outcome.status = 'skipped_missing_dependency'
        outcome.messages.append(
            f'not run: install {", ".join(unavailable)} to run this adapter'
        )
        return outcome

    outcome.process = run_adapter(
        spec,
        data_root=staging_dir / 'data',
        raw_dir=raw_dir,
        base_env=env,
        cwd=cwd,
        executable=executable,
    )

    try:
        outcome.records = discover_records(staging_dir, spec)
    except (StagingError, ValueError) as exc:
        # ValueError covers a staged file that is not strict JSON, which the
        # publisher should have made impossible; either way nothing ships.
        outcome.messages.append(str(exc))
        return outcome

    outcome.coverage = read_coverage(staging_dir)

    if not outcome.process.ok:
        partial = bool(outcome.records) and outcome.coverage is not None
        if not partial:
            reason = (
                'timed out'
                if outcome.process.timed_out
                else f'exited {outcome.process.returncode}'
            )
            outcome.messages.append(
                f'adapter {reason} and left no accounted-for records; '
                f'output tail:\n{outcome.process.tail()}'
            )
            return outcome
        if not spec.allow_partial:
            outcome.messages.append(
                'adapter reported a partial conversion and this adapter is '
                'configured to require a complete one'
            )
            return outcome
        outcome.status = 'partial'
        outcome.messages.append(
            'partial refresh: the adapter exited '
            f'{outcome.process.returncode} and reported dropped source rows'
        )
    else:
        outcome.status = 'completed'

    if not outcome.records:
        outcome.status = 'failed'
        outcome.messages.append(
            'adapter exited cleanly but produced no records; treating an '
            'empty refresh as a failure rather than an up-to-date one'
        )
        return outcome

    outcome.validation = validate_staging(staging_dir)
    if not outcome.validation.publishable:
        outcome.status = 'failed'
        outcome.messages.append(
            'validation did not pass: '
            f'{outcome.validation.valid} valid, '
            f'{outcome.validation.invalid} invalid, '
            f'{outcome.validation.warnings} warning(s)'
        )
        outcome.messages.extend(outcome.validation.problems[:20])
        if outcome.validation.stderr:
            outcome.messages.append(outcome.validation.stderr)
        return outcome

    duplicates = check_duplicates(staging_dir)
    if duplicates:
        outcome.status = 'failed'
        outcome.messages.append(
            'the adapter produced duplicate records in one batch:\n'
            f'{duplicates}'
        )
        return outcome

    known: set[str] = set() if force_full else set(known_fingerprints or ())
    for record in outcome.records:
        if record.fingerprint in known:
            outcome.skipped_unchanged.append(record)
        else:
            outcome.uploaded.append(record)
    if force_full:
        outcome.messages.append(
            'de-duplication bypassed: publishing every record'
        )

    build_upload_tree(
        outcome.uploaded,
        upload_dir,
        adapter=spec.key,
        run_date=run_date,
        run_url=run_url,
    )
    return outcome


__all__ = [
    'ALLOWED_STAGING_DIRS',
    'ALL_CREDENTIAL_ENV',
    'AdapterProcess',
    'RunOutcome',
    'StagedRecord',
    'StagingError',
    'Status',
    'ValidationSummary',
    'adapter_environment',
    'build_upload_tree',
    'check_duplicates',
    'discover_records',
    'missing_credentials',
    'missing_dependencies',
    'read_coverage',
    'run',
    'run_adapter',
    'validate_staging',
]
