#!/usr/bin/env python3
"""Validate local EEE files and folders without repository or PR access."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections.abc import Callable, Container
from pathlib import Path, PurePosixPath
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from every_eval_ever.validator.json_utils import (
    StrictJSONError,
    strict_json_loads,
)
from every_eval_ever.validator.validation_core import (
    DEFAULT_MAX_ERRORS,
    ValidationReport,
    check_path_structure,
    get_schema_fingerprint,
    get_schema_version,
    validate_file,
)

DATA_SUFFIXES = ('.json', '.jsonl')
_LOCAL_PARENT = PurePosixPath('data/local/local/local')
_LINE_LOCATION_RE = re.compile(r'^line \d+ -> ')
_MAX_GROUP_FILES = 5


class LocalRepositoryFiles(Container[str]):
    """Map logical repository paths used by checks to physical local files."""

    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files

    def __contains__(self, repo_path: object) -> bool:
        return isinstance(repo_path, str) and repo_path in self.files

    def read_text(self, repo_path: str) -> str:
        try:
            path = self.files[repo_path]
        except KeyError as exc:
            raise OSError(f'local file is not available: {repo_path}') from exc
        return path.read_text(encoding='utf-8')


def expand_directory(directory: Path) -> list[Path]:
    """Return data files below a directory without hidden or linked trees."""

    def raise_walk_error(error: OSError) -> None:
        raise error

    files: list[Path] = []
    for parent, directories, names in os.walk(
        directory, onerror=raise_walk_error
    ):
        directories[:] = [
            name for name in directories if not name.startswith('.')
        ]
        files.extend(
            Path(parent) / name
            for name in names
            if name.endswith(DATA_SUFFIXES) and not name.startswith('.')
        )
    files.sort()
    if not files:
        suffixes = ' or '.join(DATA_SUFFIXES)
        raise ValueError(
            f'directory contains no {suffixes} files: {directory.as_posix()!r}'
        )
    return files


def expand_inputs(
    values: list[str],
    *,
    on_directory: Callable[[Path, int], None] | None = None,
) -> list[Path]:
    """Expand files, folders, and glob patterns into unique local files."""
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        matches = (
            sorted(glob.glob(value, recursive='**' in value))
            if glob.has_magic(value)
            else [value]
        )
        if not matches:
            raise ValueError(f'file pattern matched no files: {value!r}')
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise ValueError(f'file or directory does not exist: {match!r}')
            if path.is_dir():
                expanded = expand_directory(path)
                if on_directory is not None:
                    on_directory(path, len(expanded))
            else:
                expanded = [path]
            for file_path in expanded:
                identity = file_path.absolute()
                if identity in seen:
                    continue
                result.append(file_path)
                seen.add(identity)
    return result


def _repo_path_under_data(path: Path) -> str | None:
    absolute = path.absolute()
    data_dir = next(
        (ancestor for ancestor in absolute.parents if ancestor.name == 'data'),
        None,
    )
    if data_dir is None:
        return None
    return absolute.relative_to(data_dir.parent).as_posix()


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        loaded = strict_json_loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, StrictJSONError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _declared_companion(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    detail = data.get('detailed_evaluation_results')
    if not isinstance(detail, dict):
        return None
    reference = detail.get('file_path')
    if not isinstance(reference, str) or not reference.strip():
        return None
    return reference.strip()


def _aggregate_sibling(sample_path: Path) -> Path | None:
    suffix = '_samples.jsonl'
    if not sample_path.name.endswith(suffix):
        return None
    return sample_path.with_name(f'{sample_path.name[: -len(suffix)]}.json')


def _logical_parent(path: Path, actual_repo_path: str | None) -> PurePosixPath:
    if actual_repo_path is not None:
        return PurePosixPath(actual_repo_path).parent

    aggregate_data: dict[str, Any] | None = None
    if path.suffix == '.json':
        aggregate_data = _load_object(path)
    elif path.suffix == '.jsonl':
        aggregate_path = _aggregate_sibling(path)
        if aggregate_path is not None and aggregate_path.is_file():
            aggregate_data = _load_object(aggregate_path)

    reference = _declared_companion(aggregate_data)
    if reference is not None:
        return PurePosixPath(reference).parent
    return _LOCAL_PARENT


def _repository_context(
    path: Path,
) -> tuple[str, LocalRepositoryFiles, str]:
    actual_repo_path = _repo_path_under_data(path)
    parent = _logical_parent(path, actual_repo_path)
    repo_path = (
        actual_repo_path
        if actual_repo_path is not None
        else (parent / path.name).as_posix()
    )

    try:
        siblings = [
            sibling
            for sibling in path.parent.iterdir()
            if sibling.is_file() and sibling.suffix in DATA_SUFFIXES
        ]
    except OSError:
        siblings = [path]
    files = {
        (parent / sibling.name).as_posix(): sibling for sibling in siblings
    }
    display_path = actual_repo_path or path.as_posix()
    return repo_path, LocalRepositoryFiles(files), display_path


def _declared_path_findings(path: Path) -> list[str]:
    if path.suffix != '.json':
        return []
    reference = _declared_companion(_load_object(path))
    return check_path_structure(reference) if reference is not None else []


def _path_warning(message: str, *, location: str = '(path)') -> dict[str, Any]:
    return {
        'loc': location,
        'msg': message,
        'type': 'path_warning',
    }


def _deduplicate_findings(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        key = (
            str(finding.get('loc', '')),
            str(finding.get('msg', '')),
            str(finding.get('type', '')),
        )
        if key not in seen:
            result.append(finding)
            seen.add(key)
    return result


def validate_local_file(
    path: Path, *, max_errors: int = DEFAULT_MAX_ERRORS
) -> ValidationReport:
    """Validate one physical file while treating datastore paths as advisory."""
    repo_path, repository, display_path = _repository_context(path)
    report = validate_file(
        path,
        max_errors=max_errors,
        repo_path=repo_path,
        available_files=repository,
        read_repo_file=repository.read_text,
        run_semantic_checks=True,
    )

    logical_findings = check_path_structure(repo_path)
    declared_findings = _declared_path_findings(path)
    path_messages = [*logical_findings, *declared_findings]
    retained_errors: list[dict[str, Any]] = []
    path_warnings: list[dict[str, Any]] = []
    for error in report.errors:
        location = str(error.get('loc', ''))
        message = str(error.get('msg', ''))
        rendered = f'{location}: {message}' if location else message
        matched_path_message = next(
            (
                path_message
                for path_message in path_messages
                if path_message in rendered
            ),
            None,
        )
        if matched_path_message is not None:
            path_warnings.append(_path_warning(matched_path_message))
        else:
            retained_errors.append(error)

    actual_findings = check_path_structure(display_path)
    path_warnings.extend(_path_warning(message) for message in actual_findings)
    report.errors = retained_errors
    report.warnings = _deduplicate_findings([*report.warnings, *path_warnings])
    report.valid = not report.errors
    return report


def report_status(report: ValidationReport) -> str:
    if not report.valid:
        return 'fail'
    return 'warn' if report.warnings else 'pass'


def report_merge_ready(report: ValidationReport) -> bool:
    return report.valid and not report.warnings


def _normalise_finding(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        'type': finding.get('type', ''),
        'location': finding.get('loc', ''),
        'message': finding.get('msg', ''),
        'input': finding.get('input'),
    }


def _summary(
    reports: list[ValidationReport], input_errors: list[dict[str, Any]]
) -> dict[str, Any]:
    statuses = [report_status(report) for report in reports]
    error_count = sum(len(report.errors) for report in reports) + len(
        input_errors
    )
    failed = statuses.count('fail')
    warned = statuses.count('warn')
    exit_code = 1 if input_errors or failed else 2 if warned else 0
    return {
        'files': len(reports),
        'passed': statuses.count('pass'),
        'warned': warned,
        'failed': failed,
        'errors': error_count,
        'warnings': sum(len(report.warnings) for report in reports),
        'exit_code': exit_code,
        'merge_ready': (
            not input_errors
            and bool(reports)
            and all(report_merge_ready(report) for report in reports)
        ),
    }


def json_payload(
    reports: list[ValidationReport],
    *,
    input_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the stable machine-readable local validation result."""
    input_errors = input_errors or []
    return {
        'validator': {
            'schema_version': get_schema_version(),
            'schema_fingerprint': get_schema_fingerprint(),
            'scope': 'local',
        },
        'summary': _summary(reports, input_errors),
        'input_errors': [_normalise_finding(error) for error in input_errors],
        'reports': [
            {
                'file': str(report.file_path),
                'status': report_status(report),
                'valid': report.valid,
                'merge_ready': report_merge_ready(report),
                'file_type': report.file_type,
                'line_count': report.line_count,
                'errors': [
                    _normalise_finding(error) for error in report.errors
                ],
                'warnings': [
                    _normalise_finding(warning) for warning in report.warnings
                ],
            }
            for report in reports
        ],
    }


def render_json(
    reports: list[ValidationReport],
    *,
    input_errors: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        json_payload(reports, input_errors=input_errors),
        indent=2,
        default=str,
    )


def _group_location(finding: dict[str, Any]) -> str:
    location = str(finding.get('loc') or '(root)')
    return _LINE_LOCATION_RE.sub('', location)


def group_findings(
    reports: list[ValidationReport], attribute: str
) -> list[tuple[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]]]:
    """Group report findings by type, normalized location, and message."""
    groups: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for report in reports:
        findings = getattr(report, attribute)
        for finding in findings:
            key = (
                str(finding.get('type') or 'validation_finding'),
                _group_location(finding),
                str(finding.get('msg') or ''),
            )
            groups.setdefault(key, []).append((report.file_path, finding))
    return list(groups.items())


def _render_finding_groups(
    reports: list[ValidationReport],
    console: Console,
    *,
    attribute: str,
    severity: str,
) -> None:
    groups = group_findings(reports, attribute)
    if not groups:
        return
    style = 'red' if severity == 'ERROR' else 'yellow'
    console.print()
    console.print(
        Text(
            f'{severity.title()}s: {len(groups)} group(s), '
            f'{sum(len(items) for _, items in groups)} occurrence(s)',
            style=f'bold {style}',
        )
    )
    for index, ((finding_type, location, message), items) in enumerate(
        groups, start=1
    ):
        files = list(dict.fromkeys(path for path, _ in items))
        shown_files = files[:_MAX_GROUP_FILES]
        lines = [
            Text(f'{location} [{finding_type}]', style=style),
            Text(message),
            Text(''),
            Text(
                f'{len(items)} occurrence(s) in {len(files)} file(s)',
                style='bold',
            ),
        ]
        lines.extend(Text(f'  {path}', style='dim') for path in shown_files)
        if len(files) > len(shown_files):
            lines.append(
                Text(
                    f'  ... and {len(files) - len(shown_files)} more',
                    style='dim',
                )
            )
        console.print(
            Panel(
                Text('\n').join(lines),
                title=f'{severity} GROUP {index}',
                title_align='left',
                border_style=style,
            )
        )


def render_summary_rich(
    reports: list[ValidationReport], console: Console
) -> None:
    summary = _summary(reports, [])
    message = (
        f'{summary["passed"]} passed, {summary["warned"]} warning-only, '
        f'{summary["failed"]} failed '
        f'({summary["errors"]} errors, {summary["warnings"]} warnings)'
    )
    if summary['failed']:
        style = 'bold red'
    elif summary['warned']:
        style = 'bold yellow'
        message += '\nValid locally, but not merge-ready. Fix all warnings.'
    else:
        style = 'bold green'
        message += '\nMerge-ready for the checks available locally.'
    console.print()
    console.print(
        Panel(Text(message, style=style), title='Summary', border_style='dim')
    )


def render_grouped_rich(
    reports: list[ValidationReport], console: Console
) -> None:
    """Render counts first, then grouped errors and warnings."""
    render_summary_rich(reports, console)
    _render_finding_groups(
        reports,
        console,
        attribute='errors',
        severity='ERROR',
    )
    _render_finding_groups(
        reports,
        console,
        attribute='warnings',
        severity='WARNING',
    )
    console.print()
    console.print(
        Text(
            'Use --format json for every individual finding, or '
            '--json-log PATH to save the detailed report.',
            style='dim',
        )
    )


def _input_error(message: str) -> dict[str, Any]:
    return {
        'loc': '(input)',
        'msg': message,
        'type': 'input_error',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python scripts/local_validate.py',
        description=(
            'Validate local EEE files or folders without accessing a PR.'
        ),
        epilog='Exit codes: 0 clean, 1 errors, 2 warnings only.',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files, folders, or glob patterns to validate.',
    )
    parser.add_argument(
        '--format',
        choices=['rich', 'json'],
        default='rich',
        dest='output_format',
        help='Output format (default: rich).',
    )
    parser.add_argument(
        '--json-log',
        type=Path,
        default=None,
        help='Write the detailed per-file JSON report to this path.',
    )
    args = parser.parse_args(argv)

    def report_directory(directory: Path, count: int) -> None:
        print(
            f'{directory.as_posix()}: validating {count} file(s) found '
            'recursively',
            file=sys.stderr,
        )

    try:
        paths = expand_inputs(args.paths, on_directory=report_directory)
    except (OSError, ValueError) as exc:
        error = _input_error(str(exc))
        if args.output_format == 'json':
            print(render_json([], input_errors=[error]))
        else:
            print(str(exc), file=sys.stderr)
        return 1

    reports = [validate_local_file(path) for path in paths]
    log_error: dict[str, Any] | None = None
    if args.json_log is not None:
        try:
            args.json_log.parent.mkdir(parents=True, exist_ok=True)
            args.json_log.write_text(render_json(reports), encoding='utf-8')
        except OSError as exc:
            log_error = _input_error(
                f'could not write JSON log {args.json_log}: {exc}'
            )
        else:
            print(f'Detailed JSON log: {args.json_log}', file=sys.stderr)

    if args.output_format == 'json':
        print(
            render_json(
                reports,
                input_errors=[log_error] if log_error is not None else None,
            )
        )
    else:
        console = Console()
        render_grouped_rich(reports, console)
        if log_error is not None:
            console.print(log_error['msg'], style='bold red')
    if log_error or any(not report.valid for report in reports):
        return 1
    if any(report.warnings for report in reports):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
