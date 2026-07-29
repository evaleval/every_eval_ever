"""CLI and compatibility exports for shared EEE validation."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from every_eval_ever.validation_core import (
    DEFAULT_MAX_ERRORS,
    ValidationReport,
    check_companion_exists,
    check_model_deployment,
    check_path_structure,
    check_score_metadata,
    format_error,
    format_warning,
    get_schema_fingerprint,
    get_schema_version,
    resolve_companion_repo_path,
    validate_aggregate,
    validate_file,
    validate_instance_file,
)

__all__ = [
    'DEFAULT_MAX_ERRORS',
    'ValidationReport',
    'check_companion_exists',
    'check_model_deployment',
    'check_path_structure',
    'check_score_metadata',
    'expand_paths',
    'format_error',
    'format_warning',
    'get_schema_fingerprint',
    'get_schema_version',
    'main',
    'render_report_github',
    'render_report_json',
    'render_report_rich',
    'render_summary_rich',
    'resolve_companion_repo_path',
    'validate_aggregate',
    'validate_file',
    'validate_instance_file',
]


class _LocalRepositoryFiles:
    """Read canonical datastore paths from one local checkout or data root."""

    def __init__(self, root: Path = Path('.')) -> None:
        self.root = root

    def __contains__(self, repo_path: object) -> bool:
        if not isinstance(repo_path, str):
            return False
        path = Path(repo_path)
        return not path.is_absolute() and (self.root / path).is_file()

    def read_text(self, repo_path: str) -> str:
        path = Path(repo_path)
        if path.is_absolute():
            raise OSError(f'expected a repository-relative path: {repo_path}')
        return (self.root / path).read_text(encoding='utf-8')


def _local_path_context(path: Path) -> tuple[str, _LocalRepositoryFiles]:
    """Map a local file to its canonical datastore path and file reader.

    Relative paths are read from the current checkout. For an absolute local
    smoke-output path, find its enclosing ``data`` directory and use the
    directory above it as the local repository root. This preserves the
    canonical path rule without requiring users to copy generated data into
    the repository just to validate it.
    """
    if not path.is_absolute():
        return path.as_posix(), _LocalRepositoryFiles()

    data_dir = next(
        (ancestor for ancestor in path.parents if ancestor.name == 'data'),
        None,
    )
    if data_dir is None:
        # Let the standard path validator explain why an arbitrary absolute
        # location cannot be treated as datastore data.
        return path.as_posix(), _LocalRepositoryFiles()
    return (
        path.relative_to(data_dir.parent).as_posix(),
        _LocalRepositoryFiles(data_dir.parent),
    )


def expand_paths(paths: list[str]) -> list[Path]:
    """Expand explicit glob arguments without accepting directories."""
    result: list[Path] = []
    seen: set[Path] = set()
    for value in paths:
        matches = (
            sorted(glob.glob(value, recursive='**' in value))
            if glob.has_magic(value)
            else [value]
        )
        if not matches:
            raise ValueError(f'file pattern matched no files: {value!r}')
        for match in matches:
            path = Path(match)
            if path.is_dir():
                raise ValueError(
                    f'directory arguments are not supported: {match!r}; '
                    'pass files or a fixed-depth glob'
                )
            if path not in seen:
                result.append(path)
                seen.add(path)
    return result


def render_report_json(reports: list[ValidationReport]) -> str:
    """Render all reports as a JSON array."""
    output = []
    for report in reports:
        output.append(
            {
                'file': str(report.file_path),
                'valid': report.valid,
                'file_type': report.file_type,
                'line_count': report.line_count,
                'errors': report.errors,
                'warnings': report.warnings,
            }
        )
    return json.dumps(output, indent=2, default=str)


def _truncate(value: object, max_len: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= max_len else text[: max_len - 3] + '...'


def render_report_rich(report: ValidationReport, console: Console) -> None:
    """Render one validation report as a terminal panel."""
    if report.valid:
        label = Text(' PASS ', style='bold white on green')
        kind = (
            'Aggregate (EvaluationLog)'
            if report.file_type == 'aggregate'
            else f'Instance (InstanceLevelEvaluationLog, {report.line_count} lines)'
        )
        console.print(
            Panel(
                Text.assemble(label, '  ', (kind, 'dim')),
                title=f'[blue underline]{report.file_path}[/]',
                title_align='left',
                border_style='green',
            )
        )
        return

    label = Text(' FAIL ', style='bold white on red')
    kind = (
        'Aggregate (EvaluationLog)'
        if report.file_type == 'aggregate'
        else 'Instance (InstanceLevelEvaluationLog)'
    )
    lines = [Text.assemble(label, '  ', (kind, 'dim')), Text('')]
    for index, error in enumerate(report.errors, 1):
        lines.append(Text(f'  {index}. {error["loc"]}', style='cyan'))
        lines.append(Text(f'     {error["msg"]}'))
        if error.get('input') is not None:
            lines.append(
                Text(f'     Got: {_truncate(error["input"])}', style='dim')
            )
        lines.append(Text(''))
    for warning in report.warnings:
        lines.append(
            Text(f'  Warning: {format_warning(warning)}', style='yellow')
        )
    console.print(
        Panel(
            Text('\n').join(lines),
            title=f'[blue underline]{report.file_path}[/]',
            title_align='left',
            border_style='red',
        )
    )


def render_summary_rich(
    reports: list[ValidationReport], console: Console
) -> None:
    """Render the aggregate pass/fail summary."""
    passed = sum(1 for report in reports if report.valid)
    failed = len(reports) - passed
    total_errors = sum(len(report.errors) for report in reports)
    if failed:
        style = 'bold red'
        message = (
            f'{failed} file(s) failed, {passed} passed '
            f'({total_errors} total errors)'
        )
    else:
        style = 'bold green'
        message = f'All {passed} file(s) passed validation'
    console.print()
    console.print(
        Panel(Text(message, style=style), title='Summary', border_style='dim')
    )


def render_report_github(reports: list[ValidationReport]) -> str:
    """Render errors and warnings as GitHub Actions annotations."""
    lines: list[str] = []
    for report in reports:
        lines.extend(
            f'::error file={report.file_path}::{format_error(error)}'
            for error in report.errors
        )
        lines.extend(
            f'::warning file={report.file_path}::{format_warning(warning)}'
            for warning in report.warnings
        )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='eee-validate',
        description='Validate EEE files using strict JSON and bundled schemas',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help=(
            'Files or glob patterns to validate. Directory arguments are not '
            'supported; use a glob to select files.'
        ),
    )
    parser.add_argument(
        '--max-errors',
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f'Maximum errors per JSONL file (default: {DEFAULT_MAX_ERRORS})',
    )
    parser.add_argument(
        '--format',
        choices=['rich', 'json', 'github'],
        default='rich',
        dest='output_format',
        help='Output format.',
    )
    args = parser.parse_args(argv)

    try:
        file_paths = expand_paths(args.paths)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not file_paths:
        print('No files found to validate.', file=sys.stderr)
        return 1

    local_contexts = [
        (path, *_local_path_context(path)) for path in file_paths
    ]
    reports = [
        validate_file(
            path,
            max_errors=args.max_errors,
            repo_path=repo_path,
            available_files=local_repository,
            read_repo_file=local_repository.read_text,
            run_semantic_checks=True,
        )
        for path, repo_path, local_repository in local_contexts
    ]

    if args.output_format == 'json':
        print(render_report_json(reports))
    elif args.output_format == 'github':
        output = render_report_github(reports)
        if output:
            print(output)
    else:
        console = Console()
        for report in reports:
            render_report_rich(report, console)
        render_summary_rich(reports, console)

    return 1 if any(not report.valid for report in reports) else 0


if __name__ == '__main__':
    raise SystemExit(main())
