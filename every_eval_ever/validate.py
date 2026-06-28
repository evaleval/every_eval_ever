"""CLI and compatibility wrapper for EEE validation.

The validation rules live in :mod:`every_eval_ever.validation_core` so the
local CLI and the datastore validator Space run the same checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from every_eval_ever.validation_core import (
    DEFAULT_MAX_ERRORS,
    ValidationReport,
    check_companion_exists,
    check_dataset_provenance,
    check_integer_counts,
    check_model_deployment,
    check_path_structure,
    check_score_metadata,
    format_error,
    format_warning,
    get_schema_fingerprint,
    get_schema_version,
    repo_path_from_path,
    validate_aggregate,
    validate_file,
    validate_instance_file,
    validate_many,
)

__all__ = [
    'DEFAULT_MAX_ERRORS',
    'ValidationReport',
    'check_companion_exists',
    'check_dataset_provenance',
    'check_integer_counts',
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
    'repo_path_from_path',
    'validate_aggregate',
    'validate_file',
    'validate_instance_file',
    'validate_many',
]

DEFAULT_WARNING_EXAMPLES = 3
WARNING_GROUP_CAP = 15


@dataclass
class _FindingGroup:
    count: int = 0
    examples: list[str] = field(default_factory=list)


def expand_paths(paths: list[str]) -> list[Path]:
    """Expand directories to .json and .jsonl files recursively."""
    result: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            for ext in ('*.json', '*.jsonl'):
                result.extend(sorted(path.rglob(ext)))
        else:
            result.append(path)
    return result


def _truncate(value: object, max_len: int = 80) -> str:
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + '...'
    return s


def _group_findings(
    reports: list[ValidationReport],
    *,
    kind: str,
    cap: int = WARNING_GROUP_CAP,
    example_cap: int = DEFAULT_WARNING_EXAMPLES,
) -> tuple[OrderedDict[str, _FindingGroup], int, int]:
    groups: OrderedDict[str, _FindingGroup] = OrderedDict()
    distinct: set[str] = set()
    total = 0

    for report in reports:
        findings = report.warnings if kind == 'warning' else report.errors
        for finding in findings:
            signature = (
                format_warning(finding)
                if kind == 'warning'
                else format_error(finding)
            )
            total += 1
            distinct.add(signature)
            group = groups.get(signature)
            if group is None:
                if len(groups) >= cap:
                    continue
                group = _FindingGroup()
                groups[signature] = group
            group.count += 1
            if len(group.examples) < example_cap:
                group.examples.append(str(report.file_path))

    return groups, total, len(distinct)


def render_report_rich(report: ValidationReport, console: Console) -> None:
    """Render a single report as a rich panel."""
    if report.valid:
        label = Text(' PASS ', style='bold white on green')
        kind = (
            'Aggregate (EvaluationLog)'
            if report.file_type == 'aggregate'
            else f'Instance (InstanceLevelEvaluationLog, {report.line_count} lines)'
        )
        if report.warnings:
            kind += f', {len(report.warnings)} warning(s)'
        header = Text.assemble(label, '  ', (kind, 'dim'))
        border_style = 'yellow' if report.warnings else 'green'
        console.print(
            Panel(
                header,
                title=f'[blue underline]{report.file_path}[/]',
                title_align='left',
                border_style=border_style,
            )
        )
        return

    label = Text(' FAIL ', style='bold white on red')
    kind = (
        'Aggregate (EvaluationLog)'
        if report.file_type == 'aggregate'
        else 'Instance (InstanceLevelEvaluationLog)'
    )
    header_line = Text.assemble(label, '  ', (kind, 'dim'))

    lines = [header_line, Text('')]
    for index, err in enumerate(report.errors, 1):
        loc_text = Text(f'  {index}. {err["loc"]}', style='cyan')
        msg_text = Text(f'     {err["msg"]}', style='default')
        lines.append(loc_text)
        lines.append(msg_text)
        if 'input' in err and err['input'] is not None:
            lines.append(
                Text(f'     Got: {_truncate(err["input"])}', style='dim')
            )
        lines.append(Text(''))

    body = Text('\n').join(lines)
    console.print(
        Panel(
            body,
            title=f'[blue underline]{report.file_path}[/]',
            title_align='left',
            border_style='red',
        )
    )


def _render_grouped_warnings(
    reports: list[ValidationReport], console: Console
) -> None:
    groups, total, distinct = _group_findings(reports, kind='warning')
    if not total:
        return

    console.print()
    console.print(
        Panel(
            Text(
                f'{total} warning(s) across {distinct} warning pattern(s)',
                style='bold yellow',
            ),
            title='Warnings',
            border_style='yellow',
        )
    )
    for signature, group in groups.items():
        console.print(f'\n{group.count} file(s)')
        console.print(f'Warning: {signature}')
        examples = ', '.join(group.examples)
        if group.count > len(group.examples):
            examples += f', ... +{group.count - len(group.examples)} more'
        console.print(f'Examples: {examples}')

    remaining = distinct - len(groups)
    if remaining > 0:
        console.print(f'\n... and {remaining} more warning pattern(s)')


def render_summary_rich(
    reports: list[ValidationReport], console: Console
) -> None:
    """Render a summary panel and grouped semantic warnings."""
    passed = sum(1 for report in reports if report.valid)
    failed = len(reports) - passed
    total_errors = sum(len(report.errors) for report in reports)

    if failed == 0:
        style = 'bold green'
        msg = f'All {passed} file(s) passed validation'
    else:
        style = 'bold red'
        msg = (
            f'{failed} file(s) failed, {passed} passed '
            f'({total_errors} total errors)'
        )

    console.print()
    console.print(
        Panel(Text(msg, style=style), title='Summary', border_style='dim')
    )
    _render_grouped_warnings(reports, console)


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


def render_report_github(reports: list[ValidationReport]) -> str:
    """Render errors and warnings as GitHub Actions annotations."""
    lines = []
    for report in reports:
        for err in report.errors:
            lines.append(
                f'::error file={report.file_path}::{err["loc"]}: {err["msg"]}'
            )
        for warning in report.warnings:
            lines.append(
                f'::warning file={report.file_path}::{format_warning(warning)}'
            )
    return '\n'.join(lines)


def _build_hf_api():
    """Create HfApi for mandatory HF checks when HF metadata is present."""
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        return HfApi()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='eee-validate',
        description='Validate EEE schema files using shared package checks',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='File or directory paths to validate (.json for aggregate, .jsonl for instance-level)',
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
        help='Output format (default: rich)',
    )
    args = parser.parse_args(argv)

    file_paths = expand_paths(args.paths)
    if not file_paths:
        print('No files found to validate.', file=sys.stderr)
        return 1

    pairs = [(repo_path_from_path(path), path) for path in file_paths]
    available_files = {repo_path for repo_path, _ in pairs}
    reports = validate_many(
        pairs,
        max_errors=args.max_errors,
        available_files=available_files,
        hf_api=_build_hf_api(),
    )

    if args.output_format == 'rich':
        console = Console()
        console.print()
        for report in reports:
            render_report_rich(report, console)
        render_summary_rich(reports, console)
        console.print()
    elif args.output_format == 'json':
        print(render_report_json(reports))
    elif args.output_format == 'github':
        output = render_report_github(reports)
        if output:
            print(output)

    return 1 if any(not report.valid for report in reports) else 0


if __name__ == '__main__':
    raise SystemExit(main())
