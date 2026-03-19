"""
Pydantic-based validation for EEE schema files.

Validates aggregate (.json) files against EvaluationLog and
instance-level (.jsonl) files against InstanceLevelEvaluationLog.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog

DEFAULT_MAX_ERRORS = 50


@dataclass
class ValidationReport:
    """Result of validating a single file."""

    file_path: Path
    valid: bool
    errors: list[dict] = field(default_factory=list)
    file_type: str = ""
    line_count: int = 0


def _format_loc(loc: tuple) -> str:
    """Format a Pydantic error location tuple as a readable path."""
    parts = []
    for part in loc:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        elif parts:
            parts.append(f" -> {part}")
        else:
            parts.append(str(part))
    return "".join(parts) if parts else "(root)"


def _pydantic_errors_to_dicts(exc: ValidationError) -> list[dict]:
    """Convert Pydantic ValidationError to a list of error dicts."""
    errors = []
    for err in exc.errors():
        errors.append(
            {
                "loc": _format_loc(err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
                "input": err.get("input"),
            }
        )
    return errors


def validate_aggregate(file_path: Path) -> ValidationReport:
    """Validate a .json file as an EvaluationLog."""
    report = ValidationReport(file_path=file_path, valid=True, file_type="aggregate")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        report.valid = False
        report.errors.append({"loc": "(file)", "msg": str(exc), "type": "io_error"})
        return report

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        report.valid = False
        report.errors.append(
            {
                "loc": f"line {exc.lineno}, col {exc.colno}",
                "msg": exc.msg,
                "type": "json_parse_error",
            }
        )
        return report

    try:
        EvaluationLog.model_validate(data)
    except ValidationError as exc:
        report.valid = False
        report.errors = _pydantic_errors_to_dicts(exc)

    return report


def _validate_instance_line(line: str, line_num: int) -> list[dict]:
    """Validate a single JSONL line. Returns a list of error dicts."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        return [
            {
                "loc": f"line {line_num}, col {exc.colno}",
                "msg": exc.msg,
                "type": "json_parse_error",
            }
        ]

    try:
        InstanceLevelEvaluationLog.model_validate(data)
    except ValidationError as exc:
        errors = _pydantic_errors_to_dicts(exc)
        for err in errors:
            err["loc"] = f"line {line_num} -> {err['loc']}"
        return errors

    return []


def validate_instance_file(
    file_path: Path, max_errors: int = DEFAULT_MAX_ERRORS
) -> ValidationReport:
    """Validate a .jsonl file as InstanceLevelEvaluationLog line-by-line."""
    report = ValidationReport(file_path=file_path, valid=True, file_type="instance")

    try:
        handle = file_path.open(encoding="utf-8")
    except OSError as exc:
        report.valid = False
        report.errors.append({"loc": "(file)", "msg": str(exc), "type": "io_error"})
        return report

    with handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            report.line_count += 1
            line_errors = _validate_instance_line(stripped, line_num)
            if not line_errors:
                continue

            report.valid = False
            remaining = max_errors - len(report.errors)
            if remaining <= 0:
                report.errors.append(
                    {
                        "loc": "(truncated)",
                        "msg": f"Error limit reached ({max_errors}). Use --max-errors to increase.",
                        "type": "truncated",
                    }
                )
                break

            if len(line_errors) > remaining:
                line_errors = line_errors[:remaining]

            report.errors.extend(line_errors)

            if len(report.errors) >= max_errors:
                report.errors.append(
                    {
                        "loc": "(truncated)",
                        "msg": f"Error limit reached ({max_errors}). Use --max-errors to increase.",
                        "type": "truncated",
                    }
                )
                break

    return report


def validate_file(
    file_path: Path, max_errors: int = DEFAULT_MAX_ERRORS
) -> ValidationReport:
    """Dispatch validation by file extension."""
    if file_path.suffix == ".json":
        return validate_aggregate(file_path)
    if file_path.suffix == ".jsonl":
        return validate_instance_file(file_path, max_errors)

    report = ValidationReport(file_path=file_path, valid=False, file_type="unsupported")
    report.errors.append(
        {
            "loc": "(file)",
            "msg": f"Unsupported file extension '{file_path.suffix}'. Expected .json or .jsonl",
            "type": "unsupported_extension",
        }
    )
    return report


def expand_paths(paths: list[str]) -> list[Path]:
    """Expand directories to .json and .jsonl files recursively."""
    result: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            for ext in ("*.json", "*.jsonl"):
                result.extend(sorted(path.rglob(ext)))
        else:
            result.append(path)
    return result


def _truncate(value: object, max_len: int = 80) -> str:
    """Truncate a repr for display."""
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def render_report_rich(report: ValidationReport, console: Console) -> None:
    """Render a single report as a rich panel."""
    if report.valid:
        label = Text(" PASS ", style="bold white on green")
        if report.file_type == "aggregate":
            kind = "Aggregate (EvaluationLog)"
        else:
            kind = f"Instance (InstanceLevelEvaluationLog, {report.line_count} lines)"
        header = Text.assemble(label, "  ", (kind, "dim"))
        console.print(
            Panel(
                header,
                title=f"[blue underline]{report.file_path}[/]",
                title_align="left",
                border_style="green",
            )
        )
        return

    label = Text(" FAIL ", style="bold white on red")
    kind = (
        "Aggregate (EvaluationLog)"
        if report.file_type == "aggregate"
        else "Instance (InstanceLevelEvaluationLog)"
    )
    header_line = Text.assemble(label, "  ", (kind, "dim"))

    lines = [header_line, Text("")]
    for index, err in enumerate(report.errors, start=1):
        lines.append(Text(f"  {index}. {err['loc']}", style="cyan"))
        lines.append(Text(f"     {err['msg']}"))
        if "input" in err and err["input"] is not None:
            lines.append(Text(f"     Got: {_truncate(err['input'])}", style="dim"))
        lines.append(Text(""))

    console.print(
        Panel(
            Text("\n").join(lines),
            title=f"[blue underline]{report.file_path}[/]",
            title_align="left",
            border_style="red",
        )
    )


def render_summary_rich(reports: list[ValidationReport], console: Console) -> None:
    """Render a summary panel."""
    passed = sum(1 for report in reports if report.valid)
    failed = len(reports) - passed
    total_errors = sum(len(report.errors) for report in reports)

    if failed == 0:
        message = f"All {passed} file(s) passed validation"
        style = "bold green"
    else:
        message = f"{failed} file(s) failed, {passed} passed ({total_errors} total errors)"
        style = "bold red"

    console.print()
    console.print(Panel(Text(message, style=style), title="Summary", border_style="dim"))


def render_report_json(reports: list[ValidationReport]) -> str:
    """Render all reports as a JSON array."""
    output = []
    for report in reports:
        output.append(
            {
                "file": str(report.file_path),
                "valid": report.valid,
                "file_type": report.file_type,
                "line_count": report.line_count,
                "errors": report.errors,
            }
        )
    return json.dumps(output, indent=2, default=str)


def render_report_github(reports: list[ValidationReport]) -> str:
    """Render errors as GitHub Actions annotations."""
    lines = []
    for report in reports:
        for err in report.errors:
            lines.append(f"::error file={report.file_path}::{err['loc']}: {err['msg']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for package-local validation."""
    parser = argparse.ArgumentParser(
        prog="every_eval_ever validate",
        description="Validate EEE schema files using Pydantic models",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File or directory paths to validate (.json for aggregate, .jsonl for instance-level)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"Maximum errors per JSONL file (default: {DEFAULT_MAX_ERRORS})",
    )
    parser.add_argument(
        "--format",
        choices=["rich", "json", "github"],
        default="rich",
        dest="output_format",
        help="Output format (default: rich)",
    )
    args = parser.parse_args(argv)

    file_paths = expand_paths(args.paths)
    if not file_paths:
        print("No files found to validate.", file=sys.stderr)
        return 1

    reports = [validate_file(path, max_errors=args.max_errors) for path in file_paths]

    if args.output_format == "rich":
        console = Console()
        console.print()
        for report in reports:
            render_report_rich(report, console)
        render_summary_rich(reports, console)
        console.print()
    elif args.output_format == "json":
        print(render_report_json(reports))
    else:
        output = render_report_github(reports)
        if output:
            print(output)

    return 1 if any(not report.valid for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
