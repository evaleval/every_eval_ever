"""
Pydantic-based validation for EEE schema files.

Validates aggregate (.json) files against EvaluationLog and
instance-level (_samples.jsonl) files against InstanceLevelEvaluationLog.

Usage:
    uv run python validate.py data/benchmark/dev/model/uuid.json
    uv run python validate.py data/benchmark/dev/model/uuid_samples.jsonl
    uv run python validate.py data/benchmark/dev/model/   # directory recurse
    uv run python validate.py --format json data/*.json
    uv run python validate.py --max-errors 10 data/*_samples.jsonl
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

from eval_types import EvaluationLog
from instance_level_types import InstanceLevelEvaluationLog

DEFAULT_MAX_ERRORS = 50


@dataclass
class ValidationReport:
    """Result of validating a single file."""

    file_path: Path
    valid: bool
    errors: list[dict] = field(default_factory=list)
    file_type: str = ""  # "aggregate" or "instance"
    line_count: int = 0  # for JSONL files


def _format_loc(loc: tuple) -> str:
    """Format a Pydantic error location tuple as a readable path."""
    parts = []
    for part in loc:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            if parts:
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
    except OSError as e:
        report.valid = False
        report.errors.append({"loc": "(file)", "msg": str(e), "type": "io_error"})
        return report

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        report.valid = False
        report.errors.append(
            {
                "loc": f"line {e.lineno}, col {e.colno}",
                "msg": e.msg,
                "type": "json_parse_error",
            }
        )
        return report

    try:
        EvaluationLog.model_validate(data)
    except ValidationError as e:
        report.valid = False
        report.errors = _pydantic_errors_to_dicts(e)

    return report


def _validate_instance_line(line: str, line_num: int) -> list[dict]:
    """Validate a single JSONL line. Returns list of error dicts."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return [
            {
                "loc": f"line {line_num}, col {e.colno}",
                "msg": e.msg,
                "type": "json_parse_error",
            }
        ]

    try:
        InstanceLevelEvaluationLog.model_validate(data)
    except ValidationError as e:
        errors = _pydantic_errors_to_dicts(e)
        for err in errors:
            err["loc"] = f"line {line_num} -> {err['loc']}"
        return errors

    return []


def validate_instance_file(
    file_path: Path, max_errors: int = DEFAULT_MAX_ERRORS
) -> ValidationReport:
    """Validate a .jsonl file as InstanceLevelEvaluationLog (line-by-line)."""
    report = ValidationReport(file_path=file_path, valid=True, file_type="instance")

    try:
        f = file_path.open(encoding="utf-8")
    except OSError as e:
        report.valid = False
        report.errors.append({"loc": "(file)", "msg": str(e), "type": "io_error"})
        return report

    with f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            report.line_count += 1
            line_errors = _validate_instance_line(stripped, line_num)

            if line_errors:
                report.valid = False

                # Respect max_errors by truncating line_errors to the remaining budget
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
    elif file_path.suffix == ".jsonl":
        return validate_instance_file(file_path, max_errors)
    else:
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
    for p in paths:
        path = Path(p)
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            for ext in ("*.json", "*.jsonl"):
                result.extend(sorted(path.rglob(ext)))
        else:
            result.append(path)  # let validate_file report the error
    return result


def _truncate(value: object, max_len: int = 80) -> str:
    """Truncate a repr for display."""
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------


def render_report_rich(report: ValidationReport, console: Console) -> None:
    """Render a single report as a rich panel."""
    if report.valid:
        label = Text(" PASS ", style="bold white on green")
        kind = "Aggregate (EvaluationLog)" if report.file_type == "aggregate" else f"Instance (InstanceLevelEvaluationLog, {report.line_count} lines)"
        header = Text.assemble(label, "  ", (kind, "dim"))
        console.print(
            Panel(
                header,
                title=f"[blue underline]{report.file_path}[/]",
                title_align="left",
                border_style="green",
            )
        )
    else:
        label = Text(" FAIL ", style="bold white on red")
        kind = "Aggregate (EvaluationLog)" if report.file_type == "aggregate" else "Instance (InstanceLevelEvaluationLog)"
        header_line = Text.assemble(label, "  ", (kind, "dim"))

        lines = [header_line, Text("")]
        for i, err in enumerate(report.errors, 1):
            loc_text = Text(f"  {i}. {err['loc']}", style="cyan")
            msg_text = Text(f"     {err['msg']}", style="default")
            lines.append(loc_text)
            lines.append(msg_text)
            if "input" in err and err["input"] is not None:
                got_text = Text(f"     Got: {_truncate(err['input'])}", style="dim")
                lines.append(got_text)
            lines.append(Text(""))

        body = Text("\n").join(lines)
        console.print(
            Panel(
                body,
                title=f"[blue underline]{report.file_path}[/]",
                title_align="left",
                border_style="red",
            )
        )


def render_summary_rich(reports: list[ValidationReport], console: Console) -> None:
    """Render a summary panel."""
    passed = sum(1 for r in reports if r.valid)
    failed = len(reports) - passed
    total_errors = sum(len(r.errors) for r in reports)

    if failed == 0:
        style = "bold green"
        msg = f"All {passed} file(s) passed validation"
    else:
        style = "bold red"
        msg = f"{failed} file(s) failed, {passed} passed ({total_errors} total errors)"

    console.print()
    console.print(Panel(Text(msg, style=style), title="Summary", border_style="dim"))


def render_report_json(reports: list[ValidationReport]) -> str:
    """Render all reports as a JSON array."""
    output = []
    for r in reports:
        output.append(
            {
                "file": str(r.file_path),
                "valid": r.valid,
                "file_type": r.file_type,
                "line_count": r.line_count,
                "errors": r.errors,
            }
        )
    return json.dumps(output, indent=2, default=str)


def render_report_github(reports: list[ValidationReport]) -> str:
    """Render errors as GitHub Actions annotations."""
    lines = []
    for r in reports:
        for err in r.errors:
            lines.append(f"::error file={r.file_path}::{err['loc']}: {err['msg']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eee-validate",
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
    args = parser.parse_args()

    file_paths = expand_paths(args.paths)
    if not file_paths:
        print("No files found to validate.", file=sys.stderr)
        sys.exit(1)

    reports = [validate_file(fp, max_errors=args.max_errors) for fp in file_paths]

    if args.output_format == "rich":
        console = Console()
        console.print()
        for report in reports:
            render_report_rich(report, console)
        render_summary_rich(reports, console)
        console.print()
    elif args.output_format == "json":
        print(render_report_json(reports))
    elif args.output_format == "github":
        output = render_report_github(reports)
        if output:
            print(output)

    if any(not r.valid for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
