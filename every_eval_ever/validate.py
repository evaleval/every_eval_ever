"""CLI and compatibility wrapper for EEE validation.

The validation rules live in :mod:`every_eval_ever.validation_core` so the
local CLI and the datastore validator Space run the same checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

from every_eval_ever.validation_core import (
    DEFAULT_MAX_ERRORS,
    ValidationReport,
    check_companion_exists,
    check_dataset_provenance,
    check_evaluator_provenance_consistency,
    check_integer_counts,
    check_model_deployment,
    check_path_structure,
    check_score_metadata,
    format_error,
    format_warning,
    get_schema_fingerprint,
    get_schema_version,
    repo_path_from_path,
    resolve_companion_repo_path,
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
    'check_evaluator_provenance_consistency',
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
    'render_report_json',
    'repo_path_from_path',
    'resolve_companion_repo_path',
    'validate_aggregate',
    'validate_file',
    'validate_instance_file',
    'validate_many',
]


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
        choices=['json'],
        default='json',
        dest='output_format',
        help=argparse.SUPPRESS,
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
        hf_api=HfApi(),
    )

    print(render_report_json(reports))

    return 1 if any(not report.valid for report in reports) else 0


if __name__ == '__main__':
    raise SystemExit(main())
