"""Validation checks shared by the local command and validator bot."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import ValidationError

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog
from every_eval_ever.json_utils import StrictJSONError, strict_json_loads
from every_eval_ever.schema import (
    get_schema_fingerprint as get_schema_fingerprint,
)
from every_eval_ever.schema import get_schema_version as get_schema_version

DEFAULT_MAX_ERRORS = 50

_EXPECTED_PATH_PARTS = 5  # data / benchmark / developer / model / filename
_UUID_RE = (
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
)
_AGGREGATE_FILE_RE = re.compile(rf'{_UUID_RE}\.json$', re.IGNORECASE)
_INSTANCE_FILE_RE = re.compile(rf'{_UUID_RE}_samples\.jsonl$', re.IGNORECASE)

_DEPLOYMENT_TYPES = ('self_deployed', 'externally_managed', 'unknown')
_MODEL_AVAILABILITY_TYPES = ('open_weights', 'closed_weights', 'unknown')


@dataclass
class ValidationReport:
    """Result of validating a single file."""

    file_path: Path
    valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    file_type: str = ''
    line_count: int = 0


@dataclass(frozen=True)
class ValidationContext:
    """Repository information needed by path and companion checks."""

    repo_path: str
    available_files: Container[str] = field(default_factory=frozenset)


CheckScope = Literal['aggregate', 'instance', 'file']
CheckSeverity = Literal['error', 'warning']


@dataclass(frozen=True)
class ValidationCheck:
    """A named validation check registered with the shared runner."""

    name: str
    scope: CheckScope
    severity: CheckSeverity
    run: Callable[[ValidationContext, dict[str, Any] | None], list[str]]


class SemanticCheckError(RuntimeError):
    """Raised when a registered semantic check cannot complete."""


@dataclass
class SemanticCheckReport:
    """Blocking and advisory findings produced by registered checks."""

    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _format_loc(loc: tuple[Any, ...]) -> str:
    parts = []
    for part in loc:
        if isinstance(part, int):
            parts.append(f'[{part}]')
        else:
            if parts:
                parts.append(f' -> {part}')
            else:
                parts.append(str(part))
    return ''.join(parts) if parts else '(root)'


def pydantic_errors_to_dicts(exc: ValidationError) -> list[dict[str, Any]]:
    """Convert Pydantic errors to the report format used by the CLI and Space."""
    errors: list[dict[str, Any]] = []
    for err in exc.errors():
        errors.append(
            {
                'loc': _format_loc(err['loc']),
                'msg': err['msg'],
                'type': err['type'],
                'input': err.get('input'),
            }
        )
    return errors


def warning_to_dict(message: str) -> dict[str, str]:
    """Convert a grouped warning string into a structured report warning."""
    if ': ' in message:
        loc, msg = message.split(': ', 1)
        return {'loc': loc, 'msg': msg, 'type': 'semantic_warning'}
    return {'loc': '', 'msg': message, 'type': 'semantic_warning'}


def semantic_error_to_dict(message: str) -> dict[str, str]:
    """Convert a grouped semantic-rule message into a blocking error."""
    if ': ' in message:
        loc, msg = message.split(': ', 1)
        return {'loc': loc, 'msg': msg, 'type': 'semantic_rule_error'}
    return {'loc': '', 'msg': message, 'type': 'semantic_rule_error'}


def format_warning(warning: dict[str, Any]) -> str:
    """Format a warning dict as the signature used for grouping."""
    loc = warning.get('loc')
    msg = warning.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def format_error(error: dict[str, Any]) -> str:
    loc = error.get('loc')
    msg = error.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def _json_error_details(
    exc: json.JSONDecodeError | StrictJSONError,
    *,
    line_num: int | None = None,
) -> tuple[str, str]:
    if isinstance(exc, json.JSONDecodeError):
        source_line = line_num if line_num is not None else exc.lineno
        return f'line {source_line}, col {exc.colno}', exc.msg
    location = f'line {line_num}' if line_num is not None else '(json)'
    return location, str(exc)


def check_path_structure(repo_path: str) -> list[str]:
    """Enforce aggregate and instance datastore paths."""
    parts = [p for p in repo_path.split('/') if p]

    if len(parts) != _EXPECTED_PATH_PARTS:
        return [
            'Unexpected path depth: expected '
            "'data/benchmark/developer/model/uuid.json' or "
            "'data/benchmark/developer/model/uuid_samples.jsonl', "
            f"got {len(parts)} components in '{repo_path}'"
        ]

    if parts[0] != 'data':
        return [f"Path does not start with 'data/': '{repo_path}'"]

    filename = parts[4]
    if not (
        _AGGREGATE_FILE_RE.fullmatch(filename)
        or _INSTANCE_FILE_RE.fullmatch(filename)
    ):
        return [
            f"Filename '{filename}' does not match '{{UUID4}}.json' or "
            f"'{{UUID4}}_samples.jsonl' in '{repo_path}'"
        ]

    return []


def resolve_companion_repo_path(
    repo_path: str, aggregate_data: dict[str, Any]
) -> str | None:
    """Resolve an aggregate's optional companion to a safe repository path."""
    detail = aggregate_data.get('detailed_evaluation_results')
    if detail is None:
        return None
    if not isinstance(detail, dict):
        raise ValueError('detailed_evaluation_results must be an object')

    reference = detail.get('file_path')
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            'detailed_evaluation_results.file_path: missing or blank companion path'
        )

    normalized_reference = reference.strip().replace('\\', '/')
    reference_path = PurePosixPath(normalized_reference)
    has_windows_drive = re.match(r'^[A-Za-z]:/', normalized_reference) is not None
    if (
        reference_path.is_absolute()
        or has_windows_drive
        or '..' in reference_path.parts
    ):
        raise ValueError(
            'detailed_evaluation_results.file_path: expected a relative '
            f'repository path without parent traversal, got {reference!r}'
        )

    if reference_path.parts and reference_path.parts[0] == 'data':
        return reference_path.as_posix()
    return (PurePosixPath(repo_path).parent / reference_path).as_posix()


def check_companion_exists(
    repo_path: str,
    aggregate_data: dict[str, Any],
    available_files: Container[str],
) -> list[str]:
    """Warn when an aggregate's declared detailed-results path is unusable."""
    detail = aggregate_data.get('detailed_evaluation_results')
    if detail is None:
        return []

    try:
        resolved_text = resolve_companion_repo_path(repo_path, aggregate_data)
    except ValueError as exc:
        return [str(exc)]
    if resolved_text is None:
        return []

    warnings: list[str] = []
    reference = detail['file_path']
    reference_path = PurePosixPath(reference.strip().replace('\\', '/'))
    for path_error in check_path_structure(resolved_text):
        warnings.append(
            'detailed_evaluation_results.file_path: '
            f'declared companion has invalid datastore path: {path_error}'
        )
    declared_format = detail.get('format')
    if declared_format == 'jsonl' and reference_path.suffix != '.jsonl':
        warnings.append(
            'detailed_evaluation_results.file_path: format is jsonl but '
            f'path is {reference!r}'
        )

    if resolved_text not in available_files:
        warnings.append(
            'detailed_evaluation_results.file_path: referenced companion '
            f'{resolved_text!r} was not found in the dataset or this batch'
        )
    return warnings


def check_score_metadata(data: dict[str, Any]) -> list[str]:
    """Validate supplied bounds and require them for continuous metrics."""
    warnings: list[str] = []
    results = data.get('evaluation_results')
    if not isinstance(results, list):
        return warnings

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        metric = result.get('metric_config')
        if not isinstance(metric, dict):
            continue
        score_type = metric.get('score_type')
        if score_type is not None and (
            not isinstance(score_type, str) or not score_type.strip()
        ):
            warnings.append(
                f"evaluation_results[{index}].metric_config: invalid 'score_type'"
            )

        raw_lo = metric.get('min_score')
        raw_hi = metric.get('max_score')
        lo = _metric_bound(raw_lo)
        hi = _metric_bound(raw_hi)
        requires_bounds = score_type == 'continuous'
        for key, raw_value, value in (
            ('min_score', raw_lo, lo),
            ('max_score', raw_hi, hi),
        ):
            if value is None and (requires_bounds or raw_value is not None):
                warnings.append(
                    f'evaluation_results[{index}].metric_config: missing or '
                    f"invalid '{key}'"
                )

        score_details = result.get('score_details')
        if not isinstance(score_details, dict):
            continue
        score = score_details.get('score')
        if not _is_finite_number(score):
            warnings.append(
                f'evaluation_results[{index}].score_details.score: expected a '
                f'finite number, got {score!r}'
            )
            continue
        if lo is not None and hi is not None and lo > hi:
            warnings.append(
                f'evaluation_results[{index}].metric_config: min_score '
                f'{raw_lo} is greater than max_score {raw_hi}'
            )
            continue
        if lo is not None and hi is not None and (score < lo or score > hi):
            warnings.append(
                f'evaluation_results[{index}]: score {score} is outside '
                f'[min_score={raw_lo}, max_score={raw_hi}]'
            )
    return warnings


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metric_bound(value: Any) -> float | None:
    """Return a comparable metric bound, including the strict-JSON infinity form."""
    if value == 'Infinity':
        return math.inf
    if value == '-Infinity':
        return -math.inf
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
    ):
        return float(value)
    return None


def check_model_deployment(data: dict[str, Any]) -> list[str]:
    """Require independent deployment-control and weight-availability axes.

    ``deployment_type`` describes who controlled the inference deployment;
    ``model_availability`` describes whether model weights are available.
    Neither value constrains the other. This rule deliberately performs no
    provider-specific existence check.
    """
    warnings: list[str] = []
    model_info = data.get('model_info')
    if not isinstance(model_info, dict):
        return warnings

    details = model_info.get('additional_details')
    if not isinstance(details, dict):
        details = {}

    deployment_type = details.get('deployment_type')
    if deployment_type is None:
        warnings.append(
            "model_info.additional_details: missing 'deployment_type' "
            f'(expected {"|".join(_DEPLOYMENT_TYPES)})'
        )
    elif deployment_type not in _DEPLOYMENT_TYPES:
        warnings.append(
            'model_info.additional_details.deployment_type: expected one of '
            f'{list(_DEPLOYMENT_TYPES)}, got {deployment_type!r}'
        )

    availability = details.get('model_availability')
    if availability is None:
        warnings.append(
            "model_info.additional_details: missing 'model_availability' "
            f'(expected {"|".join(_MODEL_AVAILABILITY_TYPES)})'
        )
    elif availability not in _MODEL_AVAILABILITY_TYPES:
        warnings.append(
            'model_info.additional_details.model_availability: expected one '
            f'of {list(_MODEL_AVAILABILITY_TYPES)}, got {availability!r}'
        )
    return warnings


def _file_check_path(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    return check_path_structure(context.repo_path)


def _aggregate_check_companion(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_companion_exists(
        context.repo_path, data, context.available_files
    )


def _aggregate_check_score_metadata(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_score_metadata(data)


def _aggregate_check_model_deployment(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_model_deployment(data)


REGISTERED_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck('path structure', 'file', 'error', _file_check_path),
    ValidationCheck(
        'companion file', 'aggregate', 'error', _aggregate_check_companion
    ),
    ValidationCheck(
        'score metadata', 'aggregate', 'error', _aggregate_check_score_metadata
    ),
    ValidationCheck(
        'model deployment',
        'aggregate',
        'error',
        _aggregate_check_model_deployment,
    ),
)


def run_registered_checks(
    context: ValidationContext,
    *,
    file_type: Literal['aggregate', 'instance'],
    data: dict[str, Any] | None,
    checks: tuple[ValidationCheck, ...] = REGISTERED_CHECKS,
) -> SemanticCheckReport:
    """Run registered checks and preserve their explicit severity."""
    report = SemanticCheckReport()
    for check in checks:
        if check.scope not in {'file', file_type}:
            continue
        try:
            messages = check.run(context, data)
        except Exception as exc:
            raise SemanticCheckError(
                f'{check.name} check did not complete: '
                f'{type(exc).__name__}: {exc or "<no detail>"}'
            ) from exc
        if check.severity == 'error':
            report.errors.extend(
                semantic_error_to_dict(message) for message in messages
            )
        elif check.severity == 'warning':
            report.warnings.extend(
                warning_to_dict(message) for message in messages
            )
        else:
            raise SemanticCheckError(
                f'{check.name} check has unsupported severity {check.severity!r}'
            )
    return report


def validate_aggregate(
    file_path: Path,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Validate an aggregate file, optionally including bot-only checks."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='aggregate'
    )
    try:
        raw = file_path.read_text(encoding='utf-8')
    except OSError as exc:
        report.valid = False
        report.errors.append(
            {'loc': '(file)', 'msg': str(exc), 'type': 'io_error'}
        )
        return report

    try:
        loaded = strict_json_loads(raw)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        location, message = _json_error_details(exc)
        report.valid = False
        report.errors.append(
            {
                'loc': location,
                'msg': message,
                'type': 'json_parse_error',
            }
        )
        return report

    data = loaded if isinstance(loaded, dict) else None
    try:
        EvaluationLog.model_validate(loaded)
    except ValidationError as exc:
        report.valid = False
        report.errors = pydantic_errors_to_dicts(exc)

    if run_semantic_checks:
        if repo_path is None:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': 'repo_path is required for repository validation',
                    'type': 'semantic_check_error',
                }
            )
            return report
        if available_files is None:
            available_files = frozenset({repo_path})
        context = ValidationContext(
            repo_path=repo_path,
            available_files=available_files,
        )
        try:
            semantic_report = run_registered_checks(
                context, file_type='aggregate', data=data
            )
            report.errors.extend(semantic_report.errors)
            report.warnings.extend(semantic_report.warnings)
            if semantic_report.errors:
                report.valid = False
        except SemanticCheckError as exc:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': str(exc),
                    'type': 'semantic_check_error',
                }
            )

    return report


def _validate_instance_line(line: str, line_num: int) -> list[dict[str, Any]]:
    try:
        data = strict_json_loads(line)
    except (json.JSONDecodeError, StrictJSONError) as exc:
        location, message = _json_error_details(exc, line_num=line_num)
        return [
            {
                'loc': location,
                'msg': message,
                'type': 'json_parse_error',
            }
        ]

    try:
        InstanceLevelEvaluationLog.model_validate(data)
    except ValidationError as exc:
        errors = pydantic_errors_to_dicts(exc)
        for error in errors:
            error['loc'] = f'line {line_num} -> {error["loc"]}'
        return errors

    return []


def validate_instance_file(
    file_path: Path,
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Validate a JSONL file, optionally including bot-only checks."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='instance'
    )
    try:
        handle = file_path.open(encoding='utf-8')
    except OSError as exc:
        report.valid = False
        report.errors.append(
            {'loc': '(file)', 'msg': str(exc), 'type': 'io_error'}
        )
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
                        'loc': '(truncated)',
                        'msg': (
                            f'Error limit reached ({max_errors}). '
                            'Use --max-errors to increase.'
                        ),
                        'type': 'truncated',
                    }
                )
                break
            report.errors.extend(line_errors[:remaining])
            if len(report.errors) >= max_errors:
                report.errors.append(
                    {
                        'loc': '(truncated)',
                        'msg': (
                            f'Error limit reached ({max_errors}). '
                            'Use --max-errors to increase.'
                        ),
                        'type': 'truncated',
                    }
                )
                break

    if run_semantic_checks:
        if repo_path is None:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': 'repo_path is required for repository validation',
                    'type': 'semantic_check_error',
                }
            )
            return report
        if available_files is None:
            available_files = frozenset({repo_path})
        context = ValidationContext(
            repo_path=repo_path,
            available_files=available_files,
        )
        try:
            semantic_report = run_registered_checks(
                context, file_type='instance', data=None
            )
            report.errors.extend(semantic_report.errors)
            report.warnings.extend(semantic_report.warnings)
            if semantic_report.errors:
                report.valid = False
        except SemanticCheckError as exc:
            report.valid = False
            report.errors.append(
                {
                    'loc': '(semantic checks)',
                    'msg': str(exc),
                    'type': 'semantic_check_error',
                }
            )

    return report


def validate_file(
    file_path: Path,
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    run_semantic_checks: bool = False,
) -> ValidationReport:
    """Dispatch validation by extension."""
    if file_path.suffix == '.json':
        return validate_aggregate(
            file_path,
            repo_path=repo_path,
            available_files=available_files,
            run_semantic_checks=run_semantic_checks,
        )
    if file_path.suffix == '.jsonl':
        return validate_instance_file(
            file_path,
            max_errors=max_errors,
            repo_path=repo_path,
            available_files=available_files,
            run_semantic_checks=run_semantic_checks,
        )

    report = ValidationReport(
        file_path=file_path, valid=False, file_type='unsupported'
    )
    report.errors.append(
        {
            'loc': '(file)',
            'msg': (
                f"Unsupported file extension '{file_path.suffix}'. "
                'Expected .json or .jsonl'
            ),
            'type': 'unsupported_extension',
        }
    )
    return report
