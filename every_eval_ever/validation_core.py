"""Shared validation checks for Every Eval Ever data.

This module is the source of truth for package CLI validation and the
datastore validator Space.  It intentionally keeps orchestration out: callers
provide local files, repo-relative paths, available companion files, and an
optional ``HfApi`` for required Hugging Face existence checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import ValidationError

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog
from every_eval_ever.schema import schema_json, schema_text

DEFAULT_MAX_ERRORS = 50

_EXPECTED_PATH_PARTS = 5  # data / benchmark / developer / model / filename
_UUID_FILE_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
    r'(?:_samples)?\.jsonl?$',
    re.IGNORECASE,
)
_COUNT_FIELDS = frozenset(
    {'num_samples', 'num_bootstrap_samples', 'samples_number'}
)

_DEPLOYMENT_TYPES = ('api', 'local', 'unknown')
_AVAILABILITY_BY_DEPLOYMENT: dict[str, tuple[str, ...]] = {
    'api': ('closed_source', 'open_weights_deployment', 'other'),
    'local': ('hf', 'unavailable', 'other'),
}

_existence_cache: dict[tuple[str, str], tuple[bool | None, str | None]] = {}


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
    """Context supplied by CLI or Space orchestration for semantic checks."""

    local_path: Path
    repo_path: str
    available_files: Container[str] = field(default_factory=frozenset)
    hf_api: Any = None


CheckScope = Literal['aggregate', 'instance', 'file']


@dataclass(frozen=True)
class ValidationCheck:
    """A named validation check registered with the shared runner."""

    name: str
    scope: CheckScope
    run: Callable[[ValidationContext, dict[str, Any] | None], list[str]]


def get_schema_version() -> str:
    """Read the bundled aggregate schema version."""
    data = schema_json('eval.schema.json')
    version = data.get('version')
    if not isinstance(version, str) or not version.strip():
        raise ValueError("eval.schema.json missing or empty 'version' field")
    return version.strip()


def get_schema_fingerprint() -> str:
    """SHA-256 of the bundled aggregate and instance schema files."""
    hasher = hashlib.sha256()
    hasher.update(schema_text('eval.schema.json').encode())
    hasher.update(schema_text('instance_level_eval.schema.json').encode())
    return hasher.hexdigest()


def repo_path_from_path(path: Path) -> str:
    """Best-effort repo-relative path for local CLI use.

    If an absolute local path contains a ``data`` component, warnings should use
    the datastore path from that point onward. Otherwise the supplied path is
    used as-is.
    """
    raw = path.as_posix()
    parts = list(path.parts)
    if 'data' in parts:
        data_index = parts.index('data')
        return '/'.join(parts[data_index:])
    return raw


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


def format_warning(warning: dict[str, Any]) -> str:
    """Format a warning dict as the signature used for grouping."""
    loc = warning.get('loc')
    msg = warning.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def format_error(error: dict[str, Any]) -> str:
    loc = error.get('loc')
    msg = error.get('msg', '')
    return f'{loc}: {msg}' if loc else str(msg)


def check_path_structure(repo_path: str) -> list[str]:
    """Warn unless path matches data/{benchmark}/{developer}/{model}/{uuid}.json[l]."""
    parts = [p for p in repo_path.split('/') if p]

    if len(parts) != _EXPECTED_PATH_PARTS:
        return [
            'Unexpected path depth: expected '
            "'data/benchmark/developer/model/uuid.json[l]', "
            f"got {len(parts)} components in '{repo_path}'"
        ]

    if parts[0] != 'data':
        return [f"Path does not start with 'data/': '{repo_path}'"]

    if not _UUID_FILE_RE.match(parts[4]):
        return [
            f"Filename '{parts[4]}' does not match "
            f"'{{UUID4}}[_samples].json[l]' in '{repo_path}'"
        ]

    return []


def check_companion_exists(
    repo_path: str,
    aggregate_data: dict[str, Any],
    available_files: Container[str],
) -> list[str]:
    """Warn when aggregate detailed results point to a missing JSONL companion."""
    detail = aggregate_data.get('detailed_evaluation_results')
    if not isinstance(detail, dict) or not detail.get('file_path'):
        return []

    folder = Path(repo_path).parent
    uuid = Path(repo_path).stem
    expected = {
        str(folder / f'{uuid}.jsonl'),
        str(folder / f'{uuid}_samples.jsonl'),
    }
    if not any(path in available_files for path in expected):
        return [
            f"Companion .jsonl for '{Path(repo_path).name}' not found "
            'in the dataset or this PR'
        ]
    return []


def check_score_metadata(data: dict[str, Any]) -> list[str]:
    """Warn on missing score type/bounds and score values outside bounds."""
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
        for key in ('score_type', 'min_score', 'max_score'):
            if key not in metric:
                warnings.append(
                    f"evaluation_results[{index}].metric_config: missing '{key}'"
                )

        score_details = result.get('score_details')
        if not isinstance(score_details, dict):
            continue
        score = score_details.get('score')
        lo = metric.get('min_score')
        hi = metric.get('max_score')
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and isinstance(lo, (int, float))
            and not isinstance(lo, bool)
            and isinstance(hi, (int, float))
            and not isinstance(hi, bool)
            and (score < lo or score > hi)
        ):
            warnings.append(
                f'evaluation_results[{index}]: score {score} is outside '
                f'[min_score={lo}, max_score={hi}]'
            )
    return warnings


def check_integer_counts(data: dict[str, Any]) -> list[str]:
    """Warn when count fields are present but not plain integers."""
    warnings: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f'{path}.{key}'
                if key in _COUNT_FIELDS and value is not None:
                    if isinstance(value, bool) or not isinstance(value, int):
                        warnings.append(
                            f'{child}: expected integer count, got {value!r}'
                        )
                walk(value, child)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, f'{path}[{index}]')

    walk(data, '$')
    return warnings


def _hf_exists(
    api: Any, kind: str, repo_id: str
) -> tuple[bool | None, str | None]:
    """Return (exists, error); exists is None when verification failed."""
    key = (kind, repo_id)
    if key in _existence_cache:
        return _existence_cache[key]
    try:
        if kind == 'model':
            api.model_info(repo_id)
        else:
            api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        result = (False, None)
    except Exception as exc:
        detail = f'{type(exc).__name__}: {exc}'
        result = (None, detail)
    else:
        result = (True, None)
    _existence_cache[key] = result
    return result


def check_model_deployment(data: dict[str, Any], api: Any = None) -> list[str]:
    """Warn on missing/invalid model deployment metadata.

    ``model_info.additional_details.deployment_type`` is required and must be
    ``api``, ``local``, or ``unknown``.  For ``api`` and ``local``,
    ``model_availability`` is also required.  When availability is ``hf``,
    Hugging Face model existence verification is required.
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
            '(expected api|local|unknown)'
        )
        return warnings
    if deployment_type not in _DEPLOYMENT_TYPES:
        warnings.append(
            'model_info.additional_details.deployment_type: expected one of '
            f'{list(_DEPLOYMENT_TYPES)}, got {deployment_type!r}'
        )
        return warnings

    availability = details.get('model_availability')
    allowed = _AVAILABILITY_BY_DEPLOYMENT.get(deployment_type)
    if allowed is not None:
        if availability is None:
            warnings.append(
                "model_info.additional_details: missing 'model_availability' "
                f'for deployment_type={deployment_type!r} '
                f'(expected one of {list(allowed)})'
            )
        elif availability not in allowed:
            warnings.append(
                'model_info.additional_details.model_availability: expected '
                f'one of {list(allowed)} for deployment_type={deployment_type!r}, '
                f'got {availability!r}'
            )

    if availability == 'hf':
        model_id = model_info.get('id')
        if not isinstance(model_id, str) or not model_id:
            warnings.append(
                "model_info.id: missing model id for model_availability='hf'"
            )
        elif api is None:
            warnings.append(
                f'model_info.id {model_id!r}: HuggingFace model existence '
                "check required because model_availability is 'hf', but no "
                'HfApi was provided'
            )
        else:
            exists, error = _hf_exists(api, 'model', model_id)
            if exists is False:
                warnings.append(
                    f'model_info.id {model_id!r}: not found on HuggingFace '
                    "(model_availability is 'hf')"
                )
            elif exists is None:
                warnings.append(
                    f'model_info.id {model_id!r}: HuggingFace model '
                    f'existence check did not complete: {error}'
                )
    return warnings


def check_dataset_provenance(
    data: dict[str, Any], api: Any = None
) -> list[str]:
    """Warn on weak dataset provenance and verify HF dataset repos."""
    warnings: list[str] = []
    results = data.get('evaluation_results')
    if not isinstance(results, list):
        return warnings

    other_count = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        source_data = result.get('source_data')
        if not isinstance(source_data, dict):
            continue
        source_type = source_data.get('source_type')
        if source_type == 'hf_dataset':
            repo = source_data.get('hf_repo')
            if not isinstance(repo, str) or not repo:
                warnings.append(
                    f'evaluation_results[{index}].source_data: source_type '
                    "is 'hf_dataset' but 'hf_repo' is missing"
                )
            elif api is None:
                warnings.append(
                    f'evaluation_results[{index}].source_data: HuggingFace '
                    f'dataset existence check required for {repo!r}, but no '
                    'HfApi was provided'
                )
            else:
                exists, error = _hf_exists(api, 'dataset', repo)
                if exists is False:
                    warnings.append(
                        f'evaluation_results[{index}].source_data: HF dataset '
                        f'{repo!r} not found'
                    )
                elif exists is None:
                    warnings.append(
                        f'evaluation_results[{index}].source_data: HF dataset '
                        f'existence check for {repo!r} did not complete: {error}'
                    )
        elif source_type == 'other':
            other_count += 1

    if other_count:
        warnings.append(
            f"{other_count} evaluation_results use dataset source_type 'other' "
            '(no URL/HF repo provenance)'
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
    return check_score_metadata(data or {})


def _aggregate_check_integer_counts(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    return check_integer_counts(data or {})


def _aggregate_check_model_deployment(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    return check_model_deployment(data or {}, context.hf_api)


def _aggregate_check_dataset_provenance(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    return check_dataset_provenance(data or {}, context.hf_api)


REGISTERED_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck('path structure', 'file', _file_check_path),
    ValidationCheck('companion file', 'aggregate', _aggregate_check_companion),
    ValidationCheck(
        'score metadata', 'aggregate', _aggregate_check_score_metadata
    ),
    ValidationCheck(
        'integer counts', 'aggregate', _aggregate_check_integer_counts
    ),
    ValidationCheck(
        'model deployment', 'aggregate', _aggregate_check_model_deployment
    ),
    ValidationCheck(
        'dataset provenance', 'aggregate', _aggregate_check_dataset_provenance
    ),
)


def run_registered_checks(
    context: ValidationContext,
    *,
    file_type: Literal['aggregate', 'instance'],
    data: dict[str, Any] | None,
    checks: tuple[ValidationCheck, ...] = REGISTERED_CHECKS,
) -> list[dict[str, Any]]:
    """Run registered semantic checks and return structured warnings."""
    warnings: list[dict[str, Any]] = []
    for check in checks:
        if check.scope not in {'file', file_type}:
            continue
        try:
            messages = check.run(context, data)
        except Exception as exc:
            messages = [
                f'{check.name} check did not complete: '
                f'{type(exc).__name__}: {exc or "<no detail>"}'
            ]
        warnings.extend(warning_to_dict(message) for message in messages)
    return warnings


def validate_aggregate(
    file_path: Path,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    hf_api: Any = None,
    run_semantic_checks: bool = True,
) -> ValidationReport:
    """Validate a .json file as an EvaluationLog plus semantic warnings."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='aggregate'
    )
    repo_path = repo_path or repo_path_from_path(file_path)
    if available_files is None:
        available_files = frozenset({repo_path})

    try:
        raw = file_path.read_text(encoding='utf-8')
    except OSError as exc:
        report.valid = False
        report.errors.append(
            {'loc': '(file)', 'msg': str(exc), 'type': 'io_error'}
        )
        return report

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        report.valid = False
        report.errors.append(
            {
                'loc': f'line {exc.lineno}, col {exc.colno}',
                'msg': exc.msg,
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
        context = ValidationContext(
            local_path=file_path,
            repo_path=repo_path,
            available_files=available_files,
            hf_api=hf_api,
        )
        report.warnings = run_registered_checks(
            context, file_type='aggregate', data=data
        )

    return report


def _validate_instance_line(line: str, line_num: int) -> list[dict[str, Any]]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        return [
            {
                'loc': f'line {line_num}, col {exc.colno}',
                'msg': exc.msg,
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
    run_semantic_checks: bool = True,
) -> ValidationReport:
    """Validate a .jsonl file as InstanceLevelEvaluationLog line-by-line."""
    report = ValidationReport(
        file_path=file_path, valid=True, file_type='instance'
    )
    repo_path = repo_path or repo_path_from_path(file_path)
    if available_files is None:
        available_files = frozenset({repo_path})

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
        context = ValidationContext(
            local_path=file_path,
            repo_path=repo_path,
            available_files=available_files,
        )
        report.warnings = run_registered_checks(
            context, file_type='instance', data=None
        )

    return report


def validate_file(
    file_path: Path,
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    repo_path: str | None = None,
    available_files: Container[str] | None = None,
    hf_api: Any = None,
    run_semantic_checks: bool = True,
) -> ValidationReport:
    """Dispatch validation by extension."""
    if file_path.suffix == '.json':
        return validate_aggregate(
            file_path,
            repo_path=repo_path,
            available_files=available_files,
            hf_api=hf_api,
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


def validate_many(
    files: list[tuple[str, Path]],
    *,
    max_errors: int = DEFAULT_MAX_ERRORS,
    available_files: Container[str] | None = None,
    hf_api: Any = None,
) -> list[ValidationReport]:
    """Validate repo-path/local-path pairs with a shared context."""
    available = (
        frozenset(repo_path for repo_path, _ in files)
        if available_files is None
        else available_files
    )
    return [
        validate_file(
            local_path,
            max_errors=max_errors,
            repo_path=repo_path,
            available_files=available,
            hf_api=hf_api,
        )
        for repo_path, local_path in files
    ]
