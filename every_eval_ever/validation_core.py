"""Shared validation checks for Every Eval Ever data.

This module is the source of truth for package CLI validation and the
datastore validator Space.  It intentionally keeps orchestration out: callers
provide local files, repo-relative paths, available companion files, and an
optional ``HfApi`` for required Hugging Face existence checks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from huggingface_hub.errors import RepositoryNotFoundError
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
_UUID_FILE_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
    r'(?:_samples)?\.jsonl?$',
    re.IGNORECASE,
)
_COUNT_FIELDS = frozenset(
    {'num_samples', 'num_bootstrap_samples', 'samples_number'}
)

_DEPLOYMENT_TYPES = ('self_deployed', 'externally_managed', 'unknown')
_MODEL_AVAILABILITY_TYPES = ('open_weights', 'closed_weights', 'unknown')
_EVALUATOR_RELATIONSHIP_TYPES = (
    'first_party',
    'third_party',
    'collaborative',
    'other',
)

# Compatibility surface: the validator Space clears this cache between jobs.
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

    reference_path = PurePosixPath(reference.strip().replace('\\', '/'))
    if reference_path.is_absolute() or '..' in reference_path.parts:
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
    declared_format = detail.get('format')
    if declared_format == 'jsonl' and reference_path.suffix != '.jsonl':
        warnings.append(
            'detailed_evaluation_results.file_path: format is jsonl but '
            f'path is {reference!r}'
        )

    available = {
        PurePosixPath(path.replace('\\', '/')).as_posix()
        for path in available_files
    }
    if resolved_text not in available:
        warnings.append(
            'detailed_evaluation_results.file_path: referenced companion '
            f'{resolved_text!r} was not found in the dataset or this batch'
        )
    return warnings


def check_score_metadata(data: dict[str, Any]) -> list[str]:
    """Warn on absent/invalid score metadata and inconsistent bounds."""
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
        if not isinstance(score_type, str) or not score_type.strip():
            warnings.append(
                f'evaluation_results[{index}].metric_config: missing or invalid '
                "'score_type'"
            )

        raw_lo = metric.get('min_score')
        raw_hi = metric.get('max_score')
        lo = _metric_bound(raw_lo)
        hi = _metric_bound(raw_hi)
        for key, value in (('min_score', lo), ('max_score', hi)):
            if value is None:
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


def check_nonempty_evaluation_results(data: dict[str, Any]) -> list[str]:
    """Reject aggregate documents that contain no evaluation result records."""
    results = data.get('evaluation_results')
    if isinstance(results, list) and not results:
        return ['evaluation_results: expected at least one evaluation result']
    return []


def check_explicit_unknown_metadata(data: dict[str, Any]) -> list[str]:
    """Require stable descriptive metadata or the literal ``unknown``.

    This policy intentionally lives above the generated schema. Descriptive
    provenance may use ``unknown``; identity-critical values are handled by a
    separate rule and may not use a missing-value fallback.
    """
    findings: list[str] = []

    def require_text(parent: Any, key: str, path: str) -> None:
        value = parent.get(key) if isinstance(parent, dict) else None
        if isinstance(value, str) and value.strip():
            return
        findings.append(
            f"{path}: missing or blank; use 'unknown' when unavailable"
        )

    source_metadata = data.get('source_metadata')
    require_text(source_metadata, 'source_name', 'source_metadata.source_name')
    require_text(
        source_metadata,
        'source_organization_name',
        'source_metadata.source_organization_name',
    )

    eval_library = data.get('eval_library')
    require_text(eval_library, 'name', 'eval_library.name')
    require_text(eval_library, 'version', 'eval_library.version')

    model_info = data.get('model_info')
    require_text(model_info, 'name', 'model_info.name')
    require_text(model_info, 'developer', 'model_info.developer')
    require_text(
        model_info, 'inference_platform', 'model_info.inference_platform'
    )
    inference_engine = (
        model_info.get('inference_engine')
        if isinstance(model_info, dict)
        else None
    )
    require_text(inference_engine, 'name', 'model_info.inference_engine.name')
    require_text(
        inference_engine, 'version', 'model_info.inference_engine.version'
    )
    return findings


def check_identity_fields(data: dict[str, Any]) -> list[str]:
    """Reject missing identity components instead of collapsing to unknown."""
    findings: list[str] = []
    model_info = data.get('model_info')
    model_id = model_info.get('id') if isinstance(model_info, dict) else None
    if not isinstance(model_id, str) or not model_id.strip():
        findings.append('model_info.id: missing or blank identity field')

    results = data.get('evaluation_results')
    if not isinstance(results, list):
        return findings
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        evaluation_name = result.get('evaluation_name')
        if not isinstance(evaluation_name, str) or not evaluation_name.strip():
            findings.append(
                f'evaluation_results[{index}].evaluation_name: missing or blank identity field'
            )
        source_data = result.get('source_data')
        dataset_name = (
            source_data.get('dataset_name')
            if isinstance(source_data, dict)
            else None
        )
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            findings.append(
                f'evaluation_results[{index}].source_data.dataset_name: '
                'missing or blank identity field'
            )
        metric = result.get('metric_config')
        metric_id = (
            metric.get('metric_id') if isinstance(metric, dict) else None
        )
        metric_name = (
            metric.get('metric_name') if isinstance(metric, dict) else None
        )
        if not any(
            isinstance(value, str) and value.strip()
            for value in (metric_id, metric_name)
        ):
            findings.append(
                f'evaluation_results[{index}].metric_config: requires a '
                'non-blank metric_id or metric_name'
            )
    return findings


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


def _hf_dataset_exists(
    api: Any, repo_id: str
) -> tuple[bool | None, str | None]:
    """Return (exists, error); exists is None when verification failed."""
    key = ('dataset', repo_id)
    if key in _existence_cache:
        return _existence_cache[key]
    try:
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
    """Require independent deployment-control and weight-availability axes.

    ``deployment_type`` describes who controlled the inference deployment;
    ``model_availability`` describes whether model weights are available.
    Neither value constrains the other. ``api`` is retained in the signature
    for compatibility with callers that supply validation context, but this
    rule deliberately performs no provider-specific existence check.
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


def check_evaluator_provenance_consistency(
    data: dict[str, Any],
) -> list[str]:
    """Verify optional score-level provenance against the aggregate grouping.

    Adapters may preserve their inferred evaluator relationship and inference
    reason in ``score_details.details``. When they do, those two fields form an
    auditable contract: the inferred relationship must be valid and must match
    the file-level relationship used to group the evaluation results.

    Records without these optional fields remain valid. The validator does not
    repeat adapter-specific URL or organization inference.
    """
    findings: list[str] = []
    source_metadata = data.get('source_metadata')
    aggregate_relationship = (
        source_metadata.get('evaluator_relationship')
        if isinstance(source_metadata, dict)
        else None
    )
    results = data.get('evaluation_results')
    if not isinstance(results, list):
        return findings

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        score_details = result.get('score_details')
        details = (
            score_details.get('details')
            if isinstance(score_details, dict)
            else None
        )
        if not isinstance(details, dict):
            continue

        inferred = details.get('inferred_evaluator_relationship')
        reason = details.get('relationship_inference_reason')
        prefix = f'evaluation_results[{index}].score_details.details'
        if inferred is None and reason is None:
            continue
        if inferred is None:
            findings.append(
                f'{prefix}.inferred_evaluator_relationship: required when '
                'relationship_inference_reason is present'
            )
            continue
        if inferred not in _EVALUATOR_RELATIONSHIP_TYPES:
            findings.append(
                f'{prefix}.inferred_evaluator_relationship: expected one of '
                f'{list(_EVALUATOR_RELATIONSHIP_TYPES)}, got {inferred!r}'
            )
            continue
        if not isinstance(reason, str) or not reason.strip():
            findings.append(
                f'{prefix}.relationship_inference_reason: required when '
                'inferred_evaluator_relationship is present'
            )
        if inferred != aggregate_relationship:
            findings.append(
                f'{prefix}.inferred_evaluator_relationship: {inferred!r} '
                'does not match '
                f'source_metadata.evaluator_relationship '
                f'{aggregate_relationship!r}'
            )
    return findings


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
                exists, error = _hf_dataset_exists(api, repo)
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
            details = source_data.get('additional_details')
            stable_identity = any(
                isinstance(source_data.get(key), str)
                and source_data[key].strip()
                for key in ('source_id', 'source_version')
            ) or (
                isinstance(details, dict)
                and any(
                    isinstance(details.get(key), str) and details[key].strip()
                    for key in (
                        'source_id',
                        'source_version',
                        'source_url',
                        'url',
                    )
                )
            )
            if not stable_identity:
                other_count += 1

    if other_count:
        warnings.append(
            f"{other_count} evaluation_results use dataset source_type 'other' "
            '(no stable source ID, version, or URL provenance)'
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


def _aggregate_check_nonempty_results(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_nonempty_evaluation_results(data)


def _aggregate_check_explicit_unknown_metadata(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_explicit_unknown_metadata(data)


def _aggregate_check_identity_fields(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_identity_fields(data)


def _aggregate_check_integer_counts(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_integer_counts(data)


def _aggregate_check_model_deployment(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_model_deployment(data, context.hf_api)


def _aggregate_check_evaluator_provenance_consistency(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_evaluator_provenance_consistency(data)


def _aggregate_check_dataset_provenance(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    if data is None:
        return []
    return check_dataset_provenance(data, context.hf_api)


def _aggregate_check_required_dataset_provenance(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    """Return provenance failures that make an HF-backed record unverifiable."""
    return [
        message
        for message in _aggregate_check_dataset_provenance(context, data)
        if "dataset source_type 'other'" not in message
    ]


def _aggregate_check_advisory_dataset_provenance(
    context: ValidationContext, data: dict[str, Any] | None
) -> list[str]:
    """Return allowed-but-weak private/custom provenance findings."""
    return [
        message
        for message in _aggregate_check_dataset_provenance(context, data)
        if "dataset source_type 'other'" in message
    ]


REGISTERED_CHECKS: tuple[ValidationCheck, ...] = (
    ValidationCheck('path structure', 'file', 'error', _file_check_path),
    ValidationCheck(
        'companion file', 'aggregate', 'error', _aggregate_check_companion
    ),
    ValidationCheck(
        'nonempty evaluation results',
        'aggregate',
        'error',
        _aggregate_check_nonempty_results,
    ),
    ValidationCheck(
        'explicit unknown metadata',
        'aggregate',
        'error',
        _aggregate_check_explicit_unknown_metadata,
    ),
    ValidationCheck(
        'identity fields',
        'aggregate',
        'error',
        _aggregate_check_identity_fields,
    ),
    ValidationCheck(
        'score metadata', 'aggregate', 'error', _aggregate_check_score_metadata
    ),
    ValidationCheck(
        'integer counts', 'aggregate', 'error', _aggregate_check_integer_counts
    ),
    ValidationCheck(
        'model deployment',
        'aggregate',
        'error',
        _aggregate_check_model_deployment,
    ),
    ValidationCheck(
        'evaluator provenance consistency',
        'aggregate',
        'error',
        _aggregate_check_evaluator_provenance_consistency,
    ),
    ValidationCheck(
        'required dataset provenance',
        'aggregate',
        'error',
        _aggregate_check_required_dataset_provenance,
    ),
    ValidationCheck(
        'advisory dataset provenance',
        'aggregate',
        'warning',
        _aggregate_check_advisory_dataset_provenance,
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
        context = ValidationContext(
            local_path=file_path,
            repo_path=repo_path,
            available_files=available_files,
            hf_api=hf_api,
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

    if report.line_count == 0:
        report.valid = False
        report.errors.append(
            {
                'loc': '(file)',
                'msg': 'Instance-level JSONL must contain at least one record',
                'type': 'empty_instance_file',
            }
        )

    if run_semantic_checks:
        context = ValidationContext(
            local_path=file_path,
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
