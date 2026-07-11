"""Semantic duplicate detection for Every Eval Ever aggregate JSON files."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from every_eval_ever.json_utils import StrictJSONError, strict_json_loads
from every_eval_ever.source_index import SourceIndex

MANIFEST_PATH = 'manifest.json'
DEFAULT_DATASET_REPO_ID = 'evaleval/EEE_datastore'
FINGERPRINT_VERSION = 'eee-semantic-v2'
_QUANT_DECIMALS = 6
_QUANT_FORMAT = f'.{_QUANT_DECIMALS}f'
_TRIM_CHARS = '"\'.,;:!?()[]{} '
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


class ManifestError(RuntimeError):
    """Raised when manifest.json cannot be safely loaded or used."""


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    return value


def _require_field(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f'{context} is missing required field {key!r}')
    return mapping[key]


def _optional_field(mapping: dict[str, Any], key: str) -> Any:
    return mapping[key] if key in mapping else None


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{context} must be a non-empty string')
    return value


def _require_number(value: Any, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{context} must be a number')
    if not math.isfinite(value):
        raise ValueError(f'{context} must be finite')
    return value


def _norm_str(value: Any) -> str | None:
    if value is None:
        return None
    text = ' '.join(str(value).lower().split())
    return text.strip(_TRIM_CHARS) or None


def _quantize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('dedup identity cannot contain non-finite numbers')
        text = format(value, _QUANT_FORMAT).rstrip('0').rstrip('.')
        return '0' if text in {'', '-0'} else text
    return value


def _canon(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return _quantize(obj)
    if isinstance(obj, str):
        return _norm_str(obj)
    if isinstance(obj, dict):
        return {key: _canon(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_canon(value) for value in obj]
    return obj


def _norm_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('URL source identity requires non-empty URL strings')
    raw = value.strip()
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            f'URL source identity requires an absolute URL: {raw!r}'
        )
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def _source_identity(source_data: Any) -> dict[str, Any]:
    source_data = _require_mapping(source_data, 'aggregate source_data')
    source_type = _require_string(
        _require_field(source_data, 'source_type', 'aggregate source_data'),
        'aggregate source_data.source_type',
    )
    dataset_name = _require_string(
        _require_field(source_data, 'dataset_name', 'aggregate source_data'),
        'aggregate source_data.dataset_name',
    )
    identity = {
        'type': _norm_str(source_type),
        'name': _norm_str(dataset_name),
    }
    if source_type == 'url':
        urls = _require_field(source_data, 'url', 'URL source_data')
        if not isinstance(urls, list) or not urls:
            raise ValueError(
                'URL source identity requires a non-empty url list'
            )
        identity['urls'] = sorted(_norm_url(url) for url in urls)
    elif source_type == 'hf_dataset':
        repo = _norm_str(
            _require_string(
                _require_field(source_data, 'hf_repo', 'HF source_data'),
                'HF source_data.hf_repo',
            )
        )
        if repo is None:
            raise ValueError('HF source_data.hf_repo has no canonical content')
        identity.update(
            {
                'hf': repo,
                'config': _norm_str(_optional_field(source_data, 'hf_config')),
                'revision': _norm_str(
                    _optional_field(source_data, 'hf_revision')
                ),
                'split': _norm_str(_optional_field(source_data, 'hf_split')),
            }
        )
        sample_ids = _optional_field(source_data, 'sample_ids')
        if sample_ids is not None:
            if not isinstance(sample_ids, list) or not all(
                isinstance(item, str) for item in sample_ids
            ):
                raise ValueError(
                    'HF source sample_ids must be a list of strings'
                )
            normalized_sample_ids = [_norm_str(item) for item in sample_ids]
            if any(item is None for item in normalized_sample_ids):
                raise ValueError('HF source sample_ids must not be blank')
            identity['sample_ids'] = sorted(normalized_sample_ids)
    elif source_type == 'other':
        raw_details = _optional_field(source_data, 'additional_details')
        details = (
            None
            if raw_details is None
            else _require_mapping(
                raw_details, 'other source additional_details'
            )
        )
        source_id = _optional_field(source_data, 'source_id')
        if source_id is None and details is not None:
            source_id = _optional_field(details, 'source_id')
        source_version = _optional_field(source_data, 'source_version')
        if source_version is None and details is not None:
            source_version = _optional_field(details, 'source_version')
        source_url = (
            _optional_field(details, 'source_url')
            if details is not None
            else None
        )
        if source_url is None and details is not None:
            source_url = _optional_field(details, 'url')
        if source_id is not None:
            identity['source_id'] = _norm_str(
                _require_string(source_id, 'other source source_id')
            )
        if source_version is not None:
            identity['source_version'] = _norm_str(
                _require_string(source_version, 'other source source_version')
            )
        if source_url is not None:
            identity['source_url'] = _norm_url(source_url)
        if not any(
            key in identity
            for key in ('source_id', 'source_version', 'source_url')
        ):
            raise ValueError(
                "source_type 'other' requires source_id, source_version, or "
                'additional_details.source_url for safe duplicate detection'
            )
    else:
        raise ValueError(f'unsupported source_type for dedup: {source_type!r}')
    return identity


def _metric_identity(metric_config: Any) -> dict[str, Any]:
    metric_config = _require_mapping(
        metric_config, 'evaluation result.metric_config'
    )
    metric_id = _optional_field(metric_config, 'metric_id')
    if metric_id is None:
        metric_id = _optional_field(metric_config, 'metric_name')
    if metric_id is not None:
        metric_id = _require_string(
            metric_id, 'evaluation result.metric_config metric identifier'
        )
    raw_parameters = _optional_field(metric_config, 'metric_parameters')
    parameters = (
        None
        if raw_parameters is None
        else _require_mapping(
            raw_parameters, 'evaluation result.metric_config.metric_parameters'
        )
    )
    return {
        'id': _norm_str(metric_id),
        'kind': _norm_str(_optional_field(metric_config, 'metric_kind')),
        'params': _canon(parameters),
        'score_type': _norm_str(_optional_field(metric_config, 'score_type')),
        'unit': _norm_str(_optional_field(metric_config, 'metric_unit')),
        'lower_is_better': _require_field(
            metric_config,
            'lower_is_better',
            'evaluation result.metric_config',
        ),
    }


def _generation_config_identity(
    generation_config: Any,
) -> dict[str, Any] | None:
    if generation_config is None:
        return None
    generation_config = _require_mapping(
        generation_config, 'evaluation result.generation_config'
    )
    args = _optional_field(generation_config, 'generation_args')
    if args is None:
        return {'present': True, 'args': None}
    args = _require_mapping(
        args, 'evaluation result.generation_config.generation_args'
    )
    raw_plan = _optional_field(args, 'eval_plan')
    plan = (
        None
        if raw_plan is None
        else _require_mapping(raw_plan, 'generation_args.eval_plan')
    )
    raw_limits = _optional_field(args, 'eval_limits')
    limits = (
        None
        if raw_limits is None
        else _require_mapping(raw_limits, 'generation_args.eval_limits')
    )
    return {
        'temp': _quantize(_optional_field(args, 'temperature')),
        'top_p': _quantize(_optional_field(args, 'top_p')),
        'top_k': _optional_field(args, 'top_k'),
        'max_tokens': _optional_field(args, 'max_tokens'),
        'reasoning': _optional_field(args, 'reasoning'),
        'plan': (
            _norm_str(_optional_field(plan, 'name'))
            if plan is not None
            else None
        ),
        'time_limit': (
            _optional_field(limits, 'time_limit')
            if limits is not None
            else None
        ),
        'msg_limit': (
            _optional_field(limits, 'message_limit')
            if limits is not None
            else None
        ),
        'token_limit': (
            _optional_field(limits, 'token_limit')
            if limits is not None
            else None
        ),
        'max_attempts': _optional_field(args, 'max_attempts'),
    }


def _result_identity(result: dict[str, Any]) -> dict[str, Any]:
    score_details = _require_mapping(
        _require_field(result, 'score_details', 'evaluation result'),
        'evaluation result.score_details',
    )
    return {
        'eval': _norm_str(
            _require_string(
                _require_field(result, 'evaluation_name', 'evaluation result'),
                'evaluation result.evaluation_name',
            )
        ),
        'src': _source_identity(
            _require_field(result, 'source_data', 'evaluation result')
        ),
        'metric': _metric_identity(
            _require_field(result, 'metric_config', 'evaluation result')
        ),
        'score': _quantize(
            _require_number(
                _require_field(
                    score_details, 'score', 'evaluation result.score_details'
                ),
                'evaluation result.score_details.score',
            )
        ),
        'gen': _generation_config_identity(
            _optional_field(result, 'generation_config')
        ),
    }


def compute_aggregate_identity(data: dict[str, Any]) -> str:
    """Hash the normalized semantic identity of an aggregate eval record."""
    model_info = _require_mapping(
        _require_field(data, 'model_info', 'aggregate'), 'aggregate.model_info'
    )
    eval_library = _require_mapping(
        _require_field(data, 'eval_library', 'aggregate'),
        'aggregate.eval_library',
    )
    raw_results = _require_field(data, 'evaluation_results', 'aggregate')
    if not isinstance(raw_results, list):
        raise ValueError('aggregate evaluation_results must be a list')
    result_items: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError(
                'aggregate evaluation_results entries must be objects'
            )
        result_items.append(item)
    identity = {
        'model': _norm_str(
            _require_string(
                _require_field(model_info, 'id', 'aggregate.model_info'),
                'aggregate.model_info.id',
            )
        ),
        'lib': _norm_str(
            _require_string(
                _require_field(eval_library, 'name', 'aggregate.eval_library'),
                'aggregate.eval_library.name',
            )
        ),
        'results': [_result_identity(item) for item in result_items],
    }
    canonical = _canon(identity)
    canonical['results'] = sorted(
        canonical['results'],
        key=lambda item: json.dumps(item, sort_keys=True),
    )
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_fingerprint(content: bytes) -> str:
    """Compute the semantic duplicate fingerprint for aggregate JSON bytes."""
    try:
        data = strict_json_loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError, StrictJSONError) as exc:
        raise ValueError(
            'dedup fingerprint requires aggregate JSON content'
        ) from exc
    if not isinstance(data, dict):
        raise ValueError('dedup fingerprint requires an aggregate JSON object')
    evaluation_results = _require_field(
        data, 'evaluation_results', 'aggregate JSON object'
    )
    if not isinstance(evaluation_results, list):
        raise ValueError('aggregate evaluation_results must be a list')
    return compute_aggregate_identity(data)


def compute_file_fingerprint(local_path: str | Path) -> str:
    with Path(local_path).open('rb') as handle:
        return compute_fingerprint(handle.read())


def collection_key(file_path: str) -> str:
    """Return the datastore collection root for scoped duplicate comparison."""
    parts = file_path.split('/')
    if len(parts) >= 2 and parts[0] == 'data' and parts[1]:
        return f'data/{parts[1]}'
    raise ValueError(
        'duplicate comparison requires a repository path under '
        f"'data/<collection>/': {file_path!r}"
    )


@dataclass(frozen=True)
class DedupResult:
    """Deduplication result for one aggregate file."""

    file_path: str
    fingerprint: str
    duplicate_of: str | None = None
    matched_manifest_path: str | None = None


@dataclass(frozen=True)
class DedupReport:
    """Aggregated deduplication report."""

    results: Sequence[DedupResult] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)


def load_manifest(
    api: Any = None,
    *,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    manifest_path: str = MANIFEST_PATH,
    revision: str = 'main',
) -> dict[str, Any]:
    """Download and validate datastore manifest.json from Hugging Face."""
    download_options: dict[str, Any] = {}
    if api is not None:
        if not hasattr(api, 'endpoint') or not hasattr(api, 'token'):
            raise TypeError('api must provide Hugging Face endpoint and token')
        download_options['endpoint'] = api.endpoint
        download_options['token'] = api.token
    try:
        manifest_file = hf_hub_download(
            repo_id=dataset_repo_id,
            filename=manifest_path,
            repo_type='dataset',
            revision=revision,
            **download_options,
        )
        manifest = strict_json_loads(
            Path(manifest_file).read_text(encoding='utf-8')
        )
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        raise ManifestError(
            f'{manifest_path} not found in {dataset_repo_id}'
        ) from exc
    except Exception as exc:
        raise ManifestError(
            f'Failed to load {manifest_path} from {dataset_repo_id}'
        ) from exc

    validate_manifest(manifest, manifest_path=manifest_path)
    return manifest


def validate_manifest(
    manifest: dict[str, Any], *, manifest_path: str = MANIFEST_PATH
) -> None:
    if not isinstance(manifest, dict):
        raise ManifestError(f'{manifest_path} must contain a JSON object')
    if 'fingerprint_version' not in manifest:
        raise ManifestError(
            f"{manifest_path} is missing required field 'fingerprint_version'"
        )
    if manifest['fingerprint_version'] != FINGERPRINT_VERSION:
        raise ManifestError(
            f'{manifest_path} fingerprint_version must be '
            f'{FINGERPRINT_VERSION!r}, got {manifest["fingerprint_version"]!r}'
        )
    if 'files' not in manifest:
        raise ManifestError(
            f"{manifest_path} is missing required field 'files'"
        )
    files = manifest['files']
    if not isinstance(files, dict):
        raise ManifestError(
            f"{manifest_path} must contain an object field 'files'"
        )
    for path, entry in files.items():
        if not isinstance(path, str) or not isinstance(entry, dict):
            raise ManifestError(
                f'{manifest_path} has invalid file entry {path!r}'
            )
        if not path.endswith('.json'):
            raise ManifestError(
                f'{manifest_path} file entry must reference aggregate JSON: {path!r}'
            )
        try:
            collection_key(path)
        except ValueError as exc:
            raise ManifestError(
                f'{manifest_path} has invalid repository path {path!r}: {exc}'
            ) from exc
        if 'fingerprint' not in entry or not isinstance(
            entry['fingerprint'], str
        ):
            raise ManifestError(
                f'{manifest_path} entry {path!r} is missing fingerprint'
            )
        if not _SHA256_RE.fullmatch(entry['fingerprint']):
            raise ManifestError(
                f'{manifest_path} entry {path!r} has invalid SHA-256 fingerprint'
            )
    if 'sources' in manifest:
        try:
            SourceIndex.from_manifest(manifest)
        except ValueError as exc:
            raise ManifestError(
                f'{manifest_path} has invalid pre-download source index: {exc}'
            ) from exc


def build_dedup_report(
    file_fingerprints: dict[str, str],
    manifest: dict[str, Any],
) -> DedupReport:
    """Compare candidate fingerprints against manifest and same-batch files."""
    validate_manifest(manifest)
    results: list[DedupResult] = []
    manifest_files: dict[str, dict[str, Any]] = manifest['files']
    manifest_fingerprint_to_path: dict[tuple[str, str], str] = {
        (collection_key(path), entry['fingerprint']): path
        for path, entry in manifest_files.items()
    }
    batch_fingerprint_to_path: dict[tuple[str, str], str] = {}

    for file_path in sorted(file_fingerprints):
        fingerprint = file_fingerprints[file_path]
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(
            fingerprint
        ):
            raise ValueError(
                f'candidate {file_path!r} has invalid SHA-256 fingerprint'
            )
        key = (collection_key(file_path), fingerprint)
        matched_manifest_path = manifest_fingerprint_to_path.get(key)
        batch_match = batch_fingerprint_to_path.get(key)
        duplicate_of = None
        if (
            matched_manifest_path is not None
            and matched_manifest_path != file_path
        ):
            duplicate_of = matched_manifest_path
        elif matched_manifest_path is None and batch_match is not None:
            duplicate_of = batch_match
        results.append(
            DedupResult(
                file_path=file_path,
                fingerprint=fingerprint,
                duplicate_of=duplicate_of,
                matched_manifest_path=matched_manifest_path,
            )
        )
        batch_fingerprint_to_path.setdefault(key, file_path)

    return DedupReport(results=tuple(results))


def check_duplicates(
    file_paths: list[str],
    local_paths: dict[str, str | Path],
    manifest: dict[str, Any],
) -> DedupReport:
    """Compute fingerprints for aggregate JSON files and compare to manifest."""
    file_fingerprints: dict[str, str] = {}
    for file_path in sorted(file_paths):
        if not file_path.endswith('.json'):
            raise ValueError(
                f'Duplicate check only accepts .json files: {file_path}'
            )
        local_path = local_paths.get(file_path)
        if local_path is None:
            raise ValueError(
                f'Duplicate check requires a local path for {file_path}'
            )
        try:
            file_fingerprints[file_path] = compute_file_fingerprint(local_path)
        except Exception as exc:
            raise ValueError(
                f'Duplicate check failed for {file_path}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

    report = build_dedup_report(file_fingerprints, manifest)
    return report


def empty_manifest() -> dict[str, Any]:
    """Return an explicit empty manifest for same-batch-only comparison."""
    return {
        'fingerprint_version': FINGERPRINT_VERSION,
        'files': {},
        'sources': {},
    }
