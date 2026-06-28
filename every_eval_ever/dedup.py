"""Semantic duplicate detection for Every Eval Ever aggregate JSON files."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

MANIFEST_PATH = 'manifest.json'
DEFAULT_DATASET_REPO_ID = 'evaleval/EEE_datastore'
_QUANT_DECIMALS = 6
_WS_RE = re.compile(r'\s+')


class ManifestError(RuntimeError):
    """Raised when manifest.json cannot be safely loaded or used."""


def _norm_str(value: Any) -> str | None:
    if value is None:
        return None
    text = _WS_RE.sub(' ', str(value).strip().lower())
    return text.strip('"\'.,;:!?()[]{} ') or None


def _quantize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return repr(round(float(value), _QUANT_DECIMALS))
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


def _source_identity(source_data: Any) -> dict[str, Any]:
    if not isinstance(source_data, dict):
        return {}
    identity = {
        'type': _norm_str(source_data.get('source_type')),
        'name': _norm_str(source_data.get('dataset_name')),
    }
    if source_data.get('source_type') == 'hf_dataset':
        repo = _norm_str(source_data.get('hf_repo'))
        if repo:
            identity['hf'] = repo
    return identity


def _metric_identity(metric_config: Any) -> dict[str, Any]:
    if not isinstance(metric_config, dict):
        return {}
    metric_id = metric_config.get('metric_id') or metric_config.get(
        'metric_name'
    )
    return {
        'id': _norm_str(metric_id),
        'kind': _norm_str(metric_config.get('metric_kind')),
        'params': _canon(metric_config.get('metric_parameters') or {}),
        'score_type': _norm_str(metric_config.get('score_type')),
        'unit': _norm_str(metric_config.get('metric_unit')),
        'lower_is_better': metric_config.get('lower_is_better'),
    }


def _generation_config_identity(
    generation_config: Any,
) -> dict[str, Any] | None:
    if not isinstance(generation_config, dict):
        return None
    args = generation_config.get('generation_args')
    if not isinstance(args, dict):
        return None
    plan = args.get('eval_plan') or {}
    limits = args.get('eval_limits') or {}
    return {
        'temp': _quantize(args.get('temperature')),
        'top_p': _quantize(args.get('top_p')),
        'top_k': args.get('top_k'),
        'max_tokens': args.get('max_tokens'),
        'reasoning': args.get('reasoning'),
        'plan': _norm_str(plan.get('name')) if isinstance(plan, dict) else None,
        'time_limit': (
            limits.get('time_limit') if isinstance(limits, dict) else None
        ),
        'msg_limit': (
            limits.get('message_limit') if isinstance(limits, dict) else None
        ),
        'token_limit': (
            limits.get('token_limit') if isinstance(limits, dict) else None
        ),
        'max_attempts': args.get('max_attempts'),
    }


def _result_identity(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    score_details = result.get('score_details') or {}
    return {
        'eval': _norm_str(result.get('evaluation_name')),
        'src': _source_identity(result.get('source_data')),
        'metric': _metric_identity(result.get('metric_config')),
        'score': _quantize(score_details.get('score')),
        'gen': _generation_config_identity(result.get('generation_config')),
    }


def compute_aggregate_identity(data: dict[str, Any]) -> str:
    """Hash the normalized semantic identity of an aggregate eval record."""
    model_info = data.get('model_info') or {}
    eval_library = data.get('eval_library') or {}
    raw_results = data.get('evaluation_results') or []
    identity = {
        'model': _norm_str(model_info.get('id')),
        'lib': _norm_str(eval_library.get('name')),
        'results': [
            result
            for result in (_result_identity(item) for item in raw_results)
            if result is not None
        ],
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
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            'dedup fingerprint requires aggregate JSON content'
        ) from exc
    if isinstance(data, dict) and isinstance(
        data.get('evaluation_results'), list
    ):
        return compute_aggregate_identity(data)
    raise ValueError('dedup fingerprint requires aggregate JSON content')


def compute_file_fingerprint(local_path: str | Path) -> str:
    with Path(local_path).open('rb') as handle:
        return compute_fingerprint(handle.read())


def collection_key(file_path: str) -> str:
    """Return the datastore collection root for scoped duplicate comparison."""
    parts = file_path.split('/')
    if len(parts) >= 2 and parts[0] == 'data' and parts[1]:
        return f'data/{parts[1]}'
    return ''


@dataclass
class DedupResult:
    """Deduplication result for one aggregate file."""

    file_path: str
    fingerprint: str
    duplicate_of: str | None = None


@dataclass
class DedupReport:
    """Aggregated deduplication report."""

    results: list[DedupResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_duplicates(self) -> bool:
        return any(result.duplicate_of is not None for result in self.results)


def load_manifest(
    api: HfApi,
    *,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    manifest_path: str = MANIFEST_PATH,
    revision: str = 'main',
) -> dict[str, Any]:
    """Download and validate datastore manifest.json from Hugging Face."""
    try:
        manifest_file = hf_hub_download(
            repo_id=dataset_repo_id,
            filename=manifest_path,
            repo_type='dataset',
            revision=revision,
        )
        with Path(manifest_file).open(encoding='utf-8') as handle:
            manifest = json.load(handle)
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
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise ManifestError(
            f"{manifest_path} must contain an object field 'files'"
        )
    for path, entry in files.items():
        if not isinstance(path, str) or not isinstance(entry, dict):
            raise ManifestError(
                f'{manifest_path} has invalid file entry {path!r}'
            )
        if not isinstance(entry.get('fingerprint'), str):
            raise ManifestError(
                f'{manifest_path} entry {path!r} is missing fingerprint'
            )


def build_dedup_report(
    file_fingerprints: dict[str, str],
    manifest: dict[str, Any],
) -> DedupReport:
    """Compare candidate fingerprints against manifest and same-batch files."""
    validate_manifest(manifest)
    report = DedupReport()
    manifest_files = manifest.get('files', {})
    fingerprint_to_path: dict[tuple[str, str], str] = {
        (collection_key(path), entry['fingerprint']): path
        for path, entry in manifest_files.items()
        if isinstance(entry, dict) and isinstance(entry.get('fingerprint'), str)
    }

    for file_path in sorted(file_fingerprints):
        fingerprint = file_fingerprints[file_path]
        result = DedupResult(file_path=file_path, fingerprint=fingerprint)
        key = (collection_key(file_path), fingerprint)
        existing_path = fingerprint_to_path.get(key)
        if existing_path is not None and existing_path != file_path:
            result.duplicate_of = existing_path
        report.results.append(result)
        fingerprint_to_path.setdefault(key, file_path)

    return report


def check_duplicates(
    file_paths: list[str],
    local_paths: dict[str, str | Path],
    manifest: dict[str, Any],
) -> DedupReport:
    """Compute fingerprints for aggregate JSON files and compare to manifest."""
    file_fingerprints: dict[str, str] = {}
    warnings: list[str] = []
    for file_path in sorted(file_paths):
        if not file_path.endswith('.json'):
            continue
        local_path = local_paths.get(file_path)
        if local_path is None:
            warnings.append(
                f'Duplicate check skipped {file_path}: local path was not provided'
            )
            continue
        try:
            file_fingerprints[file_path] = compute_file_fingerprint(local_path)
        except Exception as exc:
            warnings.append(
                f'Duplicate check skipped {file_path}: '
                f'{type(exc).__name__}: {exc}'
            )

    report = build_dedup_report(file_fingerprints, manifest)
    report.warnings.extend(warnings)
    return report
