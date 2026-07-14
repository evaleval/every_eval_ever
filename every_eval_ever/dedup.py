"""Semantic duplicate detection for Every Eval Ever aggregate JSON files."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from every_eval_ever.json_utils import StrictJSONError, strict_json_loads

MANIFEST_PATH = 'manifest.json'
DEFAULT_DATASET_REPO_ID = 'evaleval/EEE_datastore'
FINGERPRINT_VERSION = 'eee-semantic-v3'
INSTANCE_DEDUP_UNSUPPORTED = (
    'instance-level JSONL deduplication is not supported yet'
)
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
    text = ' '.join(str(value).casefold().split())
    return text or None


def _quantize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('dedup identity cannot contain non-finite numbers')
        if value == 0:
            return '0'
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return value


def _canon(obj: Any) -> Any:
    """Canonicalize JSON containers without recursive Python calls."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return _quantize(obj)
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, (dict, list)):
        return obj

    root: dict[str, Any] | list[Any] = {} if isinstance(obj, dict) else []
    stack: list[
        tuple[dict[str, Any] | list[Any], dict[str, Any] | list[Any]]
    ] = [(obj, root)]
    while stack:
        source, target = stack.pop()
        items = (
            source.items() if isinstance(source, dict) else enumerate(source)
        )
        for key, value in items:
            if isinstance(value, dict):
                child: dict[str, Any] | list[Any] = {}
                if isinstance(target, list):
                    target.append(child)
                else:
                    target[key] = child
                stack.append((value, child))
            elif isinstance(value, list):
                child = []
                if isinstance(target, list):
                    target.append(child)
                else:
                    target[key] = child
                stack.append((value, child))
            elif isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                canonical_value = _quantize(value)
                if isinstance(target, list):
                    target.append(canonical_value)
                else:
                    target[key] = canonical_value
            else:
                if isinstance(target, list):
                    target.append(value)
                else:
                    target[key] = value
    return root


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
    metric_name = _optional_field(metric_config, 'metric_name')
    if metric_name is not None:
        metric_name = _require_string(
            metric_name, 'evaluation result.metric_config.metric_name'
        )
    if metric_id is None and metric_name is None:
        raise ValueError(
            'evaluation result.metric_config requires metric_id or metric_name'
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
        'id': _norm_str(metric_id if metric_id is not None else metric_name),
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
    """Return the complete normalized generation configuration."""
    if generation_config is None:
        return None
    generation_config = _require_mapping(
        generation_config, 'evaluation result.generation_config'
    )
    return _canon(generation_config)


def _model_identity(model_info: dict[str, Any]) -> dict[str, str | None]:
    """Return model identity including weights and execution environment."""
    raw_details = _optional_field(model_info, 'additional_details')
    details = (
        {}
        if raw_details is None
        else _require_mapping(
            raw_details, 'aggregate.model_info.additional_details'
        )
    )

    def optional_detail(key: str) -> str | None:
        value = _optional_field(details, key)
        if value is None:
            return None
        return _norm_str(
            _require_string(
                value, f'aggregate.model_info.additional_details.{key}'
            )
        )

    raw_engine = _optional_field(model_info, 'inference_engine')
    engine = (
        {}
        if raw_engine is None
        else _require_mapping(
            raw_engine, 'aggregate.model_info.inference_engine'
        )
    )
    return {
        'id': _norm_str(
            _require_string(
                _require_field(model_info, 'id', 'aggregate.model_info'),
                'aggregate.model_info.id',
            )
        ),
        'revision': optional_detail('model_revision'),
        'precision': optional_detail('precision'),
        'deployment_type': optional_detail('deployment_type'),
        'inference_platform': _norm_str(
            _optional_field(model_info, 'inference_platform')
        ),
        'inference_engine_name': _norm_str(_optional_field(engine, 'name')),
        'inference_engine_version': _norm_str(
            _optional_field(engine, 'version')
        ),
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
        'generation_config': _generation_config_identity(
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
    if not raw_results:
        raise ValueError(
            'aggregate evaluation_results must contain at least one result'
        )
    result_items: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError(
                'aggregate evaluation_results entries must be objects'
            )
        result_items.append(item)
    identity = {
        'model': _model_identity(model_info),
        'evaluation_library': {
            'name': _norm_str(
                _require_string(
                    _require_field(
                        eval_library, 'name', 'aggregate.eval_library'
                    ),
                    'aggregate.eval_library.name',
                )
            ),
            'version': _norm_str(
                _require_string(
                    _require_field(
                        eval_library, 'version', 'aggregate.eval_library'
                    ),
                    'aggregate.eval_library.version',
                )
            ),
        },
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
    """Deduplication result or explicit unsupported-file disposition."""

    file_path: str
    fingerprint: str | None
    duplicate_of: str | None = None
    matched_manifest_path: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class DedupReport:
    """Aggregated deduplication report."""

    results: Sequence[DedupResult] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class DedupSelection:
    """Package-owned acceptance decision for a candidate file batch."""

    accepted_paths: tuple[str, ...]
    duplicate_results: tuple[DedupResult, ...]
    skipped_results: tuple[DedupResult, ...]


def _candidate_kind(file_path: str) -> str:
    """Classify a datastore candidate while enforcing the current data/ scope."""
    collection_key(file_path)
    if file_path.endswith('.jsonl'):
        return 'instance'
    if file_path.endswith('.json'):
        return 'aggregate'
    raise ValueError(
        'Duplicate check only accepts aggregate .json or instance .jsonl '
        f'paths under data/: {file_path!r}'
    )


class DedupSession:
    """One manifest-backed duplicate comparison flow for all callers.

    A session can consume multiple batches while retaining intra-request
    fingerprints. Aggregate JSON files are fingerprinted and compared globally.
    Instance JSONL files receive an explicit skipped result until an instance
    identity contract exists.
    """

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self._manifest = dict(manifest)
        validate_manifest(self._manifest)
        self._manifest_fingerprint_to_paths: dict[str, list[str]] = {}
        for path, entry in sorted(self._manifest['files'].items()):
            self._manifest_fingerprint_to_paths.setdefault(
                entry['fingerprint'], []
            ).append(path)
        self._candidate_fingerprint_to_path: dict[str, str] = {}
        self._results: list[DedupResult] = []
        self._seen_paths: set[str] = set()

    @property
    def report(self) -> DedupReport:
        return DedupReport(results=tuple(self._results))

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            result.file_path: result.fingerprint
            for result in self._results
            if result.fingerprint is not None
        }

    def _reserve_path(self, file_path: str) -> None:
        if file_path in self._seen_paths:
            raise ValueError(
                f'Duplicate check received candidate path more than once: {file_path!r}'
            )
        self._seen_paths.add(file_path)

    def _add_fingerprint(self, file_path: str, fingerprint: str) -> DedupResult:
        if _candidate_kind(file_path) != 'aggregate':
            raise ValueError(
                f'Only aggregate JSON can have a semantic fingerprint: {file_path!r}'
            )
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(
            fingerprint
        ):
            raise ValueError(
                f'candidate {file_path!r} has invalid SHA-256 fingerprint'
            )

        manifest_matches = self._manifest_fingerprint_to_paths.get(
            fingerprint, []
        )
        matched_manifest_path = next(
            (path for path in manifest_matches if path != file_path),
            file_path if file_path in manifest_matches else None,
        )
        batch_match = self._candidate_fingerprint_to_path.get(fingerprint)
        duplicate_of = None
        if (
            matched_manifest_path is not None
            and matched_manifest_path != file_path
        ):
            duplicate_of = matched_manifest_path
        elif matched_manifest_path is None and batch_match is not None:
            duplicate_of = batch_match

        result = DedupResult(
            file_path=file_path,
            fingerprint=fingerprint,
            duplicate_of=duplicate_of,
            matched_manifest_path=matched_manifest_path,
        )
        self._results.append(result)
        self._candidate_fingerprint_to_path.setdefault(fingerprint, file_path)
        return result

    def add_fingerprints(
        self, file_fingerprints: Mapping[str, str]
    ) -> DedupReport:
        """Add already-computed aggregate fingerprints to this session."""
        added: list[DedupResult] = []
        for file_path, fingerprint in sorted(file_fingerprints.items()):
            self._reserve_path(file_path)
            added.append(self._add_fingerprint(file_path, fingerprint))
        return DedupReport(results=tuple(added))

    def check_files(
        self,
        file_paths: Sequence[str],
        local_paths: Mapping[str, str | Path],
    ) -> DedupReport:
        """Classify, fingerprint, and compare one candidate batch."""
        added: list[DedupResult] = []
        for file_path in sorted(file_paths):
            self._reserve_path(file_path)
            if _candidate_kind(file_path) == 'instance':
                result = DedupResult(
                    file_path=file_path,
                    fingerprint=None,
                    skipped_reason=INSTANCE_DEDUP_UNSUPPORTED,
                )
                self._results.append(result)
                added.append(result)
                continue

            local_path = local_paths.get(file_path)
            if local_path is None:
                raise ValueError(
                    f'Duplicate check requires a local path for {file_path}'
                )
            try:
                fingerprint = compute_file_fingerprint(local_path)
            except Exception as exc:
                raise ValueError(
                    f'Duplicate check failed for {file_path}: '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
            added.append(self._add_fingerprint(file_path, fingerprint))
        return DedupReport(results=tuple(added))


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


def build_dedup_report(
    file_fingerprints: dict[str, str],
    manifest: dict[str, Any],
) -> DedupReport:
    """Compare aggregate fingerprints through the shared session flow."""
    session = DedupSession(manifest)
    session.add_fingerprints(file_fingerprints)
    return session.report


def check_duplicates(
    file_paths: list[str],
    local_paths: dict[str, str | Path],
    manifest: dict[str, Any],
) -> DedupReport:
    """Run the shared manifest-backed flow for aggregate and instance files."""
    session = DedupSession(manifest)
    session.check_files(file_paths, local_paths)
    return session.report


def select_unique_files(
    file_paths: Sequence[str],
    local_paths: Mapping[str, str | Path],
    manifest: dict[str, Any],
) -> DedupSelection:
    """Return the files accepted by the canonical manifest-backed flow.

    Aggregate candidates already represented by the manifest, or duplicated
    elsewhere in the same batch, are rejected. Explicitly unsupported file
    kinds such as instance JSONL are reported separately and are not silently
    treated as deduplicated.
    """
    report = check_duplicates(list(file_paths), dict(local_paths), manifest)
    accepted: list[str] = []
    duplicates: list[DedupResult] = []
    skipped: list[DedupResult] = []
    for result in report.results:
        if result.skipped_reason is not None:
            skipped.append(result)
        elif (
            result.duplicate_of is not None
            or result.matched_manifest_path is not None
        ):
            duplicates.append(result)
        else:
            accepted.append(result.file_path)
    return DedupSelection(
        accepted_paths=tuple(accepted),
        duplicate_results=tuple(duplicates),
        skipped_results=tuple(skipped),
    )


def empty_manifest() -> dict[str, Any]:
    """Return a new manifest seed for semantic duplicate detection."""
    return {
        'fingerprint_version': FINGERPRINT_VERSION,
        'files': {},
    }
