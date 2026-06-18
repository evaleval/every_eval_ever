from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import get_close_matches
from math import isfinite
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests
import yaml
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import EntryNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Column, Table
from rich.text import Text

from every_eval_ever.eval_types import EvaluationLog, EvaluationResult

SOURCE_NAME = 'EvalEval'
MANIFEST_VERSION = 1
DEFAULT_DATASTORE_REVISION = 'main'
DEFAULT_DATASTORE_REPO = 'evaleval/EEE_datastore'
DEFAULT_BENCHMARKS = ('gpqa', 'hle', 'mmlu_pro', 'gsm8k')
DEFAULT_PR_COMMIT_DESCRIPTION = (
    'Adds EvalEval Community Evals YAML entries with source backlinks to EEE '
    'aggregate records.\n\n'
    'Contributor: evaleval'
)
AUDIT_ERROR_STATUS = 'audit_error'


class HFEvalsError(RuntimeError):
    """Raised when HF Community Evals export cannot proceed safely."""


class ReviewProgress:
    def add_task(self, description: str, total: int | None = None) -> int:
        return 0

    def update(
        self,
        task_id: int,
        *,
        advance: int = 0,
        description: str | None = None,
        total: int | None = None,
    ) -> None:
        return None


class RichReviewProgress(ReviewProgress):
    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.rich_task_id: TaskID | None = None
        self.next_task_id = 0
        self.total_by_task: dict[int, int] = {}
        self.completed_by_task: dict[int, int] = {}
        self.active_task_id: int | None = None

    def add_task(self, description: str, total: int | None = None) -> int:
        self.next_task_id += 1
        task_id = self.next_task_id
        task_total = total or 0
        self.total_by_task[task_id] = task_total
        self.completed_by_task[task_id] = 0
        self.active_task_id = task_id
        if self.rich_task_id is None:
            self.rich_task_id = self.progress.add_task(
                description,
                total=task_total,
            )
        else:
            self.progress.update(
                self.rich_task_id,
                description=description,
                completed=0,
                total=task_total,
            )
        return task_id

    def update(
        self,
        task_id: int,
        *,
        advance: int = 0,
        description: str | None = None,
        total: int | None = None,
    ) -> None:
        if self.rich_task_id is None:
            self.rich_task_id = self.progress.add_task(
                description or 'Reviewing',
                total=0,
            )
        if total is not None:
            self.total_by_task[task_id] = total
        self.active_task_id = task_id
        self.completed_by_task[task_id] = (
            self.completed_by_task.get(task_id, 0) + advance
        )
        kwargs: dict[str, Any] = {
            'completed': self.completed_by_task[task_id],
            'total': self.total_by_task.get(task_id, 0),
        }
        if description is not None:
            kwargs['description'] = description
        self.progress.update(self.rich_task_id, **kwargs)


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_id: str
    task_id: str
    yaml_name: str
    dataset_aliases: tuple[str, ...] = ()
    preferred_metric_ids: tuple[str, ...] = ()


BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    'gpqa': BenchmarkConfig(
        'Idavidrein/gpqa',
        'diamond',
        'gpqa_diamond',
        ('reasoningMIA/gpqa_diamond',),
    ),
    'hle': BenchmarkConfig(
        'cais/hle',
        'default',
        'hle',
        preferred_metric_ids=('hle.accuracy', 'hle/accuracy'),
    ),
    'mmlu_pro': BenchmarkConfig(
        'TIGER-Lab/MMLU-Pro',
        'mmlu_pro',
        'mmlu_pro',
        preferred_metric_ids=(
            'mmlu_pro/overall',
            'mmlu-pro::chain-of-thought-correctness',
        ),
    ),
    'gsm8k': BenchmarkConfig('openai/gsm8k', 'gsm8k', 'gsm8k', ('gsm8k',)),
}
BENCHMARK_ALIASES = {
    'gpqa_diamond': 'gpqa',
}
HF_TIMEOUT_SECONDS = 10
OPEN_WEIGHT_MODEL_PREFIXES = ('openai/gpt-oss',)
API_ONLY_PROVIDER_PREFIXES = (
    'anthropic',
    'gemini',
    'grok',
    'mistral',
    'openai',
    'xai',
)
GPQA_SUBSET_NOTES = {
    'diamond': 'GPQA Diamond',
    'gpqa_diamond': 'GPQA Diamond',
    'main': 'GPQA chain-of-thought',
    'chain_of_thought': 'GPQA chain-of-thought',
    'cot': 'GPQA chain-of-thought',
}
EVAL_RESULT_PATH_FAMILIES = {
    'gpqa': (
        '.eval_results/gpqa_diamond.yaml',
        '.eval_results/gpqa-diamond.yaml',
        '.eval_results/gpqa.yaml',
    ),
    'gsm8k': ('.eval_results/gsm8k.yaml',),
    'hle': ('.eval_results/hle.yaml',),
    'mmlu_pro': (
        '.eval_results/mmlu_pro.yaml',
        '.eval_results/mmlu-pro.yaml',
    ),
}


def normalize_benchmark(value: str) -> str:
    return value.strip().lower().replace('-', '_')


def parse_benchmarks(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_BENCHMARKS)
    benchmarks = [
        BENCHMARK_ALIASES.get(normalize_benchmark(item), normalize_benchmark(item))
        for item in raw.split(',')
    ]
    benchmarks = [item for item in benchmarks if item]
    unknown = sorted(set(benchmarks) - set(BENCHMARK_CONFIGS))
    if unknown:
        raise HFEvalsError(f'Unsupported benchmark(s): {", ".join(unknown)}')
    if not benchmarks:
        raise HFEvalsError('At least one benchmark is required.')
    return benchmarks


def parse_datastore_locator(value: str) -> tuple[str, str | None]:
    raw = value.strip()
    if not raw:
        raise HFEvalsError('Datastore must be <hf_dataset_repo>[@<revision>].')
    if raw.count('@') > 1:
        raise HFEvalsError('Datastore must be <hf_dataset_repo>[@<revision>].')
    if '@' in raw:
        repo_id, revision = (
            part.strip().strip('/') for part in raw.split('@', 1)
        )
    else:
        repo_id = raw.strip().strip('/')
        revision = None
    if not repo_id or '/' not in repo_id:
        raise HFEvalsError('Datastore repo must look like org/dataset.')
    if revision is not None and not revision:
        raise HFEvalsError('Datastore revision must not be empty.')
    return repo_id, revision


def resolve_datastore_locator(value: str, *, api: HfApi) -> tuple[str, str]:
    repo_id, revision = parse_datastore_locator(value)
    if revision is not None:
        return repo_id, revision

    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type='dataset',
            revision=DEFAULT_DATASTORE_REVISION,
        )
    except Exception as exc:  # noqa: BLE001
        raise HFEvalsError(
            f'Unable to resolve latest datastore commit for {repo_id}'
        ) from exc

    sha = getattr(info, 'sha', None)
    if sha is None and isinstance(info, dict):
        sha = info.get('sha')
    if not isinstance(sha, str) or not sha.strip():
        raise HFEvalsError(
            f'HF dataset repo did not return a commit sha for {repo_id}'
        )
    return repo_id, sha.strip()


def dump_yaml_entries(entries: list[dict[str, Any]]) -> str:
    return yaml.safe_dump(
        entries,
        sort_keys=False,
        allow_unicode=False,
        width=88,
    )


def _is_real_hf_api(api: HfApi) -> bool:
    return api.__class__ is HfApi


def _hf_model_api_url(repo_id: str) -> str:
    return f'https://huggingface.co/api/models/{quote(repo_id, safe="/")}'


def _http_model_info(repo_id: str) -> dict[str, Any]:
    try:
        response = requests.get(
            _hf_model_api_url(repo_id), timeout=HF_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        raise HFEvalsError(f'Unable to check HF model repo: {repo_id}') from exc
    if response.status_code == 404:
        raise HFEvalsError(f'HF model repo does not exist: {repo_id}')
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HFEvalsError(f'Unable to check HF model repo: {repo_id}') from exc
    loaded = response.json()
    if not isinstance(loaded, dict):
        raise HFEvalsError(f'HF model API returned invalid data: {repo_id}')
    return loaded


def _repo_exists(api: HfApi, repo_id: str) -> None:
    if _is_real_hf_api(api):
        _http_model_info(repo_id)
        return
    try:
        api.model_info(repo_id)
    except Exception as exc:  # noqa: BLE001 - preserve HF client details
        raise HFEvalsError(f'HF model repo does not exist: {repo_id}') from exc


def _datastore_blob_url(
    path: str,
    *,
    datastore_revision: str,
    datastore_repo: str = DEFAULT_DATASTORE_REPO,
) -> str:
    repo = datastore_repo.strip().strip('/')
    if not repo:
        raise HFEvalsError('Datastore repo must not be empty.')
    revision = datastore_revision.strip()
    if not revision:
        raise HFEvalsError('Datastore revision must not be empty.')
    return (
        f'https://huggingface.co/datasets/{quote(repo, safe="/")}/blob/'
        f'{quote(revision, safe="")}/'
        f'{quote(path, safe="/")}'
    )


def _date_from_result(log: EvaluationLog, result: EvaluationResult) -> str | None:
    value = result.evaluation_timestamp or log.evaluation_timestamp
    if value is None:
        return None
    try:
        if value.replace('.', '', 1).isdigit():
            return datetime.fromtimestamp(float(value), tz=UTC).date().isoformat()
    except ValueError:
        pass
    if len(value) >= 10:
        return value[:10]
    raise HFEvalsError(f'Invalid evaluation timestamp: {value!r}')


def _dataset_ids_for_config(config: BenchmarkConfig) -> set[str]:
    return {
        config.dataset_id.strip().lower(),
        *(alias.strip().lower() for alias in config.dataset_aliases),
    }


def _result_matches_dataset(
    result: EvaluationResult, config: BenchmarkConfig
) -> bool:
    if result.source_data.source_type == 'hf_dataset':
        hf_repo = (result.source_data.hf_repo or '').strip().lower()
        if hf_repo in _dataset_ids_for_config(config):
            return True
        additional_details = result.source_data.additional_details or {}
        if isinstance(additional_details, dict):
            benchmark_hf_repo = (
                str(additional_details.get('benchmark_hf_repo') or '')
                .strip()
                .lower()
            )
            if benchmark_hf_repo in _dataset_ids_for_config(config):
                return True
        if hf_repo:
            return False
        dataset_name = normalize_benchmark(result.source_data.dataset_name)
        return dataset_name == normalize_benchmark(config.task_id)

    if result.source_data.source_type == 'url':
        dataset_urls = [
            url.strip().lower().rstrip('/')
            for url in result.source_data.url
            if isinstance(url, str)
        ]
        return any(
            url.endswith(f'/datasets/{dataset_id}')
            for dataset_id in _dataset_ids_for_config(config)
            for url in dataset_urls
        )

    return False


def _result_matches_preferred_metric(
    result: EvaluationResult, config: BenchmarkConfig
) -> bool:
    if not config.preferred_metric_ids:
        return True
    allowed = {item.strip().lower() for item in config.preferred_metric_ids}
    result_ids = {
        item.strip().lower()
        for item in (
            result.evaluation_result_id,
            result.metric_config.metric_id,
        )
        if isinstance(item, str)
    }
    return bool(allowed & result_ids)


def _result_for_dataset(
    log: EvaluationLog, config: BenchmarkConfig
) -> EvaluationResult | None:
    matches = [
        result
        for result in log.evaluation_results
        if _result_matches_dataset(result, config)
        and _result_matches_preferred_metric(result, config)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        ids = [
            result.evaluation_result_id or result.evaluation_name
            for result in matches
        ]
        raise HFEvalsError(
            f'{log.evaluation_id} has {len(matches)} matching '
            f'evaluation_results for {config.dataset_id}: {ids}'
        )
    return matches[0]


def _results_for_supported_datasets(
    log: EvaluationLog,
) -> list[tuple[str, BenchmarkConfig, EvaluationResult]]:
    results: list[tuple[str, BenchmarkConfig, EvaluationResult]] = []
    for benchmark, config in BENCHMARK_CONFIGS.items():
        matches = [
            result
            for result in log.evaluation_results
            if _result_matches_dataset(result, config)
            and _result_matches_preferred_metric(result, config)
        ]
        if len(matches) > 1:
            ids = [
                result.evaluation_result_id or result.evaluation_name
                for result in matches
            ]
            raise HFEvalsError(
                f'{log.evaluation_id} has {len(matches)} matching '
                f'evaluation_results for {config.dataset_id}: {ids}'
            )
        if matches:
            results.append((benchmark, config, matches[0]))
    return results


def _community_eval_entry(
    *,
    config: BenchmarkConfig,
    task_id: str,
    value: float | int,
    date: str | None,
    source_url: str,
    notes: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'dataset': {'id': config.dataset_id, 'task_id': task_id},
        'value': value,
        'source': {'url': source_url, 'name': SOURCE_NAME},
    }
    if date is not None:
        entry['date'] = date
    if notes is not None:
        entry['notes'] = notes
    return entry


def _gpqa_variant_notes(result: EvaluationResult) -> str | None:
    source_data = result.source_data
    hf_repo = (source_data.hf_repo or '').strip().lower()
    dataset_name = normalize_benchmark(source_data.dataset_name).replace(' ', '_')
    result_id = (result.evaluation_result_id or '').strip().lower()
    metric_name = (result.metric_config.metric_name or '').strip().lower()

    if (
        hf_repo == 'human-centered-eval/openeval'
        and dataset_name == 'gpqa'
        and (
            result_id == 'gpqa::chain-of-thought-correctness'
            or metric_name == 'chain_of_thought_correctness'
        )
    ):
        return 'GPQA chain-of-thought'

    if dataset_name == 'gpqa_diamond' or result_id.startswith('gpqa_diamond/'):
        return 'GPQA Diamond'

    return None


def _community_eval_notes(benchmark: str, result: EvaluationResult) -> str | None:
    if benchmark == 'gpqa':
        return _gpqa_variant_notes(result)
    return None


def _community_eval_notes_for_subset(
    benchmark: str,
    subset: str | None,
) -> str | None:
    if subset is None:
        return None
    if benchmark != 'gpqa':
        return None
    normalized_subset = normalize_benchmark(subset)
    try:
        return GPQA_SUBSET_NOTES[normalized_subset]
    except KeyError as exc:
        allowed = ', '.join(sorted(GPQA_SUBSET_NOTES))
        raise HFEvalsError(
            f'Unsupported subset for gpqa: {subset!r}; expected one of {allowed}.'
        ) from exc


def _community_eval_task_id(
    benchmark: str,
    config: BenchmarkConfig,
    result: EvaluationResult,
    notes: str | None,
) -> str:
    if benchmark == 'gpqa':
        if notes == 'GPQA chain-of-thought':
            return 'main'
        if notes == 'GPQA Diamond':
            return 'diamond'
    return config.task_id


def _community_eval_value(result: EvaluationResult) -> float | int:
    score = result.score_details.score
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(float(score))
    ):
        raise HFEvalsError('score_details.score must be numeric')

    value: float
    metric_unit = (result.metric_config.metric_unit or '').strip().lower()
    max_score = result.metric_config.max_score
    if metric_unit in {'percent', 'percentage'} or max_score == 100:
        value = float(score)
    elif metric_unit == 'proportion' or max_score == 1:
        value = float(score) * 100
    else:
        raise HFEvalsError(
            'Cannot convert score to 0-100 Community Evals value without '
            f'metric_unit=proportion/percent or max_score=1/100 for '
            f'{result.evaluation_result_id!r}.'
        )

    if value < 0 or value > 100:
        raise HFEvalsError(
            f'Community Evals value for {result.evaluation_result_id!r} '
            f'must be in the 0-100 range, got {value}.'
        )
    return round(value, 10)


def _target_path(config: BenchmarkConfig) -> str:
    return f'.eval_results/{config.yaml_name}.yaml'


def _entry_is_ready(entry: dict[str, Any]) -> bool:
    return entry.get('status', 'ready') == 'ready'


def _entry_has_yaml_preview(entry: dict[str, Any]) -> bool:
    return entry.get('status', 'ready') in {'ready', AUDIT_ERROR_STATUS}


def _api_only_skip_reason(log: EvaluationLog) -> str | None:
    platform = (log.model_info.inference_platform or '').strip().lower()
    developer = (log.model_info.developer or '').strip().lower()
    model_id = log.model_info.id.strip().lower()
    model_name = log.model_info.name.strip().lower()
    if any(
        model_id == prefix or model_id.startswith(prefix)
        for prefix in OPEN_WEIGHT_MODEL_PREFIXES
    ):
        return None
    if any(
        model_name == prefix or model_name.startswith(prefix)
        for prefix in OPEN_WEIGHT_MODEL_PREFIXES
    ):
        return None
    values = (platform, developer, model_id, model_name)
    for prefix in API_ONLY_PROVIDER_PREFIXES:
        if any(value == prefix or value.startswith(f'{prefix}/') for value in values):
            return f'api_only_or_closed_provider:{prefix}'
    if 'gemini' in model_id or 'gemini' in model_name:
        return 'api_only_or_closed_provider:gemini'
    return None


def _safe_index_path(row: dict[str, Any], field: str, *, line_ref: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HFEvalsError(f'{line_ref}: missing {field}')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise HFEvalsError(f'{line_ref}: unsafe {field}: {value}')
    return value


def _index_subset(row: dict[str, Any], *, line_ref: str) -> str | None:
    value = row.get('subset')
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise HFEvalsError(f'{line_ref}: subset must be a non-empty string')
    return value.strip()


def _reject_unsupported_row_sources(row: dict[str, Any], *, line_ref: str) -> None:
    unsupported = [
        field for field in ('url', 'local_path') if row.get(field) is not None
    ]
    if unsupported:
        fields = ', '.join(unsupported)
        raise HFEvalsError(
            f'{line_ref}: unsupported aggregate row source field(s): {fields}; '
            'use object_path from the datastore index'
        )


def _validate_instance_level_reference(
    row: dict[str, Any],
    *,
    line_ref: str,
) -> None:
    available = row.get('instance_level_available')
    if not isinstance(available, bool):
        raise HFEvalsError(f'{line_ref}: instance_level_available must be boolean')
    if not available:
        unexpected = [
            field
            for field in (
                'instance_level_path',
                'instance_level_size_bytes',
                'instance_sha',
            )
            if row.get(field) is not None
        ]
        if unexpected:
            raise HFEvalsError(
                f'{line_ref}: instance_level_available is false but '
                f'instance provenance is present: {", ".join(unexpected)}'
            )
        return

    _safe_index_path(row, 'instance_level_path', line_ref=line_ref)
    size = row.get('instance_level_size_bytes')
    if not isinstance(size, int):
        raise HFEvalsError(f'{line_ref}: instance_level_size_bytes must be an integer')
    instance_sha = row.get('instance_sha')
    if not isinstance(instance_sha, str) or not instance_sha:
        raise HFEvalsError(f'{line_ref}: missing instance_sha')
    if len(instance_sha) != 64 or any(
        char not in '0123456789abcdef' for char in instance_sha.lower()
    ):
        raise HFEvalsError(f'{line_ref}: instance_sha must be a sha256 hex digest')


def _index_trace_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for field in ('legacy_path', 'object_path', 'subset'):
        value = row.get(field)
        if value is not None:
            fields[field] = value
    return fields


def _resolve_index_jsonl_path(index_path: Path) -> Path:
    resolved = index_path.resolve()
    if not resolved.is_dir():
        return resolved

    aggregate_jsonl = resolved / 'aggregate.jsonl'
    if not aggregate_jsonl.exists():
        raise HFEvalsError(
            f'Index directory must contain aggregate.jsonl: {resolved}'
        )
    if not aggregate_jsonl.is_file():
        raise HFEvalsError(
            f'Index directory aggregate.jsonl must be a file: {aggregate_jsonl}'
        )
    return aggregate_jsonl


def _load_index_rows(index_jsonl: Path) -> list[dict[str, Any]]:
    if not index_jsonl.exists():
        raise HFEvalsError(f'Index JSONL does not exist: {index_jsonl}')
    if not index_jsonl.is_file():
        raise HFEvalsError(f'Index JSONL must be a file: {index_jsonl}')

    rows: list[dict[str, Any]] = []
    with index_jsonl.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HFEvalsError(
                    f'{index_jsonl}:{line_number}: invalid JSONL row: {exc}'
                ) from exc
            if not isinstance(row, dict):
                raise HFEvalsError(
                    f'{index_jsonl}:{line_number}: JSONL row must be an object'
                )
            row['_index_line'] = line_number
            rows.append(row)
    if not rows:
        raise HFEvalsError(f'Index JSONL has no rows: {index_jsonl}')
    return rows


def _safe_collection_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise HFEvalsError('Collection name must not be empty.')
    if name != value:
        raise HFEvalsError('Collection name must not include surrounding whitespace.')
    if name.endswith('.jsonl'):
        raise HFEvalsError('Pass the collection name without the .jsonl suffix.')
    if '/' in name or '\\' in name:
        raise HFEvalsError(
            'Collection name must be a single by_collection file stem.'
        )
    parts = Path(name).parts
    if parts != (name,) or name in {'.', '..'}:
        raise HFEvalsError(
            'Collection name must be a single by_collection file stem.'
        )
    return name


def _collection_index_path(collection_name: str) -> str:
    collection_name = _safe_collection_name(collection_name)
    return f'flat/indexes/by_collection/{collection_name}.jsonl'


def _collection_stems_from_repo_files(paths: list[str]) -> list[str]:
    prefix = 'flat/indexes/by_collection/'
    suffix = '.jsonl'
    stems = set()
    for path in paths:
        if not isinstance(path, str):
            continue
        if not path.startswith(prefix) or not path.endswith(suffix):
            continue
        filename = path[len(prefix) :]
        if '/' in filename:
            continue
        stem = filename[: -len(suffix)]
        if stem:
            stems.add(stem)
    return sorted(stems, key=str.lower)


def _normalized_collection_stem(value: str) -> str:
    return (
        value.lower()
        .replace('-', '')
        .replace('_', '')
        .replace(' ', '')
        .replace('.', '')
    )


def _nearby_collection_stems(collection_name: str, stems: list[str]) -> list[str]:
    normalized_requested = _normalized_collection_stem(collection_name)
    suggestions = []
    for stem in stems:
        normalized_stem = _normalized_collection_stem(stem)
        if (
            normalized_requested in normalized_stem
            or normalized_stem in normalized_requested
        ):
            suggestions.append(stem)
    for stem in get_close_matches(collection_name, stems, n=5, cutoff=0.55):
        if stem not in suggestions:
            suggestions.append(stem)
    return suggestions[:5]


def _collection_suggestion_text(
    *,
    api: HfApi,
    datastore_repo: str,
    datastore_revision: str,
    collection_name: str,
) -> str:
    try:
        paths = api.list_repo_files(
            repo_id=datastore_repo,
            repo_type='dataset',
            revision=datastore_revision,
        )
    except Exception as exc:  # noqa: BLE001
        return f'Unable to list available collection stems: {exc}'
    stems = _collection_stems_from_repo_files(list(paths))
    if not stems:
        return 'No collection JSONL files were found under flat/indexes/by_collection.'
    suggestions = _nearby_collection_stems(collection_name, stems)
    if not suggestions:
        return 'No nearby collection stems found.'
    return f'Nearby collection stems: {", ".join(suggestions)}'


def _download_collection_index_jsonl(
    *,
    api: HfApi,
    datastore_repo: str,
    datastore_revision: str,
    collection_name: str,
    download_file: Callable[..., str] | None = None,
) -> tuple[str, Path]:
    download_file = download_file or hf_hub_download
    collection_index_path = _collection_index_path(collection_name)
    try:
        local_path = download_file(
            repo_id=datastore_repo,
            repo_type='dataset',
            filename=collection_index_path,
            revision=datastore_revision,
        )
    except Exception as exc:  # noqa: BLE001
        suggestion_text = _collection_suggestion_text(
            api=api,
            datastore_repo=datastore_repo,
            datastore_revision=datastore_revision,
            collection_name=collection_name,
        )
        raise HFEvalsError(
            f'Unable to download required collection index file '
            f'{collection_index_path} from {datastore_repo}@{datastore_revision}. '
            f'{suggestion_text}'
        ) from exc
    return collection_index_path, Path(local_path)


def _candidate_duplicate_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    dataset = entry['yaml_entry']['dataset']
    return (
        str(entry['model_repo']).strip().lower(),
        str(dataset['id']).strip().lower(),
        str(dataset['task_id']).strip(),
    )


def _numeric_score(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HFEvalsError(f'{context}: score value must be numeric')
    score = float(value)
    if not isfinite(score):
        raise HFEvalsError(f'{context}: score value must be finite')
    return score


def _scores_equal(left: Any, right: Any) -> bool:
    return abs(
        _numeric_score(left, context='left score')
        - _numeric_score(right, context='right score')
    ) <= 1e-9


def _read_online_indexed_record(
    *,
    datastore_repo: str,
    datastore_revision: str,
    object_path: str,
    row: dict[str, Any],
    line_ref: str,
    download_file: Callable[..., str] | None = None,
) -> EvaluationLog:
    download_file = download_file or hf_hub_download
    try:
        local_path = download_file(
            repo_id=datastore_repo,
            repo_type='dataset',
            filename=object_path,
            revision=datastore_revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise HFEvalsError(
            f'{line_ref}: unable to download {object_path} from '
            f'{datastore_repo}@{datastore_revision}'
        ) from exc

    return _parse_indexed_record_bytes(
        data=Path(local_path).read_bytes(),
        row=row,
        line_ref=line_ref,
        source_ref=object_path,
    )


def _parse_indexed_record_bytes(
    *,
    data: bytes,
    row: dict[str, Any],
    line_ref: str,
    source_ref: str,
) -> EvaluationLog:
    expected_size = row.get('size_bytes')
    if expected_size is not None:
        if not isinstance(expected_size, int):
            raise HFEvalsError(f'{line_ref}: size_bytes must be an integer')
        if len(data) != expected_size:
            raise HFEvalsError(
                f'{line_ref}: size_bytes mismatch for {source_ref}: '
                f'expected {expected_size}, got {len(data)}'
            )

    expected_sha = row.get('sha256')
    if not isinstance(expected_sha, str) or not expected_sha:
        raise HFEvalsError(f'{line_ref}: missing sha256')
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha:
        raise HFEvalsError(
            f'{line_ref}: sha256 mismatch for {source_ref}: '
            f'expected {expected_sha}, got {actual_sha}'
        )

    try:
        raw = json.loads(data.decode('utf-8'))
        log = EvaluationLog.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise HFEvalsError(f'{line_ref}: invalid EEE aggregate JSON: {exc}') from exc
    return log


def _candidate_from_record_result(
    *,
    benchmark: str,
    config: BenchmarkConfig,
    record_path: str,
    log: EvaluationLog,
    result: EvaluationResult,
    model_repo: str,
    source_url: str,
    source: str,
    status: str,
    hf_check_error: str | None,
    subset: str | None = None,
) -> dict[str, Any]:
    value = _community_eval_value(result)
    notes = _community_eval_notes(benchmark, result)
    subset_notes = _community_eval_notes_for_subset(benchmark, subset)
    if subset_notes is not None:
        if notes is not None and notes != subset_notes:
            raise HFEvalsError(
                f'Index subset {subset!r} conflicts with aggregate variant '
                f'{notes!r}.'
            )
        notes = subset_notes
    task_id = _community_eval_task_id(benchmark, config, result, notes)
    yaml_entry = _community_eval_entry(
        config=config,
        task_id=task_id,
        value=value,
        date=_date_from_result(log, result),
        source_url=source_url,
        notes=notes,
    )
    entry = {
        'status': status,
        'benchmark': benchmark,
        'model_repo': model_repo,
        'target_path': _target_path(config),
        'eee_evaluation_id': log.evaluation_id,
        'eee_evaluation_result_id': result.evaluation_result_id,
        'eee_record_path': record_path,
        'source': source,
        'yaml_entry': yaml_entry,
        'pr_title': f'Add EvalEval {task_id} result for {model_repo}',
        'pr_body': (
            'Adds a Hugging Face Community Evals result from '
            f'{SOURCE_NAME} with a backlink to the source EEE record.'
        ),
    }
    if hf_check_error is not None:
        entry['hf_check_error'] = hf_check_error
    return entry


def build_index_manifest(
    *,
    index_jsonl: Path,
    datastore: str,
    benchmarks: list[str],
    output_path: Path | None = None,
    api: HfApi | None = None,
    check_hf: bool = True,
    download_file: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Build HF Community Evals candidates from online flat datastore rows."""

    api = api or HfApi()
    index_jsonl = _resolve_index_jsonl_path(index_jsonl)
    rows = _load_index_rows(index_jsonl)
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_keys: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    repo_check_cache: dict[str, HFEvalsError | None] = {}
    datastore_repo, datastore_revision = resolve_datastore_locator(
        datastore, api=api
    )

    def cached_repo_error(repo_id: str) -> HFEvalsError | None:
        if repo_id not in repo_check_cache:
            try:
                _repo_exists(api, repo_id)
                repo_check_cache[repo_id] = None
            except HFEvalsError as exc:
                repo_check_cache[repo_id] = exc
        return repo_check_cache[repo_id]

    for row in rows:
        line_number = row['_index_line']
        line_ref = f'{index_jsonl}:{line_number}'

        raw_benchmark = row.get('benchmark')
        if not isinstance(raw_benchmark, str) or not raw_benchmark.strip():
            errors.append(f'{line_ref}: missing benchmark')
            continue
        try:
            _reject_unsupported_row_sources(row, line_ref=line_ref)
            subset = _index_subset(row, line_ref=line_ref)
        except HFEvalsError as exc:
            errors.append(str(exc))
            continue
        normalized_benchmark = BENCHMARK_ALIASES.get(
            normalize_benchmark(raw_benchmark),
            normalize_benchmark(raw_benchmark),
        )
        if normalized_benchmark not in BENCHMARK_CONFIGS:
            skipped.append(
                {
                    'reason': 'unsupported_index_benchmark',
                    'benchmark': raw_benchmark,
                    'index_path': index_jsonl.as_posix(),
                    'index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            continue
        if normalized_benchmark not in benchmarks:
            skipped.append(
                {
                    'reason': 'benchmark_not_selected',
                    'benchmark': raw_benchmark,
                    'index_path': index_jsonl.as_posix(),
                    'index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            continue

        record_type = row.get('record_type')
        if record_type != 'aggregate':
            skipped.append(
                {
                    'reason': 'non_aggregate_index_row',
                    'record_type': record_type,
                    'benchmark': raw_benchmark,
                    'index_path': index_jsonl.as_posix(),
                    'index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            continue

        try:
            object_path = _safe_index_path(row, 'object_path', line_ref=line_ref)
            record_ref = object_path
            source_url = _datastore_blob_url(
                object_path,
                datastore_repo=datastore_repo,
                datastore_revision=datastore_revision,
            )
            source_mode = 'online_flat_index_jsonl'
            log = _read_online_indexed_record(
                datastore_repo=datastore_repo,
                datastore_revision=datastore_revision,
                object_path=object_path,
                row=row,
                line_ref=line_ref,
                download_file=download_file,
            )
        except HFEvalsError as exc:
            errors.append(str(exc))
            continue

        api_only_reason = _api_only_skip_reason(log)
        if api_only_reason is not None:
            skipped.append(
                {
                    'reason': api_only_reason,
                    'benchmark': raw_benchmark,
                    'eee_evaluation_id': log.evaluation_id,
                    'eee_record_path': record_ref,
                    'index_path': index_jsonl.as_posix(),
                    'index_line': line_number,
                    **_index_trace_fields(row),
                    'model_id': log.model_info.id,
                }
            )
            continue

        raw_model_repo = log.model_info.id
        if not isinstance(raw_model_repo, str) or not raw_model_repo.strip():
            errors.append(f'{line_ref}: record has no model_info.id')
            continue
        model_repo = raw_model_repo.strip()

        status = 'ready'
        hf_check_error: str | None = None
        if check_hf:
            error = cached_repo_error(model_repo)
            if error is not None:
                status = 'missing_hf_model'
                hf_check_error = str(error)

        config = BENCHMARK_CONFIGS[normalized_benchmark]
        try:
            result = _result_for_dataset(log, config)
        except HFEvalsError as exc:
            errors.append(f'{line_ref}: {exc}')
            continue
        if result is None:
            skipped.append(
                {
                    'reason': 'no_matching_evaluation_result',
                    'benchmark': raw_benchmark,
                    'eee_evaluation_id': log.evaluation_id,
                    'eee_record_path': record_ref,
                    'index_path': index_jsonl.as_posix(),
                    'index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            continue

        try:
            entry = _candidate_from_record_result(
                benchmark=normalized_benchmark,
                config=config,
                record_path=record_ref,
                log=log,
                result=result,
                model_repo=model_repo,
                source_url=source_url,
                source=source_mode,
                status=status,
                hf_check_error=hf_check_error,
                subset=subset,
            )
        except HFEvalsError as exc:
            errors.append(f'{line_ref}: {exc}')
            continue

        entry['index_path'] = index_jsonl.as_posix()
        entry['index_line'] = line_number
        entry['flat_object_path'] = record_ref
        for field in (
            'legacy_path',
            'object_uuid',
            'subset',
            'sha256',
            'size_bytes',
            'eval_schema_version',
            'instance_object_path',
            'instance_sha256',
            'instance_size_bytes',
        ):
            value = row.get(field)
            if value is not None:
                entry[field] = value

        dataset = entry['yaml_entry']['dataset']
        duplicate_key = (
            model_repo.lower(),
            dataset['id'],
            dataset['task_id'],
            entry['yaml_entry'].get('notes'),
        )
        existing_entry = seen_keys.get(duplicate_key)
        if existing_entry is not None:
            if existing_entry['yaml_entry'] == entry['yaml_entry']:
                skipped.append(
                    {
                        'reason': 'duplicate_candidate_same_entry',
                        'model_repo': model_repo,
                        'eee_evaluation_id': log.evaluation_id,
                        'eee_record_path': record_ref,
                        'index_path': index_jsonl.as_posix(),
                        'index_line': line_number,
                        **_index_trace_fields(row),
                    }
                )
                continue
            errors.append(
                f'{line_ref}: duplicate candidate for {model_repo} '
                f'{dataset["id"]}/{dataset["task_id"]} with different '
                'YAML values.'
            )
            continue
        seen_keys[duplicate_key] = entry
        entries.append(entry)

    manifest = {
        'version': MANIFEST_VERSION,
        'created_at': datetime.now(tz=UTC).isoformat(),
        'benchmarks': benchmarks,
        'hf_checks': check_hf,
        'source_url_mode': 'online_flat_index_jsonl',
        'datastore': f'{datastore_repo}@{datastore_revision}',
        'datastore_input': datastore,
        'datastore_repo': datastore_repo,
        'datastore_revision': datastore_revision,
        'index_jsonl': index_jsonl.as_posix(),
        'entries': entries,
        'skipped': skipped,
        'errors': errors,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    if errors:
        raise HFEvalsError('\n'.join(errors))

    return manifest


def build_collection_manifest(
    *,
    collection_name: str,
    datastore: str,
    output_path: Path | None = None,
    api: HfApi | None = None,
    check_hf: bool = True,
    download_file: Callable[..., str] | None = None,
    progress: ReviewProgress | None = None,
) -> dict[str, Any]:
    """Build HF Community Evals candidates from a datastore collection."""

    progress = progress or ReviewProgress()
    api = api or HfApi()
    collection_name = _safe_collection_name(collection_name)
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_keys: dict[tuple[str, str, str], dict[str, Any]] = {}
    repo_check_cache: dict[str, HFEvalsError | None] = {}
    setup_task = progress.add_task('Resolving datastore revision', total=4)
    datastore_repo, datastore_revision = resolve_datastore_locator(
        datastore, api=api
    )
    progress.update(
        setup_task,
        advance=1,
        description=f'Downloading collection index {collection_name}.jsonl',
    )
    collection_index_path, collection_index_jsonl = _download_collection_index_jsonl(
        api=api,
        datastore_repo=datastore_repo,
        datastore_revision=datastore_revision,
        collection_name=collection_name,
        download_file=download_file,
    )
    progress.update(setup_task, advance=1, description='Reading collection rows')
    rows = _load_index_rows(collection_index_jsonl)
    progress.update(
        setup_task,
        advance=1,
        description=f'Loaded {len(rows)} collection rows',
    )

    def cached_repo_error(repo_id: str) -> HFEvalsError | None:
        if repo_id not in repo_check_cache:
            try:
                _repo_exists(api, repo_id)
                repo_check_cache[repo_id] = None
            except HFEvalsError as exc:
                repo_check_cache[repo_id] = exc
        return repo_check_cache[repo_id]

    row_task = progress.add_task(
        f'Processing {len(rows)} aggregate rows',
        total=len(rows),
    )
    for row_number, row in enumerate(rows, start=1):
        line_number = row['_index_line']
        line_ref = f'{collection_index_path}:{line_number}'
        raw_benchmark = row.get('benchmark')
        row_label = f'row {row_number}/{len(rows)}'

        try:
            _reject_unsupported_row_sources(row, line_ref=line_ref)
            subset = _index_subset(row, line_ref=line_ref)
            _validate_instance_level_reference(row, line_ref=line_ref)
        except HFEvalsError as exc:
            errors.append(str(exc))
            progress.update(row_task, advance=1)
            continue

        record_type = row.get('record_type')
        if record_type != 'aggregate':
            skipped.append(
                {
                    'reason': 'non_aggregate_collection_row',
                    'record_type': record_type,
                    'benchmark': raw_benchmark,
                    'collection_index_path': collection_index_path,
                    'collection_index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            progress.update(row_task, advance=1)
            continue

        try:
            object_path = _safe_index_path(row, 'object_path', line_ref=line_ref)
            progress.update(
                row_task,
                description=f'{row_label}: downloading {object_path}',
            )
            source_url = _datastore_blob_url(
                object_path,
                datastore_repo=datastore_repo,
                datastore_revision=datastore_revision,
            )
            log = _read_online_indexed_record(
                datastore_repo=datastore_repo,
                datastore_revision=datastore_revision,
                object_path=object_path,
                row=row,
                line_ref=line_ref,
                download_file=download_file,
            )
        except HFEvalsError as exc:
            errors.append(str(exc))
            progress.update(row_task, advance=1)
            continue

        api_only_reason = _api_only_skip_reason(log)
        if api_only_reason is not None:
            skipped.append(
                {
                    'reason': api_only_reason,
                    'benchmark': raw_benchmark,
                    'eee_evaluation_id': log.evaluation_id,
                    'eee_record_path': object_path,
                    'collection_index_path': collection_index_path,
                    'collection_index_line': line_number,
                    **_index_trace_fields(row),
                    'model_id': log.model_info.id,
                }
            )
            progress.update(row_task, advance=1)
            continue

        raw_model_repo = log.model_info.id
        if not isinstance(raw_model_repo, str) or not raw_model_repo.strip():
            errors.append(f'{line_ref}: record has no model_info.id')
            progress.update(row_task, advance=1)
            continue
        model_repo = raw_model_repo.strip()
        progress.update(row_task, description=f'{row_label}: checking {model_repo}')

        status = 'ready'
        hf_check_error: str | None = None
        if check_hf:
            error = cached_repo_error(model_repo)
            if error is not None:
                status = 'missing_hf_model'
                hf_check_error = str(error)

        try:
            supported_results = _results_for_supported_datasets(log)
        except HFEvalsError as exc:
            errors.append(f'{line_ref}: {exc}')
            progress.update(row_task, advance=1)
            continue
        if not supported_results:
            skipped.append(
                {
                    'reason': 'no_supported_hf_dataset_result',
                    'benchmark': raw_benchmark,
                    'eee_evaluation_id': log.evaluation_id,
                    'eee_record_path': object_path,
                    'collection_index_path': collection_index_path,
                    'collection_index_line': line_number,
                    **_index_trace_fields(row),
                }
            )
            progress.update(row_task, advance=1)
            continue

        for benchmark, config, result in supported_results:
            try:
                entry = _candidate_from_record_result(
                    benchmark=benchmark,
                    config=config,
                    record_path=object_path,
                    log=log,
                    result=result,
                    model_repo=model_repo,
                    source_url=source_url,
                    source='online_collection_index_jsonl',
                    status=status,
                    hf_check_error=hf_check_error,
                    subset=subset,
                )
            except HFEvalsError as exc:
                errors.append(f'{line_ref}: {exc}')
                continue

            entry['collection'] = collection_name
            entry['collection_index_path'] = collection_index_path
            entry['collection_index_line'] = line_number
            entry['flat_object_path'] = object_path
            for field in (
                'legacy_path',
                'object_uuid',
                'subset',
                'sha256',
                'size_bytes',
                'eval_schema_version',
                'instance_level_available',
                'instance_level_path',
                'instance_level_sha256',
                'instance_level_size_bytes',
                'instance_sha',
                'instance_object_path',
                'instance_sha256',
                'instance_size_bytes',
            ):
                value = row.get(field)
                if value is not None:
                    entry[field] = value

            duplicate_key = _candidate_duplicate_key(entry)
            existing_entry = seen_keys.get(duplicate_key)
            if existing_entry is not None:
                if _scores_equal(
                    existing_entry['yaml_entry']['value'],
                    entry['yaml_entry']['value'],
                ):
                    skipped.append(
                        {
                            'reason': 'duplicate_candidate_same_score',
                            'model_repo': model_repo,
                            'eee_evaluation_id': log.evaluation_id,
                            'eee_record_path': object_path,
                            'collection_index_path': collection_index_path,
                            'collection_index_line': line_number,
                            **_index_trace_fields(row),
                        }
                    )
                    continue
                dataset = entry['yaml_entry']['dataset']
                errors.append(
                    f'{line_ref}: duplicate candidate for {model_repo} '
                    f'{dataset["id"]}/{dataset["task_id"]} with different '
                    'scores.'
                )
                continue
            seen_keys[duplicate_key] = entry
            entries.append(entry)
        progress.update(row_task, advance=1)

    progress.update(row_task, description=f'Processed {len(rows)} aggregate rows')

    manifest = {
        'version': MANIFEST_VERSION,
        'created_at': datetime.now(tz=UTC).isoformat(),
        'collection': collection_name,
        'benchmarks': list(DEFAULT_BENCHMARKS),
        'hf_checks': check_hf,
        'source_url_mode': 'online_collection_index_jsonl',
        'datastore': f'{datastore_repo}@{datastore_revision}',
        'datastore_input': datastore,
        'datastore_repo': datastore_repo,
        'datastore_revision': datastore_revision,
        'collection_jsonl': collection_index_path,
        'entries': entries,
        'skipped': skipped,
        'errors': errors,
    }

    if output_path is not None:
        _write_manifest(manifest, output_path)

    if errors:
        raise HFEvalsError('\n'.join(errors))

    progress.update(
        setup_task,
        advance=1,
        description=(
            f'Built manifest: {len(entries)} entries, {len(skipped)} skipped, '
            f'{len(errors)} errors'
        ),
    )
    return manifest


def _path_family_for_entry(entry: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    benchmark = entry.get('benchmark')
    if not isinstance(benchmark, str):
        raise HFEvalsError('Manifest entry benchmark must be a string.')
    paths = EVAL_RESULT_PATH_FAMILIES.get(benchmark)
    if paths is None:
        return benchmark, (entry['target_path'],)
    return benchmark, paths


def _repo_eval_tree(
    api: HfApi,
    repo_id: str,
    revision: str,
) -> dict[str, dict[str, Any]]:
    try:
        items = list(
            api.list_repo_tree(
                repo_id,
                '.eval_results',
                recursive=True,
                expand=False,
                revision=revision,
                repo_type='model',
                token=True,
            )
        )
    except EntryNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ == 'EntryNotFoundError':
            return {}
        raise HFEvalsError(
            f'Unable to list .eval_results for {repo_id}@{revision}'
        ) from exc

    tree: dict[str, dict[str, Any]] = {}
    for item in items:
        path = getattr(item, 'path', None) or getattr(item, 'rfilename', None)
        if not path:
            continue
        tree[path] = {
            'blob_id': getattr(item, 'blob_id', None) or path,
            'size': getattr(item, 'size', None),
        }
    return tree


def _discussion_number(discussion: Any) -> int | None:
    value = getattr(discussion, 'num', None)
    if value is None:
        url = getattr(discussion, 'url', '')
        if isinstance(url, str) and '/discussions/' in url:
            value = url.rsplit('/discussions/', 1)[-1].strip('/')
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _discussion_url(repo_id: str, discussion: Any) -> str:
    url = getattr(discussion, 'url', None)
    if isinstance(url, str) and url:
        return url
    number = _discussion_number(discussion)
    if number is None:
        return f'https://huggingface.co/{repo_id}/discussions'
    return f'https://huggingface.co/{repo_id}/discussions/{number}'


def _discussion_revision(discussion: Any) -> str | None:
    revision = getattr(discussion, 'git_reference', None)
    if isinstance(revision, str) and revision:
        return revision
    number = _discussion_number(discussion)
    if number is None:
        return None
    return f'refs/pr/{number}'


def _open_pull_requests(api: HfApi, repo_id: str) -> list[Any]:
    try:
        return list(
            api.get_repo_discussions(
                repo_id,
                repo_type='model',
                discussion_type='pull_request',
                discussion_status='open',
                token=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise HFEvalsError(f'Unable to list open PRs for {repo_id}') from exc


def _candidate_comment(entry: dict[str, Any]) -> str:
    yaml_entry = entry['yaml_entry']
    dataset = yaml_entry['dataset']
    source = yaml_entry['source']
    benchmark = f'{dataset["id"]}/{dataset["task_id"]}'
    source_name = source.get('name') or SOURCE_NAME
    source_url = source['url']
    value = yaml_entry['value']
    return (
        f'This model scores {value} on {benchmark} run by {source_name}, '
        'but it is different from the currently posted score. '
        f'See {source_url} for full details.'
    )


def _already_present_comment(entry: dict[str, Any]) -> str:
    yaml_entry = entry['yaml_entry']
    dataset = yaml_entry['dataset']
    return (
        'Already present, will not open PR: '
        f'{entry["model_repo"]} has {dataset["id"]}/{dataset["task_id"]} '
        f'with score {yaml_entry["value"]}.'
    )


def _eval_yaml_paths(tree: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        path
        for path in tree
        if path.startswith('.eval_results/')
        and path.rsplit('.', 1)[-1].lower() in {'yaml', 'yml'}
    )


def _download_model_file_text(
    *,
    repo_id: str,
    revision: str,
    path: str,
    download_file: Callable[..., str] | None = None,
) -> str:
    download_file = download_file or hf_hub_download
    try:
        local_path = download_file(
            repo_id=repo_id,
            repo_type='model',
            filename=path,
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise HFEvalsError(
            f'Unable to download {path} from {repo_id}@{revision}'
        ) from exc
    return Path(local_path).read_text(encoding='utf-8')


def _load_eval_yaml_entries(
    *,
    repo_id: str,
    revision: str,
    path: str,
    download_file: Callable[..., str] | None = None,
) -> list[dict[str, Any]]:
    text = _download_model_file_text(
        repo_id=repo_id,
        revision=revision,
        path=path,
        download_file=download_file,
    )
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HFEvalsError(
            f'Invalid YAML in {repo_id}@{revision}:{path}: {exc}'
        ) from exc
    if not isinstance(loaded, list):
        raise HFEvalsError(
            f'Eval results YAML must be a list in {repo_id}@{revision}:{path}'
        )
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(loaded, start=1):
        if not isinstance(item, dict):
            raise HFEvalsError(
                f'Eval results item {index} must be an object in '
                f'{repo_id}@{revision}:{path}'
            )
        entries.append(item)
    return entries


def _yaml_dataset_key(yaml_entry: dict[str, Any]) -> tuple[str, str] | None:
    dataset = yaml_entry.get('dataset')
    if not isinstance(dataset, dict):
        return None
    dataset_id = dataset.get('id')
    task_id = dataset.get('task_id')
    if not isinstance(dataset_id, str) or not isinstance(task_id, str):
        return None
    return dataset_id.strip().lower(), task_id.strip()


def _candidate_yaml_dataset_key(entry: dict[str, Any]) -> tuple[str, str]:
    dataset = entry['yaml_entry']['dataset']
    return str(dataset['id']).strip().lower(), str(dataset['task_id']).strip()


def _classify_existing_yaml_entries(
    *,
    candidate: dict[str, Any],
    yaml_entries: list[dict[str, Any]],
    context: str,
) -> dict[str, Any] | None:
    candidate_key = _candidate_yaml_dataset_key(candidate)
    candidate_value = candidate['yaml_entry']['value']
    for item in yaml_entries:
        if _yaml_dataset_key(item) != candidate_key:
            continue
        if 'value' not in item:
            raise HFEvalsError(f'{context}: matching entry is missing value')
        if _scores_equal(item['value'], candidate_value):
            return {
                'status': 'already_present',
                'existing_value': item['value'],
                'comment': _already_present_comment(candidate),
            }
        return {
            'status': 'score_conflict',
            'existing_value': item['value'],
            'comment': _candidate_comment(candidate),
        }
    return None


def audit_manifest_for_hf_eval_duplicates(
    manifest: dict[str, Any],
    *,
    api: HfApi | None = None,
    download_file: Callable[..., str] | None = None,
    progress: ReviewProgress | None = None,
) -> dict[str, Any]:
    """Check candidate YAML entries against main .eval_results and open PRs."""

    progress = progress or ReviewProgress()
    api = api or HfApi()
    entries = [
        (entry_index, entry)
        for entry_index, entry in enumerate(manifest.get('entries', []))
        if _entry_is_ready(entry)
    ]
    main_tree_cache: dict[str, dict[str, dict[str, Any]]] = {}
    main_yaml_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    open_pr_cache: dict[str, list[Any]] = {}
    pr_tree_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    pr_yaml_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    audit_task = progress.add_task(
        f'Auditing {len(entries)} ready candidates',
        total=len(entries),
    )

    def cached_main_tree(repo_id: str) -> dict[str, dict[str, Any]]:
        if repo_id not in main_tree_cache:
            main_tree_cache[repo_id] = _repo_eval_tree(
                api, repo_id, DEFAULT_DATASTORE_REVISION
            )
        return main_tree_cache[repo_id]

    def cached_prs(repo_id: str) -> list[Any]:
        if repo_id not in open_pr_cache:
            open_pr_cache[repo_id] = _open_pull_requests(api, repo_id)
        return open_pr_cache[repo_id]

    def cached_pr_tree(
        repo_id: str, revision: str
    ) -> dict[str, dict[str, Any]]:
        key = (repo_id, revision)
        if key not in pr_tree_cache:
            pr_tree_cache[key] = _repo_eval_tree(api, repo_id, revision)
        return pr_tree_cache[key]

    def cached_yaml(
        repo_id: str,
        revision: str,
        path: str,
    ) -> list[dict[str, Any]]:
        if revision == DEFAULT_DATASTORE_REVISION:
            key = (repo_id, path)
            if key not in main_yaml_cache:
                main_yaml_cache[key] = _load_eval_yaml_entries(
                    repo_id=repo_id,
                    revision=revision,
                    path=path,
                    download_file=download_file,
                )
            return main_yaml_cache[key]
        key = (repo_id, revision, path)
        if key not in pr_yaml_cache:
            pr_yaml_cache[key] = _load_eval_yaml_entries(
                repo_id=repo_id,
                revision=revision,
                path=path,
                download_file=download_file,
            )
        return pr_yaml_cache[key]

    for entry_index, entry in entries:
        repo_id = entry['model_repo']
        benchmark = entry.get('benchmark')
        progress.update(
            audit_task,
            description=f'Auditing {repo_id} {entry["target_path"]}',
        )
        try:
            main_tree = cached_main_tree(repo_id)
        except HFEvalsError as exc:
            errors.append(
                {
                    'entry_index': entry_index,
                    'model_repo': repo_id,
                    'benchmark': benchmark,
                    'target_path': entry['target_path'],
                    'stage': 'list_main_eval_results',
                    'error': str(exc),
                }
            )
            progress.update(audit_task, advance=1)
            continue

        for path in _eval_yaml_paths(main_tree):
            try:
                yaml_entries = cached_yaml(
                    repo_id, DEFAULT_DATASTORE_REVISION, path
                )
                match = _classify_existing_yaml_entries(
                    candidate=entry,
                    yaml_entries=yaml_entries,
                    context=f'{repo_id}@main:{path}',
                )
            except HFEvalsError as exc:
                errors.append(
                    {
                        'entry_index': entry_index,
                        'model_repo': repo_id,
                        'benchmark': benchmark,
                        'target_path': entry['target_path'],
                        'stage': 'read_main_eval_results',
                        'path': path,
                        'error': str(exc),
                    }
                )
                continue
            if match is None:
                continue
            findings.append(
                {
                    'type': f'existing_eval_results_{match["status"]}',
                    'status': match['status'],
                    'entry_index': entry_index,
                    'model_repo': repo_id,
                    'benchmark': benchmark,
                    'target_path': entry['target_path'],
                    'candidate_path': entry['target_path'],
                    'existing_path': path,
                    'existing_value': match['existing_value'],
                    'candidate_value': entry['yaml_entry']['value'],
                    'candidate_source_url': entry['yaml_entry']['source']['url'],
                    'comment': match['comment'],
                }
            )

        try:
            discussions = cached_prs(repo_id)
        except HFEvalsError as exc:
            errors.append(
                {
                    'entry_index': entry_index,
                    'model_repo': repo_id,
                    'benchmark': benchmark,
                    'target_path': entry['target_path'],
                    'stage': 'list_open_prs',
                    'error': str(exc),
                }
            )
            progress.update(audit_task, advance=1)
            continue

        for discussion in discussions:
            revision = _discussion_revision(discussion)
            if revision is None:
                errors.append(
                    {
                        'entry_index': entry_index,
                        'model_repo': repo_id,
                        'benchmark': benchmark,
                        'target_path': entry['target_path'],
                        'stage': 'resolve_pr_revision',
                        'error': f'No PR revision for {_discussion_url(repo_id, discussion)}',
                    }
                )
                continue
            try:
                pr_tree = cached_pr_tree(repo_id, revision)
            except HFEvalsError as exc:
                errors.append(
                    {
                        'entry_index': entry_index,
                        'model_repo': repo_id,
                        'benchmark': benchmark,
                        'target_path': entry['target_path'],
                        'stage': 'list_open_pr_eval_results',
                        'pr_url': _discussion_url(repo_id, discussion),
                        'error': str(exc),
                    }
                )
                continue

            changed_paths = []
            for path in _eval_yaml_paths(pr_tree):
                main_blob = main_tree.get(path, {}).get('blob_id')
                pr_blob = pr_tree[path].get('blob_id')
                if main_blob != pr_blob:
                    changed_paths.append(path)
            for path in changed_paths:
                try:
                    yaml_entries = cached_yaml(repo_id, revision, path)
                    match = _classify_existing_yaml_entries(
                        candidate=entry,
                        yaml_entries=yaml_entries,
                        context=f'{repo_id}@{revision}:{path}',
                    )
                except HFEvalsError as exc:
                    errors.append(
                        {
                            'entry_index': entry_index,
                            'model_repo': repo_id,
                            'benchmark': benchmark,
                            'target_path': entry['target_path'],
                            'stage': 'read_open_pr_eval_results',
                            'pr_url': _discussion_url(repo_id, discussion),
                            'path': path,
                            'error': str(exc),
                        }
                    )
                    continue
                if match is None:
                    continue
                findings.append(
                    {
                        'type': f'open_pr_eval_results_{match["status"]}',
                        'status': match['status'],
                        'entry_index': entry_index,
                        'model_repo': repo_id,
                        'benchmark': benchmark,
                        'target_path': entry['target_path'],
                        'candidate_path': entry['target_path'],
                        'pr_url': _discussion_url(repo_id, discussion),
                        'pr_title': getattr(discussion, 'title', None),
                        'paths': [path],
                        'existing_value': match['existing_value'],
                        'candidate_value': entry['yaml_entry']['value'],
                        'candidate_source_url': entry['yaml_entry']['source']['url'],
                        'comment': match['comment'],
                    }
                )

        progress.update(audit_task, advance=1)

    return {
        'created_at': datetime.now(tz=UTC).isoformat(),
        'candidate_count': len(entries),
        'finding_count': len(findings),
        'error_count': len(errors),
        'findings': findings,
        'errors': errors,
    }


def _apply_duplicate_audit_to_manifest(
    manifest: dict[str, Any],
    duplicate_audit: dict[str, Any],
) -> None:
    priority = {'already_present': 1, 'score_conflict': 2}
    selected: dict[int, tuple[int, str, list[dict[str, Any]]]] = {}
    for finding in duplicate_audit.get('findings', []):
        entry_index = finding.get('entry_index')
        status = finding.get('status')
        if not isinstance(entry_index, int) or status not in priority:
            continue
        rank = priority[status]
        existing = selected.get(entry_index)
        if existing is None:
            selected[entry_index] = (rank, status, [finding])
            continue
        existing_rank, existing_status, findings = existing
        findings.append(finding)
        if rank > existing_rank:
            selected[entry_index] = (rank, status, findings)
        else:
            selected[entry_index] = (existing_rank, existing_status, findings)

    entries = manifest.get('entries', [])
    if not isinstance(entries, list):
        raise HFEvalsError('Manifest entries must be a list.')
    for entry_index, (_rank, status, findings) in selected.items():
        if entry_index < 0 or entry_index >= len(entries):
            raise HFEvalsError(
                f'Duplicate audit referenced missing manifest entry {entry_index}.'
            )
        entry = entries[entry_index]
        if not isinstance(entry, dict):
            raise HFEvalsError(
                f'Manifest entry {entry_index} must be an object.'
            )
        if not _entry_is_ready(entry):
            continue
        entry['status'] = status
        entry['duplicate_audit_findings'] = findings

    errors_by_entry: dict[int, list[dict[str, Any]]] = {}
    for error in duplicate_audit.get('errors', []):
        entry_index = error.get('entry_index')
        if not isinstance(entry_index, int):
            continue
        errors_by_entry.setdefault(entry_index, []).append(error)

    for entry_index, audit_errors in errors_by_entry.items():
        if entry_index < 0 or entry_index >= len(entries):
            raise HFEvalsError(
                f'Duplicate audit referenced missing manifest entry {entry_index}.'
            )
        entry = entries[entry_index]
        if not isinstance(entry, dict):
            raise HFEvalsError(
                f'Manifest entry {entry_index} must be an object.'
            )
        entry['duplicate_audit_errors'] = audit_errors
        if _entry_is_ready(entry):
            entry['status'] = AUDIT_ERROR_STATUS


def _write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _review_from_manifest(
    *,
    manifest: dict[str, Any],
    manifest_output_path: Path,
    yaml_output_dir: Path,
    review_output_path: Path,
    duplicate_audit: dict[str, Any],
) -> dict[str, Any]:
    _apply_duplicate_audit_to_manifest(manifest, duplicate_audit)
    _write_manifest(manifest, manifest_output_path)
    yaml_result = _write_yaml_from_manifest(manifest, yaml_output_dir)
    ready_entries = [
        entry for entry in manifest['entries'] if _entry_is_ready(entry)
    ]
    audit_blocked_entries = [
        entry
        for entry in manifest['entries']
        if entry.get('status') == AUDIT_ERROR_STATUS
    ]
    global_audit_errors = [
        error
        for error in duplicate_audit.get('errors', [])
        if not isinstance(error.get('entry_index'), int)
    ]
    review = {
        'created_at': datetime.now(tz=UTC).isoformat(),
        'manifest_path': manifest_output_path.as_posix(),
        'yaml_output_dir': yaml_output_dir.as_posix(),
        'yaml_count': yaml_result['count'],
        'yaml_files': yaml_result['written'],
        'can_open_prs': len(ready_entries) > 0 and not global_audit_errors,
        'audit_blocked_entries': audit_blocked_entries,
        'global_audit_errors': global_audit_errors,
        'missing_hf_models': [
            entry
            for entry in manifest['entries']
            if entry.get('status') == 'missing_hf_model'
        ],
        'manifest': manifest,
        'duplicate_audit': duplicate_audit,
    }
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return review


def review_index_for_hf_evals(
    *,
    index_jsonl: Path,
    datastore: str,
    benchmarks: list[str],
    manifest_output_path: Path,
    yaml_output_dir: Path,
    review_output_path: Path,
    api: HfApi | None = None,
    check_hf: bool = True,
    download_file: Callable[..., str] | None = None,
) -> dict[str, Any]:
    api = api or HfApi()
    manifest = build_index_manifest(
        index_jsonl=index_jsonl,
        datastore=datastore,
        benchmarks=benchmarks,
        output_path=None,
        api=api,
        check_hf=check_hf,
        download_file=download_file,
    )
    duplicate_audit = audit_manifest_for_hf_eval_duplicates(
        manifest,
        api=api,
        download_file=download_file,
    )
    return _review_from_manifest(
        manifest=manifest,
        manifest_output_path=manifest_output_path,
        yaml_output_dir=yaml_output_dir,
        review_output_path=review_output_path,
        duplicate_audit=duplicate_audit,
    )


def review_collection_for_hf_evals(
    *,
    collection_name: str,
    datastore: str,
    manifest_output_path: Path,
    yaml_output_dir: Path,
    review_output_path: Path,
    api: HfApi | None = None,
    check_hf: bool = True,
    download_file: Callable[..., str] | None = None,
    progress: ReviewProgress | None = None,
    force: bool = False,
) -> dict[str, Any]:
    progress = progress or ReviewProgress()
    api = api or HfApi()
    collection_name = _safe_collection_name(collection_name)

    manifest = None
    if not force:
        cached_review = _load_cached_collection_review(
            review_output_path=review_output_path,
            yaml_output_dir=yaml_output_dir,
            collection_name=collection_name,
            datastore=datastore,
            check_hf=check_hf,
        )
        if cached_review is not None:
            cache_task = progress.add_task('Using cached review', total=1)
            progress.update(cache_task, advance=1, description='Used cached review')
            return cached_review

        manifest = _load_cached_collection_manifest(
            manifest_output_path=manifest_output_path,
            collection_name=collection_name,
            datastore=datastore,
            check_hf=check_hf,
        )
        if manifest is not None:
            cache_task = progress.add_task('Using cached manifest', total=1)
            progress.update(
                cache_task,
                advance=1,
                description='Used cached manifest; starting audit',
            )

    if manifest is None:
        manifest = build_collection_manifest(
            collection_name=collection_name,
            datastore=datastore,
            output_path=manifest_output_path,
            api=api,
            check_hf=check_hf,
            download_file=download_file,
            progress=progress,
        )
    duplicate_audit = audit_manifest_for_hf_eval_duplicates(
        manifest,
        api=api,
        download_file=download_file,
        progress=progress,
    )
    return _review_from_manifest(
        manifest=manifest,
        manifest_output_path=manifest_output_path,
        yaml_output_dir=yaml_output_dir,
        review_output_path=review_output_path,
        duplicate_audit=duplicate_audit,
    )


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get('version') != MANIFEST_VERSION:
        raise HFEvalsError(
            f'Unsupported manifest version: {manifest.get("version")!r}'
        )
    entries = manifest.get('entries')
    if not isinstance(entries, list):
        raise HFEvalsError('Manifest entries must be a list.')
    errors = manifest.get('errors') or []
    if errors:
        raise HFEvalsError('Manifest contains errors; rebuild it first.')
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise HFEvalsError('Manifest must be a JSON object.')
    return _validate_manifest(manifest)


def _collection_cache_matches(
    manifest: dict[str, Any],
    *,
    collection_name: str,
    datastore: str,
    check_hf: bool,
) -> bool:
    return (
        manifest.get('version') == MANIFEST_VERSION
        and manifest.get('collection') == collection_name
        and manifest.get('datastore_input') == datastore
        and manifest.get('hf_checks') is check_hf
        and manifest.get('source_url_mode') == 'online_collection_index_jsonl'
    )


def _manifest_has_duplicate_audit_state(manifest: dict[str, Any]) -> bool:
    audit_statuses = {'already_present', 'score_conflict', AUDIT_ERROR_STATUS}
    for entry in manifest.get('entries', []):
        if not isinstance(entry, dict):
            continue
        if entry.get('status') in audit_statuses:
            return True
        if (
            'duplicate_audit_findings' in entry
            or 'duplicate_audit_errors' in entry
        ):
            return True
    return False


def _load_cached_collection_manifest(
    *,
    manifest_output_path: Path,
    collection_name: str,
    datastore: str,
    check_hf: bool,
) -> dict[str, Any] | None:
    if not manifest_output_path.exists():
        return None
    try:
        manifest = load_manifest(manifest_output_path)
    except (json.JSONDecodeError, OSError) as exc:
        raise HFEvalsError(
            f'Cached manifest is not readable: {manifest_output_path}'
        ) from exc
    if not _collection_cache_matches(
        manifest,
        collection_name=collection_name,
        datastore=datastore,
        check_hf=check_hf,
    ):
        return None
    if _manifest_has_duplicate_audit_state(manifest):
        raise HFEvalsError(
            f'Cached manifest is post-audit but {manifest_output_path.parent / "review.json"} '
            'is missing or does not match. Move the cached output directory aside '
            'before rebuilding.'
        )
    return manifest


def _load_cached_collection_review(
    *,
    review_output_path: Path,
    yaml_output_dir: Path,
    collection_name: str,
    datastore: str,
    check_hf: bool,
) -> dict[str, Any] | None:
    if not review_output_path.exists():
        return None
    try:
        review = json.loads(review_output_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        raise HFEvalsError(
            f'Cached review is not readable: {review_output_path}'
        ) from exc
    if not isinstance(review, dict):
        raise HFEvalsError(f'Cached review must be an object: {review_output_path}')
    manifest = review.get('manifest')
    if not isinstance(manifest, dict):
        raise HFEvalsError(
            f'Cached review is missing its manifest: {review_output_path}'
        )
    if not _collection_cache_matches(
        manifest,
        collection_name=collection_name,
        datastore=datastore,
        check_hf=check_hf,
    ):
        return None
    for field in (
        'duplicate_audit',
        'can_open_prs',
        'audit_blocked_entries',
        'global_audit_errors',
        'missing_hf_models',
    ):
        if field not in review:
            raise HFEvalsError(
                f'Cached review is missing {field}: {review_output_path}'
            )
    yaml_result = _write_yaml_from_manifest(manifest, yaml_output_dir)
    review['yaml_output_dir'] = yaml_output_dir.as_posix()
    review['yaml_count'] = yaml_result['count']
    review['yaml_files'] = yaml_result['written']
    review_output_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return review


def _write_yaml_from_manifest(
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _validate_manifest(manifest)
    grouped: dict[tuple[str, str], tuple[str, str, list[dict[str, Any]]]] = {}
    for entry in manifest['entries']:
        if not _entry_has_yaml_preview(entry):
            continue
        model_repo = entry['model_repo']
        target_path = entry['target_path']
        key = (model_repo.lower(), target_path)
        if key not in grouped:
            grouped[key] = (model_repo, target_path, [])
        grouped[key][2].append(entry['yaml_entry'])

    written: list[str] = []
    for model_repo, target_path, yaml_entries in sorted(grouped.values()):
        path = output_dir / model_repo / target_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_yaml_entries(yaml_entries), encoding='utf-8')
        written.append(path.as_posix())

    return {'written': written, 'count': len(written)}


def write_yaml_from_manifest(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    return _write_yaml_from_manifest(load_manifest(manifest_path), output_dir)


def create_prs_from_manifest(
    manifest_path: Path,
    *,
    limit: int | None,
    yes_i_reviewed: bool,
    commit_message: str,
    api: HfApi | None = None,
    commit_description: str = DEFAULT_PR_COMMIT_DESCRIPTION,
    stream: bool = False,
) -> dict[str, Any]:
    if not yes_i_reviewed:
        raise HFEvalsError('Refusing to create PRs without --yes-i-reviewed.')
    if not commit_message.strip():
        raise HFEvalsError('Commit message must not be empty.')
    if not commit_description.strip():
        raise HFEvalsError('Commit description must not be empty.')
    manifest = load_manifest(manifest_path)
    api = api or HfApi()

    grouped: dict[str, tuple[str, dict[str, list[dict[str, Any]]]]] = {}
    for entry in manifest['entries']:
        if not _entry_is_ready(entry):
            continue
        model_repo = entry['model_repo']
        repo_key = model_repo.lower()
        if repo_key not in grouped:
            grouped[repo_key] = (model_repo, {})
        by_path = grouped[repo_key][1]
        by_path.setdefault(entry['target_path'], []).append(entry['yaml_entry'])

    created: list[dict[str, Any]] = []
    total_repos = len(grouped)
    for repo_index, (model_repo, by_path) in enumerate(
        sorted(grouped.values(), key=lambda item: item[0].lower())
    ):
        if limit is not None and repo_index >= limit:
            break
        if stream:
            print(
                f'[{repo_index + 1}/{total_repos}] preparing {model_repo}',
                flush=True,
            )

        operations: list[CommitOperationAdd] = []
        for target_path, new_entries in sorted(by_path.items()):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=target_path,
                    path_or_fileobj=dump_yaml_entries(new_entries).encode('utf-8'),
                )
            )

        if not operations:
            if stream:
                print(
                    f'[{repo_index + 1}/{total_repos}] no changes {model_repo}',
                    flush=True,
                )
            continue

        try:
            info = api.create_commit(
                repo_id=model_repo,
                repo_type='model',
                operations=operations,
                commit_message=commit_message,
                commit_description=commit_description,
                revision=DEFAULT_DATASTORE_REVISION,
                create_pr=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise HFEvalsError(f'Unable to create PR for {model_repo}') from exc

        pr_url = getattr(info, 'pr_url', None)
        commit_url = getattr(info, 'commit_url', None)
        created.append(
            {
                'model_repo': model_repo,
                'commit': str(info),
                'commit_url': commit_url,
                'pr_url': pr_url,
                'updated_existing_pr': False,
            }
        )
        if stream:
            print(
                f'[{repo_index + 1}/{total_repos}] '
                f'created {model_repo}: '
                f'{pr_url or commit_url or info}',
                flush=True,
            )

    return {
        'created': created,
        'count': len(created),
        'failed': [],
        'failed_count': 0,
        'skipped': [],
        'skipped_count': 0,
    }


APPROVAL_PHRASE = 'OPEN PRS'


def _panel(
    renderable: object,
    *,
    title: str | None = None,
    border_style: str = 'yellow',
) -> Panel:
    return Panel(
        renderable,
        title=title,
        border_style=border_style,
        expand=False,
    )


def _render_interrupted_prompt(console: Console) -> None:
    console.line()
    console.print(_panel('PR submission cancelled.', border_style='yellow'))


def _default_paths(collection_name: str) -> tuple[Path, Path, Path]:
    stem = _safe_collection_name(collection_name).replace(' ', '_')
    base = Path('outputs') / f'community_evals_converter_{stem}'
    return (
        base / 'manifest.json',
        base / 'yamls',
        base / 'review.json',
    )


def _render_summary(console: Console, review: dict) -> None:
    manifest = review['manifest']
    audit = review['duplicate_audit']
    missing_models = review['missing_hf_models']

    table = Table(title='Community Evals Converter', show_header=True, header_style='bold')
    table.add_column('Item')
    table.add_column('Count', justify='right')
    table.add_row('records converted', str(len(manifest['entries'])))
    table.add_row(
        'ready records',
        str(len([entry for entry in manifest['entries'] if _entry_is_ready(entry)])),
    )
    table.add_row(
        'already present',
        str(
            len(
                [
                    entry
                    for entry in manifest['entries']
                    if entry.get('status') == 'already_present'
                ]
            )
        ),
    )
    table.add_row(
        'score conflicts',
        str(
            len(
                [
                    entry
                    for entry in manifest['entries']
                    if entry.get('status') == 'score_conflict'
                ]
            )
        ),
    )
    table.add_row(
        'audit-blocked records',
        str(len(review.get('audit_blocked_entries', []))),
    )
    table.add_row('preview YAML files', str(review['yaml_count']))
    table.add_row('skipped records', str(len(manifest['skipped'])))
    table.add_row('missing HF models', str(len(missing_models)))
    table.add_row('existing score findings', str(audit['finding_count']))
    table.add_row('audit errors', str(audit['error_count']))
    console.print(table)

    console.print(f'Manifest: {review["manifest_path"]}')
    console.print(f'YAML dir:  {review["yaml_output_dir"]}')


def _render_review_details(console: Console, review: dict) -> None:
    max_rows = 20
    rows: list[tuple[str, str, str, str, str | Text]] = []

    def datastore_record_url(path: object) -> object:
        raw_path = str(path or '')
        if not raw_path.startswith('flat/'):
            return path
        manifest = review['manifest']
        datastore_repo = manifest.get('datastore_repo')
        datastore_revision = manifest.get('datastore_revision')
        if not isinstance(datastore_repo, str) or not isinstance(
            datastore_revision,
            str,
        ):
            return path
        return _datastore_blob_url(
            raw_path,
            datastore_repo=datastore_repo,
            datastore_revision=datastore_revision,
        )

    def where_cell(value: object) -> str | Text:
        text = str(value or '')
        if text.startswith(('http://', 'https://')):
            return Text(text, style=f'link {text}')
        return text

    def add(
        issue: str,
        model: object,
        details: object,
        action: str,
        where: object,
    ) -> None:
        if len(rows) >= max_rows:
            return
        rows.append(
            (
                str(issue or ''),
                str(model or ''),
                str(details or ''),
                action,
                where_cell(where),
            )
        )

    for error in review['duplicate_audit']['errors']:
        entry_index = error.get('entry_index')
        action = 'block entry' if isinstance(entry_index, int) else 'block all'
        add(
            'audit_error',
            error.get('model_repo'),
            error.get('error'),
            action,
            error.get('pr_url') or error.get('path') or error.get('stage'),
        )

    findings = review['duplicate_audit']['findings']
    score_conflicts = [
        item for item in findings if item.get('status') == 'score_conflict'
    ]
    already_present = [
        item for item in findings if item.get('status') == 'already_present'
    ]
    for item in score_conflicts:
        where = item.get('existing_path') or item.get('pr_url') or ''
        paths = item.get('paths')
        if paths:
            details = (
                f'{item.get("existing_value")} -> {item.get("candidate_value")}; '
                f'existing score differs from EvalEval; {", ".join(paths)}'
            )
        else:
            details = (
                f'{item.get("existing_value")} -> {item.get("candidate_value")}; '
                'existing score differs from EvalEval.'
            )
        add(
            'score_conflict',
            item.get('model_repo'),
            details,
            'exclude',
            where,
        )

    if already_present:
        add(
            'already_present',
            f'{len(already_present)} models',
            'Same-score result already exists; excluded from PRs.',
            'exclude',
            '.eval_results',
        )

    for entry in review['missing_hf_models']:
        add(
            'missing_hf_model',
            entry.get('model_repo'),
            entry.get('hf_check_error'),
            'exclude',
            entry.get('yaml_entry', {}).get('source', {}).get('url')
            or datastore_record_url(entry.get('eee_record_path')),
        )

    for item in review['manifest']['skipped']:
        line = item.get('collection_index_line') or item.get('index_line') or ''
        add(
            'skipped',
            item.get('model_id'),
            item.get('reason'),
            f'line {line}' if line else 'skip',
            datastore_record_url(
                item.get('eee_record_path') or item.get('object_path')
            ),
        )

    if not rows:
        return

    total = (
        len(review['duplicate_audit']['errors'])
        + len(score_conflicts)
        + (1 if already_present else 0)
        + len(review['missing_hf_models'])
        + len(review['manifest']['skipped'])
    )
    table = Table(
        title='Needs Attention',
        show_header=True,
        header_style='bold cyan',
        show_lines=True,
    )
    table.add_column('Issue', no_wrap=True)
    table.add_column('Model', overflow='fold', ratio=2, max_width=30)
    table.add_column('Details', overflow='fold', ratio=4)
    table.add_column('Action', no_wrap=True)
    table.add_column('Where', no_wrap=True, overflow='ellipsis', ratio=4)
    for row in rows:
        table.add_row(*row)
    if total > len(rows):
        table.caption = (
            f'Showing {len(rows)} of {total} attention items. '
            'Full data is in review JSON.'
        )
    console.print(table)


def _render_not_ready(console: Console, review: dict) -> None:
    audit_blocked_count = len(review.get('audit_blocked_entries', []))
    global_audit_error_count = len(review.get('global_audit_errors', []))
    if global_audit_error_count:
        message = (
            f'{global_audit_error_count} global audit error(s) blocked PR '
            'submission. Local YAML previews were still written when possible.'
        )
    elif audit_blocked_count:
        message = (
            f'{audit_blocked_count} candidate(s) had audit errors, and no '
            'clean ready entries remain. Local YAML previews were still '
            'written for inspection.'
        )
    else:
        message = 'No clean ready entries are available. PRs were not submitted.'
    console.print(
        _panel(
            message,
            title='PRs Not Submitted',
            border_style='yellow',
        )
    )


def _render_ready(console: Console, review: dict) -> None:
    audit_blocked_count = len(review.get('audit_blocked_entries', []))
    message = (
        'Clean ready entries are available. Existing same-score duplicates '
        'and score conflicts have been excluded from submission.'
    )
    if audit_blocked_count:
        message += (
            f'\n\n{audit_blocked_count} candidate(s) had audit errors and '
            'will not be submitted. Their local YAML previews remain under '
            f'{review["yaml_output_dir"]}.'
        )
    console.print(
        _panel(
            message,
            title='Ready',
            border_style='green',
        )
    )


def _prompt_commit_message(console: Console) -> str | None:
    try:
        message = Prompt.ask('Commit message').strip()
    except (EOFError, KeyboardInterrupt):
        _render_interrupted_prompt(console)
        return None
    if not message:
        console.print(
            _panel('Commit message is required.', title='PRs Not Submitted')
        )
        return None
    return message


def _submit_prs(
    console: Console,
    manifest_output: Path,
    *,
    commit_message: str,
) -> int:
    try:
        result = create_prs_from_manifest(
            manifest_path=manifest_output,
            limit=None,
            yes_i_reviewed=True,
            commit_message=commit_message,
            stream=True,
        )
    except HFEvalsError as exc:
        console.print(_panel(str(exc), title='PR Creation Failed', border_style='red'))
        return 1
    console.print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _maybe_submit_prs(
    console: Console,
    review: dict,
    manifest_output: Path,
) -> int:
    if not review['can_open_prs']:
        _render_not_ready(console, review)
        return 0

    _render_ready(console, review)
    if not _approve_pr_submission(console, review):
        return 0
    commit_message = _prompt_commit_message(console)
    if commit_message is None:
        return 0
    return _submit_prs(
        console,
        manifest_output,
        commit_message=commit_message,
    )


def _ready_entries_by_repo(review: dict) -> dict[str, list[str]]:
    by_repo: dict[str, set[str]] = {}
    for entry in review['manifest']['entries']:
        if entry.get('status', 'ready') != 'ready':
            continue
        repo = str(entry['model_repo'])
        by_repo.setdefault(repo, set()).add(str(entry['target_path']))
    return {
        repo: sorted(paths)
        for repo, paths in sorted(by_repo.items(), key=lambda item: item[0].lower())
    }


def _approve_pr_submission(console: Console, review: dict) -> bool:
    by_repo = _ready_entries_by_repo(review)
    if not by_repo:
        console.print(_panel('No ready entries to submit.', border_style='yellow'))
        return False

    table = Table(
        title='PR Submission Approval',
        show_header=True,
        header_style='bold',
        show_lines=True,
    )
    table.add_column('Model repo')
    table.add_column('Files')
    for repo, paths in by_repo.items():
        table.add_row(repo, '\n'.join(paths))
    console.print(table)
    console.print(
        _panel(
            f'Type {APPROVAL_PHRASE!r} to submit these PRs. '
            'Anything else cancels.',
            title='Approval Required',
            border_style='yellow',
        )
    )
    try:
        answer = Prompt.ask('Approval').strip()
    except (EOFError, KeyboardInterrupt):
        _render_interrupted_prompt(console)
        return False
    if answer != APPROVAL_PHRASE:
        console.print(_panel('PR submission cancelled.', border_style='yellow'))
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Review an EEE datastore collection for HF Community Evals.',
    )
    parser.add_argument(
        'collection_name',
        help='Collection file stem under flat/indexes/by_collection/<name>.jsonl.',
    )
    parser.add_argument(
        '--datastore',
        default=DEFAULT_DATASTORE_REPO,
        help=(
            'Online HF dataset locator in the form <repo> or '
            '<repo>@<revision>. Defaults to evaleval/EEE_datastore and '
            'resolves the current main commit.'
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Ignore cached review/manifest outputs and rebuild from datastore.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        collection_name = _safe_collection_name(args.collection_name)
        manifest_output, yaml_dir, review_output = _default_paths(collection_name)
        with Progress(
            SpinnerColumn(),
            TextColumn(
                '[bold blue]{task.description}',
                table_column=Column(width=48, no_wrap=True, overflow='ellipsis'),
            ),
            BarColumn(bar_width=28),
            TextColumn(
                '{task.completed:>4.0f}/{task.total:<4.0f}',
                table_column=Column(width=10, no_wrap=True),
            ),
            TimeElapsedColumn(),
            console=console,
            expand=False,
        ) as rich_progress:
            review = review_collection_for_hf_evals(
                collection_name=collection_name,
                datastore=args.datastore,
                manifest_output_path=manifest_output,
                yaml_output_dir=yaml_dir,
                review_output_path=review_output,
                progress=RichReviewProgress(rich_progress),
                force=args.force,
            )
    except HFEvalsError as exc:
        console.print(_panel(str(exc), title='Review Failed', border_style='red'))
        return 1

    _render_summary(console, review)
    _render_review_details(console, review)
    console.print(f'Review JSON: {review_output.as_posix()}')

    return _maybe_submit_prs(console, review, manifest_output)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
