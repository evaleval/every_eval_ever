#!/usr/bin/env python3
"""Convert OpenEval response data into Every Eval Ever aggregate records.

Data source:
- OpenEval dataset: https://huggingface.co/datasets/human-centered-eval/OpenEval

Usage:
    uv run python -m utils.openeval.adapter --output-dir data/openeval
    uv run python -m utils.openeval.adapter --include-instances
    uv run python -m utils.openeval.adapter --input-json sample.json

The offline JSON payload shape is:
    {
      "bench": [...],
      "response": [...]
    }
where rows match the Hugging Face ``bench`` and ``response`` table rows.
Pass ``--include-instances`` with an ``item`` collection to write sibling
``*_samples.jsonl`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    GenerationArgs,
    GenerationConfig,
    HashAlgorithm,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    datastore_repo_file_path,
    get_developer,
    get_model_id,
    require_identity,
    sanitize_filename,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    datastore_output_dir,
    default_failure_report_path,
    save_failure_report,
)
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
)

HF_REPO_ID = 'human-centered-eval/OpenEval'
HF_REVISION = 'main'
DEFAULT_OUTPUT_DIR = 'data/openeval'
SOURCE_NAME = 'OpenEval'
SOURCE_ORGANIZATION = 'Human-Centered Eval'
SOURCE_ORGANIZATION_URL = 'https://open-eval.github.io/'
HF_DATASET_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}'
GITHUB_URL = 'https://github.com/open-eval/OpenEval'


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str
    instance_path: Path | None = None
    instance_count: int = 0
    binary_result_ids: set[str] = field(default_factory=set)


@dataclass
class MetricAccumulator:
    benchmark: dict[str, Any]
    metric_name: str
    values: list[float] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    metric_models: set[str] = field(default_factory=set)
    extra_artifact_types: set[str] = field(default_factory=set)
    sample_ids: list[str] = field(default_factory=list)

    def add(
        self, value: float, response_id: str, metric: dict[str, Any]
    ) -> None:
        self.values.append(value)
        self.response_ids.append(response_id)
        sample_id = response_item_id(response_id)
        if sample_id is not None:
            self.sample_ids.append(sample_id)
        models = metric.get('models')
        if isinstance(models, list):
            self.metric_models.update(str(model) for model in models if model)
        artifacts = metric.get('extra_artifacts')
        if isinstance(artifacts, dict) and isinstance(
            artifacts.get('type'), list
        ):
            self.extra_artifact_types.update(
                str(kind) for kind in artifacts['type'] if kind
            )


@dataclass
class ModelGroup:
    generation_params: dict[str, Any]
    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)
    instance_path: Path | None = None
    instance_handle: TextIO | None = None
    instance_count: int = 0


@dataclass(frozen=True)
class PendingInstance:
    response_id: str
    sample_id: str
    benchmark: dict[str, Any]
    metric_name: str
    score: float
    raw_input: str
    references: list[str]
    output: list[str]
    metadata: dict[str, str]


@dataclass(frozen=True)
class OpenEvalAggregationResult:
    groups: dict[tuple[str, str, str], ModelGroup]
    total_responses: int
    failures: list[SourceRecordFailure]
    exclusions: list[SourceRecordExclusion]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert OpenEval HF dataset results to EEE format.'
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Read an offline JSON payload instead of fetching from HF.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--revision',
        default=HF_REVISION,
        help=f'Hugging Face dataset revision (default: {HF_REVISION}).',
    )
    parser.add_argument(
        '--limit-responses',
        type=int,
        help='Limit live/offline responses for smoke runs.',
    )
    parser.add_argument(
        '--max-response-shards',
        type=int,
        help='Limit downloaded HF response parquet shards for smoke runs.',
    )
    parser.add_argument(
        '--allow-unknown-benchmark',
        action='store_true',
        help='Keep responses whose benchmark cannot be matched from response_id.',
    )
    parser.add_argument(
        '--include-instances',
        action='store_true',
        help='Also write instance-level *_samples.jsonl files.',
    )
    return parser.parse_args()


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    return {
        key: stringify(value)
        for key, value in details.items()
        if value not in (None, '')
    }


def normalize_slug(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw)
    raw = raw.replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    raw = re.sub(r'-{2,}', '-', raw).strip('-')
    return raw or 'unknown'


def load_payload(input_json: Path) -> dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('--input-json must contain a JSON object.')
    return payload


def extract_collection(payload: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f'Expected {name!r} payload to be a list or object.')

    for key in (name, f'{name}s', 'data', 'rows', 'items', 'results'):
        value = payload.get(key)
        if isinstance(value, list):
            rows = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('row'), dict):
                    rows.append(item['row'])
                elif isinstance(item, dict):
                    rows.append(item)
            return rows
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(item, dict) for item in value.values())
        ):
            return list(value.values())

    if payload and all(isinstance(item, dict) for item in payload.values()):
        return list(payload.values())

    raise ValueError(f'Could not find a list of {name!r} records.')


def validate_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], Iterable[Any]]:
    benches = extract_collection(payload.get('bench'), 'bench')
    response_payload = payload.get('response') or payload.get('responses')
    responses: Iterable[Any]
    if isinstance(response_payload, list):
        responses = [
            item['row']
            if isinstance(item, dict) and isinstance(item.get('row'), dict)
            else item
            for item in response_payload
        ]
    elif isinstance(response_payload, dict):
        responses = extract_collection(response_payload, 'response')
    elif response_payload is not None:
        responses = response_payload
    else:
        raise ValueError('Could not find a list of response records.')
    return benches, responses


def item_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    item_payload = payload.get('item') or payload.get('items')
    if item_payload is None:
        return []
    return extract_collection(item_payload, 'item')


def build_index(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    index = {}
    for row in rows:
        value = row.get(key)
        if value not in (None, ''):
            index[str(value)] = row
    return index


def build_benchmark_index(
    benches: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index = build_index(benches, 'benchmark_name')
    for benchmark in benches:
        name = benchmark.get('benchmark_name')
        if name not in (None, ''):
            index[normalize_slug(name)] = benchmark
    return index


def fetch_payload(
    revision: str = HF_REVISION,
    max_response_shards: int | None = None,
    include_instances: bool = False,
) -> dict[str, Any]:
    """Download public OpenEval parquet shards and aggregate them to rows.

    The project already depends on ``huggingface_hub`` and ``duckdb``. Using
    parquet shards avoids a large dependency on ``datasets`` and avoids the
    HF row API pagination path for the 500k+ response table.
    """
    try:
        import duckdb
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            'OpenEval live fetch requires huggingface_hub and duckdb.'
        ) from exc

    api = HfApi()
    info = api.dataset_info(HF_REPO_ID, revision=revision)
    files = api.list_repo_files(
        HF_REPO_ID, repo_type='dataset', revision=revision
    )
    bench_files = [path for path in files if path.startswith('bench/')]
    item_files = sorted(path for path in files if path.startswith('item/'))
    response_files = sorted(
        path for path in files if path.startswith('response/')
    )
    total_response_shards = len(response_files)
    if max_response_shards is not None:
        response_files = response_files[:max_response_shards]
    if not bench_files or not response_files:
        raise ValueError(
            f'Could not find bench/response parquet files in {HF_REPO_ID}.'
        )
    if include_instances and not item_files:
        raise ValueError(f'Could not find item parquet files in {HF_REPO_ID}.')

    local_bench = [
        hf_hub_download(
            HF_REPO_ID,
            path,
            repo_type='dataset',
            revision=revision,
        )
        for path in bench_files
    ]
    local_response = [
        hf_hub_download(
            HF_REPO_ID,
            path,
            repo_type='dataset',
            revision=revision,
        )
        for path in response_files
    ]
    local_item = []
    if include_instances:
        local_item = [
            hf_hub_download(
                HF_REPO_ID,
                path,
                repo_type='dataset',
                revision=revision,
            )
            for path in item_files
        ]

    con = duckdb.connect()
    bench_cursor = con.execute('SELECT * FROM read_parquet(?)', [local_bench])
    bench_columns = [item[0] for item in bench_cursor.description]
    benches = [dict(zip(bench_columns, row)) for row in bench_cursor.fetchall()]
    items = []
    if include_instances:
        item_cursor = con.execute('SELECT * FROM read_parquet(?)', [local_item])
        item_columns = [item[0] for item in item_cursor.description]
        items = [dict(zip(item_columns, row)) for row in item_cursor.fetchall()]
    # Keep response rows lazy. The full response table is large enough that
    # materializing it would make the adapter harder to run on ordinary laptops.
    # Payloads returned by this live fetch path are therefore consumed once by
    # make_logs().
    responses = _response_rows_from_parquet(
        con, local_response, include_instances=include_instances
    )
    return {
        'bench': benches,
        'item': items,
        'response': responses,
        'source_metadata': {
            'hf_revision': revision,
            'hf_commit': getattr(info, 'sha', None),
            'downloaded_response_shards': len(response_files),
            'total_response_shards': total_response_shards,
            'max_response_shards': max_response_shards,
            'include_instances': include_instances,
        },
    }


def _response_rows_from_parquet(
    con: Any,
    parquet_paths: list[str],
    include_instances: bool = False,
) -> Iterable[dict[str, Any]]:
    columns = ['response_id', 'model', 'scores']
    if include_instances:
        columns.extend(['item_adaptation', 'response_content'])
    query = (
        f'SELECT {", ".join(columns)} '
        'FROM read_parquet(?) '
        'WHERE scores IS NOT NULL'
    )
    cursor = con.execute(query, [parquet_paths])
    names = [item[0] for item in cursor.description]
    while True:
        rows = cursor.fetchmany(10000)
        if not rows:
            break
        for row in rows:
            yield dict(zip(names, row))


def response_benchmark_id(response_id: str) -> str | None:
    """Extract the benchmark prefix from an OpenEval response id.

    OpenEval response ids start with the item id. The item id has the shape
    ``<benchmark>_<timestamp>_<row_index>``, while the response id appends
    model/run suffixes after that.
    """
    match = re.match(r'^(.+?)_\d{8}T\d{6}Z_\d+(?:_|$)', response_id)
    if match:
        return match.group(1)
    return None


def response_item_id(response_id: str) -> str | None:
    match = re.match(r'^(.+?_\d{8}T\d{6}Z_\d+)(?:_|$)', response_id)
    if match:
        return match.group(1)
    return None


def benchmark_for_response_id(
    response_id: str, benchmark_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    parsed = response_benchmark_id(response_id)
    if parsed:
        for candidate in (parsed, normalize_slug(parsed)):
            if candidate in benchmark_index:
                return benchmark_index[candidate]

    for benchmark_name in sorted(benchmark_index, key=len, reverse=True):
        if response_id.startswith(f'{benchmark_name}_'):
            return benchmark_index[benchmark_name]
    return {
        'benchmark_name': 'unknown',
        'benchmark_version': '',
        'paper_url': None,
        'dataset_url': HF_DATASET_URL,
        'benchmark_tags': [],
    }


def numeric_score_values(
    scores: Any,
) -> list[tuple[str, float, dict[str, Any]]]:
    result = numeric_score_values_result(scores, 'OpenEval score')
    result.raise_if_incomplete()
    return result.records


def numeric_score_values_result(
    scores: Any,
    source_ref: str,
) -> SourceConversionResult[tuple[str, float, dict[str, Any]]]:
    """Parse every metric independently and retain malformed score details."""
    failures: list[SourceRecordFailure] = []
    if not isinstance(scores, dict):
        return SourceConversionResult(
            source_name=f'{source_ref} metrics',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='scores must be an object',
                    source_record=scores,
                )
            ],
        )
    metrics = scores.get('metric') or []
    values = scores.get('value') or []
    if not isinstance(metrics, list) or not isinstance(values, list):
        return SourceConversionResult(
            source_name=f'{source_ref} metrics',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='scores.metric and scores.value must be lists',
                    source_record=scores,
                )
            ],
        )

    pairs: list[tuple[str, float, dict[str, Any]]] = []
    total_records = max(len(metrics), len(values))
    if total_records == 0:
        failures.append(
            SourceRecordFailure(
                source_ref=source_ref,
                reason='scores contains no metric/value entries',
                source_record=scores,
            )
        )
        total_records = 1
    for index in range(max(len(metrics), len(values))):
        metric_ref = f'{source_ref} metric {index}'
        try:
            metric = metrics[index]
            value = values[index]
            if not isinstance(metric, dict):
                raise ValueError('metric definition must be an object')
            name = metric.get('name')
            if name in (None, ''):
                raise ValueError('metric name is missing')
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f'score must be finite, got {value!r}')
            pairs.append((str(name), score, metric))
        except (IndexError, TypeError, ValueError) as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=metric_ref,
                    reason=str(exc),
                    source_record={
                        'metric': metrics[index]
                        if index < len(metrics)
                        else None,
                        'value': values[index] if index < len(values) else None,
                    },
                )
            )
    return SourceConversionResult(
        source_name=f'{source_ref} metrics',
        total_records=total_records,
        records=pairs,
        failures=failures,
    )


def model_name(response: dict[str, Any]) -> str:
    model = response.get('model')
    if isinstance(model, dict):
        value = model.get('name')
        if value not in (None, ''):
            return str(value)
    return 'unknown'


def model_size(response: dict[str, Any]) -> str | None:
    model = response.get('model')
    if isinstance(model, dict) and model.get('size') not in (None, ''):
        return str(model['size'])
    return None


def generation_parameters(response: dict[str, Any]) -> dict[str, Any]:
    model = response.get('model')
    if not isinstance(model, dict):
        return {}
    adaptation = model.get('model_adaptation')
    if not isinstance(adaptation, dict):
        return {}
    params = adaptation.get('generation_parameters')
    if isinstance(params, dict):
        return params
    if isinstance(params, str) and params.strip():
        try:
            decoded = json.loads(params)
        except json.JSONDecodeError:
            return {'raw_generation_parameters': params}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def generation_key(params: dict[str, Any]) -> str:
    if not params:
        return 'default'
    blob = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


def make_generation_config(params: dict[str, Any]) -> GenerationConfig | None:
    if not params:
        return None

    def maybe_float(name: str) -> float | None:
        value = params.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def maybe_int(name: str) -> int | None:
        value = params.get(name)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    additional = {
        key: value
        for key, value in params.items()
        if key not in {'temperature', 'top_p', 'top_k', 'max_tokens'}
    }
    return GenerationConfig(
        generation_args=GenerationArgs(
            temperature=maybe_float('temperature'),
            top_p=maybe_float('top_p'),
            top_k=maybe_float('top_k'),
            max_tokens=maybe_int('max_tokens'),
        ),
        additional_details=stringify_details(additional) or None,
    )


def normalize_model_info(
    name: str, size: str | None
) -> tuple[ModelInfo, str, str]:
    name = require_identity(name, 'OpenEval model name')
    developer = require_identity(
        get_developer(name), 'OpenEval model developer'
    )
    model_id = require_identity(
        get_model_id(name, developer), 'OpenEval model id'
    )
    model_slug = require_identity(
        normalize_slug(model_id.split('/', 1)[-1], name),
        'OpenEval model path name',
    )
    details = stringify_details({'raw_model_name': name, 'model_size': size})
    return (
        ModelInfo(
            name=name,
            id=model_id,
            developer=developer,
            additional_details=details,
        ),
        normalize_slug(developer),
        model_slug,
    )


def aggregate_scores_result(
    benches: list[dict[str, Any]],
    responses: Iterable[Any],
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
    items: list[dict[str, Any]] | None = None,
    include_instances: bool = False,
) -> OpenEvalAggregationResult:
    benchmark_index = build_benchmark_index(benches)
    item_index = build_index(items or [], 'item_id')
    groups: dict[tuple[str, str, str], ModelGroup] = {}
    seen_results: set[tuple[str, str]] = set()
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    total_responses = 0

    try:
        for count, response in enumerate(responses, start=1):
            if limit_responses is not None and count > limit_responses:
                break
            total_responses += 1
            if not isinstance(response, dict):
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'OpenEval response row {count}',
                        reason='response must be an object',
                        source_record=response,
                    )
                )
                continue
            response_id = str(response.get('response_id') or '')
            if not response_id:
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'OpenEval response row {count}',
                        reason='response_id is missing',
                        source_record=response,
                    )
                )
                continue

            benchmark = benchmark_for_response_id(response_id, benchmark_index)
            if (
                benchmark.get('benchmark_name') == 'unknown'
                and not allow_unknown_benchmark
            ):
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'OpenEval response {response_id!r}',
                        reason=(
                            f'Could not match OpenEval response_id '
                            f'{response_id!r} to a benchmark. Pass '
                            '--allow-unknown-benchmark to keep unmatched rows '
                            'under the unknown benchmark.'
                        ),
                        source_record=response,
                    )
                )
                continue
            name = model_name(response)
            size = model_size(response)
            params = generation_parameters(response)
            key = (name, size or '', generation_key(params))
            group = groups.setdefault(
                key,
                ModelGroup(generation_params=params),
            )

            sample_id = response_item_id(response_id) or response_id
            score_result = numeric_score_values_result(
                response.get('scores'),
                f'OpenEval response {response_id!r}',
            )
            failures.extend(score_result.failures)
            for metric_name, score, metric in score_result.records:
                seen_key = (name, response_id, metric_name)
                if seen_key in seen_results:
                    exclusions.append(
                        SourceRecordExclusion(
                            source_ref=(
                                f'OpenEval response {response_id!r} metric '
                                f'{metric_name!r}'
                            ),
                            reason='duplicate model/response/metric score',
                            source_record=response,
                        )
                    )
                    continue
                accumulator_key = result_key(benchmark, metric_name)
                accumulator = group.metrics.setdefault(
                    accumulator_key,
                    MetricAccumulator(
                        benchmark=benchmark, metric_name=metric_name
                    ),
                )
                try:
                    if include_instances:
                        item = item_index.get(sample_id)
                        append_pending_instance(
                            group,
                            PendingInstance(
                                response_id=response_id,
                                sample_id=sample_id,
                                benchmark=benchmark,
                                metric_name=metric_name,
                                score=score,
                                raw_input=input_text(item, response),
                                references=reference_texts(item),
                                output=response_texts(response),
                                metadata=instance_metadata(
                                    response_id,
                                    benchmark,
                                    metric_name,
                                    metric,
                                    item,
                                    response,
                                ),
                            ),
                        )
                    accumulator.add(score, response_id, metric)
                    seen_results.add(seen_key)
                except Exception as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=(
                                f'OpenEval response {response_id!r} metric '
                                f'{metric_name!r}'
                            ),
                            reason=str(exc),
                            source_record=response,
                        )
                    )
    finally:
        close_instance_files(groups)

    return OpenEvalAggregationResult(
        groups=groups,
        total_responses=total_responses,
        failures=failures,
        exclusions=exclusions,
    )


def result_key(benchmark: dict[str, Any], metric_name: str) -> str:
    return (
        f'{normalize_slug(benchmark.get("benchmark_name"))}.'
        f'{normalize_slug(metric_name)}'
    )


def values_are_binary(values: list[float]) -> bool:
    return bool(values) and all(value in {0.0, 1.0} for value in values)


def metric_bounds(values: list[float]) -> tuple[float, float, str, str]:
    if values_are_binary(values):
        return 0.0, 1.0, 'proportion', 'binary_values'
    if values and all(0.0 <= value <= 1.0 for value in values):
        return 0.0, 1.0, 'score', 'normalized_observed_values'
    observed_min = min(values) if values else 0.0
    observed_max = max(values) if values else 1.0
    return (
        min(0.0, observed_min),
        max(1.0, observed_max),
        'points',
        'observed_values',
    )


def benchmark_name(benchmark: dict[str, Any]) -> str:
    return str(benchmark.get('benchmark_name') or 'unknown')


def benchmark_url(benchmark: dict[str, Any]) -> str | None:
    for key in ('dataset_url', 'paper_url'):
        value = benchmark.get(key)
        if isinstance(value, str) and value.startswith(('http://', 'https://')):
            return value
    return None


def make_evaluation_result(
    accumulator: MetricAccumulator,
    generation_config: GenerationConfig | None,
) -> EvaluationResult:
    values = accumulator.values
    score = sum(values) / len(values)
    min_score, max_score, unit, bounds_source = metric_bounds(values)
    benchmark = accumulator.benchmark
    bench_slug = normalize_slug(benchmark_name(benchmark))
    metric_slug = normalize_slug(accumulator.metric_name)
    unique_sample_count = len(set(accumulator.sample_ids)) or len(values)
    stddev = None
    stderr = None
    if len(values) > 1:
        variance = sum((value - score) ** 2 for value in values) / (
            len(values) - 1
        )
        stddev = math.sqrt(variance)
        stderr = stddev / math.sqrt(len(values))

    urls = [HF_DATASET_URL, GITHUB_URL]
    extra_url = benchmark_url(benchmark)
    if extra_url:
        urls.append(extra_url)

    tags = benchmark.get('benchmark_tags')
    return EvaluationResult(
        evaluation_result_id=f'{bench_slug}::{metric_slug}',
        evaluation_name=f'openeval.{bench_slug}.{metric_slug}',
        source_data=SourceDataHf(
            dataset_name=benchmark_name(benchmark),
            source_type='hf_dataset',
            hf_repo=HF_REPO_ID,
            hf_split='train',
            samples_number=unique_sample_count,
            additional_details=stringify_details(
                {
                    'benchmark_name': benchmark_name(benchmark),
                    'benchmark_version': benchmark.get('benchmark_version'),
                    'paper_url': benchmark.get('paper_url'),
                    'dataset_url': benchmark.get('dataset_url'),
                    'source_urls_json': urls,
                }
            ),
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f'Mean OpenEval score for {accumulator.metric_name} '
                f'on {benchmark_name(benchmark)}.'
            ),
            metric_id=f'openeval.{bench_slug}.{metric_slug}',
            metric_name=accumulator.metric_name,
            metric_kind='benchmark_score',
            metric_unit=unit,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=min_score,
            max_score=max_score,
            additional_details=stringify_details(
                {
                    'benchmark_tags_json': tags,
                    'aggregation': 'mean',
                    'raw_metric_name': accumulator.metric_name,
                    'response_count': len(values),
                    'unique_sample_count': unique_sample_count,
                    'score_values_are_binary': values_are_binary(values),
                    'bounds_source': bounds_source,
                    'metric_models_json': sorted(accumulator.metric_models),
                    'extra_artifact_types_json': sorted(
                        accumulator.extra_artifact_types
                    ),
                }
            ),
        ),
        score_details=ScoreDetails(
            score=score,
            uncertainty=Uncertainty(
                standard_error=StandardError(value=stderr, method='analytic')
                if stderr is not None
                else None,
                standard_deviation=stddev,
                num_samples=len(values),
            ),
            details=stringify_details(
                {
                    'min_instance_score': min(values),
                    'max_instance_score': max(values),
                    'response_count': len(values),
                    'example_response_ids_json': accumulator.response_ids[:5],
                }
            ),
        ),
        generation_config=generation_config,
    )


def list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [stringify(item) for item in value if item not in (None, '')]
    if value in (None, ''):
        return []
    return [stringify(value)]


def maybe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def item_content(item: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    content = item.get('item_content')
    return content if isinstance(content, dict) else {}


def input_text(item: dict[str, Any] | None, response: dict[str, Any]) -> str:
    content = item_content(item)
    inputs = list_of_strings(content.get('input'))
    if inputs:
        return '\n'.join(inputs)

    adaptation = response.get('item_adaptation')
    if isinstance(adaptation, dict):
        request_input = list_of_strings(adaptation.get('request_input'))
        if request_input:
            return '\n'.join(request_input)
    return ''


def reference_texts(item: dict[str, Any] | None) -> list[str]:
    content = item_content(item)
    references = []
    for raw in list_of_strings(content.get('references')):
        parsed = maybe_json(raw)
        if isinstance(parsed, dict):
            output = parsed.get('output')
            if isinstance(output, dict) and output.get('text') not in (
                None,
                '',
            ):
                references.append(str(output['text']))
                continue
        references.append(raw)
    return references


def response_texts(response: dict[str, Any]) -> list[str]:
    texts = []
    for raw in list_of_strings(response.get('response_content')):
        parsed = maybe_json(raw)
        if isinstance(parsed, dict) and parsed.get('text') not in (None, ''):
            texts.append(str(parsed['text']))
        else:
            texts.append(raw)
    return texts


def sample_hash(raw_input: str, references: list[str]) -> str:
    payload = json.dumps(
        {'raw': raw_input, 'reference': references},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def instance_metadata(
    response_id: str,
    benchmark: dict[str, Any],
    metric_name: str,
    metric: dict[str, Any],
    item: dict[str, Any] | None,
    response: dict[str, Any],
) -> dict[str, str]:
    item = item if isinstance(item, dict) else {}
    item_meta = item.get('item_metadata') if isinstance(item, dict) else None
    item_adaptation = response.get('item_adaptation')
    metadata = {
        'response_id': response_id,
        'benchmark_name': benchmark_name(benchmark),
        'metric_name': metric_name,
        'raw_metric_name': metric.get('name'),
        'metric_models_json': metric.get('models') or [],
        'extra_artifact_types_json': (
            metric.get('extra_artifacts', {}).get('type')
            if isinstance(metric.get('extra_artifacts'), dict)
            else []
        ),
        'item_schema_version': item.get('schema_version'),
        'item_metadata_json': item_meta
        if isinstance(item_meta, dict)
        else None,
        'item_adaptation_json': (
            item_adaptation if isinstance(item_adaptation, dict) else None
        ),
    }
    return stringify_details(metadata)


def make_instance_log(
    instance: PendingInstance,
    evaluation_id: str,
    model_id: str,
    binary_result_ids: set[str] | None = None,
) -> InstanceLevelEvaluationLog:
    bench_slug = normalize_slug(benchmark_name(instance.benchmark))
    metric_slug = normalize_slug(instance.metric_name)
    evaluation_result_id = f'{bench_slug}::{metric_slug}'
    extracted_value = instance.output[0] if instance.output else ''
    is_binary_metric = evaluation_result_id in (binary_result_ids or set())
    metadata = {
        **instance.metadata,
        'is_correct_applicable': stringify(is_binary_metric),
        'is_correct_rule': (
            'score == 1.0 for binary 0/1 metric'
            if is_binary_metric
            else 'false for non-binary metric; use evaluation.score'
        ),
    }

    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        model_id=model_id,
        evaluation_name=f'openeval.{bench_slug}.{metric_slug}',
        evaluation_result_id=evaluation_result_id,
        sample_id=instance.sample_id,
        sample_hash=sample_hash(instance.raw_input, instance.references),
        interaction_type=InteractionType.single_turn,
        input=Input(raw=instance.raw_input, reference=instance.references),
        output=Output(raw=instance.output),
        answer_attribution=[
            AnswerAttributionItem(
                turn_idx=0,
                source='output.raw',
                extracted_value=extracted_value,
                extraction_method=instance.metric_name,
                is_terminal=True,
            )
        ],
        evaluation=Evaluation(
            score=instance.score,
            is_correct=is_binary_metric and instance.score == 1.0,
        ),
        metadata=metadata,
    )


def pending_instance_to_json(instance: PendingInstance) -> str:
    return json.dumps(
        {
            'response_id': instance.response_id,
            'sample_id': instance.sample_id,
            'benchmark': instance.benchmark,
            'metric_name': instance.metric_name,
            'score': instance.score,
            'raw_input': instance.raw_input,
            'references': instance.references,
            'output': instance.output,
            'metadata': instance.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def pending_instance_from_json(line: str) -> PendingInstance:
    data = json.loads(line)
    return PendingInstance(
        response_id=str(data['response_id']),
        sample_id=str(data['sample_id']),
        benchmark=data['benchmark'],
        metric_name=str(data['metric_name']),
        score=float(data['score']),
        raw_input=str(data['raw_input']),
        references=list_of_strings(data.get('references')),
        output=list_of_strings(data.get('output')),
        metadata=stringify_details(data.get('metadata') or {}),
    )


def append_pending_instance(
    group: ModelGroup, instance: PendingInstance
) -> None:
    if group.instance_path is None:
        handle = tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            prefix='openeval-instances-',
            suffix='.jsonl',
            delete=False,
        )
        group.instance_path = Path(handle.name)
        group.instance_handle = handle

    if group.instance_handle is None:
        group.instance_handle = group.instance_path.open('a', encoding='utf-8')
    group.instance_handle.write(pending_instance_to_json(instance) + '\n')
    group.instance_count += 1


def close_instance_files(
    groups: dict[tuple[str, str, str], ModelGroup],
) -> None:
    for group in groups.values():
        if group.instance_handle is not None:
            group.instance_handle.close()
            group.instance_handle = None


def source_metadata(
    revision: str,
    payload_metadata: dict[str, Any] | None = None,
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
    include_instances: bool = False,
) -> SourceMetadata:
    payload_metadata = payload_metadata or {}
    downloaded_shards = payload_metadata.get('downloaded_response_shards')
    total_shards = payload_metadata.get('total_response_shards')
    partial_export = (
        limit_responses is not None
        or payload_metadata.get('max_response_shards') is not None
        or (
            isinstance(downloaded_shards, int)
            and isinstance(total_shards, int)
            and downloaded_shards < total_shards
        )
    )
    return SourceMetadata(
        source_name=SOURCE_NAME,
        source_type='evaluation_run',
        source_organization_name=SOURCE_ORGANIZATION,
        source_organization_url=SOURCE_ORGANIZATION_URL,
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details=stringify_details(
            {
                'hf_repo': HF_REPO_ID,
                'hf_dataset_url': HF_DATASET_URL,
                'github_url': GITHUB_URL,
                'hf_revision': revision,
                'hf_commit': payload_metadata.get('hf_commit'),
                'downloaded_response_shards': payload_metadata.get(
                    'downloaded_response_shards'
                ),
                'total_response_shards': payload_metadata.get(
                    'total_response_shards'
                ),
                'max_response_shards': payload_metadata.get(
                    'max_response_shards'
                ),
                'limit_responses': limit_responses,
                'include_instances': include_instances
                or payload_metadata.get('include_instances'),
                'partial_export': partial_export,
                'allow_unknown_benchmark': allow_unknown_benchmark,
                'source_role': 'aggregator',
            }
        ),
    )


def _make_log_bundle(
    name: str,
    size: str,
    gen_key: str,
    group: ModelGroup,
    metadata: SourceMetadata,
    timestamp: str,
) -> LogBundle | None:
    results = [metric for metric in group.metrics.values() if metric.values]
    if not results:
        return None
    model_info, developer, model_slug = normalize_model_info(name, size or None)
    generation_config = make_generation_config(group.generation_params)
    if generation_config is not None:
        details = generation_config.additional_details or {}
        generation_config.additional_details = {
            **details,
            'generation_config_hash': gen_key,
        }
    results = [
        make_evaluation_result(metric, generation_config) for metric in results
    ]

    sanitized_model_id = model_info.id.replace('/', '_')
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(f'openeval/{sanitized_model_id}/{gen_key}/{timestamp}'),
        retrieved_timestamp=timestamp,
        source_metadata=metadata,
        eval_library=EvalLibrary(name='OpenEval', version='unknown'),
        model_info=model_info,
        evaluation_results=sorted(
            results, key=lambda item: item.evaluation_result_id or ''
        ),
    )
    binary_result_ids = {
        result.evaluation_result_id
        for result in results
        if result.evaluation_result_id
        and (result.metric_config.additional_details or {}).get(
            'score_values_are_binary'
        )
        == 'true'
    }
    return LogBundle(
        log=log,
        developer=developer,
        model=model_slug,
        instance_path=group.instance_path,
        instance_count=group.instance_count,
        binary_result_ids=binary_result_ids,
    )


def make_logs(
    payload: dict[str, Any],
    retrieved_timestamp: str | None = None,
    revision: str = HF_REVISION,
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
    include_instances: bool = False,
) -> list[LogBundle]:
    result = make_logs_result(
        payload,
        retrieved_timestamp=retrieved_timestamp,
        revision=revision,
        limit_responses=limit_responses,
        allow_unknown_benchmark=allow_unknown_benchmark,
        include_instances=include_instances,
    )
    if result.failures:
        for bundle in result.records:
            if bundle.instance_path is not None:
                bundle.instance_path.unlink(missing_ok=True)
    result.raise_if_incomplete()
    return result.records


def make_logs_result(
    payload: dict[str, Any],
    retrieved_timestamp: str | None = None,
    revision: str = HF_REVISION,
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
    include_instances: bool = False,
) -> SourceConversionResult[LogBundle]:
    benches, responses = validate_payload(payload)
    items = item_rows(payload) if include_instances else []
    if include_instances and not items:
        raise ValueError(
            'OpenEval instance-level output requires item records. '
            'Include an "item" collection in --input-json or use live fetch.'
        )
    timestamp = retrieved_timestamp or str(time.time())
    aggregation = aggregate_scores_result(
        benches,
        responses,
        limit_responses,
        allow_unknown_benchmark=allow_unknown_benchmark,
        items=items,
        include_instances=include_instances,
    )
    payload_metadata = (
        payload.get('source_metadata')
        if isinstance(payload.get('source_metadata'), dict)
        else {}
    )
    metadata = source_metadata(
        revision,
        payload_metadata,
        limit_responses,
        allow_unknown_benchmark,
        include_instances,
    )

    bundles: list[LogBundle] = []
    failures = list(aggregation.failures)
    for (name, size, gen_key), group in sorted(aggregation.groups.items()):
        try:
            bundle = _make_log_bundle(
                name,
                size,
                gen_key,
                group,
                metadata,
                timestamp,
            )
            if bundle is not None:
                bundles.append(bundle)
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'OpenEval model group {name!r}/{gen_key}',
                    reason=str(exc),
                    source_record={
                        'model_name': name,
                        'model_size': size,
                        'generation_parameters': group.generation_params,
                        'response_ids': sorted(
                            {
                                response_id
                                for metric in group.metrics.values()
                                for response_id in metric.response_ids
                            }
                        ),
                    },
                )
            )
            if group.instance_path is not None:
                group.instance_path.unlink(missing_ok=True)

    if not bundles and not failures:
        raise ValueError('OpenEval: converted 0 source records')
    return SourceConversionResult(
        source_name='OpenEval responses',
        total_records=aggregation.total_responses,
        records=bundles,
        failures=failures,
        exclusions=aggregation.exclusions,
    )


def export_logs(bundles: list[LogBundle], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    collection = output_dir.name
    logs: list[EvaluationLog] = []
    file_uuids = [str(uuid.uuid4()) for _ in bundles]
    try:
        with tempfile.TemporaryDirectory(
            prefix='eee-openeval-publication-'
        ) as staging:
            staging_root = Path(staging)
            for bundle, file_uuid in zip(bundles, file_uuids, strict=True):
                log = bundle.log.model_copy(deep=True)
                if bundle.instance_path is not None and bundle.instance_count:
                    sample_rows = []
                    for line in bundle.instance_path.read_text(
                        encoding='utf-8'
                    ).splitlines():
                        if not line.strip():
                            continue
                        instance = pending_instance_from_json(line)
                        sample_rows.append(
                            make_instance_log(
                                instance,
                                log.evaluation_id,
                                log.model_info.id,
                                bundle.binary_result_ids,
                            )
                        )
                    if len(sample_rows) != bundle.instance_count:
                        raise ValueError(
                            'OpenEval instance spool row count changed before '
                            'publication'
                        )
                    sample_content = ''.join(
                        json.dumps(
                            InstanceLevelEvaluationLog.model_validate(
                                row.model_dump()
                            ).model_dump(mode='json', exclude_none=True),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + '\n'
                        for row in sample_rows
                    ).encode('utf-8')
                    sample_name = f'{file_uuid}_samples.jsonl'
                    staged_dir = datastore_output_dir(
                        staging_root,
                        collection,
                        log.model_info.id,
                        log.model_info.developer,
                    )
                    staged_dir.mkdir(parents=True, exist_ok=True)
                    (staged_dir / sample_name).write_bytes(sample_content)
                    log.detailed_evaluation_results = DetailedEvaluationResults(
                        format=Format.jsonl,
                        file_path=datastore_repo_file_path(
                            collection,
                            log.model_info.id,
                            log.model_info.developer,
                            sample_name,
                        ),
                        hash_algorithm=HashAlgorithm.sha256,
                        checksum=hashlib.sha256(sample_content).hexdigest(),
                        total_rows=len(sample_rows),
                    )
                logs.append(log)

            return publish_evaluation_logs(
                logs,
                output_dir.parent,
                file_uuids,
                staged_output_dir=staging_root,
                collection_override=collection,
            )
    finally:
        for bundle in bundles:
            if bundle.instance_path is not None:
                bundle.instance_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    if args.input_json is not None:
        payload = load_payload(args.input_json)
    else:
        max_response_shards = args.max_response_shards
        if args.limit_responses is not None and max_response_shards is None:
            max_response_shards = 1
        payload = fetch_payload(
            args.revision,
            max_response_shards,
            include_instances=args.include_instances,
        )

    result = make_logs_result(
        payload,
        revision=args.revision,
        limit_responses=args.limit_responses,
        allow_unknown_benchmark=args.allow_unknown_benchmark,
        include_instances=args.include_instances,
    )
    paths = export_logs(result.records, args.output_dir)
    for path in paths:
        print(path)
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            default_failure_report_path(args.output_dir),
        )
        print(f'Provenance report: {report_path}')
    result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} OpenEval model log(s).')
