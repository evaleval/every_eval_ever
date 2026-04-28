#!/usr/bin/env python3
"""Convert OpenEval response data into Every Eval Ever aggregate records.

Data source:
- OpenEval dataset: https://huggingface.co/datasets/human-centered-eval/OpenEval

Usage:
    uv run python -m utils.openeval.adapter --output-dir data/openeval
    uv run python -m utils.openeval.adapter --input-json sample.json

The offline JSON payload shape is:
    {
      "bench": [...],
      "response": [...]
    }
where rows match the Hugging Face ``bench`` and ``response`` table rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
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
    get_developer,
    get_model_id,
    sanitize_filename,
    save_evaluation_log,
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


@dataclass
class MetricAccumulator:
    benchmark: dict[str, Any]
    metric_name: str
    values: list[float] = field(default_factory=list)
    response_ids: list[str] = field(default_factory=list)
    metric_models: set[str] = field(default_factory=set)
    extra_artifact_types: set[str] = field(default_factory=set)

    def add(
        self, value: float, response_id: str, metric: dict[str, Any]
    ) -> None:
        self.values.append(value)
        self.response_ids.append(response_id)
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
) -> tuple[list[dict[str, Any]], Iterable[dict[str, Any]]]:
    benches = extract_collection(payload.get('bench'), 'bench')
    response_payload = payload.get('response') or payload.get('responses')
    if isinstance(response_payload, (list, dict)):
        responses: Iterable[dict[str, Any]] = extract_collection(
            response_payload, 'response'
        )
    elif response_payload is not None:
        responses = response_payload
    else:
        raise ValueError('Could not find a list of response records.')
    return benches, responses


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

    con = duckdb.connect()
    bench_cursor = con.execute('SELECT * FROM read_parquet(?)', [local_bench])
    bench_columns = [item[0] for item in bench_cursor.description]
    benches = [dict(zip(bench_columns, row)) for row in bench_cursor.fetchall()]
    # Keep response rows lazy. The full response table is large enough that
    # materializing it would make the adapter harder to run on ordinary laptops.
    # Payloads returned by this live fetch path are therefore consumed once by
    # make_logs().
    responses = _response_rows_from_parquet(con, local_response)
    return {
        'bench': benches,
        'response': responses,
        'source_metadata': {
            'hf_revision': revision,
            'hf_commit': getattr(info, 'sha', None),
            'downloaded_response_shards': len(response_files),
            'total_response_shards': total_response_shards,
            'max_response_shards': max_response_shards,
        },
    }


def _response_rows_from_parquet(
    con: Any, parquet_paths: list[str]
) -> Iterable[dict[str, Any]]:
    query = (
        'SELECT response_id, model, scores '
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
    if not isinstance(scores, dict):
        return []
    metrics = scores.get('metric') or []
    values = scores.get('value') or []
    if not isinstance(metrics, list) or not isinstance(values, list):
        return []

    pairs = []
    for metric, value in zip(metrics, values, strict=False):
        if not isinstance(metric, dict):
            continue
        name = metric.get('name')
        if name in (None, ''):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(score) or math.isinf(score):
            continue
        pairs.append((str(name), score, metric))
    return pairs


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
    developer = get_developer(name)
    model_id = get_model_id(name, developer)
    model_slug = normalize_slug(model_id.split('/', 1)[-1], name)
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


def aggregate_scores(
    benches: list[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
) -> dict[tuple[str, str, str], ModelGroup]:
    benchmark_index = build_benchmark_index(benches)
    groups: dict[tuple[str, str, str], ModelGroup] = {}

    for count, response in enumerate(responses, start=1):
        if limit_responses is not None and count > limit_responses:
            break
        response_id = str(response.get('response_id') or '')
        if not response_id:
            continue

        benchmark = benchmark_for_response_id(response_id, benchmark_index)
        if (
            benchmark.get('benchmark_name') == 'unknown'
            and not allow_unknown_benchmark
        ):
            raise ValueError(
                f'Could not match OpenEval response_id {response_id!r} '
                'to a benchmark. Pass --allow-unknown-benchmark to keep '
                'unmatched rows under the unknown benchmark.'
            )
        name = model_name(response)
        size = model_size(response)
        params = generation_parameters(response)
        key = (name, size or '', generation_key(params))
        group = groups.setdefault(
            key,
            ModelGroup(generation_params=params),
        )

        for metric_name, score, metric in numeric_score_values(
            response.get('scores')
        ):
            accumulator_key = result_key(benchmark, metric_name)
            accumulator = group.metrics.setdefault(
                accumulator_key,
                MetricAccumulator(benchmark=benchmark, metric_name=metric_name),
            )
            accumulator.add(score, response_id, metric)

    return groups


def result_key(benchmark: dict[str, Any], metric_name: str) -> str:
    return (
        f'{normalize_slug(benchmark.get("benchmark_name"))}.'
        f'{normalize_slug(metric_name)}'
    )


def metric_bounds(values: list[float]) -> tuple[float, float, str]:
    if values and all(0.0 <= value <= 1.0 for value in values):
        return 0.0, 1.0, 'proportion'
    observed_min = min(values) if values else 0.0
    observed_max = max(values) if values else 1.0
    return (
        min(0.0, observed_min),
        max(1.0, observed_max),
        'points',
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
    min_score, max_score, unit = metric_bounds(values)
    benchmark = accumulator.benchmark
    bench_slug = normalize_slug(benchmark_name(benchmark))
    metric_slug = normalize_slug(accumulator.metric_name)
    stddev = None
    stderr = None
    if len(values) > 1:
        variance = sum((value - score) ** 2 for value in values) / len(values)
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
            samples_number=len(values),
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


def source_metadata(
    revision: str,
    payload_metadata: dict[str, Any] | None = None,
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
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
                'partial_export': partial_export,
                'allow_unknown_benchmark': allow_unknown_benchmark,
                'source_role': 'aggregator',
            }
        ),
    )


def make_logs(
    payload: dict[str, Any],
    retrieved_timestamp: str | None = None,
    revision: str = HF_REVISION,
    limit_responses: int | None = None,
    allow_unknown_benchmark: bool = False,
) -> list[LogBundle]:
    benches, responses = validate_payload(payload)
    timestamp = retrieved_timestamp or str(time.time())
    groups = aggregate_scores(
        benches,
        responses,
        limit_responses,
        allow_unknown_benchmark=allow_unknown_benchmark,
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
    )

    bundles: list[LogBundle] = []
    for (name, size, gen_key), group in sorted(groups.items()):
        model_info, developer, model_slug = normalize_model_info(
            name, size or None
        )
        generation_config = make_generation_config(group.generation_params)
        if generation_config is not None:
            details = generation_config.additional_details or {}
            generation_config.additional_details = {
                **details,
                'generation_config_hash': gen_key,
            }
        results = [
            make_evaluation_result(metric, generation_config)
            for metric in group.metrics.values()
            if metric.values
        ]
        if not results:
            continue

        sanitized_model_id = model_info.id.replace('/', '_')
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=(
                f'openeval/{sanitized_model_id}/{gen_key}/{timestamp}'
            ),
            retrieved_timestamp=timestamp,
            source_metadata=metadata,
            eval_library=EvalLibrary(name='OpenEval', version='unknown'),
            model_info=model_info,
            evaluation_results=sorted(
                results, key=lambda item: item.evaluation_result_id or ''
            ),
        )
        bundles.append(
            LogBundle(log=log, developer=developer, model=model_slug)
        )

    return bundles


def export_logs(bundles: list[LogBundle], output_dir: Path) -> list[Path]:
    paths = []
    for bundle in bundles:
        paths.append(
            save_evaluation_log(
                bundle.log,
                output_dir,
                bundle.developer,
                bundle.model,
            )
        )
    return paths


def run(args: argparse.Namespace) -> int:
    if args.input_json is not None:
        payload = load_payload(args.input_json)
    else:
        max_response_shards = args.max_response_shards
        if args.limit_responses is not None and max_response_shards is None:
            max_response_shards = 1
        payload = fetch_payload(args.revision, max_response_shards)

    bundles = make_logs(
        payload,
        revision=args.revision,
        limit_responses=args.limit_responses,
        allow_unknown_benchmark=args.allow_unknown_benchmark,
    )
    paths = export_logs(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} OpenEval model log(s).')
