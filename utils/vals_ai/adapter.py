#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

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
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
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

SOURCE_NAME = 'Vals.ai'
SOURCE_ORGANIZATION_URL = 'https://www.vals.ai'
BENCHMARKS_URL = f'{SOURCE_ORGANIZATION_URL}/benchmarks'
OUTPUT_DIR = 'data/vals-ai'
USER_AGENT = 'every-eval-ever vals-ai adapter'
ASTRO_UNDEFINED = object()
NAMESPACE_DEVELOPER_ALIASES = {
    'grok': 'xai',
    'kimi': 'moonshotai',
}


@dataclass(frozen=True)
class ValsMetric:
    benchmark_slug: str
    benchmark_name: str
    benchmark_updated: str | None
    dataset_type: str | None
    industry: str | None
    task_key: str
    task_name: str
    model_id: str
    metrics: dict[str, Any]
    source_url: str


@dataclass(frozen=True)
class ScoreScale:
    metric_unit: str
    metric_name: str
    metric_kind: str
    max_score: float


@dataclass(frozen=True)
class EvaluationBundle:
    log: EvaluationLog
    developer: str
    model_name: str


class AstroIslandParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.props: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != 'astro-island':
            return

        attr_map = {name: value for name, value in attrs}
        component_url = attr_map.get('component-url') or ''
        if 'BenchmarkView' not in component_url:
            return

        props = attr_map.get('props')
        if props:
            self.props.append(props)


def fetch_text(url: str) -> str:
    request = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode('utf-8')
    except URLError as exc:
        raise RuntimeError(f'Failed to fetch {url}: {exc}') from exc


def extract_benchmark_slugs(index_html: str) -> list[str]:
    slugs = set(
        re.findall(r'href=["\']/benchmarks/([A-Za-z0-9_-]+)', index_html)
    )
    return sorted(slugs)


def decode_astro_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], int):
            return ASTRO_UNDEFINED
        if len(value) == 2 and isinstance(value[0], int):
            tag, payload = value
            if tag == 0:
                return decode_astro_value(payload)
            if tag == 1:
                return [
                    item
                    for item in (decode_astro_value(v) for v in payload)
                    if item is not ASTRO_UNDEFINED
                ]
            raise ValueError(f'Unsupported Astro serialized value tag: {tag}')
        return [
            item
            for item in (decode_astro_value(v) for v in value)
            if item is not ASTRO_UNDEFINED
        ]

    if isinstance(value, dict):
        return {
            key: decoded
            for key, inner in value.items()
            if (decoded := decode_astro_value(inner)) is not ASTRO_UNDEFINED
        }

    return value


def extract_benchmark_view(page_html: str) -> dict[str, Any]:
    parser = AstroIslandParser()
    parser.feed(page_html)

    for raw_props in parser.props:
        props = parse_astro_props(raw_props)
        decoded = decode_astro_value(props)
        benchmark_view = decoded.get('benchmarkView')
        if not isinstance(benchmark_view, dict):
            continue

        candidate = benchmark_view.get('default') or benchmark_view
        if isinstance(candidate, dict) and {
            'metadata',
            'tasks',
        }.issubset(candidate):
            return candidate

    raise ValueError('Could not find BenchmarkView data in Vals.ai page')


def parse_astro_props(raw_props: str) -> dict[str, Any]:
    try:
        return json.loads(raw_props)
    except json.JSONDecodeError:
        return json.loads(unescape(raw_props))


def normalize_benchmark_page(page_html: str, source_url: str) -> dict[str, Any]:
    view = extract_benchmark_view(page_html)
    metadata = view.get('metadata') or {}
    tasks = view.get('tasks') or {}
    if not isinstance(metadata, dict) or not isinstance(tasks, dict):
        raise ValueError('BenchmarkView payload has invalid metadata/tasks')

    return {
        'metadata': metadata,
        'tasks': tasks,
        'source_url': source_url,
    }


def extract_collection(
    *,
    input_json: Path | None = None,
    benchmark_slugs: list[str] | None = None,
    base_url: str = SOURCE_ORGANIZATION_URL,
) -> dict[str, Any]:
    if input_json is not None:
        return json.loads(input_json.read_text(encoding='utf-8'))

    index_html = fetch_text(f'{base_url.rstrip("/")}/benchmarks')
    slugs = benchmark_slugs or extract_benchmark_slugs(index_html)
    benchmarks = []
    for slug in slugs:
        source_url = f'{base_url.rstrip("/")}/benchmarks/{slug}'
        page_html = fetch_text(source_url)
        try:
            benchmarks.append(normalize_benchmark_page(page_html, source_url))
        except Exception as exc:
            raise ValueError(
                f'Failed to parse Vals.ai benchmark page {slug!r} '
                f'at {source_url}'
            ) from exc

    return {
        'source_url': f'{base_url.rstrip("/")}/benchmarks',
        'benchmarks': benchmarks,
    }


def iter_vals_metrics(payload: dict[str, Any]) -> list[ValsMetric]:
    metrics: list[ValsMetric] = []
    for benchmark in payload.get('benchmarks', []):
        metadata = benchmark.get('metadata') or {}
        tasks = benchmark.get('tasks') or {}
        source_url = str(benchmark.get('source_url') or BENCHMARKS_URL)
        raw_benchmark_slug = metadata.get('slug') or metadata.get(
            'benchmark_id'
        )
        if not raw_benchmark_slug:
            raise ValueError('Vals.ai benchmark payload is missing a slug')
        benchmark_slug = str(raw_benchmark_slug)
        benchmark_name = str(metadata.get('benchmark') or benchmark_slug)
        task_names = metadata.get('tasks') or {}
        if not isinstance(tasks, dict):
            continue

        for task_key, model_rows in tasks.items():
            if not isinstance(model_rows, dict):
                raise ValueError(
                    f'Vals.ai task payload is not an object for '
                    f'{benchmark_slug}/{task_key}'
                )
            task_name = (
                task_names.get(task_key)
                if isinstance(task_names, dict)
                else None
            )
            for model_id, row in model_rows.items():
                if not model_id or not isinstance(row, dict):
                    raise ValueError(
                        f'Vals.ai model row is invalid for '
                        f'{benchmark_slug}/{task_key}/{model_id!r}'
                    )
                score = row.get('accuracy')
                if score is None:
                    continue
                try:
                    float(score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Non-numeric Vals.ai score for '
                        f'{benchmark_slug}/{task_key}/{model_id}: {score!r}'
                    ) from exc
                metrics.append(
                    ValsMetric(
                        benchmark_slug=benchmark_slug,
                        benchmark_name=benchmark_name,
                        benchmark_updated=_optional_str(
                            metadata.get('updated')
                        ),
                        dataset_type=_optional_str(
                            metadata.get('dataset_type')
                        ),
                        industry=_optional_str(metadata.get('industry')),
                        task_key=str(task_key),
                        task_name=str(task_name or task_key),
                        model_id=str(model_id),
                        metrics=row,
                        source_url=source_url,
                    )
                )

    return metrics


def build_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str], list[ValsMetric]]:
    grouped: dict[tuple[str, str], list[ValsMetric]] = {}
    canonical_to_raw: dict[tuple[str, str], str] = {}
    for metric in iter_vals_metrics(payload):
        provider = _optional_str(metric.metrics.get('provider'))
        developer = _developer_from_vals_id(metric.model_id, provider)
        canonical_model_id = _canonical_model_id(metric.model_id, developer)
        key = (metric.benchmark_slug, canonical_model_id)
        existing_raw_id = canonical_to_raw.get(key)
        if existing_raw_id is not None and existing_raw_id != metric.model_id:
            raise ValueError(
                'Vals.ai model IDs collide after canonicalization for '
                f'{metric.benchmark_slug}: {existing_raw_id!r} and '
                f'{metric.model_id!r} both map to {canonical_model_id!r}'
            )
        canonical_to_raw[key] = metric.model_id
        grouped.setdefault(key, []).append(metric)
    return grouped


def build_score_scales(payload: dict[str, Any]) -> dict[str, ScoreScale]:
    by_benchmark: dict[str, list[float]] = {}
    for metric in iter_vals_metrics(payload):
        by_benchmark.setdefault(metric.benchmark_slug, []).append(
            float(metric.metrics['accuracy'])
        )

    return {
        benchmark_slug: _score_scale(scores)
        for benchmark_slug, scores in by_benchmark.items()
    }


def validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get('benchmarks'), list):
        raise ValueError('Vals.ai payload must contain a benchmarks list')

    if not iter_vals_metrics(payload):
        raise ValueError(
            'Vals.ai payload did not contain any scored model rows'
        )


def make_logs(
    payload: dict[str, Any],
    *,
    retrieved_timestamp: str | None = None,
) -> list[EvaluationBundle]:
    validate_payload(payload)
    retrieved_timestamp = retrieved_timestamp or str(time.time())
    bundles: list[EvaluationBundle] = []
    score_scales = build_score_scales(payload)

    for (benchmark_slug, model_id), rows in sorted(
        build_index(payload).items()
    ):
        score_scale = score_scales[benchmark_slug]
        if score_scale.metric_unit != 'percent':
            continue
        first = rows[0]
        provider = _optional_str(first.metrics.get('provider'))
        developer = model_id.split('/', 1)[0]
        vals_model_id = first.model_id
        model_name = _model_name_from_id(model_id)

        result_rows = sorted(rows, key=lambda row: row.task_key)
        results = [
            make_result(
                row,
                score_scale=score_scale,
            )
            for row in result_rows
        ]

        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=(
                f'vals-ai/{benchmark_slug}/'
                f'{sanitize_filename(model_id)}/{retrieved_timestamp}'
            ),
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name=f'Vals.ai Leaderboard - {first.benchmark_name}',
                source_type=SourceType.documentation,
                source_organization_name=SOURCE_NAME,
                source_organization_url=SOURCE_ORGANIZATION_URL,
                evaluator_relationship=EvaluatorRelationship.third_party,
                additional_details=_clean_details(
                    {
                        'benchmark_slug': benchmark_slug,
                        'benchmark_name': first.benchmark_name,
                        'benchmark_updated': first.benchmark_updated,
                        'dataset_type': first.dataset_type,
                        'industry': first.industry,
                        'leaderboard_page_url': first.source_url,
                        'extraction_method': 'static_astro_benchmark_view_props',
                    }
                ),
            ),
            eval_library=EvalLibrary(name=SOURCE_NAME, version='unknown'),
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                additional_details=_clean_details(
                    {
                        'vals_model_id': vals_model_id,
                        'vals_provider': provider,
                    }
                ),
            ),
            evaluation_results=results,
        )
        bundles.append(
            EvaluationBundle(
                log=log, developer=developer, model_name=model_name
            )
        )

    return bundles


def make_result(
    row: ValsMetric,
    *,
    score_scale: ScoreScale,
) -> EvaluationResult:
    score = float(row.metrics['accuracy'])
    stderr = _optional_float(row.metrics.get('stderr'))
    details = _clean_details(
        {
            'benchmark_slug': row.benchmark_slug,
            'benchmark_name': row.benchmark_name,
            'benchmark_updated': row.benchmark_updated,
            'task_key': row.task_key,
            'task_name': row.task_name,
            'dataset_type': row.dataset_type,
            'industry': row.industry,
            'raw_score': row.metrics.get('accuracy'),
            'raw_stderr': row.metrics.get('stderr'),
            'latency': row.metrics.get('latency'),
            'cost_per_test': row.metrics.get('cost_per_test'),
            'temperature': row.metrics.get('temperature'),
            'top_p': row.metrics.get('top_p'),
            'max_output_tokens': row.metrics.get('max_output_tokens'),
            'reasoning': row.metrics.get('reasoning'),
            'reasoning_effort': row.metrics.get('reasoning_effort'),
            'verbosity': row.metrics.get('verbosity'),
            'compute_effort': row.metrics.get('compute_effort'),
            'provider': row.metrics.get('provider'),
        }
    )

    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(
            standard_error=StandardError(
                value=stderr,
                method='vals_reported',
            )
        )

    return EvaluationResult(
        evaluation_result_id=(
            f'{row.benchmark_slug}:{row.task_key}:{row.model_id}:score'
        ),
        evaluation_name=f'vals_ai.{row.benchmark_slug}.{row.task_key}',
        source_data=make_source_data(row),
        metric_config=MetricConfig(
            evaluation_description=(
                f'{score_scale.metric_name} reported by Vals.ai for '
                f'{row.benchmark_name} ({row.task_name}).'
            ),
            metric_id=(
                f'vals_ai.{row.benchmark_slug}.'
                f'{row.task_key}.{score_scale.metric_kind}'
            ),
            metric_name=score_scale.metric_name,
            metric_kind=score_scale.metric_kind,
            metric_unit=score_scale.metric_unit,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=score_scale.max_score,
            additional_details=_clean_details(
                {
                    'score_scale': (
                        'percent_0_to_100'
                        if score_scale.metric_unit == 'percent'
                        else 'source_reported_points'
                    ),
                    'max_score_source': (
                        'fixed_percentage_bound'
                        if score_scale.metric_unit == 'percent'
                        else 'not_exported_without_explicit_source_bounds'
                    ),
                    'leaderboard_page_url': row.source_url,
                }
            ),
        ),
        score_details=ScoreDetails(
            score=score,
            details=details,
            uncertainty=uncertainty,
        ),
        generation_config=make_generation_config(row.metrics),
    )


def make_generation_config(metrics: dict[str, Any]) -> GenerationConfig | None:
    explicit_generation_args = {
        'temperature': _optional_float(metrics.get('temperature')),
        'top_p': _optional_float(metrics.get('top_p')),
        'max_tokens': _optional_positive_int(metrics.get('max_output_tokens')),
    }
    generation_args = None
    if any(value is not None for value in explicit_generation_args.values()):
        generation_args = GenerationArgs(**explicit_generation_args)

    additional_details = _clean_details(
        {
            'reasoning': metrics.get('reasoning'),
            'reasoning_effort': metrics.get('reasoning_effort'),
            'verbosity': metrics.get('verbosity'),
            'compute_effort': metrics.get('compute_effort'),
        }
    )
    if generation_args is not None or additional_details:
        return GenerationConfig(
            generation_args=generation_args,
            additional_details=additional_details,
        )
    return None


def make_source_data(row: ValsMetric) -> SourceDataPrivate | SourceDataUrl:
    details = _clean_details(
        {
            'benchmark_slug': row.benchmark_slug,
            'task_key': row.task_key,
            'dataset_type': row.dataset_type,
            'leaderboard_page_url': row.source_url,
        }
    )
    dataset_name = f'{row.benchmark_name} - {row.task_name}'
    if row.dataset_type == 'private':
        return SourceDataPrivate(
            dataset_name=dataset_name,
            source_type='other',
            additional_details=details,
        )

    return SourceDataUrl(
        dataset_name=dataset_name,
        source_type='url',
        url=[row.source_url],
        additional_details=details,
    )


def export_logs(
    bundles: list[EvaluationBundle],
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    paths = []
    for bundle in bundles:
        paths.append(
            save_evaluation_log(
                bundle.log,
                output_dir,
                bundle.developer,
                bundle.model_name,
            )
        )
    return paths


def save_raw_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )


def _developer_from_vals_id(vals_model_id: str, provider: str | None) -> str:
    if '/' in vals_model_id:
        namespace = _slug(vals_model_id.split('/', 1)[0])
        return NAMESPACE_DEVELOPER_ALIASES.get(namespace, namespace)
    if provider:
        return _slug(provider)
    return get_developer(vals_model_id)


def _canonical_model_id(vals_model_id: str, developer: str) -> str:
    model_name = _model_name_from_id(vals_model_id)
    return get_model_id(model_name, developer)


def _model_name_from_id(vals_model_id: str) -> str:
    if '/' in vals_model_id:
        return vals_model_id.split('/', 1)[1]
    return vals_model_id


def _slug(value: str) -> str:
    return sanitize_filename(value.lower().replace(' ', '-'))


def _score_scale(scores: list[float]) -> ScoreScale:
    if scores and max(scores) <= 100.0 and min(scores) >= 0.0:
        return ScoreScale(
            metric_unit='percent',
            metric_name='Accuracy',
            metric_kind='accuracy',
            max_score=100.0,
        )
    return ScoreScale(
        metric_unit='points',
        metric_name='Score',
        metric_kind='score',
        max_score=max(scores) if scores else 0.0,
    )


def _clean_details(values: dict[str, Any]) -> dict[str, str] | None:
    details = {
        key: _detail_value(value)
        for key, value in values.items()
        if value is not None and value is not ASTRO_UNDEFINED
    }
    return details or None


def _detail_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None or value is ASTRO_UNDEFINED:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value is ASTRO_UNDEFINED:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value is ASTRO_UNDEFINED:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_positive_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 1:
        return None
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Vals.ai benchmark leaderboards to EEE JSON.'
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory for EEE JSON files (default: {OUTPUT_DIR})',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Read a previously saved normalized Vals.ai payload.',
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help=(
            'Save the fetched and normalized Vals.ai payload for '
            'replay/debugging.'
        ),
    )
    parser.add_argument(
        '--benchmark',
        action='append',
        dest='benchmarks',
        help='Benchmark slug to fetch. Can be repeated. Defaults to all slugs.',
    )
    parser.add_argument(
        '--base-url',
        default=SOURCE_ORGANIZATION_URL,
        help=f'Vals.ai base URL (default: {SOURCE_ORGANIZATION_URL})',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = extract_collection(
        input_json=args.input_json,
        benchmark_slugs=args.benchmarks,
        base_url=args.base_url,
    )
    if args.save_raw_json is not None:
        save_raw_payload(payload, args.save_raw_json)

    bundles = make_logs(payload)
    paths = export_logs(bundles, args.output_dir)
    print(f'Saved {len(paths)} Vals.ai evaluation logs to {args.output_dir}')


if __name__ == '__main__':
    main()
