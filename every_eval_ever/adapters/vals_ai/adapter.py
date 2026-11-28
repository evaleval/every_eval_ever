#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
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
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    get_developer,
    get_model_id,
    raw_capture,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

SOURCE_NAME = 'Vals.ai'
SOURCE_ORGANIZATION_URL = 'https://www.vals.ai'
BENCHMARKS_URL = f'{SOURCE_ORGANIZATION_URL}/benchmarks'
API_URL = 'https://benchmcp.vals.ai'
OUTPUT_DIR = 'data/vals-ai'
USER_AGENT = 'every-eval-ever vals-ai adapter'
ASTRO_UNDEFINED = object()
NAMESPACE_DEVELOPER_ALIASES = {
    'grok': 'xai',
    'kimi': 'moonshotai',
    'meta-llama': 'meta',
    'qwen': 'alibaba',
    'togethercomputer': 'together',
}
# Leading route segments naming the serving platform, not the developer.
ROUTE_PREFIXES = frozenset({'baseten', 'openrouter', 'together'})
# Hosts whose two-segment routes (`fireworks/kimi-k3`) omit the developer.
HOSTED_ROUTE_PREFIXES = ROUTE_PREFIXES | {'fireworks', 'groq'}


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
    max_score: float | None


@dataclass(frozen=True)
class EvaluationBundle:
    log: EvaluationLog
    developer: str
    model_name: str


@dataclass(frozen=True)
class ValsModelIdentity:
    """Schema fields resolved from one Vals.ai model route."""

    raw_id: str
    developer: str
    model_name: str
    model_id: str
    inference_platform: str | None
    inference_engine: str | None = None
    route: str | None = None
    baseline_model: str | None = None


class NoBenchmarkViewError(ValueError):
    """A source (API artifact listing or page) carries no leaderboard view."""


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
        props = attr_map.get('props')
        # Matches both the inline `benchmarkView` and the loader's
        # `benchmarkViewUrl`, in whichever island component carries them.
        if props and 'benchmarkView' in props:
            self.props.append(props)


def fetch_text(url: str) -> str:
    request = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
            content_type = response.headers.get('Content-Type')
    except URLError as exc:
        raise RuntimeError(f'Failed to fetch {url}: {exc}') from exc
    raw_capture.record(url=url, content=body, content_type=content_type)
    return body.decode('utf-8')


def fetch_json(url: str) -> Any:
    return json.loads(fetch_text(url))


def list_benchmark_slugs(api_url: str = API_URL) -> list[str]:
    """Return every benchmark slug the Vals.ai API lists."""
    listing = fetch_json(f'{api_url.rstrip("/")}/benchmarks')
    slugs = listing.get('benchmarks') if isinstance(listing, dict) else None
    if (
        not isinstance(slugs, list)
        or not slugs
        or not all(isinstance(slug, str) and slug for slug in slugs)
    ):
        raise ValueError(
            'Vals.ai API /benchmarks did not return a list of benchmark slugs'
        )
    return sorted(set(slugs))


def benchmark_view_url(slug: str, api_url: str = API_URL) -> str:
    """Return the API URL of a benchmark's single ``benchmark_view`` artifact.

    Raises ``NoBenchmarkViewError`` when the artifact listing is well formed
    but has no ``benchmark_view`` entry: the benchmark has no leaderboard.
    """
    benchmark_url = f'{api_url.rstrip("/")}/benchmarks/{quote(slug, safe="")}'
    listing = fetch_json(f'{benchmark_url}/artifacts')
    artifacts = listing.get('artifacts') if isinstance(listing, dict) else None
    if not isinstance(artifacts, list) or not all(
        isinstance(artifact, dict) for artifact in artifacts
    ):
        raise ValueError('artifact listing has no "artifacts" list')
    paths = [
        artifact.get('path')
        for artifact in artifacts
        if artifact.get('type') == 'benchmark_view'
    ]
    if not paths:
        types = sorted({str(artifact.get('type')) for artifact in artifacts})
        raise NoBenchmarkViewError(
            'API lists no benchmark_view artifact '
            f'(artifact types: {", ".join(types) or "none"})'
        )
    if len(paths) != 1 or not isinstance(paths[0], str) or not paths[0]:
        raise ValueError(
            f'expected one benchmark_view artifact, found {len(paths)}'
        )
    return f'{benchmark_url}/artifacts/{quote(paths[0])}'


def normalize_benchmark_view(
    view: Any,
    source_url: str,
    benchmark_view_url: str | None = None,
) -> dict[str, Any]:
    if not isinstance(view, dict):
        raise ValueError('BenchmarkView payload must be an object')
    metadata = view.get('metadata') or {}
    tasks = view.get('tasks') or {}
    if not isinstance(metadata, dict) or not isinstance(tasks, dict):
        raise ValueError('BenchmarkView payload has invalid metadata/tasks')

    normalized = {
        'metadata': metadata,
        'tasks': tasks,
        'source_url': source_url,
    }
    if benchmark_view_url is not None:
        normalized['benchmark_view_url'] = benchmark_view_url
    return normalized


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


def _benchmark_view_props(page_html: str) -> list[dict[str, Any]]:
    parser = AstroIslandParser()
    parser.feed(page_html)
    return [
        decode_astro_value(parse_astro_props(raw_props))
        for raw_props in parser.props
    ]


def extract_benchmark_view_url(page_html: str) -> str | None:
    """Return the JSON URL a ``BenchmarkViewLoader`` island fetches, if any."""
    for decoded in _benchmark_view_props(page_html):
        view_url = decoded.get('benchmarkViewUrl')
        if isinstance(view_url, str) and view_url:
            return view_url
    return None


def extract_benchmark_view(page_html: str) -> dict[str, Any]:
    for decoded in _benchmark_view_props(page_html):
        benchmark_view = decoded.get('benchmarkView')
        if not isinstance(benchmark_view, dict):
            continue

        candidate = benchmark_view.get('default') or benchmark_view
        if isinstance(candidate, dict) and {
            'metadata',
            'tasks',
        }.issubset(candidate):
            return candidate

    raise NoBenchmarkViewError(
        'Could not find BenchmarkView data in Vals.ai page'
    )


def parse_astro_props(raw_props: str) -> dict[str, Any]:
    try:
        return json.loads(raw_props)
    except json.JSONDecodeError:
        return json.loads(unescape(raw_props))


def normalize_benchmark_page(page_html: str, source_url: str) -> dict[str, Any]:
    return normalize_benchmark_view(
        extract_benchmark_view(page_html), source_url
    )


def load_benchmark_page(page_html: str, source_url: str) -> dict[str, Any]:
    """Read a benchmark's leaderboard from the JSON its page loads, or inline."""
    view_url = extract_benchmark_view_url(page_html)
    if view_url is None:
        return normalize_benchmark_page(page_html, source_url)

    view_url = urljoin(source_url, view_url)
    return normalize_benchmark_view(
        json.loads(fetch_text(view_url)), source_url, view_url
    )


def load_api_benchmark(
    slug: str,
    *,
    api_url: str = API_URL,
    base_url: str = SOURCE_ORGANIZATION_URL,
) -> dict[str, Any]:
    """Fetch one benchmark's leaderboard view from the Vals.ai API."""
    view_url = benchmark_view_url(slug, api_url)
    response = fetch_json(view_url)
    view = response.get('data') if isinstance(response, dict) else None
    metadata = view.get('metadata') if isinstance(view, dict) else None
    # The site page lives at the view's own slug, which can differ from the
    # API slug (API `web_search` -> page `web_search_backends`).
    page_slug = (
        metadata.get('slug')
        if isinstance(metadata, dict) and isinstance(metadata.get('slug'), str)
        else slug
    )
    return normalize_benchmark_view(
        view, f'{base_url.rstrip("/")}/benchmarks/{page_slug}', view_url
    )


def load_benchmark(
    slug: str,
    *,
    api_url: str = API_URL,
    base_url: str = SOURCE_ORGANIZATION_URL,
) -> dict[str, Any]:
    """Fetch a benchmark from the API, falling back to scraping its page.

    Raises ``NoBenchmarkViewError`` only when both sources agree there is no
    leaderboard: the API lists no ``benchmark_view`` artifact and the page
    loads but carries no BenchmarkView data. Any other error is a failure.
    """
    try:
        return load_api_benchmark(slug, api_url=api_url, base_url=base_url)
    except Exception as api_exc:
        page_url = f'{base_url.rstrip("/")}/benchmarks/{slug}'
        try:
            page_html = fetch_text(page_url)
            benchmark = load_benchmark_page(page_html, page_url)
        except Exception as page_exc:
            if isinstance(api_exc, NoBenchmarkViewError) and isinstance(
                page_exc, NoBenchmarkViewError
            ):
                raise NoBenchmarkViewError(
                    f'API: {api_exc}; page: {page_exc}'
                ) from page_exc
            raise ValueError(
                f'API: {api_exc}; page fallback: {page_exc}'
            ) from page_exc
        benchmark['api_fallback_reason'] = str(api_exc)
        return benchmark


def _benchmark_failure(
    source_ref: str, slug: str, exc: Exception
) -> SourceRecordFailure:
    return SourceRecordFailure(
        source_ref=source_ref,
        reason=f'Failed to fetch or parse Vals.ai benchmark {slug!r}: {exc}',
        source_record={'benchmark_slug': slug},
    )


def extract_collection(
    *,
    input_json: Path | None = None,
    benchmark_slugs: list[str] | None = None,
    base_url: str = SOURCE_ORGANIZATION_URL,
    api_url: str = API_URL,
) -> dict[str, Any]:
    """Fetch the Vals.ai leaderboards, or replay a saved payload.

    Without explicit ``benchmark_slugs``, every benchmark the API lists is
    fetched first; benchmarks the site index lists that the API did not cover
    are then scraped from their pages. Benchmarks are deduplicated on the
    view's own slug, keeping the first (API) copy.
    """
    if input_json is not None:
        return json.loads(input_json.read_text(encoding='utf-8'))

    api_url = api_url.rstrip('/')
    base_url = base_url.rstrip('/')
    source_failures: list[SourceRecordFailure] = []
    source_exclusions: list[SourceRecordExclusion] = []
    discovery_errors: dict[str, str] = {}

    if benchmark_slugs:
        api_slugs, site_slugs = benchmark_slugs, []
    else:
        api_slugs, site_slugs = [], []
        try:
            api_slugs = list_benchmark_slugs(api_url)
        except Exception as exc:
            discovery_errors['api'] = str(exc)
        try:
            site_slugs = extract_benchmark_slugs(
                fetch_text(f'{base_url}/benchmarks')
            )
        except Exception as exc:
            discovery_errors['site'] = str(exc)
        if not api_slugs and not site_slugs:
            raise ValueError(
                f'Could not list Vals.ai benchmarks: {discovery_errors}'
            )

    by_key: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []

    def add(benchmark: dict[str, Any], slug: str) -> None:
        metadata = benchmark['metadata']
        key = str(metadata.get('slug') or metadata.get('benchmark_id') or slug)
        kept = by_key.get(key)
        if kept is None:
            by_key[key] = benchmark
            return
        # API views are added first, so the API copy wins over a scraped one.
        duplicates.append(
            {
                'benchmark_slug': key,
                'kept': kept.get('benchmark_view_url') or kept['source_url'],
                'dropped': benchmark.get('benchmark_view_url')
                or benchmark['source_url'],
            }
        )

    for slug in api_slugs:
        try:
            add(load_benchmark(slug, api_url=api_url, base_url=base_url), slug)
        except NoBenchmarkViewError as exc:
            source_exclusions.append(
                SourceRecordExclusion(
                    source_ref=f'{api_url}/benchmarks/{slug}/artifacts',
                    reason=(
                        f'Vals.ai benchmark {slug!r} has no leaderboard: {exc}'
                    ),
                    source_record={'benchmark_slug': slug},
                )
            )
        except Exception as exc:
            source_failures.append(
                _benchmark_failure(f'{api_url}/benchmarks/{slug}', slug, exc)
            )

    covered = set(api_slugs) | set(by_key)
    for slug in site_slugs:
        if slug in covered:
            continue
        page_url = f'{base_url}/benchmarks/{slug}'
        try:
            add(load_benchmark_page(fetch_text(page_url), page_url), slug)
        except Exception as exc:
            source_failures.append(_benchmark_failure(page_url, slug, exc))

    benchmarks = list(by_key.values())
    payload: dict[str, Any] = {
        'source_url': f'{api_url}/benchmarks',
        'benchmarks': benchmarks,
        'source_failures': [
            failure.model_dump() for failure in source_failures
        ],
    }
    if source_exclusions:
        payload['source_exclusions'] = [
            exclusion.model_dump() for exclusion in source_exclusions
        ]
    if discovery_errors:
        payload['discovery_errors'] = discovery_errors
    if duplicates:
        payload['duplicate_benchmarks'] = duplicates
    return payload


def iter_vals_metrics(payload: dict[str, Any]) -> list[ValsMetric]:
    result = iter_vals_metrics_result(payload)
    result.raise_if_incomplete()
    return result.records


def iter_vals_metrics_result(
    payload: dict[str, Any],
) -> SourceConversionResult[ValsMetric]:
    """Parse Vals.ai model/task rows without losing valid sibling rows."""
    metrics: list[ValsMetric] = []
    failures: list[SourceRecordFailure] = []
    total_records = 0
    for benchmark_index, benchmark in enumerate(payload.get('benchmarks', [])):
        if not isinstance(benchmark, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'benchmark row {benchmark_index}',
                    reason='Vals.ai benchmark must be an object',
                    source_record=benchmark,
                )
            )
            total_records += 1
            continue
        metadata = benchmark.get('metadata') or {}
        tasks = benchmark.get('tasks') or {}
        source_url = str(benchmark.get('source_url') or BENCHMARKS_URL)
        if not isinstance(metadata, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'benchmark row {benchmark_index}',
                    reason='Vals.ai benchmark metadata must be an object',
                    source_record=benchmark,
                )
            )
            total_records += 1
            continue
        raw_benchmark_slug = metadata.get('slug') or metadata.get(
            'benchmark_id'
        )
        if not raw_benchmark_slug:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'benchmark row {benchmark_index}',
                    reason='Vals.ai benchmark payload is missing a slug',
                    source_record=benchmark,
                )
            )
            total_records += 1
            continue
        benchmark_slug = str(raw_benchmark_slug)
        benchmark_name = str(metadata.get('benchmark') or benchmark_slug)
        task_names = metadata.get('tasks') or {}
        if not isinstance(tasks, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'Vals.ai benchmark {benchmark_slug!r}',
                    reason='benchmark tasks must be an object',
                    source_record=benchmark,
                )
            )
            total_records += 1
            continue

        for task_key, model_rows in tasks.items():
            if not isinstance(model_rows, dict):
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'{benchmark_slug}/{task_key}',
                        reason='Vals.ai task payload is not an object',
                        source_record=model_rows,
                    )
                )
                total_records += 1
                continue
            task_name = (
                task_names.get(task_key)
                if isinstance(task_names, dict)
                else None
            )
            for model_id, row in model_rows.items():
                total_records += 1
                source_ref = f'{benchmark_slug}/{task_key}/{model_id}'
                try:
                    if not model_id or not isinstance(row, dict):
                        raise ValueError('Vals.ai model row must be an object')
                    score = row.get('accuracy')
                    if score is None:
                        raise ValueError(
                            'Vals.ai model row is missing accuracy'
                        )
                    try:
                        parsed_score = float(score)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'Non-numeric Vals.ai score: {score!r}'
                        ) from exc
                    if not math.isfinite(parsed_score):
                        raise ValueError(
                            f'Vals.ai score must be finite, got {score!r}'
                        )
                except (TypeError, ValueError) as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=source_ref,
                            reason=str(exc),
                            source_record=row,
                        )
                    )
                    continue
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

    return SourceConversionResult(
        source_name='Vals.ai source model/task rows',
        total_records=total_records,
        records=metrics,
        failures=failures,
    )


def build_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str], list[ValsMetric]]:
    result = _group_metrics(iter_vals_metrics(payload))
    result.raise_if_incomplete()
    return dict(result.records)


def _group_metrics(
    metrics: list[ValsMetric],
) -> SourceConversionResult[tuple[tuple[str, str], list[ValsMetric]]]:
    raw_groups: dict[tuple[str, str], list[ValsMetric]] = {}
    for metric in metrics:
        raw_groups.setdefault(
            (metric.benchmark_slug, metric.model_id),
            [],
        ).append(metric)

    grouped: list[tuple[tuple[str, str], list[ValsMetric]]] = []
    canonical_to_raw: dict[tuple[str, str], str] = {}
    failures: list[SourceRecordFailure] = []
    for (benchmark_slug, raw_id), rows in sorted(raw_groups.items()):
        try:
            identity = _rows_identity(raw_id, rows)
        except (TypeError, ValueError) as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{benchmark_slug}/{raw_id}',
                    reason=str(exc),
                    source_record=[asdict(row) for row in rows],
                )
            )
            continue
        canonical_key = (
            benchmark_slug,
            '|'.join(
                (
                    identity.model_id,
                    (
                        _slug(identity.inference_platform)
                        if identity.inference_platform
                        else ''
                    ),
                    identity.inference_engine or '',
                )
            ),
        )
        existing_raw_id = canonical_to_raw.get(canonical_key)
        if existing_raw_id is not None and existing_raw_id != raw_id:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{benchmark_slug}/{raw_id}',
                    reason=(
                        'Vals.ai model IDs collide after canonicalization: '
                        f'{existing_raw_id!r} and {raw_id!r} both map to '
                        f'{identity.model_id!r} on '
                        f'{identity.inference_platform or "unknown platform"!r}'
                    ),
                    source_record=[asdict(row) for row in rows],
                )
            )
            continue
        canonical_to_raw[canonical_key] = raw_id
        grouped.append(((benchmark_slug, raw_id), rows))

    return SourceConversionResult(
        source_name='Vals.ai model groups',
        total_records=len(raw_groups),
        records=grouped,
        failures=failures,
    )


def build_score_scales(payload: dict[str, Any]) -> dict[str, ScoreScale]:
    return {
        benchmark_slug: _score_scale(scores)
        for benchmark_slug, scores in _scores_by_benchmark(
            iter_vals_metrics(payload)
        ).items()
    }


def _scores_by_benchmark(
    metrics: list[ValsMetric],
) -> dict[str, list[float]]:
    by_benchmark: dict[str, list[float]] = {}
    for metric in metrics:
        by_benchmark.setdefault(metric.benchmark_slug, []).append(
            float(metric.metrics['accuracy'])
        )
    return by_benchmark


def validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get('benchmarks'), list):
        raise ValueError('Vals.ai payload must contain a benchmarks list')


def _make_bundle(
    benchmark_slug: str,
    vals_model_id: str,
    rows: list[ValsMetric],
    score_scale: ScoreScale,
    retrieved_timestamp: str,
) -> EvaluationBundle:
    first = rows[0]
    provider = _optional_str(first.metrics.get('provider'))
    identity = _rows_identity(vals_model_id, rows)
    results = [
        make_result(row, score_scale=score_scale)
        for row in sorted(rows, key=lambda row: row.task_key)
    ]

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'vals-ai/{benchmark_slug}/'
            f'{sanitize_filename(vals_model_id)}/{retrieved_timestamp}'
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
            name=identity.model_name,
            id=identity.model_id,
            developer=identity.developer,
            inference_platform=identity.inference_platform,
            inference_engine=(
                InferenceEngine(name=identity.inference_engine)
                if identity.inference_engine
                else None
            ),
            additional_details=_clean_details(
                {
                    'vals_model_id': vals_model_id,
                    'vals_provider': provider,
                    'vals_route': identity.route,
                    'vals_baseline_model': identity.baseline_model,
                }
            ),
        ),
        evaluation_results=results,
    )
    return EvaluationBundle(
        log=log,
        developer=identity.developer,
        model_name=identity.model_name,
    )


def convert_logs(
    payload: dict[str, Any],
    *,
    retrieved_timestamp: str | None = None,
) -> SourceConversionResult[EvaluationBundle]:
    validate_payload(payload)
    timestamp = retrieved_timestamp or str(time.time())
    source_failures = [
        SourceRecordFailure(
            source_ref=str(failure.get('source_ref') or 'unknown source'),
            reason=str(failure.get('reason') or 'unknown failure'),
            source_record=failure.get('source_record'),
        )
        for failure in payload.get('source_failures', [])
        if isinstance(failure, dict)
    ]
    source_exclusions = [
        SourceRecordExclusion(
            source_ref=str(exclusion.get('source_ref') or 'unknown source'),
            reason=str(exclusion.get('reason') or 'unknown exclusion'),
            source_record=exclusion.get('source_record'),
        )
        for exclusion in payload.get('source_exclusions', [])
        if isinstance(exclusion, dict)
    ]
    metric_result = iter_vals_metrics_result(payload)
    if (
        not metric_result.records
        and not metric_result.failures
        and not source_failures
    ):
        raise ValueError(
            'Vals.ai payload did not contain any scored model rows'
        )
    score_scales = {
        benchmark_slug: _score_scale(scores)
        for benchmark_slug, scores in _scores_by_benchmark(
            metric_result.records
        ).items()
    }
    group_result = _group_metrics(metric_result.records)
    bundles = []
    failures = [
        *source_failures,
        *metric_result.failures,
        *group_result.failures,
    ]

    for (benchmark_slug, vals_model_id), rows in group_result.records:
        try:
            bundles.append(
                _make_bundle(
                    benchmark_slug,
                    vals_model_id,
                    rows,
                    score_scales[benchmark_slug],
                    timestamp,
                )
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{benchmark_slug}/{vals_model_id}',
                    reason=str(exc),
                    source_record=[asdict(row) for row in rows],
                )
            )

    if not bundles and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref='Vals.ai payload',
                reason='converted 0 scored source records',
            )
        )
    return SourceConversionResult(
        source_name='Vals.ai',
        total_records=(
            metric_result.total_records
            + len(source_failures)
            + len(source_exclusions)
        ),
        records=bundles,
        failures=failures,
        exclusions=source_exclusions,
    )


def make_logs(
    payload: dict[str, Any],
    *,
    retrieved_timestamp: str | None = None,
) -> list[EvaluationBundle]:
    result = convert_logs(
        payload,
        retrieved_timestamp=retrieved_timestamp,
    )
    result.raise_if_incomplete()
    return result.records


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
            'harness': row.metrics.get('harness'),
            'harness_config': row.metrics.get('harness_config'),
            'paired_accuracy': row.metrics.get('paired_accuracy'),
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
            score_type=(
                ScoreType.continuous
                if score_scale.max_score is not None
                else None
            ),
            min_score=0.0 if score_scale.max_score is not None else None,
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
                        else 'not_provided'
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
            'harness': metrics.get('harness'),
            'harness_config': metrics.get('harness_config'),
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
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=bundle.log,
            base_dir=output_dir,
            developer=bundle.developer,
            model_name=bundle.model_name,
        )
        for bundle in bundles
    )


def save_raw_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding='utf-8',
    )


def _canonical_developer(namespace: str) -> str:
    slug = _slug(namespace)
    return NAMESPACE_DEVELOPER_ALIASES.get(slug, slug)


def _rows_identity(raw_id: str, rows: list[ValsMetric]) -> ValsModelIdentity:
    provider = _optional_str(rows[0].metrics.get('provider'))
    baseline_model = next(
        (
            str(paired['baseline_model'])
            for row in rows
            if isinstance(paired := row.metrics.get('paired_accuracy'), dict)
            and paired.get('baseline_model')
        ),
        None,
    )
    return resolve_model_identity(raw_id, provider, baseline_model)


def resolve_model_identity(
    vals_model_id: str,
    provider: str | None,
    baseline_model: str | None = None,
) -> ValsModelIdentity:
    """Assign Vals route components to explicit model schema fields.

    A host route that Vals pairs with a ``baseline_model`` (the developer's
    own route for the same model) takes its developer and model from that
    baseline, and its inference platform from ``provider``.
    """
    raw_id = require_identity(vals_model_id, 'Vals.ai model id')
    parts = raw_id.split('/')
    if any(not part for part in parts):
        raise ValueError(
            f'Vals.ai model id contains an empty route: {raw_id!r}'
        )

    if (
        baseline_model is not None
        and baseline_model != raw_id
        and len(parts) > 1
        and _slug(parts[0]) in HOSTED_ROUTE_PREFIXES
    ):
        baseline = resolve_model_identity(baseline_model, None)
        return ValsModelIdentity(
            raw_id=raw_id,
            developer=baseline.developer,
            model_name=baseline.model_name,
            model_id=baseline.model_id,
            inference_platform=provider,
            route=parts[0],
            baseline_model=baseline_model,
        )

    inference_platform = provider
    inference_engine = None
    route = None

    if (
        len(parts) == 5
        and _slug(parts[0]) == 'together'
        and _slug(parts[1]) == 'langston'
        and _slug(parts[2]) == 'nim'
    ):
        # Vals exposes this as a Together route through its Langston service
        # and NVIDIA NIM. These are infrastructure, not model-name segments.
        developer = _canonical_developer(parts[3])
        model_name = parts[4]
        inference_engine = 'NIM'
        route = '/'.join(parts[:3])
    elif len(parts) == 3 and _slug(parts[0]) in ROUTE_PREFIXES:
        developer = _canonical_developer(parts[1])
        model_name = parts[2]
        route = parts[0]
    elif len(parts) == 2:
        developer = _canonical_developer(parts[0])
        model_name = parts[1]
    elif len(parts) == 1:
        developer = (
            _canonical_developer(provider)
            if provider
            else get_developer(raw_id)
        )
        model_name = raw_id
    else:
        raise ValueError(
            f'Unsupported Vals.ai model route structure: {raw_id!r}'
        )

    developer = require_identity(
        developer,
        f'Vals.ai developer for model {raw_id!r}',
    )
    model_name = require_identity(
        model_name,
        f'Vals.ai model name for {raw_id!r}',
    )
    model_id = get_model_id(model_name, developer)
    return ValsModelIdentity(
        raw_id=raw_id,
        developer=developer,
        model_name=model_name,
        model_id=model_id,
        inference_platform=inference_platform,
        inference_engine=inference_engine,
        route=route,
    )


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
        max_score=None,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help=(
            'Vals.ai site URL used for leaderboard page links '
            f'(default: {SOURCE_ORGANIZATION_URL})'
        ),
    )
    parser.add_argument(
        '--api-url',
        default=API_URL,
        help=f'Vals.ai benchmark data API URL (default: {API_URL})',
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected source rows and reasons here. Defaults beside '
            '--output-dir when any row fails.'
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = extract_collection(
        input_json=args.input_json,
        benchmark_slugs=args.benchmarks,
        base_url=args.base_url,
        api_url=args.api_url,
    )
    if args.save_raw_json is not None:
        save_raw_payload(payload, args.save_raw_json)

    result = convert_logs(payload)
    paths = export_logs(result.records, args.output_dir)
    print(f'Saved {len(paths)} Vals.ai evaluation logs to {args.output_dir}')
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
    if result.failures:
        result.raise_if_incomplete()


if __name__ == '__main__':
    main()
