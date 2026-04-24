#!/usr/bin/env python3
"""Convert LLM Stats API data into Every Eval Ever aggregate records.

Data source:
- LLM Stats API: https://llm-stats.com/developer

Usage:
    LLM_STATS_API_KEY=... uv run python -m utils.llm_stats.adapter

The adapter consumes a combined offline payload with top-level ``models``,
``benchmarks``, and ``scores`` keys when ``--input-json`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    FetchError,
    fetch_json,
    get_developer,
    sanitize_filename,
    save_evaluation_log,
)

DEFAULT_BASE_URL = 'https://api.llm-stats.com'
ATTRIBUTION_URL = 'https://llm-stats.com/'
DEVELOPER_PAGE_URL = 'https://llm-stats.com/developer'
DEFAULT_OUTPUT_DIR = 'data/llm-stats'

RELATIONSHIP_VALUES = {item.value for item in EvaluatorRelationship}

MODEL_ID_KEYS = (
    'id',
    'slug',
    'model_id',
    'modelId',
    'llm_id',
    'llmId',
    'name',
)
BENCHMARK_ID_KEYS = (
    'id',
    'slug',
    'benchmark_id',
    'benchmarkId',
    'dataset_id',
    'datasetId',
    'name',
)
SCORE_VALUE_KEYS = (
    'score',
    'value',
    'raw_score',
    'rawScore',
    'normalized_score',
    'normalizedScore',
)
PROVENANCE_KEYS = (
    'evaluator_relationship',
    'relationship',
    'provenance',
    'provenance_label',
    'source_type',
    'source_kind',
    'verification',
    'verification_level',
    'verification_tier',
    'submitted_by',
)
URL_KEYS = (
    'citation_url',
    'citationUrl',
    'source_url',
    'sourceUrl',
    'source_urls',
    'sourceUrls',
    'sources',
    'url',
    'paper_url',
    'paperUrl',
    'model_card_url',
    'modelCardUrl',
    'system_card_url',
    'systemCardUrl',
    'reference_url',
    'referenceUrl',
)
MODEL_DETAIL_KEYS = (
    'id',
    'slug',
    'name',
    'model_id',
    'modelId',
    'model_name',
    'modelName',
    'display_name',
    'displayName',
    'organization_id',
    'organizationId',
    'organization_name',
    'organizationName',
    'organization_slug',
    'organizationSlug',
    'context_window',
    'contextWindow',
    'context_length',
    'contextLength',
    'max_context',
    'maxContext',
    'max_input_tokens',
    'maxInputTokens',
    'max_output_tokens',
    'maxOutputTokens',
    'modalities',
    'input_modalities',
    'inputModalities',
    'output_modalities',
    'outputModalities',
    'pricing',
    'price',
    'input_price',
    'inputPrice',
    'output_price',
    'outputPrice',
    'input_cost_per_million',
    'inputCostPerMillion',
    'output_cost_per_million',
    'outputCostPerMillion',
    'release_date',
    'releaseDate',
    'announcement_date',
    'announcementDate',
    'license',
    'status',
    'multimodal',
    'param_count',
    'paramCount',
    'is_open_source',
    'isOpenSource',
)
BENCHMARK_DETAIL_KEYS = (
    'id',
    'slug',
    'name',
    'display_name',
    'displayName',
    'category',
    'categories',
    'modality',
    'language',
    'status',
    'verified',
    'self_reported',
    'selfReported',
    'model_count',
    'modelCount',
    'last_updated',
    'lastUpdated',
)


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert LLM Stats API data to Every Eval Ever format.'
    )
    parser.add_argument(
        '--api-key',
        help='LLM Stats API key. Defaults to LLM_STATS_API_KEY.',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help=f'LLM Stats API base URL (default: {DEFAULT_BASE_URL}).',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help=(
            'Use a combined JSON payload with models, benchmarks, and scores '
            'instead of fetching live data.'
        ),
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help=(
            'Save the raw payload used for conversion. If the path ends in '
            '.json, a combined JSON file is written; otherwise a directory of '
            'endpoint payloads is created.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
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
        if value is not None
    }


def normalize_slug(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw)
    raw = raw.replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    raw = re.sub(r'-{2,}', '-', raw).strip('-')
    return raw or 'unknown'


def is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def nested_get(row: dict[str, Any], dotted_key: str) -> Any:
    current: Any = row
    for part in dotted_key.split('.'):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = nested_get(row, key) if '.' in key else row.get(key)
        if value not in (None, ''):
            return value
    return None


def parse_float(value: Any) -> float | None:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        value = value.strip().removesuffix('%').replace(',', '')
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_collection(payload: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        raise ValueError(f'Expected {name!r} payload to be a list or object.')

    for key in (name, 'data', 'items', 'results'):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = value.get(name)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if value and all(isinstance(item, dict) for item in value.values()):
                return list(value.values())

    if payload and all(isinstance(item, dict) for item in payload.values()):
        return list(payload.values())

    raise ValueError(f'Could not find a list of {name!r} records.')


def load_payload(input_json: Path) -> dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('--input-json must contain a JSON object.')
    return payload


def endpoint_headers(api_key: str) -> dict[str, str]:
    return {
        'Authorization': f'Bearer {api_key}',
        'x-api-key': api_key,
    }


def api_url(base_url: str, path: str) -> str:
    return f'{base_url.rstrip("/")}/{path.lstrip("/")}'


def fetch_payload(api_key: str, base_url: str) -> dict[str, Any]:
    headers = endpoint_headers(api_key)
    models = fetch_json(api_url(base_url, '/v1/models'), headers=headers)
    benchmarks = fetch_json(
        api_url(base_url, '/leaderboard/benchmarks'),
        headers=headers,
    )

    try:
        scores = fetch_json(api_url(base_url, '/v1/scores'), headers=headers)
    except FetchError:
        scores = fetch_benchmark_score_payloads(
            extract_collection(benchmarks, 'benchmarks'),
            base_url,
            headers,
        )

    return {
        'models': models,
        'benchmarks': benchmarks,
        'scores': scores,
    }


def fetch_benchmark_score_payloads(
    benchmarks: list[dict[str, Any]],
    base_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        benchmark_id = benchmark_source_id(benchmark)
        if benchmark_id == 'unknown':
            continue

        try:
            detail = fetch_json(
                api_url(base_url, f'/leaderboard/benchmarks/{benchmark_id}'),
                headers=headers,
            )
        except FetchError as exc:
            print(f'Skipping benchmark {benchmark_id!r}: {exc}')
            continue

        scores.extend(scores_from_benchmark_detail(detail, benchmark))

    return scores


def scores_from_benchmark_detail(
    detail: dict[str, Any],
    benchmark_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = benchmark_summary or {}
    benchmark = {
        **summary,
        'benchmark_id': detail.get('benchmark_id')
        or detail.get('id')
        or summary.get('benchmark_id'),
        'name': detail.get('name')
        or detail.get('benchmark_name')
        or summary.get('name'),
        'description': detail.get('description')
        or detail.get('benchmark_description')
        or summary.get('description'),
        'categories': detail.get('categories') or summary.get('categories'),
        'modality': detail.get('modality') or summary.get('modality'),
        'max_score': detail.get('max_score') or summary.get('max_score'),
        'verified': detail.get('verified') or summary.get('verified'),
        'paper_link': detail.get('paper_link') or summary.get('paper_link'),
        'implementation_link': detail.get('implementation_link')
        or summary.get('implementation_link'),
    }

    entries = detail.get('entries')
    if not isinstance(entries, list):
        entries = detail.get('models')
    if not isinstance(entries, list):
        return []

    scores = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score_value = first_present(
            entry, ('score', 'benchmark_score', 'normalized_score')
        )
        if score_value in (None, ''):
            continue

        score = dict(entry)
        score['benchmark'] = benchmark
        score['model_id'] = entry.get('model_id')
        score['model_name'] = entry.get('model_name')
        score['score'] = score_value
        score.setdefault(
            'id',
            f'{benchmark_source_id(benchmark)}::{entry.get("model_id", "unknown")}',
        )
        if entry.get('self_reported_source'):
            score['source_url'] = entry['self_reported_source']
        scores.append(score)

    return scores


def maybe_save_raw_json(payload: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return

    if path.suffix.lower() == '.json':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        return

    path.mkdir(parents=True, exist_ok=True)
    for endpoint in ('models', 'benchmarks', 'scores'):
        endpoint_path = path / f'{endpoint}.json'
        endpoint_path.write_text(
            json.dumps(payload.get(endpoint), indent=2, sort_keys=True),
            encoding='utf-8',
        )
    (path / 'combined.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def validate_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        extract_collection(payload.get('models'), 'models'),
        extract_collection(payload.get('benchmarks'), 'benchmarks'),
        extract_collection(payload.get('scores'), 'scores'),
    )


def candidate_ids(row: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    ids: set[str] = set()
    for key in keys:
        value = first_present(row, (key,))
        if isinstance(value, dict):
            nested = first_present(value, ('id', 'slug', 'name'))
            if nested not in (None, ''):
                ids.add(str(nested))
        elif value not in (None, ''):
            ids.add(str(value))
    return ids


def build_index(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for candidate in candidate_ids(row, keys):
            index[candidate] = row
    return index


def score_model_ref(score: dict[str, Any]) -> str | None:
    model = score.get('model')
    if isinstance(model, dict):
        value = first_present(model, MODEL_ID_KEYS)
        return str(value) if value not in (None, '') else None
    value = first_present(
        score,
        ('model_id', 'modelId', 'model_slug', 'modelSlug', 'llm_id', 'llmId'),
    )
    if value not in (None, ''):
        return str(value)
    return str(model) if isinstance(model, str) and model else None


def score_benchmark_ref(score: dict[str, Any]) -> str | None:
    benchmark = score.get('benchmark')
    if isinstance(benchmark, dict):
        value = first_present(benchmark, BENCHMARK_ID_KEYS)
        return str(value) if value not in (None, '') else None
    value = first_present(
        score,
        (
            'benchmark_id',
            'benchmarkId',
            'benchmark_slug',
            'benchmarkSlug',
            'dataset_id',
            'datasetId',
        ),
    )
    if value not in (None, ''):
        return str(value)
    return str(benchmark) if isinstance(benchmark, str) and benchmark else None


def resolve_model(
    score: dict[str, Any],
    model_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    embedded = score.get('model')
    if isinstance(embedded, dict):
        return embedded

    ref = score_model_ref(score)
    if ref and ref in model_index:
        return model_index[ref]

    fallback_name = first_present(
        score, ('model_name', 'modelName', 'model_display_name')
    )

    fallback = {
        'id': ref or fallback_name or 'unknown',
        'model_id': ref or fallback_name or 'unknown',
        'name': fallback_name or ref or 'unknown',
        'model_name': fallback_name or ref or 'unknown',
    }
    for key in (
        'organization_id',
        'organizationId',
        'organization_name',
        'organizationName',
        'organization_slug',
        'organizationSlug',
        'context_window',
        'contextWindow',
        'input_cost_per_million',
        'inputCostPerMillion',
        'output_cost_per_million',
        'outputCostPerMillion',
        'release_date',
        'releaseDate',
        'announcement_date',
        'announcementDate',
        'multimodal',
        'param_count',
        'paramCount',
        'is_open_source',
        'isOpenSource',
    ):
        value = first_present(score, (key,))
        if value not in (None, ''):
            fallback[key] = value
    return fallback


def resolve_benchmark(
    score: dict[str, Any],
    benchmark_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    embedded = score.get('benchmark')
    if isinstance(embedded, dict):
        return embedded

    ref = score_benchmark_ref(score)
    if ref and ref in benchmark_index:
        return benchmark_index[ref]

    fallback_name = first_present(
        score, ('benchmark_name', 'benchmarkName', 'dataset_name')
    )
    return {
        'id': ref or fallback_name or 'unknown',
        'name': fallback_name or ref or 'unknown',
    }


def provider_value(model: dict[str, Any]) -> tuple[str | None, str | None]:
    provider = first_present(
        model,
        (
            'provider',
            'developer',
            'organization',
            'creator',
            'model_creator',
            'modelCreator',
        ),
    )
    if isinstance(provider, dict):
        slug = first_present(provider, ('slug', 'id', 'name'))
        name = first_present(provider, ('name', 'slug', 'id'))
        return (
            str(slug) if slug not in (None, '') else None,
            str(name) if name not in (None, '') else None,
        )
    if provider not in (None, ''):
        text = str(provider)
        return text, text

    slug = first_present(
        model,
        (
            'provider_slug',
            'providerSlug',
            'developer_slug',
            'developerSlug',
            'organization_id',
            'organizationId',
            'organization_slug',
            'organizationSlug',
        ),
    )
    name = first_present(
        model,
        (
            'provider_name',
            'providerName',
            'developer_name',
            'developerName',
            'organization_name',
            'organizationName',
        ),
    )
    return (
        str(slug) if slug not in (None, '') else None,
        str(name) if name not in (None, '') else None,
    )


def model_source_id(model: dict[str, Any]) -> str:
    value = first_present(model, MODEL_ID_KEYS)
    return str(value) if value not in (None, '') else 'unknown'


def benchmark_source_id(benchmark: dict[str, Any]) -> str:
    value = first_present(benchmark, BENCHMARK_ID_KEYS)
    return str(value) if value not in (None, '') else 'unknown'


def model_display_name(model: dict[str, Any]) -> str:
    value = first_present(
        model,
        (
            'name',
            'model_name',
            'modelName',
            'display_name',
            'displayName',
            'id',
            'slug',
        ),
    )
    return str(value) if value not in (None, '') else 'unknown'


def benchmark_display_name(benchmark: dict[str, Any]) -> str:
    value = first_present(
        benchmark, ('name', 'display_name', 'displayName', 'id', 'slug')
    )
    return str(value) if value not in (None, '') else 'unknown'


def split_model_id(value: str) -> tuple[str | None, str | None]:
    if '/' not in value:
        return None, value
    developer, model = value.split('/', 1)
    return developer or None, model or None


def normalize_model_info(model: dict[str, Any]) -> tuple[ModelInfo, str, str]:
    raw_id = model_source_id(model)
    raw_developer_from_id, raw_model_from_id = split_model_id(raw_id)
    provider_slug, provider_name = provider_value(model)

    name = model_display_name(model)
    developer_hint = (
        provider_slug or raw_developer_from_id or get_developer(name)
    )
    developer = normalize_slug(developer_hint, 'unknown')

    raw_slug = first_present(model, ('slug', 'model_slug', 'modelSlug'))
    model_hint = raw_slug or raw_model_from_id or raw_id or name
    model_slug = normalize_slug(model_hint, name)

    additional_details = make_model_details(model)
    additional_details.update(
        stringify_details(
            {
                'raw_model_id': raw_id,
                'raw_model_name': name,
                'raw_provider_slug': provider_slug,
                'raw_provider_name': provider_name,
            }
        )
    )

    return (
        ModelInfo(
            name=name,
            id=f'{developer}/{model_slug}',
            developer=developer,
            additional_details=additional_details,
        ),
        developer,
        model_slug,
    )


def make_model_details(model: dict[str, Any]) -> dict[str, str]:
    details: dict[str, Any] = {}
    for key in MODEL_DETAIL_KEYS:
        value = first_present(model, (key,))
        if value not in (None, ''):
            details[f'raw_{normalize_slug(key, key).replace("-", "_")}'] = value

    provider = first_present(
        model,
        (
            'provider',
            'developer',
            'organization',
            'creator',
            'model_creator',
            'modelCreator',
        ),
    )
    if isinstance(provider, dict):
        details['raw_provider_json'] = provider

    return stringify_details(details)


def relationship_from_score(score: dict[str, Any]) -> str:
    explicit = first_present(score, ('evaluator_relationship',))
    if isinstance(explicit, str) and explicit in RELATIONSHIP_VALUES:
        return explicit

    if (
        score.get('is_self_reported') is True
        or score.get('self_reported') is True
    ):
        return EvaluatorRelationship.first_party.value
    if score.get('isSelfReported') is True or score.get('selfReported') is True:
        return EvaluatorRelationship.first_party.value

    labels = []
    for key in PROVENANCE_KEYS:
        value = first_present(score, (key,))
        if value not in (None, '') and not isinstance(value, (dict, list)):
            labels.append(str(value))

    text = ' '.join(labels).lower().replace('-', '_').replace(' ', '_')
    if not text:
        return EvaluatorRelationship.other.value

    if any(
        marker in text
        for marker in (
            'first_party',
            'model_developer',
            'provider',
            'official',
            'model_card',
            'system_card',
            'self_reported',
            'selfreported',
        )
    ):
        return EvaluatorRelationship.first_party.value
    if any(
        marker in text
        for marker in (
            'third_party',
            'independent',
            'external',
            'benchmark_runner',
            'thirdparty',
        )
    ):
        return EvaluatorRelationship.third_party.value
    if 'collaborative' in text or 'joint' in text:
        return EvaluatorRelationship.collaborative.value

    return EvaluatorRelationship.other.value


def provenance_details(score: dict[str, Any]) -> dict[str, str]:
    details: dict[str, Any] = {}
    raw_fields: dict[str, Any] = {}
    for key in PROVENANCE_KEYS:
        value = first_present(score, (key,))
        if value not in (None, ''):
            raw_fields[key] = value
    if raw_fields:
        details['raw_provenance_fields_json'] = raw_fields
        details['raw_provenance_label'] = ' '.join(
            stringify(v) for v in raw_fields.values()
        )
    else:
        details['raw_provenance_label'] = 'unknown'

    for key in ('verified', 'is_verified', 'isVerified', 'status'):
        value = first_present(score, (key,))
        if value not in (None, ''):
            details[f'raw_{normalize_slug(key).replace("-", "_")}'] = value

    return stringify_details(details)


def extract_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.startswith(('http://', 'https://')) else []
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            urls.extend(extract_urls(item))
        return urls
    if isinstance(value, dict):
        urls = []
        for key in URL_KEYS + ('href',):
            if key in value:
                urls.extend(extract_urls(value[key]))
        return urls
    return []


def dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    out = []
    for url in urls:
        if url and url not in seen:
            out.append(url)
            seen.add(url)
    return out


def llm_stats_model_url(model: dict[str, Any]) -> str | None:
    raw_id = model_source_id(model)
    if raw_id == 'unknown':
        return None
    return f'https://llm-stats.com/models/{normalize_slug(raw_id)}'


def llm_stats_benchmark_url(benchmark: dict[str, Any]) -> str | None:
    raw_id = benchmark_source_id(benchmark)
    if raw_id == 'unknown':
        return None
    return f'https://llm-stats.com/benchmarks/{normalize_slug(raw_id)}'


def score_source_urls(
    score: dict[str, Any],
    model: dict[str, Any],
    benchmark: dict[str, Any],
    base_url: str,
) -> list[str]:
    urls: list[str] = []
    for key in URL_KEYS:
        urls.extend(extract_urls(first_present(score, (key,))))
    for key in URL_KEYS:
        urls.extend(extract_urls(first_present(benchmark, (key,))))

    model_url = llm_stats_model_url(model)
    benchmark_url = llm_stats_benchmark_url(benchmark)
    if model_url:
        urls.append(model_url)
    if benchmark_url:
        urls.append(benchmark_url)
    raw_benchmark_id = benchmark_source_id(benchmark)
    if raw_benchmark_id != 'unknown':
        urls.append(
            api_url(base_url, f'/leaderboard/benchmarks/{raw_benchmark_id}')
        )
    return dedupe_urls(urls)


def score_value_and_field(
    score: dict[str, Any],
) -> tuple[float | None, str | None, Any]:
    for key in SCORE_VALUE_KEYS:
        raw_value = first_present(score, (key,))
        value = parse_float(raw_value)
        if value is not None:
            return value, key, raw_value
    return None, None, None


def score_unit(score: dict[str, Any], benchmark: dict[str, Any]) -> str | None:
    value = first_present(
        score,
        ('unit', 'score_unit', 'scoreUnit', 'metric_unit', 'metricUnit'),
    )
    if value not in (None, ''):
        return str(value)
    value = first_present(
        benchmark,
        ('unit', 'score_unit', 'scoreUnit', 'metric_unit', 'metricUnit'),
    )
    return str(value) if value not in (None, '') else None


def lower_is_better(benchmark: dict[str, Any]) -> bool:
    value = first_present(
        benchmark, ('lower_is_better', 'lowerIsBetter', 'lower_better')
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {'true', '1', 'yes'}
    return False


def metric_bounds_and_unit(
    score_value: float,
    raw_unit: str | None,
    benchmark: dict[str, Any],
) -> tuple[float, float, str, str]:
    min_raw = first_present(
        benchmark,
        ('min_score', 'minScore', 'minimum_score', 'minimumScore', 'min'),
    )
    max_raw = first_present(
        benchmark,
        ('max_score', 'maxScore', 'maximum_score', 'maximumScore', 'max'),
    )
    min_score = parse_float(min_raw)
    max_score = parse_float(max_raw)

    normalized_unit = normalize_metric_unit(raw_unit)
    if min_score is not None and max_score is not None:
        if score_value > max_score and max_score == 1.0 and score_value <= 100:
            return 0.0, 100.0, 'percent', 'inferred_percent_from_score'
        if not normalized_unit:
            normalized_unit = 'proportion' if max_score == 1.0 else 'points'
        return min_score, max_score, normalized_unit, 'benchmark_bounds'

    if normalized_unit == 'percent' or (1.0 < score_value <= 100.0):
        return 0.0, 100.0, 'percent', 'inferred_percent'
    if 0.0 <= score_value <= 1.0:
        return 0.0, 1.0, normalized_unit or 'proportion', 'inferred_proportion'

    return (
        0.0,
        max(1.0, score_value),
        normalized_unit or 'points',
        'observed_value',
    )


def normalize_metric_unit(raw_unit: str | None) -> str | None:
    if raw_unit is None:
        return None
    text = raw_unit.strip().lower()
    if text in {'%', 'percent', 'percentage'}:
        return 'percent'
    if text in {'proportion', 'fraction', 'ratio'}:
        return 'proportion'
    if text in {'point', 'points', 'score'}:
        return 'points'
    return text or None


def benchmark_metric_kind(benchmark: dict[str, Any]) -> str:
    value = first_present(
        benchmark,
        (
            'metric_kind',
            'metricKind',
            'metric_type',
            'metricType',
            'score_type',
            'scoreType',
        ),
    )
    return str(value) if value not in (None, '') else 'benchmark_score'


def benchmark_metric_name(benchmark: dict[str, Any]) -> str:
    value = first_present(
        benchmark,
        ('metric_name', 'metricName', 'score_name', 'scoreName'),
    )
    if value not in (None, ''):
        return str(value)
    return f'{benchmark_display_name(benchmark)} score'


def make_source_data(
    score: dict[str, Any],
    model: dict[str, Any],
    benchmark: dict[str, Any],
    base_url: str,
) -> SourceDataUrl:
    raw_benchmark_id = benchmark_source_id(benchmark)
    raw_model_id = model_source_id(model)
    urls = score_source_urls(score, model, benchmark, base_url)
    return SourceDataUrl(
        dataset_name=benchmark_display_name(benchmark),
        source_type='url',
        url=urls,
        additional_details=stringify_details(
            {
                'raw_benchmark_id': raw_benchmark_id,
                'raw_model_id': raw_model_id,
                'source_role': 'aggregator',
            }
        ),
    )


def make_metric_details(
    benchmark: dict[str, Any],
    raw_score_field: str | None,
    bound_strategy: str,
) -> dict[str, str]:
    details: dict[str, Any] = {
        'raw_benchmark_id': benchmark_source_id(benchmark),
        'raw_score_field': raw_score_field,
        'bound_strategy': bound_strategy,
    }
    for key in BENCHMARK_DETAIL_KEYS:
        value = first_present(benchmark, (key,))
        if value not in (None, ''):
            details[f'raw_{normalize_slug(key).replace("-", "_")}'] = value
    return stringify_details(details)


def make_score_details(
    score: dict[str, Any],
    model: dict[str, Any],
    benchmark: dict[str, Any],
    score_value: float,
    raw_score_field: str | None,
    raw_score_value: Any,
    raw_unit: str | None,
    urls: list[str],
) -> ScoreDetails:
    details: dict[str, Any] = {
        'raw_score': raw_score_value,
        'raw_score_unit': raw_unit,
        'raw_score_field': raw_score_field,
        'raw_model_id': model_source_id(model),
        'raw_benchmark_id': benchmark_source_id(benchmark),
        'source_urls_json': urls,
    }

    score_id = first_present(score, ('id', 'score_id', 'scoreId'))
    if score_id not in (None, ''):
        details['raw_score_id'] = score_id

    details.update(provenance_details(score))
    return ScoreDetails(
        score=score_value,
        details=stringify_details(details),
    )


def make_evaluation_result(
    score: dict[str, Any],
    model: dict[str, Any],
    benchmark: dict[str, Any],
    base_url: str,
) -> EvaluationResult | None:
    value, raw_score_field, raw_score_value = score_value_and_field(score)
    if value is None:
        return None

    raw_unit = score_unit(score, benchmark)
    min_score, max_score, metric_unit, bound_strategy = metric_bounds_and_unit(
        value, raw_unit, benchmark
    )
    raw_benchmark_id = benchmark_source_id(benchmark)
    benchmark_slug = normalize_slug(raw_benchmark_id)
    score_id = first_present(score, ('id', 'score_id', 'scoreId'))
    result_id_suffix = normalize_slug(score_id) if score_id else 'score'
    urls = score_source_urls(score, model, benchmark, base_url)

    timestamp = first_present(
        score,
        (
            'evaluation_timestamp',
            'evaluationTimestamp',
            'evaluated_at',
            'evaluatedAt',
            'created_at',
            'createdAt',
            'updated_at',
            'updatedAt',
            'date',
        ),
    )

    return EvaluationResult(
        evaluation_result_id=f'{benchmark_slug}::{result_id_suffix}',
        evaluation_name=f'llm_stats.{benchmark_slug}',
        source_data=make_source_data(score, model, benchmark, base_url),
        evaluation_timestamp=str(timestamp)
        if timestamp not in (None, '')
        else None,
        metric_config=MetricConfig(
            evaluation_description=(
                first_present(benchmark, ('description', 'summary'))
                or f'Score on {benchmark_display_name(benchmark)} reported by LLM Stats.'
            ),
            metric_id=f'llm_stats.{benchmark_slug}.score',
            metric_name=benchmark_metric_name(benchmark),
            metric_kind=benchmark_metric_kind(benchmark),
            metric_unit=metric_unit,
            lower_is_better=lower_is_better(benchmark),
            score_type=ScoreType.continuous,
            min_score=min_score,
            max_score=max_score,
            additional_details=make_metric_details(
                benchmark, raw_score_field, bound_strategy
            ),
        ),
        score_details=make_score_details(
            score,
            model,
            benchmark,
            value,
            raw_score_field,
            raw_score_value,
            raw_unit,
            urls,
        ),
    )


def source_metadata(
    relationship: str,
    base_url: str,
) -> SourceMetadata:
    clean_base = base_url.rstrip('/')
    return SourceMetadata(
        source_name=f'LLM Stats API: {relationship} scores',
        source_type='documentation',
        source_organization_name='LLM Stats',
        source_organization_url=ATTRIBUTION_URL,
        evaluator_relationship=EvaluatorRelationship(relationship),
        additional_details={
            'models_endpoint': f'{clean_base}/v1/models',
            'benchmarks_endpoint': f'{clean_base}/leaderboard/benchmarks',
            'scores_endpoint': (
                f'{clean_base}/leaderboard/benchmarks/{{benchmark_id}}'
            ),
            'developer_page_url': DEVELOPER_PAGE_URL,
            'attribution_url': ATTRIBUTION_URL,
            'attribution_required': 'true',
            'source_role': 'aggregator',
        },
    )


def make_logs(
    payload: dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    retrieved_timestamp: str | None = None,
) -> list[LogBundle]:
    models, benchmarks, scores = validate_payload(payload)
    model_index = build_index(models, MODEL_ID_KEYS)
    benchmark_index = build_index(benchmarks, BENCHMARK_ID_KEYS)
    timestamp = retrieved_timestamp or str(time.time())

    groups: dict[tuple[str, str, str], list[EvaluationResult]] = defaultdict(
        list
    )
    model_infos: dict[tuple[str, str, str], ModelInfo] = {}

    for score in scores:
        model = resolve_model(score, model_index)
        benchmark = resolve_benchmark(score, benchmark_index)
        model_info, developer, model_slug = normalize_model_info(model)
        relationship = relationship_from_score(score)
        result = make_evaluation_result(score, model, benchmark, base_url)
        if result is None:
            continue

        key = (developer, model_slug, relationship)
        groups[key].append(result)
        model_infos[key] = model_info

    bundles: list[LogBundle] = []
    for (developer, model_slug, relationship), results in sorted(
        groups.items()
    ):
        model_info = model_infos[(developer, model_slug, relationship)]
        sanitized_model_id = model_info.id.replace('/', '_')
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=(
                f'llm-stats/{relationship}/{sanitized_model_id}/{timestamp}'
            ),
            retrieved_timestamp=timestamp,
            source_metadata=source_metadata(relationship, base_url),
            eval_library=EvalLibrary(name='LLM Stats', version='unknown'),
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
    if args.save_raw_json is not None and is_subpath(
        args.save_raw_json, args.output_dir
    ):
        raise SystemExit(
            '--save-raw-json must point outside --output-dir so raw API '
            'snapshots are not validated as EvaluationLog files.'
        )

    if args.input_json is not None:
        payload = load_payload(args.input_json)
    else:
        api_key = args.api_key or os.environ.get('LLM_STATS_API_KEY')
        if not api_key:
            raise SystemExit(
                'Missing API key. Set LLM_STATS_API_KEY or pass --api-key.'
            )
        payload = fetch_payload(api_key, args.base_url)

    maybe_save_raw_json(payload, args.save_raw_json)
    bundles = make_logs(payload, args.base_url)
    paths = export_logs(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} LLM Stats model log(s).')
