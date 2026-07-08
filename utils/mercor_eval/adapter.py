#!/usr/bin/env python3
"""Convert Mercor Evaluation Exports API leaderboards to EEE records.

Usage:
    MERCOR_EVAL_API_EVALEVAL_KEY=... \
      uv run python -m utils.mercor_eval.adapter

For deterministic offline replay:
    uv run python -m utils.mercor_eval.adapter \
      --input-json tests/data/mercor_eval/api_payload.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import requests

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    ConfidenceInterval,
    EvalLibrary,
    EvalLimits,
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
    SourceType,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    sanitize_filename,
    save_evaluation_log,
)

DEFAULT_BASE_URL = 'https://coil.mercor.com/external/evals/v1'
DEFAULT_OUTPUT_DIR = Path('data')
API_KEY_ENV = 'MERCOR_EVAL_API_EVALEVAL_KEY'
API_SCHEMA_VERSION = '1.0'
DEFAULT_TIMEOUT = 60
DEFAULT_PAGE_SIZE = 500

PROVIDER_ALIASES = {
    'anthropic': 'anthropic',
    'gemini': 'google',
    'google': 'google',
    'moonshot-ai': 'moonshot',
    'moonshotai': 'moonshot',
    'openai': 'openai',
    'xai': 'xai',
}

FetchPage = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class LogBundle:
    """An EEE log and its datastore path components."""

    log: EvaluationLog
    benchmark_slug: str
    developer: str
    model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Mercor evaluation exports to Every Eval Ever.'
    )
    parser.add_argument(
        '--api-key',
        help=f'Mercor API key. Defaults to {API_KEY_ENV}.',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help=f'Mercor API base URL (default: {DEFAULT_BASE_URL}).',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Use an offline benchmarks/leaderboards payload.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output root directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--page-size',
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f'Leaderboard page size (default: {DEFAULT_PAGE_SIZE}).',
    )
    return parser.parse_args()


def normalize_slug(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw)
    raw = raw.replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    raw = re.sub(r'-{2,}', '-', raw).strip('-')
    return raw or fallback


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    return {
        key: stringify(value)
        for key, value in details.items()
        if value not in (None, '')
    }


def optional_float(value: Any) -> float | None:
    if value in (None, ''):
        return None
    return float(value)


def optional_positive_int(value: Any) -> int | None:
    if value in (None, ''):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def canonical_developer(provider: Any) -> str:
    provider_slug = normalize_slug(provider)
    return PROVIDER_ALIASES.get(provider_slug, provider_slug)


def resolve_model_identity(
    model_name: str,
    provider: Any,
) -> tuple[str, str, str]:
    if '/' in model_name:
        namespace, unqualified_name = model_name.split('/', 1)
        if namespace and unqualified_name:
            developer = canonical_developer(namespace)
            return (
                developer,
                f'{developer}/{unqualified_name}',
                normalize_slug(unqualified_name),
            )

    developer = canonical_developer(provider)
    return developer, f'{developer}/{model_name}', normalize_slug(model_name)


def resolve_api_key(explicit: str | None) -> str:
    api_key = explicit or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f'Mercor API key required via --api-key or {API_KEY_ENV}.'
        )
    return api_key


def validate_api_envelope(payload: Any, endpoint: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f'Mercor {endpoint} endpoint returned a non-object payload.'
        )
    schema_version = payload.get('schemaVersion')
    if schema_version != API_SCHEMA_VERSION:
        raise ValueError(
            f'Mercor {endpoint} schemaVersion must be '
            f'{API_SCHEMA_VERSION!r}, got {schema_version!r}.'
        )
    return payload


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f'Failed to fetch Mercor endpoint {url}: {exc}'
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f'Failed to parse JSON from Mercor endpoint {url}: {exc}'
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f'Mercor endpoint {url} returned a non-object payload.'
        )
    return payload


def fetch_payload(
    api_key: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    fetch_page: FetchPage = request_json,
) -> dict[str, Any]:
    if not 1 <= page_size <= 500:
        raise ValueError(
            'Mercor leaderboard page size must be between 1 and 500.'
        )

    base_url = base_url.rstrip('/')
    headers = {'X-API-Key': api_key}
    benchmarks = fetch_page(
        f'{base_url}/benchmarks',
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )
    validate_api_envelope(benchmarks, 'benchmarks')

    rows: list[dict[str, Any]] = []
    offset = 0
    data_as_of = None
    while True:
        page = fetch_page(
            f'{base_url}/leaderboards',
            headers=headers,
            params={'limit': page_size, 'offset': offset},
            timeout=DEFAULT_TIMEOUT,
        )
        validate_api_envelope(page, 'leaderboards')
        page_rows = page.get('rows')
        total = page.get('total')
        if not isinstance(page_rows, list) or not isinstance(total, int):
            raise ValueError('Invalid Mercor leaderboard pagination envelope.')
        if not page_rows and offset < total:
            raise ValueError(
                'Mercor leaderboard pagination stopped before total.'
            )
        rows.extend(page_rows)
        data_as_of = page.get('dataAsOf') or data_as_of
        offset += len(page_rows)
        if offset >= total:
            break

    return {
        'benchmarks': benchmarks,
        'leaderboards': {
            'schemaVersion': API_SCHEMA_VERSION,
            'rows': rows,
            'total': len(rows),
            'limit': page_size,
            'offset': 0,
            'dataAsOf': data_as_of,
        },
    }


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('--input-json must contain a JSON object.')
    return payload


def make_source_data(
    benchmark: dict[str, Any],
    base_url: str,
) -> SourceDataHf:
    benchmark_id = str(benchmark['benchmarkId'])
    benchmark_slug = normalize_slug(benchmark['benchmarkName'])
    query = urlencode({'benchmarkId': benchmark_id})
    return SourceDataHf(
        dataset_name=benchmark_slug,
        source_type='hf_dataset',
        hf_repo=f'mercor/{benchmark_slug}',
        additional_details=stringify_details(
            {
                'benchmark_id': benchmark_id,
                'num_tasks': benchmark.get('numTasks'),
                'domains': benchmark.get('domains'),
                'access': 'authenticated_api',
                'api_url': f'{base_url.rstrip("/")}/leaderboards?{query}',
            }
        ),
    )


def result_scope_slug(value: str) -> str:
    return normalize_slug(value).replace('-', '_')


def evaluation_result_id(
    parent_evaluation_id: str,
    evaluation_name: str,
    metric_id: str,
    metric_parameters: dict[str, str | float | bool | int | None] | None,
) -> str:
    suffix = metric_id
    if metric_parameters:
        parts = []
        for key, value in sorted(metric_parameters.items()):
            if value is not None:
                parts.append(f'{normalize_slug(key)}_{normalize_slug(value)}')
        if parts:
            suffix = f'{suffix}__{"__".join(parts)}'
    return (
        f'{parent_evaluation_id}#{result_scope_slug(evaluation_name)}#{suffix}'
    )


def make_generation_config(config: dict[str, Any]) -> GenerationConfig:
    agent_details = stringify_details(
        {
            'agent_name': config.get('agentName'),
            'agent_config_id': config.get('agentConfigId'),
        }
    )
    return GenerationConfig(
        generation_args=GenerationArgs(
            temperature=optional_float(config.get('temperature')),
            max_tokens=optional_positive_int(config.get('maxTokens')),
            agentic_eval_config=AgenticEvalConfig(
                additional_details=agent_details or None,
            ),
            eval_limits=EvalLimits(
                time_limit=optional_positive_int(config.get('timeoutSec')),
                message_limit=optional_positive_int(config.get('maxSteps')),
            ),
        ),
        additional_details=stringify_details(
            {
                'reasoning_effort': config.get('reasoningEffort'),
                'verbosity': config.get('verbosity'),
                'summary': config.get('summary'),
            }
        )
        or None,
    )


def make_metric_result(
    *,
    benchmark: dict[str, Any],
    row: dict[str, Any],
    base_url: str,
    parent_evaluation_id: str,
    evaluation_name: str,
    metric_id: str,
    metric_name: str,
    metric_kind: str,
    value: Any,
    metric_unit: str = 'proportion',
    metric_parameters: dict[str, str | float | bool | int | None] | None = None,
    ci95: dict[str, Any] | None = None,
) -> EvaluationResult:
    uncertainty = None
    if ci95 is not None:
        uncertainty = Uncertainty(
            confidence_interval=ConfidenceInterval(
                lower=float(ci95['lower']),
                upper=float(ci95['upper']),
                confidence_level=0.95,
                method='seeded_percentile_bootstrap',
            ),
            num_samples=int(benchmark['numTasks']),
        )

    config = row['model'].get('config') or {}
    return EvaluationResult(
        evaluation_result_id=evaluation_result_id(
            parent_evaluation_id,
            evaluation_name,
            metric_id,
            metric_parameters,
        ),
        evaluation_name=evaluation_name,
        source_data=make_source_data(benchmark, base_url),
        evaluation_timestamp=row.get('evaluatedAt'),
        metric_config=MetricConfig(
            evaluation_description=(
                f'{evaluation_name} {metric_name} reported by Mercor for '
                f'{benchmark["benchmarkName"]}.'
            ),
            metric_id=metric_id,
            metric_name=metric_name,
            metric_kind=metric_kind,
            metric_unit=metric_unit,
            metric_parameters=metric_parameters,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
            additional_details={'raw_scope': evaluation_name},
        ),
        score_details=ScoreDetails(
            score=float(value),
            details={'num_trials': str(row['numTrials'])},
            uncertainty=uncertainty,
        ),
        generation_config=make_generation_config(config),
    )


def make_results(
    row: dict[str, Any],
    benchmark: dict[str, Any],
    base_url: str,
    parent_evaluation_id: str,
) -> list[EvaluationResult]:
    metrics = row.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Mercor leaderboard row is missing metrics.')

    pass_at_1 = metrics.get('passAt1')
    pass_at_8 = metrics.get('passAt8')
    if not isinstance(pass_at_1, dict) or not isinstance(pass_at_8, dict):
        raise ValueError('Mercor leaderboard row is missing Pass@k metrics.')

    results = [
        make_metric_result(
            benchmark=benchmark,
            row=row,
            base_url=base_url,
            parent_evaluation_id=parent_evaluation_id,
            evaluation_name='Overall',
            metric_id='pass_at_k',
            metric_name='Pass@1',
            metric_kind='pass_rate',
            metric_parameters={'k': 1},
            value=pass_at_1['value'],
            ci95=pass_at_1.get('ci95'),
        ),
        make_metric_result(
            benchmark=benchmark,
            row=row,
            base_url=base_url,
            parent_evaluation_id=parent_evaluation_id,
            evaluation_name='Overall',
            metric_id='pass_at_k',
            metric_name='Pass@8',
            metric_kind='pass_rate',
            metric_parameters={'k': 8},
            value=pass_at_8['value'],
            ci95=pass_at_8.get('ci95'),
        ),
        make_metric_result(
            benchmark=benchmark,
            row=row,
            base_url=base_url,
            parent_evaluation_id=parent_evaluation_id,
            evaluation_name='Overall',
            metric_id='pass_hat_k',
            metric_name='Pass^8',
            metric_kind='pass_rate',
            metric_parameters={'k': 8, 'estimator': 'naive'},
            value=metrics['passHat8'],
        ),
        make_metric_result(
            benchmark=benchmark,
            row=row,
            base_url=base_url,
            parent_evaluation_id=parent_evaluation_id,
            evaluation_name='Overall',
            metric_id='mean_score',
            metric_name='Mean Score',
            metric_kind='score',
            value=metrics['meanScore'],
        ),
    ]

    per_domain = metrics.get('perDomainPassAt1')
    if not isinstance(per_domain, dict):
        raise ValueError('Mercor leaderboard row is missing perDomainPassAt1.')
    for domain, value in sorted(per_domain.items()):
        results.append(
            make_metric_result(
                benchmark=benchmark,
                row=row,
                base_url=base_url,
                parent_evaluation_id=parent_evaluation_id,
                evaluation_name=str(domain),
                metric_id='pass_at_k',
                metric_name=f'Pass@1 - {domain}',
                metric_kind='pass_rate',
                metric_parameters={'k': 1},
                value=value,
            )
        )
    return results


def make_bundles(
    payload: dict[str, Any],
    *,
    retrieved_timestamp: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> list[LogBundle]:
    benchmark_envelope = validate_api_envelope(
        payload.get('benchmarks'), 'benchmarks'
    )
    leaderboard_envelope = validate_api_envelope(
        payload.get('leaderboards'), 'leaderboards'
    )

    benchmarks = benchmark_envelope.get('benchmarks')
    rows = leaderboard_envelope.get('rows')
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError('Mercor benchmarks payload contains no benchmarks.')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Mercor leaderboard payload contains no rows.')

    benchmark_index = {}
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict) or not benchmark.get('benchmarkId'):
            raise ValueError(
                'Mercor benchmarks payload contains an invalid row.'
            )
        benchmark_index[str(benchmark['benchmarkId'])] = benchmark

    retrieved_timestamp = retrieved_timestamp or str(time.time())
    api_data_as_of = leaderboard_envelope.get(
        'dataAsOf'
    ) or benchmark_envelope.get('dataAsOf')
    bundles = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                'Mercor leaderboard payload contains an invalid row.'
            )
        benchmark_ref = row.get('benchmark')
        if not isinstance(benchmark_ref, dict):
            raise ValueError('Mercor leaderboard row is missing benchmark.')
        benchmark_id = str(benchmark_ref.get('id') or '')
        benchmark = benchmark_index.get(benchmark_id)
        if benchmark is None:
            raise ValueError(
                f'Mercor leaderboard references unknown benchmark {benchmark_id!r}.'
            )

        model = row.get('model')
        if not isinstance(model, dict):
            raise ValueError('Mercor leaderboard row is missing model.')
        config = model.get('config') or {}
        if not isinstance(config, dict):
            raise ValueError('Mercor model config must be an object.')
        provider = config.get('provider') or 'unknown'
        model_name = str(model.get('name') or config.get('model') or 'unknown')
        developer, model_id, output_model_name = resolve_model_identity(
            model_name,
            provider,
        )
        mercor_model_id = str(model.get('id') or 'unknown')
        evaluation_id = str(row.get('evaluationId') or 'unknown')
        benchmark_slug = normalize_slug(benchmark['benchmarkName'])
        log_evaluation_id = (
            f'{benchmark_slug}/{developer}_{output_model_name}/'
            f'{retrieved_timestamp}'
        )

        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=log_evaluation_id,
            evaluation_timestamp=row.get('evaluatedAt'),
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name=(
                    f'Mercor {benchmark["benchmarkName"]} Leaderboard'
                ),
                source_type=SourceType.evaluation_run,
                source_organization_name='Mercor',
                source_organization_url='https://www.mercor.com',
                evaluator_relationship=EvaluatorRelationship.first_party,
                additional_details=stringify_details(
                    {
                        'benchmark_id': benchmark_id,
                        'benchmark_name': benchmark.get('benchmarkName'),
                        'evaluation_id': evaluation_id,
                        'api_schema_version': API_SCHEMA_VERSION,
                        'data_as_of': api_data_as_of,
                    }
                ),
            ),
            eval_library=EvalLibrary(
                name='Mercor Evaluation Exports API',
                version=API_SCHEMA_VERSION,
            ),
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                inference_platform=str(provider),
                additional_details=stringify_details(
                    {
                        'mercor_model_id': mercor_model_id,
                        'provider': provider,
                        'run_config': config,
                    }
                ),
            ),
            evaluation_results=make_results(
                row,
                benchmark,
                base_url,
                log_evaluation_id,
            ),
        )
        bundles.append(
            LogBundle(
                log=log,
                benchmark_slug=benchmark_slug,
                developer=developer,
                model=output_model_name,
            )
        )
    return bundles


def export_bundles(
    bundles: list[LogBundle],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    paths = []
    for bundle in bundles:
        paths.append(
            save_evaluation_log(
                bundle.log,
                Path(output_dir) / bundle.benchmark_slug,
                bundle.developer,
                bundle.model,
            )
        )
    return paths


def main() -> None:
    args = parse_args()
    if args.input_json:
        payload = load_payload(args.input_json)
    else:
        payload = fetch_payload(
            resolve_api_key(args.api_key),
            base_url=args.base_url,
            page_size=args.page_size,
        )

    bundles = make_bundles(payload, base_url=args.base_url)
    paths = export_bundles(bundles, args.output_dir)
    for path in paths:
        print(f'Saved: {path}')
    print(f'Done! Generated {len(paths)} Mercor evaluation record(s).')


if __name__ == '__main__':
    main()
