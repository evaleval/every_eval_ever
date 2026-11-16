#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
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
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    FetchError,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    require_finite_number,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

SOURCE_URL = 'https://artificialanalysis.ai/api/v2/language/models'
FREE_SOURCE_URL = 'https://artificialanalysis.ai/api/v2/language/models/free'
API_REFERENCE_URL = 'https://artificialanalysis.ai/data-api/docs'
METHODOLOGY_URL = 'https://artificialanalysis.ai/methodology'
ATTRIBUTION_URL = 'https://artificialanalysis.ai/'
BENCHMARK_NAME = 'artificial-analysis-llms'
OUTPUT_DIR = f'data/{BENCHMARK_NAME}'


@dataclass(frozen=True)
class MetricSpec:
    evaluation_name: str
    source_section: str
    source_key: str
    metric_name: str
    evaluation_description: str
    metric_kind: str
    metric_unit: str
    lower_is_better: bool
    min_score: float
    max_score: float
    include_prompt_options: bool = False
    fallback_keys: tuple[str, ...] = ()


METRIC_SPECS = [
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_intelligence_index',
        source_section='evaluations',
        source_key='artificial_analysis_intelligence_index',
        metric_name='Artificial Analysis Intelligence Index',
        evaluation_description='Artificial Analysis composite intelligence index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_coding_index',
        source_section='evaluations',
        source_key='artificial_analysis_coding_index',
        metric_name='Artificial Analysis Coding Index',
        evaluation_description='Artificial Analysis composite coding index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_math_index',
        source_section='evaluations',
        source_key='artificial_analysis_math_index',
        metric_name='Artificial Analysis Math Index',
        evaluation_description='Artificial Analysis composite math index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_agentic_index',
        source_section='evaluations',
        source_key='artificial_analysis_agentic_index',
        metric_name='Artificial Analysis Agentic Index',
        evaluation_description='Artificial Analysis composite agentic index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_finance_and_accounting_index',
        source_section='evaluations',
        source_key='artificial_analysis_finance_and_accounting_index',
        metric_name='Artificial Analysis Finance & Accounting Index',
        evaluation_description='Artificial Analysis Finance & Accounting capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_strategy_and_ops_index',
        source_section='evaluations',
        source_key='artificial_analysis_strategy_and_ops_index',
        metric_name='Artificial Analysis Strategy & Ops Index',
        evaluation_description='Artificial Analysis Strategy & Ops capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_legal_index',
        source_section='evaluations',
        source_key='artificial_analysis_legal_index',
        metric_name='Artificial Analysis Legal Index',
        evaluation_description='Artificial Analysis Legal capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_healthcare_and_medical_index',
        source_section='evaluations',
        source_key='artificial_analysis_healthcare_and_medical_index',
        metric_name='Artificial Analysis Healthcare & Medical Index',
        evaluation_description='Artificial Analysis Healthcare & Medical capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_engineering_index',
        source_section='evaluations',
        source_key='artificial_analysis_engineering_index',
        metric_name='Artificial Analysis Engineering Index',
        evaluation_description='Artificial Analysis Engineering capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.artificial_analysis_economics_index',
        source_section='evaluations',
        source_key='artificial_analysis_economics_index',
        metric_name='Artificial Analysis Economics Index',
        evaluation_description='Artificial Analysis Economics capability index.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.mmlu_pro',
        source_section='evaluations',
        source_key='mmlu_pro',
        metric_name='MMLU-Pro',
        evaluation_description='Benchmark score on MMLU-Pro.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.gpqa',
        source_section='evaluations',
        source_key='gpqa',
        metric_name='GPQA',
        evaluation_description='Benchmark score on GPQA.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
        fallback_keys=('gpqa_diamond',),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.hle',
        source_section='evaluations',
        source_key='hle',
        metric_name="Humanity's Last Exam",
        evaluation_description="Benchmark score on Humanity's Last Exam.",
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.livecodebench',
        source_section='evaluations',
        source_key='livecodebench',
        metric_name='LiveCodeBench',
        evaluation_description='Benchmark score on LiveCodeBench.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.scicode',
        source_section='evaluations',
        source_key='scicode',
        metric_name='SciCode',
        evaluation_description='Benchmark score on SciCode.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.math_500',
        source_section='evaluations',
        source_key='math_500',
        metric_name='MATH-500',
        evaluation_description='Benchmark score on MATH-500.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.aime',
        source_section='evaluations',
        source_key='aime',
        metric_name='AIME',
        evaluation_description='Benchmark score on AIME.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.aime_25',
        source_section='evaluations',
        source_key='aime_25',
        metric_name='AIME 2025',
        evaluation_description='Benchmark score on AIME 2025.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.ifbench',
        source_section='evaluations',
        source_key='ifbench',
        metric_name='IFBench',
        evaluation_description='Benchmark score on IFBench.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.lcr',
        source_section='evaluations',
        source_key='lcr',
        metric_name='AA-LCR',
        evaluation_description='Benchmark score on AA-LCR.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
        fallback_keys=('aa_lcr',),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.terminalbench_hard',
        source_section='evaluations',
        source_key='terminalbench_hard',
        metric_name='Terminal-Bench Hard',
        evaluation_description='Benchmark score on Terminal-Bench Hard.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.tau2',
        source_section='evaluations',
        source_key='tau2',
        metric_name='tau2',
        evaluation_description='Benchmark score on tau2.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.tau2_telecom',
        source_section='evaluations',
        source_key='tau2_telecom',
        metric_name='tau2-bench Telecom',
        evaluation_description='Benchmark score on tau2-bench Telecom.',
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.price_1m_blended_3_to_1',
        source_section='pricing',
        source_key='price_1m_blended_3_to_1',
        metric_name='Price per 1M tokens (blended 3:1)',
        evaluation_description='Blended price per 1M tokens using a 3:1 input-to-output ratio.',
        metric_kind='cost',
        metric_unit='usd_per_1m_tokens',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.price_1m_input_tokens',
        source_section='pricing',
        source_key='price_1m_input_tokens',
        metric_name='Price per 1M input tokens',
        evaluation_description='Price per 1M input tokens in USD.',
        metric_kind='cost',
        metric_unit='usd_per_1m_tokens',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.price_1m_output_tokens',
        source_section='pricing',
        source_key='price_1m_output_tokens',
        metric_name='Price per 1M output tokens',
        evaluation_description='Price per 1M output tokens in USD.',
        metric_kind='cost',
        metric_unit='usd_per_1m_tokens',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.median_output_tokens_per_second',
        source_section='model',
        source_key='median_output_tokens_per_second',
        metric_name='Median output tokens per second',
        evaluation_description='Median output generation speed reported by Artificial Analysis.',
        metric_kind='throughput',
        metric_unit='tokens_per_second',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
        include_prompt_options=True,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.median_time_to_first_token_seconds',
        source_section='model',
        source_key='median_time_to_first_token_seconds',
        metric_name='Median time to first token',
        evaluation_description='Median time to first token reported by Artificial Analysis.',
        metric_kind='latency',
        metric_unit='seconds',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
        include_prompt_options=True,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.median_time_to_first_answer_token',
        source_section='model',
        source_key='median_time_to_first_answer_token',
        metric_name='Median time to first answer token',
        evaluation_description='Median time to first answer token reported by Artificial Analysis.',
        metric_kind='latency',
        metric_unit='seconds',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
        include_prompt_options=True,
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.mean_output_tokens_per_second',
        source_section='performance_v3',
        source_key='mean_output_tokens_per_second',
        metric_name='Mean output tokens per second (Performance v3)',
        evaluation_description='Headline end-to-end output speed reported by Artificial Analysis Performance v3.',
        metric_kind='throughput',
        metric_unit='tokens_per_second',
        lower_is_better=False,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.mean_time_to_first_token_seconds',
        source_section='performance_v3',
        source_key='mean_time_to_first_token_seconds',
        metric_name='Mean time to first token (Performance v3)',
        evaluation_description='Mean time to first token reported by Artificial Analysis Performance v3.',
        metric_kind='latency',
        metric_unit='seconds',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    ),
    MetricSpec(
        evaluation_name='artificial_analysis.mean_time_to_first_answer_token_seconds',
        source_section='performance_v3',
        source_key='mean_time_to_first_answer_token_seconds',
        metric_name='Mean time to first answer token (Performance v3)',
        evaluation_description='Mean time to first answer token reported by Artificial Analysis Performance v3.',
        metric_kind='latency',
        metric_unit='seconds',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Artificial Analysis LLM API data to Every Eval Ever format.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Directory to write evaluation logs (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Use a saved API response instead of fetching live data.',
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help='Write the raw API response used for this run to the given file.',
    )
    parser.add_argument(
        '--api-key',
        help='Artificial Analysis API key. Defaults to ARTIFICIAL_ANALYSIS_API_KEY.',
    )
    return parser.parse_args(argv)


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normalize_slug(value: str | None, fallback: str) -> str:
    raw = require_identity(value or fallback, 'Artificial Analysis identity')
    raw = raw.lower()
    sanitized = sanitize_filename(raw)
    sanitized = sanitized.replace(' ', '-').replace('_', '-')
    while '--' in sanitized:
        sanitized = sanitized.replace('--', '-')
    sanitized = sanitized.strip('-')
    if not sanitized:
        raise ValueError(
            'Artificial Analysis identity has no usable path characters'
        )
    return sanitized


def make_model_path_name(model: dict[str, Any]) -> str:
    name = require_identity(model.get('name'), 'Artificial Analysis model name')
    return normalize_slug(model.get('slug'), name)


def load_payload(input_json: Path) -> dict[str, Any]:
    return json.loads(input_json.read_text(encoding='utf-8'))


MAX_PAGINATION_PAGES = 50


def _is_tier_forbidden(exc: FetchError) -> bool:
    response = getattr(exc.__cause__, 'response', None)
    if getattr(response, 'status_code', None) != 403:
        return False
    headers = getattr(response, 'headers', None) or {}
    content_type = str(headers.get('Content-Type') or headers.get('content-type') or '')
    return 'text/html' not in content_type.lower()


def fetch_payload(api_key: str) -> dict[str, Any]:
    headers = {'x-api-key': api_key}
    endpoint = SOURCE_URL
    try:
        first_page = fetch_json(endpoint, headers=headers)
    except FetchError as exc:
        if not _is_tier_forbidden(exc):
            raise
        endpoint = FREE_SOURCE_URL
        first_page = fetch_json(endpoint, headers=headers)

    if not isinstance(first_page, dict):
        raise ValueError('Artificial Analysis API response must be a JSON object.')
    all_models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _append_unique(rows: list[dict[str, Any]]) -> None:
        for item in rows:
            raw_id = item.get('id') if isinstance(item, dict) else None
            key = str(raw_id) if raw_id is not None else None
            if key is not None:
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            all_models.append(item)

    _append_unique(validate_payload(first_page))
    pagination = first_page.get('pagination')
    page = 1
    while (
        isinstance(pagination, dict)
        and pagination.get('has_more') is True
        and page < MAX_PAGINATION_PAGES
    ):
        page += 1
        sep = '&' if '?' in endpoint else '?'
        next_payload = fetch_json(f'{endpoint}{sep}page={page}', headers=headers)
        if not isinstance(next_payload, dict):
            raise ValueError(
                f'Artificial Analysis page {page} response must be a JSON object.'
            )
        _append_unique(validate_payload(next_payload))
        pagination = next_payload.get('pagination')

    if isinstance(pagination, dict) and pagination.get('has_more') is True:
        raise ValueError(
            f'Artificial Analysis pagination exceeded {MAX_PAGINATION_PAGES} pages.'
        )

    merged = dict(first_page)
    merged['data'] = all_models
    merged['api_endpoint'] = endpoint
    return merged


def validate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get('data')
    if not isinstance(data, list):
        raise ValueError(
            "Expected payload['data'] to be a list of model objects."
        )
    return data


def maybe_save_raw_json(payload: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
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


def is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def get_metric_value(
    model: dict[str, Any], spec: MetricSpec
) -> tuple[float, str, str] | None:
    raw_metric_field = spec.source_key
    if spec.source_section == 'model':
        value = model.get(spec.source_key)
        raw_value_field = spec.source_key
        if value is None:
            perf = model.get('performance')
            if isinstance(perf, dict):
                value = perf.get(spec.source_key)
                if value is not None:
                    raw_value_field = f'performance.{spec.source_key}'
                elif spec.source_key == 'median_time_to_first_answer_token':
                    alt = 'median_time_to_first_answer_token_seconds'
                    value = perf.get(alt)
                    if value is not None:
                        raw_metric_field = alt
                        raw_value_field = f'performance.{alt}'
    else:
        container = model.get(spec.source_section, {})
        if not isinstance(container, dict):
            return None
        value = container.get(spec.source_key)
        raw_value_field = f'{spec.source_section}.{spec.source_key}'
        if value is None:
            for alt in spec.fallback_keys:
                value = container.get(alt)
                if value is not None:
                    raw_metric_field = alt
                    raw_value_field = f'{spec.source_section}.{alt}'
                    break

    if value is None:
        return None
    score = require_finite_number(
        value, f'Artificial Analysis metric {raw_value_field}'
    )
    return score, raw_metric_field, raw_value_field


def make_source_data(endpoint: str = SOURCE_URL) -> SourceDataUrl:
    return SourceDataUrl(
        dataset_name='Artificial Analysis LLM API',
        source_type='url',
        url=[endpoint],
    )


def make_source_metadata_details(
    payload: dict[str, Any],
) -> dict[str, str]:
    prompt_options = payload.get('prompt_options', {})
    endpoint = str(payload.get('api_endpoint') or SOURCE_URL)
    details = {
        'api_endpoint': endpoint,
        'api_reference_url': API_REFERENCE_URL,
        'methodology_url': METHODOLOGY_URL,
        'attribution_url': ATTRIBUTION_URL,
        'attribution_required': 'true',
        'endpoint_scope': 'llms',
    }
    if payload.get('tier') is not None:
        details['api_tier'] = stringify(payload['tier'])

    if isinstance(prompt_options, dict):
        for key in ('prompt_length', 'parallel_queries'):
            if key in prompt_options and prompt_options[key] is not None:
                details[key] = stringify(prompt_options[key])

    return details


def make_model_info(model: dict[str, Any]) -> tuple[ModelInfo, str, str]:
    creator = model.get('model_creator') or {}
    if not isinstance(creator, dict):
        raise ValueError('Artificial Analysis model_creator must be an object')
    creator_name = require_identity(
        creator.get('name'), 'Artificial Analysis model creator name'
    )
    developer = normalize_slug(creator.get('slug'), creator_name)
    model_path_name = make_model_path_name(model)

    additional_details = {
        'raw_model_id': stringify(model['id']),
        'raw_model_name': stringify(model['name']),
        'raw_model_slug': stringify(model.get('slug')),
        'raw_creator_id': stringify(creator.get('id')),
        'raw_creator_name': stringify(creator.get('name')),
    }
    if creator.get('slug') is not None:
        additional_details['raw_creator_slug'] = stringify(creator['slug'])
    if model.get('release_date') is not None:
        additional_details['release_date'] = stringify(model['release_date'])

    return (
        ModelInfo(
            name=str(model['name']),
            id=f'{developer}/{model_path_name}',
            developer=developer,
            inference_platform='unknown',
            additional_details=additional_details,
        ),
        developer,
        model_path_name,
    )


def make_metric_config(
    spec: MetricSpec,
    prompt_options: dict[str, Any],
    *,
    raw_metric_field: str | None = None,
) -> MetricConfig:
    max_score = spec.max_score
    bound_strategy = 'unbounded' if math.isinf(max_score) else 'fixed'
    additional_details = {
        'raw_metric_field': raw_metric_field or spec.source_key,
        'bound_strategy': bound_strategy,
    }

    metric_parameters: dict[str, str | float | bool | None] | None = None
    if spec.include_prompt_options and isinstance(prompt_options, dict):
        metric_parameters = {}
        for key in ('prompt_length', 'parallel_queries'):
            if key in prompt_options:
                metric_parameters[key] = prompt_options[key]
        if not metric_parameters:
            metric_parameters = None

    return MetricConfig(
        evaluation_description=spec.evaluation_description,
        metric_id=spec.evaluation_name,
        metric_name=spec.metric_name,
        metric_kind=spec.metric_kind,
        metric_unit=spec.metric_unit,
        metric_parameters=metric_parameters,
        lower_is_better=spec.lower_is_better,
        score_type=ScoreType.continuous,
        min_score=spec.min_score,
        max_score=max_score,
        additional_details=additional_details,
    )


def maybe_make_uncertainty(
    model: dict[str, Any], spec: MetricSpec
) -> Uncertainty | None:
    if spec.source_key != 'median_output_tokens_per_second':
        return None

    perf = model.get('performance')
    perf_dict = perf if isinstance(perf, dict) else {}
    lower = model.get('tokens_per_second_q025')
    if lower is None:
        lower = perf_dict.get('tokens_per_second_q025')
    upper = model.get('tokens_per_second_q975')
    if upper is None:
        upper = perf_dict.get('tokens_per_second_q975')
    if lower is None or upper is None:
        return None

    return Uncertainty(
        confidence_interval={
            'lower': require_finite_number(
                lower, 'Artificial Analysis confidence interval lower'
            ),
            'upper': require_finite_number(
                upper, 'Artificial Analysis confidence interval upper'
            ),
            'confidence_level': 0.95,
            'method': 'source_reported',
        }
    )


def make_score_details(
    model: dict[str, Any],
    spec: MetricSpec,
    score: float,
    *,
    raw_value_field: str | None = None,
) -> ScoreDetails:
    details: dict[str, str] = {
        'raw_model_id': stringify(model['id']),
        'raw_value_field': raw_value_field
        or (
            spec.source_key
            if spec.source_section == 'model'
            else f'{spec.source_section}.{spec.source_key}'
        ),
    }

    uncertainty = maybe_make_uncertainty(model, spec)

    return ScoreDetails(
        score=score,
        details=details,
        uncertainty=uncertainty,
    )


def make_evaluation_results(
    model: dict[str, Any],
    payload: dict[str, Any],
    *,
    failures: list[SourceRecordFailure] | None = None,
    source_ref: str | None = None,
) -> list[EvaluationResult]:
    prompt_options = payload.get('prompt_options', {})
    endpoint = str(payload.get('api_endpoint') or SOURCE_URL)
    source_data = make_source_data(endpoint)
    results: list[EvaluationResult] = []

    for spec in METRIC_SPECS:
        try:
            extracted = get_metric_value(model, spec)
            if extracted is None:
                continue
            score, raw_metric_field, raw_value_field = extracted
            results.append(
                EvaluationResult(
                    evaluation_result_id=spec.evaluation_name,
                    evaluation_name=spec.evaluation_name,
                    source_data=source_data,
                    metric_config=make_metric_config(
                        spec,
                        prompt_options,
                        raw_metric_field=raw_metric_field,
                    ),
                    score_details=make_score_details(
                        model,
                        spec,
                        score,
                        raw_value_field=raw_value_field,
                    ),
                )
            )
        except Exception as exc:
            if failures is None:
                raise
            failures.append(
                SourceRecordFailure(
                    source_ref=(
                        f'{source_ref or model.get("id", "model")} '
                        f'metric {spec.source_section}.{spec.source_key}'
                    ),
                    reason=str(exc),
                    source_record={
                        'model': model,
                        'metric_section': spec.source_section,
                        'metric_key': spec.source_key,
                    },
                )
            )

    if not results:
        raise ValueError('model has no usable evaluation metrics')

    return results


def make_log(
    model: dict[str, Any],
    payload: dict[str, Any],
    retrieved_timestamp: str,
    *,
    failures: list[SourceRecordFailure] | None = None,
    source_ref: str | None = None,
) -> tuple[EvaluationLog, str, str]:
    model_info, developer, model_path_name = make_model_info(model)

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'{BENCHMARK_NAME}/{developer}/{model_path_name}/'
            f'{retrieved_timestamp}'
        ),
        retrieved_timestamp=retrieved_timestamp,
        source_metadata={
            'source_name': 'Artificial Analysis LLM API',
            'source_type': 'documentation',
            'source_organization_name': 'Artificial Analysis',
            'source_organization_url': ATTRIBUTION_URL,
            'evaluator_relationship': EvaluatorRelationship.third_party,
            'additional_details': make_source_metadata_details(payload),
        },
        eval_library=EvalLibrary(
            name='Artificial Analysis',
            version='unknown',
            additional_details={
                'api_reference_url': API_REFERENCE_URL,
            },
        ),
        model_info=model_info,
        evaluation_results=make_evaluation_results(
            model,
            payload,
            failures=failures,
            source_ref=source_ref,
        ),
    )

    return log, developer, model_path_name


def convert_models(
    models: list[dict[str, Any]],
    payload: dict[str, Any],
    output_dir: Path,
    retrieved_timestamp: str,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert usable models while retaining model and metric failures."""
    outputs = []
    failures = []
    for index, model in enumerate(models):
        source_ref = f'Artificial Analysis model row {index}'
        failure_count_before = len(failures)
        try:
            log, developer, model_path_name = make_log(
                model,
                payload,
                retrieved_timestamp,
                failures=failures,
                source_ref=source_ref,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=log,
                    base_dir=output_dir,
                    developer=developer,
                    model_name=model_path_name,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=(
                        f'no output written: {exc}'
                        if len(failures) > failure_count_before
                        else str(exc)
                    ),
                    source_record=model,
                )
            )
    return SourceConversionResult(
        source_name='Artificial Analysis',
        total_records=len(models),
        records=outputs,
        failures=failures,
    )


def run(args: argparse.Namespace) -> int:
    if args.save_raw_json is not None and is_subpath(
        args.save_raw_json, args.output_dir
    ):
        raise SystemExit(
            '--save-raw-json must point outside --output-dir, otherwise '
            'the validator will try to parse the raw API payload as an '
            'EvaluationLog.'
        )

    if args.input_json is not None:
        payload = load_payload(args.input_json)
    else:
        api_key = args.api_key or os.environ.get('ARTIFICIAL_ANALYSIS_API_KEY')
        if not api_key:
            raise SystemExit(
                'Missing API key. Set ARTIFICIAL_ANALYSIS_API_KEY or pass --api-key.'
            )
        payload = fetch_payload(api_key)

    maybe_save_raw_json(payload, args.save_raw_json)
    models = validate_payload(payload)
    retrieved_timestamp = str(time.time())

    result = convert_models(
        models,
        payload,
        args.output_dir,
        retrieved_timestamp,
    )
    paths = save_evaluation_logs(result.records)
    for output_path in paths:
        print(output_path)
    if result.failures:
        report_path = save_failure_report(
            result,
            default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} Artificial Analysis model logs.')
