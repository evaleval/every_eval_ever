#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    require_finite_number,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.fetch import fetch_text
from every_eval_ever.helpers.io import require_identity

SOURCE_URL = 'https://artificialanalysis.ai/api/v2/language/models/free'
LEADERBOARD_URL = 'https://artificialanalysis.ai/leaderboards/models'
CRITPT_V2_URL = 'https://artificialanalysis.ai/evaluations/critpt-v2'
# Model fields merged in from the website, keyed by the page they come from.
SITE_SECTIONS = {
    'site_scores': LEADERBOARD_URL,
    'site_critpt_v2': CRITPT_V2_URL,
}
MAX_API_PAGES = 100
API_REFERENCE_URL = 'https://artificialanalysis.ai/api-reference'
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
    max_score: float | None = None
    use_observed_max: bool = False
    include_prompt_options: bool = False


def _index_spec(key: str, label: str) -> MetricSpec:
    return MetricSpec(
        evaluation_name=f'artificial_analysis.{key}',
        source_section='evaluations',
        source_key=key,
        metric_name=f'Artificial Analysis {label}',
        evaluation_description=f'Artificial Analysis {label}.',
        metric_kind='index',
        metric_unit='points',
        lower_is_better=False,
        min_score=0.0,
        use_observed_max=True,
    )


def _site_score_spec(
    name: str, section: str, key: str, metric_name: str, description: str
) -> MetricSpec:
    return MetricSpec(
        evaluation_name=f'artificial_analysis.{name}',
        source_section=section,
        source_key=key,
        metric_name=metric_name,
        evaluation_description=description,
        metric_kind='benchmark_score',
        metric_unit='proportion',
        lower_is_better=False,
        min_score=0.0,
        max_score=1.0,
    )


def _speed_spec(
    name: str,
    section: str,
    key: str,
    metric_name: str,
    description: str,
    *,
    latency: bool,
) -> MetricSpec:
    return MetricSpec(
        evaluation_name=f'artificial_analysis.{name}',
        source_section=section,
        source_key=key,
        metric_name=metric_name,
        evaluation_description=description,
        metric_kind='latency' if latency else 'throughput',
        metric_unit='seconds' if latency else 'tokens_per_second',
        lower_is_better=latency,
        min_score=0.0,
        use_observed_max=True,
        include_prompt_options=True,
    )


METRIC_SPECS = [
    _index_spec('artificial_analysis_intelligence_index', 'Intelligence Index'),
    _index_spec(
        'artificial_analysis_finance_and_accounting_index',
        'Finance & Accounting Index',
    ),
    _index_spec(
        'artificial_analysis_strategy_and_ops_index', 'Strategy & Ops Index'
    ),
    _index_spec('artificial_analysis_legal_index', 'Legal Index'),
    _index_spec(
        'artificial_analysis_healthcare_and_medical_index',
        'Healthcare & Medical Index',
    ),
    _index_spec('artificial_analysis_engineering_index', 'Engineering Index'),
    _index_spec('artificial_analysis_economics_index', 'Economics Index'),
    _site_score_spec(
        'gpqa',
        'site_scores',
        'gpqa-diamond',
        'GPQA Diamond',
        'Benchmark score on GPQA Diamond.',
    ),
    _site_score_spec(
        'hle',
        'site_scores',
        'humanitys-last-exam',
        "Humanity's Last Exam",
        "Benchmark score on Humanity's Last Exam.",
    ),
    _site_score_spec(
        'livecodebench',
        'site_scores',
        'livecodebench',
        'LiveCodeBench',
        'Benchmark score on LiveCodeBench.',
    ),
    _site_score_spec(
        'scicode',
        'site_scores',
        'scicode',
        'SciCode',
        'Benchmark score on SciCode.',
    ),
    _site_score_spec(
        'aime_25',
        'site_scores',
        'aime-2025',
        'AIME 2025',
        'Benchmark score on AIME 2025.',
    ),
    _site_score_spec(
        'ifbench',
        'site_scores',
        'ifbench',
        'IFBench',
        'Benchmark score on IFBench.',
    ),
    _site_score_spec(
        'lcr',
        'site_scores',
        'artificial-analysis-long-context-reasoning',
        'AA-LCR',
        'Benchmark score on AA-LCR.',
    ),
    _site_score_spec(
        'terminalbench_hard',
        'site_scores',
        'terminalbench-hard',
        'Terminal-Bench Hard',
        'Benchmark score on Terminal-Bench Hard.',
    ),
    _site_score_spec(
        'tau2',
        'site_scores',
        'tau2-bench',
        'tau2-bench',
        'Benchmark score on tau2-bench.',
    ),
    _site_score_spec(
        'critpt',
        'site_scores',
        'critpt',
        'CritPt',
        "Artificial Analysis leaderboard 'critpt' score.",
    ),
    _site_score_spec(
        'critpt_v2',
        'site_critpt_v2',
        'critptAgentic',
        'CritPt v2',
        "CritPt v2 score: the 'critptAgentic' field of the Artificial "
        'Analysis CritPt v2 evaluation page.',
    ),
    _site_score_spec(
        'critpt_v2_hard',
        'site_scores',
        'critpt-v2-hard',
        'CritPt v2 (hard subset)',
        'CritPt v2 score on its hard subset.',
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
        use_observed_max=True,
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
        use_observed_max=True,
    ),
    _speed_spec(
        'median_output_tokens_per_second',
        'performance',
        'median_output_tokens_per_second',
        'Median output tokens per second',
        'Median output generation speed reported by Artificial Analysis.',
        latency=False,
    ),
    _speed_spec(
        'median_time_to_first_token_seconds',
        'performance',
        'median_time_to_first_token_seconds',
        'Median time to first token',
        'Median time to first token reported by Artificial Analysis.',
        latency=True,
    ),
    _speed_spec(
        'median_time_to_first_answer_token',
        'performance',
        'median_time_to_first_answer_token_seconds',
        'Median time to first answer token',
        'Median time to first answer token reported by Artificial Analysis.',
        latency=True,
    ),
    _speed_spec(
        'mean_output_tokens_per_second',
        'performance_v3',
        'mean_output_tokens_per_second',
        'Mean output tokens per second',
        'Performance v3 mean output generation speed.',
        latency=False,
    ),
    _speed_spec(
        'mean_time_to_first_token_seconds',
        'performance_v3',
        'mean_time_to_first_token_seconds',
        'Mean time to first token',
        'Performance v3 mean time to first token.',
        latency=True,
    ),
    _speed_spec(
        'mean_time_to_first_answer_token_seconds',
        'performance_v3',
        'mean_time_to_first_answer_token_seconds',
        'Mean time to first answer token',
        'Performance v3 mean time to first answer token.',
        latency=True,
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


def fetch_payload(api_key: str) -> dict[str, Any]:
    """Fetch every page of the free-tier language model list."""
    data: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for page in range(1, MAX_API_PAGES + 1):
        body = fetch_json(
            f'{SOURCE_URL}?page={page}',
            headers={'x-api-key': api_key},
        )
        if not isinstance(body, dict):
            raise ValueError('Artificial Analysis API page must be an object')
        first = first or body
        data.extend(validate_payload(body))
        pagination = body.get('pagination') or {}
        if not pagination.get('has_more'):
            break
    else:
        raise ValueError(
            f'Artificial Analysis API returned more than {MAX_API_PAGES} pages'
        )
    return {
        'tier': first.get('tier'),
        'intelligence_index_version': first.get('intelligence_index_version'),
        'pages_fetched': page,
        'data': data,
    }


def flight_text(page_html: str) -> str:
    """Return the concatenated Next.js flight data embedded in a page."""
    chunks = re.findall(
        r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', page_html
    )
    if not chunks:
        raise ValueError('page has no embedded Next.js flight data')
    return ''.join(json.loads(f'"{chunk}"') for chunk in chunks)


def parse_leaderboard_models(page_html: str) -> list[dict[str, Any]]:
    """Return the ``initialModels`` rows of the models leaderboard page."""
    text = flight_text(page_html)
    marker = '"initialModels":'
    start = text.find(marker)
    if start < 0:
        raise ValueError('leaderboard page has no initialModels list')
    rows, _ = json.JSONDecoder().raw_decode(text, start + len(marker))
    if not isinstance(rows, list) or not rows:
        raise ValueError('leaderboard initialModels list is empty')
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get('id'), str)
            or not isinstance(row.get('scores'), dict)
        ):
            raise ValueError('leaderboard model row lacks an id or scores')
    return rows


def parse_critpt_v2_models(page_html: str) -> dict[str, dict[str, Any]]:
    """Return ``{model id: {'critptAgentic': value}}`` from the CritPt v2 page."""
    text = flight_text(page_html)
    decoder = json.JSONDecoder()
    found: dict[str, dict[str, Any]] = {}
    for match in re.finditer(r'\{"id":"[0-9a-f-]{36}","slug":"', text):
        try:
            row, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(row, dict) and 'critptAgentic' in row:
            found.setdefault(row['id'], {'critptAgentic': row['critptAgentic']})
    if not found:
        raise ValueError('CritPt v2 page has no model rows with critptAgentic')
    return found


def fetch_site() -> dict[str, Any]:
    """Scrape the website's per-benchmark scores; page errors are returned."""
    site: dict[str, Any] = {'errors': []}
    for key, url, parse in (
        ('leaderboard_models', LEADERBOARD_URL, parse_leaderboard_models),
        ('critpt_v2', CRITPT_V2_URL, parse_critpt_v2_models),
    ):
        try:
            site[key] = parse(fetch_text(url))
        except Exception as exc:
            site['errors'].append({'url': url, 'reason': str(exc)})
    return site


def merge_site_scores(
    models: list[dict[str, Any]], site: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[SourceRecordExclusion]]:
    """Attach website scores to API models with the same model id."""
    leaderboard = {
        row['id']: row for row in site.get('leaderboard_models') or []
    }
    critpt_v2 = site.get('critpt_v2') or {}
    merged = []
    for model in models:
        extra: dict[str, Any] = {}
        row = leaderboard.get(model.get('id'))
        if row is not None:
            extra['site_scores'] = row['scores']
        if model.get('id') in critpt_v2:
            extra['site_critpt_v2'] = critpt_v2[model['id']]
        merged.append({**model, **extra})

    api_ids = {model.get('id') for model in models}
    exclusions = [
        SourceRecordExclusion(
            source_ref=f'{url} model {model_id}',
            reason='no API model has this id, so it has no model identity',
            source_record={'id': model_id, 'name': name},
        )
        for url, rows in (
            (
                LEADERBOARD_URL,
                {k: v.get('name') for k, v in leaderboard.items()},
            ),
            (CRITPT_V2_URL, {k: None for k in critpt_v2}),
        )
        for model_id, name in rows.items()
        if model_id not in api_ids
    ]
    return merged, exclusions


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


def get_metric_value(model: dict[str, Any], spec: MetricSpec) -> float | None:
    if spec.source_section == 'model':
        value = model.get(spec.source_key)
    else:
        container = model.get(spec.source_section, {})
        if not isinstance(container, dict):
            return None
        value = container.get(spec.source_key)

    if value is None:
        return None
    return require_finite_number(
        value, f'Artificial Analysis metric {spec.source_key}'
    )


def compute_observed_max_scores(
    models: list[dict[str, Any]],
) -> dict[str, float]:
    observed_max_scores: dict[str, float] = {}

    for spec in METRIC_SPECS:
        if not spec.use_observed_max:
            continue

        values = [
            value
            for model in models
            if (value := _valid_metric_value(model, spec)) is not None
        ]
        if values:
            observed_max_scores[spec.evaluation_name] = max(values)

    return observed_max_scores


def _valid_metric_value(
    model: dict[str, Any], spec: MetricSpec
) -> float | None:
    """Return a numeric value for aggregate bounds; row conversion reports errors."""
    try:
        return get_metric_value(model, spec)
    except (TypeError, ValueError):
        return None


def make_source_data(spec: MetricSpec) -> SourceDataUrl:
    page = SITE_SECTIONS.get(spec.source_section)
    if page is not None:
        return SourceDataUrl(
            dataset_name='Artificial Analysis website',
            source_type='url',
            url=[page],
        )
    return SourceDataUrl(
        dataset_name='Artificial Analysis LLM API',
        source_type='url',
        url=[SOURCE_URL],
    )


def make_source_metadata_details(
    payload: dict[str, Any],
) -> dict[str, str]:
    prompt_options = payload.get('prompt_options', {})
    details = {
        'api_endpoint': SOURCE_URL,
        'api_reference_url': API_REFERENCE_URL,
        'methodology_url': METHODOLOGY_URL,
        'attribution_url': ATTRIBUTION_URL,
        'attribution_required': 'true',
        'endpoint_scope': 'llms',
        'leaderboard_url': LEADERBOARD_URL,
        'critpt_v2_url': CRITPT_V2_URL,
    }
    for key in ('tier', 'intelligence_index_version'):
        if payload.get(key) is not None:
            details[f'api_{key}'] = stringify(payload[key])

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
        'raw_creator_slug': stringify(creator.get('slug')),
    }
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
    observed_max_scores: dict[str, float],
    prompt_options: dict[str, Any],
) -> MetricConfig:
    max_score = spec.max_score
    if max_score is None:
        max_score = observed_max_scores.get(spec.evaluation_name)
    if max_score is None:
        raise ValueError(
            f'No max_score available for metric {spec.evaluation_name!r}.'
        )

    additional_details = {
        'raw_metric_field': spec.source_key,
        'bound_strategy': (
            'observed_max_from_snapshot' if spec.use_observed_max else 'fixed'
        ),
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

    lower = model.get('tokens_per_second_q025')
    upper = model.get('tokens_per_second_q975')
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
) -> ScoreDetails:
    details: dict[str, str] = {
        'raw_model_id': stringify(model['id']),
    }

    if spec.source_section == 'model':
        details['raw_value_field'] = spec.source_key
    else:
        details['raw_value_field'] = f'{spec.source_section}.{spec.source_key}'

    uncertainty = maybe_make_uncertainty(model, spec)

    return ScoreDetails(
        score=score,
        details=details,
        uncertainty=uncertainty,
    )


def make_evaluation_results(
    model: dict[str, Any],
    payload: dict[str, Any],
    observed_max_scores: dict[str, float],
    *,
    failures: list[SourceRecordFailure] | None = None,
    source_ref: str | None = None,
) -> list[EvaluationResult]:
    prompt_options = payload.get('prompt_options', {})
    results: list[EvaluationResult] = []

    for spec in METRIC_SPECS:
        try:
            score = get_metric_value(model, spec)
            if score is None:
                continue
            results.append(
                EvaluationResult(
                    evaluation_result_id=spec.evaluation_name,
                    evaluation_name=spec.evaluation_name,
                    source_data=make_source_data(spec),
                    metric_config=make_metric_config(
                        spec, observed_max_scores, prompt_options
                    ),
                    score_details=make_score_details(model, spec, score),
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
    observed_max_scores: dict[str, float],
    retrieved_timestamp: str,
    *,
    failures: list[SourceRecordFailure] | None = None,
    source_ref: str | None = None,
) -> tuple[EvaluationLog, str, str]:
    model_info, developer, model_path_name = make_model_info(model)

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'{BENCHMARK_NAME}/{developer}/{model_path_name}/{retrieved_timestamp}'
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
            observed_max_scores,
            failures=failures,
            source_ref=source_ref,
        ),
    )

    return log, developer, model_path_name


def convert_models(
    models: list[dict[str, Any]],
    payload: dict[str, Any],
    output_dir: Path,
    observed_max_scores: dict[str, float],
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
                observed_max_scores,
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
        payload['site'] = fetch_site()

    maybe_save_raw_json(payload, args.save_raw_json)
    site = payload.get('site') or {}
    models, exclusions = merge_site_scores(validate_payload(payload), site)
    observed_max_scores = compute_observed_max_scores(models)
    retrieved_timestamp = str(time.time())

    result = convert_models(
        models,
        payload,
        args.output_dir,
        observed_max_scores,
        retrieved_timestamp,
    )
    result.exclusions.extend(exclusions)
    result.failures.extend(
        SourceRecordFailure(
            source_ref=error['url'],
            reason=f'website scores not read: {error["reason"]}',
        )
        for error in site.get('errors', [])
    )
    paths = save_evaluation_logs(result.records)
    for output_path in paths:
        print(output_path)
    if result.failures or result.exclusions:
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
