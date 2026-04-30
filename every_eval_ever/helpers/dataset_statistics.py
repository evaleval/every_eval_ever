"""Descriptive and uncertainty-aware summaries for Every Eval Ever."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SEP = '=' * 72
SUB = '-' * 72

REPO_ID = 'evaleval/EEE_datastore'
FOLDER_PATH = 'viewer_parquets'
HUGGING_FACE_DATASTORE = f'datasets/{REPO_ID}/{FOLDER_PATH}/**/*.parquet'

CONTINUOUS_SCORE_TYPE = 'continuous'
STABILIZATION_WEIGHT = 5.0
BOOTSTRAP_ITERATIONS = 400
RANDOM_SEED = 20260429
SCORE_GROUP_KEYS = (
    'benchmark',
    'evaluation_name',
    'metric_id',
    'metric_name',
    'metric_kind',
    'metric_unit',
)
METADATA_FIELD_CANDIDATES = (
    {'key': 'generation_config_present', 'label': 'generation config'},
    {'key': 'generation_temperature', 'label': 'temperature'},
    {'key': 'generation_max_tokens', 'label': 'max tokens'},
    {'key': 'generation_agentic_config_present', 'label': 'agentic config'},
    {'key': 'inference_engine', 'label': 'runtime/platform'},
    {'key': 'source_locator', 'label': 'source URL / HF repo'},
    {'key': 'source_organization_url', 'label': 'source org URL'},
    {'key': 'evaluator_relationship', 'label': 'evaluator relationship'},
    {'key': 'detailed_results_file', 'label': 'detailed results'},
    {'key': 'has_uncertainty', 'label': 'uncertainty'},
    {'key': 'uncertainty_num_samples', 'label': 'sample count'},
    {'key': 'metric_id', 'label': 'metric ID'},
    {'key': 'metric_kind', 'label': 'metric kind'},
    {'key': 'metric_unit', 'label': 'metric unit'},
    {'key': 'model_parameters', 'label': 'model parameters'},
    {'key': 'model_license', 'label': 'model license'},
)


def read_data(datastore: str) -> list[str]:
    from huggingface_hub import HfFileSystem

    hffs = HfFileSystem()
    files = hffs.glob(datastore)
    return [f'hf://{f}' for f in files if f.endswith('dataset.parquet')]


def load_schema_table(con: Any, table: str) -> None:
    schema_urls = read_data(HUGGING_FACE_DATASTORE)
    if not schema_urls:
        raise RuntimeError('No schema parquet files found')
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_parquet(?, union_by_name=true, filename=true)
        """,
        [schema_urls],
    )


def extract_result_rows(con: Any, schema_table: str) -> list[dict[str, Any]]:
    rows = con.execute(
        f"""
        SELECT
            schema_version,
            evaluation_id,
            model_info.id AS model_id,
            model_info.developer AS model_developer,
            model_info.inference_platform AS inference_engine,
            er.evaluation_name AS evaluation_name,
            er.source_data.dataset_name AS benchmark,
            er.metric_config.score_type AS score_type,
            er.metric_config.lower_is_better AS lower_is_better,
            TRY_CAST(er.metric_config.min_score AS DOUBLE) AS min_score,
            TRY_CAST(er.metric_config.max_score AS DOUBLE) AS max_score,
            TRY_CAST(er.score_details.score AS DOUBLE) AS score,
            er.score_details.uncertainty IS NOT NULL AS has_uncertainty,
            er.metric_config.metric_id AS metric_id,
            er.metric_config.metric_name AS metric_name,
            er.metric_config.metric_kind AS metric_kind,
            er.metric_config.metric_unit AS metric_unit,
            source_metadata.source_organization_name AS source_organization,
            er.generation_config IS NOT NULL AS generation_config_present,
            TRY_CAST(
                er.generation_config.generation_args.temperature AS DOUBLE
            ) AS generation_temperature,
            TRY_CAST(
                er.generation_config.generation_args.max_tokens AS BIGINT
            ) AS generation_max_tokens,
            er.generation_config.generation_args.agentic_eval_config IS NOT NULL
                AS generation_agentic_config_present,
            er.source_data.source_type AS source_data_type,
            er.source_data.hf_repo AS source_hf_repo,
            er.source_data.url AS source_urls,
            source_metadata.source_organization_url
                AS source_organization_url,
            source_metadata.evaluator_relationship AS evaluator_relationship,
            detailed_evaluation_results.file_path AS detailed_results_file,
            TRY_CAST(
                er.score_details.uncertainty.num_samples AS BIGINT
            ) AS uncertainty_num_samples,
            to_json(model_info.additional_details)
                AS model_additional_details_json
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        """
    ).fetchall()
    columns = [
        'schema_version',
        'evaluation_id',
        'model_id',
        'model_developer',
        'inference_engine',
        'evaluation_name',
        'benchmark',
        'score_type',
        'lower_is_better',
        'min_score',
        'max_score',
        'score',
        'has_uncertainty',
        'metric_id',
        'metric_name',
        'metric_kind',
        'metric_unit',
        'source_organization',
        'generation_config_present',
        'generation_temperature',
        'generation_max_tokens',
        'generation_agentic_config_present',
        'source_data_type',
        'source_hf_repo',
        'source_urls',
        'source_organization_url',
        'evaluator_relationship',
        'detailed_results_file',
        'uncertainty_num_samples',
        'model_additional_details_json',
    ]
    extracted = []
    for row in rows:
        item = dict(zip(columns, row))
        source_urls = item.get('source_urls')
        item['source_locator'] = item.get('source_hf_repo') or source_urls
        model_details = parse_json_mapping(
            item.pop('model_additional_details_json', None)
        )
        item['model_parameters'] = (
            model_details.get('params_billions')
            or model_details.get('parameters')
            or model_details.get('parameter_count')
        )
        item['model_license'] = model_details.get('license')
        extracted.append(item)
    return extracted


def parse_json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_score(
    score: float,
    min_score: float,
    max_score: float,
    lower_is_better: bool,
) -> float:
    normalized = (score - min_score) / (max_score - min_score)
    if lower_is_better:
        normalized = 1.0 - normalized
    return normalized


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {
            'count': 0,
            'min': None,
            'median': None,
            'mean': None,
            'max': None,
            'stddev': None,
        }
    return {
        'count': len(vals),
        'min': min(vals),
        'median': statistics.median(vals),
        'mean': statistics.mean(vals),
        'max': max(vals),
        'stddev': statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def shared_evaluation_key(row: dict[str, Any]) -> str:
    parts = [
        *(row.get(key) for key in SCORE_GROUP_KEYS),
        row.get('score_type'),
        row.get('min_score'),
        row.get('max_score'),
        bool(row.get('lower_is_better')),
    ]
    return json.dumps(parts, sort_keys=False, separators=(',', ':'))


def quality_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        'total_result_rows': len(rows),
        'missing_score': 0,
        'missing_bounds': 0,
        'zero_width_bounds': 0,
        'incompatible_score_type': 0,
        'out_of_range': 0,
        'missing_metadata': 0,
        'has_uncertainty': 0,
    }
    for row in rows:
        score = row.get('score')
        min_score = row.get('min_score')
        max_score = row.get('max_score')
        if score is None:
            counts['missing_score'] += 1
        if min_score is None or max_score is None:
            counts['missing_bounds'] += 1
        elif min_score == max_score:
            counts['zero_width_bounds'] += 1
        elif score is not None and not min_score <= score <= max_score:
            counts['out_of_range'] += 1
        if row.get('score_type') != CONTINUOUS_SCORE_TYPE:
            counts['incompatible_score_type'] += 1
        if not row.get('model_id') or not row.get('benchmark'):
            counts['missing_metadata'] += 1
        if row.get('has_uncertainty'):
            counts['has_uncertainty'] += 1
    return counts


def valid_normalized_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    valid = []
    exclusions = {
        'missing_score': 0,
        'missing_bounds': 0,
        'zero_width_bounds': 0,
        'incompatible_score_type': 0,
        'out_of_range': 0,
    }
    for row in rows:
        score = row.get('score')
        min_score = row.get('min_score')
        max_score = row.get('max_score')
        if score is None:
            exclusions['missing_score'] += 1
            continue
        if min_score is None or max_score is None:
            exclusions['missing_bounds'] += 1
            continue
        if min_score == max_score:
            exclusions['zero_width_bounds'] += 1
            continue
        if row.get('score_type') != CONTINUOUS_SCORE_TYPE:
            exclusions['incompatible_score_type'] += 1
            continue
        if not min_score <= score <= max_score:
            exclusions['out_of_range'] += 1
            continue
        normalized = normalize_score(
            float(score),
            float(min_score),
            float(max_score),
            bool(row.get('lower_is_better')),
        )
        valid_row = dict(row)
        valid_row['normalized_score'] = normalized
        valid_row['shared_evaluation_key'] = shared_evaluation_key(row)
        valid.append(valid_row)
    return valid, exclusions


def distinct_count(rows: list[dict[str, Any]], key: str) -> int:
    return len({row.get(key) for row in rows if row.get(key) is not None})


def count_values(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, int | str]]:
    counts = Counter(
        str(row.get(key)) for row in rows if row.get(key) is not None
    )
    return [
        {'value': value, 'count': count}
        for value, count in counts.most_common()
    ]


def count_values_with_unknown(
    rows: list[dict[str, Any]], key: str, unknown: str = 'unknown'
) -> list[dict[str, int | str]]:
    counts = Counter()
    for row in rows:
        value = row.get(key)
        normalized = unknown if value is None else str(value).strip()
        counts[normalized or unknown] += 1
    return [
        {'value': value, 'count': count}
        for value, count in counts.most_common()
    ]


def models_per_benchmark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmark = row.get('benchmark')
        if benchmark is not None:
            grouped[str(benchmark)].append(row)

    summaries = []
    for benchmark, items in grouped.items():
        summaries.append(
            {
                'benchmark': benchmark,
                'unique_models': distinct_count(items, 'model_id'),
                'result_rows': len(items),
            }
        )
    summaries.sort(
        key=lambda item: (
            -int(item['unique_models']),
            -int(item['result_rows']),
            str(item['benchmark']),
        )
    )
    return summaries


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def benchmark_name(row: dict[str, Any]) -> str:
    value = row.get('benchmark')
    if value is None:
        return 'unknown'
    text = str(value).strip()
    return text or 'unknown'


def format_benchmark_label(benchmark: str, result_rows: int) -> str:
    return f'{benchmark} (n={result_rows:,})'


def field_present_rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(has_value(row.get(field)) for row in rows) / len(rows)


def metadata_completeness(
    rows: list[dict[str, Any]],
    top_benchmarks: int = 12,
    top_fields: int = 12,
) -> dict[str, Any]:
    candidate_fields = [
        field
        for field in METADATA_FIELD_CANDIDATES
        if any(field['key'] in row for row in rows)
    ]
    if not rows or not candidate_fields:
        return {
            'fields': [],
            'benchmarks': [],
            'matrix': [],
            'top_benchmark_count': top_benchmarks,
            'other_result_rows': 0,
        }

    rows_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_benchmark[benchmark_name(row)].append(row)

    field_summaries = []
    for field in candidate_fields:
        key = str(field['key'])
        benchmark_rates = [
            field_present_rate(items, key)
            for items in rows_by_benchmark.values()
        ]
        present_rate = field_present_rate(rows, key)
        missing_rate = 1.0 - present_rate
        benchmark_stddev = (
            statistics.pstdev(benchmark_rates)
            if len(benchmark_rates) > 1
            else 0.0
        )
        selection_score = missing_rate * max(benchmark_stddev, 0.05)
        field_summaries.append(
            {
                'key': key,
                'label': str(field['label']),
                'missing_rate': missing_rate,
                'benchmark_stddev': benchmark_stddev,
                'selection_score': selection_score,
            }
        )
    field_summaries.sort(
        key=lambda item: (
            -float(item['selection_score']),
            -float(item['missing_rate']),
            str(item['label']),
        )
    )
    selected_fields = field_summaries[:top_fields]

    top_benchmark_names = [
        benchmark
        for benchmark, _ in sorted(
            (
                (benchmark, len(items))
                for benchmark, items in rows_by_benchmark.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_benchmarks]
    ]
    selected_field_keys = [field['key'] for field in selected_fields]

    benchmark_summaries = []
    benchmark_groups: dict[str, list[dict[str, Any]]] = {}
    for benchmark in top_benchmark_names:
        items = rows_by_benchmark[benchmark]
        benchmark_groups[benchmark] = items
        benchmark_summaries.append(
            {
                'benchmark': benchmark,
                'label': format_benchmark_label(benchmark, len(items)),
                'result_rows': len(items),
                'overall_completeness': average_completeness(
                    items, selected_field_keys
                ),
            }
        )

    other_rows = [
        row
        for benchmark, items in rows_by_benchmark.items()
        if benchmark not in top_benchmark_names
        for row in items
    ]
    if other_rows:
        benchmark_groups['Other'] = other_rows
        benchmark_summaries.append(
            {
                'benchmark': 'Other',
                'label': format_benchmark_label('Other', len(other_rows)),
                'result_rows': len(other_rows),
                'overall_completeness': average_completeness(
                    other_rows, selected_field_keys
                ),
            }
        )

    benchmark_summaries.sort(
        key=lambda item: (
            item['benchmark'] == 'Other',
            float(item['overall_completeness']),
            str(item['benchmark']),
        )
    )

    matrix = []
    selected_fields_by_key = {
        str(field['key']): field for field in selected_fields
    }
    for benchmark_summary in benchmark_summaries:
        benchmark = str(benchmark_summary['benchmark'])
        items = benchmark_groups[benchmark]
        for field_key in selected_field_keys:
            present_rate = field_present_rate(items, str(field_key))
            field = selected_fields_by_key[str(field_key)]
            matrix.append(
                {
                    'benchmark': benchmark,
                    'benchmark_label': benchmark_summary['label'],
                    'field': str(field_key),
                    'field_label': field['label'],
                    'present_rate': present_rate,
                    'missing_rate': 1.0 - present_rate,
                    'result_rows': len(items),
                }
            )

    return {
        'fields': selected_fields,
        'benchmarks': benchmark_summaries,
        'matrix': matrix,
        'top_benchmark_count': top_benchmarks,
        'other_result_rows': len(other_rows),
    }


def average_completeness(
    rows: list[dict[str, Any]], fields: list[str]
) -> float:
    if not rows or not fields:
        return 0.0
    present = sum(
        has_value(row.get(field)) for row in rows for field in fields
    )
    return present / (len(rows) * len(fields))


def grouped_summaries(
    rows: list[dict[str, Any]],
    value_key: str,
    group_keys: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        grouped[tuple(row.get(key) for key in group_keys)].append(float(value))

    summaries = []
    for group, values in grouped.items():
        item = {key: group[index] for index, key in enumerate(group_keys)}
        item.update(numeric_summary(values))
        summaries.append(item)
    summaries.sort(key=lambda item: (-int(item['count']), str(item)))
    return summaries[:limit]


def bootstrap_interval_and_support(
    values: list[float],
    threshold: float,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[list[float | None], float | None]:
    if not values:
        return [None, None], None
    rng = random.Random(RANDOM_SEED + len(values))
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.mean(sample))
    return [
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
    ], sum(estimate > threshold for estimate in estimates) / len(estimates)


def stabilized_estimate(
    mean_score: float,
    count: int,
    corpus_mean: float,
    weight: float = STABILIZATION_WEIGHT,
) -> float:
    return (count * mean_score + weight * corpus_mean) / (count + weight)


def coverage_aware_model_summaries(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if not rows:
        return []
    corpus_mean = statistics.mean(row['normalized_score'] for row in rows)
    key_means = {
        key: statistics.mean(item['normalized_score'] for item in items)
        for key, items in _group_rows(rows, 'shared_evaluation_key').items()
    }

    summaries = []
    for model_id, items in _group_rows(rows, 'model_id').items():
        scores = [item['normalized_score'] for item in items]
        centered_scores = [
            item['normalized_score']
            - key_means[item['shared_evaluation_key']]
            + corpus_mean
            for item in items
        ]
        raw_mean = statistics.mean(scores)
        centered_mean = statistics.mean(centered_scores)
        stabilized = stabilized_estimate(raw_mean, len(scores), corpus_mean)
        rng = random.Random(RANDOM_SEED + len(scores))
        bootstrap_scores = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = [scores[rng.randrange(len(scores))] for _ in scores]
            bootstrap_scores.append(
                stabilized_estimate(
                    statistics.mean(sample), len(sample), corpus_mean
                )
            )
        interval = [
            percentile(bootstrap_scores, 0.025),
            percentile(bootstrap_scores, 0.975),
        ]
        support = sum(score > corpus_mean for score in bootstrap_scores) / len(
            bootstrap_scores
        )
        summaries.append(
            {
                'model_id': model_id,
                'result_count': len(items),
                'benchmark_count': distinct_count(items, 'benchmark'),
                'evaluation_count': distinct_count(
                    items, 'shared_evaluation_key'
                ),
                'mean_normalized_score': raw_mean,
                'benchmark_centered_score': centered_mean,
                'stabilized_score': stabilized,
                'uncertainty_interval': interval,
                'support_above_corpus_average': support,
            }
        )
    summaries.sort(
        key=lambda item: (
            -float(item['stabilized_score']),
            -int(item['evaluation_count']),
            str(item['model_id']),
        )
    )
    return summaries[:limit]


def pairwise_model_comparisons(
    rows: list[dict[str, Any]],
    min_shared_evals: int,
    top_model_limit: int,
    comparison_limit: int,
) -> list[dict[str, Any]]:
    by_model_key: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    model_counts = Counter(
        row['model_id'] for row in rows if row.get('model_id')
    )
    top_models = {
        model
        for model, _ in model_counts.most_common(top_model_limit)
        if model is not None
    }
    for row in rows:
        model_id = row.get('model_id')
        if model_id not in top_models:
            continue
        by_model_key[model_id][row['shared_evaluation_key']].append(
            row['normalized_score']
        )

    model_scores = {
        model: {
            key: statistics.mean(values)
            for key, values in scores_by_key.items()
        }
        for model, scores_by_key in by_model_key.items()
    }
    models = sorted(model_scores)
    comparisons = []
    rng = random.Random(RANDOM_SEED)
    for index, model_a in enumerate(models):
        for model_b in models[index + 1 :]:
            shared_keys = sorted(
                set(model_scores[model_a]) & set(model_scores[model_b])
            )
            if len(shared_keys) < min_shared_evals:
                continue
            diffs = [
                model_scores[model_a][key] - model_scores[model_b][key]
                for key in shared_keys
            ]
            boot_means = []
            for _ in range(BOOTSTRAP_ITERATIONS):
                sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
                boot_means.append(statistics.mean(sample))
            comparisons.append(
                {
                    'model_a': model_a,
                    'model_b': model_b,
                    'shared_evaluation_count': len(shared_keys),
                    'mean_paired_difference': statistics.mean(diffs),
                    'uncertainty_interval': [
                        percentile(boot_means, 0.025),
                        percentile(boot_means, 0.975),
                    ],
                    'support_model_a_higher': sum(
                        value > 0 for value in boot_means
                    )
                    / len(boot_means),
                }
            )
    comparisons.sort(
        key=lambda item: (
            -int(item['shared_evaluation_count']),
            -abs(float(item['mean_paired_difference'])),
            str(item['model_a']),
            str(item['model_b']),
        )
    )
    return comparisons[:comparison_limit]


def descriptive_statistics(
    rows: list[dict[str, Any]], summary_limit: int
) -> dict[str, Any]:
    valid_rows, exclusions = valid_normalized_rows(rows)
    return {
        'counts': {
            'result_rows': len(rows),
            'unique_models': distinct_count(rows, 'model_id'),
            'unique_developers': distinct_count(rows, 'model_developer'),
            'unique_benchmarks': distinct_count(rows, 'benchmark'),
            'unique_evaluations': distinct_count(rows, 'evaluation_name'),
        },
        'schema_versions': count_values(rows, 'schema_version'),
        'inference_engines': count_values_with_unknown(
            rows, 'inference_engine'
        ),
        'models_per_benchmark': models_per_benchmark(rows),
        'metadata_completeness': metadata_completeness(rows),
        'quality': quality_counts(rows),
        'normalization_exclusions': exclusions,
        'score_summaries': grouped_summaries(
            rows,
            'score',
            SCORE_GROUP_KEYS,
            summary_limit,
        ),
        'normalized_score_summaries': grouped_summaries(
            valid_rows,
            'normalized_score',
            SCORE_GROUP_KEYS,
            summary_limit,
        ),
    }


def build_statistics_report(
    rows: list[dict[str, Any]],
    summary_limit: int,
    comparison_limit: int,
    top_model_limit: int,
    min_shared_evals: int,
    descriptive_only: bool,
) -> dict[str, Any]:
    valid_rows, exclusions = valid_normalized_rows(rows)
    report = {
        'descriptive': descriptive_statistics(rows, summary_limit),
        'observational': {
            'valid_normalized_rows': len(valid_rows),
            'exclusions': exclusions,
        },
    }
    if descriptive_only:
        return report
    report['observational'].update(
        {
            'coverage_aware_model_summaries': coverage_aware_model_summaries(
                valid_rows, top_model_limit
            ),
            'pairwise_model_comparisons': pairwise_model_comparisons(
                valid_rows,
                min_shared_evals,
                top_model_limit,
                comparison_limit,
            ),
        }
    )
    return report


def _group_rows(
    rows: list[dict[str, Any]], key: str
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    return grouped


def section(title: str) -> None:
    print(f'\n{SEP}')
    print(f'  {title.upper()}')
    print(SUB)


def print_table(items: list[dict[str, Any]], columns: list[str]) -> None:
    for item in items:
        parts = []
        for column in columns:
            value = item.get(column)
            if isinstance(value, float):
                value = f'{value:.4f}'
            parts.append(f'{column}={value}')
        print('  ' + '  '.join(parts))


def print_report(report: dict[str, Any], descriptive_only: bool) -> None:
    descriptive = report['descriptive']
    section('dataset counts')
    for key, value in descriptive['counts'].items():
        print(f'  {key:<32} {value:>10,}')

    section('quality diagnostics')
    for key, value in descriptive['quality'].items():
        print(f'  {key:<32} {value:>10,}')

    section('normalization exclusions')
    for key, value in report['observational']['exclusions'].items():
        print(f'  {key:<32} {value:>10,}')

    section('inference engines')
    print_table(
        descriptive['inference_engines'][:10],
        ['value', 'count'],
    )

    section('models per benchmark')
    print_table(
        descriptive['models_per_benchmark'][:10],
        ['benchmark', 'unique_models', 'result_rows'],
    )

    section('score summaries')
    print_table(
        descriptive['score_summaries'],
        [
            'benchmark',
            'evaluation_name',
            'metric_id',
            'count',
            'mean',
            'median',
            'stddev',
        ],
    )

    section('normalized score summaries')
    print_table(
        descriptive['normalized_score_summaries'],
        [
            'benchmark',
            'evaluation_name',
            'metric_id',
            'count',
            'mean',
            'median',
            'stddev',
        ],
    )

    if descriptive_only:
        return

    section('coverage-aware model summaries')
    print_table(
        report['observational']['coverage_aware_model_summaries'],
        [
            'model_id',
            'evaluation_count',
            'stabilized_score',
            'benchmark_centered_score',
            'support_above_corpus_average',
        ],
    )

    section('pairwise model comparisons')
    print_table(
        report['observational']['pairwise_model_comparisons'],
        [
            'model_a',
            'model_b',
            'shared_evaluation_count',
            'mean_paired_difference',
            'support_model_a_higher',
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate Every Eval Ever dataset statistics.'
    )
    parser.add_argument(
        '--table', default='eee', help='Table name for the in-memory database'
    )
    parser.add_argument(
        '--stats-output',
        type=Path,
        help='Optional JSON output path for the statistics report',
    )
    parser.add_argument(
        '--summary-limit',
        default=10,
        type=int,
        help='Number of descriptive summary rows to print',
    )
    parser.add_argument(
        '--comparison-limit',
        default=50,
        type=int,
        help='Number of pairwise comparison rows to print',
    )
    parser.add_argument(
        '--top-model-limit',
        default=50,
        type=int,
        help='Number of most-covered models to include in comparisons',
    )
    parser.add_argument(
        '--min-shared-evals',
        default=5,
        type=int,
        help='Minimum shared evaluation keys for pairwise comparisons',
    )
    parser.add_argument(
        '--descriptive-only',
        action='store_true',
        help='Skip observational comparison summaries',
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.table):
        parser.error('--table must be a valid SQL identifier')
    if args.summary_limit < 1:
        parser.error('--summary-limit must be at least 1')
    if args.comparison_limit < 1:
        parser.error('--comparison-limit must be at least 1')
    if args.top_model_limit < 1:
        parser.error('--top-model-limit must be at least 1')
    if args.min_shared_evals < 1:
        parser.error('--min-shared-evals must be at least 1')
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import duckdb

    schema_table = f'{args.table}_schema'
    with duckdb.connect(':memory:') as con:
        try:
            con.execute('LOAD httpfs;')
        except duckdb.Error:
            con.execute('INSTALL httpfs;')
            con.execute('LOAD httpfs;')

        load_schema_table(con, schema_table)
        rows = extract_result_rows(con, schema_table)

    report = build_statistics_report(
        rows,
        summary_limit=args.summary_limit,
        comparison_limit=args.comparison_limit,
        top_model_limit=args.top_model_limit,
        min_shared_evals=args.min_shared_evals,
        descriptive_only=args.descriptive_only,
    )
    print_report(report, args.descriptive_only)

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(f'\nWrote statistics JSON to {args.stats_output}')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
