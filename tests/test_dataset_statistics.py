from __future__ import annotations

from every_eval_ever.helpers import dataset_statistics as stats


def row(
    model_id: str,
    benchmark: str,
    evaluation_name: str,
    score: float | None,
    *,
    min_score: float | None = 0.0,
    max_score: float | None = 1.0,
    lower_is_better: bool = False,
    score_type: str | None = 'continuous',
    inference_engine: str | None = None,
    metric_id: str | None = 'score',
    metric_name: str | None = 'Score',
    metric_kind: str | None = 'accuracy',
    metric_unit: str | None = 'proportion',
    **metadata,
) -> dict:
    result = {
        'schema_version': '0.2.2',
        'evaluation_id': f'{model_id}/{benchmark}/{evaluation_name}',
        'model_id': model_id,
        'model_developer': model_id.split('/')[0],
        'inference_engine': inference_engine,
        'benchmark': benchmark,
        'evaluation_name': evaluation_name,
        'score': score,
        'min_score': min_score,
        'max_score': max_score,
        'lower_is_better': lower_is_better,
        'score_type': score_type,
        'has_uncertainty': False,
        'metric_id': metric_id,
        'metric_name': metric_name,
        'metric_kind': metric_kind,
        'metric_unit': metric_unit,
    }
    result.update(metadata)
    return result


def test_normalization_respects_lower_is_better():
    assert stats.normalize_score(0.8, 0.0, 1.0, False) == 0.8
    assert stats.normalize_score(0.2, 0.0, 1.0, True) == 0.8


def test_invalid_rows_are_excluded_and_counted():
    rows = [
        row('a', 'bench', 'eval', None),
        row('a', 'bench', 'eval', 0.5, min_score=None),
        row('a', 'bench', 'eval', 0.5, min_score=1.0, max_score=1.0),
        row('a', 'bench', 'eval', 0.5, score_type='binary'),
        row('a', 'bench', 'eval', 2.0),
        row('a', 'bench', 'eval', 0.8),
    ]

    valid, exclusions = stats.valid_normalized_rows(rows)

    assert len(valid) == 1
    assert exclusions == {
        'missing_score': 1,
        'missing_bounds': 1,
        'zero_width_bounds': 1,
        'incompatible_score_type': 1,
        'out_of_range': 1,
    }


def test_shared_evaluation_key_includes_metric_scale_and_direction():
    base = row('a', 'bench', 'eval', 0.8)
    different_metric = row(
        'a',
        'bench',
        'eval',
        0.7,
        metric_id='cost_per_task',
        metric_name='Cost per task',
        metric_kind='cost',
        metric_unit='usd',
    )
    different_scale = row('a', 'bench', 'eval', 80.0, max_score=100.0)
    different_direction = row('a', 'bench', 'eval', 0.2, lower_is_better=True)

    assert stats.shared_evaluation_key(base) != stats.shared_evaluation_key(
        different_metric
    )
    assert stats.shared_evaluation_key(base) != stats.shared_evaluation_key(
        different_scale
    )
    assert stats.shared_evaluation_key(base) != stats.shared_evaluation_key(
        different_direction
    )


def test_min_shared_evals_filters_pairwise_comparisons():
    rows = [
        row('model/a', 'b1', 'e1', 0.9),
        row('model/b', 'b1', 'e1', 0.7),
        row('model/a', 'b2', 'e2', 0.8),
        row('model/b', 'b2', 'e2', 0.6),
    ]
    valid, _ = stats.valid_normalized_rows(rows)

    assert (
        stats.pairwise_model_comparisons(
            valid, min_shared_evals=3, top_model_limit=10, comparison_limit=10
        )
        == []
    )
    comparisons = stats.pairwise_model_comparisons(
        valid, min_shared_evals=2, top_model_limit=10, comparison_limit=10
    )
    assert len(comparisons) == 1
    assert comparisons[0]['shared_evaluation_count'] == 2


def test_stabilized_estimates_move_sparse_models_toward_corpus_mean():
    estimate = stats.stabilized_estimate(
        mean_score=1.0, count=1, corpus_mean=0.5, weight=5.0
    )

    assert 0.5 < estimate < 1.0


def test_probability_style_support_outputs_are_bounded():
    rows = [
        row('model/a', 'b1', 'e1', 0.9),
        row('model/b', 'b1', 'e1', 0.7),
        row('model/a', 'b2', 'e2', 0.8),
        row('model/b', 'b2', 'e2', 0.6),
        row('model/a', 'b3', 'e3', 0.7),
        row('model/b', 'b3', 'e3', 0.6),
    ]
    valid, _ = stats.valid_normalized_rows(rows)

    model_summaries = stats.coverage_aware_model_summaries(valid, limit=10)
    comparisons = stats.pairwise_model_comparisons(
        valid, min_shared_evals=2, top_model_limit=10, comparison_limit=10
    )

    for summary in model_summaries:
        support = summary['support_above_corpus_average']
        assert 0.0 <= support <= 1.0
    assert 0.0 <= comparisons[0]['support_model_a_higher'] <= 1.0


def test_json_report_shape():
    rows = [
        row('model/a', 'b1', 'e1', 0.9),
        row('model/b', 'b1', 'e1', 0.7),
    ]

    report = stats.build_statistics_report(
        rows,
        summary_limit=5,
        comparison_limit=5,
        top_model_limit=5,
        min_shared_evals=1,
        descriptive_only=False,
    )

    assert set(report) == {'descriptive', 'observational'}
    assert report['descriptive']['counts']['result_rows'] == 2
    assert 'inference_engines' in report['descriptive']
    assert 'models_per_benchmark' in report['descriptive']
    assert 'metadata_completeness' in report['descriptive']
    assert 'metric_id' in report['descriptive']['score_summaries'][0]
    assert 'coverage_aware_model_summaries' in report['observational']
    assert 'pairwise_model_comparisons' in report['observational']


def test_score_summaries_group_by_metric_identity():
    rows = [
        row('model/a', 'arc', 'v1_Semi_Private', 0.98),
        row(
            'model/a',
            'arc',
            'v1_Semi_Private',
            17.0,
            max_score=77.2,
            lower_is_better=True,
            metric_id='cost_per_task',
            metric_name='Cost per task',
            metric_kind='cost',
            metric_unit='usd',
        ),
    ]

    report = stats.build_statistics_report(
        rows,
        summary_limit=10,
        comparison_limit=5,
        top_model_limit=5,
        min_shared_evals=1,
        descriptive_only=True,
    )

    raw_summaries = report['descriptive']['score_summaries']
    normalized_summaries = report['descriptive']['normalized_score_summaries']

    assert {item['metric_id'] for item in raw_summaries} == {
        'score',
        'cost_per_task',
    }
    assert {item['count'] for item in raw_summaries} == {1}
    assert {item['metric_id'] for item in normalized_summaries} == {
        'score',
        'cost_per_task',
    }


def test_models_per_benchmark_dedupes_model_counts():
    rows = [
        row('model/a', 'bench-one', 'eval-a', 0.9),
        row('model/a', 'bench-one', 'eval-b', 0.8),
        row('model/b', 'bench-one', 'eval-a', 0.7),
        row('model/c', 'bench-two', 'eval-a', 0.6),
    ]

    summaries = stats.models_per_benchmark(rows)

    assert summaries == [
        {
            'benchmark': 'bench-one',
            'unique_models': 2,
            'result_rows': 3,
        },
        {
            'benchmark': 'bench-two',
            'unique_models': 1,
            'result_rows': 1,
        },
    ]


def test_inference_engine_counts_group_missing_as_unknown():
    rows = [
        row('model/a', 'bench', 'eval', 0.9, inference_engine='vllm'),
        row('model/b', 'bench', 'eval', 0.8, inference_engine=''),
        row('model/c', 'bench', 'eval', 0.7, inference_engine=None),
        row('model/d', 'bench', 'eval', 0.6, inference_engine='ollama'),
        row('model/e', 'bench', 'eval', 0.5, inference_engine='vllm'),
    ]

    assert stats.count_values_with_unknown(rows, 'inference_engine') == [
        {'value': 'vllm', 'count': 2},
        {'value': 'unknown', 'count': 2},
        {'value': 'ollama', 'count': 1},
    ]


def test_metadata_completeness_counts_present_and_missing_values():
    rows = [
        row(
            'model/a',
            'bench-a',
            'eval',
            0.9,
            inference_engine='vllm',
            generation_temperature=0.2,
            source_locator='https://example.test/data',
            has_uncertainty=True,
        ),
        row(
            'model/b',
            'bench-a',
            'eval',
            0.8,
            inference_engine='  ',
            generation_temperature=None,
            source_locator=[],
            has_uncertainty=False,
        ),
    ]

    summary = stats.metadata_completeness(
        rows, top_benchmarks=1, top_fields=16
    )
    matrix = {
        (item['benchmark'], item['field']): item
        for item in summary['matrix']
    }

    assert matrix[('bench-a', 'inference_engine')]['present_rate'] == 0.5
    assert matrix[('bench-a', 'generation_temperature')][
        'present_rate'
    ] == 0.5
    assert matrix[('bench-a', 'source_locator')]['present_rate'] == 0.5
    assert matrix[('bench-a', 'has_uncertainty')]['present_rate'] == 0.5


def test_metadata_completeness_aggregates_other_benchmarks():
    rows = [
        row('model/a', 'bench-a', 'eval', 0.9, generation_temperature=0.1),
        row('model/b', 'bench-a', 'eval', 0.8, generation_temperature=0.2),
        row('model/c', 'bench-b', 'eval', 0.7, generation_temperature=None),
        row('model/d', 'bench-b', 'eval', 0.6, generation_temperature=None),
        row('model/e', 'bench-c', 'eval', 0.5, generation_temperature=0.3),
    ]

    summary = stats.metadata_completeness(
        rows, top_benchmarks=2, top_fields=16
    )

    assert summary['other_result_rows'] == 1
    assert summary['benchmarks'][-1]['benchmark'] == 'Other'
    assert summary['benchmarks'][-1]['label'] == 'Other (n=1)'
    assert summary['benchmarks'][-1]['result_rows'] == 1
    assert any(
        item['benchmark'] == 'Other'
        and item['field'] == 'generation_temperature'
        and item['present_rate'] == 1.0
        for item in summary['matrix']
    )


def test_metadata_field_selection_favors_missing_and_uneven_fields():
    rows = [
        row(
            f'model/a-{index}',
            'bench-complete',
            'eval',
            0.9,
            generation_temperature=0.2,
            source_locator=None,
        )
        for index in range(5)
    ] + [
        row(
            f'model/b-{index}',
            'bench-missing',
            'eval',
            0.8,
            generation_temperature=None,
            source_locator=None,
        )
        for index in range(5)
    ]

    summary = stats.metadata_completeness(
        rows, top_benchmarks=2, top_fields=1
    )

    assert summary['fields'][0]['key'] == 'generation_temperature'
    assert summary['fields'][0]['selection_score'] > 0.05


def test_cli_help_uses_summary_limit_not_top_n(capsys):
    try:
        stats.parse_args(['--help'])
    except SystemExit:
        pass

    output = capsys.readouterr().out
    assert '--summary-limit' in output
    assert '--top-n' not in output
