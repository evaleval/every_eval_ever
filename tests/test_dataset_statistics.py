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
) -> dict:
    return {
        'schema_version': '0.2.2',
        'evaluation_id': f'{model_id}/{benchmark}/{evaluation_name}',
        'model_id': model_id,
        'model_developer': model_id.split('/')[0],
        'benchmark': benchmark,
        'evaluation_name': evaluation_name,
        'score': score,
        'min_score': min_score,
        'max_score': max_score,
        'lower_is_better': lower_is_better,
        'score_type': score_type,
        'has_uncertainty': False,
    }


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


def test_shared_evaluation_key_includes_score_scale_and_direction():
    base = row('a', 'bench', 'eval', 0.8)
    different_scale = row('a', 'bench', 'eval', 80.0, max_score=100.0)
    different_direction = row('a', 'bench', 'eval', 0.2, lower_is_better=True)

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
    assert 'coverage_aware_model_summaries' in report['observational']
    assert 'pairwise_model_comparisons' in report['observational']


def test_cli_help_uses_summary_limit_not_top_n(capsys):
    try:
        stats.parse_args(['--help'])
    except SystemExit:
        pass

    output = capsys.readouterr().out
    assert '--summary-limit' in output
    assert '--top-n' not in output
