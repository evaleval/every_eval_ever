import pytest

from every_eval_ever.adapters.artificial_analysis.adapter import (
    compute_observed_max_scores,
    convert_models,
)


def _model(name, creator, evaluations):
    return {
        'id': name,
        'name': name,
        'slug': name,
        'model_creator': {
            'name': creator,
            'slug': creator.lower(),
        },
        'evaluations': evaluations,
    }


def test_bad_metric_keeps_model_with_other_valid_metrics(tmp_path):
    valid = _model(
        'model-a',
        'Creator',
        {
            'artificial_analysis_intelligence_index': 50,
            'mmlu_pro': 'not-a-score',
        },
    )
    bounds = compute_observed_max_scores([valid])

    result = convert_models(
        [valid],
        {},
        tmp_path / 'data' / 'artificial-analysis-llms',
        bounds,
        '1234',
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert [
        metric.evaluation_result_id
        for metric in result.records[0].eval_log.evaluation_results
    ] == ['artificial_analysis.artificial_analysis_intelligence_index']


def test_bad_identity_does_not_discard_other_models(tmp_path):
    good = _model(
        'model-a',
        'Creator',
        {'artificial_analysis_intelligence_index': 50},
    )
    bad = _model(
        'model-b',
        '',
        {'artificial_analysis_intelligence_index': 25},
    )
    bounds = compute_observed_max_scores([good, bad])

    result = convert_models(
        [good, bad],
        {},
        tmp_path / 'data' / 'artificial-analysis-llms',
        bounds,
        '1234',
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record == bad


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('gpqa', 0.748),
        ('scicode', 0.399),
        ('lcr', 0.307),
        ('terminalbench_hard', 0.440),
    ],
)
def test_existing_frontier_metric_mapping(tmp_path, field, value):
    model = _model('model-a', 'Creator', {field: value})
    result = convert_models(
        [model], {}, tmp_path / 'data' / 'artificial-analysis-llms', {}, '1234'
    )

    assert not result.failures
    metric, = result.records[0].eval_log.evaluation_results
    assert metric.evaluation_name == f'artificial_analysis.{field}'
    assert metric.score_details.score == value
    assert metric.score_details.details['raw_value_field'] == f'evaluations.{field}'
    assert metric.metric_config.min_score == 0
    assert metric.metric_config.max_score == 1


def test_missing_frontier_fields_do_not_generate_scores(tmp_path):
    model = _model(
        'model-a',
        'Creator',
        {'gpqa': None, 'scicode': None, 'lcr': None, 'terminalbench_hard': None,
         'artificial_analysis_intelligence_index': 50},
    )
    result = convert_models(
        [model], {}, tmp_path / 'data' / 'artificial-analysis-llms',
        compute_observed_max_scores([model]), '1234',
    )

    assert not result.failures
    assert [r.evaluation_name for r in result.records[0].eval_log.evaluation_results] == [
        'artificial_analysis.artificial_analysis_intelligence_index'
    ]
