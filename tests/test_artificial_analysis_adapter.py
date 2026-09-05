from every_eval_ever.adapters.artificial_analysis.adapter import (
    METRIC_SPECS,
    convert_models,
    make_metric_config,
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

    result = convert_models(
        [valid],
        {},
        tmp_path / 'data' / 'artificial-analysis-llms',
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

    result = convert_models(
        [good, bad],
        {},
        tmp_path / 'data' / 'artificial-analysis-llms',
        '1234',
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record == bad


def _spec(name):
    return next(s for s in METRIC_SPECS if s.evaluation_name.endswith(name))


def test_an_unbounded_quantity_declares_no_ceiling():
    """A price has none, and the dearest model in a batch is not one."""
    config = make_metric_config(_spec('price_1m_input_tokens'), {})
    assert config.max_score == float('inf')
    assert config.additional_details['bound_strategy'] == 'unbounded_above'


def test_a_metric_with_no_published_range_declares_no_bounds():
    config = make_metric_config(
        _spec('artificial_analysis_intelligence_index'), {}
    )
    assert config.min_score is None
    assert config.max_score is None
    assert config.score_type is None
    assert config.additional_details['bounds_status'] == 'unknown'
