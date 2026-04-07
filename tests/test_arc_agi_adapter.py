from every_eval_ever.eval_types import EvaluationLog
from utils.arc_agi import adapter as arc_agi_adapter


ARC_AGI_ROWS = [
    {
        'datasetId': 'v1_Public_Eval',
        'modelId': 'GPT-5',
        'score': 0.72,
        'costPerTask': 4.2,
        'resultsUrl': '',
        'display': True,
    },
    {
        'datasetId': 'v1_Semi_Private',
        'modelId': 'GPT-5',
        'score': 0.68,
        'costPerTask': 5.1,
        'resultsUrl': '',
        'display': True,
    },
]


def test_compute_metric_bounds_sets_continuous_score_ranges():
    bounds = arc_agi_adapter.compute_metric_bounds(ARC_AGI_ROWS)

    assert bounds == {
        'score': {
            'min_score': 0.0,
            'max_score': 1.0,
        },
        'cost_per_task': {
            'min_score': 0.0,
            'max_score': 5.1,
        },
    }


def test_make_log_emits_continuous_score_type_for_results():
    bounds = arc_agi_adapter.compute_metric_bounds(ARC_AGI_ROWS)

    log, developer, model = arc_agi_adapter.make_log(
        ARC_AGI_ROWS,
        'openai',
        'gpt-5',
        bounds,
    )

    assert developer == 'openai'
    assert model == 'gpt-5'

    results = log['evaluation_results']
    assert [result['evaluation_result_id'] for result in results] == [
        'v1_Public_Eval::score',
        'v1_Public_Eval::cost_per_task',
        'v1_Semi_Private::score',
        'v1_Semi_Private::cost_per_task',
    ]

    for result in results:
        metric_config = result['metric_config']
        assert metric_config['score_type'] == 'continuous'
        assert 'min_score' in metric_config
        assert 'max_score' in metric_config
        assert 'metric_type' not in metric_config

    EvaluationLog.model_validate(log)
