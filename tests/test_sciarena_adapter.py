from every_eval_ever.eval_types import EvaluationLog
from utils.sciarena import adapter as sciarena_adapter


SCIARENA_ROWS = [
    {
        'modelId': 'o3',
        'rating': 1151.3779,
        'rank': 1,
        'num_battles': 1076,
        'rating_q025': 1127.1508,
        'rating_q975': 1174.3958,
        'variance': None,
        'cost_per_100_calls_usd': 3.5438,
    },
    {
        'modelId': 'GPT-5',
        'rating': 947.5,
        'rank': 2,
        'num_battles': 900,
        'rating_q025': 920.0,
        'rating_q975': 970.0,
        'variance': 8.4,
        'cost_per_100_calls_usd': 5.0,
    },
]


def test_compute_metric_bounds_uses_full_scrape_ranges():
    bounds = sciarena_adapter.compute_metric_bounds(SCIARENA_ROWS)

    assert bounds == {
        'elo': {
            'min_score': 947.5,
            'max_score': 1151.3779,
        },
        'rank': {
            'min_score': 1.0,
            'max_score': 2.0,
        },
        'cost_per_100_calls_usd': {
            'min_score': 0.0,
            'max_score': 5.0,
        },
    }


def test_make_log_emits_metric_type_and_distinct_evaluation_names():
    bounds = sciarena_adapter.compute_metric_bounds(SCIARENA_ROWS)

    log, developer, model = sciarena_adapter.make_log(SCIARENA_ROWS[0], bounds)

    assert developer == 'openai'
    assert model == 'o3'

    results = log['evaluation_results']
    assert [result['evaluation_name'] for result in results] == [
        'overall_elo',
        'overall_rank',
        'overall_cost_per_100_calls_usd',
    ]
    assert len({result['evaluation_name'] for result in results}) == len(
        results
    )

    for result in results:
        metric_config = result['metric_config']
        assert metric_config['metric_type'] == 'continuous'
        assert metric_config['score_type'] == 'continuous'
        assert 'min_score' in metric_config
        assert 'max_score' in metric_config

    EvaluationLog.model_validate(log)
