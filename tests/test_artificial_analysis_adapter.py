from every_eval_ever.adapters.artificial_analysis.adapter import (
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


def test_v2_free_tier_fallback_pagination_and_unbounded_metrics(
    monkeypatch, tmp_path
):
    import requests

    from every_eval_ever.adapters.artificial_analysis import adapter
    from every_eval_ever.helpers import FetchError

    calls: list[str] = []

    def fake_fetch_json(url, **kwargs):
        calls.append(url)
        if url == adapter.SOURCE_URL:
            resp = requests.Response()
            resp.status_code = 403
            raise FetchError('Pro tier required') from requests.HTTPError(
                '403', response=resp
            )
        if url == adapter.FREE_SOURCE_URL:
            return {
                'tier': 'free',
                'pagination': {
                    'page': 1,
                    'page_size': 1,
                    'total_pages': 2,
                    'has_more': True,
                },
                'data': [
                    {
                        'id': 'id-1',
                        'name': 'Model One',
                        'slug': 'model-one',
                        'release_date': '2026-05-01',
                        'model_creator': {'id': 'c-1', 'name': 'OpenAI'},
                        'evaluations': {
                            'artificial_analysis_intelligence_index': 55.2,
                            'artificial_analysis_finance_and_accounting_index': 61.0,
                        },
                        'pricing': {
                            'price_1m_input_tokens': 2.5,
                            'price_1m_output_tokens': 10.0,
                        },
                        'performance': {
                            'median_output_tokens_per_second': None,
                            'median_time_to_first_token_seconds': None,
                            'median_time_to_first_answer_token_seconds': None,
                        },
                        'performance_v3': {
                            'mean_output_tokens_per_second': 120.5,
                            'mean_time_to_first_token_seconds': 0.45,
                            'mean_time_to_first_answer_token_seconds': 1.25,
                        },
                    }
                ],
            }
        if url == f'{adapter.FREE_SOURCE_URL}?page=2':
            return {
                'tier': 'free',
                'pagination': {
                    'page': 2,
                    'page_size': 1,
                    'total_pages': 2,
                    'has_more': False,
                },
                'data': [
                    {
                        'id': 'id-2',
                        'name': 'Model Two',
                        'slug': 'model-two',
                        'release_date': None,
                        'model_creator': {'id': 'c-2', 'name': 'Anthropic'},
                        'evaluations': {
                            'artificial_analysis_intelligence_index': 62.0,
                        },
                        'pricing': {
                            'price_1m_input_tokens': 3.0,
                            'price_1m_output_tokens': 15.0,
                        },
                        'performance': {},
                        'performance_v3': None,
                    }
                ],
            }
        raise AssertionError(f'unexpected url: {url}')

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch_json)
    payload = adapter.fetch_payload('free-key')
    assert len(payload['data']) == 2
    assert payload['api_endpoint'] == adapter.FREE_SOURCE_URL

    converted = convert_models(
        payload['data'],
        payload,
        tmp_path / 'data' / 'artificial-analysis-llms',
        '1700000000.0',
    )
    assert converted.failures == []
    assert len(converted.records) == 2
    first_log = converted.records[0].eval_log
    assert (
        first_log.evaluation_id
        == 'artificial-analysis-llms/openai/model-one/1700000000.0'
    )
    by_id = {r.evaluation_result_id: r for r in first_log.evaluation_results}
    assert (
        by_id[
            'artificial_analysis.artificial_analysis_intelligence_index'
        ].metric_config.max_score
        == float('inf')
    )
    assert (
        by_id['artificial_analysis.mean_output_tokens_per_second']
        .score_details.score
        == 120.5
    )
