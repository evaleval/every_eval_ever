import json

import pytest

from every_eval_ever.adapters.artificial_analysis import adapter
from every_eval_ever.adapters.artificial_analysis.adapter import (
    compute_observed_max_scores,
    convert_models,
    merge_site_scores,
)


def _model(name, creator, evaluations=None, **sections):
    return {
        'id': name,
        'name': name,
        'slug': name,
        'model_creator': {'id': f'{creator}-id', 'name': creator},
        'evaluations': evaluations or {},
        **sections,
    }


def _page(text):
    """A page whose Next.js flight data is ``text``, split over two chunks."""
    half = len(text) // 2
    return ''.join(
        f'<script>self.__next_f.push([1,{json.dumps(part)}])</script>'
        for part in (text[:half], text[half:])
    )


def _convert(models, tmp_path):
    return convert_models(
        models,
        {},
        tmp_path / 'data' / 'artificial-analysis-llms',
        compute_observed_max_scores(models),
        '1234',
    )


def test_bad_metric_keeps_model_with_other_valid_metrics(tmp_path):
    valid = _model(
        'model-a',
        'Creator',
        {'artificial_analysis_intelligence_index': 50},
        pricing={'price_1m_input_tokens': 'not-a-price'},
    )

    result = _convert([valid], tmp_path)

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert [
        metric.evaluation_result_id
        for metric in result.records[0].eval_log.evaluation_results
    ] == ['artificial_analysis.artificial_analysis_intelligence_index']


def test_bad_identity_does_not_discard_other_models(tmp_path):
    good = _model(
        'model-a', 'Creator', {'artificial_analysis_intelligence_index': 50}
    )
    bad = _model('model-b', '', {'artificial_analysis_intelligence_index': 25})

    result = _convert([good, bad], tmp_path)

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record == bad


@pytest.mark.parametrize(
    ('site_key', 'name', 'value'),
    [
        ('gpqa-diamond', 'gpqa', 0.748),
        ('scicode', 'scicode', 0.399),
        ('artificial-analysis-long-context-reasoning', 'lcr', 0.307),
        ('terminalbench-hard', 'terminalbench_hard', 0.440),
        ('critpt', 'critpt', 0.317),
        ('critpt-v2-hard', 'critpt_v2_hard', 0.68),
    ],
)
def test_leaderboard_scores_map_to_evaluations(tmp_path, site_key, name, value):
    model = _model('model-a', 'Creator', site_scores={site_key: value})

    result = _convert([model], tmp_path)

    assert not result.failures
    (metric,) = result.records[0].eval_log.evaluation_results
    assert metric.evaluation_name == f'artificial_analysis.{name}'
    assert metric.score_details.score == value
    assert metric.score_details.details['raw_value_field'] == (
        f'site_scores.{site_key}'
    )
    assert metric.source_data.url == [adapter.LEADERBOARD_URL]
    assert metric.metric_config.min_score == 0
    assert metric.metric_config.max_score == 1


def test_critpt_v2_comes_from_its_evaluation_page(tmp_path):
    model = _model(
        'model-a', 'Creator', site_critpt_v2={'critptAgentic': 0.817}
    )

    result = _convert([model], tmp_path)

    (metric,) = result.records[0].eval_log.evaluation_results
    assert metric.evaluation_name == 'artificial_analysis.critpt_v2'
    assert metric.score_details.score == 0.817
    assert metric.source_data.url == [adapter.CRITPT_V2_URL]


def test_missing_scores_do_not_generate_results(tmp_path):
    model = _model(
        'model-a',
        'Creator',
        {'artificial_analysis_intelligence_index': 50},
        site_scores={'gpqa-diamond': None, 'critpt': None},
        performance={'median_output_tokens_per_second': None},
    )

    result = _convert([model], tmp_path)

    assert not result.failures
    assert [
        r.evaluation_name for r in result.records[0].eval_log.evaluation_results
    ] == ['artificial_analysis.artificial_analysis_intelligence_index']


def test_parse_leaderboard_models_reads_initial_models():
    rows = [{'id': 'm1', 'name': 'M1', 'scores': {'critpt': 0.3}}]
    page = _page(f'["$","$L4c",null,{{"initialModels":{json.dumps(rows)}}}]')

    assert adapter.parse_leaderboard_models(page) == rows


@pytest.mark.parametrize(
    'text',
    ['{"somethingElse":[]}', '{"initialModels":[]}', '{"initialModels":[{}]}'],
)
def test_leaderboard_shape_drift_fails_closed(text):
    with pytest.raises(ValueError):
        adapter.parse_leaderboard_models(_page(text))


def test_parse_critpt_v2_models_reads_model_rows():
    model_id = '593f2d23-247b-41c9-a679-cebd225f0a9a'
    row = {
        'id': model_id,
        'slug': 'm1',
        'critpt': 0.35,
        'critptAgentic': 0.86,
        'critptAgenticHard': 0.76,
    }
    page = _page(f'[{json.dumps(row, separators=(",", ":"))}]')

    assert adapter.parse_critpt_v2_models(page) == {
        model_id: {'critptAgentic': 0.86}
    }
    with pytest.raises(ValueError):
        adapter.parse_critpt_v2_models(_page('[]'))


def test_site_scores_join_on_model_id_and_report_unmatched():
    models = [_model('m1', 'Creator')]
    site = {
        'leaderboard_models': [
            {'id': 'm1', 'name': 'M1', 'scores': {'critpt': 0.3}},
            {'id': 'site-only', 'name': 'Site only', 'scores': {}},
        ],
        'critpt_v2': {'m1': {'critptAgentic': 0.8}},
    }

    merged, exclusions = merge_site_scores(models, site)

    assert merged[0]['site_scores'] == {'critpt': 0.3}
    assert merged[0]['site_critpt_v2'] == {'critptAgentic': 0.8}
    assert 'site_scores' not in models[0]
    assert [e.source_record for e in exclusions] == [
        {'id': 'site-only', 'name': 'Site only'}
    ]


def test_fetch_payload_follows_pagination(monkeypatch):
    pages = {
        1: {
            'tier': 'free',
            'intelligence_index_version': '5',
            'pagination': {'has_more': True},
            'data': [_model('m1', 'C')],
        },
        2: {'pagination': {'has_more': False}, 'data': [_model('m2', 'C')]},
    }
    urls = []

    def fake_fetch_json(url, headers=None):
        urls.append(url)
        assert headers == {'x-api-key': 'key'}
        return pages[int(url.rsplit('=', 1)[1])]

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch_json)

    payload = adapter.fetch_payload('key')

    assert urls == [
        f'{adapter.SOURCE_URL}?page=1',
        f'{adapter.SOURCE_URL}?page=2',
    ]
    assert [m['id'] for m in payload['data']] == ['m1', 'm2']
    assert payload['intelligence_index_version'] == '5'
    assert payload['pages_fetched'] == 2


def test_site_page_errors_are_returned_not_raised(monkeypatch):
    def fake_fetch_text(url):
        raise RuntimeError(f'down: {url}')

    monkeypatch.setattr(adapter, 'fetch_text', fake_fetch_text)

    site = adapter.fetch_site()

    assert [error['url'] for error in site['errors']] == [
        adapter.LEADERBOARD_URL,
        adapter.CRITPT_V2_URL,
    ]
    assert 'leaderboard_models' not in site
