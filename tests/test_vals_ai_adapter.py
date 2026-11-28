from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.adapters.vals_ai import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file

FIXTURE_PATH = (
    Path(__file__).parent / 'data' / 'vals_ai' / 'finance_agent_payload.json'
)


def sample_payload() -> dict:
    return {
        'source_url': 'https://www.vals.ai/benchmarks',
        'benchmarks': [
            {
                'source_url': 'https://www.vals.ai/benchmarks/finance_agent',
                'metadata': {
                    'benchmark': 'Finance Agent (v1.1)',
                    'slug': 'finance_agent',
                    'benchmark_id': 'finance_agent',
                    'updated': '2026-04-23',
                    'dataset_type': 'private',
                    'industry': 'finance',
                    'tasks': {
                        'overall': 'Overall',
                        'numerical_reasoning': 'Numerical Reasoning',
                    },
                    'models': [
                        'openai/gpt-5.4',
                        'unknown-model',
                    ],
                },
                'tasks': {
                    'overall': {
                        'openai/gpt-5.4': {
                            'accuracy': 72.222,
                            'latency': 273.714,
                            'stderr': 4.748,
                            'cost_per_test': 0.785991,
                            'temperature': 1,
                            'top_p': None,
                            'max_output_tokens': 65536,
                            'reasoning': None,
                            'reasoning_effort': 'high',
                            'verbosity': None,
                            'compute_effort': None,
                            'provider': 'OpenAI',
                        },
                    },
                    'numerical_reasoning': {
                        'openai/gpt-5.4': {
                            'accuracy': 80.0,
                            'latency': None,
                            'stderr': None,
                            'cost_per_test': None,
                            'temperature': None,
                            'provider': 'OpenAI',
                        }
                    },
                    'short_answer': {
                        'unknown-model': {
                            'accuracy': 44.0,
                            'max_output_tokens': 0,
                            'provider': 'Mystery Lab',
                        }
                    },
                },
            },
            {
                'source_url': 'https://www.vals.ai/benchmarks/poker_agent',
                'metadata': {
                    'benchmark': 'Poker Agent',
                    'slug': 'poker_agent',
                    'updated': '2025-12-23',
                    'dataset_type': 'private',
                    'industry': 'games',
                    'tasks': {'overall': 'Overall'},
                },
                'tasks': {
                    'overall': {
                        'anthropic/claude-sonnet-4-6': {
                            'accuracy': 90.0,
                            'stderr': 1.0,
                            'provider': 'Anthropic',
                        },
                        'anthropic/claude-opus-4-7': {
                            'accuracy': 1100.5,
                            'stderr': 12.5,
                            'provider': 'Anthropic',
                        },
                    }
                },
            },
        ],
    }


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    assert len(bundles) == 4

    for bundle in bundles:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.source_metadata.source_organization_name == 'Vals.ai'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert (
            validated.source_metadata.evaluator_relationship.value
            == 'third_party'
        )


def test_groups_one_benchmark_page_per_model():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    by_id = {bundle.log.evaluation_id: bundle.log for bundle in bundles}

    finance = by_id['vals-ai/finance_agent/openai_gpt-5.4/1234567890.0']
    assert finance.model_info.id == 'openai/gpt-5.4'
    assert finance.model_info.name == 'gpt-5.4'
    assert len(finance.evaluation_results) == 2
    assert {
        result.evaluation_name for result in finance.evaluation_results
    } == {
        'vals_ai.finance_agent.numerical_reasoning',
        'vals_ai.finance_agent.overall',
    }


def test_unknown_unprefixed_model_uses_provider_fallback():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    unknown = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.name == 'unknown-model'
    )

    assert unknown.model_info.id == 'mystery-lab/unknown-model'
    assert unknown.model_info.developer == 'mystery-lab'
    assert (
        unknown.model_info.additional_details['vals_provider'] == 'Mystery Lab'
    )
    result = unknown.evaluation_results[0]
    assert result.score_details.details['max_output_tokens'] == '0'
    assert result.generation_config is None


def test_routed_model_components_use_explicit_schema_fields():
    payload = sample_payload()
    raw_id = 'together/langston/nim/nvidia/llama-3.3-nemotron-super-49b-v1'
    payload['benchmarks'][0]['tasks']['overall'][raw_id] = {
        'accuracy': 55.0,
        'provider': 'Together AI',
    }

    bundles = adapter.make_logs(
        payload,
        retrieved_timestamp='1234567890.0',
    )
    routed = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.additional_details['vals_model_id'] == raw_id
    )

    assert routed.model_info.name == ('llama-3.3-nemotron-super-49b-v1')
    assert routed.model_info.id == ('nvidia/llama-3.3-nemotron-super-49b-v1')
    assert routed.model_info.developer == 'nvidia'
    assert routed.model_info.inference_platform == 'Together AI'
    assert routed.model_info.inference_engine.name == 'NIM'
    assert (
        routed.model_info.additional_details['vals_route']
        == 'together/langston/nim'
    )


def test_hosted_route_takes_developer_from_paired_baseline():
    payload = sample_payload()
    overall = payload['benchmarks'][0]['tasks']['overall']
    overall['kimi/kimi-k3'] = {'accuracy': 71.77, 'provider': 'Moonshot AI'}
    paired = {'baseline_model': 'kimi/kimi-k3', 'delta': 2.083}
    overall['fireworks/kimi-k3'] = {
        'accuracy': 73.853,
        'provider': 'Fireworks AI',
        'paired_accuracy': paired,
    }
    overall['baseten/moonshotai/Kimi-K3'] = {
        'accuracy': 67.285,
        'provider': 'Baseten',
        'paired_accuracy': paired,
    }

    bundles = adapter.make_logs(
        payload,
        retrieved_timestamp='1234567890.0',
    )
    by_raw_id = {
        bundle.log.model_info.additional_details['vals_model_id']: bundle.log
        for bundle in bundles
    }

    for raw_id, platform in [
        ('kimi/kimi-k3', 'Moonshot AI'),
        ('fireworks/kimi-k3', 'Fireworks AI'),
        ('baseten/moonshotai/Kimi-K3', 'Baseten'),
    ]:
        model_info = by_raw_id[raw_id].model_info
        assert model_info.id == 'moonshotai/kimi-k3'
        assert model_info.developer == 'moonshotai'
        assert model_info.inference_platform == platform
    hosted = by_raw_id['fireworks/kimi-k3'].model_info.additional_details
    assert hosted['vals_route'] == 'fireworks'
    assert hosted['vals_baseline_model'] == 'kimi/kimi-k3'


def test_openrouter_route_is_not_the_developer():
    payload = sample_payload()
    raw_id = 'openrouter/typesafe/jev-router'
    payload['benchmarks'][0]['tasks']['overall'][raw_id] = {
        'accuracy': 61.0,
        'provider': None,
    }

    bundles = adapter.make_logs(
        payload,
        retrieved_timestamp='1234567890.0',
    )
    routed = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.additional_details['vals_model_id'] == raw_id
    )

    assert routed.model_info.id == 'typesafe/jev-router'
    assert routed.model_info.developer == 'typesafe'
    assert routed.model_info.inference_platform is None
    assert routed.model_info.additional_details['vals_route'] == 'openrouter'


def test_harness_is_recorded_with_generation_details():
    payload = sample_payload()
    row = payload['benchmarks'][0]['tasks']['overall']['openai/gpt-5.4']
    row['harness'] = 'OpenHands'
    row['harness_config'] = 'Exa auto search with highlights'

    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    overall = next(
        result
        for bundle in bundles
        for result in bundle.log.evaluation_results
        if result.evaluation_result_id
        == 'finance_agent:overall:openai/gpt-5.4:score'
    )

    assert overall.score_details.details['harness'] == 'OpenHands'
    generation_details = overall.generation_config.additional_details
    assert generation_details['harness'] == 'OpenHands'
    assert generation_details['harness_config'] == (
        'Exa auto search with highlights'
    )


def test_preserves_source_fields_and_uncertainty():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    finance = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.id == 'openai/gpt-5.4'
    )
    overall = next(
        result
        for result in finance.evaluation_results
        if result.evaluation_name == 'vals_ai.finance_agent.overall'
    )

    assert overall.source_data.source_type == 'other'
    assert (
        overall.source_data.additional_details['leaderboard_page_url']
        == 'https://www.vals.ai/benchmarks/finance_agent'
    )
    assert finance.evaluation_timestamp is None
    assert overall.evaluation_timestamp is None
    assert overall.metric_config.metric_unit == 'percent'
    assert overall.metric_config.max_score == 100
    assert overall.score_details.score == 72.222
    assert overall.score_details.details['cost_per_test'] == '0.785991'
    assert overall.score_details.details['reasoning_effort'] == 'high'
    assert overall.score_details.uncertainty.standard_error.value == 4.748


def test_non_percent_scores_are_kept_without_invented_bounds():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    by_model = {bundle.log.model_info.id: bundle.log for bundle in bundles}

    for model_id in (
        'anthropic/claude-opus-4-7',
        'anthropic/claude-sonnet-4-6',
    ):
        result = by_model[model_id].evaluation_results[0]
        assert result.metric_config.metric_unit == 'points'
        assert result.metric_config.score_type is None
        assert result.metric_config.min_score is None
        assert result.metric_config.max_score is None
        assert (
            result.metric_config.additional_details['max_score_source']
            == 'not_provided'
        )


def test_canonical_model_id_collisions_fail_clearly():
    payload = sample_payload()
    payload['benchmarks'][0]['tasks']['overall']['gpt-5.4'] = {
        'accuracy': 70.0,
        'provider': 'openai',
    }

    try:
        adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        assert 'collide after canonicalization' in str(exc)
    else:
        raise AssertionError('expected canonicalization collision to fail')


def test_missing_benchmark_slug_fails():
    payload = sample_payload()
    payload['benchmarks'].append(
        {
            'source_url': 'https://www.vals.ai/benchmarks/bad',
            'metadata': {'benchmark': 'Bad'},
            'tasks': {'overall': {'openai/gpt-5': {'accuracy': 99.0}}},
        }
    )

    try:
        adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        assert 'missing a slug' in str(exc)
    else:
        raise AssertionError('expected missing slug to fail')


def test_non_numeric_score_fails_with_context():
    payload = sample_payload()
    payload['benchmarks'][0]['tasks']['overall']['openai/bad-score'] = {
        'accuracy': 'N/A',
        'provider': 'OpenAI',
    }

    try:
        adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        message = str(exc)
        assert 'Non-numeric Vals.ai score' in message
        assert 'finance_agent/overall/openai/bad-score' in message
    else:
        raise AssertionError('expected non-numeric score to fail')


def test_null_score_is_a_failure_without_explicit_unevaluated_status():
    payload = sample_payload()
    null_row = {
        'accuracy': None,
        'provider': 'Mystery Lab',
    }
    payload['benchmarks'][0]['tasks']['overall']['unknown-model'] = null_row

    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')

    assert result.records
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == (
        'finance_agent/overall/unknown-model'
    )
    assert result.failures[0].reason == (
        'Vals.ai model row is missing accuracy'
    )
    assert result.failures[0].source_record == null_row


def test_non_numeric_score_keeps_valid_models_in_conversion_result():
    payload = sample_payload()
    bad_row = {
        'accuracy': 'N/A',
        'provider': 'OpenAI',
    }
    payload['benchmarks'][0]['tasks']['overall']['openai/bad-score'] = bad_row

    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')

    assert result.records
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == (
        'finance_agent/overall/openai/bad-score'
    )
    assert result.failures[0].source_record == bad_row


def test_astro_payload_extraction():
    props = (
        '{&quot;benchmarkView&quot;:[0,{&quot;default&quot;:[0,{'
        '&quot;metadata&quot;:[0,{&quot;benchmark&quot;:[0,&quot;AIME&quot;],'
        '&quot;slug&quot;:[0,&quot;aime&quot;]}],'
        '&quot;tasks&quot;:[0,{&quot;overall&quot;:[0,{'
        '&quot;openai/gpt-5&quot;:[0,{&quot;accuracy&quot;:[0,95.0],'
        '&quot;showBadge&quot;:[0]}]}]}]}]}]}'
    )
    html = (
        '<astro-island component-url="/_astro/BenchmarkView.abc.js" '
        f'props="{props}"></astro-island>'
    )

    normalized = adapter.normalize_benchmark_page(
        html, 'https://www.vals.ai/benchmarks/aime'
    )

    assert normalized['metadata']['benchmark'] == 'AIME'
    assert normalized['tasks']['overall']['openai/gpt-5']['accuracy'] == 95.0
    assert 'showBadge' not in normalized['tasks']['overall']['openai/gpt-5']


def test_unknown_astro_tags_fail_loudly():
    props = (
        '{&quot;benchmarkView&quot;:[0,{&quot;metadata&quot;:[0,'
        '{&quot;benchmark&quot;:[4,&quot;2026-04-23T00:00:00.000Z&quot;],'
        '&quot;slug&quot;:[0,&quot;aime&quot;]}],&quot;tasks&quot;:[0,{}]}]}'
    )
    html = (
        '<astro-island component-url="/_astro/BenchmarkView.abc.js" '
        f'props="{props}"></astro-island>'
    )

    try:
        adapter.normalize_benchmark_page(
            html, 'https://www.vals.ai/benchmarks/aime'
        )
    except ValueError as exc:
        assert 'Unsupported Astro serialized value tag' in str(exc)
    else:
        raise AssertionError('expected unsupported Astro tag to fail')


def test_extract_collection_input_json_does_not_fetch(monkeypatch):
    def fail_fetch(_url: str) -> str:
        raise AssertionError('input_json replay should not fetch live pages')

    monkeypatch.setattr(adapter, 'fetch_text', fail_fetch)

    payload = adapter.extract_collection(input_json=FIXTURE_PATH)

    assert payload['benchmarks'][0]['metadata']['slug'] == 'finance_agent'
    assert payload['benchmarks'][0]['source_url'] == (
        'https://www.vals.ai/benchmarks/finance_agent'
    )


API = 'https://api.test'
SITE = 'https://example.test'


def _routes(monkeypatch, routes: dict[str, object]) -> list[str]:
    """Serve ``routes`` (JSON-encoding non-strings); fail any other URL."""
    calls: list[str] = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        if url not in routes:
            raise RuntimeError(f'unavailable: {url}')
        body = routes[url]
        return body if isinstance(body, str) else json.dumps(body)

    monkeypatch.setattr(adapter, 'fetch_text', fake_fetch)
    return calls


def _view(slug: str, *, archived: bool | None = None) -> dict:
    metadata = {'benchmark': slug.upper(), 'slug': slug}
    if archived is not None:
        metadata['archived'] = archived
    return {
        'metadata': metadata,
        'tasks': {'overall': {'openai/gpt-5': {'accuracy': 95.0}}},
    }


def _api_benchmark(slug: str, artifact: str, view: dict) -> dict[str, object]:
    return {
        f'{API}/benchmarks/{slug}/artifacts': {
            'benchmark': slug,
            'artifacts': [
                {'path': 'results.json', 'type': 'results'},
                {'path': artifact, 'type': 'benchmark_view'},
            ],
        },
        f'{API}/benchmarks/{slug}/artifacts/{artifact}': {
            'benchmark': slug,
            'artifact': artifact,
            'data': view,
        },
    }


def test_api_discovers_every_listed_benchmark(monkeypatch):
    calls = _routes(
        monkeypatch,
        {
            f'{API}/benchmarks': {'benchmarks': ['web_search', 'old']},
            # The artifact name and the view's slug need not match the API slug.
            **_api_benchmark(
                'web_search',
                'benchmark_view_web_search_backends.json',
                _view('web_search_backends'),
            ),
            **_api_benchmark(
                'old', 'benchmark_view_old.json', _view('old', archived=True)
            ),
        },
    )

    payload = adapter.extract_collection(api_url=API, base_url=SITE)

    assert [url for url in calls if url.startswith(SITE)] == [
        f'{SITE}/benchmarks'
    ]
    assert 'unavailable' in payload['discovery_errors']['site']
    old, web_search = payload['benchmarks']  # slugs are sorted
    assert web_search['source_url'] == f'{SITE}/benchmarks/web_search_backends'
    assert web_search['benchmark_view_url'] == (
        f'{API}/benchmarks/web_search/artifacts/'
        'benchmark_view_web_search_backends.json'
    )
    assert old['metadata']['archived'] is True
    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')
    assert len(result.records) == 2
    assert not result.failures


def _page(slug: str, accuracy: float = 95.0) -> str:
    props = (
        '{&quot;benchmarkView&quot;:[0,{&quot;metadata&quot;:[0,'
        f'{{&quot;benchmark&quot;:[0,&quot;{slug}&quot;],'
        f'&quot;slug&quot;:[0,&quot;{slug}&quot;]}}],'
        '&quot;tasks&quot;:[0,{&quot;overall&quot;:[0,{&quot;openai/gpt-5&quot;:'
        f'[0,{{&quot;accuracy&quot;:[0,{accuracy}]}}]}}]}}]}}]}}'
    )
    return (
        '<astro-island component-url="/_astro/BenchmarkView.abc.js" '
        f'props="{props}"></astro-island>'
    )


def test_api_first_then_site_remainder_deduplicated(monkeypatch):
    index = ''.join(
        f'<a href="/benchmarks/{slug}">x</a>'
        for slug in ('web_search_backends', 'alias', 'extra')
    )
    calls = _routes(
        monkeypatch,
        {
            f'{API}/benchmarks': {'benchmarks': ['web_search']},
            **_api_benchmark(
                'web_search',
                'benchmark_view_web_search_backends.json',
                _view('web_search_backends'),
            ),
            f'{SITE}/benchmarks': index,
            # A differently linked page that serves an already-fetched view.
            f'{SITE}/benchmarks/alias': _page('web_search_backends', 1.0),
            f'{SITE}/benchmarks/extra': _page('extra'),
        },
    )

    payload = adapter.extract_collection(api_url=API, base_url=SITE)

    # The view slug the API already served is never scraped.
    assert f'{SITE}/benchmarks/web_search_backends' not in calls
    assert [b['metadata']['slug'] for b in payload['benchmarks']] == [
        'web_search_backends',
        'extra',
    ]
    api_copy, extra = payload['benchmarks']
    assert api_copy['tasks']['overall']['openai/gpt-5']['accuracy'] == 95.0
    assert extra['source_url'] == f'{SITE}/benchmarks/extra'
    assert payload['duplicate_benchmarks'] == [
        {
            'benchmark_slug': 'web_search_backends',
            'kept': api_copy['benchmark_view_url'],
            'dropped': f'{SITE}/benchmarks/alias',
        }
    ]
    assert not payload['source_failures']


def test_api_listing_down_scrapes_site_index(monkeypatch):
    calls = _routes(
        monkeypatch,
        {
            f'{SITE}/benchmarks': '<a href="/benchmarks/aime">AIME</a>',
            f'{SITE}/benchmarks/aime': _page('aime'),
        },
    )

    payload = adapter.extract_collection(api_url=API, base_url=SITE)

    assert calls == [
        f'{API}/benchmarks',
        f'{SITE}/benchmarks',
        f'{SITE}/benchmarks/aime',
    ]
    assert [b['metadata']['slug'] for b in payload['benchmarks']] == ['aime']
    assert 'unavailable' in payload['discovery_errors']['api']


def test_both_listings_down_fails(monkeypatch):
    _routes(monkeypatch, {})
    try:
        adapter.extract_collection(api_url=API, base_url=SITE)
    except ValueError as exc:
        assert 'Could not list Vals.ai benchmarks' in str(exc)
    else:
        raise AssertionError('expected discovery failure')


def test_view_without_api_artifact_falls_back_to_page(monkeypatch):
    view = _view('aime')
    _routes(
        monkeypatch,
        {
            f'{API}/benchmarks/aime/artifacts': {
                'benchmark': 'aime',
                'artifacts': [{'path': 'results.json', 'type': 'results'}],
            },
            f'{SITE}/benchmarks/aime': (
                '<astro-island props="{&quot;benchmarkViewUrl&quot;:[0,'
                '&quot;/_astro/view.json&quot;]}"></astro-island>'
            ),
            f'{SITE}/_astro/view.json': view,
        },
    )

    payload = adapter.extract_collection(
        benchmark_slugs=['aime'], api_url=API, base_url=SITE
    )

    (benchmark,) = payload['benchmarks']
    assert benchmark['benchmark_view_url'] == f'{SITE}/_astro/view.json'
    assert 'no benchmark_view artifact' in benchmark['api_fallback_reason']


_NO_VIEW_ARTIFACTS = {
    f'{API}/benchmarks/mental-health/artifacts': {
        'benchmark': 'mental-health',
        'artifacts': [
            {'path': 'insights.json', 'type': 'json'},
            {'path': 'results.json', 'type': 'results'},
        ],
    },
}


def test_benchmark_without_view_in_api_or_page_is_excluded(monkeypatch):
    _routes(
        monkeypatch,
        {
            **_NO_VIEW_ARTIFACTS,
            **_api_benchmark('aime', 'benchmark_view_aime.json', _view('aime')),
            f'{SITE}/benchmarks/mental-health': (
                '<astro-island component-url="/_astro/MentalHealth.x.js" '
                'props="{&quot;section&quot;:[0,&quot;details&quot;]}">'
                '</astro-island>'
            ),
        },
    )

    payload = adapter.extract_collection(
        benchmark_slugs=['aime', 'mental-health'], api_url=API, base_url=SITE
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')

    assert len(result.records) == 1
    assert result.failures == []
    (exclusion,) = result.exclusions
    assert exclusion.source_ref == f'{API}/benchmarks/mental-health/artifacts'
    assert 'artifact types: json, results' in exclusion.reason
    assert 'Could not find BenchmarkView data' in exclusion.reason
    assert result.total_records == 2


def test_benchmark_without_api_view_and_unreachable_page_fails(monkeypatch):
    _routes(monkeypatch, _NO_VIEW_ARTIFACTS)

    payload = adapter.extract_collection(
        benchmark_slugs=['mental-health'], api_url=API, base_url=SITE
    )

    assert 'source_exclusions' not in payload
    (failure,) = payload['source_failures']
    assert 'page fallback: unavailable' in failure['reason']


def test_extract_collection_follows_benchmark_view_loader(monkeypatch):
    loader_props = (
        '{&quot;benchmarkViewUrl&quot;:[0,'
        '&quot;/_astro/benchmark_view_aime.abc123.json&quot;],'
        '&quot;benchmarkName&quot;:[0,&quot;AIME&quot;]}'
    )
    page_html = (
        '<astro-island component-url="/_astro/BenchmarkViewLoader.x.js" '
        f'props="{loader_props}"></astro-island>'
    )
    view_json = json.dumps(
        {
            'metadata': {'benchmark': 'AIME', 'slug': 'aime'},
            'tasks': {
                'overall': {'openai/gpt-5': {'accuracy': 95.0, 'harness': None}}
            },
        }
    )
    calls = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        if url == 'https://example.test/benchmarks':
            return '<html></html>'
        if url == 'https://example.test/benchmarks/aime':
            return page_html
        if url == 'https://example.test/_astro/benchmark_view_aime.abc123.json':
            return view_json
        raise AssertionError(f'unexpected fetch: {url}')

    monkeypatch.setattr(adapter, 'fetch_text', fake_fetch)

    payload = adapter.extract_collection(
        benchmark_slugs=['aime'], base_url='https://example.test'
    )

    assert calls[1:] == [
        'https://example.test/benchmarks/aime',
        'https://example.test/_astro/benchmark_view_aime.abc123.json',
    ]
    benchmark = payload['benchmarks'][0]
    assert benchmark['source_url'] == 'https://example.test/benchmarks/aime'
    assert benchmark['benchmark_view_url'] == (
        'https://example.test/_astro/benchmark_view_aime.abc123.json'
    )
    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    assert [bundle.log.model_info.id for bundle in bundles] == ['openai/gpt-5']


def test_inline_benchmark_view_in_any_island_is_read():
    props = (
        '{&quot;benchmarkView&quot;:[0,{&quot;metadata&quot;:[0,'
        '{&quot;benchmark&quot;:[0,&quot;Provider Index&quot;],'
        '&quot;slug&quot;:[0,&quot;inference_provider&quot;]}],'
        '&quot;tasks&quot;:[0,{&quot;overall&quot;:[0,{&quot;kimi/kimi-k3&quot;:'
        '[0,{&quot;accuracy&quot;:[0,71.77]}]}]}]}]}'
    )
    html = (
        '<astro-island component-url="/_astro/ProviderOverview.x.js" '
        f'props="{props}"></astro-island>'
    )

    normalized = adapter.load_benchmark_page(
        html, 'https://www.vals.ai/benchmarks/inference_provider'
    )

    assert normalized['metadata']['slug'] == 'inference_provider'
    assert 'benchmark_view_url' not in normalized


def test_page_without_benchmark_view_is_a_failure(monkeypatch):
    page_html = (
        '<astro-island component-url="/_astro/RsiBenchmarkView.x.js" '
        'props="{&quot;section&quot;:[0,&quot;details&quot;]}">'
        '</astro-island>'
    )
    monkeypatch.setattr(adapter, 'fetch_text', lambda _url: page_html)

    payload = adapter.extract_collection(
        benchmark_slugs=['rsi_index'], base_url='https://example.test'
    )

    assert payload['benchmarks'] == []
    assert (
        'Could not find BenchmarkView data'
        in (payload['source_failures'][0]['reason'])
    )


def test_benchmark_failing_api_and_page_keeps_others(monkeypatch):
    _routes(
        monkeypatch,
        _api_benchmark('aime', 'benchmark_view_aime.json', _view('aime')),
    )

    payload = adapter.extract_collection(
        benchmark_slugs=['aime', 'broken'], api_url=API, base_url=SITE
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='1234567890.0')

    assert len(result.records) == 1
    (failure,) = result.failures
    assert failure.source_ref == f'{API}/benchmarks/broken'
    assert 'API: ' in failure.reason and 'page fallback: ' in failure.reason
    assert failure.source_record == {'benchmark_slug': 'broken'}


def test_real_normalized_fixture_converts_to_schema():
    payload = adapter.extract_collection(input_json=FIXTURE_PATH)
    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')

    assert len(bundles) == 3
    assert any(
        bundle.log.model_info.id == 'openai/gpt-5.4-2026-03-05'
        for bundle in bundles
    )
    assert any(
        bundle.log.model_info.id == 'xai/grok-4-1-fast-non-reasoning'
        for bundle in bundles
    )
    assert any(
        bundle.log.model_info.id == 'ai21labs/jamba-large-1.7'
        for bundle in bundles
    )
    for bundle in bundles:
        EvaluationLog.model_validate(bundle.log.model_dump())


def test_export_paths_validate(tmp_path: Path):
    output_dir = tmp_path / 'data' / 'vals-ai'
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    paths = adapter.export_logs(bundles, output_dir)

    assert len(paths) == 4
    for path in paths:
        assert path.parent.parent.parent == output_dir
        report = validate_file(path)
        assert report.valid, report.errors
