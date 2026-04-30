from __future__ import annotations

from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file
from utils.vals_ai import adapter

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
                        'unknown-model': {
                            'accuracy': None,
                            'provider': 'Mystery Lab',
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
    assert len(bundles) == 2

    for bundle in bundles:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == '0.2.2'
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


def test_non_percent_scores_are_skipped_without_explicit_bounds():
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    model_ids = {bundle.log.model_info.id for bundle in bundles}

    assert 'anthropic/claude-opus-4-7' not in model_ids
    assert 'anthropic/claude-sonnet-4-6' not in model_ids


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


def test_extract_collection_fetches_index_and_benchmark_pages(monkeypatch):
    page_props = (
        '{&quot;benchmarkView&quot;:[0,{&quot;metadata&quot;:[0,'
        '{&quot;benchmark&quot;:[0,&quot;AIME&quot;],&quot;slug&quot;:[0,&quot;aime&quot;]}],'
        '&quot;tasks&quot;:[0,{&quot;overall&quot;:[0,{&quot;openai/gpt-5&quot;:'
        '[0,{&quot;accuracy&quot;:[0,95.0]}]}]}]}]}'
    )
    page_html = (
        '<astro-island component-url="/_astro/BenchmarkView.abc.js" '
        f'props="{page_props}"></astro-island>'
    )
    calls = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        if url == 'https://example.test/benchmarks':
            return '<a href="/benchmarks/aime">AIME</a>'
        if url == 'https://example.test/benchmarks/aime':
            return page_html
        raise AssertionError(f'unexpected fetch: {url}')

    monkeypatch.setattr(adapter, 'fetch_text', fake_fetch)

    payload = adapter.extract_collection(base_url='https://example.test')

    assert calls == [
        'https://example.test/benchmarks',
        'https://example.test/benchmarks/aime',
    ]
    assert payload['benchmarks'][0]['metadata']['benchmark'] == 'AIME'


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

    assert len(paths) == 2
    for path in paths:
        assert path.parent.parent.parent == output_dir
        report = validate_file(path)
        assert report.valid, report.errors
