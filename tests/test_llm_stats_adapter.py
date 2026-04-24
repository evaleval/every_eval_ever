from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file
from utils.llm_stats import adapter


def sample_payload() -> dict:
    return {
        'models': {
            'data': [
                {
                    'id': 'gpt-5',
                    'slug': 'gpt-5',
                    'name': 'GPT-5',
                    'provider': {'slug': 'openai', 'name': 'OpenAI'},
                    'context_window': 128000,
                    'modalities': ['text'],
                    'pricing': {'input': 1.25, 'output': 10.0},
                    'release_date': '2025-08-07',
                    'license': 'proprietary',
                },
                {
                    'id': 'claude-4-opus',
                    'slug': 'claude-4-opus',
                    'name': 'Claude 4 Opus',
                    'provider': {'slug': 'anthropic', 'name': 'Anthropic'},
                    'context_window': 200000,
                    'modalities': ['text', 'vision'],
                },
            ]
        },
        'benchmarks': {
            'data': [
                {
                    'id': 'gpqa-diamond',
                    'slug': 'gpqa-diamond',
                    'name': 'GPQA Diamond',
                    'description': 'Graduate-level science QA.',
                    'category': 'reasoning',
                    'min_score': 0,
                    'max_score': 100,
                    'metric_kind': 'accuracy',
                    'metric_unit': 'percent',
                },
                {
                    'id': 'math-500',
                    'slug': 'math-500',
                    'name': 'MATH-500',
                    'description': 'Competition mathematics benchmark.',
                    'category': 'math',
                    'min_score': 0,
                    'max_score': 1,
                    'metric_kind': 'accuracy',
                    'metric_unit': 'proportion',
                },
            ]
        },
        'scores': {
            'data': [
                {
                    'id': 'score-gpt5-gpqa',
                    'model_id': 'gpt-5',
                    'benchmark_id': 'gpqa-diamond',
                    'score': 94.2,
                    'unit': 'percent',
                    'source_type': 'model_card',
                    'verified': True,
                    'source_url': 'https://openai.com/index/gpt-5-system-card/',
                },
                {
                    'id': 'score-gpt5-math',
                    'model_id': 'gpt-5',
                    'benchmark_id': 'math-500',
                    'score': 0.91,
                    'unit': 'proportion',
                    'provenance': 'independent_runner',
                    'verification_tier': 'third_party',
                    'citation_url': 'https://example.org/independent-math-500',
                },
                {
                    'id': 'score-claude-gpqa',
                    'model_id': 'claude-4-opus',
                    'benchmark_id': 'gpqa-diamond',
                    'score': 88.5,
                    'source_url': 'https://example.org/claude-gpqa',
                },
            ]
        },
    }


def logs_by_relationship() -> dict[str, EvaluationLog]:
    bundles = adapter.make_logs(
        sample_payload(),
        base_url=adapter.DEFAULT_BASE_URL,
        retrieved_timestamp='1234567890.0',
    )
    logs = {
        bundle.log.source_metadata.evaluator_relationship.value: bundle.log
        for bundle in bundles
    }
    assert set(logs) == {'first_party', 'third_party', 'other'}
    return logs


def test_make_logs_validate_against_schema():
    for log in logs_by_relationship().values():
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_organization_name == 'LLM Stats'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert (
            validated.source_metadata.additional_details['attribution_required']
            == 'true'
        )
        assert (
            validated.source_metadata.additional_details['scores_endpoint']
            == 'https://api.llm-stats.com/v1/scores'
        )
        assert (
            validated.source_metadata.additional_details[
                'scores_endpoint_fallback'
            ]
            == 'https://api.llm-stats.com/leaderboard/benchmarks/{benchmark_id}'
        )


def test_scores_are_grouped_by_evaluator_relationship():
    logs = logs_by_relationship()

    first_party = logs['first_party']
    assert first_party.model_info.id == 'openai/gpt-5'
    assert first_party.evaluation_id.startswith(
        'llm-stats/first_party/openai_gpt-5/'
    )
    assert len(first_party.evaluation_results) == 1
    assert (
        first_party.evaluation_results[0].evaluation_name
        == 'llm_stats.gpqa-diamond'
    )

    third_party = logs['third_party']
    assert third_party.model_info.id == 'openai/gpt-5'
    assert len(third_party.evaluation_results) == 1
    assert (
        third_party.evaluation_results[0].evaluation_name
        == 'llm_stats.math-500'
    )

    other = logs['other']
    assert other.model_info.id == 'anthropic/claude-4-opus'
    assert other.source_metadata.evaluator_relationship.value == 'other'


def test_raw_citation_and_provenance_are_preserved():
    logs = logs_by_relationship()

    first_result = logs['first_party'].evaluation_results[0]
    first_details = first_result.score_details.details or {}
    assert first_details['raw_provenance_label'] == 'model_card'
    assert first_details['raw_verified'] == 'true'
    assert 'https://openai.com/index/gpt-5-system-card/' in json.loads(
        first_details['source_urls_json']
    )
    assert (
        'https://openai.com/index/gpt-5-system-card/'
        in first_result.source_data.url
    )
    assert first_result.metric_config.metric_unit == 'percent'
    assert first_result.metric_config.max_score == 100

    other_result = logs['other'].evaluation_results[0]
    other_details = other_result.score_details.details or {}
    assert other_details['raw_provenance_label'] == 'unknown'


def test_export_paths_follow_datastore_layout(tmp_path: Path):
    output_dir = tmp_path / 'data' / 'llm-stats'
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    paths = adapter.export_logs(bundles, output_dir)

    assert len(paths) == 3
    for path in paths:
        assert path.suffix == '.json'
        assert path.parent.parent.parent == output_dir
        report = validate_file(path)
        assert report.valid, report.errors

    assert (output_dir / 'openai' / 'gpt-5').is_dir()
    assert (output_dir / 'anthropic' / 'claude-4-opus').is_dir()


def test_scores_from_live_benchmark_detail_shape():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'description': 'Graduate-level science questions.',
        'max_score': 1.0,
        'models': [
            {
                'rank': 1,
                'model_id': 'gpt-5.5',
                'model_name': 'GPT-5.5',
                'organization_id': 'openai',
                'organization_name': 'OpenAI',
                'score': 0.936,
                'normalized_score': 0.936,
                'verified': False,
                'self_reported': True,
                'self_reported_source': 'https://openai.com/index/introducing-gpt-5-5/',
                'analysis_method': 'GPQA Diamond. Reasoning effort xhigh.',
            }
        ],
    }

    scores = adapter.scores_from_benchmark_detail(detail)

    assert len(scores) == 1
    assert scores[0]['benchmark']['benchmark_id'] == 'gpqa'
    assert scores[0]['model_id'] == 'gpt-5.5'
    assert (
        scores[0]['source_url']
        == 'https://openai.com/index/introducing-gpt-5-5/'
    )
    assert adapter.relationship_from_score(scores[0]) == 'first_party'


def test_scores_from_live_benchmark_detail_handles_empty_model_id():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'models': [
            {
                'model_id': None,
                'score': 0.936,
            }
        ],
    }

    scores = adapter.scores_from_benchmark_detail(detail)

    assert scores[0]['id'] == 'gpqa::unknown'


def test_live_benchmark_scores_preserve_score_level_organization():
    detail = {
        'benchmark_id': 'gpqa',
        'name': 'GPQA',
        'models': [
            {
                'model_id': 'gpt-5.5',
                'model_name': 'GPT-5.5',
                'organization_id': 'openai',
                'organization_name': 'OpenAI',
                'score': 0.936,
                'self_reported': True,
                'self_reported_source': 'https://openai.com/index/introducing-gpt-5-5/',
            }
        ],
    }
    payload = {
        'models': [],
        'benchmarks': [],
        'scores': adapter.scores_from_benchmark_detail(detail),
    }

    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')

    assert len(bundles) == 1
    assert bundles[0].developer == 'openai'
    assert bundles[0].model == 'gpt-5.5'
    assert bundles[0].log.model_info.id == 'openai/gpt-5.5'


def test_relationship_accepts_canonical_values_from_provenance_keys():
    assert (
        adapter.relationship_from_score({'relationship': 'collaborative'})
        == 'collaborative'
    )
    assert (
        adapter.relationship_from_score({'verification_tier': 'third_party'})
        == 'third_party'
    )


def test_unknown_source_urls_fall_back_to_attribution_url():
    payload = {
        'models': [],
        'benchmarks': [],
        'scores': [{'score': 0.5}],
    }

    bundles = adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    result = bundles[0].log.evaluation_results[0]

    assert result.source_data.url == [adapter.ATTRIBUTION_URL]
    EvaluationLog.model_validate(bundles[0].log.model_dump())
