"""Unit tests for the ARC Prize leaderboard adapter."""

from __future__ import annotations

import json

import pytest

from every_eval_ever.adapters.arc_agi import adapter
from every_eval_ever.eval_types import EvaluationLog


def sample_payload() -> dict:
    return {
        'datasets': [
            {'id': 'v1_Semi_Private', 'displayName': 'ARC-AGI-1', 'url': ''},
            {'id': 'v2_Semi_Private', 'displayName': 'ARC-AGI-2', 'url': ''},
        ],
        'providers': [
            {'id': 'Human', 'displayName': 'Human', 'url': ''},
            {'id': 'ARC Prize 2024', 'displayName': 'ARC Prize 2024', 'url': ''},
            {'id': 'Anthropic', 'displayName': 'Anthropic', 'url': ''},
            {'id': 'OpenAI', 'displayName': 'OpenAI', 'url': ''},
            {'id': 'New Lab', 'displayName': 'New Lab', 'url': ''},
        ],
        'models': [
            {
                'id': '2025_human_panel',
                'displayName': 'Human Panel',
                'providerId': 'Human',
                'modelType': None,
                'modelGroup': 'Human',
                'modelReleaseDate': None,
            },
            {
                'id': 'ARChitects',
                'displayName': 'ARChitects',
                'providerId': 'ARC Prize 2024',
                'modelType': 'Custom',
                'modelGroup': 'Kaggle',
            },
            {
                'id': 'anthropic-claude-fable-5-high',
                'displayName': 'Claude Fable 5 (High)',
                'providerId': 'Anthropic',
                'modelType': 'CoT',
                'modelGroup': 'anthropic-claude-fable-5',
                'modelReleaseDate': '2026-05-01',
            },
            {
                'id': 'o4-mini',
                'displayName': 'o4-mini',
                'providerId': 'OpenAI',
                'modelType': 'CoT',
            },
            {
                'id': 'openai-o4-mini',
                'displayName': 'o4-mini',
                'providerId': 'OpenAI',
                'modelType': 'CoT',
            },
            {
                'id': 'shiny-new-model',
                'displayName': 'Shiny New Model',
                'providerId': 'New Lab',
            },
        ],
        'evaluations': [
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': '2025_human_panel',
                'score': 0.98,
                'costPerTask': 17,
                'resultsUrl': '',
                'display': True,
            },
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': 'ARChitects',
                'score': 0.535,
                'cost': 50,
                'resultsUrl': '',
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'anthropic-claude-fable-5-high',
                'score': 0.29,
                'costPerTask': 8.42,
                'resultsUrl': '',
                'display': True,
                'labelOffsetX': 12,
            },
            # Two raw ids that slugify to the same canonical OpenAI model,
            # on two different datasets.
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': 'o4-mini',
                'score': 0.41,
                'costPerTask': 0.23,
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'openai-o4-mini',
                'score': 0.02,
                'costPerTask': 0.31,
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'shiny-new-model',
                'score': 0.05,
                'costPerTask': 1.0,
                'display': True,
            },
            # Hidden rows never convert.
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'hidden-model',
                'score': 0.99,
                'display': False,
            },
        ],
    }


def convert(payload: dict) -> dict[str, EvaluationLog]:
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    result.raise_if_incomplete()
    return {log.model_info.id: log for log, _, _ in result.records}


def test_converts_each_canonical_model_once():
    logs = convert(sample_payload())
    assert sorted(logs) == [
        'anthropic/claude-fable-5-high',
        'arcprize/2025-human-panel',
        'community/architects',
        'new-lab/shiny-new-model',
        'openai/o4-mini',
    ]


def test_developer_comes_from_provider_table():
    logs = convert(sample_payload())
    assert logs['anthropic/claude-fable-5-high'].model_info.developer == 'anthropic'
    # Kaggle-winner systems keep the historical 'community' developer.
    assert logs['community/architects'].model_info.developer == 'community'
    # The human panel keeps the historical 'arcprize' developer.
    assert logs['arcprize/2025-human-panel'].model_info.developer == 'arcprize'
    # An unmapped provider falls back to a slug of its id.
    assert logs['new-lab/shiny-new-model'].model_info.developer == 'new-lab'


def test_aliases_merge_into_one_log():
    logs = convert(sample_payload())
    log = logs['openai/o4-mini']
    aliases = json.loads(
        log.model_info.additional_details['raw_model_aliases_json']
    )
    assert aliases == ['o4-mini', 'openai-o4-mini']
    result_ids = [r.evaluation_result_id for r in log.evaluation_results]
    assert result_ids == [
        'v1_Semi_Private::score',
        'v1_Semi_Private::cost_per_task',
        'v2_Semi_Private::score',
        'v2_Semi_Private::cost_per_task',
    ]


def test_score_and_cost_results_carry_source_fields():
    logs = convert(sample_payload())
    log = logs['anthropic/claude-fable-5-high']
    score_result, cost_result = log.evaluation_results
    assert score_result.evaluation_result_id == 'v2_Semi_Private::score'
    assert score_result.score_details.score == 0.29
    assert score_result.metric_config.max_score == 1.0
    assert (
        score_result.source_data.additional_details['dataset_display_name']
        == 'ARC-AGI-2'
    )
    # Chart-layout fields stay out of the record.
    assert 'labelOffsetX' not in score_result.score_details.details

    assert cost_result.evaluation_result_id == 'v2_Semi_Private::cost_per_task'
    assert cost_result.score_details.score == 8.42
    assert cost_result.metric_config.lower_is_better is True
    # Bounds come from the largest observed value across the payload.
    assert cost_result.metric_config.max_score == 17.0


def test_cost_metric_used_when_cost_per_task_missing():
    logs = convert(sample_payload())
    log = logs['community/architects']
    result_ids = [r.evaluation_result_id for r in log.evaluation_results]
    assert result_ids == ['v1_Semi_Private::score', 'v1_Semi_Private::cost']


def test_model_metadata_from_models_table():
    logs = convert(sample_payload())
    details = logs['anthropic/claude-fable-5-high'].model_info.additional_details
    assert details['source_model_type'] == 'CoT'
    assert details['source_provider_id'] == 'Anthropic'
    assert details['model_release_date'] == '2026-05-01'
    assert (
        logs['anthropic/claude-fable-5-high'].model_info.name
        == 'Claude Fable 5 (High)'
    )


def test_unknown_model_id_is_an_accounted_failure():
    payload = sample_payload()
    payload['evaluations'].append(
        {
            'datasetId': 'v2_Semi_Private',
            'modelId': 'not-in-models-json',
            'score': 0.5,
            'display': True,
        }
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    assert len(result.failures) == 1
    assert 'not in models.json' in result.failures[0].reason
    with pytest.raises(ValueError):
        result.raise_if_incomplete()


def test_out_of_range_score_is_an_accounted_failure():
    payload = sample_payload()
    payload['evaluations'].append(
        {
            'datasetId': 'v2_Semi_Private',
            'modelId': 'o4-mini',
            'score': 1.5,
            'display': True,
        }
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    assert len(result.failures) == 1
    assert 'proportion' in result.failures[0].reason


def test_bare_list_payload_is_rejected(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps([{'modelId': 'x'}]), encoding='utf-8')
    with pytest.raises(ValueError, match='evaluations'):
        adapter.load_payload_file(path)


def test_run_writes_records_and_replays_offline(tmp_path):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps(sample_payload()), encoding='utf-8')
    out_dir = tmp_path / 'out'
    args = adapter.parse_args(
        [
            '--input-json',
            str(payload_path),
            '--output-dir',
            str(out_dir),
        ]
    )
    written = adapter.run(args)
    assert written == 5
    files = sorted(out_dir.glob('*/*/*.json'))
    assert len(files) == 5
    record = json.loads(files[0].read_text(encoding='utf-8'))
    assert record['schema_version']
    assert record['evaluation_id'].startswith('arc-agi/')
