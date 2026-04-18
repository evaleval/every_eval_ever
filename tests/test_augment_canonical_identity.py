"""Tests for canonical identity augmentation of legacy datastore payloads."""

from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.augment_canonical_identity import (
    augment_aggregate_file,
    augment_aggregate_payload,
)


def _base_payload(evaluation_id: str, evaluation_results: list[dict]) -> dict:
    return {
        'schema_version': '0.2.2',
        'evaluation_id': evaluation_id,
        'retrieved_timestamp': '1234567890',
        'source_metadata': {
            'source_type': 'documentation',
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'third_party',
        },
        'eval_library': {'name': 'unknown', 'version': 'unknown'},
        'model_info': {'name': 'test-model', 'id': 'org/test-model'},
        'evaluation_results': evaluation_results,
    }


def _base_result(
    evaluation_name: str,
    description: str,
    *,
    min_score: float = 0.0,
    max_score: float = 1.0,
    lower_is_better: bool = False,
) -> dict:
    return {
        'evaluation_name': evaluation_name,
        'source_data': {
            'dataset_name': 'test-ds',
            'source_type': 'hf_dataset',
            'hf_repo': 'org/test-ds',
        },
        'metric_config': {
            'evaluation_description': description,
            'lower_is_better': lower_is_better,
            'score_type': 'continuous',
            'min_score': min_score,
            'max_score': max_score,
        },
        'score_details': {'score': 0.5},
    }


def test_global_mmlu_lite_backfills_accuracy_metric():
    payload = _base_payload(
        'global-mmlu-lite/org_model/123',
        [
            _base_result(
                'Arabic',
                'Global MMLU Lite - Arabic',
            )
        ],
    )

    augmented, changed_results, _, ids_changed = augment_aggregate_payload(
        payload, benchmark_family='global-mmlu-lite'
    )

    result = augmented['evaluation_results'][0]
    metric_config = result['metric_config']

    assert changed_results == 1
    assert ids_changed is True
    assert result['evaluation_name'] == 'Arabic'
    assert metric_config['metric_id'] == 'accuracy'
    assert metric_config['metric_name'] == 'Accuracy'
    assert metric_config['metric_kind'] == 'accuracy'
    assert metric_config['metric_unit'] == 'proportion'
    assert result['evaluation_result_id'].endswith('#arabic#accuracy')


def test_wordle_arena_updates_aggregate_and_samples(tmp_path: Path):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'wordle_arena'
        / 'anthropic'
        / 'claude-haiku-4.5'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = _base_payload(
        'wordle_arena/anthropic_claude-haiku-4.5/1776347262.820056',
        [
            _base_result(
                'wordle_arena_win_rate',
                'Win rate on Wordle Arena',
            ),
            _base_result(
                'wordle_arena_avg_attempts',
                'Average guesses used per game on Wordle Arena',
                min_score=1.0,
                max_score=6.0,
                lower_is_better=True,
            ),
        ],
    )
    payload['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': 'run_samples.jsonl',
        'total_rows': 1,
    }
    aggregate_path.write_text(json.dumps(payload), encoding='utf-8')

    samples_path = aggregate_path.with_name('run_samples.jsonl')
    sample_rows = [
        {
            'schema_version': 'instance_level_eval_0.2.2',
            'evaluation_id': payload['evaluation_id'],
            'model_id': 'org/test-model',
            'evaluation_name': 'wordle_arena_win_rate',
            'sample_id': 'sample-1',
            'interaction_type': 'single_turn',
            'input': {'raw': 'guess', 'reference': ['guess']},
            'output': {'raw': ['guess']},
            'answer_attribution': [
                {
                    'turn_idx': 0,
                    'source': 'output.raw',
                    'extracted_value': 'guess',
                    'extraction_method': 'exact_match',
                    'is_terminal': True,
                }
            ],
            'evaluation': {'score': 1.0, 'is_correct': True},
        },
        {
            'schema_version': 'instance_level_eval_0.2.2',
            'evaluation_id': payload['evaluation_id'],
            'model_id': 'org/test-model',
            'evaluation_name': 'wordle_arena_win_rate',
            'sample_id': 'sample-2',
            'interaction_type': 'single_turn',
            'input': {'raw': 'guess', 'reference': ['guess']},
            'output': {'raw': ['guess']},
            'answer_attribution': [
                {
                    'turn_idx': 0,
                    'source': 'output.raw',
                    'extracted_value': 'guess',
                    'extraction_method': 'exact_match',
                    'is_terminal': True,
                }
            ],
            'evaluation': {'score': 0.0, 'is_correct': False},
        },
    ]
    samples_path.write_text(
        '\n'.join(json.dumps(row) for row in sample_rows) + '\n',
        encoding='utf-8',
    )

    report = augment_aggregate_file(aggregate_path, write_changes=True)

    assert report.aggregate_changed is True
    assert report.sample_changed is True

    updated_payload = json.loads(aggregate_path.read_text(encoding='utf-8'))
    win_rate = updated_payload['evaluation_results'][0]
    avg_attempts = updated_payload['evaluation_results'][1]

    assert win_rate['evaluation_name'] == 'wordle_arena'
    assert win_rate['metric_config']['metric_id'] == 'win_rate'
    assert (
        win_rate['metric_config']['additional_details']['raw_evaluation_name']
        == 'wordle_arena_win_rate'
    )
    assert avg_attempts['evaluation_name'] == 'wordle_arena'
    assert avg_attempts['metric_config']['metric_id'] == 'mean_attempts'
    assert updated_payload['detailed_evaluation_results']['total_rows'] == 2

    updated_rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert all(row['evaluation_name'] == 'wordle_arena' for row in updated_rows)
    assert all(
        row['evaluation_result_id'] == win_rate['evaluation_result_id']
        for row in updated_rows
    )


def test_apex_agents_splits_metric_semantics_from_evaluation_name():
    payload = _base_payload(
        'apex-agents/org_model/123',
        [
            _base_result(
                'Overall Pass@8',
                'Overall Pass@8 (dataset card / paper snapshot).',
            ),
            _base_result(
                'Investment Banking Mean Score',
                'Investment banking world mean rubric score.',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='apex-agents'
    )

    pass_at_8 = augmented['evaluation_results'][0]
    mean_score = augmented['evaluation_results'][1]

    assert changed_results == 2
    assert pass_at_8['evaluation_name'] == 'Overall'
    assert pass_at_8['metric_config']['metric_id'] == 'pass_at_k'
    assert pass_at_8['metric_config']['metric_parameters'] == {'k': 8}
    assert mean_score['evaluation_name'] == 'Investment Banking'
    assert mean_score['metric_config']['metric_id'] == 'mean_score'


def test_bfcl_uses_metric_id_to_restore_eval_slice():
    payload = _base_payload(
        'bfcl/org_model/123',
        [
            {
                **_base_result(
                    'bfcl.multi_turn.accuracy',
                    'Multi-turn accuracy',
                    max_score=100.0,
                ),
                'metric_config': {
                    'metric_id': 'bfcl.multi_turn.accuracy',
                    'metric_name': 'Accuracy',
                    'lower_is_better': False,
                    'score_type': 'continuous',
                    'min_score': 0.0,
                    'max_score': 100.0,
                },
            }
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='bfcl'
    )

    result = augmented['evaluation_results'][0]
    assert changed_results == 1
    assert result['evaluation_name'] == 'multi_turn'
    assert result['metric_config']['metric_id'] == 'bfcl.multi_turn.accuracy'
    assert (
        result['metric_config']['additional_details']['raw_evaluation_name']
        == 'bfcl.multi_turn.accuracy'
    )


def test_helm_description_drives_metric_identity_and_mean_row_renaming():
    payload = _base_payload(
        'helm_lite/org_model/123',
        [
            _base_result(
                'Mean win rate',
                'How many models this model outperforms on average (over columns).',
            ),
            _base_result(
                'NarrativeQA',
                'F1 on NarrativeQA',
            ),
            _base_result(
                'IFEval',
                'IFEval Strict Acc on IFEval',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='helm_lite'
    )

    mean_win_rate = augmented['evaluation_results'][0]
    narrative_qa = augmented['evaluation_results'][1]
    ifeval = augmented['evaluation_results'][2]

    assert changed_results == 3
    assert mean_win_rate['evaluation_name'] == 'helm_lite'
    assert mean_win_rate['metric_config']['metric_id'] == 'win_rate'
    assert narrative_qa['evaluation_name'] == 'NarrativeQA'
    assert narrative_qa['metric_config']['metric_id'] == 'f1'
    assert ifeval['metric_config']['metric_id'] == 'strict_accuracy'
