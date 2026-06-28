"""Tests for canonical identity augmentation of legacy datastore payloads."""

from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.augment_canonical_identity import (
    augment_aggregate_file,
    augment_aggregate_payload,
)
from every_eval_ever.check_canonical_identity import check_paths
from every_eval_ever.upgrade_schema_version import (
    upgrade_aggregate_file,
    upgrade_instance_file,
)
from every_eval_ever.validate import validate_file


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


def test_swe_leaderboards_backfill_single_score_metric_identity():
    payload = _base_payload(
        'swe-polybench-leaderboard/org_model/123',
        [
            _base_result(
                'SWE-PolyBench Verified (Python)',
                'Fraction of Python GitHub issues resolved (0.0–1.0)',
            )
        ],
    )

    augmented, changed_results, _, ids_changed = augment_aggregate_payload(
        payload, benchmark_family='swe-polybench-leaderboard'
    )

    result = augmented['evaluation_results'][0]
    metric_config = result['metric_config']

    assert changed_results == 1
    assert ids_changed is True
    assert result['evaluation_name'] == 'SWE-PolyBench Verified (Python)'
    assert metric_config['metric_id'] == 'swe_polybench_leaderboard.score'
    assert metric_config['metric_name'] == 'Score'
    assert metric_config['metric_kind'] == 'score'
    assert metric_config['metric_unit'] == 'proportion'
    assert result['evaluation_result_id'].endswith(
        '#swe_polybench_verified_python#swe_polybench_leaderboard_score'
    )


def test_nested_datastore_paths_are_normalized_and_samples_move(tmp_path: Path):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'multi-swe-bench-leaderboard'
        / 'typescript'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = _base_payload(
        'multi-swe-bench-leaderboard/openai_gpt-5/123',
        [
            _base_result(
                'Multi-SWE-Bench (TypeScript)',
                'Fraction of TypeScript issues resolved (0.0-1.0)',
            )
        ],
    )
    payload['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': 'run_samples.jsonl',
        'total_rows': 0,
    }
    aggregate_path.write_text(json.dumps(payload), encoding='utf-8')

    samples_path = aggregate_path.with_name('run_samples.jsonl')
    samples_path.write_text(
        json.dumps(
            {
                'schema_version': 'instance_level_eval_0.2.2',
                'evaluation_id': payload['evaluation_id'],
                'model_id': 'openai/gpt-5',
                'evaluation_name': 'Multi-SWE-Bench (TypeScript)',
                'sample_id': 'sample-1',
                'interaction_type': 'single_turn',
                'input': {'raw': 'issue', 'reference': ['issue']},
                'output': {'raw': ['patch']},
                'answer_attribution': [],
                'evaluation': {'score': 1.0, 'is_correct': True},
            }
        )
        + '\n',
        encoding='utf-8',
    )

    report = augment_aggregate_file(aggregate_path, write_changes=True)

    expected_aggregate_path = (
        tmp_path
        / 'data'
        / 'multi-swe-bench-leaderboard'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )
    expected_samples_path = expected_aggregate_path.with_name('run_samples.jsonl')

    assert report.path_changed is True
    assert report.file_path == expected_aggregate_path
    assert report.sample_file_path == expected_samples_path
    assert aggregate_path.exists() is False
    assert samples_path.exists() is False
    assert expected_aggregate_path.exists() is True
    assert expected_samples_path.exists() is True

    updated_payload = json.loads(expected_aggregate_path.read_text(encoding='utf-8'))
    updated_result = updated_payload['evaluation_results'][0]
    assert updated_result['metric_config']['metric_id'] == (
        'multi_swe_bench_leaderboard.score'
    )
    assert updated_payload['detailed_evaluation_results']['total_rows'] == 1

    updated_rows = [
        json.loads(line)
        for line in expected_samples_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert updated_rows[0]['evaluation_result_id'] == updated_result['evaluation_result_id']


def test_nested_datastore_paths_report_target_path_in_dry_run(tmp_path: Path):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'swe-polybench-leaderboard'
        / 'pb-verified'
        / 'python'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = _base_payload(
        'swe-polybench-leaderboard/openai_gpt-5/123',
        [
            _base_result(
                'SWE-PolyBench Verified (Python)',
                'Fraction of Python GitHub issues resolved (0.0-1.0)',
            )
        ],
    )
    aggregate_path.write_text(json.dumps(payload), encoding='utf-8')

    report = augment_aggregate_file(aggregate_path, write_changes=False)

    expected_aggregate_path = (
        tmp_path
        / 'data'
        / 'swe-polybench-leaderboard'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )

    assert report.path_changed is True
    assert report.original_file_path == aggregate_path
    assert report.file_path == expected_aggregate_path
    assert aggregate_path.exists() is True
    assert expected_aggregate_path.exists() is False


def test_pr72_style_faulty_data_is_fully_repaired_by_current_workflow(
    tmp_path: Path,
):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'multi-swe-bench-leaderboard'
        / 'typescript'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = _base_payload(
        'multi-swe-bench-leaderboard/openai_gpt-5/123',
        [
            _base_result(
                'Multi-SWE-Bench (TypeScript)',
                'Fraction of TypeScript GitHub issues resolved (0.0-1.0)',
            )
        ],
    )
    aggregate_path.write_text(json.dumps(payload), encoding='utf-8')

    pre_validate = validate_file(aggregate_path)
    pre_audit = check_paths([str(tmp_path / 'data')])
    schema_upgrade = upgrade_aggregate_file(aggregate_path, write_changes=True)
    report = augment_aggregate_file(aggregate_path, write_changes=True)

    fixed_path = (
        tmp_path
        / 'data'
        / 'multi-swe-bench-leaderboard'
        / 'openai'
        / 'gpt-5'
        / 'run.json'
    )
    fixed_payload = json.loads(fixed_path.read_text(encoding='utf-8'))
    fixed_result = fixed_payload['evaluation_results'][0]
    post_validate = validate_file(fixed_path)
    post_audit = check_paths([str(tmp_path / 'data')])

    assert pre_validate.valid is True
    assert pre_audit.has_issues is True
    assert schema_upgrade.changed is False
    assert report.path_changed is True
    assert aggregate_path.exists() is False
    assert fixed_path.exists() is True
    assert fixed_result['metric_config']['metric_id'] == (
        'multi_swe_bench_leaderboard.score'
    )
    assert fixed_result['evaluation_result_id'].endswith(
        '#multi_swe_bench_typescript#multi_swe_bench_leaderboard_score'
    )
    assert post_validate.valid is True
    assert post_audit.has_issues is False


def test_pr57_style_faulty_data_is_fully_repaired_by_current_workflow(
    tmp_path: Path,
):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'swe-bench-verified-mini'
        / 'google'
        / 'gemini-2.0-flash-001'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = _base_payload(
        'swe-bench-verified-mini/google_gemini-2.0-flash-001/1744346409.0',
        [
            _base_result(
                'mean on inspect_evals/swe_bench_verified_mini for scorer swe_bench_scorer',
                'mean',
            ),
            _base_result(
                'std on inspect_evals/swe_bench_verified_mini for scorer swe_bench_scorer',
                'std',
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
    samples_path.write_text(
        json.dumps(
            {
                'schema_version': '0.2.2',
                'evaluation_id': 'run_samples',
                'model_id': 'google/gemini-2.0-flash-001',
                'evaluation_name': 'inspect_evals/swe_bench_verified_mini',
                'evaluation_result_id': None,
                'sample_id': 'django__django-11790',
                'sample_hash': 'abc123',
                'interaction_type': 'agentic',
                'input': {'raw': 'issue', 'reference': ['issue']},
                'output': None,
                'messages': [
                    {'turn_idx': 0, 'role': 'user', 'content': 'issue'},
                    {
                        'turn_idx': 1,
                        'role': 'assistant',
                        'content': 'attempted patch',
                    },
                ],
                'answer_attribution': [
                    {
                        'turn_idx': 1,
                        'source': 'messages[1].content',
                        'extracted_value': 'attempted patch',
                        'extraction_method': 'exact_match',
                        'is_terminal': True,
                    }
                ],
                'evaluation': {
                    'score': 0.0,
                    'is_correct': False,
                    'num_turns': 2,
                    'tool_calls_count': 0,
                },
            }
        )
        + '\n',
        encoding='utf-8',
    )

    pre_sample_validate = validate_file(samples_path)
    pre_audit = check_paths([str(tmp_path / 'data')])
    schema_upgrade = upgrade_instance_file(samples_path, write_changes=True)
    report = augment_aggregate_file(aggregate_path, write_changes=True)

    fixed_payload = json.loads(aggregate_path.read_text(encoding='utf-8'))
    mean_result, std_result = fixed_payload['evaluation_results']
    fixed_rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    fixed_row = fixed_rows[0]
    post_sample_validate = validate_file(samples_path)
    post_audit = check_paths([str(tmp_path / 'data')])

    assert pre_sample_validate.valid is False
    assert pre_audit.has_issues is True
    assert schema_upgrade.changed is True
    assert report.path_changed is False
    assert mean_result['evaluation_name'] == 'swe-bench-verified-mini'
    assert mean_result['metric_config']['metric_id'] == 'mean_score'
    assert mean_result['metric_config']['metric_name'] == 'Mean Score'
    assert mean_result['metric_config']['metric_kind'] == 'score'
    assert std_result['evaluation_name'] == 'swe-bench-verified-mini'
    assert std_result['metric_config']['metric_id'] == 'standard_deviation'
    assert std_result['metric_config']['metric_name'] == 'Standard Deviation'
    assert std_result['metric_config']['metric_kind'] == 'standard_deviation'
    assert fixed_row['schema_version'] == 'instance_level_eval_0.2.2'
    assert fixed_row['evaluation_id'] == fixed_payload['evaluation_id']
    assert fixed_row['evaluation_name'] == 'swe-bench-verified-mini'
    assert fixed_row['evaluation_result_id'] == mean_result['evaluation_result_id']
    assert post_sample_validate.valid is True
    assert post_audit.has_issues is False


def test_alphaxiv_faulty_data_is_fully_repaired_by_current_workflow(
    tmp_path: Path,
):
    aggregate_path = (
        tmp_path
        / 'data'
        / 'alphaxiv'
        / 'Gaia2'
        / 'unknown'
        / 'GPT-4o'
        / 'run.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    payload = {
        'schema_version': '0.2.0',
        'evaluation_id': 'Gaia2/GPT-4o/1771591481.616601',
        'retrieved_timestamp': '1771591481.616601',
        'source_metadata': {
            'source_name': 'alphaXiv State of the Art',
            'source_type': 'documentation',
            'source_organization_name': 'alphaXiv',
            'evaluator_relationship': 'third_party',
        },
        'model_info': {'id': 'GPT-4o', 'name': 'GPT-4o', 'developer': 'unknown'},
        'evaluation_results': [
            {
                'evaluation_name': 'Overall Performance on Gaia2',
                'source_data': {
                    'dataset_name': 'Gaia2',
                    'source_type': 'url',
                    'url': ['https://www.alphaxiv.org/abs/example'],
                },
                'metric_config': {
                    'lower_is_better': False,
                    'score_type': 'continuous',
                    'min_score': 0.0,
                    'max_score': 100.0,
                    'evaluation_description': 'Overall performance on Gaia2.',
                    'additional_details': {
                        'alphaxiv_is_primary': 'True',
                        'alphaxiv_y_axis': 'Score',
                    },
                },
                'score_details': {'score': 67.4},
            }
        ],
    }
    aggregate_path.write_text(json.dumps(payload), encoding='utf-8')

    pre_validate = validate_file(aggregate_path)
    pre_audit = check_paths([str(tmp_path / 'data')])
    schema_upgrade = upgrade_aggregate_file(aggregate_path, write_changes=True)
    report = augment_aggregate_file(aggregate_path, write_changes=True)

    fixed_path = (
        tmp_path / 'data' / 'alphaxiv' / 'unknown' / 'GPT-4o' / 'run.json'
    )
    fixed_payload = json.loads(fixed_path.read_text(encoding='utf-8'))
    fixed_result = fixed_payload['evaluation_results'][0]
    post_validate = validate_file(fixed_path)
    post_audit = check_paths([str(tmp_path / 'data')])

    assert pre_validate.valid is False
    assert pre_audit.has_issues is True
    assert schema_upgrade.changed is True
    assert report.path_changed is True
    assert aggregate_path.exists() is False
    assert fixed_path.exists() is True
    assert fixed_payload['eval_library'] == {
        'name': 'alphaxiv',
        'version': 'unknown',
    }
    assert fixed_result['evaluation_name'] == 'Gaia2'
    assert fixed_result['metric_config']['metric_id'] == (
        'overall_performance_on_gaia2'
    )
    assert fixed_result['metric_config']['metric_name'] == (
        'Overall Performance on Gaia2'
    )
    assert fixed_result['metric_config']['metric_kind'] == 'score'
    assert fixed_result['metric_config']['metric_unit'] == 'points'
    assert (
        fixed_result['metric_config']['additional_details']['raw_evaluation_name']
        == 'Overall Performance on Gaia2'
    )
    assert fixed_result['evaluation_result_id'].endswith(
        '#gaia2#overall_performance_on_gaia2'
    )
    assert post_validate.valid is True
    assert post_audit.has_issues is False


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


def test_ifeval_inspect_rows_split_slice_and_metric_identity():
    payload = _base_payload(
        'IFEval/org_model/123',
        [
            _base_result(
                'prompt_strict_acc on inspect_evals/ifeval for scorer instruction_following',
                'prompt strict accuracy',
            ),
            _base_result(
                'final_stderr on inspect_evals/ifeval for scorer instruction_following',
                'final standard error',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='IFEval'
    )

    strict_acc = augmented['evaluation_results'][0]
    final_stderr = augmented['evaluation_results'][1]

    assert changed_results == 2
    assert strict_acc['evaluation_name'] == 'prompt_strict'
    assert strict_acc['metric_config']['metric_id'] == 'accuracy'
    assert strict_acc['metric_config']['metric_name'] == 'Accuracy'
    assert strict_acc['metric_config']['metric_kind'] == 'accuracy'
    assert final_stderr['evaluation_name'] == 'final'
    assert final_stderr['metric_config']['metric_id'] == 'standard_error'
    assert final_stderr['metric_config']['metric_name'] == 'Standard Error'
    assert final_stderr['metric_config']['metric_kind'] == 'standard_error'


def test_agentharm_inspect_rows_cover_overall_and_category_metrics():
    payload = _base_payload(
        'agentharm/org_model/123',
        [
            _base_result(
                'inspect_evals/avg_full_score on inspect_evals/agentharm for scorer combined_scorer',
                'Average full score on agentharm',
            ),
            _base_result(
                'Cybercrime_avg_refusals on inspect_evals/agentharm for scorer combined_scorer',
                'Average refusals on Cybercrime subset',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='agentharm'
    )

    avg_full_score = augmented['evaluation_results'][0]
    cybercrime_refusals = augmented['evaluation_results'][1]

    assert changed_results == 2
    assert avg_full_score['evaluation_name'] == 'agentharm'
    assert avg_full_score['metric_config']['metric_id'] == 'average_full_score'
    assert avg_full_score['metric_config']['metric_name'] == 'Average Full Score'
    assert avg_full_score['metric_config']['metric_kind'] == 'score'
    assert cybercrime_refusals['evaluation_name'] == 'Cybercrime'
    assert (
        cybercrime_refusals['metric_config']['metric_id']
        == 'average_refusal_rate'
    )
    assert (
        cybercrime_refusals['metric_config']['metric_name']
        == 'Average Refusal Rate'
    )
    assert (
        cybercrime_refusals['metric_config']['metric_kind']
        == 'refusal_rate'
    )


def test_generic_inspect_eval_benchmarks_backfill_accuracy_mean_and_std():
    payload = _base_payload(
        'cyse2_vulnerability_exploit/org_model/123',
        [
            _base_result(
                'accuracy on inspect_evals/cyse2_vulnerability_exploit for scorer vul_exploit_scorer',
                'accuracy',
            ),
            _base_result(
                'mean on inspect_evals/cyse2_vulnerability_exploit for scorer vul_exploit_scorer',
                'mean',
            ),
            _base_result(
                'std on inspect_evals/cyse2_vulnerability_exploit for scorer vul_exploit_scorer',
                'std',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='cyse2_vulnerability_exploit'
    )

    accuracy = augmented['evaluation_results'][0]
    mean_score = augmented['evaluation_results'][1]
    standard_deviation = augmented['evaluation_results'][2]

    assert changed_results == 3
    assert accuracy['evaluation_name'] == 'cyse2_vulnerability_exploit'
    assert accuracy['metric_config']['metric_id'] == 'accuracy'
    assert mean_score['evaluation_name'] == 'cyse2_vulnerability_exploit'
    assert mean_score['metric_config']['metric_id'] == 'mean_score'
    assert mean_score['metric_config']['metric_name'] == 'Mean Score'
    assert mean_score['metric_config']['metric_kind'] == 'score'
    assert standard_deviation['evaluation_name'] == 'cyse2_vulnerability_exploit'
    assert (
        standard_deviation['metric_config']['metric_id']
        == 'standard_deviation'
    )
    assert (
        standard_deviation['metric_config']['metric_name']
        == 'Standard Deviation'
    )
    assert (
        standard_deviation['metric_config']['metric_kind']
        == 'standard_deviation'
    )


def test_fibble_arena_supports_2_to_4_lie_variants():
    payload = _base_payload(
        'fibble_arena/org_model/123',
        [
            _base_result(
                'fibble_arena_3lies_win_rate',
                'Win rate on Fibble³ Arena (3 lies)',
            ),
            _base_result(
                'fibble_arena_4lies_avg_attempts',
                'Average guesses used per game on Fibble⁴ Arena (4 lies)',
                min_score=1.0,
                max_score=8.0,
                lower_is_better=True,
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='fibble_arena'
    )

    win_rate = augmented['evaluation_results'][0]
    avg_attempts = augmented['evaluation_results'][1]

    assert changed_results == 2
    assert win_rate['evaluation_name'] == 'fibble_arena_3lies'
    assert win_rate['metric_config']['metric_id'] == 'win_rate'
    assert avg_attempts['evaluation_name'] == 'fibble_arena_4lies'
    assert avg_attempts['metric_config']['metric_id'] == 'mean_attempts'


def test_fibble_numbered_families_backfill_metrics_without_renaming_family():
    payload = _base_payload(
        'fibble5_arena/org_model/123',
        [
            _base_result(
                'fibble5_arena_win_rate',
                'Win rate on Fibble5 Arena',
            ),
            _base_result(
                'fibble5_arena_avg_attempts',
                'Average attempts on Fibble5 Arena',
                min_score=1.0,
                max_score=8.0,
                lower_is_better=True,
            ),
            _base_result(
                'fibble5_arena_avg_latency_ms',
                'Average latency on Fibble5 Arena',
                min_score=0.0,
                max_score=5000.0,
                lower_is_better=True,
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='fibble5_arena'
    )

    win_rate = augmented['evaluation_results'][0]
    avg_attempts = augmented['evaluation_results'][1]
    avg_latency = augmented['evaluation_results'][2]

    assert changed_results == 3
    assert win_rate['evaluation_name'] == 'fibble5_arena'
    assert win_rate['metric_config']['metric_id'] == 'win_rate'
    assert avg_attempts['evaluation_name'] == 'fibble5_arena'
    assert avg_attempts['metric_config']['metric_id'] == 'mean_attempts'
    assert avg_latency['evaluation_name'] == 'fibble5_arena'
    assert avg_latency['metric_config']['metric_id'] == 'latency_mean'


def test_helm_patch_handles_ndcg_bleu_and_acc_abbreviations():
    payload = _base_payload(
        'helm_classic/org_model/123',
        [
            _base_result(
                'MS MARCO (TREC)',
                'NDCG@10 on MS MARCO (TREC)',
            ),
            _base_result(
                'WMT 2014',
                'BLEU-4 on WMT 2014',
            ),
            _base_result(
                'Omni-MATH',
                'Acc on Omni-MATH',
            ),
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='helm_classic'
    )

    ndcg = augmented['evaluation_results'][0]['metric_config']
    bleu = augmented['evaluation_results'][1]['metric_config']
    acc = augmented['evaluation_results'][2]['metric_config']

    assert changed_results == 3
    assert ndcg['metric_id'] == 'ndcg'
    assert ndcg['metric_parameters'] == {'k': 10}
    assert bleu['metric_id'] == 'bleu_4'
    assert bleu['metric_parameters'] == {'n': 4}
    assert acc['metric_id'] == 'accuracy'


def test_livecodebenchpro_pass_at_k_is_backfilled():
    payload = _base_payload(
        'livecodebenchpro/org_model/123',
        [
            _base_result(
                'Hard Problems',
                'Pass@1 on Hard Problems',
            )
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='livecodebenchpro'
    )

    metric_config = augmented['evaluation_results'][0]['metric_config']
    assert changed_results == 1
    assert metric_config['metric_id'] == 'pass_at_k'
    assert metric_config['metric_name'] == 'Pass@1'
    assert metric_config['metric_kind'] == 'pass_rate'
    assert metric_config['metric_parameters'] == {'k': 1}


def test_score_suffix_families_and_single_score_families_are_backfilled():
    ace_payload = _base_payload(
        'ace/org_model/123',
        [
            _base_result('Overall Score', 'Overall ACE score.'),
            _base_result('Gaming Score', 'Gaming domain score.'),
        ],
    )
    ace_augmented, changed_results, _, _ = augment_aggregate_payload(
        ace_payload, benchmark_family='ace'
    )

    overall = ace_augmented['evaluation_results'][0]
    gaming = ace_augmented['evaluation_results'][1]

    assert changed_results == 2
    assert overall['evaluation_name'] == 'ace'
    assert overall['metric_config']['metric_id'] == 'ace.score'
    assert gaming['evaluation_name'] == 'Gaming'
    assert gaming['metric_config']['metric_id'] == 'ace.score'

    tau_payload = _base_payload(
        'tau-bench-2/airline/org_model/123',
        [
            _base_result(
                'tau-bench-2/airline',
                'Tau Bench 2 benchmark evaluation (airline subset)',
            )
        ],
    )
    tau_augmented, tau_changed, _, _ = augment_aggregate_payload(
        tau_payload, benchmark_family='tau-bench-2_airline'
    )
    tau_metric = tau_augmented['evaluation_results'][0]['metric_config']
    assert tau_changed == 1
    assert tau_metric['metric_id'] == 'tau_bench_2_airline.score'
    assert tau_metric['metric_unit'] == 'proportion'

    la_payload = _base_payload(
        'la_leaderboard/org_model/123',
        [
            _base_result(
                'la_leaderboard',
                'La Leaderboard benchmark score',
                min_score=0.0,
                max_score=100.0,
            )
        ],
    )
    la_augmented, la_changed, _, _ = augment_aggregate_payload(
        la_payload, benchmark_family='la_leaderboard'
    )
    la_metric = la_augmented['evaluation_results'][0]['metric_config']
    assert la_changed == 1
    assert la_metric['metric_id'] == 'la_leaderboard.score'
    assert la_metric['metric_unit'] == 'points'


def test_theory_of_mind_patch_splits_metric_from_eval_name():
    payload = _base_payload(
        'theory_of_mind/org_model/123',
        [
            _base_result(
                'accuracy on theory_of_mind for scorer model_graded_fact',
                'accuracy',
            )
        ],
    )

    augmented, changed_results, _, _ = augment_aggregate_payload(
        payload, benchmark_family='theory_of_mind'
    )

    result = augmented['evaluation_results'][0]
    assert changed_results == 1
    assert result['evaluation_name'] == 'theory_of_mind'
    assert result['metric_config']['metric_id'] == 'accuracy'
