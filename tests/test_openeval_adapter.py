from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog
from every_eval_ever.validate import validate_file
from utils.openeval import adapter


def sample_payload() -> dict:
    return {
        'bench': [
            {
                'benchmark_name': 'ifeval',
                'benchmark_version': 'v1',
                'paper_url': 'https://arxiv.org/abs/2311.07911',
                'dataset_url': 'https://huggingface.co/datasets/google/IFEval',
                'benchmark_tags': ['instruction following'],
            },
            {
                'benchmark_name': 'mmlu_pro',
                'benchmark_version': '',
                'paper_url': 'https://neurips.cc/virtual/2024/poster/97435',
                'dataset_url': 'https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro',
                'benchmark_tags': ['knowledge', 'multiple_choice'],
            },
        ],
        'item': [
            {
                'item_id': 'ifeval_20260305T211125Z_0',
                'schema_version': '1.0',
                'item_metadata': {'source': 'ifeval'},
                'item_content': {
                    'input': ['Write one sentence.'],
                    'references': ['{"output":{"text":"A sentence."}}'],
                },
            },
            {
                'item_id': 'ifeval_20260305T211125Z_1',
                'schema_version': '1.0',
                'item_metadata': {'source': 'ifeval'},
                'item_content': {
                    'input': ['Write two words.'],
                    'references': ['{"output":{"text":"two words"}}'],
                },
            },
            {
                'item_id': 'mmlu-pro_20260305T211125Z_0',
                'schema_version': '1.0',
                'item_metadata': {'source': 'mmlu_pro'},
                'item_content': {
                    'input': ['Which answer is correct?'],
                    'references': ['{"output":{"text":"B"}}'],
                },
            },
        ],
        'response': [
            {
                'response_id': 'ifeval_20260305T211125Z_0_gemma-2b-it_0',
                'model': {
                    'name': 'gemma-2b-it',
                    'size': '2b',
                    'model_adaptation': {
                        'generation_parameters': (
                            '{"temperature": 0.0, "top_p": 1, '
                            '"top_k": 1, "max_tokens": 4096, '
                            '"do_sample": false}'
                        )
                    },
                },
                'scores': {
                    'metric': [{'name': 'ifeval_strict_accuracy'}],
                    'value': [0.0],
                },
                'response_content': ['{"text":"One sentence."}'],
                'item_adaptation': {'request_input': ['Write one sentence.']},
            },
            {
                'response_id': 'ifeval_20260305T211125Z_1_gemma-2b-it_0',
                'model': {
                    'name': 'gemma-2b-it',
                    'size': '2b',
                    'model_adaptation': {
                        'generation_parameters': (
                            '{"temperature": 0.0, "top_p": 1, '
                            '"top_k": 1, "max_tokens": 4096, '
                            '"do_sample": false}'
                        )
                    },
                },
                'scores': {
                    'metric': [{'name': 'ifeval_strict_accuracy'}],
                    'value': [1.0],
                },
                'response_content': ['two words'],
                'item_adaptation': {'request_input': ['Write two words.']},
            },
            {
                'response_id': 'mmlu-pro_20260305T211125Z_0_gpt-4o_0',
                'model': {
                    'name': 'gpt-4o',
                    'model_adaptation': {
                        'generation_parameters': '{"temperature": 0.0}'
                    },
                },
                'scores': {
                    'metric': [
                        {
                            'name': 'accuracy',
                            'models': ['judge-model'],
                            'extra_artifacts': {'type': ['judge_trace']},
                        },
                        {'name': 'macro_f1'},
                    ],
                    'value': [0.75, 0.6],
                },
                'response_content': ['B'],
                'item_adaptation': {
                    'request_input': ['Which answer is correct?']
                },
            },
        ],
    }


def logs_by_model() -> dict[str, EvaluationLog]:
    bundles = adapter.make_logs(
        sample_payload(),
        retrieved_timestamp='1234567890.0',
        revision='test-revision',
    )
    return {bundle.log.model_info.name: bundle.log for bundle in bundles}


def test_make_logs_validate_against_schema():
    logs = logs_by_model()

    assert set(logs) == {'gemma-2b-it', 'gpt-4o'}
    for log in logs.values():
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_name == 'OpenEval'
        assert validated.source_metadata.source_type.value == 'evaluation_run'
        assert (
            validated.source_metadata.source_organization_name
            == 'Human-Centered Eval'
        )
        assert (
            validated.source_metadata.additional_details['hf_repo']
            == 'human-centered-eval/OpenEval'
        )
        assert (
            validated.source_metadata.additional_details['partial_export']
            == 'false'
        )


def test_scores_are_aggregated_by_model_benchmark_and_metric():
    gemma = logs_by_model()['gemma-2b-it']

    assert gemma.model_info.developer == 'google'
    assert gemma.model_info.id == 'google/gemma-2b-it'
    assert len(gemma.evaluation_results) == 1

    result = gemma.evaluation_results[0]
    assert result.evaluation_name == 'openeval.ifeval.ifeval-strict-accuracy'
    assert result.metric_config.metric_id == (
        'openeval.ifeval.ifeval-strict-accuracy'
    )
    assert result.score_details.score == 0.5
    assert result.score_details.uncertainty.num_samples == 2
    assert result.metric_config.min_score == 0.0
    assert result.metric_config.max_score == 1.0
    assert result.metric_config.metric_unit == 'proportion'
    assert (
        result.metric_config.additional_details['score_values_are_binary']
        == 'true'
    )
    assert result.source_data.source_type == 'hf_dataset'
    assert result.source_data.hf_repo == 'human-centered-eval/OpenEval'
    assert result.source_data.dataset_name == 'ifeval'


def test_benchmark_prefix_matching_handles_underscores():
    gpt = logs_by_model()['gpt-4o']

    assert len(gpt.evaluation_results) == 2
    names = {result.evaluation_name for result in gpt.evaluation_results}
    assert names == {
        'openeval.mmlu-pro.accuracy',
        'openeval.mmlu-pro.macro-f1',
    }
    for result in gpt.evaluation_results:
        assert result.source_data.dataset_name == 'mmlu_pro'

    accuracy = {
        result.metric_config.metric_name: result
        for result in gpt.evaluation_results
    }['accuracy']
    assert (
        accuracy.metric_config.additional_details['metric_models_json']
        == '["judge-model"]'
    )
    assert (
        accuracy.metric_config.additional_details['extra_artifact_types_json']
        == '["judge_trace"]'
    )
    assert accuracy.metric_config.metric_unit == 'score'
    assert (
        accuracy.metric_config.additional_details['score_values_are_binary']
        == 'false'
    )
    assert accuracy.metric_config.additional_details['bounds_source'] == (
        'normalized_observed_values'
    )


def test_generation_config_hash_is_preserved():
    log = logs_by_model()['gemma-2b-it']

    assert '/default/' not in log.evaluation_id
    result = log.evaluation_results[0]
    assert result.generation_config.additional_details['generation_config_hash']


def test_extract_collection_accepts_hf_rows_shape():
    rows = {
        'rows': [
            {'row_idx': 0, 'row': {'benchmark_name': 'ifeval'}},
            {'row_idx': 1, 'row': {'benchmark_name': 'mmlu_pro'}},
        ]
    }

    assert adapter.extract_collection(rows, 'bench') == [
        {'benchmark_name': 'ifeval'},
        {'benchmark_name': 'mmlu_pro'},
    ]


def test_partial_exports_are_marked_in_source_metadata():
    payload = sample_payload() | {
        'source_metadata': {
            'hf_commit': 'abc123',
            'downloaded_response_shards': 1,
            'total_response_shards': 13,
        }
    }

    bundles = adapter.make_logs(
        payload,
        retrieved_timestamp='1234567890.0',
    )

    details = bundles[0].log.source_metadata.additional_details
    assert details['hf_commit'] == 'abc123'
    assert details['partial_export'] == 'true'
    assert details['downloaded_response_shards'] == '1'
    assert details['total_response_shards'] == '13'


def test_unknown_benchmark_ids_raise_by_default():
    payload = {
        'bench': [{'benchmark_name': 'ifeval'}],
        'response': [
            {
                'response_id': 'newbench_20260305T211125Z_0_gpt-4o_0',
                'model': {'name': 'gpt-4o'},
                'scores': {
                    'metric': [{'name': 'accuracy'}],
                    'value': [1.0],
                },
            }
        ],
    }

    try:
        adapter.make_logs(payload, retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        assert 'Could not match OpenEval response_id' in str(exc)
    else:
        raise AssertionError('Expected unmatched benchmark to raise')


def test_export_paths_follow_datastore_layout(tmp_path: Path):
    output_dir = tmp_path / 'data' / 'openeval'
    bundles = adapter.make_logs(
        sample_payload(), retrieved_timestamp='1234567890.0'
    )
    paths = adapter.export_logs(bundles, output_dir)

    assert len(paths) == 2
    for path in paths:
        assert path.suffix == '.json'
        assert path.parent.parent.parent == output_dir
        report = validate_file(path)
        assert report.valid, report.errors

    assert (output_dir / 'google' / 'gemma-2b-it').is_dir()
    assert (output_dir / 'openai' / 'gpt-4o').is_dir()


def test_include_instances_writes_valid_jsonl_sidecar(tmp_path: Path):
    output_dir = tmp_path / 'data' / 'openeval'
    bundles = adapter.make_logs(
        sample_payload(),
        retrieved_timestamp='1234567890.0',
        include_instances=True,
    )

    gemma_bundle = next(
        bundle
        for bundle in bundles
        if bundle.log.model_info.name == 'gemma-2b-it'
    )
    assert gemma_bundle.instance_count == 2
    assert gemma_bundle.instance_path is not None
    assert (
        gemma_bundle.log.source_metadata.additional_details['include_instances']
        == 'true'
    )
    first = adapter.pending_instance_from_json(
        gemma_bundle.instance_path.read_text(encoding='utf-8').splitlines()[0]
    )
    first_log = adapter.make_instance_log(
        first,
        gemma_bundle.log.evaluation_id,
        gemma_bundle.log.model_info.id,
        gemma_bundle.binary_result_ids,
    )
    assert first_log.evaluation_id == gemma_bundle.log.evaluation_id
    assert first_log.evaluation_result_id == 'ifeval::ifeval-strict-accuracy'
    assert first_log.evaluation.is_correct is False
    assert first_log.metadata['is_correct_applicable'] == 'true'
    assert first.sample_id == 'ifeval_20260305T211125Z_0'
    assert first.raw_input == 'Write one sentence.'
    assert first.references == ['A sentence.']
    assert first.output == ['One sentence.']

    paths = adapter.export_logs([gemma_bundle], output_dir)
    aggregate_path = paths[0]
    sample_path = aggregate_path.with_name(
        f'{aggregate_path.stem}_samples.jsonl'
    )
    assert sample_path.exists()

    aggregate = EvaluationLog.model_validate(
        json.loads(aggregate_path.read_text(encoding='utf-8'))
    )
    assert aggregate.detailed_evaluation_results is not None
    assert aggregate.detailed_evaluation_results.total_rows == 2
    assert aggregate.detailed_evaluation_results.format.value == 'jsonl'
    assert aggregate.detailed_evaluation_results.file_path == sample_path.name
    assert aggregate.detailed_evaluation_results.checksum

    report = validate_file(sample_path)
    assert report.valid, report.errors
    rows = [
        InstanceLevelEvaluationLog.model_validate(json.loads(line))
        for line in sample_path.read_text(encoding='utf-8').splitlines()
    ]
    assert {row.evaluation_id for row in rows} == {aggregate.evaluation_id}


def test_include_instances_requires_item_records():
    payload = sample_payload()
    payload.pop('item')

    try:
        adapter.make_logs(
            payload,
            retrieved_timestamp='1234567890.0',
            include_instances=True,
        )
    except ValueError as exc:
        assert 'requires item records' in str(exc)
    else:
        raise AssertionError('Expected missing item records to raise')


def test_duplicate_response_metrics_are_deduped():
    payload = sample_payload()
    payload['response'].append(dict(payload['response'][0]))

    bundles = adapter.make_logs(
        payload,
        retrieved_timestamp='1234567890.0',
        include_instances=True,
    )
    gemma_bundle = next(
        bundle
        for bundle in bundles
        if bundle.log.model_info.name == 'gemma-2b-it'
    )

    assert gemma_bundle.instance_count == 2
    result = gemma_bundle.log.evaluation_results[0]
    assert result.score_details.uncertainty.num_samples == 2
    assert result.source_data.samples_number == 2


def test_non_binary_instance_scores_do_not_claim_correctness():
    bundles = adapter.make_logs(
        sample_payload(),
        retrieved_timestamp='1234567890.0',
        include_instances=True,
    )
    gpt_bundle = next(
        bundle for bundle in bundles if bundle.log.model_info.name == 'gpt-4o'
    )
    rows = [
        adapter.pending_instance_from_json(line)
        for line in gpt_bundle.instance_path.read_text(
            encoding='utf-8'
        ).splitlines()
    ]
    accuracy = next(row for row in rows if row.metric_name == 'accuracy')

    instance = adapter.make_instance_log(
        accuracy,
        gpt_bundle.log.evaluation_id,
        gpt_bundle.log.model_info.id,
        gpt_bundle.binary_result_ids,
    )

    assert instance.evaluation.score == 0.75
    assert instance.evaluation.is_correct is False
    assert instance.metadata['is_correct_applicable'] == 'false'
