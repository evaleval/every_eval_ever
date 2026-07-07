from __future__ import annotations

import copy
import json

import pytest

from every_eval_ever.dedup import (
    check_duplicates,
    compute_aggregate_identity,
    compute_fingerprint,
)


def _base() -> dict:
    return {
        'schema_version': '0.2.2',
        'evaluation_id': 'gsm8k/llama/111',
        'retrieved_timestamp': '111',
        'evaluation_timestamp': '2025-01-01T00:00:00Z',
        'source_metadata': {
            'source_type': 'evaluation_run',
            'source_organization_name': 'Org',
            'evaluator_relationship': 'third_party',
        },
        'model_info': {
            'id': 'meta-llama/Llama-3.1-8B-Instruct',
            'name': 'Llama',
        },
        'eval_library': {'name': 'lm_eval', 'version': '0.4'},
        'detailed_evaluation_results': {'file_path': 'x.jsonl'},
        'additional_details': {'freeform': 'ignored'},
        'evaluation_results': [
            {
                'evaluation_name': 'GSM8K',
                'source_data': {
                    'source_type': 'hf_dataset',
                    'dataset_name': 'gsm8k',
                    'hf_repo': 'openai/gsm8k',
                },
                'metric_config': {
                    'metric_id': 'accuracy',
                    'metric_kind': 'accuracy',
                    'metric_unit': 'proportion',
                    'score_type': 'continuous',
                    'lower_is_better': False,
                    'metric_parameters': {'k': 1},
                },
                'score_details': {'score': 0.95, 'details': {'n': '500'}},
                'generation_config': {
                    'generation_args': {
                        'temperature': 0.0,
                        'reasoning': False,
                        'max_tokens': 512,
                    }
                },
                'additional_details': {'note': 'ignored'},
            }
        ],
    }


def _first_result(data: dict) -> dict:
    return data['evaluation_results'][0]


def _write(tmp_path, spec: dict[str, dict]) -> dict[str, str]:
    local_paths = {}
    for repo_path, payload in spec.items():
        local_path = tmp_path / repo_path.replace('/', '__')
        local_path.write_text(json.dumps(payload), encoding='utf-8')
        local_paths[repo_path] = str(local_path)
    return local_paths


def test_non_identity_fields_do_not_change_fingerprint():
    baseline = compute_aggregate_identity(_base())

    data = _base()
    data['evaluation_id'] = 'new-id'
    data['retrieved_timestamp'] = '999'
    data['schema_version'] = '0.3.0'
    data['source_metadata'] = {'source_type': 'documentation'}
    data['additional_details'] = {'anything': 'else'}
    _first_result(data)['additional_details'] = {'different': 'freeform'}
    _first_result(data)['score_details']['details'] = {'different': 'freeform'}

    assert compute_aggregate_identity(data) == baseline


def test_identity_fields_change_fingerprint():
    baseline = compute_aggregate_identity(_base())
    cases = []

    changed_score = _base()
    _first_result(changed_score)['score_details']['score'] = 0.96
    cases.append(changed_score)

    changed_temp = _base()
    _first_result(changed_temp)['generation_config']['generation_args'][
        'temperature'
    ] = 0.5
    cases.append(changed_temp)

    changed_model = _base()
    changed_model['model_info']['id'] = 'openai/gpt-4o'
    cases.append(changed_model)

    for case in cases:
        assert compute_aggregate_identity(case) != baseline


def test_results_order_is_invariant():
    data = _base()
    second = copy.deepcopy(_first_result(data))
    second['evaluation_name'] = 'MMLU'
    second['metric_config']['metric_id'] = 'mmlu_accuracy'
    data['evaluation_results'].append(second)
    forward = compute_aggregate_identity(data)
    data['evaluation_results'].reverse()
    assert compute_aggregate_identity(data) == forward


def test_none_results_are_ignored_but_non_object_results_are_rejected():
    data = _base()
    baseline = compute_aggregate_identity(data)

    data_with_none = _base()
    data_with_none['evaluation_results'].append(None)
    assert compute_aggregate_identity(data_with_none) == baseline

    data_with_bad_result = _base()
    data_with_bad_result['evaluation_results'].append('not a result')
    with pytest.raises(ValueError, match='entries must be objects'):
        compute_aggregate_identity(data_with_bad_result)


def test_large_integer_identity_fields_are_not_float_coerced():
    first = _base()
    second = _base()
    first_args = _first_result(first)['generation_config']['generation_args']
    second_args = _first_result(second)['generation_config']['generation_args']
    first_args['max_tokens'] = 2**53
    second_args['max_tokens'] = 2**53 + 1

    assert compute_aggregate_identity(first) != compute_aggregate_identity(
        second
    )


def test_compute_fingerprint_rejects_non_aggregate_json():
    with pytest.raises(ValueError, match='aggregate JSON'):
        compute_fingerprint(b'{"x": 1}')


def test_manifest_and_intra_batch_dedup_are_collection_scoped(tmp_path):
    existing = _base()
    candidate = _base()
    manifest = {
        'files': {
            'data/gsm8k/openai/model/existing.json': {
                'fingerprint': compute_fingerprint(
                    json.dumps(existing).encode()
                )
            },
            'data/other/openai/model/existing.json': {
                'fingerprint': compute_fingerprint(
                    json.dumps(existing).encode()
                )
            },
        }
    }
    local = _write(
        tmp_path,
        {
            'data/gsm8k/openai/model/candidate.json': candidate,
            'data/gsm8k/openai/model/second.json': candidate,
        },
    )

    report = check_duplicates(list(local), local, manifest)
    by_path = {result.file_path: result for result in report.results}

    assert (
        by_path['data/gsm8k/openai/model/candidate.json'].duplicate_of
        == 'data/gsm8k/openai/model/existing.json'
    )
    assert (
        by_path['data/gsm8k/openai/model/second.json'].duplicate_of
        == 'data/gsm8k/openai/model/existing.json'
    )


def test_distinct_scores_are_not_duplicates(tmp_path):
    first = _base()
    second = _base()
    _first_result(second)['score_details']['score'] = 0.71
    local = _write(
        tmp_path,
        {
            'data/gsm8k/model/one.json': first,
            'data/gsm8k/model/two.json': second,
        },
    )

    report = check_duplicates(list(local), local, {'files': {}})

    assert all(result.duplicate_of is None for result in report.results)


def test_missing_local_path_is_reported_as_warning():
    report = check_duplicates(
        ['data/gsm8k/openai/model/missing.json'], {}, {'files': {}}
    )

    assert report.results == []
    assert report.warnings == [
        'Duplicate check skipped data/gsm8k/openai/model/missing.json: '
        'local path was not provided'
    ]


def test_check_duplicates_rejects_non_json_paths():
    with pytest.raises(ValueError, match='only accepts .json files'):
        check_duplicates(['data/gsm8k/model/readme.txt'], {}, {'files': {}})
