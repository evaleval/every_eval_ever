"""Focused coverage for the scoped validator rules retained from PR #194."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator
from pydantic import ValidationError

from every_eval_ever.converters import (
    SCHEMA_VERSION as CONVERTER_SCHEMA_VERSION,
)
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    ModelInfo,
)
from every_eval_ever.helpers import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION
from every_eval_ever.schema import get_schema_version, schema_json
from every_eval_ever.validate import (
    check_path_structure,
    check_score_metadata,
    validate_aggregate,
    validate_instance_file,
)
from every_eval_ever.validate import (
    main as validate_main,
)

UUID = '550e8400-e29b-41d4-a716-446655440000'
AGGREGATE_REPO_PATH = f'data/bench/dev/model/{UUID}.json'
COMPANION_REPO_PATH = f'data/bench/dev/model/{UUID}_samples.jsonl'
CURRENT_SCHEMA_VERSION = get_schema_version()


def test_schema_versions_are_consistent():
    assert ADAPTER_SCHEMA_VERSION == CURRENT_SCHEMA_VERSION
    assert CONVERTER_SCHEMA_VERSION == CURRENT_SCHEMA_VERSION
    assert (
        schema_json('instance_level_eval.schema.json')['version']
        == CURRENT_SCHEMA_VERSION
    )


def valid_aggregate() -> dict:
    return {
        'schema_version': CURRENT_SCHEMA_VERSION,
        'evaluation_id': 'bench/dev_model/123',
        'retrieved_timestamp': '123',
        'source_metadata': {
            'source_type': 'evaluation_run',
            'source_organization_name': 'Test',
            'evaluator_relationship': 'third_party',
        },
        'eval_library': {'name': 'unknown', 'version': 'unknown'},
        'model_info': {
            'name': 'model',
            'id': 'dev/model',
            'additional_details': {
                'deployment_type': 'unknown',
                'model_availability': 'unknown',
            },
        },
        'evaluation_results': [
            {
                'evaluation_name': 'bench',
                'source_data': {
                    'dataset_name': 'bench',
                    'source_type': 'other',
                },
                'metric_config': {
                    'lower_is_better': False,
                    'score_type': 'binary',
                },
                'score_details': {'score': 1.0},
            }
        ],
    }


def write_aggregate(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / UUID / f'{UUID}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def valid_sample() -> dict:
    return {
        'schema_version': CURRENT_SCHEMA_VERSION,
        'evaluation_id': valid_aggregate()['evaluation_id'],
        'model_id': valid_aggregate()['model_info']['id'],
        'evaluation_name': 'bench',
        'sample_id': 'sample-1',
        'interaction_type': 'single_turn',
        'input': {'raw': 'question', 'reference': ['answer']},
        'output': {'raw': ['answer']},
        'answer_attribution': [
            {
                'turn_idx': 0,
                'source': 'output.raw',
                'extracted_value': 'answer',
                'extraction_method': 'exact_match',
                'is_terminal': True,
            }
        ],
        'evaluation': {'score': 1.0, 'is_correct': True},
    }


def write_samples(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / UUID / f'{UUID}_samples.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(f'{json.dumps(row)}\n' for row in rows),
        encoding='utf-8',
    )
    return path


def test_path_components_cannot_use_reserved_data_name():
    for repo_path in (
        f'data/data/dev/model/{UUID}.json',
        f'data/bench/data/model/{UUID}.json',
        f'data/bench/dev/data/{UUID}.json',
    ):
        errors = check_path_structure(repo_path)
        assert len(errors) == 1
        assert 'reserved datastore name' in errors[0]


def test_path_components_reject_nonportable_windows_names():
    for component in ('bad:name', 'CON', 'model.', 'bad\x01name'):
        repo_path = f'data/bench/dev/{component}/{UUID}.json'
        errors = check_path_structure(repo_path)
        assert len(errors) == 1
        assert 'portable filesystem names' in errors[0]


def validate_data(
    tmp_path: Path,
    data: dict,
    *,
    available_files: set[str] | None = None,
    repo_files: dict[str, str] | None = None,
):
    available = available_files or {AGGREGATE_REPO_PATH}
    if repo_files is None:
        repo_files = (
            {COMPANION_REPO_PATH: ''}
            if COMPANION_REPO_PATH in available
            else {}
        )
    return validate_aggregate(
        write_aggregate(tmp_path, data),
        repo_path=AGGREGATE_REPO_PATH,
        available_files=available,
        read_repo_file=repo_files.__getitem__,
        run_semantic_checks=True,
    )


def test_path_structure_accepts_only_datastore_file_conventions():
    assert check_path_structure(AGGREGATE_REPO_PATH) == []
    assert check_path_structure(COMPANION_REPO_PATH) == []
    assert check_path_structure(f'data/bench/dev/model/{UUID}.jsonl')
    assert check_path_structure(f'data/bench/dev/model/{UUID}_samples.json')
    assert check_path_structure(f'data/bench/sub/dev/model/{UUID}.json')
    assert check_path_structure(f'/data/bench/dev/model/{UUID}.json')
    assert check_path_structure(f'data/bench/../model/{UUID}.json')
    assert check_path_structure(f'data/bench/dev/model/{UUID}_Samples.jsonl')


def test_companion_check_uses_declared_path(tmp_path):
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': 'different_samples.jsonl',
    }
    report = validate_data(
        tmp_path,
        data,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        repo_files={COMPANION_REPO_PATH: json.dumps(valid_sample()) + '\n'},
    )
    assert report.valid is False
    assert any(
        'different_samples.jsonl' in error['msg'] for error in report.errors
    )


def test_companion_check_accepts_existing_declared_repository_path(tmp_path):
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
    }
    report = validate_data(
        tmp_path,
        data,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        repo_files={COMPANION_REPO_PATH: json.dumps(valid_sample()) + '\n'},
    )
    assert report.valid is True, report.errors


def test_aggregate_requires_tag_when_samples_sibling_exists(tmp_path):
    report = validate_data(
        tmp_path,
        valid_aggregate(),
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
    )

    assert report.valid is False
    assert any(
        'detailed_evaluation_results is required' in error['msg']
        for error in report.errors
    )


def test_companion_must_use_full_path_same_folder_and_uuid(tmp_path):
    other_uuid = '550e8400-e29b-41d4-a716-446655440001'
    for reference in (
        f'{UUID}_samples.jsonl',
        f'data/bench/dev/model/{other_uuid}_samples.jsonl',
        f'data/other-bench/dev/model/{UUID}_samples.jsonl',
        f'data/bench/other-dev/model/{UUID}_samples.jsonl',
        f'data/bench/dev/other-model/{UUID}_samples.jsonl',
    ):
        data = valid_aggregate()
        data['detailed_evaluation_results'] = {
            'format': 'jsonl',
            'file_path': reference,
        }
        report = validate_data(tmp_path, data)

        assert report.valid is False
        assert any(
            'expected exactly' in error['msg'] for error in report.errors
        )


def test_pair_ids_and_total_rows_must_match(tmp_path):
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
        'total_rows': 2,
    }
    sample = valid_sample()
    sample['evaluation_id'] = 'different-evaluation'
    sample['model_id'] = 'different/model'
    samples_text = f'{json.dumps(sample)}\n'

    report = validate_data(
        tmp_path,
        data,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        repo_files={COMPANION_REPO_PATH: samples_text},
    )

    assert report.valid is False
    messages = [error['msg'] for error in report.errors]
    assert any('samples evaluation_id' in message for message in messages)
    assert any('samples model_id' in message for message in messages)
    assert any('total_rows' in error['loc'] for error in report.errors)


def test_samples_requires_aggregate_that_points_back(tmp_path):
    sample_path = write_samples(tmp_path, [valid_sample()])
    report = validate_instance_file(
        sample_path,
        repo_path=COMPANION_REPO_PATH,
        available_files={COMPANION_REPO_PATH},
        read_repo_file={}.get,
        run_semantic_checks=True,
    )
    assert report.valid is False
    assert any(
        'requires sibling aggregate' in error['msg'] for error in report.errors
    )

    aggregate = valid_aggregate()
    report = validate_instance_file(
        sample_path,
        repo_path=COMPANION_REPO_PATH,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        read_repo_file={
            AGGREGATE_REPO_PATH: json.dumps(aggregate),
        }.__getitem__,
        run_semantic_checks=True,
    )
    assert report.valid is False
    assert any(
        'must declare detailed_evaluation_results' in error['msg']
        for error in report.errors
    )

    aggregate['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
        'total_rows': 1,
    }
    report = validate_instance_file(
        sample_path,
        repo_path=COMPANION_REPO_PATH,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        read_repo_file={
            AGGREGATE_REPO_PATH: json.dumps(aggregate),
        }.__getitem__,
        run_semantic_checks=True,
    )
    assert report.valid is True, report.errors


def test_samples_reject_aggregate_pointing_to_another_full_path(tmp_path):
    sample_path = write_samples(tmp_path, [valid_sample()])
    aggregate = valid_aggregate()
    aggregate['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/other-model/{UUID}_samples.jsonl',
        'total_rows': 1,
    }

    report = validate_instance_file(
        sample_path,
        repo_path=COMPANION_REPO_PATH,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        read_repo_file={
            AGGREGATE_REPO_PATH: json.dumps(aggregate),
        }.__getitem__,
        run_semantic_checks=True,
    )

    assert report.valid is False
    assert any('expected exactly' in error['msg'] for error in report.errors)


def test_companion_check_supports_bot_file_lookup(tmp_path):
    class BotFileLookup:
        def __contains__(self, path: object) -> bool:
            return path == COMPANION_REPO_PATH

    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
    }
    report = validate_aggregate(
        write_aggregate(tmp_path, data),
        repo_path=AGGREGATE_REPO_PATH,
        available_files=BotFileLookup(),
        read_repo_file=lambda path: json.dumps(valid_sample()) + '\n',
        run_semantic_checks=True,
    )

    assert report.valid is True, report.errors


def test_companion_check_enforces_companion_path_convention(tmp_path):
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': 'details.jsonl',
    }
    report = validate_data(
        tmp_path,
        data,
        available_files={
            AGGREGATE_REPO_PATH,
            'data/bench/dev/model/details.jsonl',
        },
    )

    assert report.valid is False
    assert any('expected exactly' in error['msg'] for error in report.errors)


def test_companion_check_rejects_absolute_and_parent_paths(tmp_path):
    for file_path in (
        '/tmp/details.jsonl',
        '../details.jsonl',
        r'C:\tmp\details.jsonl',
    ):
        data = valid_aggregate()
        data['detailed_evaluation_results'] = {
            'format': 'jsonl',
            'file_path': file_path,
        }
        report = validate_data(tmp_path, data)
        assert report.valid is False
        assert any(
            'expected exactly' in error['msg'] for error in report.errors
        )


def test_score_bounds_are_required_only_for_continuous_metrics():
    data = valid_aggregate()
    metric = data['evaluation_results'][0]['metric_config']
    assert check_score_metadata(data) == []

    metric.pop('score_type')
    assert check_score_metadata(data) == []

    metric['score_type'] = 'continuous'
    findings = check_score_metadata(data)
    assert any('min_score' in finding for finding in findings)
    assert any('max_score' in finding for finding in findings)


def test_score_bounds_accept_infinity_and_reject_reversed_ranges():
    data = valid_aggregate()
    metric = data['evaluation_results'][0]['metric_config']
    metric.update(
        {
            'score_type': 'continuous',
            'min_score': '-Infinity',
            'max_score': 'Infinity',
        }
    )
    assert check_score_metadata(data) == []

    metric.update({'min_score': 2, 'max_score': 1})
    assert any(
        'greater than max_score' in finding
        for finding in check_score_metadata(data)
    )


def test_model_unknown_placeholders_pass_but_missing_raw_fields_fail(tmp_path):
    assert validate_data(tmp_path, valid_aggregate()).valid is True

    missing = valid_aggregate()
    missing['model_info'].pop('additional_details')
    report = validate_data(tmp_path, missing)
    assert report.valid is False
    assert any('deployment_type' in error['msg'] for error in report.errors)
    assert any('model_availability' in error['msg'] for error in report.errors)


def test_model_info_objects_default_new_fields_to_unknown():
    model = ModelInfo(name='model', id='dev/model')
    assert model.additional_details == {
        'deployment_type': 'unknown',
        'model_availability': 'unknown',
    }
    with pytest.raises(ValidationError, match='deployment_type'):
        ModelInfo(
            name='model',
            id='dev/model',
            additional_details={
                'deployment_type': 'banana',
                'model_availability': 'unknown',
            },
        )


def test_judge_models_require_raw_deployment_metadata(tmp_path):
    data = valid_aggregate()
    data['evaluation_results'][0]['metric_config']['llm_scoring'] = {
        'judges': [
            {
                'model_info': {
                    'name': 'judge',
                    'id': 'dev/judge',
                    'developer': 'dev',
                }
            }
        ]
    }

    report = validate_data(tmp_path, data)

    assert report.valid is False
    assert any(
        'judges[0].model_info.additional_details' in error['loc']
        for error in report.errors
    )


def test_json_schema_requires_and_constrains_model_metadata():
    schema = json.loads(
        Path('every_eval_ever/schemas/eval.schema.json').read_text(
            encoding='utf-8'
        )
    )
    validator = Draft7Validator(schema)

    assert list(validator.iter_errors(valid_aggregate())) == []

    missing = valid_aggregate()
    missing['model_info'].pop('additional_details')
    messages = [error.message for error in validator.iter_errors(missing)]
    assert any('additional_details' in message for message in messages)

    invalid = valid_aggregate()
    invalid['model_info']['additional_details']['deployment_type'] = 'banana'
    messages = [error.message for error in validator.iter_errors(invalid)]
    assert any('banana' in message for message in messages)


def test_json_schema_accepts_metric_with_unknown_bounds():
    schema = json.loads(
        Path('every_eval_ever/schemas/eval.schema.json').read_text(
            encoding='utf-8'
        )
    )
    data = valid_aggregate()
    metric = data['evaluation_results'][0]['metric_config']
    metric.pop('score_type')

    assert list(Draft7Validator(schema).iter_errors(data)) == []


def test_detailed_results_requires_jsonl_path():
    with pytest.raises(ValidationError):
        DetailedEvaluationResults()
    with pytest.raises(ValidationError):
        DetailedEvaluationResults(format='json', file_path='details.json')
    with pytest.raises(ValidationError):
        DetailedEvaluationResults(
            format='jsonl', file_path=f'{UUID}_samples.jsonl'
        )

    detailed = DetailedEvaluationResults(
        format='jsonl', file_path=COMPANION_REPO_PATH
    )
    assert detailed.file_path == COMPANION_REPO_PATH


def test_strict_json_rejects_nonfinite_tokens_and_duplicate_keys(tmp_path):
    path = write_aggregate(tmp_path, valid_aggregate())
    path.write_text('{"score": NaN}', encoding='utf-8')
    report = validate_aggregate(path, repo_path=AGGREGATE_REPO_PATH)
    assert report.valid is False
    assert 'non-finite JSON number' in report.errors[0]['msg']

    path.write_text('{"score": 1e999}', encoding='utf-8')
    report = validate_aggregate(path, repo_path=AGGREGATE_REPO_PATH)
    assert report.valid is False
    assert 'non-finite JSON number' in report.errors[0]['msg']

    path.write_text('{"x": 1, "x": 2}', encoding='utf-8')
    report = validate_aggregate(path, repo_path=AGGREGATE_REPO_PATH)
    assert report.valid is False
    assert 'duplicate JSON object key' in report.errors[0]['msg']


def test_uncertainty_values_must_be_raw_finite_numbers():
    data = valid_aggregate()
    data['evaluation_results'][0]['score_details']['uncertainty'] = {
        'standard_deviation': 'Infinity',
        'standard_error': {'value': '0.1'},
        'confidence_interval': {'lower': 0.1, 'upper': 'inf'},
    }

    findings = check_score_metadata(data)

    assert len(findings) == 3
    assert all('expected a finite number' in finding for finding in findings)


def test_empty_samples_companion_is_rejected(tmp_path):
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
    }

    report = validate_data(
        tmp_path,
        data,
        available_files={AGGREGATE_REPO_PATH, COMPANION_REPO_PATH},
        repo_files={COMPANION_REPO_PATH: ''},
    )

    assert report.valid is False
    assert any(
        'at least one JSONL row' in error['msg'] for error in report.errors
    )


def test_repository_checks_require_a_repository_path(tmp_path):
    report = validate_aggregate(
        write_aggregate(tmp_path, valid_aggregate()),
        run_semantic_checks=True,
    )

    assert report.valid is False
    assert any(
        'repo_path is required' in error['msg'] for error in report.errors
    )


def test_local_command_runs_the_same_repository_checks(
    tmp_path, monkeypatch, capsys
):
    aggregate_path = (
        tmp_path / 'data' / 'bench' / 'dev' / 'model' / f'{UUID}.json'
    )
    aggregate_path.parent.mkdir(parents=True)
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': COMPANION_REPO_PATH,
    }
    aggregate_path.write_text(json.dumps(data), encoding='utf-8')
    companion_path = aggregate_path.with_name(f'{UUID}_samples.jsonl')
    companion_path.write_text(
        json.dumps(valid_sample()) + '\n', encoding='utf-8'
    )
    monkeypatch.chdir(tmp_path)

    exit_code = validate_main(
        ['--format', 'json', aggregate_path.relative_to(tmp_path).as_posix()]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)[0]['valid'] is True

    exit_code = validate_main(['--format', 'json', 'data/*/*/*/*.json*'])
    reports = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(reports) == 2
    assert all(report['valid'] for report in reports)

    companion_path.unlink()
    exit_code = validate_main(
        ['--format', 'json', aggregate_path.relative_to(tmp_path).as_posix()]
    )

    assert exit_code == 1
    errors = json.loads(capsys.readouterr().out)[0]['errors']
    assert any('was not found' in error['msg'] for error in errors)
