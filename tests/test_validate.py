"""Tests for validate.py — Pydantic-based EEE schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from every_eval_ever import cli
from every_eval_ever.converters import (
    SCHEMA_VERSION as CONVERTER_SCHEMA_VERSION,
)
from every_eval_ever.helpers import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION
from every_eval_ever.schema import (
    get_schema_fingerprint,
    get_schema_version,
    schema_json,
    schema_text,
)
from every_eval_ever.validate import (
    check_companion_exists,
    check_dataset_provenance,
    check_evaluator_provenance_consistency,
    check_integer_counts,
    check_model_deployment,
    check_path_structure,
    check_score_metadata,
    expand_paths,
    render_report_json,
    resolve_companion_repo_path,
    validate_aggregate,
    validate_file,
    validate_instance_file,
    validate_many,
)


def test_validate_cli_defaults_to_json_and_rejects_removed_output_modes():
    parser = cli.build_parser()

    args = parser.parse_args(['validate', 'data/example.json'])
    assert args.output_format == 'json'

    for removed_format in ('rich', 'github'):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    'validate',
                    '--format',
                    removed_format,
                    'data/example.json',
                ]
            )


def test_schema_metadata_has_one_runtime_source():
    version = get_schema_version()

    assert version == schema_json()['version']
    assert version == ADAPTER_SCHEMA_VERSION
    assert version == CONVERTER_SCHEMA_VERSION
    assert len(get_schema_fingerprint()) == 64


def test_documented_and_packaged_schemas_match():
    repo_root = Path(__file__).parents[1]

    assert (repo_root / 'eval.schema.json').read_text() == schema_text(
        'eval.schema.json'
    )
    assert (
        repo_root / 'instance_level_eval.schema.json'
    ).read_text() == schema_text('instance_level_eval.schema.json')


from every_eval_ever.validation_core import (
    SemanticCheckError,
    ValidationCheck,
    ValidationContext,
    run_registered_checks,
)

# ---------------------------------------------------------------------------
# Helpers — minimal valid data fixtures
# ---------------------------------------------------------------------------

VALID_AGGREGATE: dict = {
    'schema_version': '0.2.2',
    'evaluation_id': 'test/model/123',
    'retrieved_timestamp': '1234567890',
    'source_metadata': {
        'source_type': 'evaluation_run',
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': 'first_party',
    },
    'eval_library': {'name': 'inspect_ai', 'version': '0.3.0'},
    'model_info': {'name': 'test-model', 'id': 'org/test-model'},
    'evaluation_results': [
        {
            'evaluation_name': 'test_eval',
            'source_data': {
                'dataset_name': 'test_ds',
                'source_type': 'hf_dataset',
                'hf_repo': 'org/test-ds',
            },
            'metric_config': {
                'lower_is_better': False,
                'score_type': 'binary',
            },
            'score_details': {'score': 0.95},
        }
    ],
}

VALID_SINGLE_TURN: dict = {
    'schema_version': 'instance_level_eval_0.2.2',
    'evaluation_id': 'test/model/123',
    'model_id': 'org/test-model',
    'evaluation_name': 'test_eval',
    'sample_id': 'sample_001',
    'interaction_type': 'single_turn',
    'input': {'raw': 'What is 2+2?', 'reference': ['4']},
    'output': {'raw': ['4']},
    'answer_attribution': [
        {
            'turn_idx': 0,
            'source': 'output.raw',
            'extracted_value': '4',
            'extraction_method': 'exact_match',
            'is_terminal': True,
        }
    ],
    'evaluation': {'score': 1.0, 'is_correct': True},
}

VALID_MULTI_TURN: dict = {
    'schema_version': 'instance_level_eval_0.2.2',
    'evaluation_id': 'test/model/123',
    'model_id': 'org/test-model',
    'evaluation_name': 'test_eval',
    'sample_id': 'sample_002',
    'interaction_type': 'multi_turn',
    'input': {'raw': 'Solve this problem', 'reference': ['42']},
    'messages': [
        {'turn_idx': 0, 'role': 'user', 'content': 'Solve this problem'},
        {'turn_idx': 1, 'role': 'assistant', 'content': 'The answer is 42'},
    ],
    'answer_attribution': [
        {
            'turn_idx': 1,
            'source': 'messages[1].content',
            'extracted_value': '42',
            'extraction_method': 'regex',
            'is_terminal': True,
        }
    ],
    'evaluation': {'score': 1.0, 'is_correct': True, 'num_turns': 2},
}


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding='utf-8')
    return p


def _write_jsonl(tmp_path: Path, name: str, lines: list[dict | str]) -> Path:
    p = tmp_path / name
    text_lines = []
    for item in lines:
        if isinstance(item, str):
            text_lines.append(item)
        else:
            text_lines.append(json.dumps(item))
    p.write_text('\n'.join(text_lines) + '\n', encoding='utf-8')
    return p


def _schema_messages(schema_name: str, data: dict) -> list[str]:
    validator = Draft7Validator(schema_json(schema_name))
    return [error.message for error in validator.iter_errors(data)]


# ===================================================================
# Aggregate validation tests
# ===================================================================


class TestAggregateValidation:
    def test_valid_json_passes(self, tmp_path: Path):
        fp = _write_json(tmp_path, 'valid.json', VALID_AGGREGATE)
        report = validate_aggregate(fp, run_semantic_checks=False)
        assert report.valid is True
        assert report.errors == []
        assert report.file_type == 'aggregate'

    def test_missing_required_field(self, tmp_path: Path):
        data = {**VALID_AGGREGATE}
        del data['evaluation_id']
        fp = _write_json(tmp_path, 'missing.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('evaluation_id' in e['loc'] for e in report.errors)

    def test_extra_field_on_evaluation_log_fails(self, tmp_path: Path):
        data = {**VALID_AGGREGATE, 'unexpected_field': 'oops'}
        fp = _write_json(tmp_path, 'extra.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('unexpected_field' in e['loc'] for e in report.errors)

    def test_extra_field_on_generation_args_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['evaluation_results'][0]['generation_config'] = {
            'generation_args': {'temperature': 0.7, 'unknown_param': 'bad'}
        }
        fp = _write_json(tmp_path, 'extra_gen.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('unknown_param' in e['loc'] for e in report.errors)

    def test_score_type_levels_without_level_names_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['evaluation_results'][0]['metric_config'] = {
            'lower_is_better': False,
            'score_type': 'levels',
            # missing level_names and has_unknown_level
        }
        fp = _write_json(tmp_path, 'levels.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('level_names' in e['msg'] for e in report.errors)

    def test_score_type_continuous_without_min_score_fails(
        self, tmp_path: Path
    ):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['evaluation_results'][0]['metric_config'] = {
            'lower_is_better': False,
            'score_type': 'continuous',
            # missing min_score and max_score
        }
        fp = _write_json(tmp_path, 'continuous.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('min_score' in e['msg'] for e in report.errors)

    def test_hf_source_requires_repo(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['evaluation_results'][0]['source_data'] = {
            'dataset_name': 'test',
            'source_type': 'hf_dataset',
        }
        fp = _write_json(tmp_path, 'disc.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('hf_repo' in error['loc'] for error in report.errors)

    def test_source_data_wrong_source_type_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['evaluation_results'][0]['source_data'] = {
            'dataset_name': 'test',
            'source_type': 'invalid_type',
        }
        fp = _write_json(tmp_path, 'bad_source.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False

    def test_additional_details_non_string_values_fail(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['model_info']['additional_details'] = {'params_billions': 8.357}
        fp = _write_json(tmp_path, 'nonstr.json', data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any('string' in e['msg'] for e in report.errors)

    def test_json_parse_error(self, tmp_path: Path):
        fp = tmp_path / 'bad.json'
        fp.write_text('{invalid json}', encoding='utf-8')
        report = validate_aggregate(fp)
        assert report.valid is False
        assert report.errors[0]['type'] == 'json_parse_error'


class TestJSONSchemaContracts:
    def test_valid_aggregate_matches_json_schema(self):
        assert _schema_messages('eval.schema.json', VALID_AGGREGATE) == []

    def test_detailed_results_schema_requires_format_and_path(self):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data['detailed_evaluation_results'] = {}
        messages = _schema_messages('eval.schema.json', data)
        assert any("'format' is a required property" in msg for msg in messages)
        assert any(
            "'file_path' is a required property" in msg for msg in messages
        )

    def test_missing_score_type_does_not_trigger_levels_condition(self):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        del data['evaluation_results'][0]['metric_config']['score_type']
        messages = _schema_messages('eval.schema.json', data)
        assert not any('level_names' in msg for msg in messages)

    def test_multiturn_schema_requires_nonempty_messages_and_num_turns(self):
        data = json.loads(json.dumps(VALID_MULTI_TURN))
        data['messages'] = []
        del data['evaluation']['num_turns']
        messages = _schema_messages('instance_level_eval.schema.json', data)
        assert any('should be non-empty' in msg for msg in messages)
        assert any(
            "'num_turns' is a required property" in msg for msg in messages
        )


# ===================================================================
# Instance-level validation tests
# ===================================================================


class TestInstanceLevelValidation:
    def test_valid_single_turn_passes(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, 'valid.jsonl', [VALID_SINGLE_TURN])
        report = validate_instance_file(fp, run_semantic_checks=False)
        assert report.valid is True
        assert report.line_count == 1

    def test_valid_multi_turn_passes(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, 'multi.jsonl', [VALID_MULTI_TURN])
        report = validate_instance_file(fp, run_semantic_checks=False)
        assert report.valid is True

    def test_single_turn_with_messages_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_SINGLE_TURN))
        data['messages'] = [
            {'turn_idx': 0, 'role': 'user', 'content': 'hi'},
        ]
        fp = _write_jsonl(tmp_path, 'bad_st.jsonl', [data])
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any('must not have messages' in e['msg'] for e in report.errors)

    def test_multi_turn_without_messages_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_MULTI_TURN))
        del data['messages']
        fp = _write_jsonl(tmp_path, 'no_msgs.jsonl', [data])
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any('requires messages' in e['msg'] for e in report.errors)

    def test_invalid_line_in_middle_reports_correct_line_number(
        self, tmp_path: Path
    ):
        bad_line = {**VALID_SINGLE_TURN}
        del bad_line['evaluation_id']
        fp = _write_jsonl(
            tmp_path,
            'mid.jsonl',
            [VALID_SINGLE_TURN, bad_line, VALID_SINGLE_TURN],
        )
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any('line 2' in e['loc'] for e in report.errors)

    def test_json_parse_error_reports_line_number(self, tmp_path: Path):
        fp = _write_jsonl(
            tmp_path, 'parse.jsonl', [VALID_SINGLE_TURN, '{bad json}']
        )
        report = validate_instance_file(fp)
        assert report.valid is False
        assert report.errors[0]['type'] == 'json_parse_error'
        assert 'line 2' in report.errors[0]['loc']

    def test_empty_jsonl_fails(self, tmp_path: Path):
        fp = tmp_path / 'empty.jsonl'
        fp.write_text('', encoding='utf-8')
        report = validate_instance_file(fp, run_semantic_checks=False)
        assert report.valid is False
        assert report.line_count == 0
        assert report.errors[0]['type'] == 'empty_instance_file'

    def test_blank_lines_skipped(self, tmp_path: Path):
        lines = [
            json.dumps(VALID_SINGLE_TURN),
            '',
            '  ',
            json.dumps(VALID_SINGLE_TURN),
        ]
        fp = tmp_path / 'blanks.jsonl'
        fp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        report = validate_instance_file(fp, run_semantic_checks=False)
        assert report.valid is True
        assert report.line_count == 2


# ===================================================================
# File dispatch and CLI tests
# ===================================================================


class TestFileDispatch:
    def test_json_dispatches_to_aggregate(self, tmp_path: Path):
        fp = _write_json(tmp_path, 'test.json', VALID_AGGREGATE)
        report = validate_file(fp)
        assert report.file_type == 'aggregate'

    def test_jsonl_dispatches_to_instance(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, 'test.jsonl', [VALID_SINGLE_TURN])
        report = validate_file(fp)
        assert report.file_type == 'instance'

    def test_unsupported_extension(self, tmp_path: Path):
        fp = tmp_path / 'test.csv'
        fp.write_text('a,b,c', encoding='utf-8')
        report = validate_file(fp)
        assert report.valid is False
        assert report.errors[0]['type'] == 'unsupported_extension'

    def test_directory_expansion(self, tmp_path: Path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        _write_json(sub, 'a.json', VALID_AGGREGATE)
        _write_jsonl(sub, 'b.jsonl', [VALID_SINGLE_TURN])
        (sub / 'c.txt').write_text('ignored')
        paths = expand_paths([str(sub)])
        extensions = {p.suffix for p in paths}
        assert '.json' in extensions
        assert '.jsonl' in extensions
        assert '.txt' not in extensions


class TestMaxErrors:
    def test_max_errors_caps_output(self, tmp_path: Path):
        bad_line = {**VALID_SINGLE_TURN}
        del bad_line['evaluation_id']
        lines = [bad_line] * 100
        fp = _write_jsonl(tmp_path, 'many.jsonl', lines)
        report = validate_instance_file(
            fp, max_errors=5, run_semantic_checks=False
        )
        assert report.valid is False
        # Should have at most 5 real errors + 1 truncation message
        assert len(report.errors) <= 6
        assert any(e['type'] == 'truncated' for e in report.errors)


class TestOutputFormats:
    def test_json_output_is_valid_json(self, tmp_path: Path):
        fp = _write_json(tmp_path, 'test.json', VALID_AGGREGATE)
        report = validate_file(fp, run_semantic_checks=False)
        output = render_report_json([report])
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]['valid'] is True


class TestExitCode:
    def test_exit_code_0_on_pass(self, tmp_path: Path):
        fp = _write_json(tmp_path, 'pass.json', VALID_AGGREGATE)
        report = validate_file(fp, run_semantic_checks=False)
        assert report.valid is True

    def test_exit_code_1_on_failure(self, tmp_path: Path):
        data = {**VALID_AGGREGATE}
        del data['evaluation_id']
        fp = _write_json(tmp_path, 'fail.json', data)
        report = validate_file(fp)
        assert report.valid is False


class TestSemanticWarnings:
    def test_registered_check_failure_is_not_downgraded_to_warning(
        self, tmp_path: Path
    ):
        def broken_check(context, data):
            raise RuntimeError('boom')

        context = ValidationContext(
            local_path=tmp_path / 'record.json',
            repo_path='data/bench/dev/model/record.json',
        )
        check = ValidationCheck('broken', 'aggregate', 'error', broken_check)
        with pytest.raises(
            SemanticCheckError, match='broken check did not complete'
        ):
            run_registered_checks(
                context,
                file_type='aggregate',
                data={},
                checks=(check,),
            )

    def test_path_structure_matches_validator_bot(self):
        good = (
            'data/gsm8k/openai/gpt-4o/550e8400-e29b-41d4-a716-446655440000.json'
        )
        bad = 'data/gsm8k/file.json'
        assert check_path_structure(good) == []
        assert 'Unexpected path depth' in check_path_structure(bad)[0]

    def test_companion_warning_uses_available_files(self):
        uuid = '550e8400-e29b-41d4-a716-446655440000'
        repo_path = f'data/bench/dev/model/{uuid}.json'
        data = {'detailed_evaluation_results': {'file_path': f'{uuid}.jsonl'}}
        assert (
            check_companion_exists(
                repo_path, data, {f'data/bench/dev/model/{uuid}.jsonl'}
            )
            == []
        )
        warnings = check_companion_exists(repo_path, data, {repo_path})
        assert 'referenced companion' in warnings[0]

    def test_companion_validates_declared_path_not_uuid_alias(self):
        uuid = '550e8400-e29b-41d4-a716-446655440000'
        repo_path = f'data/bench/dev/model/{uuid}.json'
        alias = f'data/bench/dev/model/{uuid}.jsonl'
        data = {
            'detailed_evaluation_results': {
                'format': 'jsonl',
                'file_path': 'different.jsonl',
            }
        }
        warnings = check_companion_exists(repo_path, data, {alias})
        assert len(warnings) == 1
        assert 'different.jsonl' in warnings[0]

    def test_companion_accepts_explicit_relative_or_repo_path(self):
        repo_path = 'data/bench/dev/model/aggregate.json'
        companion = 'data/bench/dev/model/details.jsonl'
        relative = {
            'detailed_evaluation_results': {
                'format': 'jsonl',
                'file_path': 'details.jsonl',
            }
        }
        rooted = {
            'detailed_evaluation_results': {
                'format': 'jsonl',
                'file_path': companion,
            }
        }
        assert check_companion_exists(repo_path, relative, {companion}) == []
        assert check_companion_exists(repo_path, rooted, {companion}) == []
        assert resolve_companion_repo_path(repo_path, relative) == companion
        assert resolve_companion_repo_path(repo_path, rooted) == companion

    def test_companion_rejects_parent_traversal(self):
        warnings = check_companion_exists(
            'data/bench/dev/model/aggregate.json',
            {
                'detailed_evaluation_results': {
                    'format': 'jsonl',
                    'file_path': '../details.jsonl',
                }
            },
            set(),
        )
        assert 'parent traversal' in warnings[0]

    def test_score_metadata_missing_and_bounds_warn(self):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        warnings = check_score_metadata(data)
        assert any("invalid 'min_score'" in warning for warning in warnings)
        assert any("invalid 'max_score'" in warning for warning in warnings)

        data['evaluation_results'][0]['metric_config'].update(
            {'score_type': 'continuous', 'min_score': 0, 'max_score': 1}
        )
        data['evaluation_results'][0]['score_details']['score'] = 1.5
        warnings = check_score_metadata(data)
        assert any(
            'outside [min_score=0, max_score=1]' in warning
            for warning in warnings
        )

    def test_score_metadata_rejects_null_nonfinite_and_reversed_bounds(self):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        metric = data['evaluation_results'][0]['metric_config']
        metric.update({'min_score': None, 'max_score': float('nan')})
        warnings = check_score_metadata(data)
        assert any("invalid 'min_score'" in warning for warning in warnings)
        assert any("invalid 'max_score'" in warning for warning in warnings)

        metric.update({'min_score': 2, 'max_score': 1})
        warnings = check_score_metadata(data)
        assert any('greater than max_score' in warning for warning in warnings)

    def test_score_metadata_accepts_strict_json_infinity_bounds(self):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        metric = data['evaluation_results'][0]['metric_config']
        metric.update(
            {
                'score_type': 'continuous',
                'min_score': '-Infinity',
                'max_score': 'Infinity',
            }
        )
        data['evaluation_results'][0]['score_details']['score'] = 1.5

        assert check_score_metadata(data) == []

    def test_nonstandard_or_ambiguous_json_is_rejected(self, tmp_path: Path):
        nan_path = tmp_path / 'nan.json'
        nan_path.write_text('{"score": NaN}', encoding='utf-8')
        nan_report = validate_aggregate(nan_path)
        assert nan_report.valid is False
        assert 'non-finite JSON number' in nan_report.errors[0]['msg']

        duplicate_path = tmp_path / 'duplicate.json'
        duplicate_path.write_text('{"x": 1, "x": 2}', encoding='utf-8')
        duplicate_report = validate_aggregate(duplicate_path)
        assert duplicate_report.valid is False
        assert 'duplicate JSON object key' in duplicate_report.errors[0]['msg']

    def test_integer_count_warning(self):
        warnings = check_integer_counts(
            {'score_details': {'uncertainty': {'num_samples': 10.0}}}
        )
        assert any('num_samples' in warning for warning in warnings)

    def test_model_deployment_axes_are_required_and_independent(self):
        base = {'model_info': {'id': 'org/model', 'additional_details': {}}}
        findings = check_model_deployment(base)
        assert len(findings) == 2
        assert any('deployment_type' in finding for finding in findings)
        assert any('model_availability' in finding for finding in findings)

        self_deployed_closed = {
            'model_info': {
                'id': 'org/model',
                'additional_details': {
                    'deployment_type': 'self_deployed',
                    'model_availability': 'closed_weights',
                },
            }
        }
        externally_managed_open = {
            'model_info': {
                'id': 'org/model',
                'additional_details': {
                    'deployment_type': 'externally_managed',
                    'model_availability': 'open_weights',
                },
            }
        }
        assert check_model_deployment(self_deployed_closed) == []
        assert check_model_deployment(externally_managed_open) == []

        legacy_values = {
            'model_info': {
                'id': 'org/model',
                'additional_details': {
                    'deployment_type': 'api',
                    'model_availability': 'closed_source',
                },
            }
        }
        findings = check_model_deployment(legacy_values)
        assert len(findings) == 2
        assert any("got 'api'" in finding for finding in findings)
        assert any("got 'closed_source'" in finding for finding in findings)

    def test_dataset_provenance_requires_hf_api_for_hf_dataset(self):
        data = {
            'evaluation_results': [
                {
                    'source_data': {
                        'source_type': 'hf_dataset',
                        'hf_repo': 'org/dataset',
                    }
                },
                {'source_data': {'source_type': 'other'}},
            ]
        }
        warnings = check_dataset_provenance(data)
        assert any('no HfApi was provided' in warning for warning in warnings)
        assert any("source_type 'other'" in warning for warning in warnings)

    def test_evaluator_provenance_must_match_aggregate_group(self):
        data = {
            'source_metadata': {'evaluator_relationship': 'third_party'},
            'evaluation_results': [
                {
                    'score_details': {
                        'details': {
                            'inferred_evaluator_relationship': 'first_party',
                            'relationship_inference_reason': (
                                'source_matches_model_developer'
                            ),
                        }
                    }
                }
            ],
        }

        findings = check_evaluator_provenance_consistency(data)

        assert len(findings) == 1
        assert 'does not match' in findings[0]

    def test_evaluator_provenance_contract_is_conditional_and_complete(self):
        matching = {
            'source_metadata': {'evaluator_relationship': 'first_party'},
            'evaluation_results': [
                {
                    'score_details': {
                        'details': {
                            'inferred_evaluator_relationship': 'first_party',
                            'relationship_inference_reason': (
                                'source_matches_model_developer'
                            ),
                        }
                    }
                },
                {'score_details': {'score': 0.5}},
            ],
        }
        assert check_evaluator_provenance_consistency(matching) == []

        incomplete = json.loads(json.dumps(matching))
        del incomplete['evaluation_results'][0]['score_details']['details'][
            'relationship_inference_reason'
        ]
        findings = check_evaluator_provenance_consistency(incomplete)
        assert len(findings) == 1
        assert 'relationship_inference_reason' in findings[0]

    def test_validate_many_preserves_explicit_empty_available_files(
        self, tmp_path: Path
    ):
        uuid = '550e8400-e29b-41d4-a716-446655440000'
        aggregate = json.loads(json.dumps(VALID_AGGREGATE))
        aggregate['detailed_evaluation_results'] = {
            'format': 'jsonl',
            'file_path': f'{uuid}.jsonl',
        }
        json_path = _write_json(tmp_path, f'{uuid}.json', aggregate)
        jsonl_path = _write_jsonl(
            tmp_path, f'{uuid}.jsonl', [VALID_SINGLE_TURN]
        )
        reports = validate_many(
            [
                (f'data/bench/dev/model/{uuid}.json', json_path),
                (f'data/bench/dev/model/{uuid}.jsonl', jsonl_path),
            ],
            available_files=set(),
        )

        aggregate_report = reports[0]
        assert aggregate_report.valid is False
        assert any(
            'referenced companion' in error['msg']
            for error in aggregate_report.errors
        )
