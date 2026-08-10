"""Tests for the standalone local folder validator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import local_validate
from tests.test_validation_scope import UUID, valid_aggregate, valid_sample


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(f'{json.dumps(row)}\n' for row in rows),
        encoding='utf-8',
    )
    return path


def canonical_aggregate(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / 'data' / 'bench' / 'dev' / 'model' / f'{UUID}.json',
        valid_aggregate(),
    )


def test_canonical_record_is_merge_ready(tmp_path: Path):
    report = local_validate.validate_local_file(canonical_aggregate(tmp_path))

    assert report.valid is True
    assert report.errors == []
    assert report.warnings == []
    assert local_validate.report_status(report) == 'pass'
    assert local_validate.report_merge_ready(report) is True


def test_noncanonical_record_is_valid_but_not_merge_ready(tmp_path: Path):
    path = write_json(tmp_path / 'results.json', valid_aggregate())

    report = local_validate.validate_local_file(path)

    assert report.valid is True
    assert report.errors == []
    assert report.warnings
    assert all(warning['type'] == 'path_warning' for warning in report.warnings)
    assert local_validate.report_status(report) == 'warn'
    assert local_validate.report_merge_ready(report) is False


def test_non_uuid_filename_is_advisory_in_canonical_tree(tmp_path: Path):
    path = write_json(
        tmp_path / 'data' / 'bench' / 'dev' / 'model' / 'results.json',
        valid_aggregate(),
    )

    report = local_validate.validate_local_file(path)

    assert report.valid is True
    assert any('UUID4' in warning['msg'] for warning in report.warnings)


def test_wrong_canonical_depth_is_advisory(tmp_path: Path):
    path = write_json(
        tmp_path / 'data' / 'bench' / 'model' / f'{UUID}.json',
        valid_aggregate(),
    )

    report = local_validate.validate_local_file(path)

    assert report.valid is True
    assert report.errors == []
    assert len(report.warnings) == 1
    assert 'Unexpected path depth' in report.warnings[0]['msg']


def test_arbitrary_folder_uses_physical_companion(tmp_path: Path):
    folder = tmp_path / 'arbitrary' / 'output'
    aggregate_path = folder / f'{UUID}.json'
    sample_path = folder / f'{UUID}_samples.jsonl'
    aggregate = valid_aggregate()
    aggregate['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/model/{UUID}_samples.jsonl',
        'total_rows': 1,
    }
    write_json(aggregate_path, aggregate)
    write_jsonl(sample_path, [valid_sample()])

    aggregate_report = local_validate.validate_local_file(aggregate_path)
    sample_report = local_validate.validate_local_file(sample_path)

    assert aggregate_report.valid is True, aggregate_report.errors
    assert sample_report.valid is True, sample_report.errors
    assert aggregate_report.warnings
    assert sample_report.warnings


def test_companion_content_mismatch_remains_blocking(tmp_path: Path):
    folder = tmp_path / 'flat'
    aggregate_path = folder / f'{UUID}.json'
    sample_path = folder / f'{UUID}_samples.jsonl'
    aggregate = valid_aggregate()
    aggregate['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/model/{UUID}_samples.jsonl',
        'total_rows': 2,
    }
    write_json(aggregate_path, aggregate)
    write_jsonl(sample_path, [valid_sample()])

    report = local_validate.validate_local_file(aggregate_path)

    assert report.valid is False
    assert any('total_rows' in error['loc'] for error in report.errors)
    assert local_validate.report_status(report) == 'fail'


def test_registered_semantic_rules_are_not_bypassed(tmp_path: Path):
    aggregate = valid_aggregate()
    del aggregate['model_info']['additional_details']['deployment_type']
    path = write_json(tmp_path / f'{UUID}.json', aggregate)

    report = local_validate.validate_local_file(path)

    assert report.valid is False
    assert any('deployment_type' in error['msg'] for error in report.errors)


def test_recursive_discovery_skips_hidden_and_linked_directories(
    tmp_path: Path,
):
    wanted = write_json(tmp_path / 'nested' / 'wanted.json', valid_aggregate())
    write_json(tmp_path / '.venv' / 'ignored.json', valid_aggregate())
    link = tmp_path / 'linked'
    link.symlink_to(wanted.parent, target_is_directory=True)

    paths = local_validate.expand_inputs([str(tmp_path), str(wanted)])

    assert paths == [wanted]


def test_unrelated_json_is_validated_not_skipped(tmp_path: Path):
    path = write_json(tmp_path / 'package.json', {'name': 'not-an-eval'})

    report = local_validate.validate_local_file(path)

    assert report.valid is False
    assert report.errors


def test_json_output_is_detailed_and_stdout_stays_parseable(
    tmp_path: Path, capsys
):
    folder = tmp_path / 'flat'
    write_json(folder / 'results.json', valid_aggregate())

    exit_code = local_validate.main(['--format', 'json', folder.as_posix()])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload['validator']['scope'] == 'local'
    assert payload['validator']['schema_version']
    assert payload['validator']['schema_fingerprint']
    assert payload['summary']['warned'] == 1
    assert payload['summary']['exit_code'] == 2
    assert payload['summary']['merge_ready'] is False
    assert payload['reports'][0]['status'] == 'warn'
    warning = payload['reports'][0]['warnings'][0]
    assert set(warning) == {'type', 'location', 'message', 'input'}
    assert warning['type'] == 'path_warning'
    assert warning['message']
    assert 'validating 1 file(s)' in captured.err


def test_json_output_describes_input_errors(tmp_path: Path, capsys):
    exit_code = local_validate.main(['--format', 'json', tmp_path.as_posix()])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload['summary']['errors'] == 1
    assert payload['summary']['exit_code'] == 1
    assert payload['summary']['merge_ready'] is False
    assert payload['input_errors'][0]['type'] == 'input_error'
    assert (
        'contains no .json or .jsonl' in (payload['input_errors'][0]['message'])
    )


def test_json_output_gives_agents_field_level_errors(tmp_path: Path, capsys):
    aggregate = valid_aggregate()
    del aggregate['evaluation_id']
    path = write_json(tmp_path / 'results.json', aggregate)

    exit_code = local_validate.main(['--format', 'json', path.as_posix()])

    payload = json.loads(capsys.readouterr().out)
    error = payload['reports'][0]['errors'][0]
    assert exit_code == 1
    assert payload['reports'][0]['status'] == 'fail'
    assert payload['summary']['merge_ready'] is False
    assert error['type']
    assert 'evaluation_id' in error['location']
    assert error['message']
    assert 'input' in error


def test_rich_output_groups_repeated_errors(tmp_path: Path, capsys):
    folder = tmp_path / 'flat'
    aggregate = valid_aggregate()
    del aggregate['evaluation_id']
    write_json(folder / 'first.json', aggregate)
    write_json(folder / 'second.json', aggregate)

    exit_code = local_validate.main([folder.as_posix()])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '0 passed, 0 warning-only, 2 failed' in captured.out
    assert 'Errors: 1 group(s), 2 occurrence(s)' in captured.out
    assert '2 occurrence(s) in 2 file(s)' in captured.out


def test_json_log_keeps_individual_warning_details(tmp_path: Path, capsys):
    folder = tmp_path / 'flat'
    path = write_json(folder / 'results.json', valid_aggregate())
    log_path = tmp_path / 'reports' / 'local-validation.json'

    exit_code = local_validate.main(
        ['--json-log', log_path.as_posix(), path.as_posix()]
    )

    captured = capsys.readouterr()
    payload = json.loads(log_path.read_text(encoding='utf-8'))
    assert exit_code == 2
    assert payload['summary']['warned'] == 1
    assert payload['reports'][0]['file'] == str(path)
    assert payload['reports'][0]['warnings']
    assert payload['reports'][0]['warnings'][0]['message']
    assert f'Detailed JSON log: {log_path}' in captured.err


def test_rich_output_marks_warning_only_run(tmp_path: Path, capsys):
    path = write_json(tmp_path / 'results.json', valid_aggregate())

    exit_code = local_validate.main([path.as_posix()])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert 'WARN' in captured.out
    assert 'not merge-ready' in captured.out


def test_clean_run_exits_zero(tmp_path: Path, capsys):
    path = canonical_aggregate(tmp_path)

    exit_code = local_validate.main([path.as_posix()])

    assert exit_code == 0
    assert '1 passed, 0 warning-only, 0 failed' in capsys.readouterr().out
