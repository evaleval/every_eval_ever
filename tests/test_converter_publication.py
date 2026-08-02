from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from every_eval_ever import cli
from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
from every_eval_ever.converters.lm_eval.instance_level_adapter import (
    LMEvalInstanceLevelAdapter,
)
from every_eval_ever.helpers.io import SourceRecordsError, datastore_output_dir
from every_eval_ever.validate import validate_file

LM_EVAL_DIR = Path('tests/data/lm_eval')
RESULTS_FILE = LM_EVAL_DIR / 'results_2026-01-21T03-44-18.458309.json'
SAMPLES_FILE = (
    LM_EVAL_DIR / 'samples_math_perturbed_full_2026-01-21T03-44-18.458309.jsonl'
)
FILE_UUID = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'


def make_lm_eval_log():
    return LMEvalAdapter().transform_from_file(
        RESULTS_FILE,
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'first_party',
        },
    )[0]


def output_dir(base_dir: Path, log) -> Path:
    return datastore_output_dir(
        base_dir,
        log.evaluation_results[0].source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
    )


def test_publish_evaluation_logs_uses_canonical_validated_path(
    tmp_path: Path,
):
    log = make_lm_eval_log()

    paths = publish_evaluation_logs([log], tmp_path / 'data', [FILE_UUID])

    assert paths == [output_dir(tmp_path / 'data', log) / f'{FILE_UUID}.json']
    report = validate_file(paths[0])
    assert report.valid, report.errors


def test_sidecar_and_aggregate_rollback_preserves_competing_file(
    tmp_path: Path, monkeypatch
):
    log = make_lm_eval_log()
    staging_dir = tmp_path / 'staging'
    staged_model_dir = output_dir(staging_dir, log)
    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
        samples_path=SAMPLES_FILE,
        evaluation_id=log.evaluation_id,
        model_id=log.model_info.id,
        task_name='math_perturbed_full',
        output_dir=str(staged_model_dir),
        file_uuid=FILE_UUID,
        collection=log.evaluation_results[0].source_data.dataset_name,
        developer=log.model_info.developer,
    )
    assert detailed is not None
    log.detailed_evaluation_results = detailed

    final_dir = output_dir(tmp_path / 'data', log)
    aggregate_path = final_dir / f'{FILE_UUID}.json'
    sample_path = final_dir / f'{FILE_UUID}_samples.jsonl'
    original_open = Path.open

    def racing_open(path: Path, mode='r', *args, **kwargs):
        if path == aggregate_path and mode == 'xb':
            with original_open(path, 'xb') as handle:
                handle.write(b'competing process')
            raise FileExistsError(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', racing_open)

    with pytest.raises(FileExistsError):
        publish_evaluation_logs(
            [log],
            tmp_path / 'data',
            [FILE_UUID],
            staged_output_dir=staging_dir,
        )

    assert not sample_path.exists()
    assert aggregate_path.read_bytes() == b'competing process'


def test_publisher_rejects_tampered_staged_samples(tmp_path: Path):
    log = make_lm_eval_log()
    staging_dir = tmp_path / 'staging'
    staged_model_dir = output_dir(staging_dir, log)
    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
        samples_path=SAMPLES_FILE,
        evaluation_id=log.evaluation_id,
        model_id=log.model_info.id,
        task_name='math_perturbed_full',
        output_dir=str(staged_model_dir),
        file_uuid=FILE_UUID,
        collection=log.evaluation_results[0].source_data.dataset_name,
        developer=log.model_info.developer,
    )
    assert detailed is not None
    log.detailed_evaluation_results = detailed
    sample_path = staged_model_dir / Path(detailed.file_path).name
    lines = sample_path.read_text(encoding='utf-8').splitlines()
    first_row = json.loads(lines[0])
    first_row['model_id'] = 'other/model'
    lines[0] = json.dumps(first_row)
    sample_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='model_id does not match'):
        publish_evaluation_logs(
            [log],
            tmp_path / 'data',
            [FILE_UUID],
            staged_output_dir=staging_dir,
        )

    assert not list((tmp_path / 'data').rglob('*.json*'))


def test_publisher_rejects_valid_looking_sample_path_in_another_folder(
    tmp_path: Path,
):
    log = make_lm_eval_log()
    staging_dir = tmp_path / 'staging'
    staged_model_dir = output_dir(staging_dir, log)
    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
        samples_path=SAMPLES_FILE,
        evaluation_id=log.evaluation_id,
        model_id=log.model_info.id,
        task_name='math_perturbed_full',
        output_dir=str(staged_model_dir),
        file_uuid=FILE_UUID,
        collection=log.evaluation_results[0].source_data.dataset_name,
        developer=log.model_info.developer,
    )
    assert detailed is not None
    detailed.file_path = (
        f'data/other-collection/developer/model/{FILE_UUID}_samples.jsonl'
    )
    log.detailed_evaluation_results = detailed

    with pytest.raises(ValueError, match='repository path and UUID'):
        publish_evaluation_logs(
            [log],
            tmp_path / 'data',
            [FILE_UUID],
            staged_output_dir=staging_dir,
        )

    assert not (tmp_path / 'data').exists()


def test_publisher_rejects_non_uuid4_filename(tmp_path: Path):
    log = make_lm_eval_log()

    with pytest.raises(ValueError, match='UUIDv4'):
        publish_evaluation_logs(
            [log],
            tmp_path / 'data',
            [str(uuid.uuid1())],
        )


def test_publisher_rejects_distinct_models_with_colliding_routes(
    tmp_path: Path,
):
    first = make_lm_eval_log()
    second = first.model_copy(deep=True)
    first.model_info.id = 'developer/family/model'
    second.model_info.id = 'developer/family_model'

    with pytest.raises(ValueError, match='same datastore directory'):
        publish_evaluation_logs(
            [first, second],
            tmp_path / 'data',
            [FILE_UUID, '9e6e9282-51d0-49ef-9728-61f53c235c37'],
        )

    assert not (tmp_path / 'data').exists()


def test_publisher_rollback_removes_only_its_empty_directories(
    tmp_path: Path, monkeypatch
):
    first = make_lm_eval_log()
    second = first.model_copy(deep=True)
    second.model_info.id = 'other-developer/other-model'
    second_uuid = '9e6e9282-51d0-49ef-9728-61f53c235c37'
    second_path = output_dir(tmp_path / 'data', second) / f'{second_uuid}.json'
    original_open = Path.open

    def failing_open(path: Path, mode='r', *args, **kwargs):
        if path == second_path and mode == 'xb':
            raise OSError('simulated write failure')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', failing_open)

    with pytest.raises(OSError, match='simulated write failure'):
        publish_evaluation_logs(
            [first, second],
            tmp_path / 'data',
            [FILE_UUID, second_uuid],
        )

    assert not (tmp_path / 'data').exists()


def test_lm_eval_cli_publishes_valid_batch(tmp_path: Path):
    output_dir = tmp_path / 'data'

    result = cli.main(
        [
            'convert',
            'lm_eval',
            '--log-path',
            str(RESULTS_FILE),
            '--output-dir',
            str(output_dir),
            '--source-organization-name',
            'TestOrg',
            '--evaluator-relationship',
            'first_party',
        ]
    )

    assert result == 0
    paths = sorted(output_dir.rglob('*.json'))
    assert len(paths) == 2
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


def test_lm_eval_cli_sample_failure_publishes_other_valid_tasks(
    tmp_path: Path,
):
    output_dir = tmp_path / 'data'

    with pytest.raises(SourceRecordsError, match='no upstream samples'):
        cli.main(
            [
                'convert',
                'lm_eval',
                '--log-path',
                str(RESULTS_FILE),
                '--output-dir',
                str(output_dir),
                '--source-organization-name',
                'TestOrg',
                '--evaluator-relationship',
                'first_party',
                '--include-samples',
            ]
        )

    aggregate_paths = list(output_dir.rglob('*.json'))
    sample_paths = list(output_dir.rglob('*.jsonl'))
    assert len(aggregate_paths) == 2
    assert len(sample_paths) == 1
    for aggregate_path in aggregate_paths:
        assert validate_file(aggregate_path).valid
    assert validate_file(sample_paths[0]).valid

    aggregates = [
        json.loads(path.read_text(encoding='utf-8')) for path in aggregate_paths
    ]
    assert sum('detailed_evaluation_results' in log for log in aggregates) == 1

    report_path = tmp_path / 'adapter_reports' / 'lm_eval_samples_failures.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['converted_records'] == 1
    assert report['failed_record_count'] == 1
    failure_record = report['failed_records'][0]['source_record']
    assert failure_record['task_name'] == 'math_rephrased_full'
    assert failure_record['searched_directory'] == str(LM_EVAL_DIR)
    assert failure_record['expected_samples_pattern'] == (
        'samples_math_rephrased_full_*.jsonl'
    )
