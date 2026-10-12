from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
from every_eval_ever.converters.lm_eval.instance_level_adapter import (
    LMEvalInstanceLevelAdapter,
)
from every_eval_ever.helpers.io import datastore_output_dir
from every_eval_ever.validator.validation_core import validate_file

LM_EVAL_DIR = Path('tests/data/lm_eval')
RESULTS_FILE = LM_EVAL_DIR / 'results_2026-01-21T03-44-18.458309.json'
SAMPLES_FILE = (
    LM_EVAL_DIR / 'samples_math_perturbed_full_2026-01-21T03-44-18.458309.jsonl'
)
FILE_UUID = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'


def _log():
    return LMEvalAdapter().transform_from_file(
        RESULTS_FILE,
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'first_party',
        },
    )[0]


def _out(base: Path, log) -> Path:
    return datastore_output_dir(
        base,
        log.evaluation_results[0].source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
    )


def test_publisher_compresses_final_samples_and_updates_pointer(tmp_path: Path):
    log = _log()
    staging = tmp_path / 'staging'
    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
        samples_path=SAMPLES_FILE,
        evaluation_id=log.evaluation_id,
        model_id=log.model_info.id,
        task_name='math_perturbed_full',
        output_dir=str(_out(staging, log)),
        file_uuid=FILE_UUID,
        collection=log.evaluation_results[0].source_data.dataset_name,
        developer=log.model_info.developer,
    )
    assert detailed is not None
    log.detailed_evaluation_results = detailed

    [aggregate_path] = publish_evaluation_logs(
        [log],
        tmp_path / 'data',
        [FILE_UUID],
        staged_output_dir=staging,
        aggregate_compression='none',
        samples_compression='gz',
    )
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    detail = aggregate['detailed_evaluation_results']
    assert detail['file_path'].endswith(f'{FILE_UUID}_samples.jsonl.gz')
    sample_path = aggregate_path.with_name(f'{FILE_UUID}_samples.jsonl.gz')
    assert (
        detail['checksum']
        == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    )
    with gzip.open(sample_path, 'rt', encoding='utf-8') as file:
        rows = [json.loads(line) for line in file if line.strip()]
    assert len(rows) == detail['total_rows']
    report = validate_file(aggregate_path)
    assert report.valid, report.errors


def test_publisher_compresses_aggregate_at_shared_boundary(tmp_path: Path):
    log = _log()
    [aggregate_path] = publish_evaluation_logs(
        [log], tmp_path / 'data', [FILE_UUID], aggregate_compression='gz'
    )
    assert aggregate_path.name == f'{FILE_UUID}.json.gz'
    with gzip.open(aggregate_path, 'rt', encoding='utf-8') as file:
        assert json.load(file)['evaluation_id'] == log.evaluation_id
