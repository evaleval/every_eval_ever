"""Cron records must be identifiable as cron records, and still validate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import every_eval_ever.eval_types as ET
from every_eval_ever.cron.stamp import (
    ADAPTER_KEY,
    CRON_ADDITION_TYPE,
    INFERRED_MODEL_FIELDS,
    RUN_DATE_KEY,
    RUN_URL_KEY,
    TYPE_OF_ADDITION_KEY,
    UNKNOWN,
    UNKNOWN_FIELDS_KEY,
    StampConflict,
    aggregate_records,
    stamp_file,
    stamp_payload,
    stamp_tree,
)
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_log
from every_eval_ever.validate import validate_file

RUN_DATE = '2026-08-11'


def _log(
    *,
    model_details: dict[str, str] | None = None,
    source_details: dict[str, str] | None = None,
    model_id: str = 'dev/model',
) -> ET.EvaluationLog:
    return ET.EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'src/{model_id}/1700000000.0',
        retrieved_timestamp='1700000000.0',
        source_metadata=ET.SourceMetadata(
            source_name='Example Leaderboard',
            source_type='documentation',
            source_organization_name='Example',
            evaluator_relationship='third_party',
            additional_details=source_details,
        ),
        eval_library=ET.EvalLibrary(name='unknown', version='unknown'),
        model_info=ET.ModelInfo(
            name='model',
            id=model_id,
            developer='dev',
            additional_details=model_details
            if model_details is not None
            else {
                'deployment_type': 'externally_managed',
                'model_availability': 'closed_weights',
            },
        ),
        evaluation_results=[
            ET.EvaluationResult(
                evaluation_name='example',
                source_data=ET.SourceDataUrl(
                    dataset_name='example-collection',
                    source_type='url',
                    url=['https://example.invalid/leaderboard'],
                ),
                metric_config=ET.MetricConfig(
                    metric_name='accuracy',
                    lower_is_better=False,
                    score_type=ET.ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ET.ScoreDetails(score=0.5),
            )
        ],
    )


def _write(tmp_path: Path, log: ET.EvaluationLog) -> Path:
    return save_evaluation_log(
        log,
        base_dir=tmp_path / 'data' / 'example-collection',
        developer=log.model_info.developer,
        model_name=log.model_info.id.split('/')[-1],
    )


def test_stamp_records_that_the_cron_added_the_record_and_when():
    stamped, _ = stamp_payload(
        json.loads(_log().model_dump_json(exclude_none=True)),
        adapter='vals_ai',
        run_date=RUN_DATE,
    )
    details = stamped['source_metadata']['additional_details']
    assert details[TYPE_OF_ADDITION_KEY] == CRON_ADDITION_TYPE
    assert details[RUN_DATE_KEY] == RUN_DATE
    assert details[ADAPTER_KEY] == 'vals_ai'


def test_stamp_values_are_all_strings():
    # The schema types additional_details as a string map.
    stamped, _ = stamp_payload(
        json.loads(_log().model_dump_json(exclude_none=True)),
        adapter='vals_ai',
        run_date=RUN_DATE,
        run_url='https://example.invalid/run/1',
    )
    details = stamped['source_metadata']['additional_details']
    assert all(isinstance(value, str) for value in details.values())
    assert details[RUN_URL_KEY] == 'https://example.invalid/run/1'


def test_stamp_keeps_details_the_adapter_already_recorded():
    stamped, _ = stamp_payload(
        json.loads(
            _log(source_details={'leaderboard_version': '1.2'}).model_dump_json(
                exclude_none=True
            )
        ),
        adapter='hle',
        run_date=RUN_DATE,
    )
    details = stamped['source_metadata']['additional_details']
    assert details['leaderboard_version'] == '1.2'
    assert details[TYPE_OF_ADDITION_KEY] == CRON_ADDITION_TYPE


def test_stamped_record_still_validates(tmp_path: Path):
    path = _write(tmp_path, _log())
    stamp_file(path, adapter='vals_ai', run_date=RUN_DATE)
    report = validate_file(path)
    assert report.valid, report.errors


def test_stamp_leaves_the_rest_of_the_record_byte_identical(tmp_path: Path):
    path = _write(tmp_path, _log())
    before = json.loads(path.read_text(encoding='utf-8'))
    stamp_file(path, adapter='vals_ai', run_date=RUN_DATE)
    after = json.loads(path.read_text(encoding='utf-8'))

    del after['source_metadata']['additional_details']
    assert after == before


def test_stamp_does_not_overwrite_deployment_axes_the_source_stated(
    tmp_path: Path,
):
    path = _write(tmp_path, _log())
    unknown = stamp_file(path, adapter='vals_ai', run_date=RUN_DATE)
    details = json.loads(path.read_text(encoding='utf-8'))['model_info'][
        'additional_details'
    ]
    assert unknown == []
    assert details['deployment_type'] == 'externally_managed'
    assert details['model_availability'] == 'closed_weights'
    assert (
        UNKNOWN_FIELDS_KEY
        not in (
            json.loads(path.read_text(encoding='utf-8'))['source_metadata'][
                'additional_details'
            ]
        )
    )


def test_unknown_deployment_axes_are_named_on_the_record():
    payload = json.loads(_log().model_dump_json(exclude_none=True))
    payload['model_info']['additional_details'] = {}

    stamped, unknown = stamp_payload(
        payload, adapter='mt_bench', run_date=RUN_DATE
    )

    assert unknown == list(INFERRED_MODEL_FIELDS)
    model_details = stamped['model_info']['additional_details']
    assert model_details['deployment_type'] == UNKNOWN
    assert model_details['model_availability'] == UNKNOWN
    # Recorded on the record itself, so a later fix can find exactly these.
    source_details = stamped['source_metadata']['additional_details']
    assert source_details[UNKNOWN_FIELDS_KEY] == (
        'deployment_type,model_availability'
    )


def test_one_stated_axis_is_not_reported_as_unknown():
    payload = json.loads(_log().model_dump_json(exclude_none=True))
    payload['model_info']['additional_details'] = {
        'model_availability': 'open_weights'
    }

    stamped, unknown = stamp_payload(
        payload, adapter='mt_bench', run_date=RUN_DATE
    )

    assert unknown == ['deployment_type']
    details = stamped['model_info']['additional_details']
    assert details['deployment_type'] == UNKNOWN
    assert details['model_availability'] == 'open_weights'


def test_record_with_another_addition_type_is_refused():
    payload = json.loads(
        _log(source_details={TYPE_OF_ADDITION_KEY: 'manual'}).model_dump_json(
            exclude_none=True
        )
    )
    with pytest.raises(StampConflict, match='manual'):
        stamp_payload(payload, adapter='vals_ai', run_date=RUN_DATE)


def test_stamping_an_already_stamped_record_is_stable(tmp_path: Path):
    path = _write(tmp_path, _log())
    stamp_file(path, adapter='vals_ai', run_date=RUN_DATE)
    first = path.read_text(encoding='utf-8')
    stamp_file(path, adapter='vals_ai', run_date=RUN_DATE)
    assert path.read_text(encoding='utf-8') == first


def test_stamp_tree_covers_every_record_and_counts_unknowns(tmp_path: Path):
    _write(tmp_path, _log(model_id='dev/first'))
    _write(
        tmp_path,
        _log(model_id='dev/second', model_details={}),
    )

    summary = stamp_tree(
        tmp_path / 'data', adapter='vals_ai', run_date=RUN_DATE
    )

    assert summary.stamped == 2
    assert summary.unknown_inferred == {
        'deployment_type': 1,
        'model_availability': 1,
    }
    for path in summary.paths:
        details = json.loads(path.read_text(encoding='utf-8'))[
            'source_metadata'
        ]['additional_details']
        assert details[TYPE_OF_ADDITION_KEY] == CRON_ADDITION_TYPE


def test_sample_companions_are_not_treated_as_aggregates(tmp_path: Path):
    path = _write(tmp_path, _log())
    companion = path.with_name(f'{path.stem}_samples.jsonl')
    companion.write_text('{}\n', encoding='utf-8')

    found = aggregate_records(tmp_path / 'data')

    assert found == [path]


def test_a_companion_named_with_a_json_extension_is_not_an_aggregate(
    tmp_path: Path,
):
    # Recognised by the `_samples` stem, not the extension, so widening the glob
    # to '*.json*' cannot start pulling companions into the stamp.
    path = _write(tmp_path, _log())
    companion = path.with_name(f'{path.stem}_samples.json')
    companion.write_text('{}\n', encoding='utf-8')

    assert aggregate_records(tmp_path / 'data') == [path]


def test_stamping_a_tree_leaves_companions_untouched(tmp_path: Path):
    path = _write(tmp_path, _log())
    companion = path.with_name(f'{path.stem}_samples.jsonl')
    companion.write_text('{"untouched": true}\n', encoding='utf-8')

    stamp_tree(tmp_path / 'data', adapter='openeval', run_date=RUN_DATE)

    assert companion.read_text(encoding='utf-8') == '{"untouched": true}\n'
