"""Cron records must be findable later without scanning the datastore."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from every_eval_ever.cron import provenance
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validator.check_duplicate_entries import normalized_hash
from every_eval_ever.validator.validation_core import check_model_deployment

REFERENCE_RECORD = (
    Path(__file__).resolve().parent
    / 'data'
    / 'skill_reference_conversion'
    / 'data'
    / 'demo-source'
    / 'demo-org'
    / 'demo-model'
    / 'f3a1c0de-4b2e-4c1a-9f6d-1b7e5a2c8d40.json'
)
RUN_DATE = date(2026, 8, 10)


@pytest.fixture
def record() -> dict[str, Any]:
    return json.loads(REFERENCE_RECORD.read_text(encoding='utf-8'))


def details(record: dict[str, Any]) -> dict[str, str]:
    return record['source_metadata']['additional_details']


def test_stamp_records_type_date_adapter_and_run(record) -> None:
    stamped = provenance.stamp_cron_provenance(
        record,
        adapter='hle',
        run_date=RUN_DATE,
        run_url='https://github.com/evaleval/every_eval_ever/actions/runs/1',
    )

    assert details(stamped) == {
        'source_role': 'aggregator',
        'type_of_addition': 'cron',
        'cron_run_date': '2026-08-10',
        'cron_adapter': 'hle',
        'cron_run_url': (
            'https://github.com/evaleval/every_eval_ever/actions/runs/1'
        ),
    }
    assert provenance.is_cron_record(stamped)


def test_a_record_that_knows_both_inferred_fields_says_nothing(record) -> None:
    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    assert provenance.UNKNOWN_INFERRED_KEY not in details(stamped)


@pytest.mark.parametrize(
    ('known', 'expected'),
    [
        ({}, 'deployment_type,model_availability'),
        (
            {'deployment_type': 'externally_managed'},
            'model_availability',
        ),
        (
            {'model_availability': 'open_weights'},
            'deployment_type',
        ),
        (
            {
                'deployment_type': 'unknown',
                'model_availability': 'open_weights',
            },
            'deployment_type',
        ),
    ],
)
def test_the_inferred_fields_left_unknown_are_named(
    record, known, expected
) -> None:
    """The ticket defers these two, so a back-fill needs to find them.

    Both default to ``unknown`` in the model, so a record that omits one says
    the same thing as one that spells it out.
    """
    record['model_info']['additional_details'] = dict(known)

    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    assert details(stamped)[provenance.UNKNOWN_INFERRED_KEY] == expected


def test_run_url_is_omitted_when_not_running_in_ci(record) -> None:
    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    assert provenance.RUN_URL_KEY not in details(stamped)


def test_stamp_adds_details_when_the_adapter_set_none(record) -> None:
    del record['source_metadata']['additional_details']

    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    assert details(stamped)['type_of_addition'] == 'cron'


def test_stamp_leaves_the_input_untouched(record) -> None:
    before = json.dumps(record, sort_keys=True)

    provenance.stamp_cron_provenance(record, adapter='hle', run_date=RUN_DATE)

    assert json.dumps(record, sort_keys=True) == before


def test_stamp_changes_nothing_but_source_metadata(record) -> None:
    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    del stamped['source_metadata']
    expected = dict(record)
    del expected['source_metadata']
    assert stamped == expected


def test_stamped_record_still_validates(record) -> None:
    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    EvaluationLog.model_validate(stamped)


def test_stamp_does_not_disturb_the_deployment_axes(record) -> None:
    """The inferred axes are the adapter's business, not the cron's."""
    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )

    assert check_model_deployment(stamped) == []
    assert (
        stamped['model_info']['additional_details']
        == (record['model_info']['additional_details'])
    )


def test_a_record_without_the_axes_still_defaults_to_unknown(record) -> None:
    """An adapter that sets neither axis publishes ``unknown``/``unknown``.

    The library fills both in on validation, which is what the ticket asks
    for; pinning it here means a codegen change that dropped the default
    would fail loudly instead of silently shipping records without the axes.
    """
    del record['model_info']['additional_details']

    stamped = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )
    published = EvaluationLog.model_validate(stamped).model_dump(
        mode='json', exclude_none=True
    )

    assert published['model_info']['additional_details'] == {
        'deployment_type': 'unknown',
        'model_availability': 'unknown',
    }
    assert check_model_deployment(published) == []


def test_restamping_the_same_run_is_idempotent(record) -> None:
    once = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )
    twice = provenance.stamp_cron_provenance(
        once, adapter='hle', run_date=RUN_DATE
    )

    assert twice == once


@pytest.mark.parametrize(
    ('key', 'value', 'kwargs'),
    [
        ('type_of_addition', 'manual', {}),
        ('cron_run_date', '2020-01-01', {}),
        ('cron_adapter', 'mmlu_pro', {}),
        (
            'cron_run_url',
            'https://example.com/other',
            {'run_url': 'https://example.com/mine'},
        ),
    ],
)
def test_stamp_refuses_to_overwrite_a_conflicting_value(
    record, key, value, kwargs
) -> None:
    record['source_metadata']['additional_details'][key] = value

    with pytest.raises(provenance.ProvenanceConflictError, match=key):
        provenance.stamp_cron_provenance(
            record, adapter='hle', run_date=RUN_DATE, **kwargs
        )


def test_stamp_rejects_a_record_without_source_metadata(record) -> None:
    del record['source_metadata']

    with pytest.raises(ValueError, match='source_metadata'):
        provenance.stamp_cron_provenance(
            record, adapter='hle', run_date=RUN_DATE
        )


def test_stamp_rejects_non_object_additional_details(record) -> None:
    record['source_metadata']['additional_details'] = 'aggregator'

    with pytest.raises(ValueError, match='must be an object'):
        provenance.stamp_cron_provenance(
            record, adapter='hle', run_date=RUN_DATE
        )


def test_stamp_rejects_an_invalid_record(record) -> None:
    del record['evaluation_id']

    with pytest.raises(Exception):
        provenance.stamp_cron_provenance(
            record, adapter='hle', run_date=RUN_DATE
        )


def test_stamping_changes_the_duplicate_fingerprint(record) -> None:
    """Why the runner must fingerprint before stamping.

    ``normalized_hash`` ignores ``evaluation_id`` and ``retrieved_timestamp``
    but not ``cron_run_date``, so a record fingerprinted after stamping would
    look new every single day and the de-duplication ledger would never
    match anything.
    """
    stamped_today = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=RUN_DATE
    )
    stamped_tomorrow = provenance.stamp_cron_provenance(
        record, adapter='hle', run_date=date(2026, 8, 11)
    )

    assert normalized_hash(record) == normalized_hash(record)
    assert normalized_hash(stamped_today) != normalized_hash(stamped_tomorrow)
    assert normalized_hash(stamped_today) != normalized_hash(record)


def test_is_cron_record_is_false_for_a_hand_submitted_record(record) -> None:
    assert not provenance.is_cron_record(record)
    assert not provenance.is_cron_record({})
    assert not provenance.is_cron_record({'source_metadata': 'nope'})
