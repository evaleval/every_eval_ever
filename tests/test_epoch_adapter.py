"""Offline tests for the Epoch AI Benchmarking Hub adapter."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from every_eval_ever.adapters.epoch_ai import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import SourceRecordsError

FIXTURES = Path(__file__).parent / 'fixtures' / 'epoch_ai'


@pytest.fixture(scope='module')
def zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path in sorted(FIXTURES.glob('*.csv')):
            archive.writestr(
                zipfile.ZipInfo(path.name, (2026, 11, 13, 0, 0, 0)),
                path.read_bytes(),
            )
    return buffer.getvalue()


@pytest.fixture(scope='module')
def result(zip_bytes):
    dataset = adapter.load_zip(zip_bytes, 'fixture')
    return adapter.convert(dataset, '1.0', adapter.Registry())


def _by_version(result, benchmark):
    return {
        log.model_info.additional_details['epoch_model_version']: log
        for log in result.records
        if log.evaluation_results[0].evaluation_result_id == benchmark
    }


def test_row_accounting(result):
    # 3 gpqa + 3 aider + 4 mmlu + 2 os_world
    assert result.total_records == 12
    assert len(result.records) + len(result.failures) == 12
    reasons = {f.source_ref: f.reason for f in result.failures}
    assert 'no Model version' in reasons['aider_polyglot_external.csv:4']
    assert 'no Model version' in reasons['os_world_external.csv:2']
    assert 'no Organization' in reasons['mmlu_external.csv:4']
    assert "blank 'EM'" in reasons['mmlu_external.csv:5']
    assert len(result.failures) == 4
    assert len(result.records) == 8
    assert [e.source_ref for e in result.exclusions] == [
        'benchmark_metadata.csv:HLE'
    ]


def test_epoch_run_row(result):
    log = _by_version(result, 'gpqa-diamond')['grok-3-beta']
    res = log.evaluation_results[0]
    assert res.score_details.score == 0.7575757575757576  # Best score, not mean
    assert res.score_details.details['mean_score'] == '0.6758207070707071'
    assert res.metric_config.metric_name == 'Best score (across scorers)'
    assert (res.metric_config.min_score, res.metric_config.max_score) == (0, 1)
    assert res.metric_config.metric_unit == 'proportion'
    meta = log.source_metadata
    assert meta.evaluator_relationship.value == 'third_party'
    assert meta.additional_details['epoch_score_origin'] == 'epoch_run'
    assert meta.additional_details['license'] == 'CC-BY-4.0'
    assert meta.additional_details['epoch_snapshot'] == '2026-11-13'
    assert log.model_info.id == 'xai/grok-3-beta'
    assert log.model_info.additional_details['deployment_type'] == (
        'externally_managed'
    )
    assert log.model_info.additional_details['model_availability'] == (
        'closed_weights'
    )
    assert log.evaluation_id == (
        'epoch-ai/gpqa-diamond/grok-3-beta/FQ8Ykq4JWRSXnyF9rbmLc7/2026-11-13'
    )


def test_external_row_keeps_provenance_and_percent_scale(result):
    log = _by_version(result, 'aider-polyglot')['gpt-oss-120b_high']
    res = log.evaluation_results[0]
    assert res.score_details.score == 41.8
    assert res.metric_config.metric_unit == 'percent'
    assert (res.metric_config.min_score, res.metric_config.max_score) == (
        0, 100
    )
    meta = log.source_metadata
    assert meta.evaluator_relationship.value == 'other'
    assert meta.additional_details['epoch_score_origin'] == 'external'
    assert meta.additional_details['epoch_source'] == 'Aider LLM Leaderboards'
    # a blank provenance cell is absent, never invented
    assert 'epoch_source_link' not in meta.additional_details
    other = _by_version(result, 'aider-polyglot')['grok-3-beta']
    assert other.source_metadata.additional_details['epoch_source_link'] == (
        'https://aider.chat/docs/leaderboards/#polyglot-leaderboard'
    )
    assert 'Source' not in res.score_details.details  # provenance not duplicated


def test_variant_suffix_leaves_model_id(result):
    log = _by_version(result, 'gpqa-diamond')['claude-opus-4-5-20251101_16K']
    assert log.model_info.id == 'anthropic/claude-opus-4-5-20251101'
    assert log.model_info.additional_details['thinking_budget'] == '16K'
    log = _by_version(result, 'gpqa-diamond')['Qwen3-32B_none']
    assert log.model_info.id.endswith('/Qwen3-32B')
    assert log.model_info.additional_details['reasoning_effort'] == 'none'
    assert log.model_info.additional_details['model_availability'] == (
        'open_weights'
    )
    # a trailing token that is not a reasoning setting stays in the id
    assert adapter.split_variant('MiniCPM-V-2_6') == ('MiniCPM-V-2_6', None)
    assert adapter.split_variant('gpt-5_high') == ('gpt-5', 'high')


def test_organization_fallback_from_model_metadata(result):
    # the os_world row has a blank Organization; model_metadata supplies it
    log = _by_version(result, 'osworld')['claude-opus-4-5-20251101_16K']
    assert log.model_info.id.startswith('anthropic/')


def test_publish_and_validate(zip_bytes, tmp_path):
    from every_eval_ever.validator import validate_file

    args = adapter.parse_args([
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    archive = tmp_path / 'benchmark_data.zip'
    archive.write_bytes(zip_bytes)
    args.input = archive
    with pytest.raises(SourceRecordsError):  # failures exit non-zero
        adapter.run(args)
    files = sorted((tmp_path / 'data' / adapter.COLLECTION).glob('*/*/*.json'))
    assert len(files) == 8
    for path in files:
        EvaluationLog.model_validate(json.loads(path.read_text()))
        assert validate_file(path).valid
    report = json.loads(
        (tmp_path / 'adapter_reports' / f'{adapter.COLLECTION}_failures.json')
        .read_text()
    )
    assert report['converted_records'] == 8
    assert report['failed_record_count'] == 4


def test_duplicate_rows_are_excluded_not_published_twice(zip_bytes):
    dataset = adapter.load_zip(zip_bytes, 'fixture')
    rows = dataset.tables['gpqa_diamond.csv']
    rows.append({**rows[0], 'id': 'a-second-row-id'})
    result = adapter.convert(dataset, '1.0', adapter.Registry())
    assert len(result.records) == 8
    duplicates = [e for e in result.exclusions if 'duplicate' in e.reason]
    assert [e.source_ref for e in duplicates] == ['gpqa_diamond.csv:5']


def test_catalog_entry_fetches_by_url_on_a_weekly_schedule():
    from every_eval_ever.adapters import catalog

    spec = catalog.get('epoch_ai')
    argv = spec.argv('/tmp/staging/data')
    assert argv[argv.index('--url') + 1] == adapter.DATA_URL
    assert '--input' not in argv
    assert spec.collections == (adapter.COLLECTION,)
    assert (spec.cadence, spec.weekday) == ('weekly', 5)
    assert spec.captures_raw  # the zip is snapshotted via raw_capture
