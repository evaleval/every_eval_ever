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
    # archive-level facts stay out of the record (they would defeat
    # cross-run de-duplication); raw capture keeps the zip and its sha256
    assert 'epoch_snapshot' not in meta.additional_details
    assert 'epoch_zip_sha256' not in meta.additional_details
    assert log.eval_library.name == 'inspect_ai'
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
    # external data keeps its original licence: no blanket CC BY claim
    assert 'license' not in meta.additional_details
    assert 'retains its original licensing' in (
        meta.additional_details['license_note'])
    assert meta.additional_details['epoch_contribution_license'] == 'CC-BY-4.0'
    assert log.eval_library.name == 'unknown'


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
    assert spec.captures_raw


# -- review fixes -----------------------------------------------------------


def _fixture_zip(*, extra: dict[str, bytes] | None = None,
                 date=(2026, 11, 13, 0, 0, 0),
                 edit=None) -> bytes:
    """The fixture archive, optionally with extra members, other member
    timestamps, or one CSV's text edited."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path in sorted(FIXTURES.glob('*.csv')):
            data = path.read_bytes()
            if edit is not None and path.name == edit[0]:
                data = data.replace(edit[1], edit[2], 1)
            archive.writestr(zipfile.ZipInfo(path.name, date), data)
        for name, data in (extra or {}).items():
            archive.writestr(zipfile.ZipInfo(name, date), data)
    return buffer.getvalue()


def _published_fingerprints(content: bytes, tmp_path: Path, tag: str):
    """Run the adapter as the cron does and fingerprint what it publishes."""
    from every_eval_ever.cron.runner import record_fingerprint

    archive = tmp_path / f'{tag}.zip'
    archive.write_bytes(content)
    out = tmp_path / tag / 'data' / adapter.COLLECTION
    args = adapter.parse_args(['--output-dir', str(out)])
    args.input = archive
    with pytest.raises(SourceRecordsError):  # the fixture has failing rows
        adapter.run(args)
    return {
        record_fingerprint(json.loads(path.read_text()))
        for path in out.glob('*/*/*.json')
    }


def test_unrelated_archive_changes_do_not_make_records_new(tmp_path):
    base = _published_fingerprints(_fixture_zip(), tmp_path, 'base')
    assert len(base) == 8
    eci_only = _fixture_zip(extra={
        'epoch_capabilities_index/eci_scores.csv': b'Model version,ECI\nx,1\n',
    })
    assert _published_fingerprints(eci_only, tmp_path, 'eci') == base
    repackaged = _fixture_zip(date=(2026, 11, 14, 0, 0, 0))
    assert _published_fingerprints(repackaged, tmp_path, 'repack') == base
    one_score = _fixture_zip(
        edit=('mmlu_external.csv', b'v1:0,0.77,', b'v1:0,0.78,'))
    assert one_score != _fixture_zip()
    changed = _published_fingerprints(one_score, tmp_path, 'score')
    assert len(changed - base) == 1 and len(base - changed) == 1


def _dataset_with(rows_by_table: dict[str, list[dict]], models: list[dict]):
    dataset = adapter.load_zip(_fixture_zip(), 'fixture')
    for table, rows in rows_by_table.items():
        template = dataset.tables[table][0]
        dataset.tables[table].extend(
            {**{k: '' for k in template}, **row} for row in rows)
    template = dataset.tables['model_metadata.csv'][0]
    dataset.tables['model_metadata.csv'].extend(
        {**{k: '' for k in template}, **row} for row in models)
    return dataset


def test_qualified_and_provider_spellings_share_one_model_id():
    dataset = _dataset_with(
        {'aider_polyglot_external.csv': [
            {'Model version': 'openai/gpt-oss-120b_high',
             'Percent correct': '40.0', 'Organization': 'OpenAI',
             'id': 'recQualified'},
            {'Model version': 'chutes/gpt-oss-120b',
             'Percent correct': '39.0', 'Organization': 'OpenAI',
             'id': 'recProvider'},
        ]},
        [{'model_version': 'openai/gpt-oss-120b_high',
          'model_group': 'gpt-oss-120b',
          'accessibility': 'Open weights (unrestricted)'},
         {'model_version': 'chutes/gpt-oss-120b',
          'model_group': 'gpt-oss-120b',
          'accessibility': 'Open weights (unrestricted)'}],
    )
    result = adapter.convert(dataset, '1.0', adapter.Registry())
    logs = _by_version(result, 'aider-polyglot')
    plain = logs['gpt-oss-120b_high']
    qualified = logs['openai/gpt-oss-120b_high']
    served = logs['chutes/gpt-oss-120b']
    assert plain.model_info.id == qualified.model_info.id == (
        served.model_info.id) == 'openai/gpt-oss-120b'
    # the settings stay distinguishable
    assert qualified.model_info.additional_details['reasoning_effort'] == 'high'
    assert 'reasoning_effort' not in served.model_info.additional_details
    assert served.model_info.additional_details['inference_provider'] == (
        'chutes')
    assert qualified.evaluation_id != plain.evaluation_id
    assert adapter.canonical_model_name(
        'fireworks/kimi-k2p5', {'kimi-k2.5', 'fireworks/kimi-k2p5'}
    )[0] == 'kimi-k2.5'


def test_collaborative_models_use_their_published_repository():
    dataset = _dataset_with(
        {'mmlu_external.csv': [
            {'Model version': 'bloom', 'EM': '0.3',
             'Organization': 'Hugging Face,BigScience', 'id': 'recBloom'},
            {'Model version': 'starcoder2-3b', 'EM': '0.4',
             'Organization': 'Hugging Face,ServiceNow,Nvidia,BigCode',
             'id': 'recStarcoder'},
        ]},
        [{'model_version': v, 'model_group': v,
          'accessibility': 'Open weights (restricted use)'}
         for v in ('bloom', 'starcoder2-3b')],
    )
    result = adapter.convert(dataset, '1.0', adapter.Registry())
    logs = _by_version(result, 'mmlu')
    assert logs['bloom'].model_info.id == 'bigscience/bloom'
    assert logs['starcoder2-3b'].model_info.id == 'bigcode/starcoder2-3b'
    details = logs['starcoder2-3b'].model_info.additional_details
    assert json.loads(details['epoch_organizations']) == [
        'Hugging Face', 'ServiceNow', 'Nvidia', 'BigCode']
    assert details['model_id_strategy'] == 'reviewed_override'


def test_training_compute_is_kept_and_counts_for_duplicates():
    row = {'Model version': 'grok-3-beta', 'Percent correct': '53.3',
           'Organization': 'xAI', 'Training compute (FLOP)': '4.6e+26',
           'Training compute notes': 'Estimate from reported GPU count'}
    dataset = _dataset_with(
        {'aider_polyglot_external.csv': [
            {**row, 'Model version': 'grok-4', 'id': 'recA'},
            {**row, 'Model version': 'grok-4', 'id': 'recB',
             'Training compute notes': 'A different note'},
        ]},
        [],
    )
    result = adapter.convert(dataset, '1.0', adapter.Registry())
    kept = [log for log in result.records
            if log.model_info.additional_details['epoch_model_version']
            == 'grok-4']
    assert len(kept) == 2  # differing notes are not an exact duplicate
    details = kept[0].model_info.additional_details
    assert details['epoch_training_compute_flop'] == '4.6e+26'
    assert details['epoch_training_compute_notes'] == (
        'Estimate from reported GPU count')
    assert not [e for e in result.exclusions if 'duplicate' in e.reason]


def test_global_metric_ids_where_the_column_is_one():
    log = _by_version(
        adapter.convert(adapter.load_zip(_fixture_zip(), 'fixture'), '1.0'),
        'mmlu')['Baichuan-7B']
    assert log.evaluation_results[0].metric_config.metric_id == 'exact_match'
    log = _by_version(
        adapter.convert(adapter.load_zip(_fixture_zip(), 'fixture'), '1.0'),
        'gpqa-diamond')['grok-3-beta']
    assert log.evaluation_results[0].metric_config.metric_id == (
        'epoch-ai.gpqa-diamond.score')


def test_fetch_records_the_zip_in_raw_capture(monkeypatch, tmp_path):
    from every_eval_ever.helpers import raw_capture

    content = _fixture_zip()

    class Response:
        headers = {'Content-Type': 'application/zip'}

        def __init__(self, body):
            self.content = body

        def raise_for_status(self):
            return None

    calls = []
    monkeypatch.setattr(
        adapter.requests, 'get',
        lambda url, **kw: calls.append(url) or Response(content))
    sink = raw_capture.activate(tmp_path / 'raw')
    try:
        out = tmp_path / 'data' / adapter.COLLECTION
        args = adapter.parse_args(
            ['--url', adapter.DATA_URL, '--output-dir', str(out)])
        with pytest.raises(SourceRecordsError):
            adapter.run(args)
    finally:
        raw_capture.deactivate()
    assert calls == [adapter.DATA_URL]
    entries = sink.entries()
    assert [e['url'] for e in entries] == [adapter.DATA_URL]
    import hashlib
    assert entries[0]['sha256'] == hashlib.sha256(content).hexdigest()
    log = json.loads(next(out.glob('*/*/*.json')).read_text())
    assert log['source_metadata']['additional_details']['retrieved_from'] == (
        adapter.DATA_URL)
