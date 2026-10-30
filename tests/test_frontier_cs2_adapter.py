"""Unit tests for the FrontierCS 2 benchmark adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters import catalog
from every_eval_ever.adapters.frontier_cs2 import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file

FIXTURES_DIR = Path(__file__).parent / 'data' / 'frontier_cs2'
SAMPLE_RESULTS_FILE = FIXTURES_DIR / 'sample_formal_results.json'
SAMPLE_MANIFEST_FILE = FIXTURES_DIR / 'sample_trace_manifest.tsv'


@pytest.fixture
def sample_payload() -> dict:
    return json.loads(SAMPLE_RESULTS_FILE.read_text(encoding='utf-8'))


@pytest.fixture
def sample_manifest() -> dict[str, dict[str, str]]:
    text = SAMPLE_MANIFEST_FILE.read_text(encoding='utf-8')
    return adapter.parse_trace_manifest(text)


def test_catalog_registration() -> None:
    """The adapter must be registered in the catalog with valid specification."""
    spec = catalog.get('frontier_cs2')
    assert spec.key == 'frontier_cs2'
    assert spec.collections == ('frontier-cs-2',)
    assert spec.output_scope == 'collection'
    assert spec.captures_raw is True
    assert spec.skip_if_unchanged is True
    assert spec.cadence == 'weekly'
    assert spec.weekday == 0


def test_offline_conversion_validates_schema_and_gate(
    sample_payload: dict,
    sample_manifest: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """Sample records must convert cleanly and pass all schema + semantic checks."""
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result = adapter.convert_payload(
        formal_results=sample_payload,
        manifest_by_run_id=sample_manifest,
        output_dir=output_dir,
        retrieved_timestamp='1730000000.0',
    )

    assert result.total_records == 2
    assert len(result.records) == 2
    assert not result.failures

    written = adapter.save_evaluation_logs(result.records)
    assert len(written) == 2

    for path in written:
        report = validate_file(path)
        assert report.valid, f'{path}: {report.errors}'
        assert not report.warnings, f'{path}: {report.warnings}'


def test_model_identity_and_deployment_axes(
    sample_payload: dict,
    sample_manifest: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """Models must adhere to canonical identities and deployment enums."""
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result = adapter.convert_payload(
        formal_results=sample_payload,
        manifest_by_run_id=sample_manifest,
        output_dir=output_dir,
        retrieved_timestamp='1730000000.0',
    )

    logs_by_id = {out.eval_log.model_info.id: out.eval_log for out in result.records}
    assert 'google/gemini-4-argon' in logs_by_id
    assert 'openai/gpt-6-astra' in logs_by_id

    gemini_log = logs_by_id['google/gemini-4-argon']
    assert gemini_log.model_info.additional_details.get('deployment_type') == 'externally_managed'
    assert gemini_log.model_info.additional_details.get('model_availability') == 'closed_weights'

    astra_log = logs_by_id['openai/gpt-6-astra']
    assert astra_log.model_info.additional_details.get('deployment_type') == 'externally_managed'
    assert astra_log.model_info.additional_details.get('model_availability') == 'closed_weights'


def test_null_score_handled_as_unscored_failure(
    sample_payload: dict,
    sample_manifest: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """A run with null score must be mapped to 0.0 with failure details."""
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result = adapter.convert_payload(
        formal_results=sample_payload,
        manifest_by_run_id=sample_manifest,
        output_dir=output_dir,
        retrieved_timestamp='1730000000.0',
    )

    gemini_log = next(out.eval_log for out in result.records if out.eval_log.model_info.id == 'google/gemini-4-argon')
    failed_eval = next(r for r in gemini_log.evaluation_results if r.evaluation_result_id == 'graph_mining_method_selection/gemini-agy')

    assert failed_eval.score_details.score == 0.0
    assert failed_eval.score_details.details['score_status'] == 'unscored_failure'
    assert failed_eval.score_details.details['failure_code'] == 'agent_exited_without_final_submission'


def test_parse_args_defaults() -> None:
    """CLI parser defaults should match catalog expectations."""
    args = adapter.parse_args([])
    assert args.output_dir == adapter.DEFAULT_OUTPUT_DIR
    assert args.emit_source_version is False
