"""Unit tests for the Scale SEAL Humanity's Last Exam adapter."""

from __future__ import annotations

from every_eval_ever.eval_types import EvaluationLog
from utils.hle import adapter


def sample_rows() -> list[dict]:
    return [
        {
            'model': 'gemini-3.1-pro-preview (thinking high)',
            'version': '',
            'rank': 1,
            'score': 46.44,
            'confidenceInterval_upper': 1.96,
            'contaminationMessage': 'warning text',
            'company': 'google',
            'createdAt': '2026-04-10T15:51:06.000Z',
            'deprecated': False,
            'calibrationError': 51,
            'maxScore': 49.852,
        },
        {
            'model': 'claude-opus-4-7',
            'version': '',
            'rank': 6,
            'score': 36.2,
            'confidenceInterval_upper': 1.88,
            'company': 'anthropic',
            'createdAt': '2026-03-01T00:00:00.000Z',
            'deprecated': False,
            'calibrationError': 47,
            'maxScore': 49.852,
        },
        {
            'model': 'kimi-k2',
            'version': '',
            'rank': 30,
            'score': 12.0,
            'confidenceInterval_upper': 1.5,
            'company': 'moonshot',
            'createdAt': '2025-12-01T00:00:00.000Z',
            'deprecated': False,
            'calibrationError': None,
            'maxScore': 49.852,
        },
        # Row missing a score must be silently dropped.
        {
            'model': 'broken-row',
            'company': 'openai',
            'score': None,
        },
    ]


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    assert len(bundles) == 3
    for log, _, _ in bundles:
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_organization_name == 'Scale'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert (
            validated.source_metadata.evaluator_relationship.value
            == 'third_party'
        )


def test_accuracy_confidence_interval_is_score_plus_minus_half_width():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    by_id = {b[0].model_info.id: b[0] for b in bundles}
    gemini = by_id['google/gemini-3.1-pro-preview-thinking-high']
    accuracy = next(
        r
        for r in gemini.evaluation_results
        if r.evaluation_result_id == 'hle/accuracy'
    )
    ci = accuracy.score_details.uncertainty.confidence_interval
    assert ci.lower == 46.44 - 1.96
    assert ci.upper == 46.44 + 1.96
    assert ci.confidence_level == 0.95


def test_calibration_error_optional_and_lower_is_better():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    by_id = {b[0].model_info.id: b[0] for b in bundles}

    claude = by_id['anthropic/claude-opus-4-7']
    calib = next(
        r
        for r in claude.evaluation_results
        if r.evaluation_result_id == 'hle/calibration_error'
    )
    assert calib.metric_config.lower_is_better is True
    assert calib.score_details.score == 47.0

    kimi = by_id['moonshotai/kimi-k2']
    calib_ids = {r.evaluation_result_id for r in kimi.evaluation_results}
    assert 'hle/calibration_error' not in calib_ids
    assert 'hle/accuracy' in calib_ids


def test_company_to_developer_mapping_is_applied():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    devs = {b[1] for b in bundles}
    assert 'moonshotai' in devs
    assert 'moonshot' not in devs


def test_evaluation_id_uses_sanitized_model_id():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    ids = {b[0].evaluation_id for b in bundles}
    assert 'hle/google_gemini-3.1-pro-preview-thinking-high/123.0' in ids


def test_duplicate_canonical_id_raises():
    rows = sample_rows() + [
        {
            'model': 'CLAUDE-OPUS-4-7',
            'company': 'anthropic',
            'score': 36.2,
            'rank': 6,
            'confidenceInterval_upper': 1.88,
        }
    ]
    try:
        adapter.make_logs(rows, retrieved_timestamp='123.0')
    except ValueError as exc:
        assert 'Duplicate model id' in str(exc)
    else:
        raise AssertionError('expected duplicate-model-id failure')


def test_metric_config_uses_percent_scale_with_judge():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    log = bundles[0][0]
    accuracy = next(
        r
        for r in log.evaluation_results
        if r.evaluation_result_id == 'hle/accuracy'
    )
    assert accuracy.metric_config.min_score == 0.0
    assert accuracy.metric_config.max_score == 100.0
    assert accuracy.metric_config.metric_unit == 'percent'
    assert accuracy.metric_config.lower_is_better is False
    judge = accuracy.metric_config.llm_scoring.judges[0]
    assert judge.model_info.id == adapter.JUDGE_MODEL_ID


def test_parse_rsc_payload_handles_minimal_chunk():
    rows_json = (
        '[{\\"model\\":\\"foo\\",\\"company\\":\\"openai\\",\\"rank\\":1,'
        '\\"score\\":12.3,\\"confidenceInterval_upper\\":0.5,'
        '\\"calibrationError\\":40}]'
    )
    html = (
        f'<html><script>self.__next_f.push([1,"{rows_json}"])</script></html>'
    )
    rows = adapter.parse_rsc_payload(html)
    assert rows == [
        {
            'model': 'foo',
            'company': 'openai',
            'rank': 1,
            'score': 12.3,
            'confidenceInterval_upper': 0.5,
            'calibrationError': 40,
        }
    ]


def test_parse_rsc_payload_raises_when_no_payload_present():
    try:
        adapter.parse_rsc_payload('<html><body>nothing here</body></html>')
    except ValueError as exc:
        assert 'No __next_f payload' in str(exc)
    else:
        raise AssertionError('expected ValueError for missing payload')
