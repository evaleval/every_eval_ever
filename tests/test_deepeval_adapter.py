"""Unit tests for the DeepEval converter.

The committed fixture is a real export: deepeval 4.3.4's
`deepeval test run` over a six-case suite using only deterministic
metrics (ExactMatch, PatternMatch, and ToolCorrectness in exact-match
mode), written to a results folder as test_run_20261122_001137.json.
The smaller payloads built inline below are synthetic examples of the
format, used to reach branches a single real export does not contain
(errored metrics, conversational cases, missing optional fields).
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from every_eval_ever.converters.deepeval.adapter import DeepEvalAdapter
from every_eval_ever.converters.deepeval.instance_level_adapter import (
    DeepEvalInstanceLevelAdapter,
)
from every_eval_ever.converters.deepeval.utils import (
    metric_key,
    parse_run_timestamp,
)
from every_eval_ever.eval_types import EvaluatorRelationship
from every_eval_ever.helpers.io import SourceRecordsError

DATA_DIR = Path('tests/data/deepeval')
RUN_FILE = DATA_DIR / 'test_run_20261122_001137.json'
MODEL_ID = 'example/support-agent-7b'


def _make_metadata_args(**overrides):
    args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'model_id': MODEL_ID,
    }
    args.update(overrides)
    return args


def _convert_fixture():
    adapter = DeepEvalAdapter()
    logs = adapter.transform_from_file(RUN_FILE, _make_metadata_args())
    assert len(logs) == 1
    return adapter, logs[0]


# ── Utility tests ──────────────────────────────────────────────────────


def test_metric_key_normalizes_display_names():
    assert metric_key('Exact Match') == 'exact_match'
    assert metric_key('Tool Correctness') == 'tool_correctness'
    assert metric_key('GEval') == 'geval'
    assert metric_key('  Answer Relevancy ') == 'answer_relevancy'


def test_parse_run_timestamp():
    # The file name carries the run's local time; the epoch follows the
    # converting machine's timezone, as it does in the converter.
    expected = int(datetime(2026, 11, 22, 0, 11, 37).timestamp())
    assert parse_run_timestamp('test_run_20261122_001137.json') == expected
    assert parse_run_timestamp('.latest_run_full.json') is None
    assert parse_run_timestamp('test_run_20261340_001137.json') is None


# ── Aggregate conversion of the real export ────────────────────────────


def test_fixture_maps_run_level_fields():
    _, log = _convert_fixture()
    # The export's file name carries the run time (2026-11-22 00:11:37
    # local); the record keys on it, in the converting machine's
    # timezone, rather than on the conversion time.
    run_key = str(int(datetime(2026, 11, 22, 0, 11, 37).timestamp()))
    assert log.evaluation_id == (
        f'test_support_agent_eval/{MODEL_ID}/{run_key}'
    )
    assert log.evaluation_timestamp == run_key
    assert log.model_info.id == MODEL_ID
    assert log.model_info.name == MODEL_ID
    assert log.model_info.developer == 'example'
    assert log.eval_library.name == 'deepeval'
    assert log.source_metadata.source_name == 'deepeval'
    # Every fixture metric has known bounds, so no unknown-bounds count
    # is reported (the key is present only when the count is positive).
    assert log.source_metadata.additional_details is None


def test_fixture_maps_one_result_per_metric():
    _, log = _convert_fixture()
    results = {r.evaluation_result_id: r for r in log.evaluation_results}
    assert set(results) == {
        'exact_match',
        'tool_correctness',
        'pattern_match',
    }

    exact = results['exact_match']
    assert exact.evaluation_name == 'test_support_agent_eval'
    assert exact.metric_config.metric_name == 'exact_match'
    # The registry carries exact match, so the id joins across sources.
    assert exact.metric_config.metric_id == 'exact-match'
    assert exact.metric_config.min_score == 0.0
    assert exact.metric_config.max_score == 1.0
    assert exact.metric_config.metric_parameters == {'threshold': 0.5}
    assert (
        exact.metric_config.additional_details['deepeval_metric_name']
        == 'Exact Match'
    )
    assert exact.score_details.score == pytest.approx(2 / 3)
    assert exact.score_details.details == {
        'passes': '2',
        'fails': '1',
        'errors': '0',
    }
    uncertainty = exact.score_details.uncertainty
    assert uncertainty.num_samples == 3
    assert uncertainty.standard_deviation == pytest.approx(0.5773502691896257)
    assert uncertainty.standard_error.value == pytest.approx(1 / 3)
    assert uncertainty.standard_error.method == 'analytic'
    assert exact.source_data.dataset_name == 'test_support_agent_eval'
    assert exact.source_data.additional_details == {
        'test_file': 'test_support_agent_eval.py'
    }

    tool = results['tool_correctness']
    assert tool.metric_config.metric_id == 'deepeval.tool_correctness'
    assert tool.score_details.score == pytest.approx(0.5)
    # The run file names the judge model but carries no judge prompt, so
    # the judge lands in additional_details, not the typed llm_scoring.
    assert tool.metric_config.llm_scoring is None
    assert (
        tool.metric_config.additional_details['evaluation_model'] == 'fake-llm'
    )

    pattern = results['pattern_match']
    assert pattern.metric_config.metric_id == 'deepeval.pattern_match'
    assert pattern.metric_config.metric_parameters == {'threshold': 1.0}


def test_fixture_instance_rows_join_to_results():
    adapter, log = _convert_fixture()
    raw = adapter.raw_by_evaluation_id[log.evaluation_id]
    rows = DeepEvalInstanceLevelAdapter().transform_samples(
        raw,
        evaluation_id=log.evaluation_id,
        model_id=MODEL_ID,
        evaluation_name='test_support_agent_eval',
    )
    # Six test cases, one scored by two metrics.
    assert len(rows) == 7
    by_pair = {(row.sample_id, row.evaluation_result_id): row for row in rows}
    assert by_pair[('0', 'exact_match')].evaluation.score == 1.0
    assert by_pair[('0', 'exact_match')].evaluation.is_correct is True
    assert by_pair[('1', 'exact_match')].evaluation.score == 0.0
    assert by_pair[('1', 'exact_match')].evaluation.is_correct is False
    assert by_pair[('4', 'pattern_match')].evaluation.score == 1.0
    assert by_pair[('4', 'exact_match')].evaluation.score == 1.0

    row = by_pair[('2', 'tool_correctness')]
    assert row.input.raw == 'Where is my order 12345?'
    assert row.output.raw == ['Your order 12345 is in transit.']
    assert row.metadata['evaluation_model'] == 'fake-llm'
    assert row.metadata['threshold'] == '0.5'
    assert row.metadata['test_case_name'] == 'test_tool_correctness_lookup'
    assert row.sample_hash
    assert row.token_usage is None
    assert row.performance is None


# ── Synthetic payloads: branches the real export does not reach ────────


def _run_payload(**overrides):
    payload = {
        'testFile': 'test_synthetic.py',
        'testCases': [
            {
                'name': 'case_a',
                'input': 'q',
                'actualOutput': 'a',
                'expectedOutput': 'a',
                'success': True,
                'metricsData': [
                    {
                        'name': 'Exact Match',
                        'threshold': 0.5,
                        'success': True,
                        'score': 1.0,
                    }
                ],
                'order': 0,
            }
        ],
        'conversationalTestCases': [],
        'metricsScores': [
            {
                'metric': 'Exact Match',
                'scores': [1.0],
                'passes': 1,
                'fails': 0,
                'errors': 0,
            }
        ],
        'testPassed': 1,
        'testFailed': 0,
    }
    payload.update(overrides)
    return payload


def _convert_payload(payload, tmp_path, name='test_run_20260101_120000.json'):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    adapter = DeepEvalAdapter()
    return adapter, adapter.transform_from_file(path, _make_metadata_args())[0]


def test_dataset_alias_wins_over_test_file_stem(tmp_path):
    payload = _run_payload(
        datasetAlias='support-dataset',
        datasetId='ds-123',
        datasetVersion='3',
    )
    _, log = _convert_payload(payload, tmp_path)
    result = log.evaluation_results[0]
    assert result.evaluation_name == 'support-dataset'
    assert result.source_data.dataset_name == 'support-dataset'
    assert result.source_data.additional_details['dataset_id'] == 'ds-123'
    assert result.source_data.additional_details['dataset_version'] == '3'
    assert log.evaluation_id.startswith(f'support-dataset/{MODEL_ID}/')


def test_wrapped_latest_test_run_payload_is_unwrapped(tmp_path):
    _, log = _convert_payload(
        {'testRunData': _run_payload()}, tmp_path, name='snapshot.json'
    )
    assert log.evaluation_results[0].metric_config.metric_name == (
        'exact_match'
    )
    # No run timestamp in the file name: the record keys on its stem.
    assert log.evaluation_id.endswith('/snapshot')
    assert log.evaluation_timestamp is None


def test_missing_model_id_is_an_error(tmp_path):
    path = tmp_path / 'test_run_20260101_120000.json'
    path.write_text(json.dumps(_run_payload()), encoding='utf-8')
    args = _make_metadata_args()
    del args['model_id']
    with pytest.raises(ValueError, match='model_id is required'):
        DeepEvalAdapter().transform_from_file(path, args)


def test_errored_metric_is_counted_not_published(tmp_path):
    payload = _run_payload()
    payload['metricsScores'].append(
        {
            'metric': 'Answer Relevancy',
            'scores': [],
            'passes': 0,
            'fails': 0,
            'errors': 1,
        }
    )
    _, log = _convert_payload(payload, tmp_path)
    assert len(log.evaluation_results) == 1
    assert log.source_metadata.additional_details == {
        'metrics_without_scores': '1'
    }


def test_run_with_no_scored_metric_is_an_error(tmp_path):
    payload = _run_payload(
        metricsScores=[
            {
                'metric': 'Answer Relevancy',
                'scores': [],
                'passes': 0,
                'fails': 0,
                'errors': 2,
            }
        ]
    )
    with pytest.raises(ValueError, match='no metric with a score'):
        _convert_payload(payload, tmp_path)


def test_unknown_metric_keeps_unknown_bounds(tmp_path):
    payload = _run_payload()
    payload['metricsScores'] = [
        {
            'metric': 'My Custom Score',
            'scores': [3.5],
            'passes': 1,
            'fails': 0,
            'errors': 0,
        }
    ]
    payload['testCases'][0]['metricsData'] = [
        {
            'name': 'My Custom Score',
            'threshold': 0.5,
            'success': True,
            'score': 3.5,
        }
    ]
    _, log = _convert_payload(payload, tmp_path)
    result = log.evaluation_results[0]
    assert result.metric_config.metric_name == 'my_custom_score'
    assert result.metric_config.metric_id == 'deepeval.my_custom_score'
    assert result.metric_config.min_score is None
    assert result.metric_config.max_score is None
    assert result.score_details.score == pytest.approx(3.5)
    # A single score has a spread of zero and no standard error.
    uncertainty = result.score_details.uncertainty
    assert uncertainty.num_samples == 1
    assert uncertainty.standard_deviation == 0.0
    assert uncertainty.standard_error is None
    assert log.source_metadata.additional_details == {
        'metrics_with_unknown_bounds': '1'
    }


def test_mixed_thresholds_are_not_summarized(tmp_path):
    payload = _run_payload()
    payload['testCases'].append(
        {
            'name': 'case_b',
            'input': 'q2',
            'actualOutput': 'b',
            'expectedOutput': 'b',
            'success': True,
            'metricsData': [
                {
                    'name': 'Exact Match',
                    'threshold': 0.9,
                    'success': True,
                    'score': 1.0,
                }
            ],
            'order': 1,
        }
    )
    payload['metricsScores'][0]['scores'] = [1.0, 1.0]
    payload['metricsScores'][0]['passes'] = 2
    _, log = _convert_payload(payload, tmp_path)
    assert log.evaluation_results[0].metric_config.metric_parameters is None


def test_conversational_cases_are_counted_not_expanded(tmp_path):
    payload = _run_payload()
    payload['conversationalTestCases'] = [
        {
            'name': 'convo',
            'turns': [
                {
                    'role': 'user',
                    'content': 'hi',
                    'metricsData': [
                        {
                            'name': 'Exact Match',
                            'threshold': 0.5,
                            'success': True,
                            'score': 1.0,
                        }
                    ],
                }
            ],
        }
    ]
    adapter, log = _convert_payload(payload, tmp_path)
    assert log.source_metadata.additional_details == {
        'conversational_test_cases': '1'
    }
    raw = adapter.raw_by_evaluation_id[log.evaluation_id]
    rows = DeepEvalInstanceLevelAdapter().transform_samples(
        raw, log.evaluation_id, MODEL_ID, 'test_synthetic'
    )
    assert len(rows) == 1  # only the single-turn case produces a row


def test_errored_case_metric_produces_no_row(tmp_path):
    payload = _run_payload()
    payload['testCases'][0]['metricsData'].append(
        {
            'name': 'Answer Relevancy',
            'threshold': 0.5,
            'success': False,
            'score': None,
            'error': 'judge unavailable',
        }
    )
    rows = DeepEvalInstanceLevelAdapter().transform_samples(
        payload, 'eval-id', MODEL_ID, 'test_synthetic'
    )
    assert [row.evaluation_result_id for row in rows] == ['exact_match']


def test_case_token_counts_and_latency_reach_rows():
    payload = _run_payload()
    case = payload['testCases'][0]
    case['inputTokenCount'] = 11
    case['outputTokenCount'] = 7
    case['completionTime'] = 0.25
    rows = DeepEvalInstanceLevelAdapter().transform_samples(
        payload, 'eval-id', MODEL_ID, 'test_synthetic'
    )
    assert rows[0].token_usage.input_tokens == 11
    assert rows[0].token_usage.output_tokens == 7
    assert rows[0].token_usage.total_tokens == 18
    assert rows[0].performance.latency_ms == pytest.approx(250.0)


def test_directory_mode_reports_bad_files(tmp_path):
    (tmp_path / 'test_run_20261122_001137.json').write_text(
        RUN_FILE.read_text(encoding='utf-8'), encoding='utf-8'
    )
    (tmp_path / 'test_run_20260101_000000.json').write_text(
        '{"not": "a test run"}', encoding='utf-8'
    )
    adapter = DeepEvalAdapter()
    result = adapter.transform_from_directory_result(
        tmp_path, _make_metadata_args()
    )
    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert 'test_run_20260101_000000.json' in result.failures[0].source_ref
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()
