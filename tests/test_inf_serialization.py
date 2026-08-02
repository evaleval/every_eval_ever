"""Unbounded metric bounds: serialized as the JSON string "Infinity", scoped to
MetricConfig, with null reserved for "not provided" and NaN rejected.

Policy:
- unbounded bound = float('inf')/-inf, written as the JSON *string* "Infinity"/
  "-Infinity" (valid RFC-8259 JSON; pydantic reads it back to a float).
- the field stays typed `float | None`, so "convertible to a number" is enforced
  by the type itself; the schema pins the string form to exactly those two tokens.
- null stays INVALID for a continuous metric (missing != unbounded).
- NaN bounds are rejected.
- scoped to MetricConfig: score/latency/etc. floats keep default serialization.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import every_eval_ever.eval_types as ET
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_log
from every_eval_ever.validate import validate_file


def _mc(min_score, max_score):
    return ET.MetricConfig(
        metric_name='PSNR',
        lower_is_better=False,
        score_type=ET.ScoreType.continuous,
        min_score=min_score,
        max_score=max_score,
    )


def _log(min_score, max_score):
    return ET.EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id='src/dev_model/2026',
        retrieved_timestamp='1700000000.0',
        source_metadata=ET.SourceMetadata(
            source_name='x',
            source_type='documentation',
            source_organization_name='x',
            evaluator_relationship='third_party',
        ),
        eval_library=ET.EvalLibrary(name='unknown', version='unknown'),
        model_info=ET.ModelInfo(
            name='m',
            id='dev/model',
            developer='unknown',
            inference_platform='unknown',
            inference_engine=ET.InferenceEngine(
                name='unknown',
                version='unknown',
            ),
            additional_details={
                'deployment_type': 'unknown',
                'model_availability': 'unknown',
            },
        ),
        evaluation_results=[
            ET.EvaluationResult(
                evaluation_name='e',
                source_data=ET.SourceDataUrl(
                    dataset_name='d', source_type='url', url=['http://x']
                ),
                metric_config=_mc(min_score, max_score),
                score_details=ET.ScoreDetails(score=30.0),
            )
        ],
    )


def test_unbounded_serializes_as_infinity_string():
    dumped = _mc(float('-inf'), float('inf')).model_dump_json()
    assert '"max_score":"Infinity"' in dumped
    assert '"min_score":"-Infinity"' in dumped
    # ...and that output is STRICT-valid JSON (parses in JS/Go/jq/orjson)
    json.loads(dumped, parse_constant=_reject_constant)


def _reject_constant(c):  # would fire only on a bare Infinity/NaN token
    raise AssertionError(f'non-standard JSON token emitted: {c}')


def test_field_stays_float_and_reads_back():
    # the wire form is a string, but the Python value is a real float
    mc = _mc(0.0, float('inf'))
    assert mc.max_score == float('inf')
    reparsed = ET.MetricConfig.model_validate(json.loads(mc.model_dump_json()))
    assert reparsed.max_score == float('inf')
    # convertibility is enforced by the float type: a non-numeric string is rejected
    with pytest.raises(ValidationError):
        ET.MetricConfig.model_validate(
            {
                'metric_name': 'x',
                'lower_is_better': False,
                'score_type': 'continuous',
                'min_score': 0.0,
                'max_score': 'banana',
            }
        )
    with pytest.raises(ValidationError):
        ET.MetricConfig.model_validate(
            {
                'metric_name': 'x',
                'lower_is_better': False,
                'score_type': 'continuous',
                'min_score': '0.5',
                'max_score': 1.0,
            }
        )


def test_finite_bounds_stay_numbers():
    dumped = _mc(0.0, 100.0).model_dump_json()
    assert '"max_score":100.0' in dumped and 'Infinity' not in dumped


def test_null_bound_is_still_invalid_for_continuous():
    # missing != unbounded: null must stay rejected so it can't masquerade as inf
    with pytest.raises(ValidationError):
        _mc(0.0, None)


def test_nan_bound_is_rejected():
    with pytest.raises(ValidationError):
        _mc(0.0, float('nan'))


def test_mode_json_path_also_emits_string():
    # Regression: cli.py / instance writers serialize via
    # json.dump(model_dump(mode='json')) -- NOT model_dump_json(). The field
    # serializer must fire there too, or an inf bound leaks the bare (invalid)
    # Infinity token. Assert it's the STRING and the result is strict-valid JSON.
    d = _mc(0.0, float('inf')).model_dump(mode='json')
    assert d['max_score'] == 'Infinity'
    dumped = json.dumps(d)
    json.loads(dumped, parse_constant=_reject_constant)


def test_mode_python_keeps_native_float():
    # Python consumers get a real float, not the wire string.
    d = _mc(0.0, float('inf')).model_dump(mode='python')
    assert d['max_score'] == float('inf')


def test_no_model_wide_inf_config():
    # The fix is a field serializer on min/max_score, NOT a model-wide
    # ser_json_inf_nan -- so other MetricConfig floats / other models aren't
    # swept up (no inf/NaN-everywhere blast radius).
    assert ET.MetricConfig.model_config.get('ser_json_inf_nan') is None
    assert (
        _mc(0.0, 1.0).model_dump(mode='json')['max_score'] == 1.0
    )  # finite stays a number


def test_aggregate_round_trip_validates(tmp_path):
    path = save_evaluation_log(
        _log(0.0, float('inf')),
        tmp_path / 'data' / 'test-benchmark',
        'dev',
        'model',
    )
    report = validate_file(path)
    assert report.valid, report.errors
    mc = json.loads(path.read_text())['evaluation_results'][0]['metric_config']
    assert mc['max_score'] == 'Infinity'  # string on the wire
    # and the whole file is strict-valid JSON
    json.loads(path.read_text(), parse_constant=_reject_constant)
