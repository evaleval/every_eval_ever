import json

from every_eval_ever.adapters.hfopenllm_v2.adapter import (
    convert_model,
    convert_models,
    process_models,
)
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _model(evaluations):
    return {
        'model': {
            'name': 'example/model',
            'precision': 'bfloat16',
        },
        'metadata': {},
        'evaluations': evaluations,
    }


def test_single_model_conversion_remains_strict():
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )

    try:
        convert_model(source, '1234')
    except ValueError as exc:
        assert "Evaluation 'bbh' could not be converted" in str(exc)
    else:
        raise AssertionError('expected strict conversion to reject missing score')


def test_batch_keeps_valid_metrics_and_records_missing_metric():
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )

    result = convert_models([source], retrieved_timestamp='1234')

    assert len(result.records) == 1
    assert [
        metric.evaluation_name
        for metric in result.records[0].eval_log.evaluation_results
    ] == ['IFEval']
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == "model row 0 evaluation 'bbh'"
    assert result.failures[0].source_record == {
        'model': source['model'],
        'evaluation_key': 'bbh',
        'evaluation': source['evaluations']['bbh'],
    }


def test_process_models_writes_valid_output_and_external_failure_report(
    tmp_path,
):
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )
    output_dir = tmp_path / 'data' / 'hfopenllm_v2'

    try:
        process_models([source], str(output_dir))
    except SourceRecordsError:
        pass
    else:
        raise AssertionError('expected partial conversion to be signalled')

    outputs = list(output_dir.glob('*/*/*.json'))
    assert len(outputs) == 1
    assert validate_file(outputs[0]).valid

    report_path = (
        tmp_path / 'adapter_reports' / 'hfopenllm_v2_failures.json'
    )
    report = json.loads(report_path.read_text())
    assert report['converted_records'] == 1
    assert len(report['failed_records']) == 1
    assert not report_path.is_relative_to(output_dir)
