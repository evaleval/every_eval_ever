from importlib import import_module

convert_rows = import_module(
    'every_eval_ever.adapters.global_mmlu_lite.adapter'
).convert_rows


def _task(name, value):
    return {
        "benchmarkTaskName": name,
        "result": {
            "hasNumericResult": True,
            "numericResult": {"value": value},
        },
    }


def test_partial_task_failure_keeps_valid_model_output(tmp_path):
    row = {
        "modelVersionSlug": "openai/gpt-test",
        "modelVersionName": "GPT Test",
        "taskResults": [
            _task("valid", 0.5),
            _task("missing", None),
        ],
    }

    result = convert_rows(
        [row],
        retrieved_timestamp="1234",
        output_dir=str(tmp_path / "data" / "global-mmlu-lite"),
    )

    assert len(result.records) == 1
    assert [
        metric.evaluation_name
        for metric in result.records[0].eval_log.evaluation_results
    ] == ["valid"]
    assert len(result.failures) == 1
    assert result.failures[0].source_record == row["taskResults"][1]
