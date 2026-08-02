import json

from every_eval_ever.adapters.exgentic.adapter import (
    collect_results_from_dir,
    convert_results,
    load_results_from_dir,
)
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _result(**overrides):
    result = {
        "model_name": "openai/Azure/gpt-5.2",
        "benchmark_name": "SWE-bench",
        "agent_name": "Claude Code",
        "agent": "claude-code",
        "benchmark_score": 0.75,
        "total_sessions": 4,
    }
    result.update(overrides)
    return result


def test_convert_results_retains_valid_rows_and_raw_failure(tmp_path):
    good = _result()
    bad = _result(model_name="unmapped/model")

    result = convert_results(
        [good, bad],
        retrieved_timestamp="1234",
        output_dir=str(tmp_path / "data" / "exgentic"),
    )

    assert len(result.records) == 1
    assert result.failures[0].source_record == bad
    output = result.records[0]
    assert output.developer == "openai"
    assert output.model_name == "gpt-5.2"
    assert validate_file(
        _write_for_validation(tmp_path, output.eval_log)
    ).valid


def _write_for_validation(tmp_path, eval_log):
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps(
            eval_log.model_dump(mode="json", exclude_none=True),
            allow_nan=False,
        )
    )
    return path


def test_local_loader_preserves_good_results_when_one_config_is_bad(tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "config.json").write_text(json.dumps({"run_id": "run-1"}))
    run_dir = good_dir / "run-1"
    run_dir.mkdir()
    good = _result()
    (run_dir / "results.json").write_text(json.dumps(good))

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text(json.dumps({}))

    result = collect_results_from_dir(str(tmp_path))

    assert result.records == [good]
    assert len(result.failures) == 1
    assert result.failures[0].source_record == {}

    try:
        load_results_from_dir(str(tmp_path))
    except SourceRecordsError:
        pass
    else:
        raise AssertionError("strict loader should signal the incomplete load")
