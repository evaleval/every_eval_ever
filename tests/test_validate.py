"""Tests for validate.py — Pydantic-based EEE schema validation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from every_eval_ever.validate import (
    ValidationReport,
    expand_paths,
    validate_aggregate,
    validate_file,
    validate_instance_file,
    render_report_json,
    render_report_github,
)

# ---------------------------------------------------------------------------
# Helpers — minimal valid data fixtures
# ---------------------------------------------------------------------------

VALID_AGGREGATE: dict = {
    "schema_version": "0.2.2",
    "evaluation_id": "test/model/123",
    "retrieved_timestamp": "1234567890",
    "source_metadata": {
        "source_type": "evaluation_run",
        "source_organization_name": "TestOrg",
        "evaluator_relationship": "first_party",
    },
    "eval_library": {"name": "inspect_ai", "version": "0.3.0"},
    "model_info": {"name": "test-model", "id": "org/test-model"},
    "evaluation_results": [
        {
            "evaluation_name": "test_eval",
            "source_data": {
                "dataset_name": "test_ds",
                "source_type": "hf_dataset",
                "hf_repo": "org/test-ds",
            },
            "metric_config": {
                "lower_is_better": False,
                "score_type": "binary",
            },
            "score_details": {"score": 0.95},
        }
    ],
}

VALID_SINGLE_TURN: dict = {
    "schema_version": "instance_level_eval_0.2.2",
    "evaluation_id": "test/model/123",
    "model_id": "org/test-model",
    "evaluation_name": "test_eval",
    "sample_id": "sample_001",
    "interaction_type": "single_turn",
    "input": {"raw": "What is 2+2?", "reference": ["4"]},
    "output": {"raw": ["4"]},
    "answer_attribution": [
        {
            "turn_idx": 0,
            "source": "output.raw",
            "extracted_value": "4",
            "extraction_method": "exact_match",
            "is_terminal": True,
        }
    ],
    "evaluation": {"score": 1.0, "is_correct": True},
}

VALID_MULTI_TURN: dict = {
    "schema_version": "instance_level_eval_0.2.2",
    "evaluation_id": "test/model/123",
    "model_id": "org/test-model",
    "evaluation_name": "test_eval",
    "sample_id": "sample_002",
    "interaction_type": "multi_turn",
    "input": {"raw": "Solve this problem", "reference": ["42"]},
    "messages": [
        {"turn_idx": 0, "role": "user", "content": "Solve this problem"},
        {"turn_idx": 1, "role": "assistant", "content": "The answer is 42"},
    ],
    "answer_attribution": [
        {
            "turn_idx": 1,
            "source": "messages[1].content",
            "extracted_value": "42",
            "extraction_method": "regex",
            "is_terminal": True,
        }
    ],
    "evaluation": {"score": 1.0, "is_correct": True},
}


def _write_json(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_jsonl(tmp_path: Path, name: str, lines: list[dict | str]) -> Path:
    p = tmp_path / name
    text_lines = []
    for item in lines:
        if isinstance(item, str):
            text_lines.append(item)
        else:
            text_lines.append(json.dumps(item))
    p.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return p


# ===================================================================
# Aggregate validation tests
# ===================================================================


class TestAggregateValidation:
    def test_valid_json_passes(self, tmp_path: Path):
        fp = _write_json(tmp_path, "valid.json", VALID_AGGREGATE)
        report = validate_aggregate(fp)
        assert report.valid is True
        assert report.errors == []
        assert report.file_type == "aggregate"

    def test_missing_required_field(self, tmp_path: Path):
        data = {**VALID_AGGREGATE}
        del data["evaluation_id"]
        fp = _write_json(tmp_path, "missing.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("evaluation_id" in e["loc"] for e in report.errors)

    def test_extra_field_on_evaluation_log_fails(self, tmp_path: Path):
        data = {**VALID_AGGREGATE, "unexpected_field": "oops"}
        fp = _write_json(tmp_path, "extra.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("unexpected_field" in e["loc"] for e in report.errors)

    def test_extra_field_on_generation_args_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["evaluation_results"][0]["generation_config"] = {
            "generation_args": {"temperature": 0.7, "unknown_param": "bad"}
        }
        fp = _write_json(tmp_path, "extra_gen.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("unknown_param" in e["loc"] for e in report.errors)

    def test_score_type_levels_without_level_names_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["evaluation_results"][0]["metric_config"] = {
            "lower_is_better": False,
            "score_type": "levels",
            # missing level_names and has_unknown_level
        }
        fp = _write_json(tmp_path, "levels.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("level_names" in e["msg"] for e in report.errors)

    def test_score_type_continuous_without_min_score_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["evaluation_results"][0]["metric_config"] = {
            "lower_is_better": False,
            "score_type": "continuous",
            # missing min_score and max_score
        }
        fp = _write_json(tmp_path, "continuous.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("min_score" in e["msg"] for e in report.errors)

    def test_source_data_discriminated_error(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["evaluation_results"][0]["source_data"] = {
            "dataset_name": "test",
            "source_type": "hf_dataset",
            # valid hf_dataset, should pass
        }
        fp = _write_json(tmp_path, "disc.json", data)
        report = validate_aggregate(fp)
        assert report.valid is True

    def test_source_data_wrong_source_type_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["evaluation_results"][0]["source_data"] = {
            "dataset_name": "test",
            "source_type": "invalid_type",
        }
        fp = _write_json(tmp_path, "bad_source.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False

    def test_additional_details_non_string_values_fail(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_AGGREGATE))
        data["model_info"]["additional_details"] = {"params_billions": 8.357}
        fp = _write_json(tmp_path, "nonstr.json", data)
        report = validate_aggregate(fp)
        assert report.valid is False
        assert any("string" in e["msg"] for e in report.errors)

    def test_json_parse_error(self, tmp_path: Path):
        fp = tmp_path / "bad.json"
        fp.write_text("{invalid json}", encoding="utf-8")
        report = validate_aggregate(fp)
        assert report.valid is False
        assert report.errors[0]["type"] == "json_parse_error"


# ===================================================================
# Instance-level validation tests
# ===================================================================


class TestInstanceLevelValidation:
    def test_valid_single_turn_passes(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, "valid.jsonl", [VALID_SINGLE_TURN])
        report = validate_instance_file(fp)
        assert report.valid is True
        assert report.line_count == 1

    def test_valid_multi_turn_passes(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, "multi.jsonl", [VALID_MULTI_TURN])
        report = validate_instance_file(fp)
        assert report.valid is True

    def test_single_turn_with_messages_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_SINGLE_TURN))
        data["messages"] = [
            {"turn_idx": 0, "role": "user", "content": "hi"},
        ]
        fp = _write_jsonl(tmp_path, "bad_st.jsonl", [data])
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any("must not have messages" in e["msg"] for e in report.errors)

    def test_multi_turn_without_messages_fails(self, tmp_path: Path):
        data = json.loads(json.dumps(VALID_MULTI_TURN))
        del data["messages"]
        fp = _write_jsonl(tmp_path, "no_msgs.jsonl", [data])
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any("requires messages" in e["msg"] for e in report.errors)

    def test_invalid_line_in_middle_reports_correct_line_number(self, tmp_path: Path):
        bad_line = {**VALID_SINGLE_TURN}
        del bad_line["evaluation_id"]
        fp = _write_jsonl(
            tmp_path, "mid.jsonl", [VALID_SINGLE_TURN, bad_line, VALID_SINGLE_TURN]
        )
        report = validate_instance_file(fp)
        assert report.valid is False
        assert any("line 2" in e["loc"] for e in report.errors)

    def test_json_parse_error_reports_line_number(self, tmp_path: Path):
        fp = _write_jsonl(
            tmp_path, "parse.jsonl", [VALID_SINGLE_TURN, "{bad json}"]
        )
        report = validate_instance_file(fp)
        assert report.valid is False
        assert report.errors[0]["type"] == "json_parse_error"
        assert "line 2" in report.errors[0]["loc"]

    def test_empty_jsonl_passes(self, tmp_path: Path):
        fp = tmp_path / "empty.jsonl"
        fp.write_text("", encoding="utf-8")
        report = validate_instance_file(fp)
        assert report.valid is True
        assert report.line_count == 0

    def test_blank_lines_skipped(self, tmp_path: Path):
        lines = [json.dumps(VALID_SINGLE_TURN), "", "  ", json.dumps(VALID_SINGLE_TURN)]
        fp = tmp_path / "blanks.jsonl"
        fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = validate_instance_file(fp)
        assert report.valid is True
        assert report.line_count == 2


# ===================================================================
# File dispatch and CLI tests
# ===================================================================


class TestFileDispatch:
    def test_json_dispatches_to_aggregate(self, tmp_path: Path):
        fp = _write_json(tmp_path, "test.json", VALID_AGGREGATE)
        report = validate_file(fp)
        assert report.file_type == "aggregate"

    def test_jsonl_dispatches_to_instance(self, tmp_path: Path):
        fp = _write_jsonl(tmp_path, "test.jsonl", [VALID_SINGLE_TURN])
        report = validate_file(fp)
        assert report.file_type == "instance"

    def test_unsupported_extension(self, tmp_path: Path):
        fp = tmp_path / "test.csv"
        fp.write_text("a,b,c", encoding="utf-8")
        report = validate_file(fp)
        assert report.valid is False
        assert report.errors[0]["type"] == "unsupported_extension"

    def test_directory_expansion(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_json(sub, "a.json", VALID_AGGREGATE)
        _write_jsonl(sub, "b.jsonl", [VALID_SINGLE_TURN])
        (sub / "c.txt").write_text("ignored")
        paths = expand_paths([str(sub)])
        extensions = {p.suffix for p in paths}
        assert ".json" in extensions
        assert ".jsonl" in extensions
        assert ".txt" not in extensions


class TestMaxErrors:
    def test_max_errors_caps_output(self, tmp_path: Path):
        bad_line = {**VALID_SINGLE_TURN}
        del bad_line["evaluation_id"]
        lines = [bad_line] * 100
        fp = _write_jsonl(tmp_path, "many.jsonl", lines)
        report = validate_instance_file(fp, max_errors=5)
        assert report.valid is False
        # Should have at most 5 real errors + 1 truncation message
        assert len(report.errors) <= 6
        assert any(e["type"] == "truncated" for e in report.errors)


class TestOutputFormats:
    def test_json_output_is_valid_json(self, tmp_path: Path):
        fp = _write_json(tmp_path, "test.json", VALID_AGGREGATE)
        report = validate_file(fp)
        output = render_report_json([report])
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["valid"] is True

    def test_github_output_format(self, tmp_path: Path):
        data = {**VALID_AGGREGATE}
        del data["evaluation_id"]
        fp = _write_json(tmp_path, "fail.json", data)
        report = validate_file(fp)
        output = render_report_github([report])
        assert output.startswith("::error file=")

    def test_github_output_empty_on_pass(self, tmp_path: Path):
        fp = _write_json(tmp_path, "pass.json", VALID_AGGREGATE)
        report = validate_file(fp)
        output = render_report_github([report])
        assert output == ""


class TestExitCode:
    def test_exit_code_0_on_pass(self, tmp_path: Path):
        fp = _write_json(tmp_path, "pass.json", VALID_AGGREGATE)
        report = validate_file(fp)
        assert report.valid is True

    def test_exit_code_1_on_failure(self, tmp_path: Path):
        data = {**VALID_AGGREGATE}
        del data["evaluation_id"]
        fp = _write_json(tmp_path, "fail.json", data)
        report = validate_file(fp)
        assert report.valid is False
