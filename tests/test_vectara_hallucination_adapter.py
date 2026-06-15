from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.validate import validate_file
from utils.vectara_hallucination_leaderboard import adapter


def records():
    return adapter.make_records("1779880742.736885", offline=True)


def test_offline_records_are_schema_valid():
    generated = records()

    assert len(generated) == 3
    for model_id, _uuid_value, record in generated:
        assert record["schema_version"] == "0.2.2"
        assert record["evaluation_id"].startswith(
            f"vectara-hallucination-leaderboard/{model_id}/"
        )
        assert record["source_metadata"]["source_type"] == "documentation"
        assert record["eval_library"]["name"] == "unknown"
        assert (
            record["eval_library"]["additional_details"]["scoring_model"]
            == "Vectara HHEM-2.3"
        )


def test_source_data_distinguishes_private_eval_dataset_from_public_results():
    _model_id, _uuid_value, record = records()[0]
    result = record["evaluation_results"][0]

    source_data = result["source_data"]
    assert source_data["source_type"] == "other"
    assert "private evaluation dataset" in source_data["dataset_name"]
    assert source_data["additional_details"]["results_hf_repo"] == "vectara/results"
    assert (
        source_data["additional_details"]["availability"]
        == "Private/proprietary dataset; not publicly released to avoid overfitting."
    )


def test_scores_and_platforms_match_selected_source_rows():
    by_model = {model_id: record for model_id, _uuid_value, record in records()}

    gemini = by_model["google/gemini-2.5-flash-lite"]
    assert gemini["model_info"]["inference_platform"] == "vertex_ai"
    assert gemini["model_info"]["additional_details"]["api_model_reference"] == (
        "gemini-2.5-flash-lite"
    )

    phi = by_model["microsoft/Phi-4"]
    assert phi["model_info"]["inference_platform"] == "azure"

    qwen = by_model["qwen/qwen3-8b"]
    assert qwen["model_info"]["inference_platform"] == "dashscope"

    scores = {
        result["metric_config"]["metric_id"]: result["score_details"]["score"]
        for result in gemini["evaluation_results"]
    }
    assert scores["vectara-hallucination-leaderboard.hallucination_rate"] == 3.3
    assert scores["vectara-hallucination-leaderboard.factual_consistency_rate"] == 96.7
    assert scores["vectara-hallucination-leaderboard.answer_rate"] == 99.5
    assert scores["vectara-hallucination-leaderboard.average_summary_length"] == 95.7


def test_export_paths_follow_datastore_layout_and_validate(tmp_path: Path):
    output_dir = tmp_path / "data" / adapter.BENCHMARK
    paths = adapter.export_records(
        output_dir, retrieved_timestamp="1779880742.736885", offline=True
    )

    assert len(paths) == 3
    assert (
        output_dir
        / "google"
        / "gemini-2.5-flash-lite"
        / "65f449e7-595b-4031-8364-2b24d2d6ff95.json"
    ) in paths

    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors
        assert path.parent.parent.parent == output_dir
        assert json.loads(path.read_text())["schema_version"] == "0.2.2"
