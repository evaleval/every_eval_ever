from __future__ import annotations

from pathlib import Path

from every_eval_ever.adapters.aixamine import adapter as aix
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import EvaluationLogOutput, save_evaluation_logs
from every_eval_ever.validate import validate_file

CATALOG = [
    {
        "value": "hallucination",
        "name": "Hallucination",
        "tests": [
            {"value": "halueval", "name": "HaluEval", "description": "Hallucination eval."},
            {"value": "simpleqa", "name": "SimpleQA", "description": "Short-form factuality."},
            {"value": "factqa", "name": "FactQA", "description": "Dynamic factual QA.", "dynamic": True},
        ],
    },
    {
        "value": "fairness-bias",
        "name": "Fairness & Bias",
        "tests": [{"value": "bbq", "name": "BBQ", "description": "Bias benchmark."}],
    },
]

REPORT = {
    "_id": "r1",
    "model": "m1",
    "services": {
        "hallucination": {
            "tests": {
                "halueval": {"score": 55.3, "categories": {"QA": {"score": 44.2, "subcategories": []}}},
                "simpleqa": {"score": 30.0, "categories": {}},
            }
        },
        "fairness-bias": {"tests": {"bbq": {"score": 71.4, "categories": {}}}},
    },
    "dynamic": {
        "services": {
            "hallucination": {
                "tests": {
                    "factqa": {
                        "versions": [
                            {"score": 83.5, "testVersion": "v1",
                             "generatedAt": "2026-06-29T08:22:39.894Z",
                             "categories": {"LA": 65, "ST": 91.6}}
                        ]
                    }
                }
            }
        }
    },
}

MODEL_HF = {"name": "meta-llama/Llama-3.1-8B-Instruct", "developer": "meta-llama", "accessType": "huggingface"}
MODEL_API = {"name": "gpt-5", "developer": "OpenAI", "accessType": "openai"}


def _save_and_validate(logs, tmp_path) -> list[Path]:
    outputs = [
        EvaluationLogOutput(
            eval_log=EvaluationLog.model_validate(log.model_dump()),
            base_dir=tmp_path / "data" / collection,
            developer=dev,
            model_name=model_name,
        )
        for collection, dev, model_name, log in logs
    ]
    paths = save_evaluation_logs(outputs)
    assert paths
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors
    return paths


def test_one_log_per_service_and_validates(tmp_path):
    logs = aix.build_service_logs(REPORT, MODEL_HF, CATALOG, "123")
    collections = {c for c, _, _, _ in logs}
    assert collections == {"aixamine_hallucination", "aixamine_fairness_bias"}
    _save_and_validate(logs, tmp_path)


def test_canonical_mapping_bare_names_and_categories(tmp_path):
    logs = aix.build_service_logs(REPORT, MODEL_HF, CATALOG, "123")
    names = {r.evaluation_name for _, _, _, log in logs for r in log.evaluation_results}
    assert "bbq" in names            # canonical id for a confident match
    assert "halueval" in names       # bare aiXamine name otherwise
    assert "halueval.QA" in names    # static category sub-result
    assert "factqa" in names         # dynamic test (latest version)
    assert "factqa.LA" in names      # dynamic category sub-result


def test_api_model_marked_closed_weights(tmp_path):
    logs = aix.build_service_logs(REPORT, MODEL_API, CATALOG, "123")
    _, _, _, log = logs[0]
    assert log.model_info.id == "OpenAI/gpt-5"
    assert log.model_info.additional_details["model_availability"] == "closed_weights"
    assert log.model_info.additional_details["deployment_type"] == "externally_managed"
    _save_and_validate(logs, tmp_path)


def test_source_metadata_is_first_party_documentation(tmp_path):
    logs = aix.build_service_logs(REPORT, MODEL_HF, CATALOG, "123")
    _, _, _, log = logs[0]
    assert log.eval_library.name == "aixamine"
    assert log.source_metadata.source_type.value == "documentation"
    assert log.source_metadata.evaluator_relationship.value == "first_party"
