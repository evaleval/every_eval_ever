#!/usr/bin/env python3
"""Convert selected Vectara Hallucination Leaderboard rows to EEE JSON.

The source rows are public structured result files in ``vectara/results``. The
underlying evaluation dataset used by Vectara is private/proprietary, so the
generated EEE records keep the private evaluated dataset separate from the
public Hugging Face result source.

Usage:
    uv run python -m utils.vectara_hallucination_leaderboard.adapter \
        --output-dir /tmp/eee-vectara-hallucination
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import URLError


BENCHMARK = "vectara-hallucination-leaderboard"
SCHEMA_VERSION = "0.2.2"
SOURCE_REPO = "vectara/results"
SOURCE_COMMIT = "7c104699e98ade53dd719f79ae9f7eb281c8107d"
SOURCE_DATASET_URL = "https://huggingface.co/datasets/vectara/results"
LEADERBOARD_REPO_URL = "https://github.com/vectara/hallucination-leaderboard"
SOURCE_ORG_URL = "https://vectara.com"
DEFAULT_RETRIEVED_TIMESTAMP = "1779880742.736885"
PRIVATE_EVAL_DATASET_NAME = (
    "Vectara Hallucination Leaderboard private evaluation dataset"
)
PRIVATE_EVAL_DATASET_DESCRIPTION = (
    "Curated collection of 7700+ articles used for summarization-based "
    "hallucination evaluation."
)
PRIVATE_EVAL_DATASET_AVAILABILITY = (
    "Private/proprietary dataset; not publicly released to avoid overfitting."
)


SELECTED_MODELS = {
    "google/gemini-2.5-flash-lite": {
        "uuid": "65f449e7-595b-4031-8364-2b24d2d6ff95",
        "source_path": (
            "google/gemini-2.5-flash-lite/"
            "results_2025-12-10 14:57:20.585062.json"
        ),
        "inference_platform": "vertex_ai",
        "api_model_reference": "gemini-2.5-flash-lite",
    },
    "microsoft/Phi-4": {
        "uuid": "93cd2b22-2bd6-438a-95a6-7c32134638c5",
        "source_path": "microsoft/Phi-4/results_2025-12-10 14:57:16.944171.json",
        "inference_platform": "azure",
        "api_model_reference": "Phi-4",
    },
    "qwen/qwen3-8b": {
        "uuid": "644ea67b-89da-4310-b05c-bb4064abc2ad",
        "source_path": "qwen/qwen3-8b/results_2025-12-10 14:57:15.832674.json",
        "inference_platform": "dashscope",
        "api_model_reference": "qwen3-8b",
    },
}


OFFLINE_SOURCE_ROWS = {
    "google/gemini-2.5-flash-lite": {
        "config": {
            "model_dtype": "float16",
            "model_name": "google/gemini-2.5-flash-lite-",
            "model_sha": "main",
        },
        "results": {
            "hallucination_rate": {"hallucination_rate": 3.3},
            "factual_consistency_rate": {"factual_consistency_rate": 96.7},
            "answer_rate": {"answer_rate": 99.5},
            "average_summary_length": {"average_summary_length": 95.7},
        },
        "model_annotations": {"model_size": "large", "accessibility": "commercial"},
    },
    "microsoft/Phi-4": {
        "config": {
            "model_dtype": "float16",
            "model_name": "microsoft/Phi-4-",
            "model_sha": "main",
        },
        "results": {
            "hallucination_rate": {"hallucination_rate": 3.7},
            "factual_consistency_rate": {"factual_consistency_rate": 96.3},
            "answer_rate": {"answer_rate": 80.7},
            "average_summary_length": {"average_summary_length": 120.9},
        },
        "model_annotations": {"model_size": "small", "accessibility": "open"},
    },
    "qwen/qwen3-8b": {
        "config": {
            "model_dtype": "float16",
            "model_name": "qwen/qwen3-8b-",
            "model_sha": "main",
        },
        "results": {
            "hallucination_rate": {"hallucination_rate": 4.8},
            "factual_consistency_rate": {"factual_consistency_rate": 95.2},
            "answer_rate": {"answer_rate": 99.9},
            "average_summary_length": {"average_summary_length": 83.6},
        },
        "model_annotations": {"model_size": "small", "accessibility": "open"},
    },
}


METRICS = (
    {
        "source_key": "hallucination_rate",
        "value_key": "hallucination_rate",
        "result_id": "hallucination_rate",
        "name": "Hallucination Rate",
        "kind": "rate",
        "unit": "percent",
        "lower_is_better": True,
        "description": (
            "Percentage of generated summaries judged to contain factual "
            "inconsistencies or unsupported claims."
        ),
    },
    {
        "source_key": "factual_consistency_rate",
        "value_key": "factual_consistency_rate",
        "result_id": "factual_consistency_rate",
        "name": "Factual Consistency Rate",
        "kind": "rate",
        "unit": "percent",
        "lower_is_better": False,
        "description": "Percentage of generated summaries judged factually consistent.",
    },
    {
        "source_key": "answer_rate",
        "value_key": "answer_rate",
        "result_id": "answer_rate",
        "name": "Answer Rate",
        "kind": "rate",
        "unit": "percent",
        "lower_is_better": False,
        "description": "Percentage of prompts for which the model produced an answer.",
    },
    {
        "source_key": "average_summary_length",
        "value_key": "average_summary_length",
        "result_id": "average_summary_length",
        "name": "Average Summary Length",
        "kind": "length",
        "unit": "words",
        "lower_is_better": False,
        "description": (
            "Mean generated summary length in words; reported as a diagnostic metric."
        ),
        "diagnostic": True,
    },
)


def source_url(source_path: str) -> str:
    quoted_path = urllib.parse.quote(source_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/"
        f"{SOURCE_COMMIT}/{quoted_path}"
    )


def fetch_source_row(model_id: str, source_path: str, offline: bool) -> dict:
    if offline:
        return OFFLINE_SOURCE_ROWS[model_id]

    url = source_url(source_path)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError:
        completed = subprocess.run(
            ["curl", "-L", "-sS", "--fail", "--max-time", "30", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


def source_timestamp(source_path: str) -> str:
    stem = Path(source_path).stem
    return stem.removeprefix("results_")


def build_metric_result(
    model_id: str,
    source_path: str,
    row: dict,
    metric: dict,
    retrieved_timestamp: str,
) -> dict:
    score = row["results"][metric["source_key"]][metric["value_key"]]
    metric_id = f"{BENCHMARK}.{metric['result_id']}"
    additional_details = {
        "source_metric_key": metric["source_key"],
        "source_file": source_path,
        "source_commit": SOURCE_COMMIT,
        "source_resolve_url": source_url(source_path),
        "evaluation_timestamp_source": "Parsed from the source result filename.",
    }
    if metric.get("diagnostic"):
        additional_details["diagnostic_metric"] = "true"
        additional_details["lower_is_better_note"] = (
            "This is descriptive context rather than an optimization target."
        )

    return {
        "evaluation_result_id": (
            f"{BENCHMARK}/{model_id}/{retrieved_timestamp}#{metric['result_id']}"
        ),
        "evaluation_name": "Vectara Hallucination Leaderboard",
        "source_data": {
            "dataset_name": PRIVATE_EVAL_DATASET_NAME,
            "source_type": "other",
            "additional_details": {
                "availability": PRIVATE_EVAL_DATASET_AVAILABILITY,
                "dataset_description": PRIVATE_EVAL_DATASET_DESCRIPTION,
                "results_hf_repo": SOURCE_REPO,
                "results_dataset_url": SOURCE_DATASET_URL,
                "source_file": source_path,
                "source_commit": SOURCE_COMMIT,
                "source_resolve_url": source_url(source_path),
                "leaderboard_repository": LEADERBOARD_REPO_URL,
            },
        },
        "evaluation_timestamp": source_timestamp(source_path),
        "metric_config": {
            "evaluation_description": metric["description"],
            "metric_id": metric_id,
            "metric_name": metric["name"],
            "metric_kind": metric["kind"],
            "metric_unit": metric["unit"],
            "metric_parameters": {},
            "lower_is_better": metric["lower_is_better"],
            "score_type": "continuous",
            "min_score": 0.0,
            "max_score": 100.0 if metric["unit"] == "percent" else 1000.0,
            "additional_details": additional_details,
        },
        "score_details": {
            "score": float(score),
            "details": {"source_value_unit": metric["unit"]},
        },
    }


def build_record(
    model_id: str, spec: dict, row: dict, retrieved_timestamp: str
) -> dict:
    developer, model_name = model_id.split("/", 1)
    source_path = spec["source_path"]
    annotations = row.get("model_annotations", {})
    config = row.get("config", {})

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": f"{BENCHMARK}/{model_id}/{retrieved_timestamp}",
        "retrieved_timestamp": retrieved_timestamp,
        "evaluation_timestamp": source_timestamp(source_path),
        "source_metadata": {
            "source_name": "Vectara Hallucination Leaderboard",
            "source_type": "documentation",
            "source_organization_name": "Vectara",
            "source_organization_url": SOURCE_ORG_URL,
            "evaluator_relationship": "third_party",
            "additional_details": {
                "structured_results_dataset": SOURCE_DATASET_URL,
                "structured_results_hf_repo": SOURCE_REPO,
                "source_commit": SOURCE_COMMIT,
                "source_file": source_path,
                "source_resolve_url": source_url(source_path),
                "leaderboard_repository": LEADERBOARD_REPO_URL,
                "underlying_evaluation_dataset": PRIVATE_EVAL_DATASET_NAME,
                "underlying_evaluation_dataset_availability": (
                    PRIVATE_EVAL_DATASET_AVAILABILITY
                ),
                "scoring_model": "Vectara HHEM-2.3",
                "generation_temperature": (
                    "0 unless unavailable, per source documentation"
                ),
                "evaluation_timestamp_source": (
                    "Parsed from the source result filename."
                ),
            },
        },
        "model_info": {
            "name": model_name,
            "id": model_id,
            "developer": developer,
            "inference_platform": spec["inference_platform"],
            "additional_details": {
                "api_model_reference": spec["api_model_reference"],
                "source_model_name": str(config.get("model_name", "")),
                "model_dtype": str(config.get("model_dtype", "")),
                "model_sha": str(config.get("model_sha", "")),
                "model_size": str(annotations.get("model_size", "")),
                "accessibility": str(annotations.get("accessibility", "")),
            },
        },
        "eval_library": {
            "name": "unknown",
            "version": "unknown",
            "additional_details": {
                "leaderboard_repository": LEADERBOARD_REPO_URL,
                "structured_results_dataset": SOURCE_DATASET_URL,
                "scoring_model": "Vectara HHEM-2.3",
            },
        },
        "evaluation_results": [
            build_metric_result(model_id, source_path, row, metric, retrieved_timestamp)
            for metric in METRICS
        ],
    }


def make_records(retrieved_timestamp: str, offline: bool = False) -> list[tuple[str, str, dict]]:
    records = []
    for model_id, spec in SELECTED_MODELS.items():
        row = fetch_source_row(model_id, spec["source_path"], offline)
        records.append((model_id, spec["uuid"], build_record(model_id, spec, row, retrieved_timestamp)))
    return records


def write_record(output_dir: Path, model_id: str, uuid_value: str, record: dict) -> Path:
    developer, model_name = model_id.split("/", 1)
    model_dir = output_dir / developer / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{uuid_value}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return output_path


def export_records(output_dir: Path, retrieved_timestamp: str, offline: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_record(output_dir, model_id, uuid_value, record)
        for model_id, uuid_value, record in make_records(retrieved_timestamp, offline)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / BENCHMARK,
        help="Directory for generated benchmark records.",
    )
    parser.add_argument(
        "--retrieved-timestamp",
        default=DEFAULT_RETRIEVED_TIMESTAMP,
        help="Unix epoch timestamp string for generated EEE records.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use embedded selected source row snapshots instead of fetching from HF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_records(args.output_dir, args.retrieved_timestamp, args.offline)
    print(f"Generated {len(paths)} files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
