#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

SOURCE_URL = "https://arcprize.org/media/data/leaderboard/evaluations.json"


def make_source_data() -> dict:
    return {
        "source_type": "url",
        "dataset_name": "ARC Prize evaluations leaderboard JSON",
        "url": [SOURCE_URL],
    }


def load_rows(input_json: Path) -> list[dict]:
    return json.loads(input_json.read_text(encoding="utf-8"))


def infer_developer(raw_model_id: str) -> str:
    s = raw_model_id.strip().lower().replace("_", "-")

    if s.startswith(("openai-", "gpt-", "o1-", "o3-", "o4-", "codex")):
        return "openai"
    if s.startswith(("anthropic-", "claude")):
        return "anthropic"
    if s.startswith(("google-", "gemini")):
        return "google"
    if s.startswith(("xai-", "grok")):
        return "xai"
    if s.startswith(("qwen", "qwq")):
        return "qwen"
    if s.startswith("deepseek") or s == "r1":
        return "deepseek"
    if s.startswith("glm"):
        return "zhipu"
    if s.startswith("kimi"):
        return "moonshotai"
    if s.startswith(("mistral", "magistral")):
        return "mistralai"
    if s.startswith("llama"):
        return "meta"
    if s.startswith("olmo"):
        return "allenai"
    if s.startswith("minimax"):
        return "minimax"

    if raw_model_id in {"2025_human_panel"}:
        return "arcprize"

    return "community"


def slugify_model_name(raw_model_id: str, developer_name: str) -> str:
    s = raw_model_id.strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9.\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")

    prefix = developer_name + "-"
    if s.startswith(prefix):
        s = s[len(prefix):]

    return s


def normalize_model(raw_model_id: str) -> tuple[str, str]:
    developer_name = infer_developer(raw_model_id)
    model_name = slugify_model_name(raw_model_id, developer_name)
    return developer_name, model_name


def stringify_details(row: dict, exclude_keys: set[str]) -> dict[str, str]:
    details = {}
    for k, v in row.items():
        if k in exclude_keys:
            continue
        details[k] = str(v)
    return details


def choose_primary_raw_model_id(rows_for_canonical: list[dict], developer_name: str) -> str:
    aliases = sorted({row["modelId"] for row in rows_for_canonical})
    prefix = developer_name + "-"
    aliases.sort(
        key=lambda raw: (
            raw.lower().replace("_", "-").startswith(prefix),
            len(raw),
            raw.lower(),
        )
    )
    return aliases[0]


def choose_best_row(rows: list[dict], developer_name: str) -> dict:
    prefix = developer_name + "-"
    return sorted(
        rows,
        key=lambda row: (
            row["modelId"].lower().replace("_", "-").startswith(prefix),
            len(row["modelId"]),
            row["modelId"].lower(),
        ),
    )[0]


def make_results(rows_for_canonical: list[dict], developer_name: str) -> list[dict]:
    results = []
    by_dataset = defaultdict(list)

    for row in rows_for_canonical:
        by_dataset[row["datasetId"]].append(row)

    for dataset_id in sorted(by_dataset):
        row = choose_best_row(by_dataset[dataset_id], developer_name)
        aliases_for_dataset = sorted({r["modelId"] for r in by_dataset[dataset_id]})

        results.append(
            {
                "evaluation_result_id": f"{dataset_id}::score",
                "evaluation_name": dataset_id,
                "source_data": make_source_data(),
                "metric_config": {
                    "metric_id": "score",
                    "metric_name": "ARC score",
                    "lower_is_better": False,
                    "additional_details": {
                        "raw_metric_field": "score",
                    },
                },
                "score_details": {
                    "score": float(row["score"]),
                    "details": {
                        **stringify_details(
                            row,
                            exclude_keys={"score", "modelId"},
                        ),
                        "raw_model_id": row["modelId"],
                        "raw_model_aliases_json": json.dumps(aliases_for_dataset),
                    },
                },
            }
        )

        if "costPerTask" in row and row["costPerTask"] is not None:
            results.append(
                {
                    "evaluation_result_id": f"{dataset_id}::cost_per_task",
                    "evaluation_name": dataset_id,
                    "source_data": make_source_data(),
                    "metric_config": {
                        "metric_id": "cost_per_task",
                        "metric_name": "Cost per task",
                        "lower_is_better": True,
                        "additional_details": {
                            "raw_metric_field": "costPerTask",
                        },
                    },
                    "score_details": {
                        "score": float(row["costPerTask"]),
                        "details": {
                            **stringify_details(
                                row,
                                exclude_keys={"costPerTask", "modelId"},
                            ),
                            "raw_model_id": row["modelId"],
                            "raw_model_aliases_json": json.dumps(aliases_for_dataset),
                        },
                    },
                }
            )
        elif "cost" in row and row["cost"] is not None:
            results.append(
                {
                    "evaluation_result_id": f"{dataset_id}::cost",
                    "evaluation_name": dataset_id,
                    "source_data": make_source_data(),
                    "metric_config": {
                        "metric_id": "cost",
                        "metric_name": "Cost",
                        "lower_is_better": True,
                        "additional_details": {
                            "raw_metric_field": "cost",
                        },
                    },
                    "score_details": {
                        "score": float(row["cost"]),
                        "details": {
                            **stringify_details(
                                row,
                                exclude_keys={"cost", "modelId"},
                            ),
                            "raw_model_id": row["modelId"],
                            "raw_model_aliases_json": json.dumps(aliases_for_dataset),
                        },
                    },
                }
            )

    return results


def make_log(rows_for_canonical: list[dict], developer_name: str, model_name: str) -> tuple[dict, str, str]:
    primary_raw_model_id = choose_primary_raw_model_id(rows_for_canonical, developer_name)
    all_aliases = sorted({row["modelId"] for row in rows_for_canonical})
    ts = str(time.time())

    log = {
        "schema_version": "0.2.2",
        "evaluation_id": f"arc-agi/{developer_name}/{model_name}/{ts}",
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": "ARC Prize leaderboard JSON",
            "source_type": "documentation",
            "source_organization_name": "ARC Prize",
            "source_organization_url": "https://arcprize.org/leaderboard",
            "evaluator_relationship": "third_party",
            "additional_details": {
                "api_endpoint": SOURCE_URL,
                "filtered_to_display_true": "True",
            },
        },
        "eval_library": {
            "name": "ARC Prize leaderboard",
            "version": "unknown",
        },
        "model_info": {
            "name": primary_raw_model_id,
            "id": f"{developer_name}/{model_name}",
            "developer": developer_name,
            "additional_details": {
                "raw_model_id": primary_raw_model_id,
                "raw_model_aliases_json": json.dumps(all_aliases),
            },
        },
        "evaluation_results": make_results(rows_for_canonical, developer_name),
    }

    return log, developer_name, model_name


def write_log(log: dict, out_root: Path, developer: str, model: str) -> Path:
    out_dir = out_root / "arc-agi" / developer / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4()}.json"
    out_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input_json)
    rows = [r for r in rows if r.get("display") is True]

    by_canonical = defaultdict(list)
    for row in rows:
        developer_name, model_name = normalize_model(row["modelId"])
        by_canonical[(developer_name, model_name)].append(row)

    exported = 0
    for (developer_name, model_name), rows_for_canonical in sorted(by_canonical.items()):
        log, developer, model = make_log(rows_for_canonical, developer_name, model_name)
        out_path = write_log(log, args.output_dir, developer, model)
        print(out_path)
        exported += 1

    print(f"Exported {exported} model(s).")


if __name__ == "__main__":
    main()
