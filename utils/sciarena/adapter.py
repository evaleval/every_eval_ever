#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

# Conservative provider mapping.
# Keep the source alias in raw_model_id and derive a simple lowercase model slug.
PROVIDER_MAP = {
    "o3": "openai",
    "Claude-4.1-Opus": "anthropic",
    "GPT-5": "openai",
    "Gemini-3-Pro-Preview": "google",
    "GPT-5.1": "openai",
    "Claude-4-Opus": "anthropic",
    "GPT-5-mini": "openai",
    "Gemini-2.5-Pro": "google",
    "Grok-4": "xai",
    "Deepseek-R1-0528": "deepseek",
    "GPT-OSS-120B": "openai",
    "Qwen3-235B-A22B-Thinking-2507": "qwen",
    "o4-mini": "openai",
    "Claude-4-Sonnet": "anthropic",
    "Qwen3-235B-A22B-2507": "qwen",
    "GPT-4.1": "openai",
    "GPT-4.1-mini": "openai",
    "Qwen3-30B-A3B-Instruct-2507": "qwen",
    "Gemini-2.5-Pro-Preview": "google",
    "GLM-4.5": "zhipu",
    "Deepseek-R1": "deepseek",
    "Deepseek-V3": "deepseek",
    "Qwen3-235B-A22B": "qwen",
    "Kimi-K2": "moonshotai",
    "Grok-3": "xai",
    "QwQ-32B": "qwen",
    "Claude-3-7-Sonnet": "anthropic",
    "Gemini-2.5-Flash": "google",
    "Olmo-3.1-32B-Instruct": "allenai",
    "Qwen3-32B": "qwen",
    "Gemini-2.5-Flash-Preview": "google",
    "GPT-OSS-20B": "openai",
    "GPT-5-nano": "openai",
    "Mistral-Small-3.1": "mistralai",
    "Mistral-Medium-3": "mistralai",
    "Minimax-M1": "minimax",
    "Llama-4-Maverick": "meta",
    "Llama-4-Scout": "meta",
}

SOURCE_URL = "https://sciarena.allen.ai/api/leaderboard"


def make_source_data() -> dict:
    return {
        "source_type": "url",
        "dataset_name": "SciArena leaderboard API",
        "url": [SOURCE_URL],
    }


def load_rows(input_json: Path) -> list[dict]:
    return json.loads(input_json.read_text(encoding="utf-8"))


def compute_metric_bounds(rows: list[dict]) -> dict[str, dict[str, float]]:
    rating_values = [float(row["rating"]) for row in rows]
    rank_values = [float(row["rank"]) for row in rows]
    cost_values = [
        float(row["cost_per_100_calls_usd"])
        for row in rows
        if row.get("cost_per_100_calls_usd") is not None
    ]

    bounds = {
        "elo": {
            "min_score": min(rating_values),
            "max_score": max(rating_values),
        },
        "rank": {
            "min_score": 1.0,
            "max_score": max(rank_values),
        },
    }

    if cost_values:
        bounds["cost_per_100_calls_usd"] = {
            "min_score": 0.0,
            "max_score": max(cost_values),
        }

    return bounds


def slugify_model_name(raw_model_id: str) -> str:
    # Keep close to source aliases. Lowercase, preserve dots and hyphens.
    return raw_model_id.strip().lower()


def normalize_model(raw_model_id: str) -> tuple[str, str]:
    if raw_model_id not in PROVIDER_MAP:
        raise KeyError(
            f"No provider mapping for modelId={raw_model_id!r}. "
            "Add it to PROVIDER_MAP before exporting."
        )
    developer_name = PROVIDER_MAP[raw_model_id]
    model_name = slugify_model_name(raw_model_id)
    return developer_name, model_name


def make_results(
    row: dict, metric_bounds: dict[str, dict[str, float]]
) -> list[dict]:
    results = []

    results.append(
        {
            "evaluation_result_id": "overall::elo",
            "evaluation_name": "overall_elo",
            "source_data": make_source_data(),
            "metric_config": {
                "metric_id": "elo",
                "metric_name": "Elo rating",
                "metric_type": "continuous",
                "metric_kind": "elo",
                "metric_unit": "points",
                "lower_is_better": False,
                "score_type": "continuous",
                **metric_bounds["elo"],
                "additional_details": {
                    "raw_metric_field": "rating",
                },
            },
            "score_details": {
                "score": float(row["rating"]),
                "details": {
                    "num_battles": str(row["num_battles"]),
                    "rating_q025": str(row["rating_q025"]),
                    "rating_q975": str(row["rating_q975"]),
                    "variance": str(row["variance"]),
                },
            },
        }
    )

    results.append(
        {
            "evaluation_result_id": "overall::rank",
            "evaluation_name": "overall_rank",
            "source_data": make_source_data(),
            "metric_config": {
                "metric_id": "rank",
                "metric_name": "Rank",
                "metric_type": "continuous",
                "metric_kind": "rank",
                "metric_unit": "position",
                "lower_is_better": True,
                "score_type": "continuous",
                **metric_bounds["rank"],
            },
            "score_details": {
                "score": float(row["rank"]),
            },
        }
    )

    if row.get("cost_per_100_calls_usd") is not None:
        results.append(
            {
                "evaluation_result_id": "overall::cost_per_100_calls_usd",
                "evaluation_name": "overall_cost_per_100_calls_usd",
                "source_data": make_source_data(),
                "metric_config": {
                    "metric_id": "cost_per_100_calls_usd",
                    "metric_name": "Cost per 100 calls",
                    "metric_type": "continuous",
                    "metric_kind": "cost",
                    "metric_unit": "usd",
                    "lower_is_better": True,
                    "score_type": "continuous",
                    **metric_bounds["cost_per_100_calls_usd"],
                },
                "score_details": {
                    "score": float(row["cost_per_100_calls_usd"]),
                },
            }
        )

    return results


def make_log(
    row: dict, metric_bounds: dict[str, dict[str, float]]
) -> tuple[dict, str, str]:
    raw_model_id = row["modelId"]
    developer_name, model_name = normalize_model(raw_model_id)
    ts = str(time.time())

    log = {
        "schema_version": "0.2.2",
        "evaluation_id": f"sciarena/{developer_name}/{model_name}/{ts}",
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": "SciArena leaderboard API",
            "source_type": "documentation",
            "source_organization_name": "Ai2",
            "source_organization_url": "https://sciarena.allen.ai",
            "evaluator_relationship": "third_party",
            "additional_details": {
                "api_endpoint": SOURCE_URL,
            },
        },
        "eval_library": {
            "name": "SciArena",
            "version": "unknown",
        },
        "model_info": {
            "name": raw_model_id,
            "id": f"{developer_name}/{model_name}",
            "developer": developer_name,
            "additional_details": {
                "raw_model_id": raw_model_id,
            },
        },
        "evaluation_results": make_results(row, metric_bounds),
    }
    return log, developer_name, model_name


def write_log(log: dict, out_root: Path, developer: str, model: str) -> Path:
    out_dir = out_root / "sciarena" / developer / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4()}.json"
    out_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return out_path


def export_one(
    row: dict,
    out_root: Path,
    metric_bounds: dict[str, dict[str, float]],
) -> Path:
    log, developer, model = make_log(row, metric_bounds)
    return write_log(log, out_root, developer, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input_json)

    missing = [row["modelId"] for row in rows if row["modelId"] not in PROVIDER_MAP]
    if missing:
        raise SystemExit(f"Missing provider mappings for: {missing}")

    metric_bounds = compute_metric_bounds(rows)

    exported = 0
    for row in rows:
        out_path = export_one(row, args.output_dir, metric_bounds)
        print(out_path)
        exported += 1

    print(f"Exported {exported} model(s).")


if __name__ == "__main__":
    main()
