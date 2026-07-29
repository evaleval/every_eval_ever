#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)

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
    rating_values = []
    rank_values = []
    for row in rows:
        try:
            rating_values.append(float(row["rating"]))
            rank_values.append(float(row["rank"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rating_values or not rank_values:
        raise ValueError('SciArena has no rows with usable rating and rank')
    cost_values = [
        float(row["cost_per_100_calls_usd"])
        for row in rows
        if row.get("cost_per_100_calls_usd") is not None
        and _is_float(row["cost_per_100_calls_usd"])
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


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def slugify_model_name(raw_model_id: str) -> str:
    # Keep close to source aliases while ensuring a single path segment.
    model_name = sanitize_filename(raw_model_id.strip().lower())
    return model_name.replace("\\", "-").replace("/", "-")


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
    row: dict,
    metric_bounds: dict[str, dict[str, float]],
    retrieved_timestamp: str,
) -> tuple[dict, str, str]:
    raw_model_id = row["modelId"]
    developer_name, model_name = normalize_model(raw_model_id)

    log = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": (
            f"sciarena/{developer_name}/{model_name}/{retrieved_timestamp}"
        ),
        "retrieved_timestamp": retrieved_timestamp,
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
                "deployment_type": "unknown",
                "model_availability": "unknown",
            },
        },
        "evaluation_results": make_results(row, metric_bounds),
    }
    return log, developer_name, model_name


def convert_rows(
    rows: list[dict],
    out_root: Path,
    metric_bounds: dict[str, dict[str, float]],
    retrieved_timestamp: str,
) -> SourceConversionResult[EvaluationLogOutput]:
    outputs = []
    failures = []
    for index, row in enumerate(rows):
        try:
            raw_log, developer, model = make_log(
                row, metric_bounds, retrieved_timestamp
            )
            log = EvaluationLog.model_validate(raw_log)
            outputs.append(
                EvaluationLogOutput(
                    eval_log=log,
                    base_dir=out_root / 'sciarena',
                    developer=developer,
                    model_name=model,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'SciArena row {index}',
                    reason=str(exc),
                    source_record=row,
                )
            )
    return SourceConversionResult(
        source_name='SciArena leaderboard API',
        total_records=len(rows),
        records=outputs,
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input_json)
    retrieved_timestamp = str(time.time())

    metric_bounds = compute_metric_bounds(rows)
    result = convert_rows(
        rows,
        args.output_dir,
        metric_bounds,
        retrieved_timestamp,
    )
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(path)
    print(f"Exported {len(paths)} model(s).")
    if result.failures:
        report_path = save_failure_report(
            result,
            args.failure_report
            or default_failure_report_path(args.output_dir / 'sciarena'),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()


if __name__ == "__main__":
    main()
