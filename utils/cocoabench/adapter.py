"""
Convert CocoaBench aggregate system results into Every Eval Ever records.

This adapter is intentionally scoped to the publishable aggregate artifacts:

- per-system accuracy
- average runtime per task
- average cost per task
- total evaluation cost

It does not convert raw per-task tool-usage summaries into instance-level JSONL,
because those summaries do not contain enough information to produce valid
InstanceLevelEvaluationLog records (missing task inputs, answer attribution, and
full interaction traces).

Usage:
    uv run python -m utils.cocoabench.adapter \
      --csv /path/to/_agent_performance_time_costs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
    Uncertainty,
)
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_log

DEFAULT_OUTPUT_DIR = "data/cocoabench"
DEFAULT_BENCHMARK_REFERENCE_URLS = [
    "https://arxiv.org/abs/2604.11201",
    "https://cocoabench.github.io/",
]
DEFAULT_ROW_MAP: dict[str, dict[str, str]] = {
    "OpenClaw w/ GPT-5.4-high": {
        "agent_name": "OpenClaw",
        "agent_framework": "openclaw",
        "model_display_name": "GPT-5.4-high",
        "model_id": "openai/gpt-5.4-high",
        "developer": "OpenAI",
        "developer_slug": "openai",
        "model_slug": "gpt-5.4-high",
        "inference_platform": "openai",
    },
    "CodeX": {
        "agent_name": "CodeX",
        "agent_framework": "codex",
        "agent_organization": "OpenAI",
        "model_display_name": "CodeX",
        "model_id": "openai/codex",
        "developer": "OpenAI",
        "developer_slug": "openai",
        "model_slug": "codex",
        "inference_platform": "openai",
    },
    "Cocoa Agent w/ GPT-5.4-high": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "GPT-5.4-high",
        "model_id": "openai/gpt-5.4-high",
        "developer": "OpenAI",
        "developer_slug": "openai",
        "model_slug": "gpt-5.4-high",
        "inference_platform": "openai",
    },
    "OpenClaw w/ Claude-Sonnet-4.6-high": {
        "agent_name": "OpenClaw",
        "agent_framework": "openclaw",
        "model_display_name": "Claude-Sonnet-4.6-high",
        "model_id": "anthropic/claude-sonnet-4.6-high",
        "developer": "Anthropic",
        "developer_slug": "anthropic",
        "model_slug": "claude-sonnet-4.6-high",
        "inference_platform": "anthropic",
    },
    "Cocoa Agent w/ Gemini-3.1-pro": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "Gemini-3.1-pro",
        "model_id": "google/gemini-3.1-pro",
        "developer": "Google",
        "developer_slug": "google",
        "model_slug": "gemini-3.1-pro",
        "inference_platform": "google",
    },
    "ChatGPT Agent": {
        "agent_name": "ChatGPT Agent",
        "agent_framework": "chatgpt-agent",
        "agent_organization": "OpenAI",
        "model_display_name": "ChatGPT Agent",
        "model_id": "openai/chatgpt-agent",
        "developer": "OpenAI",
        "developer_slug": "openai",
        "model_slug": "chatgpt-agent",
        "inference_platform": "openai",
    },
    "Claude Code": {
        "agent_name": "Claude Code",
        "agent_framework": "claude-code",
        "agent_organization": "Anthropic",
        "model_display_name": "Claude Code",
        "model_id": "anthropic/claude-code",
        "developer": "Anthropic",
        "developer_slug": "anthropic",
        "model_slug": "claude-code",
        "inference_platform": "anthropic",
    },
    "Cocoa Agent w/ Gemini-Flash-3.0": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "Gemini-Flash-3.0",
        "model_id": "google/gemini-flash-3.0",
        "developer": "Google",
        "developer_slug": "google",
        "model_slug": "gemini-flash-3.0",
        "inference_platform": "google",
    },
    "Cocoa Agent w/ Claude-Sonnet-4.6-high": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "Claude-Sonnet-4.6-high",
        "model_id": "anthropic/claude-sonnet-4.6-high",
        "developer": "Anthropic",
        "developer_slug": "anthropic",
        "model_slug": "claude-sonnet-4.6-high",
        "inference_platform": "anthropic",
    },
    "Cocoa Agent w/ Kimi-k2.5": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "Kimi-k2.5",
        "model_id": "moonshotai/kimi-k2.5",
        "developer": "Moonshot AI",
        "developer_slug": "moonshotai",
        "model_slug": "kimi-k2.5",
        "inference_platform": "moonshotai",
    },
    "Cocoa Agent w/ Qwen3.5-397B-A13B": {
        "agent_name": "Cocoa Agent",
        "agent_framework": "cocoa-agent",
        "agent_organization": "CocoaBench",
        "model_display_name": "Qwen3.5-397B-A13B",
        "model_id": "qwen/qwen3.5-397b-a13b",
        "developer": "Qwen",
        "developer_slug": "qwen",
        "model_slug": "qwen3.5-397b-a13b",
        "inference_platform": "qwen",
    },
    "OpenAI Deep Research": {
        "agent_name": "OpenAI Deep Research",
        "agent_framework": "openai-deep-research",
        "agent_organization": "OpenAI",
        "model_display_name": "OpenAI Deep Research",
        "model_id": "openai/deep-research",
        "developer": "OpenAI",
        "developer_slug": "openai",
        "model_slug": "deep-research",
        "inference_platform": "openai",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CocoaBench aggregate CSV results to Every Eval Ever format"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to _agent_performance_time_costs.csv",
    )
    parser.add_argument(
        "--row-map",
        help=(
            "Optional JSON mapping from CSV Agent labels to canonical metadata. "
            "If omitted, the adapter uses the frozen built-in CocoaBench mapping."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--benchmark-version",
        default="1.0",
        help="Benchmark version string stored in metadata (default: 1.0)",
    )
    parser.add_argument(
        "--eval-library-version",
        default="1.0",
        help="Version string for eval_library.name=cocoabench (default: 1.0)",
    )
    parser.add_argument(
        "--evaluation-timestamp",
        help=(
            "Unix timestamp for when the evaluation run occurred. "
            "If omitted, only retrieved_timestamp is populated."
        ),
    )
    parser.add_argument(
        "--public-source-url",
        action="append",
        dest="public_source_urls",
        help=(
            "Public URL for the specific CocoaBench artifact used to create the "
            "records. May be repeated. If omitted, source_data.source_type will "
            "be set to 'other'."
        ),
    )
    parser.add_argument(
        "--benchmark-reference-url",
        action="append",
        dest="benchmark_reference_urls",
        help=(
            "Public benchmark reference URL, such as the CocoaBench paper or "
            "project page. May be repeated."
        ),
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_row_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return DEFAULT_ROW_MAP

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("row-map must be a JSON object keyed by CSV Agent labels")
    return data


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def stringify_details(details: dict[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, separators=(",", ":"))
        else:
            out[key] = str(value)
    return out


def compute_metric_bounds(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    avg_times = [x for x in (parse_optional_float(r.get("AvgTime_s")) for r in rows) if x is not None]
    avg_costs = [x for x in (parse_optional_float(r.get("AvgCost_USD")) for r in rows) if x is not None]
    total_costs = [x for x in (parse_optional_float(r.get("TotalCost_USD")) for r in rows) if x is not None]

    bounds: dict[str, dict[str, float]] = {
        "accuracy_percent": {
            "min_score": 0.0,
            "max_score": 100.0,
        }
    }

    if avg_times:
        bounds["avg_time_s"] = {"min_score": 0.0, "max_score": max(avg_times)}
    if avg_costs:
        bounds["avg_cost_usd"] = {"min_score": 0.0, "max_score": max(avg_costs)}
    if total_costs:
        bounds["total_cost_usd"] = {"min_score": 0.0, "max_score": max(total_costs)}

    return bounds


def make_source_data(
    benchmark_version: str,
    answered: int,
    public_source_urls: list[str],
    benchmark_reference_urls: list[str],
) -> SourceDataPrivate | SourceDataUrl:
    additional_details = {
        "samples_number": str(answered),
        "benchmark_version": benchmark_version,
    }
    if benchmark_reference_urls:
        additional_details["benchmark_reference_urls_json"] = json.dumps(
            benchmark_reference_urls, separators=(",", ":")
        )

    dataset_name = f"CocoaBench v{benchmark_version}"
    if public_source_urls:
        return SourceDataUrl(
            dataset_name=dataset_name,
            source_type="url",
            url=public_source_urls,
            additional_details=additional_details,
        )

    additional_details["artifact_visibility"] = "private"
    additional_details["artifact_provenance"] = (
        "shared_directly_by_benchmark_authors"
    )
    return SourceDataPrivate(
        dataset_name=dataset_name,
        source_type="other",
        additional_details=additional_details,
    )


def make_generation_config(
    agent_label: str,
    row_meta: dict[str, str],
    benchmark_version: str,
) -> GenerationConfig:
    agent_details = {
        "agent_label": agent_label,
        "agent_name": row_meta["agent_name"],
        "agent_framework": row_meta["agent_framework"],
        "benchmark_version": benchmark_version,
    }
    if row_meta.get("agent_organization"):
        agent_details["agent_organization"] = row_meta["agent_organization"]

    if row_meta.get("agent_version"):
        agent_details["agent_version"] = row_meta["agent_version"]

    return GenerationConfig(
        generation_args=GenerationArgs(
            agentic_eval_config=AgenticEvalConfig(
                additional_details=agent_details,
            ),
        ),
        additional_details={
            "benchmark_name": "CocoaBench",
            "benchmark_version": benchmark_version,
        },
    )


def make_accuracy_result(
    row: dict[str, str],
    row_meta: dict[str, str],
    bounds: dict[str, dict[str, float]],
    benchmark_version: str,
    public_source_urls: list[str],
    benchmark_reference_urls: list[str],
    evaluation_timestamp: str | None,
) -> EvaluationResult:
    answered = parse_optional_int(row.get("Answered")) or 0
    return EvaluationResult(
        evaluation_result_id="overall::accuracy_percent",
        evaluation_name="cocoabench.overall.accuracy_percent",
        source_data=make_source_data(
            benchmark_version,
            answered,
            public_source_urls,
            benchmark_reference_urls,
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description="Overall task success rate on CocoaBench aggregate release",
            metric_id="cocoabench.overall.accuracy_percent",
            metric_name="Accuracy",
            metric_kind="accuracy",
            metric_unit="percent",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            **bounds["accuracy_percent"],
        ),
        score_details=ScoreDetails(
            score=float(row["AccuracyPercent"]),
            details=stringify_details(
                {
                    "correct": parse_optional_int(row.get("Correct")),
                    "wrong": parse_optional_int(row.get("Wrong")),
                    "answered": answered,
                    "agent_label": row["Agent"],
                    "agent_name": row_meta["agent_name"],
                }
            ),
            uncertainty=Uncertainty(num_samples=answered) if answered else None,
        ),
        generation_config=make_generation_config(
            row["Agent"], row_meta, benchmark_version
        ),
    )


def make_optional_metric_result(
    *,
    evaluation_result_id: str,
    evaluation_name: str,
    metric_id: str,
    metric_name: str,
    metric_kind: str,
    metric_unit: str,
    evaluation_description: str,
    score: float | None,
    lower_is_better: bool,
    bounds_key: str,
    row: dict[str, str],
    row_meta: dict[str, str],
    bounds: dict[str, dict[str, float]],
    benchmark_version: str,
    public_source_urls: list[str],
    benchmark_reference_urls: list[str],
    evaluation_timestamp: str | None,
) -> EvaluationResult | None:
    if score is None:
        return None

    answered = parse_optional_int(row.get("Answered")) or 0
    return EvaluationResult(
        evaluation_result_id=evaluation_result_id,
        evaluation_name=evaluation_name,
        source_data=make_source_data(
            benchmark_version,
            answered,
            public_source_urls,
            benchmark_reference_urls,
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description=evaluation_description,
            metric_id=metric_id,
            metric_name=metric_name,
            metric_kind=metric_kind,
            metric_unit=metric_unit,
            lower_is_better=lower_is_better,
            score_type=ScoreType.continuous,
            **bounds[bounds_key],
        ),
        score_details=ScoreDetails(
            score=score,
            details=stringify_details(
                {
                    "agent_label": row["Agent"],
                    "agent_name": row_meta["agent_name"],
                    "answered": answered,
                }
            ),
        ),
        generation_config=make_generation_config(
            row["Agent"], row_meta, benchmark_version
        ),
    )


def make_log(
    row: dict[str, str],
    row_meta: dict[str, str],
    bounds: dict[str, dict[str, float]],
    benchmark_version: str,
    eval_library_version: str,
    public_source_urls: list[str],
    benchmark_reference_urls: list[str],
    source_metadata_details: dict[str, str],
    retrieved_timestamp: str,
    evaluation_timestamp: str | None,
) -> tuple[EvaluationLog, str, str]:
    model_id = row_meta["model_id"]
    if "/" not in model_id:
        raise ValueError(
            f"row-map model_id must look like developer/model, got {model_id!r}"
        )

    accuracy_result = make_accuracy_result(
        row,
        row_meta,
        bounds,
        benchmark_version,
        public_source_urls,
        benchmark_reference_urls,
        evaluation_timestamp,
    )

    avg_time = parse_optional_float(row.get("AvgTime_s"))
    avg_cost = parse_optional_float(row.get("AvgCost_USD"))
    total_cost = parse_optional_float(row.get("TotalCost_USD"))

    results = [accuracy_result]

    avg_time_result = make_optional_metric_result(
        evaluation_result_id="overall::avg_time_s",
        evaluation_name="cocoabench.overall.avg_time_seconds",
        metric_id="cocoabench.overall.avg_time_seconds",
        metric_name="Average time per task",
        metric_kind="latency",
        metric_unit="seconds",
        evaluation_description="Average task runtime in seconds",
        score=avg_time,
        lower_is_better=True,
        bounds_key="avg_time_s",
        row=row,
        row_meta=row_meta,
        bounds=bounds,
        benchmark_version=benchmark_version,
        public_source_urls=public_source_urls,
        benchmark_reference_urls=benchmark_reference_urls,
        evaluation_timestamp=evaluation_timestamp,
    )
    if avg_time_result is not None:
        results.append(avg_time_result)

    avg_cost_result = make_optional_metric_result(
        evaluation_result_id="overall::avg_cost_usd",
        evaluation_name="cocoabench.overall.avg_cost_usd",
        metric_id="cocoabench.overall.avg_cost_usd",
        metric_name="Average cost per task",
        metric_kind="cost",
        metric_unit="usd",
        evaluation_description="Average task cost in USD",
        score=avg_cost,
        lower_is_better=True,
        bounds_key="avg_cost_usd",
        row=row,
        row_meta=row_meta,
        bounds=bounds,
        benchmark_version=benchmark_version,
        public_source_urls=public_source_urls,
        benchmark_reference_urls=benchmark_reference_urls,
        evaluation_timestamp=evaluation_timestamp,
    )
    if avg_cost_result is not None:
        results.append(avg_cost_result)

    total_cost_result = make_optional_metric_result(
        evaluation_result_id="overall::total_cost_usd",
        evaluation_name="cocoabench.overall.total_cost_usd",
        metric_id="cocoabench.overall.total_cost_usd",
        metric_name="Total evaluation cost",
        metric_kind="cost",
        metric_unit="usd",
        evaluation_description="Total cost of the released evaluation run in USD",
        score=total_cost,
        lower_is_better=True,
        bounds_key="total_cost_usd",
        row=row,
        row_meta=row_meta,
        bounds=bounds,
        benchmark_version=benchmark_version,
        public_source_urls=public_source_urls,
        benchmark_reference_urls=benchmark_reference_urls,
        evaluation_timestamp=evaluation_timestamp,
    )
    if total_cost_result is not None:
        results.append(total_cost_result)

    agent_slug = row_meta["agent_framework"]
    sanitized_model_id = model_id.replace("/", "_")
    eval_timestamp = evaluation_timestamp or retrieved_timestamp
    evaluation_id = f"cocoabench/{agent_slug}__{sanitized_model_id}/{eval_timestamp}"

    model_details = {
        "agent_label": row["Agent"],
        "agent_name": row_meta["agent_name"],
        "agent_framework": row_meta["agent_framework"],
    }
    if row_meta.get("agent_organization"):
        model_details["agent_organization"] = row_meta["agent_organization"]
    if row_meta.get("agent_version"):
        model_details["agent_version"] = row_meta["agent_version"]

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        evaluation_timestamp=evaluation_timestamp,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name="CocoaBench aggregate results shared by benchmark authors",
            source_type="evaluation_run",
            source_organization_name="CocoaBench",
            source_organization_url="https://cocoabench.github.io/",
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=source_metadata_details or None,
        ),
        eval_library=EvalLibrary(
            name="cocoabench",
            version=eval_library_version,
        ),
        model_info=ModelInfo(
            name=row_meta["model_display_name"],
            id=model_id,
            developer=row_meta["developer"],
            inference_platform=row_meta.get("inference_platform"),
            additional_details=model_details,
        ),
        evaluation_results=results,
    )

    return log, row_meta["developer_slug"], row_meta["model_slug"]


def ensure_row_map_complete(
    rows: list[dict[str, str]], row_map: dict[str, dict[str, str]]
) -> None:
    missing = sorted({row["Agent"] for row in rows} - set(row_map))
    if missing:
        raise ValueError(
            "row-map is missing entries for these Agent labels: "
            + ", ".join(missing)
        )


def validate_row_meta(agent_label: str, row_meta: dict[str, str]) -> None:
    required = [
        "agent_name",
        "agent_framework",
        "model_display_name",
        "model_id",
        "developer",
        "developer_slug",
        "model_slug",
    ]
    missing = [key for key in required if not row_meta.get(key)]
    if missing:
        raise ValueError(
            f"row-map entry for {agent_label!r} is missing required keys: {missing}"
        )


def main() -> None:
    args = parse_args()

    rows = load_rows(Path(args.csv))
    row_map = load_row_map(Path(args.row_map) if args.row_map else None)
    ensure_row_map_complete(rows, row_map)

    public_source_urls = args.public_source_urls or []
    benchmark_reference_urls = (
        args.benchmark_reference_urls or DEFAULT_BENCHMARK_REFERENCE_URLS
    )
    source_metadata_details = {
        "benchmark_version": args.benchmark_version,
        "release_artifact": "aggregate_author_release",
    }

    bounds = compute_metric_bounds(rows)
    retrieved_timestamp = str(time.time())
    count = 0

    for row in rows:
        row_meta = row_map[row["Agent"]]
        validate_row_meta(row["Agent"], row_meta)
        log, developer_slug, model_slug = make_log(
            row=row,
            row_meta=row_meta,
            bounds=bounds,
            benchmark_version=args.benchmark_version,
            eval_library_version=args.eval_library_version,
            public_source_urls=public_source_urls,
            benchmark_reference_urls=benchmark_reference_urls,
            source_metadata_details=source_metadata_details,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=args.evaluation_timestamp,
        )
        filepath = save_evaluation_log(log, args.output_dir, developer_slug, model_slug)
        print(filepath)
        count += 1

    print(f"\nGenerated {count} CocoaBench records in {args.output_dir}/")


if __name__ == "__main__":
    main()
