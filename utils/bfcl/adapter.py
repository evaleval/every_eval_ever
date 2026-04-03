#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

SOURCE_CSV_URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"
SOURCE_LEADERBOARD_URL = "https://gorilla.cs.berkeley.edu/leaderboard.html"
SCHEMA_VERSION = "0.2.2"

ORG_SLUG_OVERRIDES = {
    "Anthropic": "anthropic",
    "OpenAI": "openai",
    "Google": "google",
    "Meta": "meta",
    "xAI": "xai",
    "DeepSeek": "deepseek",
    "Qwen": "qwen",
    "Zhipu AI": "zhipu",
    "Zhipu": "zhipu",
    "Mistral AI": "mistralai",
    "Mistral": "mistralai",
    "MiniMax": "minimax",
    "Moonshot AI": "moonshotai",
    "NVIDIA": "nvidia",
    "Cohere": "cohere",
    "Fireworks AI": "fireworks",
    "Berkeley Gorilla": "gorilla",
}

@dataclass(frozen=True)
class MetricSpec:
    evaluation_name: str
    column: str
    metric_id: str
    metric_name: str
    metric_kind: str
    metric_unit: str
    lower_is_better: bool
    min_score: float
    max_score: float | None = None
    use_observed_max: bool = False


METRIC_SPECS = [
    MetricSpec(
        evaluation_name="bfcl.overall.rank",
        column="Rank",
        metric_id="bfcl.overall.rank",
        metric_name="Overall rank",
        metric_kind="rank",
        metric_unit="position",
        lower_is_better=True,
        min_score=1.0,
        use_observed_max=True,
    ),
    MetricSpec(
        evaluation_name="bfcl.overall.overall_accuracy",
        column="Overall Acc",
        metric_id="bfcl.overall.overall_accuracy",
        metric_name="Overall accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.overall.total_cost_usd",
        column="Total Cost ($)",
        metric_id="bfcl.overall.total_cost_usd",
        metric_name="Total cost",
        metric_kind="cost",
        metric_unit="usd",
        lower_is_better=True,
        min_score=0.0,
        use_observed_max=True,
    ),
    MetricSpec(
        evaluation_name="bfcl.overall.latency_mean_s",
        column="Latency Mean (s)",
        metric_id="bfcl.overall.latency_mean_s",
        metric_name="Latency mean",
        metric_kind="latency",
        metric_unit="seconds",
        lower_is_better=True,
        min_score=0.0,
        use_observed_max=True,
    ),
    MetricSpec(
        evaluation_name="bfcl.overall.latency_std_s",
        column="Latency Standard Deviation (s)",
        metric_id="bfcl.overall.latency_std_s",
        metric_name="Latency standard deviation",
        metric_kind="latency",
        metric_unit="seconds",
        lower_is_better=True,
        min_score=0.0,
        use_observed_max=True,
    ),
    MetricSpec(
        evaluation_name="bfcl.overall.latency_p95_s",
        column="Latency 95th Percentile (s)",
        metric_id="bfcl.overall.latency_p95_s",
        metric_name="Latency 95th percentile",
        metric_kind="latency",
        metric_unit="seconds",
        lower_is_better=True,
        min_score=0.0,
        use_observed_max=True,
    ),
    MetricSpec(
        evaluation_name="bfcl.non_live.ast_accuracy",
        column="Non-Live AST Acc",
        metric_id="bfcl.non_live.ast_accuracy",
        metric_name="Non-live AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.non_live.simple_ast_accuracy",
        column="Non-Live Simple AST",
        metric_id="bfcl.non_live.simple_ast_accuracy",
        metric_name="Non-live simple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.non_live.multiple_ast_accuracy",
        column="Non-Live Multiple AST",
        metric_id="bfcl.non_live.multiple_ast_accuracy",
        metric_name="Non-live multiple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.non_live.parallel_ast_accuracy",
        column="Non-Live Parallel AST",
        metric_id="bfcl.non_live.parallel_ast_accuracy",
        metric_name="Non-live parallel AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.non_live.parallel_multiple_ast_accuracy",
        column="Non-Live Parallel Multiple AST",
        metric_id="bfcl.non_live.parallel_multiple_ast_accuracy",
        metric_name="Non-live parallel multiple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.live.live_accuracy",
        column="Live Acc",
        metric_id="bfcl.live.live_accuracy",
        metric_name="Live accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.live.live_simple_ast_accuracy",
        column="Live Simple AST",
        metric_id="bfcl.live.live_simple_ast_accuracy",
        metric_name="Live simple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.live.live_multiple_ast_accuracy",
        column="Live Multiple AST",
        metric_id="bfcl.live.live_multiple_ast_accuracy",
        metric_name="Live multiple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.live.live_parallel_ast_accuracy",
        column="Live Parallel AST",
        metric_id="bfcl.live.live_parallel_ast_accuracy",
        metric_name="Live parallel AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.live.live_parallel_multiple_ast_accuracy",
        column="Live Parallel Multiple AST",
        metric_id="bfcl.live.live_parallel_multiple_ast_accuracy",
        metric_name="Live parallel multiple AST accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.multi_turn.accuracy",
        column="Multi Turn Acc",
        metric_id="bfcl.multi_turn.accuracy",
        metric_name="Multi-turn accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.multi_turn.base_accuracy",
        column="Multi Turn Base",
        metric_id="bfcl.multi_turn.base_accuracy",
        metric_name="Multi-turn base accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.multi_turn.miss_function_accuracy",
        column="Multi Turn Miss Func",
        metric_id="bfcl.multi_turn.miss_function_accuracy",
        metric_name="Multi-turn missing function accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.multi_turn.miss_parameter_accuracy",
        column="Multi Turn Miss Param",
        metric_id="bfcl.multi_turn.miss_parameter_accuracy",
        metric_name="Multi-turn missing parameter accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.multi_turn.long_context_accuracy",
        column="Multi Turn Long Context",
        metric_id="bfcl.multi_turn.long_context_accuracy",
        metric_name="Multi-turn long-context accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.web_search.accuracy",
        column="Web Search Acc",
        metric_id="bfcl.web_search.accuracy",
        metric_name="Web-search accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.web_search.base_accuracy",
        column="Web Search Base",
        metric_id="bfcl.web_search.base_accuracy",
        metric_name="Web-search base accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.web_search.no_snippet_accuracy",
        column="Web Search No Snippet",
        metric_id="bfcl.web_search.no_snippet_accuracy",
        metric_name="Web-search no-snippet accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.memory.accuracy",
        column="Memory Acc",
        metric_id="bfcl.memory.accuracy",
        metric_name="Memory accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.memory.kv_accuracy",
        column="Memory KV",
        metric_id="bfcl.memory.kv_accuracy",
        metric_name="Memory KV accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.memory.vector_accuracy",
        column="Memory Vector",
        metric_id="bfcl.memory.vector_accuracy",
        metric_name="Memory vector accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.memory.recursive_summarization_accuracy",
        column="Memory Recursive Summarization",
        metric_id="bfcl.memory.recursive_summarization_accuracy",
        metric_name="Memory recursive summarization accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.relevance.relevance_detection_accuracy",
        column="Relevance Detection",
        metric_id="bfcl.relevance.relevance_detection_accuracy",
        metric_name="Relevance detection accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.relevance.irrelevance_detection_accuracy",
        column="Irrelevance Detection",
        metric_id="bfcl.relevance.irrelevance_detection_accuracy",
        metric_name="Irrelevance detection accuracy",
        metric_kind="accuracy",
        metric_unit="percentage",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.format_sensitivity.max_delta",
        column="Format Sensitivity Max Delta",
        metric_id="bfcl.format_sensitivity.max_delta",
        metric_name="Format sensitivity max delta",
        metric_kind="difference",
        metric_unit="percentage_points",
        lower_is_better=True,
        min_score=0.0,
        max_score=100.0,
    ),
    MetricSpec(
        evaluation_name="bfcl.format_sensitivity.stddev",
        column="Format Sensitivity Standard Deviation",
        metric_id="bfcl.format_sensitivity.stddev",
        metric_name="Format sensitivity standard deviation",
        metric_kind="difference",
        metric_unit="percentage_points",
        lower_is_better=True,
        min_score=0.0,
        max_score=100.0,
    ),
]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def developer_slug(org: str) -> str:
    return ORG_SLUG_OVERRIDES.get(org, slugify(org))


def model_slug(raw_model: str) -> str:
    return slugify(raw_model)


def parse_mode(raw_model: str) -> str | None:
    m = re.search(r"\(([^)]+)\)\s*$", raw_model)
    return m.group(1) if m else None


def parse_value(raw: str) -> float | None:
    raw = raw.strip()
    if raw == "" or raw.upper() == "N/A":
        return None
    if raw.endswith("%"):
        raw = raw[:-1]
    raw = raw.replace(",", "")
    return float(raw)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_source_data() -> dict:
    return {
        "source_type": "url",
        "dataset_name": "BFCL leaderboard CSV",
        "url": [SOURCE_CSV_URL],
    }


def compute_observed_max_scores(rows: list[dict]) -> dict[str, float]:
    observed_max_scores: dict[str, float] = {}
    for spec in METRIC_SPECS:
        if not spec.use_observed_max:
            continue

        values = []
        for row in rows:
            value = parse_value(row.get(spec.column, ""))
            if value is not None:
                values.append(value)

        if not values:
            raise SystemExit(
                f"Could not determine max_score for {spec.column!r}; no numeric values were found."
            )

        observed_max_scores[spec.column] = max(values)

    return observed_max_scores


def make_result(
    row: dict,
    spec: MetricSpec,
    observed_max_scores: dict[str, float],
) -> dict | None:
    value = parse_value(row[spec.column])
    if value is None:
        return None

    max_score = spec.max_score
    if spec.use_observed_max:
        max_score = observed_max_scores[spec.column]

    return {
        "evaluation_result_id": f"{spec.evaluation_name}::{spec.metric_id.split('.')[-1]}",
        "evaluation_name": spec.evaluation_name,
        "source_data": make_source_data(),
        "metric_config": {
            "metric_id": spec.metric_id,
            "metric_name": spec.metric_name,
            "metric_kind": spec.metric_kind,
            "metric_unit": spec.metric_unit,
            "lower_is_better": spec.lower_is_better,
            "score_type": "continuous",
            "min_score": spec.min_score,
            "max_score": max_score,
            "additional_details": {
                "raw_metric_field": spec.column,
            },
        },
        "score_details": {
            "score": value,
        },
    }


def make_results(row: dict, observed_max_scores: dict[str, float]) -> list[dict]:
    results = []
    for spec in METRIC_SPECS:
        result = make_result(row, spec, observed_max_scores)
        if result is not None:
            results.append(result)
    return results


def make_log(row: dict, observed_max_scores: dict[str, float]) -> tuple[dict, str, str]:
    raw_model = row["Model"]
    org = row["Organization"]
    developer = developer_slug(org)
    model = model_slug(raw_model)
    ts = str(time.time())

    additional_details = {
        "raw_model_name": raw_model,
        "organization": org,
        "license": row.get("License", ""),
    }
    mode = parse_mode(raw_model)
    if mode is not None:
        additional_details["mode"] = mode
    if row.get("Model Link"):
        additional_details["model_link"] = row["Model Link"]

    log = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": f"bfcl/{developer}/{model}/{ts}",
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": "BFCL leaderboard CSV",
            "source_type": "documentation",
            "source_organization_name": "UC Berkeley Gorilla",
            "source_organization_url": SOURCE_LEADERBOARD_URL,
            "evaluator_relationship": "third_party",
            "additional_details": {
                "csv_url": SOURCE_CSV_URL,
                "leaderboard_url": SOURCE_LEADERBOARD_URL,
                "leaderboard_version": "BFCL V4",
            },
        },
        "eval_library": {
            "name": "BFCL",
            "version": "v4",
        },
        "model_info": {
            "name": raw_model,
            "id": f"{developer}/{model}",
            "developer": developer,
            "additional_details": additional_details,
        },
        "evaluation_results": make_results(row, observed_max_scores),
    }

    return log, developer, model


def write_log(log: dict, out_root: Path, developer: str, model: str) -> Path:
    out_dir = out_root / "bfcl" / developer / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4()}.json"
    out_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return out_path


def export_one(row: dict, out_root: Path, observed_max_scores: dict[str, float]) -> Path:
    log, developer, model = make_log(row, observed_max_scores)
    return write_log(log, out_root, developer, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Exact BFCL Model cell to export. If omitted, export all rows.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    observed_max_scores = compute_observed_max_scores(rows)

    if args.model is not None:
        matches = [row for row in rows if row["Model"] == args.model]
        if not matches:
            raise SystemExit(f"Model {args.model!r} not found in {args.input_csv}")
        print(export_one(matches[0], args.output_dir, observed_max_scores))
        return

    exported = 0
    for row in rows:
        out_path = export_one(row, args.output_dir, observed_max_scores)
        print(out_path)
        exported += 1

    print(f"Exported {exported} model(s).")


if __name__ == "__main__":
    main()
