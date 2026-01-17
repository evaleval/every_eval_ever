"""
Script to convert HELM leaderboard data to the EvalEval schema format.

Supports multiple HELM variants:
- HELM_Capabilities
- HELM_Lite
- HELM_Classic
- HELM_Instruct
- HELM_MMLU

Usage:
    uv run python -m scripts.helm.adapter --leaderboard_name HELM_Lite --source_data_url <url>
"""

import math
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List

from eval_types import (
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    fetch_json,
    get_developer,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)


def parse_args():
    """Parse CLI arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--leaderboard_name",
        type=str,
        default="HELM_Capabilities",
        choices=[
            "HELM_Capabilities",
            "HELM_Lite",
            "HELM_Classic",
            "HELM_Instruct",
            "HELM_MMLU",
        ],
    )
    parser.add_argument(
        "--source_data_url",
        type=str,
        default=(
            "https://storage.googleapis.com/crfm-helm-public/"
            "capabilities/benchmark_output/releases/v1.12.0/"
            "groups/core_scenarios.json"
        ),
    )
    return parser.parse_args()


def clean_model_name(model_name: str) -> str:
    """Remove parentheses from model name."""
    return model_name.replace("(", "").replace(")", "")


def extract_generation_config(run_specs: List[str]) -> Dict[str, Any]:
    """Extract generation configuration from HELM run spec strings."""
    generation_config: Dict[str, Any] = defaultdict(list)

    for run_spec in run_specs:
        _, args_str = run_spec.split(":", 1)
        args = args_str.split(",")

        for arg in args:
            key, value = arg.split("=")
            if key == "model":
                continue
            generation_config[key].append(value)

    # Collapse values if they are identical
    for key, values in list(generation_config.items()):
        if len(set(values)) == 1:
            generation_config[key] = values[0]

    return dict(generation_config)


def extract_model_info_from_row(row: List[Dict[str, Any]], model_name: str):
    """Extract model metadata from leaderboard row."""
    run_spec_names = next(
        (cell["run_spec_names"] for cell in row if "run_spec_names" in cell),
        None,
    )

    if "(" in model_name and ")" in model_name:
        model_name = clean_model_name(model_name)

    if not run_spec_names:
        developer = get_developer(model_name)
        if developer == "unknown":
            model_id = model_name.replace(" ", "-")
        else:
            model_id = f"{developer}/{model_name.replace(' ', '-')}"
    else:
        spec = run_spec_names[0]
        args = spec.split(":", 1)[1].split(",")

        model_details = next(
            (arg.split("=", 1)[1] for arg in args if arg.startswith("model=")),
            "",
        )

        developer = model_details.split("_")[0]
        model_id = model_details.replace("_", "/")

    if developer == "unknown":
        developer = get_developer(model_name)

    return make_model_info(
        model_name=model_name,
        developer=developer,
        inference_platform="unknown",
    ), model_id


def find_column_ranges(tab_rows: List[List[Dict[str, Any]]]):
    """Determine min/max values for each metric column."""
    num_columns = len(tab_rows[0]) - 1
    mins = [0.0] * num_columns
    maxs = [0.0] * num_columns

    for row in tab_rows:
        for idx, cell in enumerate(row[1:], start=0):
            value = cell.get("value", 0)
            if value is not None:
                mins[idx] = min(mins[idx], value)
                maxs[idx] = max(maxs[idx], value)

    return mins, maxs


def convert(
    leaderboard_name: str,
    leaderboard_data: List[Dict[str, Any]],
    source_data: List[str],
):
    """Convert HELM leaderboard data into unified evaluation logs."""
    retrieved_timestamp = str(time.time())

    model_infos: Dict[str, Any] = {}
    model_ids: Dict[str, str] = {}
    model_results: Dict[str, Dict[str, EvaluationResult]] = defaultdict(dict)

    for tab in leaderboard_data:
        tab_name = tab.get("title")
        headers = tab.get("header")
        rows = tab.get("rows")

        mins, maxs = find_column_ranges(rows)

        for row in rows:
            model_name = row[0].get("value")

            if model_name not in model_infos:
                model_info, model_id = extract_model_info_from_row(row, model_name)
                model_infos[model_name] = model_info
                model_ids[model_name] = model_id

            for col_idx, (header, cell) in enumerate(zip(headers[1:], row[1:])):
                full_eval_name = header.get("value")
                short_name = (
                    full_eval_name.split()[0]
                    if "-" in full_eval_name
                    else full_eval_name
                )

                is_new_metric = (
                    tab_name.lower() == "accuracy"
                    or short_name not in model_results[model_name]
                    or "instruct" in leaderboard_name.lower()
                )

                if is_new_metric:
                    metric_config = MetricConfig(
                        evaluation_description=header.get("description"),
                        lower_is_better=header.get("lower_is_better", False),
                        min_score=(
                            0.0 if mins[col_idx] >= 0 else math.floor(mins[col_idx])
                        ),
                        max_score=(
                            1.0 if maxs[col_idx] <= 1 else math.ceil(maxs[col_idx])
                        ),
                        score_type=ScoreType.continuous,
                    )

                    generation_config = (
                        extract_generation_config(cell.get("run_spec_names", []))
                        if cell.get("run_spec_names")
                        else {}
                    )

                    model_results[model_name][short_name] = EvaluationResult(
                        evaluation_name=full_eval_name,
                        metric_config=metric_config,
                        score_details=ScoreDetails(
                            score=round(cell.get("value"), 3)
                            if cell.get("value") is not None
                            else -1,
                            details={
                                "description": cell.get("description"),
                                "tab": tab_name,
                            },
                        ),
                        generation_config=generation_config,
                    )
                else:
                    # Add extra score details under the same metric
                    existing = model_results[model_name][short_name]
                    detail_key = (
                        full_eval_name
                        if full_eval_name != existing.evaluation_name
                        else f"{full_eval_name} - {tab_name}"
                    )

                    existing.score_details.details[detail_key] = {
                        "description": cell.get("description"),
                        "tab": tab_name,
                        "score": cell.get("value"),
                    }

    # Save evaluation logs
    for model_name, results_by_metric in model_results.items():
        model_info = model_infos[model_name]
        model_id = model_ids[model_name]

        evaluation_id = (
            f"{leaderboard_name}/"
            f"{model_id.replace('/', '_')}/"
            f"{retrieved_timestamp}"
        )

        eval_log = EvaluationLog(
            schema_version="0.1.0",
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=make_source_metadata(
                source_name=leaderboard_name,
                organization_name="crfm",
                evaluator_relationship=EvaluatorRelationship.third_party,
            ),
            model_info=model_info,
            source_data=source_data,
            evaluation_results=list(results_by_metric.values()),
        )

        # Determine output path
        if model_info.developer == "unknown":
            developer = model_id
            model = model_id
        else:
            if "/" in model_id:
                developer, model = model_id.split("/", 1)
            else:
                developer = model_info.developer
                model = model_id

        filepath = save_evaluation_log(
            eval_log,
            f"data/{leaderboard_name}",
            developer,
            model,
        )
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    args = parse_args()

    leaderboard_name = args.leaderboard_name.lower()
    source_data = [args.source_data_url]

    print(f"Fetching {leaderboard_name} data from {args.source_data_url}")
    leaderboard_data = fetch_json(source_data[0])

    convert(
        leaderboard_name=leaderboard_name,
        leaderboard_data=leaderboard_data,
        source_data=source_data,
    )

    print("Done!")
