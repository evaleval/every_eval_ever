import math
import os
import time
import uuid
import requests

from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List, Optional

from eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    EvaluationResult,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceMetadata,
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


def download_leaderboard(url: str) -> Dict[str, Any]:
    """Download leaderboard JSON data."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def save_to_file(eval_log: EvaluationLog, output_path: str) -> None:
    """Serialize EvaluationLog and save it to disk."""
    try:
        json_str = eval_log.model_dump_json(indent=2, exclude_none=True)
        with open(output_path, "w") as f:
            f.write(json_str)
        print(f"Saved unified eval log to {output_path}")
    except Exception as exc:
        print(f"Failed to save eval log: {exc}")
        raise


def assign_developer_to_model_by_name(
    model_name: str
) -> Optional[str]:
    model_name_to_developer = {
        'ada': 'openai',
        'alpaca': 'stanford',
        'anthropic': 'anthropic',
        'babbage': 'openai',
        'bloom': 'bigscience',
        'cohere': 'cohere',
        'curie': 'openai',
        'davinci': 'openai',
        'falcon': 'tii-uae',
        'glm': 'zhipu-ai',
        'gpt-j': 'eleuther-ai',
        'gpt-neox': 'eleuther-ai',
        'gpt': 'openai',
        'instructpalmyra': 'writer',
        'j1': 'ai21',
        'jurassic': 'ai21',
        'llama': 'meta',
        'luminous': 'aleph-alpha',
        'mistral': 'mistral-ai',
        'mpt': 'mosaicml',
        'opt': 'meta',
        'palmyra': 'writer',
        'pythia': 'eleuther-ai',
        'redpajama': 'together',
        't0pp': 'bigscience',
        't5': 'google',
        'text-': 'openai',
        'tnlg': 'microsoft',
        'ul2': 'google',
        'vicuna': 'lmsys',
        'yalm': 'yandex'
    }

    for key, model_developer in model_name_to_developer.items():
        if model_name.lower().startswith(key):
            return model_developer
    
    return 'unknown'


def clean_model_name_from_parenthesis(model_name: str) -> str:
    return model_name.replace('(', '').replace(')', '')


def extract_generation_config_from_run_specs(
    run_specs: List[str],
) -> Dict[str, Any]:
    """
    Extract generation configuration from HELM run spec strings.
    """
    generation_config = defaultdict(list)

    for run_spec in run_specs:
        _, args_str = run_spec.split(":", 1)
        args = args_str.split(",")

        for arg in args:
            key, value = arg.split("=")
            if key == "model":
                continue
            generation_config[key].append(value)

    # Collapse values if they are identical
    for key, values in generation_config.items():
        if len(set(values)) == 1:
            generation_config[key] = values[0]

    return generation_config


def extract_model_info(row: List[Dict[str, Any]], model_name: str) -> ModelInfo:
    """
    Extract model metadata from leaderboard row.
    """
    run_spec_names = next(
        (cell["run_spec_names"] for cell in row if "run_spec_names" in cell),
        None,
    )

    if '(' in model_name and ')' in model_name:
        model_name = clean_model_name_from_parenthesis(model_name)

    if not run_spec_names:
        developer = assign_developer_to_model_by_name(model_name)
        if developer == "unknown":
            model_id = model_name.replace(" ", "-")
        else:
            model_id = f"{developer}/{model_name.replace(" ", "-")}"
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
        developer = assign_developer_to_model_by_name(model_name)

    return ModelInfo(
        name=model_name,
        id=model_id,
        developer=developer,
        inference_platform="unknown",
    )


def find_column_ranges(tab_rows: List[List[Dict[str, Any]]]):
    """
    Determine min/max values for each metric column.
    """
    num_columns = len(tab_rows[0]) - 1
    mins = [0] * num_columns
    maxs = [0] * num_columns

    for row in tab_rows:
        for idx, cell in enumerate(row[1:], start=0):
            value = cell.get("value", 0)
            mins[idx] = min(mins[idx], value)
            maxs[idx] = max(maxs[idx], value)

    return mins, maxs


def convert(
    leaderboard_name: str,
    leaderboard_data: List[Dict[str, Any]],
    source_name: str,
    source_type: str,
    source_data: List[str],
):
    """
    Convert HELM leaderboard data into unified evaluation logs.
    """
    retrieved_timestamp = str(time.time())

    model_infos: Dict[str, ModelInfo] = {}
    model_results: Dict[str, Dict[str, EvaluationResult]] = defaultdict(dict)

    for tab in leaderboard_data:
        tab_name = tab.get("title")
        headers = tab.get("header")
        rows = tab.get("rows")

        mins, maxs = find_column_ranges(rows)

        for row in rows:
            model_name = row[0].get("value")

            if model_name not in model_infos:
                model_infos[model_name] = extract_model_info(row, model_name)

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
                            0 if mins[col_idx] >= 0 else math.floor(mins[col_idx])
                        ),
                        max_score=(
                            1 if maxs[col_idx] <= 1 else math.ceil(maxs[col_idx])
                        ),
                        score_type=ScoreType.continuous,
                    )

                    generation_config = (
                        extract_generation_config_from_run_specs(
                            cell.get("run_spec_names", [])
                        )
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


    for model_name, results_by_metric in model_results.items():
        model_info = model_infos[model_name]
        evaluation_id = (
            f"{leaderboard_name}/"
            f"{model_info.id.replace('/', '_')}/"
            f"{retrieved_timestamp}"
        )

        eval_log = EvaluationLog(
            schema_version="0.1.0",
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name=source_name,
                source_type=source_type,
                source_organization_name="crfm",
                evaluator_relationship=EvaluatorRelationship.third_party,
            ),
            model_info=model_info,
            source_data=source_data,
            evaluation_results=list(results_by_metric.values()),
        )

        log_filename = f"{uuid.uuid4()}.json"

        if model_info.developer == "unknown":
            output_dir = f"data/{leaderboard_name}/{model_info.id}"
        else:
            dev, model = model_info.id.split("/")
            output_dir = f"data/{leaderboard_name}/{dev}/{model}"

        os.makedirs(output_dir, exist_ok=True)
        save_to_file(eval_log, f"{output_dir}/{log_filename}")


if __name__ == "__main__":
    args = parse_args()

    leaderboard_name = args.leaderboard_name.lower()
    source_data = [args.source_data_url]

    os.makedirs(f"data/{leaderboard_name}", exist_ok=True)

    leaderboard_data = download_leaderboard(source_data[0])

    convert(
        leaderboard_name=leaderboard_name,
        leaderboard_data=leaderboard_data,
        source_name=leaderboard_name,
        source_type="documentation",
        source_data=source_data,
    )