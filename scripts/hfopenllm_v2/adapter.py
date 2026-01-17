"""
Script to convert HuggingFace Open LLM Leaderboard v2 data to the EvalEval schema format.

Data source:
- HF Open LLM Leaderboard v2 API: https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted

Usage:
    uv run python -m scripts.hfopenllm_v2.adapter
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from eval_types import EvaluationLog, EvaluationResult, EvaluatorRelationship

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    get_developer,
    make_evaluation_result,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)


# Source URL
SOURCE_URL = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
OUTPUT_DIR = "data/hfopenllm_v2"

# Evaluation name mapping from API keys to display names
EVALUATION_MAPPING = {
    "ifeval": "IFEval",
    "bbh": "BBH",
    "math": "MATH Level 5",
    "gpqa": "GPQA",
    "musr": "MUSR",
    "mmlu_pro": "MMLU-PRO",
}

# Evaluation descriptions
EVALUATION_DESCRIPTIONS = {
    "IFEval": "Accuracy on IFEval",
    "BBH": "Accuracy on BBH",
    "MATH Level 5": "Exact Match on MATH Level 5",
    "GPQA": "Accuracy on GPQA",
    "MUSR": "Accuracy on MUSR",
    "MMLU-PRO": "Accuracy on MMLU-PRO",
}


def convert_model(model_data: Dict[str, Any], retrieved_timestamp: str) -> EvaluationLog:
    """Convert a single model's data to EvaluationLog format."""
    model_name = model_data["model"]["name"]
    developer = get_developer(model_name)

    # Build evaluation results
    eval_results: List[EvaluationResult] = []
    for eval_key, eval_data in model_data.get("evaluations", {}).items():
        display_name = eval_data.get("name", EVALUATION_MAPPING.get(eval_key, eval_key))
        description = EVALUATION_DESCRIPTIONS.get(display_name, f"Accuracy on {display_name}")

        eval_results.append(
            make_evaluation_result(
                name=display_name,
                score=eval_data.get("value", 0.0),
                description=description,
            )
        )

    # Build additional details
    additional_details = {}
    if "precision" in model_data["model"]:
        additional_details["precision"] = model_data["model"]["precision"]
    if "architecture" in model_data["model"]:
        additional_details["architecture"] = model_data["model"]["architecture"]
    if "params_billions" in model_data.get("metadata", {}):
        additional_details["params_billions"] = model_data["metadata"]["params_billions"]

    # Build model info
    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        inference_platform="unknown",
        additional_details=additional_details if additional_details else None,
    )

    # Build evaluation ID
    evaluation_id = f"hfopenllm_v2/{model_name.replace('/', '_')}/{retrieved_timestamp}"

    return EvaluationLog(
        schema_version="0.1.0",
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_data=[SOURCE_URL],
        source_metadata=make_source_metadata(
            source_name="HF Open LLM v2",
            organization_name="Hugging Face",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        model_info=model_info,
        evaluation_results=eval_results,
    )


def process_models(models_data: List[Dict[str, Any]], output_dir: str = OUTPUT_DIR):
    """Process a list of model evaluation dicts and save them."""
    retrieved_timestamp = str(time.time())
    count = 0

    for model_data in models_data:
        try:
            model_name = model_data["model"]["name"]
            developer = get_developer(model_name)

            # Parse model name for directory structure
            if "/" in model_name:
                _, model = model_name.split("/", 1)
            else:
                model = model_name

            # Convert to EvaluationLog
            eval_log = convert_model(model_data, retrieved_timestamp)

            # Save
            filepath = save_evaluation_log(eval_log, output_dir, developer, model)
            print(f"Saved: {filepath}")
            count += 1

        except Exception as e:
            model_name = model_data.get("model", {}).get("name", "unknown")
            print(f"Error processing {model_name}: {e}")

    return count


if __name__ == "__main__":
    # Load data from local file (downloaded separately)
    data_path = Path(__file__).parent / "formatted.json"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print(f"Please download the data from: {SOURCE_URL}")
        print(f"Save it as: {data_path}")
        exit(1)

    print(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        all_models = json.load(f)

    print(f"Processing {len(all_models)} models...")
    count = process_models(all_models)
    print(f"Done! Processed {count} models.")
