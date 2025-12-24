"""
Script to fetch RewardBench and RewardBench 2 leaderboard results
from HuggingFace and convert them to the EvalEval schema format.

Data sources:
- RewardBench v1: CSV from HuggingFace Space (leaderboard/final-rbv1-data.csv)
- RewardBench v2: JSON files from allenai/reward-bench-2-results dataset (eval-set/{org}/{model}.json)

Usage:
    uv run python -m scripts.rewardbench.fetch_rewardbench
"""

import csv
import io
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

from eval_types import (
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceMetadata,
)


# Data source URLs
REWARDBENCH_V1_CSV = "https://huggingface.co/spaces/allenai/reward-bench/resolve/main/leaderboard/final-rbv1-data.csv"
REWARDBENCH_V2_TREE_API = "https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/eval-set"
REWARDBENCH_V2_FILE_BASE = "https://huggingface.co/datasets/allenai/reward-bench-2-results/resolve/main/eval-set"

OUTPUT_DIR = "data/reward-bench"


def extract_model_name_from_html(html_string: str) -> str:
    """Extract the model name from an HTML anchor tag."""
    # Pattern to match content between > and <
    pattern = r">([^<]+)<"
    match = re.search(pattern, html_string)
    if match:
        # Remove any trailing markers like " *" or " ⚠️"
        name = match.group(1).strip()
        name = re.sub(r"\s*[\*⚠️]+$", "", name).strip()
        return name
    # If no HTML, return as-is (after cleaning)
    return re.sub(r"\s*[\*⚠️]+$", "", html_string).strip()


def extract_model_info(model_name: str, model_type: Optional[str] = None) -> ModelInfo:
    """Extract model information from a model name string."""
    # Clean up the model name
    clean_name = model_name.strip()
    
    # Try to extract developer from model name
    if "/" in clean_name:
        developer = clean_name.split("/")[0]
        model_id = clean_name
    else:
        developer = "unknown"
        model_id = clean_name

    additional = {}
    if model_type:
        additional["model_type"] = model_type

    return ModelInfo(
        name=clean_name,
        id=model_id,
        developer=developer,
        additional_details=additional if additional else None,
    )


def save_evaluation_log(
    eval_log: EvaluationLog,
    output_dir: str,
    model_info: ModelInfo,
) -> str:
    """Save an evaluation log to the appropriate directory."""
    # Create directory structure: output_dir/developer/model_name/
    if "/" in model_info.id:
        parts = model_info.id.split("/", 1)
        developer = parts[0]
        model_name = parts[1] if len(parts) > 1 else parts[0]
    else:
        developer = "unknown"
        model_name = model_info.id

    # Clean up names for filesystem
    developer = re.sub(r'[<>:"/\\|?*]', "_", developer)
    model_name = re.sub(r'[<>:"/\\|?*]', "_", model_name)

    dir_path = os.path.join(output_dir, developer, model_name)
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{uuid.uuid4()}.json"
    filepath = os.path.join(dir_path, filename)

    json_str = eval_log.model_dump_json(indent=2, exclude_none=True)
    with open(filepath, "w") as f:
        f.write(json_str)

    return filepath


def fetch_rewardbench_v1(retrieved_timestamp: str) -> int:
    """Fetch and process RewardBench v1 results from the CSV file."""
    print("Fetching RewardBench v1 CSV...")
    
    response = requests.get(REWARDBENCH_V1_CSV, timeout=60, allow_redirects=True)
    if response.status_code != 200:
        print(f"Error fetching RewardBench v1 CSV: {response.status_code}")
        return 0
    
    # Parse CSV
    csv_content = response.text
    reader = csv.DictReader(io.StringIO(csv_content))
    
    count = 0
    for row in reader:
        # Extract model name from HTML link
        model_html = row.get("Model", "")
        model_name = extract_model_name_from_html(model_html)
        if not model_name or model_name == "random":
            continue
            
        model_type = row.get("Model Type", "")
        model_info = extract_model_info(model_name, model_type)
        
        # Create evaluation results for each metric
        eval_results = []
        
        # Define v1 metrics with descriptions
        v1_metrics = {
            "Score": ("Overall RewardBench Score", False),
            "Chat": ("Chat accuracy - includes easy chat subsets", False),
            "Chat Hard": ("Chat Hard accuracy - includes hard chat subsets", False),
            "Safety": ("Safety accuracy - includes safety subsets", False),
            "Reasoning": ("Reasoning accuracy - includes code and math subsets", False),
            "Prior Sets (0.5 weight)": ("Prior Sets score (weighted 0.5) - includes test sets", False),
        }
        
        for metric_name, (description, lower_is_better) in v1_metrics.items():
            value = row.get(metric_name, "")
            if value and value.strip():
                try:
                    score = float(value)
                    # RewardBench v1 scores are typically 0-100, normalize to 0-1
                    if score > 1:
                        score = score / 100.0
                    
                    eval_results.append(
                        EvaluationResult(
                            evaluation_name=metric_name,
                            metric_config=MetricConfig(
                                evaluation_description=description,
                                lower_is_better=lower_is_better,
                                score_type=ScoreType.continuous,
                                min_score=0.0,
                                max_score=1.0,
                            ),
                            score_details=ScoreDetails(score=round(score, 4)),
                        )
                    )
                except ValueError:
                    pass
        
        if not eval_results:
            continue
            
        evaluation_id = f"reward-bench/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
        
        eval_log = EvaluationLog(
            schema_version="0.1.0",
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_data=["https://huggingface.co/spaces/allenai/reward-bench"],
            source_metadata=SourceMetadata(
                source_name="RewardBench",
                source_type="documentation",
                source_organization_name="Allen Institute for AI",
                source_organization_url="https://allenai.org",
                evaluator_relationship=EvaluatorRelationship.third_party,
            ),
            model_info=model_info,
            evaluation_results=eval_results,
        )
        
        filepath = save_evaluation_log(eval_log, OUTPUT_DIR, model_info)
        print(f"Saved: {filepath}")
        count += 1
    
    return count


def fetch_rewardbench_v2(retrieved_timestamp: str) -> int:
    """Fetch and process RewardBench v2 results from the HuggingFace dataset."""
    print("Fetching RewardBench v2 model list...")
    
    # First, get the list of organizations
    response = requests.get(REWARDBENCH_V2_TREE_API, timeout=60)
    if response.status_code != 200:
        print(f"Error fetching RewardBench v2 tree: {response.status_code}")
        return 0
    
    orgs = response.json()
    count = 0
    
    for org_item in orgs:
        if org_item["type"] != "directory":
            continue
            
        org_path = org_item["path"]
        org_name = org_path.split("/")[-1]
        
        print(f"  Processing organization: {org_name}")
        
        # Get models for this org
        org_tree_url = f"https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/{org_path}"
        org_response = requests.get(org_tree_url, timeout=60)
        if org_response.status_code != 200:
            print(f"    Error fetching org tree: {org_response.status_code}")
            continue
            
        model_files = org_response.json()
        
        for model_file in model_files:
            if model_file["type"] != "file" or not model_file["path"].endswith(".json"):
                continue
                
            model_path = model_file["path"]
            model_url = f"{REWARDBENCH_V2_FILE_BASE}/{'/'.join(model_path.split('/')[1:])}"
            
            # Fetch model JSON
            model_response = requests.get(model_url, timeout=60, allow_redirects=True)
            if model_response.status_code != 200:
                print(f"    Error fetching {model_path}: {model_response.status_code}")
                continue
                
            try:
                model_data = model_response.json()
            except Exception as e:
                print(f"    Error parsing JSON for {model_path}: {e}")
                continue
            
            model_name = model_data.get("model", "unknown")
            model_type = model_data.get("model_type", "")
            model_info = extract_model_info(model_name, model_type)
            
            # Define v2 metrics with descriptions (order matters for average calculation)
            v2_metrics = [
                ("Factuality", "Factuality score - measures factual accuracy"),
                ("Precise IF", "Precise Instruction Following score"),
                ("Math", "Math score - measures mathematical reasoning"),
                ("Safety", "Safety score - measures safety awareness"),
                ("Focus", "Focus score - measures response focus"),
                ("Ties", "Ties score - ability to identify tie cases"),
            ]
            
            eval_results = []
            scores_for_average = []
            
            for metric_name, description in v2_metrics:
                if metric_name in model_data and model_data[metric_name] is not None:
                    try:
                        score = float(model_data[metric_name])
                        scores_for_average.append(score)
                        eval_results.append(
                            EvaluationResult(
                                evaluation_name=metric_name,
                                metric_config=MetricConfig(
                                    evaluation_description=description,
                                    lower_is_better=False,
                                    score_type=ScoreType.continuous,
                                    min_score=0.0,
                                    max_score=1.0,
                                ),
                                score_details=ScoreDetails(score=round(score, 4)),
                            )
                        )
                    except (ValueError, TypeError):
                        pass
            
            if not eval_results:
                continue
            
            # Calculate and add mean score as the first result (like RewardBench v1's "Score")
            if scores_for_average:
                mean_score = sum(scores_for_average) / len(scores_for_average)
                eval_results.insert(
                    0,
                    EvaluationResult(
                        evaluation_name="Score",
                        metric_config=MetricConfig(
                            evaluation_description="Overall RewardBench 2 Score (mean of all metrics)",
                            lower_is_better=False,
                            score_type=ScoreType.continuous,
                            min_score=0.0,
                            max_score=1.0,
                        ),
                        score_details=ScoreDetails(score=round(mean_score, 4)),
                    )
                )
            
            evaluation_id = f"reward-bench-2/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
            
            eval_log = EvaluationLog(
                schema_version="0.1.0",
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_data=["https://huggingface.co/datasets/allenai/reward-bench-2-results"],
                source_metadata=SourceMetadata(
                    source_name="RewardBench 2",
                    source_type="documentation",
                    source_organization_name="Allen Institute for AI",
                    source_organization_url="https://allenai.org",
                    evaluator_relationship=EvaluatorRelationship.third_party,
                ),
                model_info=model_info,
                evaluation_results=eval_results,
            )
            
            filepath = save_evaluation_log(eval_log, OUTPUT_DIR, model_info)
            print(f"    Saved: {filepath}")
            count += 1
    
    return count


def main():
    """Main function to fetch and process RewardBench results."""
    retrieved_timestamp = str(time.time())

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Fetching RewardBench v1 results...")
    print("=" * 60)

    try:
        v1_count = fetch_rewardbench_v1(retrieved_timestamp)
        print(f"\nProcessed {v1_count} models from RewardBench v1")
    except Exception as e:
        print(f"Error processing RewardBench v1: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Fetching RewardBench v2 results...")
    print("=" * 60)

    try:
        v2_count = fetch_rewardbench_v2(retrieved_timestamp)
        print(f"\nProcessed {v2_count} models from RewardBench v2")
    except Exception as e:
        print(f"Error processing RewardBench v2: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
