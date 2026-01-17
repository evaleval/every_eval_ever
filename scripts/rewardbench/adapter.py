"""
Script to fetch RewardBench and RewardBench 2 leaderboard results
from HuggingFace and convert them to the EvalEval schema format.

Data sources:
- RewardBench v1: CSV from HuggingFace Space (leaderboard/final-rbv1-data.csv)
- RewardBench v2: JSON files from allenai/reward-bench-2-results dataset (eval-set/{org}/{model}.json)

Usage:
    uv run python -m scripts.rewardbench.adapter
"""

import re
import time
from typing import List, Optional

from eval_types import EvaluationLog, EvaluationResult, EvaluatorRelationship

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    fetch_csv,
    fetch_json,
    get_developer,
    make_evaluation_result,
    make_source_metadata,
    make_model_info,
    save_evaluation_log,
)


# Data source URLs
REWARDBENCH_V1_CSV = "https://huggingface.co/spaces/allenai/reward-bench/resolve/main/leaderboard/final-rbv1-data.csv"
REWARDBENCH_V2_TREE_API = "https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/eval-set"
REWARDBENCH_V2_FILE_BASE = "https://huggingface.co/datasets/allenai/reward-bench-2-results/resolve/main/eval-set"

OUTPUT_DIR = "data/reward-bench"

# RewardBench v1 metrics with descriptions
V1_METRICS = {
    "Score": "Overall RewardBench Score",
    "Chat": "Chat accuracy - includes easy chat subsets",
    "Chat Hard": "Chat Hard accuracy - includes hard chat subsets",
    "Safety": "Safety accuracy - includes safety subsets",
    "Reasoning": "Reasoning accuracy - includes code and math subsets",
    "Prior Sets (0.5 weight)": "Prior Sets score (weighted 0.5) - includes test sets",
}

# RewardBench v2 metrics with descriptions
V2_METRICS = [
    ("Factuality", "Factuality score - measures factual accuracy"),
    ("Precise IF", "Precise Instruction Following score"),
    ("Math", "Math score - measures mathematical reasoning"),
    ("Safety", "Safety score - measures safety awareness"),
    ("Focus", "Focus score - measures response focus"),
    ("Ties", "Ties score - ability to identify tie cases"),
]


def extract_model_name_from_html(html_string: str) -> str:
    """Extract the model name from an HTML anchor tag."""
    pattern = r">([^<]+)<"
    match = re.search(pattern, html_string)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"\s*[\*⚠️]+$", "", name).strip()
        return name
    return re.sub(r"\s*[\*⚠️]+$", "", html_string).strip()


def parse_score(value: str) -> Optional[float]:
    """Parse a score string, normalizing 0-100 scores to 0-1."""
    if not value or not value.strip():
        return None
    try:
        score = float(value)
        # RewardBench v1 scores are typically 0-100, normalize to 0-1
        if score > 1:
            score = score / 100.0
        return score
    except ValueError:
        return None


def fetch_rewardbench_v1(retrieved_timestamp: str) -> int:
    """Fetch and process RewardBench v1 results from the CSV file."""
    print("Fetching RewardBench v1 CSV...")

    rows = fetch_csv(REWARDBENCH_V1_CSV)
    count = 0

    for row in rows:
        # Extract model name from HTML link
        model_html = row.get("Model", "")
        model_name = extract_model_name_from_html(model_html)
        if not model_name or model_name == "random":
            continue

        model_type = row.get("Model Type", "")
        developer = get_developer(model_name)

        # Create evaluation results for each metric
        eval_results: List[EvaluationResult] = []
        for metric_name, description in V1_METRICS.items():
            score = parse_score(row.get(metric_name, ""))
            if score is not None:
                eval_results.append(
                    make_evaluation_result(
                        name=metric_name,
                        score=score,
                        description=description,
                    )
                )

        if not eval_results:
            continue

        # Build model info
        model_info = make_model_info(
            model_name=model_name,
            developer=developer,
            additional_details={"model_type": model_type} if model_type else None,
        )

        # Build evaluation log
        evaluation_id = f"reward-bench/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
        eval_log = EvaluationLog(
            schema_version="0.1.0",
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_data=["https://huggingface.co/spaces/allenai/reward-bench"],
            source_metadata=make_source_metadata(
                source_name="RewardBench",
                organization_name="Allen Institute for AI",
                organization_url="https://allenai.org",
                evaluator_relationship=EvaluatorRelationship.third_party,
            ),
            model_info=model_info,
            evaluation_results=eval_results,
        )

        # Parse model path for saving
        if "/" in model_info.id:
            dev, model = model_info.id.split("/", 1)
        else:
            dev, model = "unknown", model_info.id

        filepath = save_evaluation_log(eval_log, OUTPUT_DIR, dev, model)
        print(f"Saved: {filepath}")
        count += 1

    return count


def fetch_rewardbench_v2(retrieved_timestamp: str) -> int:
    """Fetch and process RewardBench v2 results from the HuggingFace dataset."""
    print("Fetching RewardBench v2 model list...")

    orgs = fetch_json(REWARDBENCH_V2_TREE_API)
    count = 0

    for org_item in orgs:
        if org_item["type"] != "directory":
            continue

        org_path = org_item["path"]
        org_name = org_path.split("/")[-1]
        print(f"  Processing organization: {org_name}")

        # Get models for this org
        org_tree_url = f"https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/{org_path}"
        try:
            model_files = fetch_json(org_tree_url)
        except Exception as e:
            print(f"    Error fetching org tree: {e}")
            continue

        for model_file in model_files:
            if model_file["type"] != "file" or not model_file["path"].endswith(".json"):
                continue

            model_path = model_file["path"]
            model_url = f"{REWARDBENCH_V2_FILE_BASE}/{'/'.join(model_path.split('/')[1:])}"

            try:
                model_data = fetch_json(model_url)
            except Exception as e:
                print(f"    Error fetching {model_path}: {e}")
                continue

            model_name = model_data.get("model", "unknown")
            model_type = model_data.get("model_type", "")
            developer = get_developer(model_name)

            # Build evaluation results
            eval_results: List[EvaluationResult] = []
            scores_for_average = []

            for metric_name, description in V2_METRICS:
                if metric_name in model_data and model_data[metric_name] is not None:
                    try:
                        score = float(model_data[metric_name])
                        scores_for_average.append(score)
                        eval_results.append(
                            make_evaluation_result(
                                name=metric_name,
                                score=score,
                                description=description,
                            )
                        )
                    except (ValueError, TypeError):
                        pass

            if not eval_results:
                continue

            # Add mean score as the first result
            if scores_for_average:
                mean_score = sum(scores_for_average) / len(scores_for_average)
                eval_results.insert(
                    0,
                    make_evaluation_result(
                        name="Score",
                        score=mean_score,
                        description="Overall RewardBench 2 Score (mean of all metrics)",
                    ),
                )

            # Build model info
            model_info = make_model_info(
                model_name=model_name,
                developer=developer,
                additional_details={"model_type": model_type} if model_type else None,
            )

            # Build evaluation log
            evaluation_id = f"reward-bench-2/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
            eval_log = EvaluationLog(
                schema_version="0.1.0",
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_data=["https://huggingface.co/datasets/allenai/reward-bench-2-results"],
                source_metadata=make_source_metadata(
                    source_name="RewardBench 2",
                    organization_name="Allen Institute for AI",
                    organization_url="https://allenai.org",
                    evaluator_relationship=EvaluatorRelationship.third_party,
                ),
                model_info=model_info,
                evaluation_results=eval_results,
            )

            # Parse model path for saving
            if "/" in model_info.id:
                dev, model = model_info.id.split("/", 1)
            else:
                dev, model = "unknown", model_info.id

            filepath = save_evaluation_log(eval_log, OUTPUT_DIR, dev, model)
            print(f"    Saved: {filepath}")
            count += 1

    return count


def main():
    """Main function to fetch and process RewardBench results."""
    retrieved_timestamp = str(time.time())

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
