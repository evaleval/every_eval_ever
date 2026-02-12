"""
Script to fetch Global MMLU Lite leaderboard results from Kaggle API
and convert them to the EvalEval schema format.

Data source:
- Global MMLU Lite: Kaggle Benchmarks API (cohere-labs/global-mmlu-lite)

Usage:
    uv run python -m automation_scripts.global-mmlu-lite.adapter
"""

import time
from typing import List

from eval_types import (
    ConfidenceInterval,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    Uncertainty,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    fetch_json,
    get_developer,
    make_source_metadata,
    make_model_info,
    save_evaluation_log,
)


# Data source URL
KAGGLE_API_URL = "https://www.kaggle.com/api/v1/benchmarks/cohere-labs/global-mmlu-lite/leaderboard"

OUTPUT_DIR = "data/global-mmlu-lite"

# Hardcoded source data for global-mmlu-lite
SOURCE_DATA = SourceDataUrl(
    dataset_name="global-mmlu-lite",
    source_type="url",
    url=["https://www.kaggle.com/datasets/cohere-labs/global-mmlu-lite"],
)


def parse_score(value) -> float:
    """Parse a score value, ensuring it's a float."""
    if value is None:
        return -1.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return -1.0


def make_eval_result(
    name: str,
    score: float,
    description: str,
    confidence_interval: float | None = None,
    stddev: float | None = None,
) -> EvaluationResult:
    """Create an EvaluationResult with hardcoded source_data for global-mmlu-lite."""
    uncertainty = None
    if confidence_interval is not None or stddev is not None:
        ci = None
        if confidence_interval is not None and score is not None and score >= 0:
            ci = ConfidenceInterval(
                lower=round(-confidence_interval, 4),
                upper=round(confidence_interval, 4),
                method="unknown",
            )
        uncertainty = Uncertainty(
            confidence_interval=ci,
            standard_deviation=stddev,
        )
    return EvaluationResult(
        evaluation_name=name,
        source_data=SOURCE_DATA,
        metric_config=MetricConfig(
            evaluation_description=description,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(
            score=round(score, 4) if score is not None else -1,
            uncertainty=uncertainty,
        ),
    )


def fetch_global_mmlu_lite(retrieved_timestamp: str) -> int:
    """Fetch and process Global MMLU Lite results from Kaggle API."""
    print("Fetching Global MMLU Lite leaderboard from Kaggle API...")

    try:
        data = fetch_json(KAGGLE_API_URL)
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

    # The API returns a dict with a 'rows' key containing the leaderboard entries
    rows = data.get("rows", [])
    count = 0

    for row in rows:
        # Extract model information - require slug
        model_slug = row.get("modelVersionSlug")
        if not model_slug:
            raise ValueError(f"Missing modelVersionSlug in row: {row}")

        model_name = model_slug
        model_display_name = row.get("modelVersionName", "")

        developer = get_developer(model_name)

        # Create evaluation results from task results
        eval_results: List[EvaluationResult] = []
        task_results = row.get("taskResults", [])

        for task in task_results:
            task_name = task.get("benchmarkTaskName", "")
            result_data = task.get("result", {})

            # Extract score from the result
            score_value = None
            confidence_interval = None

            if result_data.get("hasNumericResult"):
                numeric_result = result_data.get("numericResult") or result_data.get("numericResultNullable", {})
                score_value = numeric_result.get("value")

                if numeric_result.get("hasConfidenceInterval"):
                    confidence_interval = numeric_result.get("confidenceInterval")

            if score_value is not None:
                score = parse_score(score_value)
                if score >= 0:
                    eval_results.append(
                        make_eval_result(
                            name=task_name,
                            score=score,
                            description=f"Global MMLU Lite - {task_name}",
                            confidence_interval=confidence_interval,
                        )
                    )

        if not eval_results:
            continue

        # Build model info
        model_info = make_model_info(
            model_name=model_name,
            developer=developer,
            additional_details={"display_name": model_display_name} if model_display_name and model_display_name != model_name else None,
        )

        # Build evaluation log
        evaluation_id = f"global-mmlu-lite/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
        eval_log = EvaluationLog(
            schema_version="0.2.0",
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=make_source_metadata(
                source_name="Global MMLU Lite Leaderboard",
                organization_name="kaggle",
                organization_url="www.kaggle.com",
                evaluator_relationship=EvaluatorRelationship.third_party,
            ),
            model_info=model_info,
            evaluation_results=eval_results,
        )

        # Parse model path for saving: use slug for folder name (no spaces, like hfopenllm_v2)
        if "/" in model_info.id:
            dev, _ = model_info.id.split("/", 1)
        else:
            dev, _ = "unknown", model_info.id
        model_for_path = model_slug if model_slug else model_info.id.split("/")[-1]

        filepath = save_evaluation_log(eval_log, OUTPUT_DIR, dev, model_for_path)
        print(f"Saved: {filepath}")
        count += 1

    return count


def main():
    """Main function to fetch and process Global MMLU Lite results."""
    retrieved_timestamp = str(time.time())

    print("=" * 60)
    print("Fetching Global MMLU Lite results...")
    print("=" * 60)

    try:
        count = fetch_global_mmlu_lite(retrieved_timestamp)
        print(f"\nProcessed {count} models from Global MMLU Lite")
    except Exception as e:
        print(f"Error processing Global MMLU Lite: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
