"""
Script to fetch Global MMLU Lite leaderboard results from Kaggle API
and convert them to the EvalEval schema format.

Data source:
- Global MMLU Lite: Kaggle Benchmarks API (cohere-labs/global-mmlu-lite)

Usage:
    uv run python -m utils.global-mmlu-lite.adapter
"""

import time

from every_eval_ever.eval_types import (
    ConfidenceInterval,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    get_developer,
    make_model_info,
    make_source_metadata,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

# Data source URL
KAGGLE_API_URL = (
    "https://www.kaggle.com/api/v1/benchmarks/cohere-labs/global-mmlu-lite/leaderboard"
)

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
        raise ValueError("score is missing")
    try:
        score = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid score: {value!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1, got {score!r}")
    return score


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


def convert_rows(
    rows: list[dict],
    retrieved_timestamp: str,
    output_dir: str = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert valid rows and metrics while retaining rejected provenance."""
    outputs = []
    failures: list[SourceRecordFailure] = []

    for row_index, row in enumerate(rows):
        row_ref = f"leaderboard row {row_index}"
        failure_count_before = len(failures)
        try:
            model_slug = require_identity(
                row.get("modelVersionSlug"),
                "Global MMLU Lite model version slug",
            )
            model_display_name = row.get("modelVersionName", "")
            developer = require_identity(
                get_developer(model_slug),
                "Global MMLU Lite model developer",
            )

            eval_results: list[EvaluationResult] = []
            task_results = row.get("taskResults")
            if not isinstance(task_results, list):
                raise ValueError("taskResults must be a list")

            for task_index, task in enumerate(task_results):
                task_ref = f"{row_ref} task {task_index}"
                try:
                    task_name = require_identity(
                        task.get("benchmarkTaskName"),
                        "Global MMLU Lite task name",
                    )
                    result_data = task.get("result")
                    if not isinstance(result_data, dict):
                        raise ValueError("task result must be an object")
                    if not result_data.get("hasNumericResult"):
                        raise ValueError("task has no numeric result")
                    numeric_result = result_data.get(
                        "numericResult"
                    ) or result_data.get("numericResultNullable")
                    if not isinstance(numeric_result, dict):
                        raise ValueError(
                            "numeric result payload is missing"
                        )
                    score = parse_score(numeric_result.get("value"))
                    confidence_interval = (
                        numeric_result.get("confidenceInterval")
                        if numeric_result.get("hasConfidenceInterval")
                        else None
                    )
                    eval_results.append(
                        make_eval_result(
                            name=task_name,
                            score=score,
                            description=f"Global MMLU Lite - {task_name}",
                            confidence_interval=confidence_interval,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=task_ref,
                            reason=str(exc),
                            source_record=task,
                        )
                    )

            if not eval_results:
                raise ValueError("model has no usable task results")

            model_info = make_model_info(
                model_name=model_slug,
                developer=developer,
                additional_details={"display_name": model_display_name}
                if model_display_name and model_display_name != model_slug
                else None,
            )
            model_id = require_identity(
                model_info.id,
                "Global MMLU Lite model id",
            )
            if "/" not in model_id:
                raise ValueError(
                    f"model id must be developer/model: {model_id!r}"
                )
            dev, model_for_path = model_id.split("/", 1)

            evaluation_id = (
                "global-mmlu-lite/"
                f"{model_id.replace('/', '_')}/{retrieved_timestamp}"
            )
            eval_log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_metadata=make_source_metadata(
                    source_name="Global MMLU Lite Leaderboard",
                    organization_name="kaggle",
                    organization_url="https://www.kaggle.com",
                    evaluator_relationship=EvaluatorRelationship.third_party,
                ),
                eval_library=EvalLibrary(
                    name="kaggle kernel",
                    version="4",
                    additional_details={
                        "url": "https://www.kaggle.com/code/shivalikasingh95/global-mmlu-lite-sample-notebook"
                    },
                ),
                model_info=model_info,
                evaluation_results=eval_results,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=output_dir,
                    developer=dev,
                    model_name=model_for_path,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=(
                        f"no output written: {exc}"
                        if len(failures) > failure_count_before
                        else str(exc)
                    ),
                    source_record=row,
                )
            )

    return SourceConversionResult(
        source_name="Global MMLU Lite",
        total_records=len(rows),
        records=outputs,
        failures=failures,
    )


def fetch_global_mmlu_lite(retrieved_timestamp: str) -> int:
    """Fetch, convert, and publish Global MMLU Lite results."""
    print("Fetching Global MMLU Lite leaderboard from Kaggle API...")
    data = fetch_json(KAGGLE_API_URL)
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Kaggle response contains no leaderboard rows")

    result = convert_rows(rows, retrieved_timestamp)
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(f"Saved: {path}")
    if result.failures:
        report_path = save_failure_report(
            result,
            default_failure_report_path(OUTPUT_DIR),
        )
        print(f"Failure report: {report_path}")
        result.raise_if_incomplete()
    return len(paths)


def main():
    """Main function to fetch and process Global MMLU Lite results."""
    retrieved_timestamp = str(time.time())

    print("=" * 60)
    print("Fetching Global MMLU Lite results...")
    print("=" * 60)

    count = fetch_global_mmlu_lite(retrieved_timestamp)
    print(f"\nProcessed {count} models from Global MMLU Lite")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
