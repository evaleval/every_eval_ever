"""
Script to fetch AlpacaEval 1.0 and 2.0 leaderboard results from the
tatsu-lab/alpaca_eval GitHub repository and convert them to the EvalEval schema.

Data sources:
- AlpacaEval 1.0: GPT-4 judge, ~102 models
- AlpacaEval 2.0: weighted GPT-4 Turbo judge, ~222 models

Usage:
    uv run python -m utils.alpaca_eval.adapter
"""

import time
from typing import List, Optional

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
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    fetch_csv,
    get_developer,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)

# ---------------------------------------------------------------------------
# AlpacaEval 1.0 — GPT-4 judge
# ---------------------------------------------------------------------------
ALPACA_EVAL_1_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main"
    "/src/alpaca_eval/leaderboards/data_AlpacaEval"
    "/alpaca_eval_gpt4_leaderboard.csv"
)
OUTPUT_DIR_V1 = "data/alpaca_eval"

SOURCE_DATA_V1 = SourceDataUrl(
    dataset_name="alpaca_eval",
    source_type="url",
    url=["https://github.com/tatsu-lab/alpaca_eval"],
)

# ---------------------------------------------------------------------------
# AlpacaEval 2.0 — weighted GPT-4 Turbo judge
# ---------------------------------------------------------------------------
ALPACA_EVAL_2_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main"
    "/src/alpaca_eval/leaderboards/data_AlpacaEval_2"
    "/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv"
)
OUTPUT_DIR_V2 = "data/alpaca_eval_2"

SOURCE_DATA_V2 = SourceDataUrl(
    dataset_name="alpaca_eval_2",
    source_type="url",
    url=["https://github.com/tatsu-lab/alpaca_eval"],
)

ALPACA_EVAL_LIBRARY = EvalLibrary(
    name="alpaca_eval",
    version="0.6",
    additional_details={"url": "https://github.com/tatsu-lab/alpaca_eval"},
)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if not value or not value.strip():
        return None
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


def _make_uncertainty(se_value: Optional[float]) -> Optional[Uncertainty]:
    if se_value is None:
        return None
    return Uncertainty(
        standard_error=StandardError(value=se_value, method="analytic"),
    )


def _win_rate_result(
    evaluation_name: str,
    description: str,
    score: float,
    se: Optional[float],
    source_data: SourceDataUrl,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_name=evaluation_name,
        source_data=source_data,
        metric_config=MetricConfig(
            evaluation_description=description,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
        ),
        score_details=ScoreDetails(
            score=round(score, 4),
            uncertainty=_make_uncertainty(se),
        ),
    )


def _length_result(score: float, source_data: SourceDataUrl) -> EvaluationResult:
    return EvaluationResult(
        evaluation_name="avg_length",
        source_data=source_data,
        metric_config=MetricConfig(
            evaluation_description="Average response length in tokens",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100000.0,
        ),
        score_details=ScoreDetails(score=round(score, 4)),
    )


def _build_v1_results(row: dict) -> List[EvaluationResult]:
    results = []

    win_rate = _parse_float(row.get("win_rate"))
    if win_rate is not None:
        results.append(
            _win_rate_result(
                "win_rate",
                "AlpacaEval 1.0 win rate against text-davinci-003 (GPT-4 judge)",
                win_rate,
                _parse_float(row.get("standard_error")),
                SOURCE_DATA_V1,
            )
        )

    lc_wr = _parse_float(row.get("length_controlled_winrate"))
    if lc_wr is not None:
        results.append(
            _win_rate_result(
                "length_controlled_winrate",
                "AlpacaEval 1.0 length-controlled win rate (GPT-4 judge)",
                lc_wr,
                None,
                SOURCE_DATA_V1,
            )
        )

    dwr = _parse_float(row.get("discrete_win_rate"))
    if dwr is not None:
        results.append(
            _win_rate_result(
                "discrete_win_rate",
                "AlpacaEval 1.0 discrete win rate (GPT-4 judge)",
                dwr,
                None,
                SOURCE_DATA_V1,
            )
        )

    avg_len = _parse_float(row.get("avg_length"))
    if avg_len is not None:
        results.append(_length_result(avg_len, SOURCE_DATA_V1))

    return results


def _build_v2_results(row: dict) -> List[EvaluationResult]:
    results = []

    win_rate = _parse_float(row.get("win_rate"))
    if win_rate is not None:
        results.append(
            _win_rate_result(
                "win_rate",
                "AlpacaEval 2.0 win rate vs GPT-4 Preview (weighted GPT-4 Turbo judge)",
                win_rate,
                _parse_float(row.get("standard_error")),
                SOURCE_DATA_V2,
            )
        )

    lc_wr = _parse_float(row.get("length_controlled_winrate"))
    if lc_wr is not None:
        results.append(
            _win_rate_result(
                "length_controlled_winrate",
                "AlpacaEval 2.0 length-controlled win rate (weighted GPT-4 Turbo judge)",
                lc_wr,
                _parse_float(row.get("lc_standard_error")),
                SOURCE_DATA_V2,
            )
        )

    dwr = _parse_float(row.get("discrete_win_rate"))
    if dwr is not None:
        results.append(
            _win_rate_result(
                "discrete_win_rate",
                "AlpacaEval 2.0 discrete win rate (weighted GPT-4 Turbo judge)",
                dwr,
                None,
                SOURCE_DATA_V2,
            )
        )

    avg_len = _parse_float(row.get("avg_length"))
    if avg_len is not None:
        results.append(_length_result(avg_len, SOURCE_DATA_V2))

    return results


def _make_log(
    evaluation_id_prefix: str,
    model_name: str,
    row: dict,
    eval_results: List[EvaluationResult],
    retrieved_timestamp: str,
    source_metadata,
) -> EvaluationLog:
    mode = (row.get("mode") or "").strip()
    developer = get_developer(model_name)
    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        additional_details={"submission_mode": mode} if mode else None,
    )
    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f"{evaluation_id_prefix}/{model_info.id.replace('/', '_')}/{retrieved_timestamp}",
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=source_metadata,
        eval_library=ALPACA_EVAL_LIBRARY,
        model_info=model_info,
        evaluation_results=eval_results,
    )


def fetch_alpaca_eval_1(retrieved_timestamp: str) -> int:
    print("Fetching AlpacaEval 1.0 leaderboard...")
    rows = fetch_csv(ALPACA_EVAL_1_URL)
    source_metadata = make_source_metadata(
        source_name="AlpacaEval 1.0 Leaderboard",
        organization_name="tatsu-lab",
        organization_url="https://github.com/tatsu-lab/alpaca_eval",
        evaluator_relationship=EvaluatorRelationship.third_party,
    )
    count = 0
    for row in rows:
        # Both AlpacaEval CSVs use pandas index (empty-string key) for model name.
        first_key = next(iter(row))
        model_name = (row.get(first_key) or "").strip() if first_key == "" else (row.get("model") or "").strip()
        if not model_name:
            continue
        eval_results = _build_v1_results(row)
        if not eval_results:
            continue
        log = _make_log("alpaca_eval", model_name, row, eval_results, retrieved_timestamp, source_metadata)
        developer = log.model_info.developer or "unknown"
        model_slug = log.model_info.name.replace("/", "_")
        filepath = save_evaluation_log(log, OUTPUT_DIR_V1, developer, model_slug)
        print(f"Saved: {filepath}")
        count += 1
    return count


def fetch_alpaca_eval_2(retrieved_timestamp: str) -> int:
    print("Fetching AlpacaEval 2.0 leaderboard...")
    rows = fetch_csv(ALPACA_EVAL_2_URL)
    source_metadata = make_source_metadata(
        source_name="AlpacaEval 2.0 Leaderboard",
        organization_name="tatsu-lab",
        organization_url="https://github.com/tatsu-lab/alpaca_eval",
        evaluator_relationship=EvaluatorRelationship.third_party,
    )
    # The AlpacaEval 2.0 CSV uses the model name as the pandas index (first column,
    # no header). csv.DictReader assigns the empty string "" as the key for that column.
    count = 0
    for row in rows:
        # First key holds the model name (index column with no header).
        first_key = next(iter(row))
        model_name = row[first_key].strip() if first_key == "" else (row.get("model") or row.get("") or "").strip()
        if not model_name:
            continue
        eval_results = _build_v2_results(row)
        if not eval_results:
            continue
        log = _make_log("alpaca_eval_2", model_name, row, eval_results, retrieved_timestamp, source_metadata)
        developer = log.model_info.developer or "unknown"
        model_slug = log.model_info.name.replace("/", "_")
        filepath = save_evaluation_log(log, OUTPUT_DIR_V2, developer, model_slug)
        print(f"Saved: {filepath}")
        count += 1
    return count


def main():
    retrieved_timestamp = str(time.time())
    print("=" * 60)
    print("AlpacaEval leaderboard adapter")
    print("=" * 60)

    try:
        n1 = fetch_alpaca_eval_1(retrieved_timestamp)
        print(f"\nProcessed {n1} models from AlpacaEval 1.0")
    except Exception as e:
        print(f"Error processing AlpacaEval 1.0: {e}")
        import traceback
        traceback.print_exc()

    try:
        n2 = fetch_alpaca_eval_2(retrieved_timestamp)
        print(f"\nProcessed {n2} models from AlpacaEval 2.0")
    except Exception as e:
        print(f"Error processing AlpacaEval 2.0: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
