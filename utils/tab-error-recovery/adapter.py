"""
TAB Platform → EvalEval EEE Schema Adapter
Converts TAB error recovery benchmark results into EEE schema format.

Usage:
    python adapter.py --all --pretty -o output.json

Source: https://tabverified.ai
Organization: TAB Platform LLC
Evaluator relationship: third_party
"""

import json
import time
import uuid
from datetime import datetime, timezone


MODEL_MAP = {
    "gemini-3.5-flash": {"name": "Gemini 3.5 Flash", "id": "google/gemini-3.5-flash", "developer": "Google"},
    "gemini-3.1-pro": {"name": "Gemini 3.1 Pro", "id": "google/gemini-3.1-pro", "developer": "Google"},
    "claude-opus-4-6": {"name": "Claude Opus 4.6", "id": "anthropic/claude-opus-4-6", "developer": "Anthropic"},
    "claude-opus-4-7": {"name": "Claude Opus 4.7", "id": "anthropic/claude-opus-4-7", "developer": "Anthropic"},
    "gpt-5.4": {"name": "GPT-5.4", "id": "openai/gpt-5.4", "developer": "OpenAI"},
    "gpt-5.5": {"name": "GPT-5.5", "id": "openai/gpt-5.5", "developer": "OpenAI"},
    "grok-4.20": {"name": "Grok 4.20", "id": "xai/grok-4.20", "developer": "xAI"},
    "deepseek-v4-flash": {"name": "DeepSeek V4 Flash", "id": "deepseek/deepseek-v4-flash", "developer": "DeepSeek"},
    "deepseek-v4-pro": {"name": "DeepSeek V4 Pro", "id": "deepseek/deepseek-v4-pro", "developer": "DeepSeek"},
}

TAB_SOURCE_METADATA = {
    "source_name": "TAB Platform - AI Agent Verification",
    "source_type": "evaluation_run",
    "source_organization_name": "TAB Platform LLC",
    "source_organization_url": "https://tabverified.ai",
    "evaluator_relationship": "third_party",
    "additional_details": {
        "calibration_status": "135/135 PASS",
        "adversarial_audit_status": "160/160 RESISTANT",
        "harness_count": "101",
        "total_benchmarks": "340+"
    }
}

TAB_EVAL_LIBRARY = {
    "name": "TAB Platform",
    "version": "1.0.0",
    "additional_details": {
        "judge_model": "GLM-5 via OpenRouter",
        "temperature": "0.7",
        "platform_url": "https://tabverified.ai"
    }
}

CAT_DESCRIPTIONS = {
    "error_message_utilization": "Does the agent extract and use information from error messages to inform its recovery strategy?",
    "strategy_diversity": "Does the agent try genuinely different approaches when the first attempt fails?",
    "retry_storm_detection": "Does the agent recognize when retrying is futile and stop rather than creating a retry storm?",
    "graceful_degradation": "When full recovery is impossible, does the agent provide partial results and clear communication?",
}


def convert_error_recovery_to_eee(model_id, overall_score, category_scores, evaluation_timestamp=None):
    if evaluation_timestamp is None:
        evaluation_timestamp = datetime.now(timezone.utc).isoformat()

    retrieved_ts = time.time()
    model_info = MODEL_MAP.get(model_id, {"name": model_id, "id": f"unknown/{model_id}", "developer": "Unknown"})
    file_uuid = str(uuid.uuid4())
    eval_id = f"tab-error-recovery/{model_info['id']}/{retrieved_ts}"

    eval_results = []

    # Overall score
    eval_results.append({
        "evaluation_name": "TAB Error Recovery Efficiency - Overall",
        "source_data": {
            "source_type": "other",
            "dataset_name": "TAB Error Recovery Benchmark v1.0",
            "num_instances": 40,
            "additional_details": {
                "categories": "error_message_utilization, strategy_diversity, retry_storm_detection, graceful_degradation",
                "tests_per_category": "10",
                "benchmark_url": "https://tabverified.ai/static/specialty-error-recovery.html"
            }
        },
        "evaluation_timestamp": evaluation_timestamp,
        "metric_config": {
            "evaluation_description": "Measures how well an AI agent recovers from errors: utilizing error messages, trying diverse strategies, avoiding retry storms, and degrading gracefully.",
            "metric_id": "tab.error_recovery.overall",
            "metric_name": "Error Recovery Efficiency",
            "metric_kind": "accuracy",
            "metric_unit": "percent",
            "lower_is_better": False,
            "score_type": "continuous",
            "min_score": 0,
            "max_score": 100,
            "llm_scoring": {
                "judges": [{"model_id": "thudm/glm-5", "provider": "OpenRouter"}],
                "input_prompt": "Evaluate the agent's error recovery behavior across four dimensions."
            },
            "additional_details": {"harness_config": "default", "calibration_verified": "true"}
        },
        "score_details": {"score": overall_score, "details": {"test_count": "40"}}
    })

    # Per-category scores
    for cat, score in category_scores.items():
        eval_results.append({
            "evaluation_name": f"TAB Error Recovery - {cat.replace('_', ' ').title()}",
            "source_data": {
                "source_type": "other",
                "dataset_name": f"TAB Error Recovery Benchmark v1.0 - {cat}",
                "num_instances": 10
            },
            "evaluation_timestamp": evaluation_timestamp,
            "metric_config": {
                "evaluation_description": CAT_DESCRIPTIONS.get(cat, cat),
                "metric_id": f"tab.error_recovery.{cat}",
                "metric_name": cat.replace("_", " ").title(),
                "metric_kind": "accuracy",
                "metric_unit": "percent",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0,
                "max_score": 100
            },
            "score_details": {"score": score, "details": {"test_count": "10"}}
        })

    return {
        "schema_version": "0.2.0",
        "evaluation_id": eval_id,
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": str(retrieved_ts),
        "source_metadata": TAB_SOURCE_METADATA,
        "eval_library": TAB_EVAL_LIBRARY,
        "model_info": {
            "name": model_info["name"],
            "id": model_info["id"],
            "developer": model_info["developer"],
            "additional_details": {"inference_platform": "tabverified.ai"}
        },
        "evaluation_results": eval_results
    }, file_uuid


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Convert TAB error recovery data to EEE schema")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    args = parser.parse_args()

    # Example: generate for one model
    doc, file_uuid = convert_error_recovery_to_eee(
        model_id="gemini-3.5-flash",
        overall_score=26.1,
        category_scores={
            "error_message_utilization": 12.8,
            "strategy_diversity": 22.4,
            "retry_storm_detection": 64.5,
            "graceful_degradation": 4.8
        },
        evaluation_timestamp="2026-05-22T23:18:48Z"
    )

    out_path = Path(args.output_dir) / "google" / "gemini-3.5-flash"
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / f"{file_uuid}.json"

    with open(filepath, "w") as f:
        json.dump(doc, f, indent=2 if args.pretty else None)

    print(f"Generated: {filepath}")
