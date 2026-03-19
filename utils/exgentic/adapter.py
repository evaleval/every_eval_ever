"""
Script to convert Exgentic Open Agent Leaderboard results to the EvalEval schema format.

Exgentic is an open-source framework for evaluating AI agents across multiple
benchmarks (AppWorld, SWE-bench, BrowseComp+, Tau2, etc.) with different agent
frameworks (Claude Code, LiteLLM Tool Calling, SmolAgents, etc.) and models.

Each evaluation run produces a results.json file containing aggregate scores,
session counts, cost data, and per-session details. This adapter reads those
results and converts them to EEE-conformant JSON files.

Data source:
- Exgentic experiments output: results.json files produced by `exgentic batch aggregate`
- HuggingFace dataset: https://huggingface.co/datasets/Exgentic/open-agent-leaderboard-results

Usage:
    # From local experiment results
    uv run python -m utils.exgentic.adapter --results-dir /path/to/experiments

    # From HuggingFace dataset
    uv run python -m utils.exgentic.adapter --from-hf
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_types import (
    AgenticEvalConfig,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    Uncertainty,
)
from helpers import save_evaluation_log, sanitize_filename

SCHEMA_VERSION = "0.2.2"
OUTPUT_DIR = "data/exgentic"
HF_DATASET = "Exgentic/open-agent-leaderboard-results"

# Map model name prefixes to developer organizations
MODEL_DEVELOPER_MAP = {
    "claude": ("Anthropic", "anthropic"),
    "gpt": ("OpenAI", "openai"),
    "gemini": ("Google", "google"),
}


def parse_model_info(model_name: str) -> tuple[str, str, str]:
    """Extract developer display name, developer slug, and model slug from exgentic model_name.

    Exgentic model names follow the pattern: provider/platform/model-name
    e.g. 'openai/aws/claude-opus-4-5', 'openai/Azure/gpt-5.2-2025-12-11'

    Returns:
        (developer_display, developer_slug, model_slug)
    """
    parts = model_name.split("/")
    raw_model = parts[-1] if parts else model_name

    developer_display = "unknown"
    developer_slug = "unknown"
    lower = raw_model.lower()
    for prefix, (display, slug) in MODEL_DEVELOPER_MAP.items():
        if lower.startswith(prefix):
            developer_display = display
            developer_slug = slug
            break

    return developer_display, developer_slug, raw_model


def make_agent_slug(agent_name: str) -> str:
    """Convert agent display name to a URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", agent_name.lower()).strip("-")


def convert_result(result: dict, retrieved_timestamp: str) -> EvaluationLog:
    """Convert a single exgentic result dict to an EvaluationLog."""
    model_name_raw = result.get("model_name") or "unknown"
    developer_display, developer_slug, model_slug = parse_model_info(model_name_raw)
    model_id = f"{developer_slug}/{model_slug}"

    benchmark = result.get("benchmark_name") or result.get("benchmark") or "unknown"
    agent_name = result.get("agent_name") or result.get("agent") or "unknown"
    agent_framework = result.get("agent") or make_agent_slug(agent_name)
    agent_slug = make_agent_slug(agent_name)
    subset = result.get("subset_name")

    eval_name = benchmark.lower().replace(" ", "-")
    if subset:
        eval_name = f"{eval_name}/{subset}"

    score = result.get("benchmark_score")
    if score is None:
        score = result.get("average_score", 0.0)

    # Build uncertainty from session counts
    total = result.get("total_sessions")
    uncertainty = None
    if total and int(total) > 0:
        uncertainty = Uncertainty(num_samples=int(total))

    # Build score details
    details: dict[str, str] = {}
    if result.get("average_agent_cost") is not None:
        details["average_agent_cost"] = str(round(float(result["average_agent_cost"]), 2))
    if result.get("total_run_cost") is not None:
        details["total_run_cost"] = str(round(float(result["total_run_cost"]), 2))
    if result.get("average_steps") is not None:
        details["average_steps"] = str(round(float(result["average_steps"]), 2))
    if result.get("percent_finished") is not None:
        details["percent_finished"] = str(round(float(result["percent_finished"]), 4))

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataUrl(
            dataset_name=eval_name,
            source_type="url",
            url=["https://github.com/Exgentic/exgentic"],
        ),
        evaluation_timestamp=retrieved_timestamp,
        metric_config=MetricConfig(
            evaluation_description=f"{benchmark} benchmark evaluation"
            + (f" ({subset} subset)" if subset else ""),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(
            score=round(float(score), 4) if score is not None else 0.0,
            uncertainty=uncertainty,
            details=details if details else None,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    additional_details={
                        "agent_name": agent_name,
                        "agent_framework": agent_framework,
                    },
                ),
            ),
        ),
    )

    sanitized_model_id = model_id.replace("/", "_")
    evaluation_id = f"{eval_name}/{agent_slug}__{sanitized_model_id}/{retrieved_timestamp}"

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name="Exgentic Open Agent Leaderboard",
            source_type="evaluation_run",
            source_organization_name="Exgentic",
            source_organization_url="https://github.com/Exgentic",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(
            name="exgentic",
            version="0.1.0",
        ),
        model_info=ModelInfo(
            name=model_slug,
            id=model_id,
            developer=developer_display,
            additional_details={
                "agent_name": agent_name,
                "agent_framework": agent_framework,
            },
        ),
        evaluation_results=[eval_result],
    )


def load_results_from_dir(results_dir: str) -> list[dict]:
    """Recursively find and load all results.json files under a directory."""
    results = []
    base = Path(results_dir)

    for config_path in sorted(base.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text())
            run_id = config.get("run_id")
            if not run_id:
                continue
            results_path = config_path.parent / run_id / "results.json"
            if not results_path.is_file():
                continue
            payload = json.loads(results_path.read_text())
            if "benchmark_score" not in payload:
                continue
            results.append(payload)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {config_path}: {e}")
    return results


def load_results_from_hf() -> list[dict]:
    """Load results from the HuggingFace dataset (default subset with raw exgentic data)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    ds = load_dataset(HF_DATASET, split="train")
    return list(ds)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Exgentic results to Every Eval Ever format"
    )
    parser.add_argument(
        "--results-dir",
        help="Path to exgentic experiments directory containing config.json files",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help=f"Load results from HuggingFace dataset ({HF_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for EEE JSON files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not args.results_dir and not args.from_hf:
        parser.error("Specify either --results-dir or --from-hf")

    if args.results_dir:
        results = load_results_from_dir(args.results_dir)
    else:
        results = load_results_from_hf()

    if not results:
        print("No results found.")
        sys.exit(1)

    print(f"Loaded {len(results)} result(s)")

    retrieved_timestamp = str(time.time())
    count = 0

    for result in results:
        try:
            eval_log = convert_result(result, retrieved_timestamp)
            model_info = eval_log.model_info
            developer_slug = sanitize_filename(model_info.developer or "unknown")
            model_name = sanitize_filename(model_info.name)
            filepath = save_evaluation_log(
                eval_log, args.output_dir, developer_slug, model_name
            )
            print(f"  {filepath}")
            count += 1
        except Exception as e:
            benchmark = result.get("benchmark", "?")
            agent = result.get("agent", "?")
            model = result.get("model_name", "?")
            print(f"Error processing {benchmark}/{agent}/{model}: {e}")

    print(f"\nGenerated {count} file(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
