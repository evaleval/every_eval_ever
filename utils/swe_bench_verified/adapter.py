"""
Script to convert SWE-bench Verified leaderboard data to the EvalEval schema format.

Data source:
- SWE-bench experiments repo: https://github.com/swe-bench/experiments
  Cloned to a temporary directory at runtime; cleaned up on exit.

Each subdirectory under evaluation/verified/ is a submission with:
  - metadata.yaml: model/org info, tags
  - results/results.json: resolved/no_generation/no_logs instance lists

Score = len(resolved) / 500  (500 total SWE-bench Verified instances)

Usage:
    cd every_eval_ever
    .venv/bin/python -m utils.swe_bench_verified.adapter
"""

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
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
)
from every_eval_ever.helpers import SCHEMA_VERSION, get_developer, get_model_id, save_evaluation_log
from utils.swe_helpers import parse_date_from_dir

SWE_BENCH_REPO = "https://github.com/swe-bench/experiments"
SWE_BENCH_SUBDIR = "evaluation/verified"
OUTPUT_DIR = "data/swe-bench-verified-leaderboard"


def normalize_org(org) -> str:
    """Normalize org field which can be str, list, or None."""
    if isinstance(org, list):
        return ", ".join(str(o) for o in org if o)
    return str(org) if org else ""


def normalize_model_name(model) -> str:
    """Normalize a raw model value to a clean model name string.

    Handles:
    - HuggingFace URLs: https://huggingface.co/org/model → org/model
    - Plain strings returned as-is
    """
    if not model:
        return ""
    s = str(model)
    if s.startswith("https://huggingface.co/"):
        s = s[len("https://huggingface.co/"):]
    return s


def get_primary_model(tags: dict, info: dict, dir_name: str) -> str:
    """Extract the primary model name from tags, falling back to submission info."""
    raw = tags.get("model")
    # tags.model can be a list or a plain string
    if isinstance(raw, list):
        models = raw
    elif raw is not None:
        models = [raw]
    else:
        models = []

    if models:
        return normalize_model_name(models[0])
    # Fallback: use submission name from info
    return info.get("name", dir_name)


def convert_submission(submission_dir: Path, retrieved_timestamp: str, total_instances: int) -> EvaluationLog:
    """Convert a single SWE-bench submission directory to an EvaluationLog."""
    dir_name = submission_dir.name

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required to run this adapter. Install it with: pip install pyyaml"
        ) from e

    # Read metadata
    with open(submission_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    # Read results
    with open(submission_dir / "results" / "results.json") as f:
        results = json.load(f)

    tags = metadata.get("tags", {}) or {}
    info = metadata.get("info", {}) or {}

    # Primary model: first element of tags.model (list or string), fallback to submission name
    primary_model = get_primary_model(tags, info, dir_name)

    developer = get_developer(primary_model)
    model_id = get_model_id(primary_model, developer)

    # Score: resolved / total_instances
    resolved = results.get("resolved", [])
    score = len(resolved) / total_instances

    # Build additional_details (all values must be strings)
    additional_details: dict[str, str] = {
        "submission_name": str(info.get("name", "")),
        "agent_organization": normalize_org(tags.get("org", "")),
        "open_source_model": str(tags.get("os_model", "")),
        "open_source_system": str(tags.get("os_system", "")),
        "verified": str(tags.get("checked", "")),
        "attempts": str((tags.get("system") or {}).get("attempts", "")),
        "submission_dir": dir_name,
    }
    site = info.get("site")
    if site:
        additional_details["site"] = str(site)
    report = info.get("report")
    if report:
        additional_details["report"] = str(report)

    # Score details
    score_details: dict[str, str] = {
        "resolved_count": str(len(resolved)),
    }
    no_generation = results.get("no_generation", [])
    if no_generation:
        score_details["no_generation_count"] = str(len(no_generation))
    no_logs = results.get("no_logs", [])
    if no_logs:
        score_details["no_logs_count"] = str(len(no_logs))

    # Sanitize identifier components for use in evaluation_id
    sanitized_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id.replace("/", "_"))
    submission_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", dir_name)
    eval_id = (
        f"swe-bench-verified/{sanitized_id}/{submission_slug}/{retrieved_timestamp}"
    )
    evaluation_timestamp = parse_date_from_dir(dir_name)

    eval_result = EvaluationResult(
        evaluation_name="SWE-bench Verified",
        source_data=SourceDataUrl(
            dataset_name="SWE-bench Verified",
            source_type="url",
            url=["https://www.swebench.com"],
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description=(
                "Fraction of 500 verified GitHub issues resolved (0.0–1.0)"
            ),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(
            score=score,
            details=score_details,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    available_tools=[AvailableTool(name="bash")],
                ),
            ),
        ),
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=evaluation_timestamp,
        source_metadata=SourceMetadata(
            source_name="SWE-bench Verified Leaderboard",
            source_type="documentation",
            source_organization_name="SWE-bench",
            source_organization_url="https://www.swebench.com",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name="swe-bench", version="unknown"),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer if developer != "unknown" else None,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def main():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required to run this adapter. Install it with: pip install datasets"
        ) from e

    retrieved_timestamp = str(time.time())
    count = 0
    errors = 0

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    total_instances = len(ds)
    print(f"Loaded {total_instances} instances from SWE-bench/SWE-bench_Verified\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {SWE_BENCH_REPO} into {tmpdir} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", SWE_BENCH_REPO, tmpdir],
            check=True,
        )

        swe_bench_path = Path(tmpdir) / SWE_BENCH_SUBDIR
        submissions = sorted(d for d in swe_bench_path.iterdir() if d.is_dir())
        print(f"Found {len(submissions)} submission directories\n")

        for submission_dir in submissions:
            try:
                eval_log = convert_submission(submission_dir, retrieved_timestamp, total_instances)
                dev = eval_log.model_info.developer or "unknown"
                # Use model name without developer prefix for the directory
                model_name = eval_log.model_info.name.split("/")[-1]
                filepath = save_evaluation_log(eval_log, OUTPUT_DIR, dev, model_name)
                score = eval_log.evaluation_results[0].score_details.score
                print(f"  [{score:.1%}] {submission_dir.name} → {filepath}")
                count += 1
            except Exception as e:
                print(f"  ERROR {submission_dir.name}: {e}")
                errors += 1

    print(f"\nGenerated {count} files, {errors} errors → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
