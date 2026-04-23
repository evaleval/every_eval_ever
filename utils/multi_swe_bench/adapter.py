"""
Script to convert Multi-SWE-Bench leaderboard data to the EvalEval schema format.

Data source:
- Multi-SWE-bench experiments repo: https://github.com/multi-swe-bench/experiments
  Cloned to a temporary directory at runtime; cleaned up on exit.

Each subdirectory under evaluation/<lang>/verified/ is a submission with:
  - metadata.yaml: name, orgIcon, oss, site, verified
  - results/results.json: resolved/unresolved/etc instance lists

Score = len(resolved) / total_instances  (from results.json)

Usage:
    cd every_eval_ever
    .venv/bin/python -m utils.multi_swe_bench.adapter
"""

import json
import os
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
from utils.swe_helpers import parse_date_from_dir, parse_model_from_dir

MULTI_SWE_REPO = "https://github.com/multi-swe-bench/experiments"
LANGUAGES = ["c", "c++", "go", "java", "javascript", "rust", "typescript"]
OUTPUT_BASE = "data/multi-swe-bench-leaderboard"


def convert_submission(
    submission_dir: Path,
    lang: str,
    retrieved_timestamp: str,
) -> EvaluationLog:
    """Convert a single Multi-SWE-Bench submission directory to an EvaluationLog."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required to run this adapter. Install it with: pip install pyyaml"
        ) from e

    dir_name = submission_dir.name

    with open(submission_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    with open(submission_dir / "results" / "results.json") as f:
        results = json.load(f)

    total_instances = results.get("total_instances", 0)
    if total_instances == 0:
        raise ValueError(f"total_instances is 0 for {dir_name}, skipping")

    resolved = results.get("resolved", [])
    score = len(resolved) / total_instances

    agent, primary_model = parse_model_from_dir(dir_name)
    developer = get_developer(primary_model)
    model_id = get_model_id(primary_model, developer)

    sanitized_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id.replace("/", "_"))
    submission_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", dir_name)
    eval_id = f"multi-swe-bench/{lang}/{sanitized_id}/{submission_slug}/{retrieved_timestamp}"

    evaluation_timestamp = parse_date_from_dir(dir_name)

    additional_details: dict[str, str] = {
        "submission_name": str(metadata.get("name", "")),
        "language": lang,
        "oss": str(metadata.get("oss", "")),
        "site": str(metadata.get("site", "")),
        "verified": str(metadata.get("verified", "")),
        "submission_dir": dir_name,
        "agent": agent,
    }

    score_details: dict[str, str] = {
        "resolved_count": str(len(resolved)),
        "total_instances": str(total_instances),
        "submitted_instances": str(results.get("submitted_instances", "")),
        "completed_instances": str(results.get("completed_instances", "")),
        "unresolved_instances": str(results.get("unresolved_instances", "")),
        "empty_error_patch_instances": str(results.get("empty_error_patch_instances", "")),
    }

    dataset_label = f"Multi-SWE-bench ({lang})"
    eval_name = f"Multi-SWE-Bench ({lang})"

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataUrl(
            dataset_name=dataset_label,
            source_type="url",
            url=["https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench"],
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description=f"Fraction of {lang} GitHub issues resolved (0.0–1.0)",
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
            source_name="Multi-SWE-Bench Leaderboard",
            source_type="documentation",
            source_organization_name="ByteDance-Seed",
            source_organization_url="https://github.com/multi-swe-bench/experiments",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name="multi-swe-bench", version="unknown"),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer if developer != "unknown" else None,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def main():
    retrieved_timestamp = str(time.time())
    count = 0
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {MULTI_SWE_REPO} into {tmpdir} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", MULTI_SWE_REPO, tmpdir],
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
            check=True,
        )

        for lang in LANGUAGES:
            verified_path = Path(tmpdir) / "evaluation" / lang / "verified"
            if not verified_path.exists():
                print(f"  [SKIP] No verified/ dir for language: {lang}")
                continue

            submissions = sorted(d for d in verified_path.iterdir() if d.is_dir())
            print(f"\n[{lang}] Found {len(submissions)} submissions")

            for submission_dir in submissions:
                try:
                    eval_log = convert_submission(submission_dir, lang, retrieved_timestamp)
                    dev = eval_log.model_info.developer or "unknown"
                    model_name = eval_log.model_info.name.split("/")[-1]
                    filepath = save_evaluation_log(eval_log, OUTPUT_BASE, dev, model_name)
                    score = eval_log.evaluation_results[0].score_details.score
                    print(f"  [{score:.1%}] {submission_dir.name} → {filepath}")
                    count += 1
                except Exception as e:
                    print(f"  ERROR {submission_dir.name}: {e}")
                    errors += 1

    print(f"\nGenerated {count} files, {errors} errors → {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
