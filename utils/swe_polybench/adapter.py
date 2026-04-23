"""
Script to convert SWE-PolyBench leaderboard data to the EvalEval schema format.

Data source:
- SWE-PolyBench experiments repo: https://github.com/amazon-science/SWE-PolyBench
  Branch: submission. Cloned to a temporary directory at runtime; cleaned up on exit.
- HF datasets: AmazonScience/SWE-PolyBench (PB) and AmazonScience/SWE-PolyBench_Verified (PBVerified).

Each subdirectory under evaluation/{PB,PBVerified}/ is a submission with:
  - metadata.yaml: name, oss, site, pass_rate, logo
  - logs/<instance_id>_result.json: per-instance resolved status

Score = resolved_count_in_lang / total_instances_for_lang_from_hf_dataset

One EvaluationLog is written per (dataset x submission x language).

Usage:
    cd every_eval_ever
    .venv/bin/python -m utils.swe_polybench.adapter
"""

import json
import re
import subprocess
import tempfile
import time
from collections import Counter
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
    SourceDataHf,
    SourceMetadata,
)
from every_eval_ever.helpers import SCHEMA_VERSION, get_developer, get_model_id, save_evaluation_log
from utils.swe_helpers import parse_date_from_dir, parse_model_from_dir

POLY_REPO = "https://github.com/amazon-science/SWE-PolyBench"
POLY_BRANCH = "submission"

DATASETS = {
    "PB": "AmazonScience/SWE-PolyBench",
    "PBVerified": "AmazonScience/SWE-PolyBench_Verified",
}
DATASET_LABELS = {"PB": "pb", "PBVerified": "pb-verified"}
DATASET_DISPLAY = {"PB": "SWE-PolyBench", "PBVerified": "SWE-PolyBench Verified"}

OUTPUT_BASE = "data/swe-polybench-leaderboard"


def convert_submission(
    submission_dir: Path,
    ds: str,
    lang: str,
    resolved_count: int,
    patch_applied_count: int,
    no_p2p_failed_count: int,
    total_instances_for_lang: int,
    retrieved_timestamp: str,
    metadata: dict,
) -> EvaluationLog:
    dir_name = submission_dir.name
    ds_label = DATASET_LABELS[ds]
    ds_display = DATASET_DISPLAY[ds]
    hf_repo = DATASETS[ds]

    agent, primary_model = parse_model_from_dir(dir_name)
    developer = get_developer(primary_model)
    model_id = get_model_id(primary_model, developer)

    sanitized_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", model_id.replace("/", "_"))
    submission_slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", dir_name)
    eval_id = f"swe-polybench/{ds_label}/{lang}/{sanitized_id}/{submission_slug}/{retrieved_timestamp}"

    evaluation_timestamp = parse_date_from_dir(dir_name)
    score = resolved_count / total_instances_for_lang

    additional_details: dict[str, str] = {
        "submission_name": str(metadata.get("name", "")),
        "language": lang,
        "dataset": ds_label,
        "oss": str(metadata.get("oss", "")),
        "site": str(metadata.get("site", "")),
        "pass_rate": str(metadata.get("pass_rate", "")),
        "submission_dir": dir_name,
        "agent": agent,
    }

    score_details: dict[str, str] = {
        "resolved_count": str(resolved_count),
        "total_instances_for_language": str(total_instances_for_lang),
        "patch_applied_count": str(patch_applied_count),
        "no_p2p_failed_count": str(no_p2p_failed_count),
    }

    eval_name = f"{ds_display} ({lang})"
    dataset_label = f"{ds_display} ({lang})"

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataHf(
            dataset_name=dataset_label,
            source_type="hf_dataset",
            hf_repo=hf_repo,
            hf_split="test",
            samples_number=total_instances_for_lang,
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
            source_name="SWE-PolyBench Leaderboard",
            source_type="documentation",
            source_organization_name="AmazonScience",
            source_organization_url="https://github.com/amazon-science/SWE-PolyBench",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name="swe-polybench", version="unknown"),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer if developer != "unknown" else None,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def load_hf_instance_maps(ds: str) -> tuple[dict[str, str], Counter]:
    """Return (instance_id -> language, Counter of language totals) from HF dataset."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required to run this adapter. Install it with: pip install datasets"
        ) from e

    hf_repo = DATASETS[ds]
    print(f"  Loading HF dataset {hf_repo} ...")
    dataset = load_dataset(hf_repo, split="test")
    id_to_lang: dict[str, str] = {}
    lang_counts: Counter = Counter()
    for row in dataset:
        iid = row["instance_id"]
        lang = row["language"]
        id_to_lang[iid] = lang
        lang_counts[lang] += 1
    return id_to_lang, lang_counts


def process_submission(
    submission_dir: Path,
    ds: str,
    id_to_lang: dict[str, str],
    lang_counts: Counter,
    retrieved_timestamp: str,
    yaml,
) -> list[tuple[EvaluationLog, str]]:
    """Return list of (EvaluationLog, lang) for each language found in this submission."""
    dir_name = submission_dir.name
    metadata_path = submission_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {dir_name}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    logs_dir = submission_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs/ not found in {dir_name}")

    result_files = sorted(logs_dir.glob("*_result.json"))
    if not result_files:
        raise FileNotFoundError(f"No *_result.json files in {dir_name}/logs/")

    # Aggregate per language
    langs_in_submission: set[str] = set()
    resolved_by_lang: Counter = Counter()
    patch_applied_by_lang: Counter = Counter()
    no_p2p_failed_by_lang: Counter = Counter()

    unknown_ids = []
    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)
        iid = data.get("instance_id", "")
        lang = id_to_lang.get(iid)
        if lang is None:
            unknown_ids.append(iid)
            lang = "unknown"
        langs_in_submission.add(lang)
        if data.get("resolved", False):
            resolved_by_lang[lang] += 1
        if data.get("patch_applied", False):
            patch_applied_by_lang[lang] += 1
        if data.get("no_p2p_failed", False):
            no_p2p_failed_by_lang[lang] += 1

    if unknown_ids:
        print(f"    WARNING: {len(unknown_ids)} instance_ids not in HF dataset, bucketed as 'unknown'")

    # Only emit records for languages actually present in this submission's result
    # files, to avoid spurious 0-score entries for uncovered languages.
    results = []
    for lang in langs_in_submission:
        total = lang_counts.get(lang)
        if total is None:
            continue
        eval_log = convert_submission(
            submission_dir=submission_dir,
            ds=ds,
            lang=lang,
            resolved_count=resolved_by_lang.get(lang, 0),
            patch_applied_count=patch_applied_by_lang.get(lang, 0),
            no_p2p_failed_count=no_p2p_failed_by_lang.get(lang, 0),
            total_instances_for_lang=total,
            retrieved_timestamp=retrieved_timestamp,
            metadata=metadata,
        )
        results.append((eval_log, lang))
    return results


def main():
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required to run this adapter. Install it with: pip install pyyaml"
        ) from e

    retrieved_timestamp = str(time.time())
    count = 0
    errors = 0

    # Load HF datasets first
    hf_maps: dict[str, tuple[dict[str, str], Counter]] = {}
    for ds in ("PB", "PBVerified"):
        id_to_lang, lang_counts = load_hf_instance_maps(ds)
        hf_maps[ds] = (id_to_lang, lang_counts)
        print(f"  [{ds}] {sum(lang_counts.values())} instances: {dict(lang_counts)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCloning {POLY_REPO} (branch={POLY_BRANCH}) into {tmpdir} ...")
        subprocess.run(
            ["git", "clone", "--branch", POLY_BRANCH, "--depth=1", POLY_REPO, tmpdir],
            check=True,
        )

        for ds in ("PB", "PBVerified"):
            ds_label = DATASET_LABELS[ds]
            eval_path = Path(tmpdir) / "evaluation" / ds
            if not eval_path.exists():
                print(f"  [SKIP] No evaluation/{ds} dir")
                continue

            id_to_lang, lang_counts = hf_maps[ds]
            submissions = sorted(d for d in eval_path.iterdir() if d.is_dir())
            print(f"\n[{ds}] Found {len(submissions)} submissions")

            for submission_dir in submissions:
                try:
                    logs_results = process_submission(
                        submission_dir, ds, id_to_lang, lang_counts, retrieved_timestamp, yaml
                    )
                    for eval_log, lang in logs_results:
                        dev = eval_log.model_info.developer or "unknown"
                        model_name = eval_log.model_info.name.split("/")[-1]
                        filepath = save_evaluation_log(eval_log, OUTPUT_BASE, dev, model_name)
                        score = eval_log.evaluation_results[0].score_details.score
                        print(f"  [{score:.1%}] {submission_dir.name} [{lang}] → {filepath}")
                        count += 1
                except Exception as e:
                    print(f"  ERROR {submission_dir.name}: {e}")
                    errors += 1

    print(f"\nGenerated {count} files, {errors} errors → {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
