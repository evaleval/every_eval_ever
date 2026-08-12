"""
Script to fetch Vectara Hallucination Leaderboard results from the
``vectara/results`` Hugging Face dataset and convert them to the EvalEval
schema format.

Data source:
- Structured per-model result files in https://huggingface.co/datasets/vectara/results
  (pinned at SOURCE_COMMIT), one ``results_<timestamp>.json`` per model.

The leaderboard scores summaries of a private corpus of 7700+ articles using
HHEM-2.3, Vectara's commercial hallucination evaluation model. The evaluated
corpus is not publicly released, so each result carries private source data
while the public result files are recorded as provenance.

Usage:
    uv run python -m every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter
"""

import argparse
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataPrivate,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    make_model_info,
    make_source_metadata,
    raw_capture,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

SRC = "vectara-hallucination-leaderboard"
# ONE collection per SOURCE: every Vectara leaderboard record lands under this
# collection, never under a per-benchmark directory.
COLLECTION = SRC
OUTPUT_DIR = f"data/{COLLECTION}"

SOURCE_REPO = "vectara/results"
# Pinned so a refresh is reproducible; bump it deliberately to pick up new runs.
SOURCE_COMMIT = "7c104699e98ade53dd719f79ae9f7eb281c8107d"
SOURCE_DATASET_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}"
LEADERBOARD_REPO_URL = "https://github.com/vectara/hallucination-leaderboard"
SOURCE_ORG_URL = "https://vectara.com"
HF_TREE_API = (
    f"https://huggingface.co/api/datasets/{SOURCE_REPO}/tree/{SOURCE_COMMIT}"
)
RESULT_FILENAME_RE = re.compile(r"^results_(?P<timestamp>.+)\.json$")
TREE_PAGE_LIMIT = 1000

SCORING_MODEL = "Vectara HHEM-2.3"
EVAL_DATASET_NAME = f"{SRC} private evaluation corpus"
EVAL_DATASET_DESCRIPTION = (
    "Over 7700 articles spanning news, technology, science, medicine, legal, "
    "sports, business and education, in both low and high complexity, from 50 "
    "to 24K words."
)
EVAL_DATASET_AVAILABILITY = (
    "Private/proprietary corpus; not publicly released to avoid overfitting."
)
# Documented in the leaderboard README as a policy with unenumerated
# exceptions, so it is recorded as prose rather than a generation_config value
# we cannot attribute per model.
TEMPERATURE_NOTE = (
    "Temperature 0 when calling the LLMs, except where that was impossible or "
    "not available (per source documentation)."
)

# Vectara's accessibility annotation describes weight availability.
_MODEL_AVAILABILITY_BY_ACCESSIBILITY = {
    "open": "open_weights",
    "commercial": "closed_weights",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class MetricSpec:
    """One leaderboard metric and the metadata needed to describe it."""

    key: str
    metric_id: str
    metric_name: str
    metric_kind: str
    metric_unit: str
    lower_is_better: bool
    min_score: float
    max_score: float
    description: str
    # Vectara reports only a subset of the metrics per category/complexity
    # slice, so aggregate-only metrics are skipped when building breakdowns.
    in_breakdowns: bool = True
    diagnostic: bool = False


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="hallucination_rate",
        # Canonical registry metric: namespacing it would fragment the join.
        metric_id="hallucination-rate",
        metric_name="Hallucination Rate",
        metric_kind="rate",
        metric_unit="percent",
        lower_is_better=True,
        min_score=0.0,
        max_score=100.0,
        description=(
            "Percentage of generated summaries judged by HHEM-2.3 to contain "
            "factual inconsistencies or unsupported claims."
        ),
    ),
    MetricSpec(
        key="factual_consistency_rate",
        metric_id=f"{SRC}.factual-consistency-rate",
        metric_name="Factual Consistency Rate",
        metric_kind="rate",
        metric_unit="percent",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
        description=(
            "Percentage of generated summaries judged factually consistent; "
            "reported by the source as 100 minus the hallucination rate."
        ),
        in_breakdowns=False,
    ),
    MetricSpec(
        key="answer_rate",
        metric_id=f"{SRC}.answer-rate",
        metric_name="Answer Rate",
        metric_kind="rate",
        metric_unit="percent",
        lower_is_better=False,
        min_score=0.0,
        max_score=100.0,
        description=(
            "Percentage of documents for which the model produced a summary "
            "rather than refusing or being blocked by a content filter."
        ),
    ),
    MetricSpec(
        key="average_summary_length",
        metric_id=f"{SRC}.average-summary-length",
        metric_name="Average Summary Length",
        metric_kind="length",
        metric_unit="words",
        lower_is_better=False,
        min_score=0.0,
        max_score=float("inf"),
        description=(
            "Mean generated summary length in words, reported as a diagnostic "
            "alongside the hallucination rate."
        ),
        diagnostic=True,
    ),
)

# Slice containers in a source row, keyed by the row field holding them.
BREAKDOWN_FIELDS = (
    ("category", "category_results"),
    ("text_complexity", "text_complexity_results"),
)


def source_url(source_path: str) -> str:
    """Build the pinned resolve URL for one source result file."""
    quoted = urllib.parse.quote(source_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/"
        f"{SOURCE_COMMIT}/{quoted}"
    )


def list_result_files() -> list[str]:
    """List every pinned per-model result file in the source repository."""
    url = f"{HF_TREE_API}?recursive=true&limit={TREE_PAGE_LIMIT}"
    entries = fetch_json(url)
    if not isinstance(entries, list):
        raise ValueError(f"unexpected tree payload for {url}")
    if len(entries) >= TREE_PAGE_LIMIT:
        # The tree fits in one page today. Paging needs the response's Link
        # header, which fetch_json does not surface, so stop rather than
        # silently convert a truncated roster.
        raise ValueError(
            f"{SOURCE_REPO} tree returned a full page of {len(entries)} "
            "entries; add cursor pagination before refreshing"
        )
    paths = [
        entry["path"]
        for entry in entries
        if entry.get("type") == "file"
        and RESULT_FILENAME_RE.match(Path(entry.get("path", "")).name)
    ]
    if not paths:
        raise ValueError(
            f"no result files found in {SOURCE_REPO} at {SOURCE_COMMIT}"
        )
    return sorted(paths)


def fetch_source_rows(paths: list[str]) -> dict[str, dict]:
    """Fetch each pinned result file, keyed by its repository path."""
    rows: dict[str, dict] = {}
    for index, path in enumerate(paths, start=1):
        print(f"  [{index}/{len(paths)}] {path}")
        rows[path] = fetch_json(source_url(path))
    return rows


def source_timestamp(source_path: str) -> str:
    """Extract the run timestamp Vectara encodes in the result filename."""
    match = RESULT_FILENAME_RE.match(Path(source_path).name)
    if match is None:
        raise ValueError(f"unrecognized result filename: {source_path}")
    return match.group("timestamp")


def model_identity(source_path: str) -> tuple[str, str]:
    """Split a source path into its developer and model directory names."""
    parts = Path(source_path).parts
    if len(parts) != 3:
        raise ValueError(
            f"expected <developer>/<model>/<file>, got: {source_path}"
        )
    developer = require_identity(parts[0], "Vectara model developer")
    model_name = require_identity(parts[1], "Vectara model name")
    return developer, model_name


def _score(value: object, spec: MetricSpec) -> float:
    """Coerce a source score and reject values outside the declared bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{spec.key}: non-numeric score {value!r}")
    score = float(value)
    if score != score:
        raise ValueError(f"{spec.key}: score is NaN")
    if not spec.min_score <= score <= spec.max_score:
        raise ValueError(
            f"{spec.key}: score {score} outside declared bounds "
            f"[{spec.min_score}, {spec.max_score}]"
        )
    return score


def build_result(
    spec: MetricSpec,
    score: float,
    source_path: str,
    slice_kind: str,
    slice_name: str | None,
) -> EvaluationResult:
    """Build one EvaluationResult for a metric on one slice of the corpus."""
    slice_suffix = f".{slice_kind}" + (f".{slice_name}" if slice_name else "")
    metric_parameters: dict[str, str | float | bool | None] = {
        "slice_kind": slice_kind
    }
    if slice_name is not None:
        metric_parameters["slice"] = slice_name

    # Per-result details carry only what varies across results. The file,
    # commit, resolve URL, scoring model and temperature policy are constant
    # for the whole log and live once in source_metadata.
    additional_details = {"source_metric_key": spec.key}
    if spec.diagnostic:
        # lower_is_better is a required boolean with no "neither" member, so
        # say plainly that this metric is not an optimization target.
        additional_details["diagnostic_metric"] = "true"
        additional_details["direction_note"] = (
            "Descriptive context rather than an optimization target; "
            "lower_is_better is not meaningful for this metric."
        )

    dataset_name = EVAL_DATASET_NAME
    if slice_name is not None:
        dataset_name = f"{EVAL_DATASET_NAME} ({slice_kind}: {slice_name})"

    return EvaluationResult(
        evaluation_result_id=f"{SRC}{slice_suffix}.{spec.key}",
        evaluation_name=f"{SRC}{slice_suffix}",
        source_data=SourceDataPrivate(
            dataset_name=dataset_name,
            source_type="other",
        ),
        evaluation_timestamp=source_timestamp(source_path),
        metric_config=MetricConfig(
            evaluation_description=spec.description,
            metric_id=spec.metric_id,
            metric_name=spec.metric_name,
            metric_kind=spec.metric_kind,
            metric_unit=spec.metric_unit,
            metric_parameters=metric_parameters,
            lower_is_better=spec.lower_is_better,
            score_type=ScoreType.continuous,
            min_score=spec.min_score,
            max_score=spec.max_score,
            additional_details=additional_details,
        ),
        score_details=ScoreDetails(
            score=score,
            details={"source_value_unit": spec.metric_unit},
        ),
    )


def build_results(
    row: dict,
    source_path: str,
    failures: list[SourceRecordFailure],
) -> list[EvaluationResult]:
    """Build every representable result for one model, recording the rest."""
    results: list[EvaluationResult] = []
    specs_by_key = {spec.key: spec for spec in METRICS}

    aggregates = row.get("results")
    if not isinstance(aggregates, dict):
        raise ValueError("results must be an object")

    for spec in METRICS:
        ref = f"{source_path} overall {spec.key}"
        try:
            payload = aggregates.get(spec.key)
            if not isinstance(payload, dict):
                raise ValueError(f"{spec.key}: missing aggregate payload")
            results.append(
                build_result(
                    spec,
                    _score(payload.get(spec.key), spec),
                    source_path,
                    "overall",
                    None,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=ref, reason=str(exc), source_record=aggregates
                )
            )

    for slice_kind, field in BREAKDOWN_FIELDS:
        container = row.get(field)
        if container is None:
            continue
        if not isinstance(container, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=f"{source_path} {field}",
                    reason=f"{field} must be an object",
                    source_record=container,
                )
            )
            continue
        for slice_name, scores in sorted(container.items()):
            if not isinstance(scores, dict):
                failures.append(
                    SourceRecordFailure(
                        source_ref=f"{source_path} {field} {slice_name}",
                        reason="slice payload must be an object",
                        source_record=scores,
                    )
                )
                continue
            for key, value in sorted(scores.items()):
                ref = f"{source_path} {slice_kind} {slice_name} {key}"
                spec = specs_by_key.get(key)
                try:
                    if spec is None:
                        raise ValueError(f"unknown breakdown metric {key!r}")
                    results.append(
                        build_result(
                            spec,
                            _score(value, spec),
                            source_path,
                            slice_kind,
                            slice_name,
                        )
                    )
                except Exception as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=ref,
                            reason=str(exc),
                            source_record={key: value},
                        )
                    )

    return results


def build_log(
    source_path: str,
    row: dict,
    retrieved_timestamp: str,
    failures: list[SourceRecordFailure],
) -> tuple[EvaluationLog, str, str]:
    """Build one EvaluationLog per model, plus its datastore routing."""
    developer, model_name = model_identity(source_path)
    evaluation_timestamp = source_timestamp(source_path)
    config = row.get("config") or {}
    annotations = row.get("model_annotations") or {}
    accessibility = str(annotations.get("accessibility", "unknown"))

    results = build_results(row, source_path, failures)
    if not results:
        raise ValueError("model has no usable results")

    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        additional_details={
            # The pinned result files record no serving platform, and the
            # leaderboard's prose API notes cover only part of the roster.
            "deployment_type": "unknown",
            "model_availability": _MODEL_AVAILABILITY_BY_ACCESSIBILITY.get(
                accessibility, "unknown"
            ),
            "source_accessibility": accessibility,
            "source_model_name": str(config.get("model_name", "")),
            "source_model_dtype": str(config.get("model_dtype", "")),
            "source_model_sha": str(config.get("model_sha", "")),
            "source_model_size": str(annotations.get("model_size", "unknown")),
        },
    )

    eval_log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # Anchored on the source run timestamp, which changes only when Vectara
        # re-runs the evaluation, so a refresh is idempotent.
        evaluation_id=f"{SRC}/{developer}_{model_name}/{evaluation_timestamp}",
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=evaluation_timestamp,
        source_metadata=make_source_metadata(
            source_name="Vectara Hallucination Leaderboard",
            organization_name="Vectara",
            organization_url=SOURCE_ORG_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                "source_role": "aggregator",
                "structured_results_hf_repo": SOURCE_REPO,
                "structured_results_dataset": SOURCE_DATASET_URL,
                "leaderboard_repository": LEADERBOARD_REPO_URL,
                "source_commit": SOURCE_COMMIT,
                "source_file": source_path,
                "source_resolve_url": source_url(source_path),
                "scoring_model": SCORING_MODEL,
                "evaluated_corpus": EVAL_DATASET_NAME,
                "evaluated_corpus_description": EVAL_DATASET_DESCRIPTION,
                "evaluated_corpus_availability": EVAL_DATASET_AVAILABILITY,
                "generation_temperature": TEMPERATURE_NOTE,
                "evaluation_timestamp_source": (
                    "Parsed from the source result filename."
                ),
            },
        ),
        eval_library=EvalLibrary(
            name="unknown",
            version="unknown",
            additional_details={
                "scoring_model": SCORING_MODEL,
                "leaderboard_repository": LEADERBOARD_REPO_URL,
            },
        ),
        model_info=model_info,
        evaluation_results=results,
    )
    return eval_log, developer, model_name


def convert_rows(
    rows: dict[str, dict],
    retrieved_timestamp: str,
    output_dir: str = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert every fetched source row, retaining rejected provenance."""
    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []

    for source_path, row in sorted(rows.items()):
        failure_count_before = len(failures)
        try:
            if not isinstance(row, dict):
                raise ValueError("source row must be an object")
            eval_log, developer, model_name = build_log(
                source_path, row, retrieved_timestamp, failures
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=output_dir,
                    developer=developer,
                    model_name=model_name,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_path,
                    reason=(
                        f"no output written: {exc}"
                        if len(failures) > failure_count_before
                        else str(exc)
                    ),
                    source_record=row if isinstance(row, dict) else None,
                )
            )

    return SourceConversionResult(
        source_name="Vectara Hallucination Leaderboard",
        total_records=len(rows),
        records=outputs,
        failures=failures,
    )


def load_rows(input_json: Path | None, save_raw_json: Path | None) -> dict:
    """Load source rows from a saved snapshot, or fetch and optionally save."""
    if input_json is not None:
        rows = json.loads(input_json.read_text(encoding="utf-8"))
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"{input_json} contains no source rows")
        return rows

    print(f"Listing {SOURCE_REPO} result files at {SOURCE_COMMIT}...")
    raw_capture.record_pointer(
        kind="hf_dataset",
        reference=SOURCE_REPO,
        revision=SOURCE_COMMIT,
        url=SOURCE_DATASET_URL,
    )
    paths = list_result_files()
    print(f"Fetching {len(paths)} result files...")
    rows = fetch_source_rows(paths)
    if save_raw_json is not None:
        save_raw_json.parent.mkdir(parents=True, exist_ok=True)
        save_raw_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Saved raw source rows: {save_raw_json}")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Vectara Hallucination Leaderboard results to EEE records."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Datastore collection directory for generated records.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Replay a saved raw source snapshot instead of fetching.",
    )
    parser.add_argument(
        "--save-raw-json",
        type=Path,
        help="Write the fetched raw source rows to this path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    retrieved_timestamp = str(time.time())

    print("=" * 60)
    print("Fetching Vectara Hallucination Leaderboard results...")
    print("=" * 60)

    rows = load_rows(args.input_json, args.save_raw_json)
    result = convert_rows(rows, retrieved_timestamp, args.output_dir)
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(f"Saved: {path}")

    print(
        f"\nProcessed {len(paths)} of {result.total_records} models "
        f"from {SOURCE_REPO}"
    )
    if result.failures:
        report_path = save_failure_report(
            result, default_failure_report_path(args.output_dir)
        )
        print(f"Failure report: {report_path}")
        result.raise_if_incomplete()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return len(paths)


if __name__ == "__main__":
    main()
