"""Template: minimal aggregate EEE adapter (one score per (model, benchmark)).

Copy into every_eval_ever/adapters/<name>/adapter.py and replace <src>/<Platform>
and the fetch. Mirror every_eval_ever/adapters/llm_stats for anything more complex,
and every_eval_ever/adapters/bfcl for the failure-accounting shape. Build models BY
HAND — the helpers.make_* functions are stale (miss eval_library / per-result
source_data).

Publishing and drop-accounting are NOT hand-rolled: `save_evaluation_logs` validates
the whole batch before creating any file and rolls back on failure, and
`SourceConversionResult` carries the rows you could not convert.
"""
from pathlib import Path

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    save_evaluation_logs,
    save_failure_report,
)

SRC = "<src>"
# ONE collection per SOURCE, never per benchmark: records land in
# data/<COLLECTION>/<developer>/<model>/<uuid>.json. Per-benchmark directories collide
# with other sources that converted the same benchmark (fields.md §collection).
COLLECTION = "<src>"


def _result(entry):                                # one benchmark -> one result
    benchmark, score = entry["benchmark"], entry["score"]
    dataset_url = entry["dataset_url"]
    return EvaluationResult(
        evaluation_result_id=f"{SRC}.{benchmark}",     # stable join key; instances point here
        evaluation_name=f"{SRC}.{benchmark}",          # namespaced id, NOT a title
        source_data=SourceDataUrl(dataset_name=benchmark, source_type="url",
                                  # DATASET url ONLY -> a paper/leaderboard citation is NOT
                                  # dataset provenance and has no typed home; put it in
                                  # additional_details, never here (fields.md "no typed home").
                                  # `url` needs >=1 entry; for an HF dataset use
                                  # SourceDataHfDataset and set hf_repo (source_type
                                  # 'hf_dataset' without hf_repo is flagged).
                                  url=[dataset_url]),
        metric_config=MetricConfig(
            # Describes an ACCURACY metric on a 0-1 PROPORTION scale. CHANGE every field for
            # your metric -> name/id/kind/unit/direction/bounds; else you emit valid-but-wrong
            # metadata (validating != correct; see the metric_config notes in fields.md).
            metric_name="accuracy", metric_kind="accuracy", metric_unit="proportion",
            # ALWAYS set metric_id, and pick its FORM deliberately (fields.md metric_id):
            # a real GLOBAL metric takes the registry's canonical id -- plain "accuracy"
            # here, NOT f"{SRC}.accuracy", which would fragment the one join the datastore
            # exists for. Only a leaderboard-SPECIFIC construct gets namespaced
            # (f"{SRC}.overall"), and a bare "score"/"rank"/"cost" is never acceptable.
            metric_id="accuracy",
            lower_is_better=False, score_type=ScoreType.continuous,  # never omit score_type
            # BOUNDS MUST CONTAIN THE SCORE (hard error otherwise). Use the scale the
            # SOURCE's numbers are on -- proportion 0-1 / percent 0-100 / unbounded +-inf --
            # and rescale explicitly if you convert. See gotchas.md.
            min_score=0.0, max_score=1.0),
        score_details=ScoreDetails(score=score))


def make_log(row, retrieved_ts):                   # DEFAULT: one log per model
    # Source rows are DICTS (like every adapter in this repo): a dict is what you can
    # attach to a SourceRecordFailure, which requires JSON-compatible values.
    model, developer = row["model"], row["developer"]
    eval_ts = row["eval_ts"]
    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # STABLE anchor -> idempotent (NOT now). If the source is MUTABLE (re-scraped, a
        # live leaderboard), fold a source revision/run-id into this key; eval_ts alone can
        # collide across changed snapshots (fields.md timestamps). If the source lists the
        # same model under several settings (effort/temperature/scaffold/date), those axes
        # MUST appear here or the variants collapse into one record.
        evaluation_id=f"{SRC}/{developer}_{model}/{eval_ts}",
        retrieved_timestamp=retrieved_ts,                 # STRING epoch = now (record-creation)
        evaluation_timestamp=eval_ts,                     # when the eval ran
        source_metadata=SourceMetadata(
            source_name="<Platform>", source_type="documentation",
            source_organization_name="<Aggregator org>",  # NOT the model dev / a username
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={"source_role": "aggregator"}),   # str values only
        # name the harness if the format reveals it (lm-eval/inspect); else "unknown":
        eval_library=EvalLibrary(name="unknown", version="unknown"),
        model_info=ModelInfo(
            name=model, id=f"{developer}/{model}",        # canonicalize via registry
            developer=developer,                          # prefer helpers.get_developer to a
                                                          # private per-adapter developer map
            # The library defaults BOTH axes below to "unknown" (so a green *library*
            # validate hides an unset value) while the CLI/bot validate ERRORS on a missing
            # key or a non-enum value. Set the REAL value for your source:
            additional_details={
                "deployment_type": "externally_managed",  # self_deployed|externally_managed|unknown
                "model_availability": "open_weights",      # open_weights|closed_weights|unknown
            }),
        evaluation_results=[_result(entry) for entry in row["results"]])


def convert_rows(rows, out_root, retrieved_ts) -> SourceConversionResult:
    """Convert every row; keep the valid ones, ACCOUNT for the rest.

    A row you cannot represent (no model identity, non-numeric score, unknown metric
    bounds) becomes a SourceRecordFailure — never a silent skip, which would shrink an
    aggregate's denominator behind a warning. A row that is intentionally not an
    evaluation (a published random baseline) is a SourceRecordExclusion instead: it is
    reported but does NOT fail the command.
    """
    rows = list(rows)
    outputs, failures = [], []
    for index, row in enumerate(rows, start=1):
        try:
            log = make_log(row, retrieved_ts)
            outputs.append(EvaluationLogOutput(
                eval_log=EvaluationLog.model_validate(log.model_dump()),
                base_dir=out_root / COLLECTION,      # -> <out_root>/<COLLECTION>/<dev>/<model>/
                developer=row["developer"], model_name=row["model"]))
        except Exception as exc:                     # one bad row must not kill the run
            failures.append(SourceRecordFailure(
                source_ref=f"{SRC} row {index}", reason=str(exc),
                # source_record must be JSON-COMPATIBLE (dict/list/scalars) -- a tuple or
                # a custom object is rejected when the report is written.
                source_record=row))
    return SourceConversionResult(source_name=SRC, total_records=len(rows),
                                  records=outputs, failures=failures)


def fetch_rows(args):
    """PLACEHOLDER — replace with your source fetch. Yield one dict per model:
    {"model", "developer", "eval_ts", "results": [{"benchmark", "score",
    "dataset_url"}, ...]} — keep it JSON-compatible so a rejected row can be attached
    verbatim to its failure report.

    Give the real adapter --save-raw-json / --input-json so a fetched payload can be
    replayed offline: that is what makes a fixture-based test possible without mocking
    HTTP, and what lets a reviewer reproduce your numbers.
    """
    raise NotImplementedError("wire up the source fetch")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    # Default OUTSIDE the checkout: generated records belong in the HF datastore PR, not
    # in this repo. Point at data/ only for a deliberate refresh. The path must END IN
    # `data` so the validator can resolve the canonical data/<collection>/... prefix.
    ap.add_argument("--output-dir", type=Path, default=Path(f"/tmp/{SRC}-smoke/data"))
    ap.add_argument("--failure-report", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def run(args):
    import time
    retrieved_ts = str(time.time())                       # ONE timestamp per run, not per row
    result = convert_rows(fetch_rows(args), args.output_dir, retrieved_ts)
    paths = save_evaluation_logs(result.records)          # batch-validated, atomic, rollback
    print(f"wrote {len(paths)} logs -> {args.output_dir / COLLECTION}")
    if result.failures:
        report = save_failure_report(
            result,
            args.failure_report
            or default_failure_report_path(args.output_dir / COLLECTION))
        print(f"failure report: {report}")                # adapter_reports/, OUTSIDE data/
        result.raise_if_incomplete()                      # EXIT NON-ZERO on a partial refresh
    return paths


if __name__ == "__main__":   # run: uv run python -m every_eval_ever.adapters.<name>.adapter
    run(parse_args())
    # then validate the written FILES (the CLI rejects a bare dir):
    #   uv run python -m every_eval_ever validate '/tmp/<src>-smoke/data/<src>/*/*/*.json'
    #   (or scripts/validate.sh /tmp/<src>-smoke/data)
