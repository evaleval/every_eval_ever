"""aiXamine adapter — converts aiXamine public-API evaluation reports into EEE.

aiXamine (https://aixamine.qcri.org) is a safety, security and privacy-focused LLM trust-evaluation
platform. It groups 46 static + 5 dynamic tests into 9 services. This adapter
reads the aiXamine PUBLIC API and emits one aggregate EEE
log per (model, service):

    data/aixamine_<service>/<developer>/<model>/<uuid>.json

Each service log carries that service's PUBLIC tests as evaluation_results, plus a
per-category sub-result.

Usage:
    # offline, from a captured fixture bundle (report.json, model.json, services.json)
    uv run python -m every_eval_ever.adapters.aixamine.adapter --input-dir <dir>
    # live against the aiXamine public API
    uv run python -m every_eval_ever.adapters.aixamine.adapter \
        --api-url https://aixamine.qcri.org/api/v1 --report-id <id>
"""
import argparse
import json
import time
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
    SourceDataPrivate,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    save_evaluation_logs,
    save_failure_report,
)

SRC = "aixamine"
ORG_NAME = "aiXamine"
HOMEPAGE = "https://aixamine.qcri.org"
PAPER = "https://arxiv.org/abs/2608.20554"

# aiXamine tests that map to an existing canonical EEE benchmark ids
CANONICAL = {
    "simpleqa": "simpleqa",
    "truthfulqa": "truthfulqa",
    "triviaqa": "triviaqa",
    "bbq": "bbq",
    "xs-test": "xstest",
    "simple-safety": "simplesafetytests",
    "anthropic-redteam": "anthropic-red-team",
}


def collection_for(service_value):
    return f"{SRC}_{service_value.replace('-', '_')}"


def benchmark_id(test_value):
    return CANONICAL.get(test_value, test_value)


def _model_details(access_type):
    if access_type == "huggingface":
        return {"deployment_type": "self_deployed", "model_availability": "open_weights"}
    return {"deployment_type": "externally_managed", "model_availability": "closed_weights"}


def _access_date(model):
    ts = model.get("createdAt") or ""
    return ts[:10] if len(ts) >= 10 else None


def _dated_name(model):
    name = model["name"]
    if model.get("accessType") == "huggingface":
        return name
    date = _access_date(model)
    return f"{name}-{date}" if date else name


def _model_id(name, model):
    if "/" in name:
        return name
    dev = model.get("developer")
    return f"{dev}/{name}" if dev else name


def _resolve_developer(model):
    dev = (model.get("developer") or "").strip()
    if dev:
        return dev
    name = model.get("name", "")
    return name.split("/")[0] if "/" in name else None


def _metric_config(test_value, description):
    """aiXamine scores are a 0-100 rate (safe / pass / accepted); higher is better."""
    return MetricConfig(
        evaluation_description=description or None,
        metric_id=f"{SRC}.rate",
        metric_name="score",
        metric_kind="accuracy",
        metric_unit="percent",
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=100.0,
    )


def _result(service, test_value, name, description, score, categories, eval_ts):
    """One overall result for a test + dotted per-category sub-results."""
    bid = benchmark_id(test_value)
    results = [
        EvaluationResult(
            evaluation_result_id=f"{SRC}.{service}.{test_value}",
            evaluation_name=bid,
            evaluation_timestamp=eval_ts,
            source_data=SourceDataPrivate(dataset_name=bid, source_type="other"),
            metric_config=_metric_config(test_value, description),
            score_details=ScoreDetails(score=float(score)),
        )
    ]
    for cat, cscore in (categories or {}).items():
        results.append(
            EvaluationResult(
                evaluation_result_id=f"{SRC}.{service}.{test_value}.{cat}",
                evaluation_name=f"{bid}.{cat}",
                evaluation_timestamp=eval_ts,
                source_data=SourceDataPrivate(dataset_name=f"{bid} ({cat})", source_type="other"),
                metric_config=_metric_config(test_value, f"{name} — {cat}"),
                score_details=ScoreDetails(score=float(cscore)),
            )
        )
    return results


def _static_categories(cats):
    """report static categories: {cat: {score, subcategories:[...]}} -> {cat: score}."""
    return {c: v.get("score") for c, v in (cats or {}).items() if isinstance(v, dict) and v.get("score") is not None}


def build_service_logs(report, model, catalog, retrieved_ts):
    """One EvaluationLog per service present in the report (static + dynamic)."""
    name = _dated_name(model)
    developer = _resolve_developer(model)
    model_id = _model_id(name, model)
    mi = ModelInfo(name=name, id=model_id, developer=developer,
                   additional_details=_model_details(model.get("accessType")))

    # catalog lookups: service_value -> {name, tests:{test_value:{name,description}}}
    svc_meta = {s["value"]: s for s in catalog}

    logs = []  # (collection, developer, model_name, EvaluationLog)
    services = {}  # service_value -> list[EvaluationResult]

    # static
    for svc_value, svc in (report.get("services") or {}).items():
        tests = svc.get("tests") or {}
        for tv, tdata in tests.items():
            if tdata.get("score") is None:
                continue
            tmeta = _catalog_test(svc_meta, svc_value, tv)
            ets = _iso_from(tdata)
            services.setdefault(svc_value, []).extend(
                _result(svc_value, tv, tmeta.get("name", tv), tmeta.get("description"),
                        tdata["score"], _static_categories(tdata.get("categories")), ets)
            )

    # dynamic (latest version per test)
    for svc_value, svc in ((report.get("dynamic") or {}).get("services") or {}).items():
        tests = svc.get("tests") or {}
        for tv, tdata in tests.items():
            versions = tdata.get("versions") or []
            if not versions:
                continue
            v = versions[0]
            if v.get("score") is None:
                continue
            tmeta = _catalog_test(svc_meta, svc_value, tv)
            services.setdefault(svc_value, []).extend(
                _result(svc_value, tv, tmeta.get("name", tv), tmeta.get("description"),
                        v["score"], v.get("categories"), v.get("generatedAt"))
            )

    for svc_value, results in services.items():
        smeta = svc_meta.get(svc_value, {})
        collection = collection_for(svc_value)
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=f"{collection}/{developer}_{name.replace('/', '_')}",
            retrieved_timestamp=retrieved_ts,
            source_metadata=SourceMetadata(
                source_name=collection,
                source_type="documentation",
                source_organization_name=ORG_NAME,
                source_organization_url=HOMEPAGE,
                evaluator_relationship=EvaluatorRelationship.first_party,
                additional_details={
                    "service": svc_value,
                    "service_name": smeta.get("name", svc_value),
                    "paper": PAPER,
                    "homepage": HOMEPAGE,
                },
            ),
            eval_library=EvalLibrary(name=SRC, version="unknown"),
            model_info=mi,
            evaluation_results=results,
        )
        logs.append((collection, developer, name, log))
    return logs


def _catalog_test(svc_meta, svc_value, test_value):
    svc = svc_meta.get(svc_value, {})
    for t in svc.get("tests", []):
        if t.get("value") == test_value:
            return t
    return {}


def _iso_from(tdata):
    return None  # static tests carry a duration, not a wall-clock timestamp in the report


# ── Fetch ────────────────────────────────────────────────────────────────────

def _catalog(catalog):
    return catalog["services"] if isinstance(catalog, dict) and "services" in catalog else catalog


def _bundle_from_fixture(input_dir):
    d = Path(input_dir)
    return (json.load(open(d / "report.json")),
            json.load(open(d / "model.json")),
            _catalog(json.load(open(d / "services.json"))))


def _bundle_live(base, report_id):
    report = fetch_json(f"{base}/reports/{report_id}")
    model = fetch_json(f"{base}/models/{report['model']}")
    catalog = _catalog(fetch_json(f"{base}/examinations/getServices"))
    return report, model, catalog


def enumerate_reports(base, access_type=None, page_size=50, max_pages=None):
    """Page through the PUBLIC search endpoint; yield (report_id, model_id, accessType)."""
    page = 1
    while True:
        d = fetch_json(f"{base}/reports/search?page={page}&limit={page_size}")
        rows = d.get("overview") or []
        if not rows:
            break
        for r in rows:
            if access_type and r.get("accessType") != access_type:
                continue
            if r.get("report") and r.get("_id"):
                yield r["report"], r["_id"], r.get("accessType")
        total_pages = d.get("totalPages") or 1
        if page >= total_pages or (max_pages and page >= max_pages):
            break
        page += 1


def _outputs_for(report, model, catalog, out_root, retrieved_ts, outputs, failures):
    """Append this (report, model)'s service logs to outputs / failures."""
    if _resolve_developer(model) is None:
        return
    try:
        logs = build_service_logs(report, model, catalog, retrieved_ts)
    except Exception as exc:
        failures.append(SourceRecordFailure(
            source_ref=f"{SRC} report {report.get('_id')}", reason=str(exc),
            source_record={"model": model.get("name")}))
        return
    for collection, developer, model_name, log in logs:
        try:
            outputs.append(EvaluationLogOutput(
                eval_log=EvaluationLog.model_validate(log.model_dump()),
                base_dir=out_root / collection,
                developer=developer, model_name=model_name))
        except Exception as exc:
            failures.append(SourceRecordFailure(
                source_ref=f"{collection}/{model_name}", reason=str(exc),
                source_record={"collection": collection, "model": model_name}))


def build_result(args, retrieved_ts):
    out_root, outputs, failures = args.output_dir, [], []
    if args.input_dir:
        report, model, catalog = _bundle_from_fixture(args.input_dir)
        _outputs_for(report, model, catalog, out_root, retrieved_ts, outputs, failures)
        total = 1
    else:
        base = args.api_url.rstrip("/")
        catalog = _catalog(fetch_json(f"{base}/examinations/getServices"))
        if args.report_id:
            pairs = [(args.report_id, None, None)]
        else:  # --all: enumerate via public search
            pairs = list(enumerate_reports(base, args.access_type, max_pages=args.max_pages))
        total = len(pairs)
        for report_id, model_id, _ in pairs:
            try:
                report = fetch_json(f"{base}/reports/{report_id}")
                model = fetch_json(f"{base}/models/{model_id or report['model']}")
            except Exception as exc:
                failures.append(SourceRecordFailure(
                    source_ref=f"{SRC} report {report_id}", reason=str(exc),
                    source_record={"report_id": report_id}))
                continue
            _outputs_for(report, model, catalog, out_root, retrieved_ts, outputs, failures)
    return SourceConversionResult(source_name=SRC, total_records=total,
                                  records=outputs, failures=failures)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Convert aiXamine reports into EEE records")
    ap.add_argument("--output-dir", type=Path, default=Path(f"/tmp/{SRC}-smoke/data"))
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="Fixture dir with report.json/model.json/services.json (offline)")
    ap.add_argument("--api-url", type=str, default="https://aixamine.qcri.org/api/v1",
                    help="aiXamine public API base (default: production; pass a dev URL to override)")
    ap.add_argument("--report-id", type=str, default=None, help="Convert a single report")
    ap.add_argument("--all", dest="all", action="store_true",
                    help="Enumerate all public reports via /reports/search (default when no report-id/input-dir)")
    ap.add_argument("--access-type", type=str, default=None, choices=["huggingface", "openai"],
                    help="Restrict --all to open-weight (huggingface) or API (openai) models")
    ap.add_argument("--max-pages", type=int, default=None, help="Cap search pages (testing)")
    ap.add_argument("--failure-report", type=Path, default=None)
    return ap.parse_args(argv)


def run(args):
    retrieved_ts = str(time.time())
    result = build_result(args, retrieved_ts)
    paths = save_evaluation_logs(result.records)
    print(f"wrote {len(paths)} logs from {result.total_records} report(s) -> {args.output_dir}")
    if result.failures:
        rp = save_failure_report(result, args.failure_report or default_failure_report_path(args.output_dir))
        print(f"failure report: {rp}")
        result.raise_if_incomplete()
    return paths


if __name__ == "__main__":
    run(parse_args())
