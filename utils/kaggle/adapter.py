"""
Generalized adapter for Kaggle Community Benchmarks.

Kaggle Benchmarks are community-published evaluation suites. Each benchmark is
identified by an ``owner/slug`` pair (e.g. ``cohere-labs/global-mmlu-lite``) and
exposes a public leaderboard. This adapter:

1. Optionally enumerates ALL published benchmarks via Kaggle's (undocumented but
   unauthenticated) ``ListBenchmarks`` RPC, and/or accepts explicit ``owner/slug``
   pairs on the command line.
2. Fetches each benchmark's leaderboard from the public REST endpoint.
3. Converts every model row (and every task result within it) into the EvalEval
   schema, handling both numeric and boolean task results.

Data sources:
- List:        POST https://www.kaggle.com/api/i/benchmarks.BenchmarkService/ListBenchmarks
- Leaderboard: GET  https://www.kaggle.com/api/v1/benchmarks/{owner}/{slug}/leaderboard

Usage:
    # Convert specific benchmark(s)
    uv run python -m utils.kaggle.adapter \
        --benchmark cohere-labs/global-mmlu-lite \
        --output-dir /tmp/eee-kaggle

    # Discover and convert all published benchmarks (slow)
    uv run python -m utils.kaggle.adapter --all --output-dir data/kaggle

    # Smoke test: discover but only convert the first 5
    uv run python -m utils.kaggle.adapter --all --limit 5 --output-dir /tmp/eee-kaggle
"""

import argparse
import time
from typing import Dict, Iterator, List, Optional, Tuple

import requests

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
    FetchError,
    fetch_json,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)

KAGGLE_BASE = "https://www.kaggle.com"
LIST_RPC_URL = f"{KAGGLE_BASE}/api/i/benchmarks.BenchmarkService/ListBenchmarks"
LEADERBOARD_URL = KAGGLE_BASE + "/api/v1/benchmarks/{owner}/{slug}/leaderboard"

# Server-side cap on ListBenchmarks page size.
LIST_PAGE_SIZE = 200


def _kaggle_session() -> Tuple[requests.Session, str]:
    """Open a session with an anonymous XSRF token.

    The ListBenchmarks RPC is unauthenticated but requires the XSRF cookie/header
    handshake that Kaggle hands out on any page load.
    """
    session = requests.Session()
    try:
        session.get(f"{KAGGLE_BASE}/benchmarks", timeout=60)
    except requests.RequestException as e:
        raise FetchError(f"Kaggle XSRF handshake failed: {e}") from e
    xsrf = session.cookies.get("XSRF-TOKEN")
    if not xsrf:
        raise FetchError("Could not obtain XSRF-TOKEN cookie from Kaggle")
    return session, xsrf


# Kaggle aggregationType -> normalized metric_kind (for safe cross-source aggregation).
# Only map aggregation types that describe the *measured value*; generic reduction
# modes (AVERAGE, SUM) are not metric families and would conflate unrelated metrics,
# so they are intentionally omitted (the raw type is kept in source additional_details).
_AGGREGATION_TO_METRIC_KIND = {
    "PERCENTAGE_PASSED": "pass_rate",
}
# Kaggle displayType -> metric_unit. PERCENTAGES are stored as proportions in [0, 1].
_DISPLAY_TO_UNIT = {
    "PERCENTAGES": "proportion",
    "COUNTS": "count",
}


def _benchmark_owner(bench: dict) -> Optional[str]:
    """Resolve the leaderboard URL owner for a benchmark object.

    Org-owned benchmarks are addressed by the organization slug; otherwise the
    creating user's username is used.
    """
    org = bench.get("organization")
    if org and org.get("slug"):
        return org["slug"]
    return (bench.get("ownerUser") or {}).get("userName")


def _benchmark_meta(bench: dict) -> Dict[str, Optional[str]]:
    """Extract benchmark-level metadata (only present on the discovery path).

    The leaderboard endpoint omits these, but ``ListBenchmarks`` carries the
    scoring config we need to set direction (``lower_is_better``) and metric
    unit/kind, which would otherwise be guessed.
    """
    version = (bench.get("task") or {}).get("version") or {}
    return {
        "benchmark_id": bench.get("id"),
        "sort_order": version.get("sortOrder"),
        "aggregation_type": version.get("aggregationType"),
        "display_type": version.get("displayType"),
    }


def list_benchmarks() -> Iterator[Dict[str, object]]:
    """Enumerate published Kaggle benchmarks.

    Yields dicts with ``owner``, ``slug``, ``name`` and ``meta``. The leaderboard
    URL owner is the organization slug when the benchmark is org-owned, otherwise
    the creating user's username.
    """
    session, xsrf = _kaggle_session()
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "x-xsrf-token": xsrf,
    }
    page_token = ""
    while True:
        body = {"filter": {}, "pageSize": LIST_PAGE_SIZE, "pageToken": page_token}
        # On a page failure, raise after the benchmarks already yielded: the
        # caller keeps that partial progress but is told discovery was truncated,
        # so a `--all` run is never reported as a clean success when its tail is
        # missing.
        try:
            resp = session.post(LIST_RPC_URL, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise FetchError(f"ListBenchmarks page fetch failed: {e}") from e
        # A 200 error envelope or changed RPC schema would otherwise look like an
        # empty final page and end discovery as a clean success — validate the
        # shape so a broken response is a failure, not a silently empty corpus.
        if not isinstance(data, dict) or not isinstance(data.get("benchmarks"), list):
            raise FetchError(
                f"unexpected ListBenchmarks response shape (pageToken={page_token!r})"
            )
        for bench in data["benchmarks"]:
            if not bench.get("published"):
                continue
            slug = bench.get("slug")
            if not slug:
                continue
            owner = _benchmark_owner(bench)
            if not owner:
                continue
            yield {
                "owner": owner,
                "slug": slug,
                "name": bench.get("name", slug).strip(),
                "meta": _benchmark_meta(bench),
            }
        page_token = data.get("nextPageToken") or ""
        if not page_token:
            break


def fetch_leaderboard(owner: str, slug: str) -> Optional[List[dict]]:
    """Fetch a benchmark leaderboard.

    Returns the list of rows on success (possibly empty for a benchmark with no
    submissions), or ``None`` when the fetch itself failed (HTTP/network/parse
    error, including the 403 returned for non-existent/private benchmarks). The
    caller distinguishes these: an empty list is "no data", ``None`` is a failure
    that must be surfaced rather than silently treated as empty.
    """
    url = LEADERBOARD_URL.format(owner=owner, slug=slug)
    try:
        data = fetch_json(url)
    except FetchError as e:
        print(f"  ! could not fetch leaderboard for {owner}/{slug}: {e}")
        return None
    # A 200 with an error envelope or a changed schema (no list-valued `rows`) is
    # a fetch failure, not an empty leaderboard — don't let an upstream API break
    # masquerade as "no data".
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        print(f"  ! unexpected leaderboard response shape for {owner}/{slug}")
        return None
    return rows


def _build_eval_result(
    task: dict, owner: str, slug: str, meta: Optional[Dict[str, object]] = None
) -> Optional[EvaluationResult]:
    """Convert a single ``taskResult`` entry into an EvaluationResult.

    Handles numeric results (continuous score in [0, 1]) and boolean results
    (binary 0/1). Other result cases (e.g. custom tuples) are skipped.

    ``meta`` carries benchmark-level scoring config (from the discovery path)
    used to set ``lower_is_better`` and the metric unit/kind; when absent these
    fall back to sensible defaults.
    """
    meta = meta or {}
    task_name = task.get("benchmarkTaskName") or slug
    result = task.get("result", {})

    # Direction: Kaggle sorts leaderboards DESCENDING when higher is better.
    lower_is_better = meta.get("sort_order") == "ASCENDING"

    score: Optional[float] = None
    score_type: Optional[ScoreType] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    metric_kind: Optional[str] = None
    metric_unit: Optional[str] = None
    uncertainty: Optional[Uncertainty] = None

    if result.get("hasNumericResult"):
        numeric = result.get("numericResult") or {}
        value = numeric.get("value")
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        # Aggregation-derived kind/unit describe a numeric metric; they are
        # meaningless for a single boolean pass/fail, so only attach them here.
        metric_kind = _AGGREGATION_TO_METRIC_KIND.get(meta.get("aggregation_type"))
        metric_unit = _DISPLAY_TO_UNIT.get(meta.get("display_type"))
        # Kaggle's API does not expose a metric's scale. Most numeric results are
        # accuracies / pass-rates in [0, 1]; for those we can safely declare a
        # bounded continuous metric. Values outside [0, 1] (counts, percentages,
        # latencies, ...) have an unknown scale, so we leave score_type and bounds
        # unset rather than fabricate misleading [0, 1] bounds.
        # Only declare bounded [0, 1] when the value plausibly is a proportion.
        # A COUNTS metric whose value happens to fall in [0, 1] (e.g. a count of
        # 0 or 1) must not claim a max possible score of 1, so skip bounding when
        # Kaggle marks the metric as counts.
        if meta.get("display_type") != "COUNTS" and 0.0 <= score <= 1.0:
            score_type = ScoreType.continuous
            min_score, max_score = 0.0, 1.0
        # Kaggle's `confidenceInterval` is a symmetric half-width ("confidence
        # radius") around the score, so the emitted bounds must bracket the score
        # (cf. utils/hle/adapter.py), not be centered at zero.
        if numeric.get("hasConfidenceInterval"):
            try:
                ci = float(numeric.get("confidenceInterval"))
            except (TypeError, ValueError):
                ci = None
            if ci is not None:
                uncertainty = Uncertainty(
                    confidence_interval=ConfidenceInterval(
                        lower=round(score - ci, 4),
                        upper=round(score + ci, 4),
                        method="unknown",
                    )
                )
    elif result.get("hasBooleanResult"):
        score_type = ScoreType.binary
        score = 1.0 if result.get("booleanResult") else 0.0
        min_score, max_score = 0.0, 1.0
    else:
        return None

    benchmark_url = f"{KAGGLE_BASE}/benchmarks/{owner}/{slug}"
    return EvaluationResult(
        evaluation_name=task_name,
        evaluation_timestamp=result.get("evaluationDate"),
        source_data=SourceDataUrl(
            dataset_name=slug,
            source_type="url",
            url=[benchmark_url],
        ),
        metric_config=MetricConfig(
            evaluation_description=f"Kaggle Benchmarks - {task_name}",
            metric_name=task_name,
            metric_kind=metric_kind,
            metric_unit=metric_unit,
            lower_is_better=lower_is_better,
            score_type=score_type,
            min_score=min_score,
            max_score=max_score,
        ),
        score_details=ScoreDetails(
            score=round(score, 4),
            uncertainty=uncertainty,
        ),
    )


def convert_benchmark(
    owner: str,
    slug: str,
    name: str,
    rows: List[dict],
    output_dir: str,
    retrieved_timestamp: str,
    meta: Optional[Dict[str, object]] = None,
) -> int:
    """Convert all leaderboard rows for one benchmark and save them. Returns count."""
    meta = meta or {}
    benchmark_url = f"{KAGGLE_BASE}/benchmarks/{owner}/{slug}"

    source_details = {
        "platform": "kaggle",
        "benchmark_owner": owner,
        "benchmark_url": benchmark_url,
    }
    for key in ("benchmark_id", "aggregation_type", "display_type"):
        if meta.get(key) is not None:
            source_details[key] = str(meta[key])

    count = 0

    for row in rows:
        model_slug = row.get("modelVersionSlug")
        if not model_slug:
            continue
        model_display_name = row.get("modelVersionName", "")

        eval_results: List[EvaluationResult] = []
        for task in row.get("taskResults", []):
            result = _build_eval_result(task, owner, slug, meta)
            if result is not None:
                eval_results.append(result)
        if not eval_results:
            continue

        model_info = make_model_info(
            model_name=model_slug,
            additional_details={"display_name": model_display_name}
            if model_display_name and model_display_name != model_slug
            else None,
        )

        # Owner-qualify the id: benchmarks are owner/slug resources, so two
        # owners sharing a slug (with the same model and run timestamp) would
        # otherwise collide on this logical run identifier.
        evaluation_id = (
            f"{owner}/{slug}/{model_info.id.replace('/', '_')}/{retrieved_timestamp}"
        )
        eval_log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=make_source_metadata(
                source_name=f"{name} (Kaggle Benchmarks)",
                organization_name=owner,
                organization_url="https://www.kaggle.com",
                evaluator_relationship=EvaluatorRelationship.third_party,
                additional_details=source_details,
            ),
            eval_library=EvalLibrary(
                name="kaggle benchmarks",
                version="unknown",
                additional_details={"url": benchmark_url},
            ),
            model_info=model_info,
            evaluation_results=eval_results,
        )

        if "/" in model_info.id:
            dev, _ = model_info.id.split("/", 1)
        else:
            dev = "unknown"
        filepath = save_evaluation_log(eval_log, output_dir, dev, model_slug)
        print(f"  saved {filepath}")
        count += 1

    return count


def _resolve_targets(args) -> Tuple[List[Dict[str, object]], bool]:
    """Build the (de-duplicated) list of benchmarks to process from CLI args.

    Returns the targets and a flag indicating whether ``--all`` discovery was
    truncated by a fetch error, so the caller can exit non-zero instead of
    reporting an incomplete corpus as success.

    Note: the ``--benchmark`` path has no benchmark-level ``meta`` (the
    leaderboard endpoint omits it), so direction/unit enrichment is only applied
    on the ``--all`` discovery path.
    """
    targets: List[Dict[str, object]] = []
    for spec in args.benchmark or []:
        if "/" not in spec:
            raise SystemExit(f"--benchmark expects owner/slug, got: {spec!r}")
        owner, slug = spec.split("/", 1)
        targets.append({"owner": owner, "slug": slug, "name": slug, "meta": {}})

    discovery_failed = False
    if args.all:
        print("Discovering published benchmarks via Kaggle ListBenchmarks RPC...")
        try:
            for bench in list_benchmarks():
                targets.append(bench)
                if args.limit and len(targets) >= args.limit:
                    break
        except FetchError as e:
            # Keep what was discovered before the failure, but record that the
            # corpus is incomplete so the run does not exit 0.
            print(f"  ! benchmark discovery truncated: {e}")
            discovery_failed = True

    # De-duplicate by (owner, slug), preferring the entry that carries benchmark
    # meta (the discovered one) so enrichment is not lost to a bare --benchmark
    # duplicate or an explicit/--all overlap.
    deduped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for t in targets:
        key = (t["owner"], t["slug"])
        existing = deduped.get(key)
        if existing is None or (not existing.get("meta") and t.get("meta")):
            deduped[key] = t
    return list(deduped.values()), discovery_failed


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle Community Benchmarks leaderboards to the EvalEval schema."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        metavar="OWNER/SLUG",
        help="Specific benchmark to convert (repeatable), e.g. cohere-labs/global-mmlu-lite",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Discover and convert all published benchmarks via the ListBenchmarks RPC.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When using --all, stop after this many benchmarks (smoke testing).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/kaggle",
        help="Base output directory (default: data/kaggle).",
    )
    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.error("provide --benchmark OWNER/SLUG and/or --all")

    targets, discovery_failed = _resolve_targets(args)
    retrieved_timestamp = str(time.time())

    print("=" * 60)
    print(f"Converting {len(targets)} Kaggle benchmark(s) -> {args.output_dir}")
    print("=" * 60)

    total_models = 0
    total_benchmarks = 0
    failed_fetches = []
    conversion_failures = []
    empty_or_dropped = []
    for bench in targets:
        owner, slug, name = bench["owner"], bench["slug"], bench["name"]
        print(f"\n[{owner}/{slug}] {name}")
        rows = fetch_leaderboard(owner, slug)
        if rows is None:
            # Fetch failed (transient/HTTP/parse error) — distinct from an empty
            # leaderboard, so we don't report it as a clean "no data" success.
            failed_fetches.append(f"{owner}/{slug}")
            continue
        if not rows:
            print("  (no rows)")
            continue
        try:
            count = convert_benchmark(
                owner,
                slug,
                name,
                rows,
                args.output_dir,
                retrieved_timestamp,
                meta=bench.get("meta"),
            )
        except (TypeError, AttributeError, KeyError) as e:
            # A single malformed benchmark payload (e.g. taskResults: null) must
            # not abort the whole --all run — isolate it as a per-benchmark
            # failure, keep going, and exit non-zero at the end.
            print(f"  ! conversion failed for {owner}/{slug}: {e}")
            conversion_failures.append(f"{owner}/{slug}")
            continue
        print(f"  -> {count} model(s)")
        if count == 0:
            # A non-empty leaderboard that produced nothing means every row was
            # unusable (e.g. an unhandled result type) — surface it, don't hide it.
            empty_or_dropped.append(f"{owner}/{slug}")
        total_models += count
        total_benchmarks += 1

    print("\n" + "=" * 60)
    print(f"Done: {total_models} models across {total_benchmarks} benchmarks")
    if empty_or_dropped:
        print(
            f"WARNING: {len(empty_or_dropped)} non-empty leaderboard(s) produced "
            f"0 records (all rows unusable): {', '.join(empty_or_dropped)}"
        )
    if failed_fetches:
        print(
            f"WARNING: {len(failed_fetches)} benchmark(s) could not be fetched "
            f"and were skipped: {', '.join(failed_fetches)}"
        )
    if conversion_failures:
        print(
            f"WARNING: {len(conversion_failures)} benchmark(s) failed to convert "
            f"(malformed payload): {', '.join(conversion_failures)}"
        )
    if discovery_failed:
        print(
            "WARNING: benchmark discovery was truncated by a fetch error; "
            "the converted set is incomplete."
        )
    print("=" * 60)
    # Non-zero exit when any fetch/conversion failed or discovery was truncated,
    # so callers and CI don't read a partial run as a clean success.
    if failed_fetches or conversion_failures or discovery_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
