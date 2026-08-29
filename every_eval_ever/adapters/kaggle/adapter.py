"""
Generalized adapter for Kaggle Community Benchmarks.

Kaggle Benchmarks are community-published evaluation suites. Each benchmark is
identified by an ``owner/slug`` pair (e.g. ``cohere-labs/global-mmlu-lite``) and
exposes a public leaderboard. This adapter:

1. Enumerates every published benchmark through Kaggle's (undocumented but
   unauthenticated) ``ListBenchmarks`` RPC, which is also the only place the
   per-benchmark scoring configuration is published.
2. Fetches each benchmark's leaderboard from the public REST endpoint.
3. Converts every model row, and every task result within it, into one
   evaluation record.

The index is fetched even for an explicit ``--benchmark``, so a targeted run and
an ``--all`` run describe the same benchmark identically.

The source is Kaggle, the platform that published the leaderboard. Which
benchmark a record came from is in ``evaluation_name``, and who uploaded the
benchmark in ``source_metadata.additional_details``, so one collection holds
every benchmark instead of one per author.

A benchmark's overall score and its sub-benchmark scores live in one record, so
``evaluation_name`` distinguishes them: ``kaggle.<owner>.<benchmark>`` for the
aggregate and ``kaggle.<owner>.<benchmark>.<sub-benchmark>`` for each part.
Summing a file would otherwise double-count. The owner is part of the name
because a benchmark is an ``owner/slug`` resource and 15 slugs are published by
more than one owner.

What Kaggle does *not* publish is a metric's unit or scale. Across the 1,079
published benchmarks, ``displayType``/``aggregationType`` do not determine
either: values under ``PERCENTAGES``/``PERCENTAGE_PASSED`` range from -180 to
7.4e7, and only ~69% of them fall in [0, 1]. Numeric scores are therefore
emitted with no unit, no bounds and no ``score_type``, and Kaggle's raw scoring
configuration is kept in ``metric_config.additional_details`` so a later pass can
normalize what this adapter refuses to guess. ``aggregationType`` *does* name the
metric family, so it sets ``metric_kind`` and the metric's name and identifier;
``sortOrder`` is the sole source of ``lower_is_better``.

``modelVersionSlug`` mixes the model with the effort it was run at
(``claude-opus-4-6-default``). ``model_info.id`` is resolved against the
eval-card-registry, and the tier is reported as a generation setting; see
:func:`resolve_model_identity`.

Data sources:
- List:        POST https://www.kaggle.com/api/i/benchmarks.BenchmarkService/ListBenchmarks
- Leaderboard: GET  https://www.kaggle.com/api/v1/benchmarks/{owner}/{slug}/leaderboard
- Model ids:   POST https://evaleval-entity-registry.hf.space/api/v1/resolve

Usage:
    # Convert specific benchmark(s)
    uv run python -m every_eval_ever.adapters.kaggle.adapter \
        --benchmark cohere-labs/global-mmlu-lite-korean \
        --output-dir /tmp/eee-kaggle

    # Convert every published benchmark (~5 minutes)
    uv run python -m every_eval_ever.adapters.kaggle.adapter \
        --all --output-dir data/kaggle

    # Smoke test: read only the first 5 benchmarks from the index
    uv run python -m every_eval_ever.adapters.kaggle.adapter \
        --all --limit 5 --output-dir /tmp/eee-kaggle
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

from every_eval_ever.eval_types import (
    ConfidenceInterval,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    FetchError,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    get_developer,
    get_model_id,
    make_model_info,
    make_source_metadata,
    raw_capture,
    registry,
    require_finite_number,
    require_identity,
    save_evaluation_logs,
    save_failure_report,
)

SOURCE_NAME = "Kaggle Benchmarks"
SOURCE_ORGANIZATION = "Kaggle"
#: Namespace for every identifier this adapter mints, so a Kaggle metric or
#: evaluation never collides with another source's.
ID_NAMESPACE = "kaggle"
KAGGLE_BASE = "https://www.kaggle.com"
LIST_RPC_URL = f"{KAGGLE_BASE}/api/i/benchmarks.BenchmarkService/ListBenchmarks"
LEADERBOARD_URL = KAGGLE_BASE + "/api/v1/benchmarks/{owner}/{slug}/leaderboard"
OUTPUT_DIR = "data/kaggle"

# Server-side cap on ListBenchmarks page size.
LIST_PAGE_SIZE = 200

# Kaggle rate-limits a full sweep of ~1,100 leaderboards with 429s in bursts,
# so a retry is the difference between a green run and a handful of benchmarks
# lost every time. Only these statuses are retried; a 403 (private or deleted
# benchmark) is final and must not cost four more requests.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 2.0
#: Upper bound on a server-supplied ``Retry-After``, so one hostile header
#: cannot park the run until the job timeout kills it.
MAX_RETRY_AFTER_SECONDS = 60.0

#: ``task.version.sortOrder`` is the only published statement of which end of a
#: leaderboard is best, so it is the only source of ``lower_is_better`` and an
#: unrecognised value is a schema change rather than a direction to assume.
SORT_ORDER_LOWER_IS_BETTER = {
    "ASCENDING": True,
    "DESCENDING": False,
}

#: Kaggle folds the inference-effort tier into ``modelVersionSlug``, which is
#: not a model id anyone else uses: no vendor ships a ``-default`` model, and
#: `claude-sonnet-4-5-thinking-20250929` is one Anthropic model run with
#: thinking on. `-default` is Kaggle's own marker for "the provider's default
#: effort"; any other tier is named parenthetically in ``modelVersionName``
#: (`Claude Sonnet 4.5 (thinking)`), which is the only tier signal taken from
#: the source. Nothing is inferred from a model's name: a tier is stripped from
#: the id only when the registry has no canonical id for the slug as published
#: AND does have one for the stripped form, so `grok-4.20-0309-non-reasoning`
#: keeps the reasoning variant the registry considers canonical.
DEFAULT_EFFORT_TIER = "default"
_PARENTHETICAL = re.compile(r"\(([^)]+)\)\s*$")

#: ``task.version.aggregationType`` names the metric family the leaderboard
#: aggregates by; it says nothing about the scale (see the module docstring).
#: Unmapped types are not metric families, so they get an identifier and a
#: display name but no ``metric_kind``.
AGGREGATION_METRICS = {
    "PERCENTAGE_PASSED": ("pass_rate", "pass rate", "pass_rate"),
    "AVERAGE": ("mean", "mean", None),
}
#: Used when Kaggle publishes no aggregation type for a benchmark.
UNKNOWN_METRIC = ("score", "score", None)


class UnscoredTask(Exception):
    """Raised for a task result Kaggle reports as carrying no score."""


@dataclass(frozen=True)
class Benchmark:
    """One published benchmark and the scoring config its records need.

    ``sort_order`` is validated at construction: it is the only published
    source of ``lower_is_better``, so an unrecognised value must not fall
    through to a direction this adapter made up.
    """

    owner: str
    slug: str
    name: str
    sort_order: str
    benchmark_id: Optional[str] = None
    aggregation_type: Optional[str] = None
    display_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sort_order not in SORT_ORDER_LOWER_IS_BETTER:
            raise ValueError(
                f"unsupported Kaggle sortOrder {self.sort_order!r}; expected "
                f"one of {sorted(SORT_ORDER_LOWER_IS_BETTER)}"
            )

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.slug}"

    @property
    def url(self) -> str:
        return f"{KAGGLE_BASE}/benchmarks/{self.owner}/{self.slug}"

    @property
    def lower_is_better(self) -> bool:
        """Kaggle sorts a leaderboard by which end of it is best."""
        return SORT_ORDER_LOWER_IS_BETTER[self.sort_order]


@dataclass(frozen=True)
class ModelIdentity:
    """One leaderboard row's model, separated from how it was run."""

    id: str
    raw_slug: str
    effort_tier: Optional[str]
    reasoning: Optional[bool]
    provenance: dict

    @property
    def needs_review(self) -> bool:
        return registry.needs_review(self.provenance)


def _split_effort_tier(
    slug: str, display_name: str
) -> tuple[Optional[str], str, Optional[bool]]:
    """Return ``(tier, slug without it, reasoning)`` from the source's markers.

    Only Kaggle's own two markers are read: the literal ``-default`` suffix, and
    a parenthetical in the display name whose slug appears in the model slug.
    ``reasoning`` is left unknown unless that marker states it, because a model
    merely *named* Thinking is a model, not a setting.
    """
    if slug.endswith(f"-{DEFAULT_EFFORT_TIER}"):
        base = slug[: -len(DEFAULT_EFFORT_TIER) - 1]
        return DEFAULT_EFFORT_TIER, base, None
    match = _PARENTHETICAL.search(display_name or "")
    if match is None:
        return None, slug, None
    tier = match.group(1).strip()
    token = tier.lower().replace(" ", "-").replace("_", "-")
    if not token or f"-{token}" not in slug:
        return None, slug, None
    reasoning = None
    if "thinking" in token or "reasoning" in token:
        reasoning = not token.startswith("non-")
    return token, slug.replace(f"-{token}", "", 1), reasoning


def resolve_model_identity(
    slug: str, display_name: str, *, use_registry: bool = True
) -> ModelIdentity:
    """Canonicalize one leaderboard row's model id.

    The registry decides the identity. The slug as published is offered first,
    so a variant the registry considers a model of its own keeps that id. Only
    when the registry has no canonical id for it is the effort-stripped form
    tried, and only a canonical answer is adopted; otherwise the lexical
    fallback in ``helpers.developer`` is used and the provenance says so.
    """
    tier, base, reasoning = _split_effort_tier(slug, display_name)
    model_id, provenance = registry.resolve_model_id(slug, enabled=use_registry)
    if not registry.resolved_canonically(provenance) and base != slug:
        stripped_id, stripped_provenance = registry.resolve_model_id(
            base, enabled=use_registry
        )
        if registry.resolved_canonically(stripped_provenance):
            model_id = stripped_id
            provenance = {
                **stripped_provenance,
                "model_id_resolved_from": base,
            }
    if not registry.resolved_canonically(provenance):
        # No canonical id: keep the lexical developer/model form so the record
        # still lands in a plausible datastore directory, flagged for review.
        model_id = get_model_id(base, get_developer(base))
    return ModelIdentity(
        id=model_id,
        raw_slug=slug,
        effort_tier=tier,
        reasoning=reasoning,
        provenance=provenance,
    )


def open_session() -> requests.Session:
    """Open a session carrying an anonymous XSRF token.

    The ListBenchmarks RPC is unauthenticated but requires the XSRF
    cookie/header handshake that Kaggle hands out on any page load.
    """
    session = requests.Session()
    try:
        session.get(f"{KAGGLE_BASE}/benchmarks", timeout=60)
    except requests.RequestException as exc:
        raise FetchError(f"Kaggle XSRF handshake failed: {exc}") from exc
    xsrf = session.cookies.get("XSRF-TOKEN")
    if not xsrf:
        raise FetchError("Could not obtain XSRF-TOKEN cookie from Kaggle")
    session.headers.update(
        {
            "accept": "application/json",
            "x-xsrf-token": xsrf,
        }
    )
    return session


def _retry_delay(response: requests.Response, attempt: int) -> float:
    """Return how long to wait before retrying ``response``."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return BACKOFF_SECONDS**attempt


def request_json(
    session: requests.Session,
    url: str,
    *,
    json_body: Optional[dict] = None,
) -> Any:
    """Fetch JSON with bounded retries, snapshotting the bytes it converted.

    This adapter owns its HTTP call site rather than going through
    ``helpers.fetch_json``: the RPC needs a POST on a handshaken session, and a
    sweep of every leaderboard needs to tell a retryable 429 from a final 403.
    Raw capture is therefore explicit here.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            if json_body is None:
                response = session.get(url, timeout=60)
            else:
                response = session.post(url, json=json_body, timeout=60)
        except requests.RequestException as exc:
            raise FetchError(f"Failed to fetch {url}: {exc}") from exc
        if (
            response.status_code in RETRY_STATUSES
            and attempt < MAX_ATTEMPTS - 1
        ):
            time.sleep(_retry_delay(response, attempt))
            continue
        break
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
    raw_capture.record(
        url=response.url,
        content=response.content,
        content_type=response.headers.get("Content-Type"),
    )
    try:
        return response.json()
    except ValueError as exc:
        raise FetchError(f"Failed to parse JSON from {url}: {exc}") from exc


def _benchmark_owner(benchmark: dict) -> Optional[str]:
    """Resolve the leaderboard URL owner for a benchmark object.

    Org-owned benchmarks are addressed by the organization slug; otherwise the
    creating user's username is used.
    """
    organization = benchmark.get("organization")
    if organization and organization.get("slug"):
        return organization["slug"]
    return (benchmark.get("ownerUser") or {}).get("userName")


def make_benchmark(entry: dict) -> Benchmark:
    """Build a :class:`Benchmark` from one published index entry.

    Raises ``ValueError`` naming what is missing. Kaggle publishes a
    benchmark's owner, slug and sort order together with the rest of it, so an
    absent or unrecognised one is a schema change to report rather than a
    benchmark to leave quietly out of the corpus.
    """
    slug = entry.get("slug")
    if not slug:
        raise ValueError("index entry has no slug")
    owner = _benchmark_owner(entry)
    if not owner:
        raise ValueError(
            "index entry has neither an organization slug nor an owner username"
        )
    version = (entry.get("task") or {}).get("version") or {}
    benchmark_id = entry.get("id")
    return Benchmark(
        owner=owner,
        slug=slug,
        name=(entry.get("name") or slug).strip(),
        sort_order=version.get("sortOrder"),
        benchmark_id=None if benchmark_id is None else str(benchmark_id),
        aggregation_type=version.get("aggregationType"),
        display_type=version.get("displayType"),
    )


def list_benchmark_entries(session: requests.Session) -> Iterator[dict]:
    """Yield the raw index entry of every published benchmark.

    Raises :class:`FetchError` on a page failure, after yielding the entries of
    every page that succeeded, so a caller keeps partial progress while still
    being told the index is incomplete. Entries are yielded raw because a
    published benchmark this adapter cannot describe has to be accounted for by
    the caller rather than disappearing here.
    """
    page_token = ""
    while True:
        body = {
            "filter": {},
            "pageSize": LIST_PAGE_SIZE,
            "pageToken": page_token,
        }
        data = request_json(session, LIST_RPC_URL, json_body=body)
        # A 200 error envelope or a changed RPC schema would otherwise look
        # like an empty final page and end discovery as a clean success.
        if not isinstance(data, dict) or not isinstance(
            data.get("benchmarks"), list
        ):
            raise FetchError(
                "unexpected ListBenchmarks response shape "
                f"(pageToken={page_token!r})"
            )
        for entry in data["benchmarks"]:
            if isinstance(entry, dict) and entry.get("published"):
                yield entry
        page_token = data.get("nextPageToken") or ""
        if not page_token:
            return


def fetch_leaderboard(
    session: requests.Session, benchmark: Benchmark
) -> list[dict]:
    """Fetch one benchmark's leaderboard rows.

    A benchmark with no submissions answers with an empty ``rows`` list. A
    payload with no list-valued ``rows`` at all is a changed API or a 200 error
    envelope, so it raises :class:`FetchError` rather than being read as "no
    submissions" and leaving the benchmark silently unconverted.
    """
    url = LEADERBOARD_URL.format(owner=benchmark.owner, slug=benchmark.slug)
    data = request_json(session, url)
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise FetchError(f"unexpected leaderboard response shape from {url}")
    return rows


def _task_key(task_slug: str) -> str:
    """Extract the task's own slug from Kaggle's site path for it.

    Two shapes are published: ``/benchmarks/tasks/<owner>/<task>`` for a
    standalone task, and ``/benchmarks/<owner>/<task>/versions/<n>`` for a task
    that belongs to a versioned benchmark. Taking the last component of the
    latter yields the version number, which would give every task in a
    benchmark the same identity, so the version suffix is dropped first.
    """
    parts = [part for part in task_slug.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "versions":
        parts = parts[:-2]
    return parts[-1] if parts else ""


@dataclass(frozen=True)
class TaskIdentity:
    """One leaderboard column: the benchmark overall, or one sub-benchmark."""

    #: ``kaggle.<owner>.<benchmark>`` for the overall score, and
    #: ``kaggle.<owner>.<benchmark>.<sub>`` for a sub-benchmark, so a consumer
    #: can tell the aggregate from its parts instead of summing a file and
    #: double-counting. The owner is in the name because a Kaggle benchmark is
    #: an ``owner/slug`` resource and 15 slugs are published by more than one
    #: owner (``animalimagerecognition`` by six), so the slug alone would give
    #: unrelated benchmarks one identity.
    evaluation_name: str
    #: The subset a sub-benchmark scored, or the benchmark for the overall.
    dataset_name: str
    #: Page for that subset, or the benchmark page for the overall.
    url: str
    #: The display title Kaggle gives the column, kept as provenance.
    title: Optional[str]


def make_task_identity(task: dict, benchmark: Benchmark) -> TaskIdentity:
    """Name one task result relative to its benchmark.

    A benchmark's overall score is published as a task with an empty name and
    slug, and is the aggregate of the rest of the file.
    """
    title = (task.get("benchmarkTaskName") or "").strip() or None
    raw_slug = (task.get("benchmarkTaskSlug") or "").strip()
    stem = f"{ID_NAMESPACE}.{benchmark.owner}.{benchmark.slug}"
    key = _task_key(raw_slug)
    if not key:
        return TaskIdentity(
            evaluation_name=stem,
            dataset_name=benchmark.slug,
            url=benchmark.url,
            title=title,
        )
    return TaskIdentity(
        evaluation_name=f"{stem}.{key}",
        dataset_name=key,
        url=(
            f"{KAGGLE_BASE}{raw_slug}"
            if raw_slug.startswith("/")
            else benchmark.url
        ),
        title=title,
    )


def _confidence_interval(
    numeric: dict, score: float
) -> Optional[ConfidenceInterval]:
    """Convert Kaggle's uncertainty fields into bounds around ``score``.

    ``confidenceInterval`` is a symmetric half-width and
    ``unevenConfidenceInterval`` an asymmetric ``{plus, minus}`` pair; the two
    are mutually exclusive. Both are author-supplied and occasionally
    nonsensical (a negative half-width, or one equal to the score), so a
    half-width that is not a non-negative finite number yields no uncertainty
    rather than bounds that do not bracket the score.
    """
    lower_offset: Optional[float] = None
    upper_offset: Optional[float] = None
    if numeric.get("hasUnevenConfidenceInterval"):
        uneven = numeric.get("unevenConfidenceInterval") or {}
        lower_offset = _non_negative(uneven.get("minus"))
        upper_offset = _non_negative(uneven.get("plus"))
    elif numeric.get("hasConfidenceInterval"):
        half_width = _non_negative(numeric.get("confidenceInterval"))
        lower_offset = upper_offset = half_width
    if lower_offset is None or upper_offset is None:
        return None
    if lower_offset == 0.0 and upper_offset == 0.0:
        return None
    return ConfidenceInterval(
        lower=round(score - lower_offset, 4),
        upper=round(score + upper_offset, 4),
        method="unknown",
    )


def _non_negative(value: Any) -> Optional[float]:
    """Return ``value`` as a non-negative finite float, or ``None``."""
    try:
        number = require_finite_number(value, "confidence interval")
    except ValueError:
        return None
    return number if number >= 0.0 else None


def _metric_details(task: dict, benchmark: Benchmark) -> dict[str, str]:
    """Keep Kaggle's scoring configuration as record provenance.

    These describe how Kaggle renders and aggregates the metric. They are not
    reliable enough to derive a unit or bounds from, but they are the only
    description of the metric that exists, so they travel with the score.
    """
    details = {}
    if benchmark.aggregation_type:
        details["kaggle_aggregation_type"] = benchmark.aggregation_type
    if benchmark.display_type:
        details["kaggle_display_type"] = benchmark.display_type
    task_version = task.get("taskVersion")
    if task_version is not None:
        details["kaggle_task_version"] = str(task_version)
    return details


def make_generation_config(model: ModelIdentity) -> Optional[GenerationConfig]:
    """Record how the model was run, rather than folding it into its id.

    ``reasoning`` is the schema's typed field for whether chain-of-thought was
    used; there is no typed effort field, so the tier Kaggle names goes in
    ``additional_details`` as a string.
    """
    args = (
        # ``max_attempts`` defaults to 1 on the model; Kaggle never states it,
        # so it is cleared rather than published as a fact of the run.
        GenerationArgs(reasoning=model.reasoning, max_attempts=None)
        if model.reasoning is not None
        else None
    )
    details = (
        {"reasoning_effort": model.effort_tier} if model.effort_tier else None
    )
    if args is None and details is None:
        return None
    return GenerationConfig(generation_args=args, additional_details=details)


def build_eval_result(
    task: dict, benchmark: Benchmark, model: ModelIdentity
) -> EvaluationResult:
    """Convert one ``taskResult`` entry into an EvaluationResult.

    Raises :class:`UnscoredTask` when Kaggle reports no score for the task, and
    ``ValueError`` when a scored task cannot be converted.
    """
    result = task.get("result")
    if not isinstance(result, dict):
        raise ValueError("task result must be an object")
    result_case = result.get("resultCase")
    identity = make_task_identity(task, benchmark)
    metric_slug, metric_name, metric_kind = AGGREGATION_METRICS.get(
        benchmark.aggregation_type, UNKNOWN_METRIC
    )

    score_type: Optional[ScoreType] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    uncertainty: Optional[Uncertainty] = None

    if result_case == "numericResult":
        numeric = result.get("numericResult") or result.get(
            "numericResultNullable"
        )
        if not isinstance(numeric, dict):
            raise ValueError("numeric result payload is missing")
        score = require_finite_number(numeric.get("value"), "score")
        # Kaggle publishes no unit, scale or bounds for a numeric metric, and
        # its rendering hints do not imply one (see the module docstring), so
        # the score is reported as given and nothing about it is claimed.
        interval = _confidence_interval(numeric, score)
        if interval is not None:
            uncertainty = Uncertainty(confidence_interval=interval)
    elif result_case == "booleanResult":
        # A pass/fail task result genuinely is binary, whatever scale the
        # benchmark's own numeric metric turns out to be on.
        score_type = ScoreType.binary
        score = 1.0 if result.get("booleanResult") else 0.0
        min_score, max_score = 0.0, 1.0
    elif result_case == "none":
        raise UnscoredTask(f"{identity.evaluation_name} has no result")
    else:
        raise ValueError(f"unsupported result case {result_case!r}")

    details = _metric_details(task, benchmark)
    if identity.title:
        details["kaggle_task_title"] = identity.title
    return EvaluationResult(
        evaluation_name=identity.evaluation_name,
        evaluation_timestamp=result.get("evaluationDate"),
        source_data=SourceDataUrl(
            dataset_name=identity.dataset_name,
            source_type="url",
            url=[identity.url],
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f"{benchmark.name} - {identity.title}"
                if identity.title
                else f"{benchmark.name} overall"
            ),
            metric_id=f"{identity.evaluation_name}.{metric_slug}",
            metric_name=metric_name,
            metric_kind=metric_kind,
            lower_is_better=benchmark.lower_is_better,
            score_type=score_type,
            min_score=min_score,
            max_score=max_score,
            additional_details=details or None,
        ),
        score_details=ScoreDetails(
            score=round(score, 4), uncertainty=uncertainty
        ),
        generation_config=make_generation_config(model),
    )


def convert_benchmark(
    benchmark: Benchmark,
    rows: list[dict],
    retrieved_timestamp: str,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    use_registry: bool = True,
    identities: Optional[dict[tuple[str, str], ModelIdentity]] = None,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert one benchmark's leaderboard, retaining rejected provenance.

    ``identities`` is an optional cache shared across a sweep: the registry
    answer for a model slug does not change between benchmarks, and a full
    sweep sees ~100 distinct models across ~16,000 rows.
    """
    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    if identities is None:
        identities = {}

    # The source is the platform that published the leaderboard. The uploader is
    # a fact about the benchmark, not the reporting organization, so it stays in
    # additional_details rather than standing in for Kaggle.
    source_details = {
        "platform": ID_NAMESPACE,
        "benchmark_name": benchmark.name,
        "benchmark_owner": benchmark.owner,
        "benchmark_slug": benchmark.slug,
        "benchmark_url": benchmark.url,
    }
    if benchmark.benchmark_id is not None:
        source_details["benchmark_id"] = benchmark.benchmark_id

    for row_index, row in enumerate(rows):
        row_ref = f"{benchmark.ref} row {row_index}"
        failures_before = len(failures)
        try:
            model_slug = require_identity(
                row.get("modelVersionSlug"), "model version slug"
            )
            display_name = (row.get("modelVersionName") or "").strip()
            cache_key = (model_slug, display_name)
            model = identities.get(cache_key)
            if model is None:
                model = resolve_model_identity(
                    model_slug, display_name, use_registry=use_registry
                )
                identities[cache_key] = model

            task_results = row.get("taskResults")
            if not isinstance(task_results, list):
                raise ValueError("taskResults must be a list")

            eval_results: list[EvaluationResult] = []
            for task_index, task in enumerate(task_results):
                task_ref = f"{row_ref} task {task_index}"
                try:
                    eval_results.append(
                        build_eval_result(task, benchmark, model)
                    )
                except UnscoredTask as exc:
                    exclusions.append(
                        SourceRecordExclusion(
                            source_ref=task_ref,
                            reason=str(exc),
                        )
                    )
                except ValueError as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=task_ref,
                            reason=str(exc),
                            source_record=task,
                        )
                    )

            if not eval_results:
                if len(failures) > failures_before:
                    # Every task was scored but none converted: a failure, not
                    # a model Kaggle never ran.
                    raise ValueError("no task result could be converted")
                raise UnscoredTask("no scored task results")

            model_id = require_identity(model.id, "model id")
            if "/" not in model_id:
                raise ValueError(
                    f"model id must be developer/model: {model_id!r}"
                )
            path_developer, path_model = model_id.split("/", 1)
            # Neither the registry nor the lexical table could name a
            # publisher, so the record has no datastore directory to live in.
            require_identity(
                path_developer, f"developer of {model.raw_slug!r}"
            )
            # A curated set, not the whole provenance dict: a reader needs to
            # know where the id came from and whether to trust it, and every
            # record carrying `model_id_created_new: false` is noise.
            model_details = {
                "kaggle_model_version_slug": model.raw_slug,
                "model_id_resolution": str(
                    model.provenance.get("model_id_resolution")
                ),
            }
            for key in (
                "model_id_resolution_strategy",
                "model_id_review_status",
                "model_id_resolved_from",
            ):
                value = model.provenance.get(key)
                if value is not None:
                    model_details[key] = str(value)
            confidence = model.provenance.get("model_id_resolution_confidence")
            if isinstance(confidence, (int, float)):
                model_details["model_id_resolution_confidence"] = str(
                    round(float(confidence), 4)
                )
            if display_name and display_name != model.raw_slug:
                model_details["display_name"] = display_name
            if model.needs_review:
                model_details["model_id_needs_review"] = "true"
            model_info = make_model_info(
                model_name=model_id,
                developer=path_developer,
                additional_details=model_details,
            )

            # Keyed by the slug as published, not the canonical id: two rows of
            # one benchmark can resolve to the same model run at different
            # effort (`claude-sonnet-4-5[-thinking]-20250929`), and they are
            # distinct evaluations.
            evaluation_id = (
                f"{benchmark.owner}/{benchmark.slug}/"
                f"{model.raw_slug}/{retrieved_timestamp}"
            )
            eval_log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_metadata=make_source_metadata(
                    source_name=SOURCE_NAME,
                    organization_name=SOURCE_ORGANIZATION,
                    organization_url=KAGGLE_BASE,
                    evaluator_relationship=EvaluatorRelationship.third_party,
                    additional_details=source_details,
                ),
                # Kaggle runs these from a notebook definition and names no
                # harness, so there is no library to report.
                eval_library=EvalLibrary(
                    name="unknown",
                    version="unknown",
                    additional_details={"url": benchmark.url},
                ),
                model_info=model_info,
                evaluation_results=eval_results,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=output_dir,
                    developer=path_developer,
                    model_name=path_model,
                )
            )
        except UnscoredTask as exc:
            exclusions.append(
                SourceRecordExclusion(source_ref=row_ref, reason=str(exc))
            )
        # ValueError only: every rejection this loop intends is a ValueError
        # (`require_identity`, `require_finite_number`, pydantic validation,
        # and the raises above). Catching TypeError/AttributeError too would
        # turn a bug in this adapter into 16,000 quiet per-row failures instead
        # of a crash naming the line.
        except ValueError as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=(
                        f"no record written: {exc}"
                        if len(failures) > failures_before
                        else str(exc)
                    ),
                    source_record=row,
                )
            )

    return SourceConversionResult(
        source_name=f"{SOURCE_NAME} {benchmark.ref}",
        total_records=len(rows),
        records=outputs,
        failures=failures,
        exclusions=exclusions,
    )


def _parse_refs(specs: list[str] | None, flag: str) -> list[tuple[str, str]]:
    """Parse repeated ``owner/slug`` CLI values."""
    refs = []
    for spec in specs or []:
        if "/" not in spec:
            raise SystemExit(f"{flag} expects owner/slug, got: {spec!r}")
        owner, slug = spec.split("/", 1)
        refs.append((owner, slug))
    return refs


def resolve_targets(
    session: requests.Session, args: argparse.Namespace
) -> tuple[list[Benchmark], list[SourceRecordFailure]]:
    """Select the benchmarks to convert from the published index.

    ``--all`` means every published benchmark; anything the caller does not
    want in a given run is named by ``--exclude``, so which benchmarks a
    scheduled sweep skips is a declared argument in the catalog rather than a
    rule hidden in this module.

    Returns the targets and any failure that made the index incomplete — a
    truncated read, or a published benchmark this adapter cannot address or
    score. The caller keeps the partial set but must not report an incomplete
    corpus as a clean success.
    """
    requested = _parse_refs(args.benchmark, "--benchmark")
    excluded = set(_parse_refs(args.exclude, "--exclude"))

    print("Reading the published benchmark index...")
    index: dict[tuple[str, str], Benchmark] = {}
    failures: list[SourceRecordFailure] = []
    try:
        for entry in list_benchmark_entries(session):
            try:
                benchmark = make_benchmark(entry)
            except ValueError as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=(
                            "index entry "
                            f"{entry.get('slug') or entry.get('id')!r}"
                        ),
                        reason=str(exc),
                    )
                )
                continue
            index[(benchmark.owner, benchmark.slug)] = benchmark
            if args.all and args.limit and len(index) >= args.limit:
                break
    except FetchError as exc:
        reason = f"benchmark index truncated: {exc}"
        print(f"  ! {reason}")
        failures.append(
            SourceRecordFailure(source_ref="ListBenchmarks", reason=reason)
        )
    print(f"  {len(index)} benchmark(s) in the index")

    targets: dict[tuple[str, str], Benchmark] = {}
    for key in requested:
        benchmark = index.get(key)
        if benchmark is None:
            raise SystemExit(
                f"--benchmark {key[0]}/{key[1]} is not a published Kaggle "
                "benchmark (or the index could not be read)"
            )
        targets[key] = benchmark
    if args.all:
        for key, benchmark in index.items():
            if key in excluded:
                print(f"  excluding {benchmark.ref}")
                continue
            targets.setdefault(key, benchmark)
    return list(targets.values()), failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Kaggle Community Benchmarks leaderboards to EEE records."
        )
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        metavar="OWNER/SLUG",
        help=(
            "Specific benchmark to convert (repeatable), e.g. "
            "cohere-labs/global-mmlu-lite-korean"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="OWNER/SLUG",
        default=None,
        help=(
            "Benchmark to leave out of --all (repeatable). Used by the "
            "ingestion catalog to declare benchmarks another adapter owns."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every published benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "With --all, read only this many benchmarks from the index "
            "(smoke testing)."
        ),
    )
    parser.add_argument(
        "--no-registry-resolve",
        action="store_true",
        help=(
            "Skip eval-card-registry resolution of model ids and fall back to "
            "the lexical developer/model form, for an offline or deterministic "
            "run. Every record then says so in model_info.additional_details."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f"Datastore collection directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)
    if not args.benchmark and not args.all:
        parser.error("provide --benchmark OWNER/SLUG and/or --all")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    retrieved_timestamp = str(time.time())

    session = open_session()
    targets, failures = resolve_targets(session, args)

    print("=" * 60)
    print(f"Converting {len(targets)} benchmark(s) -> {args.output_dir}")
    identities: dict[tuple[str, str], ModelIdentity] = {}
    written: list[Path] = []
    total_rows = 0
    exclusions: list[SourceRecordExclusion] = []

    for benchmark in targets:
        print(f"\n[{benchmark.ref}] {benchmark.name}")
        try:
            rows = fetch_leaderboard(session, benchmark)
        except FetchError as exc:
            print(f"  ! leaderboard fetch failed: {exc}")
            failures.append(
                SourceRecordFailure(
                    source_ref=benchmark.ref,
                    reason=f"leaderboard fetch failed: {exc}",
                )
            )
            continue
        result = convert_benchmark(
            benchmark,
            rows,
            retrieved_timestamp,
            args.output_dir,
            use_registry=not args.no_registry_resolve,
            identities=identities,
        )
        total_rows += result.total_records
        failures.extend(result.failures)
        exclusions.extend(result.exclusions)
        # Published per benchmark rather than once at the end, so a sweep
        # stopped by the job timeout keeps the benchmarks it had already
        # converted. A write that fails is left to crash: the benchmarks
        # before it are already on disk, and a filesystem or routing fault is
        # not a fact about the source to file a per-record failure over.
        written.extend(save_evaluation_logs(result.records))
        print(f"  -> {len(result.records)} of {result.total_records} model(s)")

    result = SourceConversionResult[Path](
        source_name=SOURCE_NAME,
        total_records=total_rows,
        records=written,
        failures=failures,
        exclusions=exclusions,
    )

    print("\n" + "=" * 60)
    print(
        f"Saved {len(written)} record(s) from {total_rows} leaderboard row(s) "
        f"across {len(targets)} benchmark(s)"
    )
    unverified = sum(1 for model in identities.values() if model.needs_review)
    print(
        f"Model ids: {len(identities)} distinct, {unverified} unverified "
        "(model_info.additional_details.model_id_needs_review)"
    )
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result, default_failure_report_path(args.output_dir)
        )
        print(f"Failure report: {report_path}")
    print("=" * 60)
    result.raise_if_incomplete()
    return len(written)


if __name__ == "__main__":
    main()
