"""Unit tests for the generalized Kaggle Benchmarks adapter.

Every fixture mirrors a shape observed in Kaggle's live responses: the two
`benchmarkTaskSlug` spellings, the unscored `resultCase`, the symmetric and
asymmetric confidence-interval fields, the effort tier folded into
`modelVersionSlug`, and the author-supplied values that are not on any scale the
API describes.

No test touches the network. `adapter.request_json` is the single HTTP seam for
Kaggle, and `stub_registry` replaces the eval-card-registry resolver for every
test in the module.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from every_eval_ever.adapters.kaggle import adapter
from every_eval_ever.eval_types import EvaluationLog, ScoreType
from every_eval_ever.helpers import FetchError, SourceRecordsError

BENCH = adapter.Benchmark(
    owner="owner",
    slug="my-bench",
    name="My Bench",
    sort_order="DESCENDING",
    benchmark_id="42",
    aggregation_type="PERCENTAGE_PASSED",
    display_type="PERCENTAGES",
)

MODEL = adapter.ModelIdentity(
    id="google/gemma-3-27b-it",
    raw_slug="gemma-3-27b-it",
    effort_tier=None,
    reasoning=None,
    provenance={"model_id_resolution": "offline"},
)

#: Canonical ids the stubbed resolver knows, taken verbatim from the live
#: resolver's answers for these Kaggle slugs.
FAKE_CANONICAL = {
    "gemma-3-27b-it": ("google/gemma-3-27b-it", "reviewed", 1.0),
    "claude-opus-5": ("anthropic/claude-opus-5", "reviewed", 1.0),
    # `-thinking-` is a setting: the base resolves, the published slug does not.
    "claude-sonnet-4-5-20250929": (
        "anthropic/claude-sonnet-4.5-20250929",
        "reviewed",
        1.0,
    ),
    # A reasoning variant the registry considers a model of its own: the slug as
    # published resolves, so nothing may be stripped from it.
    "grok-4.20-0309-non-reasoning": (
        "xai/grok-4.20-non-reasoning",
        "reviewed",
        1.0,
    ),
    "grok-4.20-0309": ("xai/grok-4-20-0309", "draft", 0.95),
}


@pytest.fixture(autouse=True)
def stub_registry(monkeypatch):
    """Answer every registry lookup locally, so no test reaches the network.

    Mirrors the contract in `helpers/registry.py`: only a canonical answer is
    `model_id_resolution: registry`, and anything the registry does not know is
    `unresolved`. `tests/test_registry_resolver.py` pins that contract against
    the real implementation.
    """

    def fake_resolve(raw_value, *, enabled=True, timeout=15.0):
        if not enabled:
            return raw_value, {"model_id_resolution": "offline"}
        known = FAKE_CANONICAL.get(raw_value)
        if known is None:
            return raw_value, {
                "model_id_resolution": "unresolved",
                "model_id_resolution_strategy": "no_match",
            }
        canonical, status, confidence = known
        return canonical, {
            "model_id_resolution": "registry",
            "model_id_resolution_strategy": "exact",
            "model_id_resolution_confidence": confidence,
            "model_id_created_new": False,
            "model_id_review_status": status,
        }

    monkeypatch.setattr(adapter.registry, "resolve_model_id", fake_resolve)
    return fake_resolve


# ---------------------------------------------------------------------------
# Fixture builders mirroring the shape of Kaggle's leaderboard API.
# ---------------------------------------------------------------------------
def numeric_task(
    name,
    value,
    *,
    ci=None,
    uneven=None,
    date=None,
    slug="",
    task_version=None,
):
    numeric = {
        "value": value,
        "hasConfidenceInterval": ci is not None,
        "confidenceInterval": 0.0 if ci is None else ci,
        "hasUnevenConfidenceInterval": uneven is not None,
    }
    if uneven is not None:
        numeric["unevenConfidenceInterval"] = uneven
    result = {
        "resultCase": "numericResult",
        "hasNumericResult": True,
        "numericResult": numeric,
        "numericResultNullable": numeric,
        "hasBooleanResult": False,
        "booleanResult": False,
    }
    if date is not None:
        result["evaluationDate"] = date
    task = {
        "benchmarkTaskName": name,
        "benchmarkTaskSlug": slug,
        "result": result,
    }
    if task_version is not None:
        task["taskVersion"] = task_version
    return task


def boolean_task(name, passed, *, slug=""):
    return {
        "benchmarkTaskName": name,
        "benchmarkTaskSlug": slug,
        "result": {
            "resultCase": "booleanResult",
            "hasNumericResult": False,
            "hasBooleanResult": True,
            "booleanResult": passed,
            "booleanResultNullable": passed,
        },
    }


def unscored_task(name, *, slug=""):
    return {
        "benchmarkTaskName": name,
        "benchmarkTaskSlug": slug,
        "result": {
            "resultCase": "none",
            "hasNumericResult": False,
            "hasBooleanResult": False,
            "booleanResult": False,
        },
    }


def index_entry(
    owner, slug, *, published=True, sort_order="DESCENDING", org=True
):
    version = {
        "sortOrder": sort_order,
        "aggregationType": "PERCENTAGE_PASSED",
        "displayType": "PERCENTAGES",
    }
    entry = {
        "id": 7,
        "name": f"  {slug} bench  ",
        "slug": slug,
        "published": published,
        "task": {"version": version},
    }
    if org:
        entry["organization"] = {"slug": owner}
    else:
        entry["ownerUser"] = {"userName": owner}
    return entry


def build(task, benchmark=BENCH, model=MODEL):
    return adapter.build_eval_result(task, benchmark, model)


# ---------------------------------------------------------------------------
# _task_key: Kaggle publishes two site-path spellings for a task
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "slug,expected",
    [
        (
            "/benchmarks/tasks/aanaakaliil/sandbox-in-single",
            "sandbox-in-single",
        ),
        # The versioned form must not collapse to its version number, which
        # would give every task in a benchmark the same identity.
        (
            "/benchmarks/cohere-labs/global-mmlu-lite-chinese/versions/1",
            "global-mmlu-lite-chinese",
        ),
        ("/benchmarks/google/facts-grounding/versions/2", "facts-grounding"),
        ("", ""),
    ],
)
def test_task_key_reads_both_published_slug_shapes(slug, expected):
    assert adapter._task_key(slug) == expected


# ---------------------------------------------------------------------------
# Naming: the aggregate must be distinguishable from its sub-benchmarks
# ---------------------------------------------------------------------------
def test_unnamed_unslugged_task_is_the_benchmark_aggregate():
    identity = adapter.make_task_identity(numeric_task("", 0.5), BENCH)
    assert identity.evaluation_name == "kaggle.owner.my-bench"
    assert identity.dataset_name == "my-bench"
    assert identity.url == BENCH.url


def test_a_sub_benchmark_is_named_under_its_benchmark():
    identity = adapter.make_task_identity(
        numeric_task("Emotion Inference", 0.5, slug="/benchmarks/tasks/o/emo"),
        BENCH,
    )
    assert identity.evaluation_name == "kaggle.owner.my-bench.emo"
    # The subset it scored, not the parent benchmark.
    assert identity.dataset_name == "emo"
    assert identity.url == "https://www.kaggle.com/benchmarks/tasks/o/emo"
    assert identity.title == "Emotion Inference"


def test_versioned_sibling_tasks_get_distinct_identities():
    names = {
        adapter.make_task_identity(
            numeric_task(title, 0.5, slug=f"/benchmarks/google/{key}/versions/2"),
            BENCH,
        ).evaluation_name
        for title, key in (
            ("Public", "facts-public"),
            ("Private", "facts-private"),
        )
    }
    assert names == {
        "kaggle.owner.my-bench.facts-public",
        "kaggle.owner.my-bench.facts-private",
    }


def test_metric_id_and_name_describe_the_metric_not_the_eval():
    result = build(
        numeric_task("Emotion Inference", 0.5, slug="/benchmarks/tasks/o/emo")
    )
    metric = result.metric_config
    assert metric.metric_id == "kaggle.owner.my-bench.emo.pass_rate"
    assert metric.metric_name == "pass rate"
    assert metric.metric_kind == "pass_rate"
    # The column's title is provenance, not the metric's name.
    assert metric.additional_details["kaggle_task_title"] == "Emotion Inference"


def test_metric_family_is_consistent_across_a_record():
    aggregate = build(numeric_task("", 0.9))
    sub = build(boolean_task("Sub", True, slug="/benchmarks/tasks/o/sub"))
    assert aggregate.metric_config.metric_kind == "pass_rate"
    assert sub.metric_config.metric_kind == "pass_rate"


def test_a_benchmark_with_no_aggregation_type_gets_no_metric_kind():
    bench = adapter.Benchmark(
        owner="o", slug="s", name="S", sort_order="DESCENDING"
    )
    result = build(numeric_task("", 0.5), benchmark=bench)
    assert result.metric_config.metric_kind is None
    assert result.metric_config.metric_name == "score"
    assert result.metric_config.metric_id == "kaggle.o.s.score"


# ---------------------------------------------------------------------------
# build_eval_result: what may and may not be claimed about a score
# ---------------------------------------------------------------------------
def test_numeric_score_claims_no_unit_scale_or_bounds():
    # Kaggle publishes none of them and its rendering hints do not imply one,
    # so a value in [0, 1] must not be dressed up as a bounded proportion.
    result = build(numeric_task("acc", 0.87654))
    assert result.score_details.score == 0.8765
    assert result.metric_config.score_type is None
    assert result.metric_config.min_score is None
    assert result.metric_config.max_score is None
    assert result.metric_config.metric_unit is None


def test_numeric_score_is_reported_as_published():
    # A "percentage passed" of 12,000 is what the author submitted; the record
    # says so rather than rescaling or rejecting it.
    result = build(numeric_task("count", 12000.0))
    assert result.score_details.score == 12000.0
    assert result.metric_config.score_type is None


def test_kaggle_scoring_config_travels_as_provenance():
    result = build(numeric_task("acc", 0.5, task_version=3))
    assert result.metric_config.additional_details == {
        "kaggle_aggregation_type": "PERCENTAGE_PASSED",
        "kaggle_display_type": "PERCENTAGES",
        "kaggle_task_version": "3",
        "kaggle_task_title": "acc",
    }


def test_boolean_results_are_binary_on_zero_to_one():
    passed = build(boolean_task("solved", True))
    failed = build(boolean_task("solved", False))
    assert passed.metric_config.score_type == ScoreType.binary
    assert (
        passed.metric_config.min_score,
        passed.metric_config.max_score,
    ) == (0.0, 1.0)
    assert passed.score_details.score == 1.0
    assert failed.score_details.score == 0.0


def test_unscored_task_is_excluded_not_failed():
    with pytest.raises(adapter.UnscoredTask):
        build(unscored_task("unrun"))


def test_unsupported_result_case_is_a_failure():
    task = numeric_task("acc", 0.5)
    task["result"]["resultCase"] = "customResult"
    with pytest.raises(ValueError, match="unsupported result case"):
        build(task)


def test_missing_numeric_value_is_a_failure():
    with pytest.raises(ValueError, match="score"):
        build(numeric_task("acc", None))


def test_descending_leaderboard_means_higher_is_better():
    assert build(numeric_task("acc", 0.5)).metric_config.lower_is_better is False


def test_ascending_leaderboard_means_lower_is_better():
    bench = adapter.Benchmark(
        owner="o", slug="s", name="S", sort_order="ASCENDING"
    )
    result = build(numeric_task("latency", 0.3), benchmark=bench)
    assert result.metric_config.lower_is_better is True


# ---------------------------------------------------------------------------
# sortOrder is the only source of direction, so it is validated, not assumed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sort_order", ["", None, "UNSPECIFIED", "ascending"])
def test_an_unrecognised_sort_order_cannot_build_a_benchmark(sort_order):
    with pytest.raises(ValueError, match="sortOrder"):
        adapter.Benchmark(
            owner="o", slug="s", name="S", sort_order=sort_order
        )


# ---------------------------------------------------------------------------
# build_eval_result: timestamps and uncertainty
# ---------------------------------------------------------------------------
def test_evaluation_date_is_captured_as_timestamp():
    result = build(numeric_task("acc", 0.5, date="2026-06-20T16:40:52.0Z"))
    assert result.evaluation_timestamp == "2026-06-20T16:40:52.0Z"


def test_missing_evaluation_date_leaves_timestamp_unset():
    assert build(numeric_task("acc", 0.5)).evaluation_timestamp is None


def test_symmetric_confidence_interval_brackets_the_score():
    result = build(numeric_task("acc", 0.87, ci=0.02))
    interval = result.score_details.uncertainty.confidence_interval
    assert (interval.lower, interval.upper) == (0.85, 0.89)


def test_uneven_confidence_interval_uses_both_offsets():
    result = build(
        numeric_task("elo", 1137.17, uneven={"plus": 20.01, "minus": 23.42})
    )
    interval = result.score_details.uncertainty.confidence_interval
    assert (interval.lower, interval.upper) == (1113.75, 1157.18)


@pytest.mark.parametrize("ci", [-0.83, "wide", 0.0, None])
def test_an_unusable_half_width_yields_no_uncertainty(ci):
    # A negative radius cannot bracket the score, and Kaggle does publish them.
    assert build(numeric_task("acc", 0.5, ci=ci)).score_details.uncertainty is None


# ---------------------------------------------------------------------------
# Model identity: the registry decides, nothing is inferred from a name
# ---------------------------------------------------------------------------
def test_a_slug_the_registry_knows_keeps_its_canonical_id():
    model = adapter.resolve_model_identity("gemma-3-27b-it", "Gemma 3 27B")
    assert model.id == "google/gemma-3-27b-it"
    assert model.effort_tier is None
    assert model.reasoning is None
    assert model.needs_review is False


def test_the_default_effort_marker_leaves_the_id_and_is_recorded():
    model = adapter.resolve_model_identity(
        "claude-opus-5-default", "Claude Opus 5"
    )
    assert model.id == "anthropic/claude-opus-5"
    assert model.effort_tier == "default"
    # "default effort" says nothing about whether reasoning was used.
    assert model.reasoning is None
    assert model.provenance["model_id_resolved_from"] == "claude-opus-5"


def test_a_parenthetical_tier_is_stripped_only_to_reach_a_canonical_id():
    model = adapter.resolve_model_identity(
        "claude-sonnet-4-5-thinking-20250929", "Claude Sonnet 4.5 (thinking)"
    )
    assert model.id == "anthropic/claude-sonnet-4.5-20250929"
    assert model.effort_tier == "thinking"
    assert model.reasoning is True


def test_a_reasoning_variant_the_registry_owns_keeps_its_own_id():
    # The published slug resolves, so the tier is never stripped from it — the
    # registry considers this a model, and the weaker draft id for the stripped
    # form must not win.
    model = adapter.resolve_model_identity(
        "grok-4.20-0309-non-reasoning", "Grok 4.20 (Non-Reasoning)"
    )
    assert model.id == "xai/grok-4.20-non-reasoning"
    assert model.reasoning is False
    assert "model_id_resolved_from" not in model.provenance


def test_a_thinking_model_is_not_read_as_a_thinking_setting():
    # No parenthetical marker, so nothing is claimed about the run; the id
    # carries the variant because the registry says it is one.
    model = adapter.resolve_model_identity(
        "qwen3-next-80b-a3b-thinking", "Qwen 3 Next 80B Thinking"
    )
    assert model.effort_tier is None
    assert model.reasoning is None


def test_an_unresolvable_model_falls_back_and_is_flagged():
    model = adapter.resolve_model_identity("gpt-5.9-unreleased", "GPT-5.9")
    assert model.id == "openai/gpt-5.9-unreleased"
    assert model.provenance["model_id_resolution_strategy"] == "no_match"
    assert model.needs_review is True


def test_resolution_can_be_turned_off_for_an_offline_run():
    model = adapter.resolve_model_identity(
        "gemma-3-27b-it", "Gemma 3 27B", use_registry=False
    )
    assert model.id == "google/gemma-3-27b-it"
    assert model.provenance == {"model_id_resolution": "offline"}


# ---------------------------------------------------------------------------
# generation_config: how the model was run, not part of what it is
# ---------------------------------------------------------------------------
def test_reasoning_uses_the_typed_field_and_the_tier_a_string():
    model = adapter.resolve_model_identity(
        "claude-sonnet-4-5-thinking-20250929", "Claude Sonnet 4.5 (thinking)"
    )
    config = adapter.make_generation_config(model)
    assert config.generation_args.reasoning is True
    assert config.additional_details == {"reasoning_effort": "thinking"}
    # Kaggle never states an attempt count, so none is published.
    assert config.generation_args.max_attempts is None


def test_a_tier_without_a_reasoning_claim_sets_no_generation_args():
    model = adapter.resolve_model_identity(
        "claude-opus-5-default", "Claude Opus 5"
    )
    config = adapter.make_generation_config(model)
    assert config.generation_args is None
    assert config.additional_details == {"reasoning_effort": "default"}


def test_a_plain_model_gets_no_generation_config():
    assert adapter.make_generation_config(MODEL) is None


# ---------------------------------------------------------------------------
# request_json: retry policy (no network)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.url = "https://example.invalid/x"
        self.content = json.dumps(payload).encode() if payload else b""

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self._responses.pop(0)

    def post(self, url, **kwargs):
        return self.get(url, **kwargs)


def test_rate_limited_request_is_retried(monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda seconds: None)
    session = _FakeSession(
        [_FakeResponse(429), _FakeResponse(429), _FakeResponse(200, {"ok": 1})]
    )
    assert adapter.request_json(session, "https://example.invalid/x") == {
        "ok": 1
    }
    assert session.calls == 3


def test_forbidden_request_is_not_retried(monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda seconds: None)
    session = _FakeSession([_FakeResponse(403)])
    with pytest.raises(FetchError):
        adapter.request_json(session, "https://example.invalid/x")
    assert session.calls == 1


def test_retry_after_header_is_capped(monkeypatch):
    slept = []
    monkeypatch.setattr(adapter.time, "sleep", slept.append)
    session = _FakeSession(
        [
            _FakeResponse(429, headers={"Retry-After": "99999"}),
            _FakeResponse(200, {"ok": 1}),
        ]
    )
    adapter.request_json(session, "https://example.invalid/x")
    assert slept == [adapter.MAX_RETRY_AFTER_SECONDS]


# ---------------------------------------------------------------------------
# list_benchmark_entries / make_benchmark
# ---------------------------------------------------------------------------
def test_index_paginates_and_skips_unpublished(monkeypatch):
    pages = {
        "": {
            "benchmarks": [
                index_entry("cohere-labs", "one"),
                index_entry("someone", "draft", published=False),
                index_entry("someone", "two", org=False),
            ],
            "nextPageToken": "page-2",
        },
        "page-2": {"benchmarks": [index_entry("org", "three")]},
    }

    def fake_request(session, url, *, json_body=None):
        return pages[json_body["pageToken"]]

    monkeypatch.setattr(adapter, "request_json", fake_request)
    entries = list(adapter.list_benchmark_entries(object()))
    assert [entry["slug"] for entry in entries] == ["one", "two", "three"]


def test_index_entry_carries_the_published_scoring_config():
    benchmark = adapter.make_benchmark(index_entry("cohere-labs", "one"))
    assert benchmark.ref == "cohere-labs/one"
    assert benchmark.name == "one bench"
    assert benchmark.sort_order == "DESCENDING"
    assert benchmark.display_type == "PERCENTAGES"
    assert benchmark.aggregation_type == "PERCENTAGE_PASSED"
    assert benchmark.benchmark_id == "7"


def test_user_owned_benchmark_is_addressed_by_username():
    benchmark = adapter.make_benchmark(
        index_entry("someone", "two", org=False)
    )
    assert benchmark.owner == "someone"


@pytest.mark.parametrize(
    "entry,reason",
    [
        # Each defect on its own, so a fix to one cannot mask the others.
        ({"id": 1, "task": {"version": {"sortOrder": "DESCENDING"}}}, "slug"),
        (
            {
                "id": 1,
                "slug": "no-owner",
                "task": {"version": {"sortOrder": "DESCENDING"}},
            },
            "owner username",
        ),
        (index_entry("org", "no-order", sort_order=None), "sortOrder"),
        (index_entry("org", "odd-order", sort_order="SIDEWAYS"), "sortOrder"),
    ],
)
def test_an_undescribable_index_entry_says_what_is_missing(entry, reason):
    with pytest.raises(ValueError, match=reason):
        adapter.make_benchmark(entry)


def test_malformed_index_page_is_a_fetch_error(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "request_json",
        lambda session, url, *, json_body=None: {"error": "nope"},
    )
    with pytest.raises(FetchError, match="ListBenchmarks"):
        list(adapter.list_benchmark_entries(object()))


# ---------------------------------------------------------------------------
# fetch_leaderboard: failure vs empty
# ---------------------------------------------------------------------------
def test_leaderboard_without_submissions_is_empty_not_a_failure(monkeypatch):
    monkeypatch.setattr(
        adapter, "request_json", lambda session, url, **kw: {"rows": []}
    )
    assert adapter.fetch_leaderboard(object(), BENCH) == []


@pytest.mark.parametrize("payload", [{"rows": "nope"}, {"code": 429}, []])
def test_leaderboard_of_wrong_shape_is_a_fetch_error(monkeypatch, payload):
    # A payload with no list-valued `rows` must never read as "no submissions",
    # which would leave the benchmark unconverted and the run green.
    monkeypatch.setattr(
        adapter, "request_json", lambda session, url, **kw: payload
    )
    with pytest.raises(FetchError, match="unexpected leaderboard"):
        adapter.fetch_leaderboard(object(), BENCH)


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = {
        "benchmark": None,
        "exclude": None,
        "all": False,
        "limit": None,
        "no_registry_resolve": True,
        "output_dir": Path("/tmp/unused"),
    }
    base.update(overrides)
    return Namespace(**base)


def _stub_index(monkeypatch, entries):
    monkeypatch.setattr(
        adapter, "list_benchmark_entries", lambda session: iter(entries)
    )


def test_explicit_and_discovered_targets_are_deduplicated(monkeypatch):
    _stub_index(
        monkeypatch, [index_entry("o", "one"), index_entry("o", "two")]
    )
    targets, failures = adapter.resolve_targets(
        object(), _args(benchmark=["o/one", "o/one"], all=True)
    )
    assert [target.ref for target in targets] == ["o/one", "o/two"]
    assert failures == []


def test_explicit_target_carries_the_published_scoring_config(monkeypatch):
    # The leaderboard endpoint omits it, so a targeted run must describe a
    # benchmark exactly as an --all run does.
    _stub_index(
        monkeypatch, [index_entry("o", "one", sort_order="ASCENDING")]
    )
    targets, _ = adapter.resolve_targets(object(), _args(benchmark=["o/one"]))
    assert targets[0].sort_order == "ASCENDING"
    assert targets[0].display_type == "PERCENTAGES"


def test_unknown_explicit_target_is_rejected(monkeypatch):
    _stub_index(monkeypatch, [])
    with pytest.raises(SystemExit):
        adapter.resolve_targets(object(), _args(benchmark=["o/nope"]))


@pytest.mark.parametrize("flag", ["benchmark", "exclude"])
def test_a_ref_without_a_slash_is_rejected(monkeypatch, flag):
    _stub_index(monkeypatch, [])
    with pytest.raises(SystemExit):
        adapter.resolve_targets(
            object(), _args(all=True, **{flag: ["justaslug"]})
        )


def test_all_means_all_unless_a_benchmark_is_excluded(monkeypatch):
    _stub_index(
        monkeypatch, [index_entry("o", "owned"), index_entry("o", "other")]
    )
    everything, _ = adapter.resolve_targets(object(), _args(all=True))
    assert [target.ref for target in everything] == ["o/owned", "o/other"]
    trimmed, failures = adapter.resolve_targets(
        object(), _args(all=True, exclude=["o/owned"])
    )
    assert [target.ref for target in trimmed] == ["o/other"]
    assert failures == []


def test_an_excluded_benchmark_can_still_be_named_explicitly(monkeypatch):
    _stub_index(monkeypatch, [index_entry("o", "owned")])
    targets, _ = adapter.resolve_targets(
        object(),
        _args(benchmark=["o/owned"], exclude=["o/owned"], all=True),
    )
    assert [target.ref for target in targets] == ["o/owned"]


def test_truncated_index_keeps_partial_targets_and_reports_it(monkeypatch):
    def partial(session):
        yield index_entry("o", "one")
        raise FetchError("page 2 exploded")

    monkeypatch.setattr(adapter, "list_benchmark_entries", partial)
    targets, failures = adapter.resolve_targets(object(), _args(all=True))
    assert [target.ref for target in targets] == ["o/one"]
    assert [failure.source_ref for failure in failures] == ["ListBenchmarks"]
    assert "page 2 exploded" in failures[0].reason


def test_an_undescribable_published_benchmark_is_a_failure(monkeypatch):
    # It must not silently drop out of the corpus while the run stays green.
    _stub_index(
        monkeypatch,
        [
            index_entry("o", "one"),
            index_entry("o", "broken", sort_order=None),
        ],
    )
    targets, failures = adapter.resolve_targets(object(), _args(all=True))
    assert [target.ref for target in targets] == ["o/one"]
    assert [failure.source_ref for failure in failures] == [
        "index entry 'broken'"
    ]
    assert "sortOrder" in failures[0].reason


def test_limit_caps_the_sweep(monkeypatch):
    _stub_index(
        monkeypatch, [index_entry("o", f"b{index}") for index in range(5)]
    )
    targets, _ = adapter.resolve_targets(object(), _args(all=True, limit=2))
    assert len(targets) == 2


# ---------------------------------------------------------------------------
# convert_benchmark: records, exclusions, failures
# ---------------------------------------------------------------------------
def sample_rows():
    return [
        {
            "modelVersionName": "Gemma 3 27B",
            "modelVersionSlug": "gemma-3-27b-it",
            "taskResults": [
                numeric_task("", 0.9123, ci=0.01, date="2026-03-20T17:14:42Z"),
                boolean_task(
                    "Emotion Inference",
                    True,
                    slug="/benchmarks/tasks/owner/emotion-inference",
                ),
                unscored_task("Not Run"),
            ],
        },
        {
            "modelVersionName": "Nothing Scored",
            "modelVersionSlug": "claude-opus-5",
            "taskResults": [unscored_task("Not Run")],
        },
        {
            "modelVersionName": "Unknown Vendor",
            "modelVersionSlug": "some-model-nobody-registered",
            "taskResults": [numeric_task("acc", 0.5)],
        },
    ]


def convert(rows, output_dir, benchmark=BENCH):
    return adapter.convert_benchmark(
        benchmark, rows, "123.0", output_dir, use_registry=True
    )


def test_conversion_writes_records_and_accounts_for_every_row(tmp_path):
    result = convert(sample_rows(), tmp_path / "kaggle")
    assert result.total_records == 3
    assert [output.eval_log.model_info.id for output in result.records] == [
        "google/gemma-3-27b-it"
    ]
    # The all-unscored row is an exclusion (with its unscored task); a model
    # whose developer cannot be established at all is a failure.
    assert [exclusion.source_ref for exclusion in result.exclusions] == [
        "owner/my-bench row 0 task 2",
        "owner/my-bench row 1 task 0",
        "owner/my-bench row 1",
    ]
    assert len(result.failures) == 1
    assert "some-model-nobody-registered" in result.failures[0].reason


def test_converted_record_is_platform_led_and_routed_by_canonical_id(tmp_path):
    from every_eval_ever.helpers import save_evaluation_logs

    result = convert(sample_rows()[:1], tmp_path / "kaggle")
    paths = save_evaluation_logs(result.records)
    assert len(paths) == 1
    assert paths[0].parent == (
        tmp_path / "kaggle" / "google" / "gemma-3-27b-it"
    )
    log = EvaluationLog.model_validate(json.loads(paths[0].read_text()))

    # The source is the platform; the uploader is a fact about the benchmark.
    assert log.source_metadata.source_name == "Kaggle Benchmarks"
    assert log.source_metadata.source_organization_name == "Kaggle"
    details = log.source_metadata.additional_details
    assert details["benchmark_owner"] == "owner"
    assert details["benchmark_slug"] == "my-bench"
    assert details["benchmark_url"] == BENCH.url
    # No harness ran these.
    assert log.eval_library.name == "unknown"

    assert log.evaluation_id == "owner/my-bench/gemma-3-27b-it/123.0"
    model_details = log.model_info.additional_details
    assert model_details["kaggle_model_version_slug"] == "gemma-3-27b-it"
    assert model_details["model_id_resolution"] == "registry"
    assert model_details["model_id_resolution_strategy"] == "exact"
    assert "model_id_created_new" not in model_details
    assert model_details["display_name"] == "Gemma 3 27B"
    assert [
        result.evaluation_name for result in log.evaluation_results
    ] == ["kaggle.owner.my-bench", "kaggle.owner.my-bench.emotion-inference"]


def test_two_efforts_of_one_model_stay_distinct_evaluations(tmp_path):
    # Both resolve to the same canonical id, so only evaluation_id keeps them
    # apart; a shared one would collapse two runs into one row.
    rows = [
        {
            "modelVersionName": f"Claude Sonnet 4.5{suffix}",
            "modelVersionSlug": slug,
            "taskResults": [numeric_task("", 0.5)],
        }
        for slug, suffix in (
            ("claude-sonnet-4-5-20250929", ""),
            ("claude-sonnet-4-5-thinking-20250929", " (thinking)"),
        )
    ]
    result = convert(rows, tmp_path / "kaggle")
    logs = [output.eval_log for output in result.records]
    assert {log.model_info.id for log in logs} == {
        "anthropic/claude-sonnet-4.5-20250929"
    }
    assert len({log.evaluation_id for log in logs}) == 2
    configs = [log.evaluation_results[0].generation_config for log in logs]
    assert configs[0] is None
    assert configs[1].generation_args.reasoning is True


def test_one_registry_lookup_serves_every_row_of_a_model(tmp_path, monkeypatch):
    calls = []
    original = adapter.registry.resolve_model_id

    def counting(raw_value, **kwargs):
        calls.append(raw_value)
        return original(raw_value, **kwargs)

    monkeypatch.setattr(adapter.registry, "resolve_model_id", counting)
    identities = {}
    for benchmark_slug in ("one", "two"):
        benchmark = adapter.make_benchmark(index_entry("o", benchmark_slug))
        adapter.convert_benchmark(
            benchmark,
            sample_rows()[:1],
            "123.0",
            tmp_path / "kaggle",
            identities=identities,
        )
    assert calls == ["gemma-3-27b-it"]


def test_taskresults_of_the_wrong_type_fails_only_that_row(tmp_path):
    rows = sample_rows()[:1] + [
        {"modelVersionSlug": "claude-opus-5", "taskResults": None}
    ]
    result = convert(rows, tmp_path / "kaggle")
    assert len(result.records) == 1
    assert [failure.source_ref for failure in result.failures] == [
        "owner/my-bench row 1"
    ]


def test_a_row_whose_every_task_failed_is_a_failure_not_an_exclusion(tmp_path):
    # It was scored; the conversion is what went wrong, so the run must go red
    # rather than record it as a model Kaggle never ran.
    rows = [
        {
            "modelVersionSlug": "claude-opus-5",
            "taskResults": [numeric_task("acc", None)],
        }
    ]
    result = convert(rows, tmp_path / "kaggle")
    assert result.records == []
    assert result.exclusions == []
    assert [failure.source_ref for failure in result.failures] == [
        "owner/my-bench row 0 task 0",
        "owner/my-bench row 0",
    ]
    assert "no record written" in result.failures[1].reason


# ---------------------------------------------------------------------------
# main: a partial sweep is never a clean success
# ---------------------------------------------------------------------------
def _run_main(monkeypatch, tmp_path, slugs, leaderboards):
    monkeypatch.setattr(adapter, "open_session", lambda: object())
    _stub_index(monkeypatch, [index_entry("o", slug) for slug in slugs])

    def fake_fetch(session, benchmark):
        rows = leaderboards[benchmark.ref]
        if isinstance(rows, Exception):
            raise rows
        return rows

    monkeypatch.setattr(adapter, "fetch_leaderboard", fake_fetch)
    return adapter.main(["--all", "--output-dir", str(tmp_path / "data" / "k")])


def test_a_failed_leaderboard_fetch_exits_nonzero_but_keeps_good_records(
    monkeypatch, tmp_path
):
    with pytest.raises(SourceRecordsError, match="leaderboard fetch failed"):
        _run_main(
            monkeypatch,
            tmp_path,
            ["good", "bad"],
            {"o/good": sample_rows()[:1], "o/bad": FetchError("boom")},
        )
    assert list((tmp_path / "data" / "k").rglob("*.json"))
    report = tmp_path / "adapter_reports" / "k_failures.json"
    assert json.loads(report.read_text())["failed_record_count"] == 1


def test_a_clean_sweep_exits_zero(monkeypatch, tmp_path):
    written = _run_main(
        monkeypatch, tmp_path, ["good"], {"o/good": sample_rows()[:1]}
    )
    assert written == 1


def test_no_target_selector_is_a_usage_error():
    with pytest.raises(SystemExit):
        adapter.parse_args([])
