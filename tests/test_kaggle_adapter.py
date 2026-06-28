"""Unit tests for the generalized Kaggle Benchmarks adapter.

These exercise the pure transformation helpers (no network) plus an
end-to-end ``convert_benchmark`` run that writes to a tmp dir and validates
the saved records against the schema.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace

import pytest

from every_eval_ever.eval_types import EvaluationLog, ScoreType
from every_eval_ever.helpers import FetchError
from utils.kaggle import adapter


# ---------------------------------------------------------------------------
# Fixture builders mirroring the shape of Kaggle's leaderboard API.
# ---------------------------------------------------------------------------
def numeric_task(name, value, *, ci=None, date=None):
    numeric = {
        "value": value,
        "hasUnevenConfidenceInterval": False,
        "confidenceInterval": ci if ci is not None else 0.0,
        "hasConfidenceInterval": ci is not None,
    }
    result = {
        "hasNumericResult": True,
        "numericResult": numeric,
        "hasBooleanResult": False,
        "booleanResult": False,
        "customAdditionalResults": [],
        "resultCase": "numericResult",
        "hasEvaluationDate": date is not None,
    }
    if date is not None:
        result["evaluationDate"] = date
    return {"benchmarkTaskName": name, "benchmarkTaskSlug": "", "result": result}


def boolean_task(name, passed):
    return {
        "benchmarkTaskName": name,
        "result": {
            "hasNumericResult": False,
            "hasBooleanResult": True,
            "booleanResult": passed,
            "customAdditionalResults": [],
            "resultCase": "booleanResult",
        },
    }


def empty_task(name):
    return {
        "benchmarkTaskName": name,
        "result": {
            "hasNumericResult": False,
            "hasBooleanResult": False,
            "customAdditionalResults": [],
            "resultCase": "none",
        },
    }


PCT_META = {
    "benchmark_id": 42,
    "sort_order": "DESCENDING",
    "aggregation_type": "PERCENTAGE_PASSED",
    "display_type": "PERCENTAGES",
}


# ---------------------------------------------------------------------------
# _build_eval_result: result-type handling
# ---------------------------------------------------------------------------
def test_numeric_in_unit_range_is_bounded_continuous():
    r = adapter._build_eval_result(numeric_task("acc", 0.87654), "o", "s")
    assert r.score_details.score == 0.8765  # rounded to 4dp
    assert r.metric_config.score_type == ScoreType.continuous
    assert r.metric_config.min_score == 0.0
    assert r.metric_config.max_score == 1.0


def test_numeric_outside_unit_range_is_left_untyped():
    # Unknown scale (e.g. a count); must NOT fabricate [0, 1] bounds.
    r = adapter._build_eval_result(numeric_task("count", 100.0), "o", "s")
    assert r.score_details.score == 100.0
    assert r.metric_config.score_type is None
    assert r.metric_config.min_score is None
    assert r.metric_config.max_score is None


def test_boolean_results_become_binary():
    passed = adapter._build_eval_result(boolean_task("solved", True), "o", "s")
    failed = adapter._build_eval_result(boolean_task("solved", False), "o", "s")
    assert passed.metric_config.score_type == ScoreType.binary
    assert passed.score_details.score == 1.0
    assert passed.metric_config.min_score == 0.0
    assert passed.metric_config.max_score == 1.0
    assert failed.score_details.score == 0.0


def test_empty_result_is_skipped():
    assert adapter._build_eval_result(empty_task("unscored"), "o", "s") is None


def test_numeric_with_null_value_is_skipped():
    assert adapter._build_eval_result(numeric_task("x", None), "o", "s") is None


# ---------------------------------------------------------------------------
# _build_eval_result: enrichment (dates, CI, meta)
# ---------------------------------------------------------------------------
def test_evaluation_date_is_captured_as_timestamp():
    r = adapter._build_eval_result(
        numeric_task("acc", 0.5, date="2026-06-20T16:40:52.0Z"), "o", "s"
    )
    assert r.evaluation_timestamp == "2026-06-20T16:40:52.0Z"


def test_missing_evaluation_date_leaves_timestamp_unset():
    r = adapter._build_eval_result(numeric_task("acc", 0.5), "o", "s")
    assert r.evaluation_timestamp is None


def test_confidence_interval_brackets_the_score():
    # Kaggle's confidenceInterval is a symmetric half-width around the score,
    # so the bounds must bracket the score, not be centered at zero.
    r = adapter._build_eval_result(numeric_task("acc", 0.87, ci=0.02), "o", "s")
    ci = r.score_details.uncertainty.confidence_interval
    assert ci.lower == 0.85
    assert ci.upper == 0.89


def test_non_numeric_confidence_interval_is_ignored_not_fatal():
    task = numeric_task("acc", 0.5)
    task["result"]["numericResult"]["hasConfidenceInterval"] = True
    task["result"]["numericResult"]["confidenceInterval"] = "oops"
    r = adapter._build_eval_result(task, "o", "s")
    assert r.score_details.uncertainty is None


def test_no_uncertainty_when_ci_absent():
    r = adapter._build_eval_result(numeric_task("acc", 0.5), "o", "s")
    assert r.score_details.uncertainty is None


def test_meta_sets_direction_unit_and_kind():
    r = adapter._build_eval_result(numeric_task("acc", 0.9), "o", "s", PCT_META)
    assert r.metric_config.lower_is_better is False
    assert r.metric_config.metric_kind == "pass_rate"
    assert r.metric_config.metric_unit == "proportion"
    assert r.metric_config.metric_name == "acc"


def test_ascending_sort_order_means_lower_is_better():
    meta = {**PCT_META, "sort_order": "ASCENDING"}
    r = adapter._build_eval_result(numeric_task("latency", 0.3), "o", "s", meta)
    assert r.metric_config.lower_is_better is True


def test_boolean_result_does_not_get_aggregation_kind_or_unit():
    # Aggregation-derived kind/unit describe a numeric metric, not a 0/1 result.
    r = adapter._build_eval_result(boolean_task("solved", True), "o", "s", PCT_META)
    assert r.metric_config.metric_kind is None
    assert r.metric_config.metric_unit is None
    assert r.metric_config.score_type == ScoreType.binary


def test_counts_display_type_maps_to_count_unit():
    meta = {**PCT_META, "display_type": "COUNTS", "aggregation_type": None}
    r = adapter._build_eval_result(numeric_task("n", 100.0), "o", "s", meta)
    assert r.metric_config.metric_unit == "count"
    assert r.metric_config.metric_kind is None


def test_without_meta_direction_defaults_false_and_no_unit():
    r = adapter._build_eval_result(numeric_task("acc", 0.9), "o", "s")
    assert r.metric_config.lower_is_better is False
    assert r.metric_config.metric_unit is None
    assert r.metric_config.metric_kind is None


def test_task_name_falls_back_to_slug():
    task = numeric_task("", 0.5)
    r = adapter._build_eval_result(task, "owner", "myslug")
    assert r.evaluation_name == "myslug"


# ---------------------------------------------------------------------------
# _benchmark_owner / _benchmark_meta
# ---------------------------------------------------------------------------
def test_owner_prefers_organization_slug():
    bench = {
        "organization": {"slug": "cohere-labs"},
        "ownerUser": {"userName": "someuser"},
    }
    assert adapter._benchmark_owner(bench) == "cohere-labs"


def test_owner_falls_back_to_username():
    bench = {"organization": None, "ownerUser": {"userName": "someuser"}}
    assert adapter._benchmark_owner(bench) == "someuser"


def test_owner_none_when_unresolvable():
    assert adapter._benchmark_owner({}) is None


def test_benchmark_meta_extracts_scoring_config():
    bench = {
        "id": 7,
        "task": {
            "version": {
                "sortOrder": "DESCENDING",
                "aggregationType": "PERCENTAGE_PASSED",
                "displayType": "PERCENTAGES",
            }
        },
    }
    meta = adapter._benchmark_meta(bench)
    assert meta == {
        "benchmark_id": 7,
        "sort_order": "DESCENDING",
        "aggregation_type": "PERCENTAGE_PASSED",
        "display_type": "PERCENTAGES",
    }


def test_benchmark_meta_handles_missing_version():
    meta = adapter._benchmark_meta({"id": 1})
    assert meta["sort_order"] is None
    assert meta["aggregation_type"] is None


# ---------------------------------------------------------------------------
# list_benchmarks: discovery pagination + filtering (network mocked)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Returns two canned ListBenchmarks pages keyed by pageToken."""

    def __init__(self, pages):
        self._pages = pages
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        token = json["pageToken"]
        self.calls.append(token)
        return _FakeResp(self._pages[token])


def test_list_benchmarks_paginates_and_filters_unpublished(monkeypatch):
    pages = {
        "": {
            "benchmarks": [
                {
                    "slug": "global-mmlu-lite",
                    "name": " Global MMLU Lite ",
                    "published": True,
                    "organization": {"slug": "cohere-labs"},
                    "ownerUser": {"userName": "creator"},
                    "task": {"version": {"sortOrder": "DESCENDING"}},
                },
                # Unpublished -> must be filtered out.
                {"slug": "draft", "published": False, "ownerUser": {"userName": "x"}},
            ],
            "nextPageToken": "p2",
        },
        "p2": {
            "benchmarks": [
                {
                    "slug": "solo",
                    "name": "Solo",
                    "published": True,
                    "organization": None,
                    "ownerUser": {"userName": "alice"},
                }
            ],
            "nextPageToken": "",
        },
    }
    fake = _FakeSession(pages)
    monkeypatch.setattr(adapter, "_kaggle_session", lambda: (fake, "xsrf-token"))

    out = list(adapter.list_benchmarks())

    assert fake.calls == ["", "p2"]  # paginated via nextPageToken
    assert [(b["owner"], b["slug"]) for b in out] == [
        ("cohere-labs", "global-mmlu-lite"),  # org slug preferred over creator
        ("alice", "solo"),  # falls back to username
    ]
    assert out[0]["name"] == "Global MMLU Lite"  # stripped
    assert out[0]["meta"]["sort_order"] == "DESCENDING"


# ---------------------------------------------------------------------------
# fetch_leaderboard: failure vs empty contract (network mocked)
# ---------------------------------------------------------------------------
def test_fetch_leaderboard_returns_none_on_fetch_error(monkeypatch):
    def boom(url):
        raise FetchError("boom")

    monkeypatch.setattr(adapter, "fetch_json", boom)
    assert adapter.fetch_leaderboard("o", "s") is None


def test_fetch_leaderboard_returns_empty_list_for_no_submissions(monkeypatch):
    monkeypatch.setattr(adapter, "fetch_json", lambda url: {"rows": []})
    assert adapter.fetch_leaderboard("o", "s") == []


def test_fetch_leaderboard_returns_none_on_malformed_shape(monkeypatch):
    # A 200 with an error envelope / no list-valued rows is a failure, not empty.
    monkeypatch.setattr(adapter, "fetch_json", lambda url: {"error": "nope"})
    assert adapter.fetch_leaderboard("o", "s") is None


# ---------------------------------------------------------------------------
# _resolve_targets: dedup + discovery-failure flag
# ---------------------------------------------------------------------------
def test_resolve_targets_dedupes_repeated_benchmarks():
    args = Namespace(
        benchmark=["cohere-labs/global-mmlu-lite", "cohere-labs/global-mmlu-lite"],
        all=False,
        limit=None,
    )
    targets, discovery_failed = adapter._resolve_targets(args)
    assert len(targets) == 1
    assert discovery_failed is False


def test_resolve_targets_dedup_prefers_meta_carrying_entry(monkeypatch):
    # An explicit (meta-less) --benchmark overlapping an --all discovery entry
    # must keep the discovered entry's meta so enrichment isn't lost.
    def discover():
        yield {
            "owner": "cohere-labs",
            "slug": "global-mmlu-lite",
            "name": "Global MMLU Lite",
            "meta": {"sort_order": "DESCENDING", "display_type": "PERCENTAGES"},
        }

    monkeypatch.setattr(adapter, "list_benchmarks", discover)
    args = Namespace(
        benchmark=["cohere-labs/global-mmlu-lite"], all=True, limit=None
    )
    targets, _ = adapter._resolve_targets(args)
    assert len(targets) == 1
    assert targets[0]["meta"]["display_type"] == "PERCENTAGES"


def test_resolve_targets_flags_truncated_discovery(monkeypatch):
    def partial():
        yield {"owner": "o", "slug": "a", "name": "A", "meta": {}}
        raise FetchError("page 2 failed")

    monkeypatch.setattr(adapter, "list_benchmarks", partial)
    args = Namespace(benchmark=None, all=True, limit=None)
    targets, discovery_failed = adapter._resolve_targets(args)
    assert [t["slug"] for t in targets] == ["a"]  # partial progress kept
    assert discovery_failed is True


# ---------------------------------------------------------------------------
# main: non-zero exit when a fetch fails, good benchmark still written
# ---------------------------------------------------------------------------
def test_main_exits_nonzero_when_a_fetch_fails(monkeypatch, tmp_path):
    def fake_fetch(owner, slug):
        return None if owner == "bad" else sample_rows()[:1]

    monkeypatch.setattr(adapter, "fetch_leaderboard", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapter",
            "--benchmark",
            "good/one",
            "--benchmark",
            "bad/two",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        adapter.main()
    assert exc.value.code == 1
    # The good benchmark was still converted despite the other's failure.
    assert list(tmp_path.rglob("*.json"))


# ---------------------------------------------------------------------------
# convert_benchmark: end-to-end, schema-valid output
# ---------------------------------------------------------------------------
def sample_rows():
    return [
        {
            "modelVersionName": "GPT-4o",
            "modelVersionSlug": "gpt-4o-2024-05-13",
            "taskResults": [
                numeric_task("Overall", 0.71, date="2026-06-20T16:40:52Z"),
                boolean_task("Edge Case", True),
                empty_task("Unscored Task"),  # dropped, no usable score
            ],
        },
        # Row with only empty results -> produces no file.
        {
            "modelVersionName": "Blank",
            "modelVersionSlug": "blank-model",
            "taskResults": [empty_task("Nothing")],
        },
        # Row missing the slug -> skipped entirely.
        {"modelVersionName": "No Slug", "taskResults": [numeric_task("x", 0.5)]},
    ]


def test_convert_benchmark_writes_only_rows_with_results(tmp_path):
    count = adapter.convert_benchmark(
        "cohere-labs",
        "demo",
        "Demo Benchmark",
        sample_rows(),
        str(tmp_path),
        "123.0",
        meta=PCT_META,
    )
    assert count == 1  # only gpt-4o has usable results
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1


def test_convert_benchmark_output_validates_and_is_enriched(tmp_path):
    adapter.convert_benchmark(
        "cohere-labs",
        "demo",
        "Demo Benchmark",
        sample_rows(),
        str(tmp_path),
        "123.0",
        meta=PCT_META,
    )
    path = next(tmp_path.rglob("*.json"))
    log = EvaluationLog.model_validate(json.loads(path.read_text()))

    assert log.schema_version == "0.2.2"
    assert log.model_info.id == "openai/gpt-4o-2024-05-13"
    assert log.source_metadata.source_organization_name == "cohere-labs"

    details = log.source_metadata.additional_details
    assert details["platform"] == "kaggle"
    assert details["benchmark_id"] == "42"
    assert details["aggregation_type"] == "PERCENTAGE_PASSED"
    assert details["display_type"] == "PERCENTAGES"

    # Empty task dropped; numeric + boolean kept.
    names = {r.evaluation_name for r in log.evaluation_results}
    assert names == {"Overall", "Edge Case"}

    overall = next(r for r in log.evaluation_results if r.evaluation_name == "Overall")
    assert overall.evaluation_timestamp == "2026-06-20T16:40:52Z"
    assert overall.metric_config.metric_unit == "proportion"
    assert overall.metric_config.lower_is_better is False
