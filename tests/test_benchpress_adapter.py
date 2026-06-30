"""Tests for the BenchPress aggregator adapter (utils/benchpress/adapter.py)."""
import json

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file
from utils.benchpress import adapter


def sample_payload() -> dict:
    """In-memory payload (already-parsed lists, as fetch_payload returns).

    Seeds three relationships (tech_report->first_party, leaderboard->third_party,
    empty->other) and two metric types: pct (declared range, bounded) + rating
    (unbounded -> +/-inf). Includes the metadata.json freshness manifest.
    """
    return {
        "metadata": {
            "generated_at_utc": "2026-05-07T04:54:26.048511+00:00",
            "source_git_commit": "5be3b4eddf0188721ff25f00713b589b2cbed8e0",
            "source_data_dirty": False,
        },
        "models": [
            {"id": "gpt-oss-120b", "name": "gpt-oss-120B", "provider": "OpenAI",
             "release_date": "2025-08-05", "open_weights": "true"},
            {"id": "claude-opus-4.6", "name": "Claude Opus 4.6", "provider": "Anthropic",
             "open_weights": "false"},
        ],
        "benchmarks": [
            {"id": "aime_2025", "name": "AIME 2025", "category": "Math",
             "metric": "% correct", "num_problems": 30.0, "source_url": "https://maa.org/aime",
             "canonical_setting": {"metric_type": "pct", "range": [0, 100],
                                   "higher_is_better": True, "version": "AIME-2025-I+II"}},
            {"id": "codeforces_rating", "name": "Codeforces Rating", "category": "Code",
             "metric": "Elo", "source_url": None,
             "canonical_setting": {"metric_type": "rating", "higher_is_better": True}},
        ],
        "scores": [
            {"model_id": "gpt-oss-120b", "benchmark_id": "aime_2025", "score": 97.9,
             "reference_url": "https://arxiv.org/abs/2508.10925", "source_type": "tech_report",
             "audit_status": "verified", "matches_canonical": "true",
             "reported_setting": {"temperature": 0.0, "mode": "thinking", "tools": "none",
                                  "harness": "OLMES", "sampling": "pass@1", "judge": "rule-based"},
             "n_candidates": "1"},
            {"model_id": "gpt-oss-120b", "benchmark_id": "codeforces_rating", "score": 2622.0,
             "reference_url": "https://codeforces.example/x", "source_type": "leaderboard",
             "reported_setting": {"judge": "gpt-4o"}},
            {"model_id": "claude-opus-4.6", "benchmark_id": "aime_2025", "score": 93.5,
             "reference_url": "https://anthropic.com/news", "source_type": "",
             "reported_setting": {"temperature": 1.0, "mode": "thinking"}},
        ],
    }


def _logs_by_relationship():
    bundles = adapter.make_logs(sample_payload())
    return {b.log.source_metadata.evaluator_relationship.value: b for b in bundles}


def test_relationship_split():
    assert set(_logs_by_relationship()) == {"first_party", "third_party", "other"}


def test_logs_are_schema_valid():
    for bundle in adapter.make_logs(sample_payload()):
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == "0.2.2"
        assert validated.source_metadata.source_type.value == "documentation"
        assert validated.source_metadata.source_organization_name == "BenchPress"
        assert validated.eval_library.name == "BenchPress"


def test_model_id_and_evaluation_id():
    fp = _logs_by_relationship()["first_party"].log
    assert fp.model_info.id == "openai/gpt-oss-120b"
    assert fp.model_info.additional_details["benchpress_model_id"] == "gpt-oss-120b"
    # retrieved_timestamp derives from metadata.generated_at_utc
    assert fp.evaluation_id.startswith("benchpress/first_party/openai_gpt-oss-120b/")
    assert fp.retrieved_timestamp == adapter._iso_to_epoch_str(
        "2026-05-07T04:54:26.048511+00:00")


def test_citation_url_and_reported_by():
    res = _logs_by_relationship()["first_party"].log.evaluation_results[0]
    assert res.source_data.url[0] == "https://arxiv.org/abs/2508.10925"
    assert res.source_data.additional_details["reported_by"] == "arxiv.org"
    assert res.source_data.additional_details["source_role"] == "aggregator"


def test_bounded_metric_uses_declared_range():
    pct = _logs_by_relationship()["first_party"].log.evaluation_results[0].metric_config
    assert pct.score_type.value == "continuous"
    assert (pct.min_score, pct.max_score) == (0.0, 100.0)


def test_unbounded_metric_uses_infinity():
    rating = _logs_by_relationship()["third_party"].log.evaluation_results[0].metric_config
    assert rating.metric_kind == "rating"
    assert rating.min_score == float("-inf")
    assert rating.max_score == float("inf")


def test_version_provenance_recorded():
    details = _logs_by_relationship()["first_party"].log.source_metadata.additional_details
    assert details["benchpress_source_git_commit"] == "5be3b4eddf0188721ff25f00713b589b2cbed8e0"
    assert details["benchpress_generated_at_utc"] == "2026-05-07T04:54:26.048511+00:00"


def test_export_writes_infinity_token_and_validates(tmp_path):
    bundles = adapter.make_logs(sample_payload())
    paths = adapter.export_logs(bundles, tmp_path)
    assert len(paths) == 3
    # the unbounded (rating) record is serialized with the JSON Infinity token...
    inf_raws = [p.read_text() for p in paths if "Infinity" in p.read_text()]
    assert len(inf_raws) == 1
    # ...and EEE's loader reads it back as float('inf'); the record validates.
    reloaded = EvaluationLog.model_validate(json.loads(inf_raws[0]))
    assert reloaded.evaluation_results[0].metric_config.max_score == float("inf")
    for p in paths:
        report = validate_file(p)
        assert report.valid, report.errors
        assert p.parent.parent.parent == tmp_path  # <out>/<dev>/<model>/<uuid>.json
    assert (tmp_path / "openai" / "gpt-oss-120b").is_dir()
    assert (tmp_path / "anthropic" / "claude-opus-4.6").is_dir()
