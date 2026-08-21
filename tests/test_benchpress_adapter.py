"""Tests for the BenchPress aggregator adapter (every_eval_ever/adapters/benchpress/adapter.py)."""
import argparse
import json
import pathlib

import pytest

from every_eval_ever.adapters.benchpress import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file
from every_eval_ever.validator.json_utils import strict_json_loads


def sample_payload() -> dict:
    """In-memory payload (already-parsed lists, as fetch_payload returns).

    Seeds all three relationships (leaderboard -> third_party; a provider's blog
    on its own domain -> first_party; a provider-authored type on a shared host,
    and an unstated type, -> other), two metric types (pct with a declared range;
    rating, unbounded -> +/-inf), a tech_report citation shared by two providers,
    and one row BenchPress dropped.
    """
    return {
        "metadata": {
            "generated_at_utc": "2026-05-07T04:54:26.048511+00:00",
            "source_git_commit": "5be3b4eddf0188721ff25f00713b589b2cbed8e0",
            "source_data_dirty": False,
            "dataset_revision": "fbe2869d4e1581372830f02a11c64c08365cf656",
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
             "audit_status": "verified", "reported_setting": {"judge": "gpt-4o"}},
            {"model_id": "claude-opus-4.6", "benchmark_id": "aime_2025", "score": 93.5,
             "reference_url": "https://anthropic.com/news", "source_type": "",
             "audit_status": "verified",
             "reported_setting": {"temperature": 1.0, "mode": "thinking"}},
            # One tech report cited for both providers' scores: a comparison
            # table, so it is not its subjects reporting themselves either.
            {"model_id": "claude-opus-4.6", "benchmark_id": "codeforces_rating",
             "score": 2100.0, "reference_url": "https://arxiv.org/abs/2412.19437",
             "source_type": "tech_report", "audit_status": "verified"},
            {"model_id": "gpt-oss-120b", "benchmark_id": "codeforces_rating",
             "score": 2200.0, "reference_url": "https://arxiv.org/abs/2412.19437",
             "source_type": "tech_report", "audit_status": "verified"},
            # BenchPress excludes this from its own canonical matrix.
            {"model_id": "gpt-oss-120b", "benchmark_id": "aime_2025", "score": 12.0,
             "reference_url": "https://example.invalid/rumour",
             "source_type": "official_blog", "audit_status": "dropped"},
            # The provider's own domain publishing its own model's score: the one
            # case the export does identify the publisher.
            {"model_id": "gpt-oss-120b", "benchmark_id": "aime_2025", "score": 96.0,
             "reference_url": "https://openai.com/index/introducing-gpt-oss/",
             "source_type": "official_blog", "audit_status": "verified"},
        ],
    }


def _logs_by_relationship(developer: str = "openai"):
    """One developer's bundles keyed by relationship — one bundle per split."""
    result = adapter.make_logs(sample_payload())
    return {b.log.source_metadata.evaluator_relationship.value: b
            for b in result.records if b.developer == developer}


def test_relationship_split():
    assert set(_logs_by_relationship()) == {"third_party", "other", "first_party"}
    assert set(_logs_by_relationship("anthropic")) == {"other"}


def _cited_urls(result, relationship: str) -> set[str]:
    return {
        result_.source_data.url[0]
        for bundle in result.records
        if bundle.log.source_metadata.evaluator_relationship.value == relationship
        for result_ in bundle.log.evaluation_results
    }


def test_first_party_takes_the_providers_own_domain_not_just_the_type():
    """source_type says what kind of document a citation is, not who wrote it."""
    payload = sample_payload()
    # The live falsification of reading the type alone: a model card is a
    # provider-authored type, but this one is Qwen's and the model is Anthropic's.
    payload["scores"][2].update({
        "reference_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "source_type": "model_card"})
    result = adapter.make_logs(payload)

    assert _cited_urls(result, "first_party") == {
        "https://openai.com/index/introducing-gpt-oss/"}
    # arxiv 2508.10925 is a tech_report cited for one provider's model, which is as
    # close as breadth gets to publisher evidence and still names no publisher.
    # arxiv 2412.19437 is one report cited for two, so it is a comparison table.
    # The model card names an org in its path, which anyone can occupy.
    assert {"https://arxiv.org/abs/2508.10925",
            "https://arxiv.org/abs/2412.19437",
            "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507"} <= (
        _cited_urls(result, "other"))


def test_a_second_result_for_one_benchmark_is_reported_not_dropped():
    """evaluation_result_id is a join key, so a log carries one per benchmark."""
    payload = sample_payload()
    payload["scores"].append({
        "model_id": "claude-opus-4.6", "benchmark_id": "aime_2025", "score": 88.0,
        "reference_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "source_type": "model_card", "audit_status": "verified"})
    result = adapter.make_logs(payload)

    assert [f.source_ref for f in result.failures] == ["claude-opus-4.6/aime_2025"]
    assert "https://anthropic.com/news" in result.failures[0].reason

    # The same cell reported twice carries nothing new, so it is not a conflict;
    # it is accounted as an exclusion so the ledger still reconciles.
    payload["scores"][-1] = dict(payload["scores"][2])
    repeated = adapter.make_logs(payload)
    assert repeated.failures == []
    assert len(repeated.records) == len(adapter.make_logs(sample_payload()).records)
    baseline = adapter.make_logs(sample_payload())
    assert len(repeated.exclusions) == len(baseline.exclusions) + 1
    assert any('exact duplicate' in e.reason for e in repeated.exclusions)


def test_a_shared_host_never_identifies_a_publisher():
    """A domain has an owner; a path on someone else's domain does not."""
    assert adapter._provider_publishes("https://openai.com/index/x", "OpenAI")
    assert adapter._provider_publishes("https://cdn.amazon.science/p.pdf", "Amazon")
    assert adapter._provider_publishes("https://x.ai/news/grok", "xAI")
    assert not adapter._provider_publishes(
        "https://huggingface.co/Qwen/Qwen3", "Qwen")
    # Google's public bucket host serves anyone's bucket, so it names nobody.
    assert not adapter._provider_publishes(
        "https://storage.googleapis.com/x/y.pdf", "Google")
    assert not adapter._provider_publishes(None, "OpenAI")
    assert not adapter._provider_publishes("https://openai.com/x", None)


def test_scores_benchpress_dropped_are_excluded_not_failed():
    """BenchPress rejecting a row is a policy exclusion, not a conversion failure."""
    result = adapter.make_logs(sample_payload())
    assert result.total_records == 7
    assert [e.source_ref for e in result.exclusions] == ['gpt-oss-120b/aime_2025']
    assert 'dropped' in result.exclusions[0].reason
    assert result.failures == []
    result.raise_if_incomplete()  # an exclusion must not fail the run

    kept = adapter.make_logs(sample_payload(), include_unaccepted=True)
    assert kept.exclusions == []


def test_a_score_outside_its_declared_range_is_a_failure(tmp_path):
    """The export mixes scales inside one benchmark; a record cannot state both."""
    payload = sample_payload()
    payload['scores'].append({
        'model_id': 'gpt-oss-120b', 'benchmark_id': 'aime_2025', 'score': 950.0,
        'reference_url': 'https://arxiv.org/abs/2508.10925',
        'source_type': 'tech_report', 'audit_status': 'verified',
    })
    result = adapter.make_logs(payload)
    assert [f.source_ref for f in result.failures] == ['gpt-oss-120b/aime_2025']
    assert 'declared range [0.0, 100.0]' in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()

    # The valid records still publish, and nothing invalid reaches the tree.
    for path in adapter.export_logs(result.records, tmp_path / 'data' / 'benchpress'):
        assert validate_file(path).valid


def test_logs_are_schema_valid():
    for bundle in adapter.make_logs(sample_payload()).records:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == SCHEMA_VERSION
        assert validated.source_metadata.source_type.value == "documentation"
        assert validated.source_metadata.source_organization_name == "BenchPress"
        assert validated.eval_library.name == "BenchPress"


def test_model_id_and_evaluation_id():
    log = _logs_by_relationship()["other"].log
    assert log.model_info.id == "openai/gpt-oss-120b"
    assert log.model_info.additional_details["benchpress_model_id"] == "gpt-oss-120b"
    # retrieved_timestamp derives from metadata.generated_at_utc
    assert log.evaluation_id.startswith("benchpress/other/openai_gpt-oss-120b/")
    assert log.retrieved_timestamp == adapter._iso_to_epoch_str(
        "2026-05-07T04:54:26.048511+00:00")
    # The immutable dataset revision anchors the id: two content-differing
    # snapshots that share one manifest timestamp still get distinct ids.
    assert log.evaluation_id.endswith(
        "/fbe2869d4e1581372830f02a11c64c08365cf656")


def test_model_availability_is_derived_from_open_weights():
    """BenchPress ships an open_weights flag; recording a blanket 'unknown' would
    throw the one availability signal the source provides away."""
    openai = _logs_by_relationship("openai")["other"].log.model_info.additional_details
    assert openai["model_availability"] == "open_weights"   # open_weights="true"
    assert openai["deployment_type"] == "unknown"           # no serving platform recorded
    anthropic = _logs_by_relationship("anthropic")["other"].log.model_info.additional_details
    assert anthropic["model_availability"] == "closed_weights"  # open_weights="false"


def test_citation_url_and_reported_by():
    res = _logs_by_relationship()["other"].log.evaluation_results[0]
    assert res.source_data.url[0] == "https://arxiv.org/abs/2508.10925"
    assert res.source_data.additional_details["reported_by"] == "arxiv.org"
    assert res.source_data.additional_details["source_role"] == "aggregator"


def test_bounded_metric_uses_declared_range():
    pct = _logs_by_relationship()["other"].log.evaluation_results[0].metric_config
    assert pct.score_type.value == "continuous"
    assert (pct.min_score, pct.max_score) == (0.0, 100.0)


def test_unbounded_metric_uses_infinity():
    rating = _logs_by_relationship()["third_party"].log.evaluation_results[0].metric_config
    assert rating.metric_kind == "rating"
    assert rating.min_score == float("-inf")
    assert rating.max_score == float("inf")


def test_version_provenance_recorded():
    details = _logs_by_relationship()["other"].log.source_metadata.additional_details
    assert details["benchpress_source_git_commit"] == "5be3b4eddf0188721ff25f00713b589b2cbed8e0"
    assert details["benchpress_generated_at_utc"] == "2026-05-07T04:54:26.048511+00:00"
    assert details["benchpress_dataset_revision"] == (
        "fbe2869d4e1581372830f02a11c64c08365cf656")


def test_export_writes_standards_compliant_infinity_and_validates(tmp_path):
    paths = adapter.export_logs(adapter.make_logs(sample_payload()).records, tmp_path)
    assert len(paths) == 4
    inf_raws = [p.read_text() for p in paths if "Infinity" in p.read_text()]
    assert inf_raws
    # Unbounded bounds are the JSON *string* "Infinity", which a strict parser
    # accepts (a bare Infinity token would fail here) and pydantic reads as a float.
    assert '"Infinity"' in inf_raws[0]
    reloaded = EvaluationLog.model_validate(strict_json_loads(inf_raws[0]))
    assert any(result.metric_config.max_score == float("inf")
               for result in reloaded.evaluation_results)
    for p in paths:
        report = validate_file(p)
        assert report.valid, report.errors
        assert p.parent.parent.parent == tmp_path  # <out>/<dev>/<model>/<uuid>.json
    assert (tmp_path / "openai" / "gpt-oss-120b").is_dir()
    assert (tmp_path / "anthropic" / "claude-opus-4.6").is_dir()


# --------------------------------------------------------------------------- #
# per-row failure boundary
# --------------------------------------------------------------------------- #

def test_a_score_that_will_not_parse_fails_only_its_own_row(tmp_path):
    """One unusable cell must not take every valid row down with it."""
    payload = sample_payload()
    payload['scores'] += [
        {'model_id': 'gpt-oss-120b', 'benchmark_id': 'aime_2025', 'score': 'n/a',
         'reference_url': 'https://x.invalid', 'source_type': 'tech_report',
         'audit_status': 'verified'},
        {'model_id': 'claude-opus-4.6', 'benchmark_id': 'aime_2025',
         'score': float('nan'), 'reference_url': 'https://y.invalid',
         'source_type': 'tech_report', 'audit_status': 'verified'},
    ]
    result = adapter.make_logs(payload)

    assert len(result.records) == 4
    assert [f.source_ref for f in result.failures] == [
        'gpt-oss-120b/aime_2025', 'claude-opus-4.6/aime_2025']
    assert "must be numeric; got 'n/a'" in result.failures[0].reason
    assert 'must be finite' in result.failures[1].reason
    # The rejected row goes into the report, so the value can be looked at.
    assert result.failures[0].source_record['score'] == 'n/a'
    # Every source row is still counted, so the report reconciles against it.
    assert result.total_records == 9
    for path in adapter.export_logs(result.records, tmp_path / 'data' / 'benchpress'):
        assert validate_file(path).valid


def test_a_schema_error_is_recorded_against_its_source_row():
    """A field the schema rejects is bad source data, not an aborted run."""
    payload = sample_payload()
    payload['benchmarks'][0]['metric'] = 123  # metric_name must be a string
    result = adapter.make_logs(payload)

    assert len(result.records) == 3
    assert [f.source_ref for f in result.failures] == [
        'gpt-oss-120b/aime_2025', 'claude-opus-4.6/aime_2025',
        'gpt-oss-120b/aime_2025']
    assert 'ValidationError' in result.failures[0].reason


def test_an_unexpected_error_is_not_filed_as_bad_data(monkeypatch):
    """The boundary is narrow: a bug in the adapter still surfaces as a crash."""
    def boom(score, benchmark):
        raise RuntimeError('adapter bug')

    monkeypatch.setattr(adapter, 'make_evaluation_result', boom)
    with pytest.raises(RuntimeError, match='adapter bug'):
        adapter.make_logs(sample_payload())


def test_descriptive_numbers_that_will_not_parse_are_preserved(tmp_path):
    """params/num_problems only ever reach the record as text, so keep the text."""
    models = adapter._parse_models([
        {'model_id': 'm', 'params_total_M': 'about 120000',
         'params_active_M': '5100.0'},
    ])
    assert models[0]['params_total_M'] == 'about 120000'
    assert models[0]['params_active_M'] == 5100.0
    benchmarks = adapter._parse_benchmarks([
        {'benchmark_id': 'b', 'num_problems': '30 (I+II)'},
    ])
    assert benchmarks[0]['num_problems'] == '30 (I+II)'
    # The unparsed text reaches the record instead of ending the export.
    payload = sample_payload()
    payload['models'][0]['params_total_M'] = 'about 120000'
    bundle = next(b for b in adapter.make_logs(payload).records
                  if b.developer == 'openai')
    assert bundle.log.model_info.additional_details['params_total_M'] == (
        'about 120000')


def test_a_missing_id_column_is_a_structural_error_not_a_bad_row():
    """Every row lacking an id means the export changed shape, not that it is dirty."""
    with pytest.raises(KeyError):
        adapter._parse_scores([{'benchmark_id': 'b', 'score': '1.0'}])


def test_a_missing_audit_status_column_is_a_structural_error_not_an_empty_run():
    """audit_status gates accept/exclude, so a vanished column must not silently
    exclude every row and exit 0 -- it is a structural mismatch like a missing id."""
    with pytest.raises(KeyError):
        adapter._parse_scores([{'model_id': 'm', 'benchmark_id': 'b', 'score': '1.0'}])
    # A present column with an empty *value* is still just one unaccepted row.
    parsed = adapter._parse_scores(
        [{'model_id': 'm', 'benchmark_id': 'b', 'score': '1.0', 'audit_status': ''}])
    assert parsed[0]['audit_status'] is None


# --------------------------------------------------------------------------- #
# snapshot provenance
# --------------------------------------------------------------------------- #

def _record_fetches(monkeypatch, sha='ffb1f0dcb2b1a0f9c8e7d6a5b4c3d2e1f0a9b8c7'):
    """Capture every URL fetch_payload requests; answer them from sample_payload."""
    urls = []

    def fake_json(url):
        urls.append(url)
        if '/api/datasets/' in url:
            return {'sha': sha}
        return sample_payload()['metadata']

    def fake_csv(url):
        urls.append(url)
        return []

    monkeypatch.setattr(adapter, 'fetch_json', fake_json)
    monkeypatch.setattr(adapter, 'fetch_csv', fake_csv)
    return urls, sha


def test_a_symbolic_revision_is_resolved_to_one_commit(monkeypatch):
    """A branch or tag names whatever it points at now, and can move mid-run."""
    urls, sha = _record_fetches(monkeypatch)
    payload = adapter.fetch_payload('main')

    assert urls[0].endswith('/api/datasets/microsoft/benchpress-score-matrix'
                            '/revision/main')
    assert payload['metadata']['dataset_revision'] == sha
    # Every file is read at the resolved commit, never at the symbol.
    assert all(f'/resolve/{sha}/' in url for url in urls[1:])
    assert not any('/resolve/main/' in url for url in urls)


def test_a_pinned_sha_is_used_without_a_lookup(monkeypatch):
    """A full SHA is already immutable, so resolving it would only cost a request."""
    urls, _ = _record_fetches(monkeypatch)
    pinned = 'fbe2869d4e1581372830f02a11c64c08365cf656'
    payload = adapter.fetch_payload(pinned)

    assert payload['metadata']['dataset_revision'] == pinned
    assert not any('/api/datasets/' in url for url in urls)
    assert all(f'/resolve/{pinned}/' in url for url in urls)


def test_a_revision_with_a_slash_is_requested_escaped(monkeypatch):
    urls, _ = _record_fetches(monkeypatch)
    adapter.fetch_payload('refs/pr/2')

    assert urls[0].endswith('/revision/refs%2Fpr%2F2')


# --------------------------------------------------------------------------- #
# conversion accounting (the report)
# --------------------------------------------------------------------------- #

def _run_args(tmp_path, payload, **overrides) -> argparse.Namespace:
    input_json = tmp_path / 'payload.json'
    input_json.write_text(json.dumps(payload), encoding='utf-8')
    return argparse.Namespace(**{
        'input_json': input_json, 'save_raw_json': None,
        'output_dir': tmp_path / 'data' / 'benchpress',
        'retrieved_timestamp': None, 'revision': None,
        'include_unaccepted': False, **overrides})


def _report_path(tmp_path) -> pathlib.Path:
    return tmp_path / 'adapter_reports' / 'benchpress_failures.json'


def test_an_exclusions_only_run_itemizes_every_excluded_row(tmp_path):
    """A count in the console is not a record of which rows were left out."""
    payload = sample_payload()
    for score in payload['scores']:
        score['audit_status'] = 'dropped'

    assert adapter.run(_run_args(tmp_path, payload)) == 0  # exclusions do not fail

    report = json.loads(_report_path(tmp_path).read_text())
    assert report['failed_records'] == []
    assert report['excluded_record_count'] == 7
    assert [e['source_ref'] for e in report['excluded_records']] == [
        'gpt-oss-120b/aime_2025', 'gpt-oss-120b/codeforces_rating',
        'claude-opus-4.6/aime_2025', 'claude-opus-4.6/codeforces_rating',
        'gpt-oss-120b/codeforces_rating', 'gpt-oss-120b/aime_2025',
        'gpt-oss-120b/aime_2025']
    assert all('dropped' in e['reason'] for e in report['excluded_records'])
    # The excluded row itself is carried, like a failure: repeated source_refs
    # (three gpt-oss-120b/aime_2025 here) are distinguishable in the report.
    assert all('source_record' in e for e in report['excluded_records'])
    assert report['excluded_records'][0]['source_record']['model_id'] == 'gpt-oss-120b'


def test_the_accounting_report_replaces_the_previous_run_s_copy(tmp_path):
    """An earlier run's report left in place reads as this run's."""
    report_path = _report_path(tmp_path)
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({'failed_records': [{'source_ref': 'from-an-earlier-run'}]}),
        encoding='utf-8')

    adapter.run(_run_args(tmp_path, sample_payload()))

    report = json.loads(report_path.read_text())
    assert report['failed_records'] == []
    assert report['total_source_records'] == 7
    assert report['converted_records'] == 4
    # Written by an atomic swap, so no partial file is left beside it.
    assert [p.name for p in report_path.parent.iterdir()] == [report_path.name]


def test_the_report_survives_a_publication_error(tmp_path, monkeypatch):
    """It accounts for the conversion, so publication failing is when it matters."""
    def boom(bundles, output_dir):
        raise RuntimeError('disk full')

    payload = sample_payload()
    payload['scores'].append({
        'model_id': 'gpt-oss-120b', 'benchmark_id': 'aime_2025', 'score': 950.0,
        'reference_url': 'https://x.invalid', 'source_type': 'tech_report',
        'audit_status': 'verified'})
    monkeypatch.setattr(adapter, 'export_logs', boom)
    with pytest.raises(RuntimeError, match='disk full'):
        adapter.run(_run_args(tmp_path, payload))

    report = json.loads(_report_path(tmp_path).read_text())
    assert [f['source_ref'] for f in report['failed_records']] == [
        'gpt-oss-120b/aime_2025']
