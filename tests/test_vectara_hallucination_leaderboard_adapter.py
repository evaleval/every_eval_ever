"""Tests for the Vectara Hallucination Leaderboard adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.vectara_hallucination_leaderboard import (
    adapter,
)
from every_eval_ever.helpers import save_evaluation_logs
from every_eval_ever.schema import get_schema_version
from every_eval_ever.validate import validate_file

FIXTURE = (
    Path(__file__).parent
    / 'data'
    / 'vectara_hallucination_leaderboard'
    / 'source_rows.json'
)
RETRIEVED_TIMESTAMP = '1779880742.736885'

# 4 aggregate metrics, plus 3 metrics on each of 10 categories and 2 text
# complexity buckets.
EXPECTED_RESULTS_PER_MODEL = 4 + 3 * 12

SOURCE_METRIC_BY_ID = {
    'hallucination-rate': 'hallucination_rate',
    f'{adapter.SRC}.factual-consistency-rate': 'factual_consistency_rate',
    f'{adapter.SRC}.answer-rate': 'answer_rate',
    f'{adapter.SRC}.average-summary-length': 'average_summary_length',
}


@pytest.fixture
def source_rows() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def convert(rows: dict, output_dir: Path):
    return adapter.convert_rows(rows, RETRIEVED_TIMESTAMP, str(output_dir))


def expected_scores(rows: dict) -> dict:
    """Flatten the fixture into (path, slice_kind, slice, metric) -> score."""
    expected = {}
    for path, row in rows.items():
        for key, payload in row['results'].items():
            expected[(path, 'overall', None, key)] = float(payload[key])
        for slice_kind, field in adapter.BREAKDOWN_FIELDS:
            for slice_name, scores in row[field].items():
                for key, value in scores.items():
                    expected[(path, slice_kind, slice_name, key)] = float(value)
    return expected


def emitted_scores(result) -> dict:
    emitted = {}
    for output in result.records:
        log = output.eval_log
        source_file = log.source_metadata.additional_details['source_file']
        for entry in log.evaluation_results:
            config = entry.metric_config
            parameters = config.metric_parameters or {}
            key = (
                source_file,
                parameters['slice_kind'],
                parameters.get('slice'),
                SOURCE_METRIC_BY_ID[config.metric_id],
            )
            emitted[key] = entry.score_details.score
    return emitted


# ===================================================================
# Coverage and fidelity
# ===================================================================


def test_every_source_row_is_converted(source_rows, tmp_path: Path):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)

    assert result.total_records == len(source_rows)
    assert len(result.records) == len(source_rows)
    assert result.failures == []
    assert result.exclusions == []
    for output in result.records:
        assert (
            len(output.eval_log.evaluation_results)
            == EXPECTED_RESULTS_PER_MODEL
        )


def test_scores_round_trip_from_source_without_loss(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)

    assert emitted_scores(result) == expected_scores(source_rows)


def test_category_and_complexity_slices_are_labelled(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    log = result.records[0].eval_log

    names = {entry.evaluation_name for entry in log.evaluation_results}
    assert f'{adapter.SRC}.overall' in names
    assert f'{adapter.SRC}.category.business' in names
    assert f'{adapter.SRC}.text_complexity.high_complexity_text' in names

    business = [
        entry
        for entry in log.evaluation_results
        if entry.evaluation_name == f'{adapter.SRC}.category.business'
    ]
    assert len(business) == 3
    for entry in business:
        assert entry.metric_config.metric_parameters == {
            'slice_kind': 'category',
            'slice': 'business',
        }


# ===================================================================
# Metric metadata
# ===================================================================


def test_canonical_metric_id_is_not_namespaced(source_rows, tmp_path: Path):
    """A registry-canonical metric must not be namespaced, or joins split."""
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    ids = {
        entry.metric_config.metric_id
        for output in result.records
        for entry in output.eval_log.evaluation_results
    }

    assert 'hallucination-rate' in ids
    assert f'{adapter.SRC}.hallucination-rate' not in ids
    # Metrics with no canonical registry id stay namespaced.
    assert f'{adapter.SRC}.answer-rate' in ids


def test_unbounded_length_metric_declares_infinite_upper_bound(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    length = next(
        entry
        for entry in result.records[0].eval_log.evaluation_results
        if entry.metric_config.metric_id.endswith('average-summary-length')
    )

    assert length.metric_config.max_score == float('inf')
    assert length.metric_config.additional_details['diagnostic_metric'] == (
        'true'
    )


def test_emit_source_version_prints_the_pinned_commit(capsys):
    exit_code = adapter.main(['--emit-source-version'])

    assert exit_code == 0
    # One line, the pinned commit, nothing fetched — this is what the
    # scheduler compares to decide whether a run can be skipped.
    assert capsys.readouterr().out.strip() == adapter.SOURCE_COMMIT


def test_private_corpus_is_distinguished_from_public_result_files(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    log = result.records[0].eval_log
    entry = log.evaluation_results[0]

    # The evaluated corpus is private, so it is never given a public locator.
    assert entry.source_data.source_type == 'other'
    assert adapter.EVAL_DATASET_NAME in entry.source_data.dataset_name

    # The public result files are provenance for the log, not the corpus.
    details = log.source_metadata.additional_details
    assert details['structured_results_hf_repo'] == adapter.SOURCE_REPO
    assert details['source_commit'] == adapter.SOURCE_COMMIT
    assert 'not publicly released' in details['evaluated_corpus_availability']


def test_constant_provenance_is_not_repeated_per_result(
    source_rows, tmp_path: Path
):
    """Log-level constants stay at log level.

    Repeating them on all 40 results doubled the size of every record.
    """
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    log = result.records[0].eval_log

    for entry in log.evaluation_results:
        assert entry.source_data.additional_details is None
        assert set(entry.metric_config.additional_details) <= {
            'source_metric_key',
            'diagnostic_metric',
            'direction_note',
        }


# ===================================================================
# Record identity and deployment metadata
# ===================================================================


def test_schema_version_tracks_the_packaged_schema(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)

    for output in result.records:
        assert output.eval_log.schema_version == get_schema_version()


def test_evaluation_id_is_stable_across_refreshes(
    source_rows, tmp_path: Path
):
    """Idempotency must come from the key, not from a frozen timestamp."""
    first = adapter.convert_rows(source_rows, '1', str(tmp_path / 'a'))
    second = adapter.convert_rows(source_rows, '2', str(tmp_path / 'b'))

    def ids(result):
        return sorted(o.eval_log.evaluation_id for o in result.records)

    assert ids(first) == ids(second)
    assert first.records[0].eval_log.retrieved_timestamp == '1'
    assert second.records[0].eval_log.retrieved_timestamp == '2'


def test_model_availability_is_derived_from_source_accessibility(
    source_rows, tmp_path: Path
):
    result = convert(source_rows, tmp_path / 'data' / adapter.COLLECTION)
    availability = {}
    for output in result.records:
        details = output.eval_log.model_info.additional_details
        availability[output.eval_log.model_info.id] = (
            details['source_accessibility'],
            details['model_availability'],
        )

    assert availability['google/gemini-2.5-flash-lite'] == (
        'commercial',
        'closed_weights',
    )
    assert availability['microsoft/Phi-4'] == ('open', 'open_weights')
    # Not recorded by the pinned source files, so it must not be invented.
    for output in result.records:
        assert (
            output.eval_log.model_info.additional_details['deployment_type']
            == 'unknown'
        )


# ===================================================================
# Partial conversions
# ===================================================================


def test_unrepresentable_score_is_recorded_and_the_rest_still_convert(
    source_rows, tmp_path: Path
):
    rows = copy.deepcopy(source_rows)
    path = 'microsoft/Phi-4/results_2025-12-10 14:57:16.944171.json'
    rows[path]['category_results']['business']['hallucination_rate'] = 'n/a'

    result = convert(rows, tmp_path / 'data' / adapter.COLLECTION)

    assert len(result.records) == len(rows)
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert 'business' in failure.source_ref
    assert 'non-numeric' in failure.reason
    phi = next(
        output
        for output in result.records
        if output.eval_log.model_info.id == 'microsoft/Phi-4'
    )
    assert (
        len(phi.eval_log.evaluation_results)
        == EXPECTED_RESULTS_PER_MODEL - 1
    )


def test_out_of_bounds_score_is_rejected(source_rows, tmp_path: Path):
    rows = copy.deepcopy(source_rows)
    path = 'qwen/qwen3-8b/results_2025-12-10 14:57:15.832674.json'
    rows[path]['results']['answer_rate']['answer_rate'] = 101.0

    result = convert(rows, tmp_path / 'data' / adapter.COLLECTION)

    assert len(result.failures) == 1
    assert 'outside declared bounds' in result.failures[0].reason


def test_malformed_row_produces_no_output(source_rows, tmp_path: Path):
    rows = copy.deepcopy(source_rows)
    path = 'qwen/qwen3-8b/results_2025-12-10 14:57:15.832674.json'
    del rows[path]['results']

    result = convert(rows, tmp_path / 'data' / adapter.COLLECTION)

    assert len(result.records) == len(rows) - 1
    assert any(
        failure.source_ref == path and 'results must be an object' in
        failure.reason
        for failure in result.failures
    )
    with pytest.raises(Exception):
        result.raise_if_incomplete()


# ===================================================================
# Datastore publication
# ===================================================================


def test_published_records_pass_the_datastore_gate(
    source_rows, tmp_path: Path
):
    """Semantic checks on, at a real datastore path — what the merge gate runs.

    ``validate_file`` defaults ``run_semantic_checks=False``, so a green
    default-mode test can still hide gate errors.
    """
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result = convert(source_rows, output_dir)
    paths = save_evaluation_logs(result.records)

    assert len(paths) == len(source_rows)
    for path in paths:
        repo_path = str(path.relative_to(tmp_path))
        assert len(Path(repo_path).parts) == 5, repo_path
        report = validate_file(
            path,
            repo_path=repo_path,
            available_files=frozenset(),
            run_semantic_checks=True,
        )
        assert report.valid, report.errors
        assert report.warnings == [], report.warnings


def test_records_land_under_developer_and_model_directories(
    source_rows, tmp_path: Path
):
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result = convert(source_rows, output_dir)
    paths = save_evaluation_logs(result.records)

    relative = {str(path.relative_to(output_dir).parent) for path in paths}
    assert relative == {
        'google/gemini-2.5-flash-lite',
        'microsoft/Phi-4',
        'qwen/qwen3-8b',
    }
    for path in paths:
        assert path.suffix == '.json'
