"""Offline unit tests for the MTEB adapter.

Fixtures under ``tests/data/mteb/results`` are real score files from
``embeddings-benchmark/results``, trimmed to a few subsets, plus one synthetic
high-fan-out task. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.mteb import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers.eval_card_registry import Registry

FIXTURES = Path(__file__).parent / 'data' / 'mteb' / 'results'
COMMIT = '0' * 40
SMALL_MODEL = 'fixture-org/small-model'


def convert(tmp_path: Path, **kwargs):
    return adapter.convert(
        FIXTURES,
        tmp_path,
        source_commit=COMMIT,
        registry=Registry(enabled=False),
        **kwargs,
    )


def logs_by_model(result) -> dict[str, EvaluationLog]:
    return {
        output.eval_log.model_info.name: output.eval_log
        for output in result.records
    }


def results_by_task(log: EvaluationLog) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for result in log.evaluation_results:
        grouped.setdefault(result.source_data.dataset_name, []).append(result)
    return grouped


# -- shape -----------------------------------------------------------------


def test_converts_each_run_with_in_scope_tasks(tmp_path):
    result = convert(tmp_path)
    models = logs_by_model(result)
    assert set(models) == {SMALL_MODEL, 'fixture-org/closed-model'}


def test_every_record_validates_against_the_schema(tmp_path):
    for output in convert(tmp_path).records:
        EvaluationLog.model_validate(output.eval_log.model_dump())


def test_record_routes_to_the_collection_directory(tmp_path):
    output = next(
        o
        for o in convert(tmp_path).records
        if o.eval_log.model_info.name == SMALL_MODEL
    )
    assert Path(output.base_dir) == tmp_path
    assert output.developer == 'fixture-org'
    assert output.model_name == 'small-model'


def test_evaluation_id_is_stable_across_runs(tmp_path):
    first = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    second = logs_by_model(convert(tmp_path / 'again'))[SMALL_MODEL]
    assert first.evaluation_id == second.evaluation_id
    assert first.evaluation_id == (
        'mteb/fixture-org/small-model/0123456789abcdef0123456789abcdef01234567'
    )


def test_retrieved_timestamp_is_not_the_evaluation_timestamp(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    # The dump records how long each task took but never when it ran, so no
    # run timestamp is asserted.
    assert log.evaluation_timestamp is None
    assert float(log.retrieved_timestamp) > 0


# -- metrics ---------------------------------------------------------------


def test_retrieval_task_publishes_ndcg_at_10_with_registry_id(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    result = results_by_task(log)['AILAStatutes'][0]
    assert result.metric_config.metric_name == 'ndcg_at_10'
    assert result.metric_config.metric_id == 'ndcg-at-10'
    assert result.metric_config.additional_details['metric_id_status'] == (
        'eval_card_registry'
    )
    assert (result.metric_config.min_score, result.metric_config.max_score) == (
        0.0,
        1.0,
    )
    assert result.metric_config.lower_is_better is False


def test_correlation_metric_allows_negative_bounds(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    result = results_by_task(log)['STS17'][0]
    assert result.metric_config.metric_name == 'cosine_spearman'
    assert (result.metric_config.min_score, result.metric_config.max_score) == (
        -1.0,
        1.0,
    )
    assert (
        result.metric_config.additional_details['metric_id_status']
        == 'proposed'
    )


def test_legacy_reranking_spelling_is_published_under_its_own_name(tmp_path):
    """An older file spells the reranking metric ``map``, not ``map_at_1000``.

    The two never co-occur, so there is no run in which they can be checked
    against each other. The row is published as what its own file says, with
    the currently declared name recorded beside it.
    """
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    result = results_by_task(log)['AlloprofReranking'][0]
    assert result.metric_config.metric_name == 'map'
    assert result.metric_config.metric_id == 'map'
    assert result.metric_config.additional_details[
        'mteb_declared_main_score'
    ] == ('map_at_1000')


def test_score_matches_the_named_submetric_in_the_source(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    payload = json.loads(
        (
            FIXTURES
            / 'fixture-org__small-model'
            / '0123456789abcdef0123456789abcdef01234567'
            / 'AILAStatutes.json'
        ).read_text(encoding='utf-8')
    )
    entry = payload['scores']['test'][0]
    result = results_by_task(log)['AILAStatutes'][0]
    assert (
        result.score_details.score == entry['main_score'] == entry['ndcg_at_10']
    )


def test_unmatched_declared_metric_fails_the_row_rather_than_guessing(tmp_path):
    """``max_over_subqueries_map_at_1000`` also spells as ``map`` in old files.

    Publishing it as plain MAP would merge a multi-subquery protocol's score
    into the ordinary metric, so the row is failed instead.
    """
    result = convert(tmp_path)
    reasons = [f.reason for f in result.failures]
    assert any('max_over_subqueries_map_at_1000' in r for r in reasons)
    assert all('MindSmallReranking' in f.source_ref for f in result.failures)


# -- model metadata --------------------------------------------------------


def test_model_availability_comes_from_open_weights(tmp_path):
    models = logs_by_model(convert(tmp_path))
    open_details = models[SMALL_MODEL].model_info.additional_details
    closed_details = models[
        'fixture-org/closed-model'
    ].model_info.additional_details
    assert open_details['model_availability'] == 'open_weights'
    assert closed_details['model_availability'] == 'closed_weights'
    # MTEB runs the encoder locally in both cases.
    assert open_details['deployment_type'] == 'self_deployed'
    assert closed_details['deployment_type'] == 'self_deployed'


def test_eval_library_records_the_mteb_version_that_wrote_the_files(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    assert log.eval_library.name == 'mteb'
    assert log.eval_library.version != 'unknown'
    versions = log.source_metadata.additional_details['mteb_versions_in_run']
    assert log.eval_library.version in versions


def test_training_dataset_overlap_is_recorded(tmp_path):
    """The fixture model declares STS17 as a training dataset."""
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    by_task = results_by_task(log)
    sts = by_task['STS17'][0].metric_config.additional_details
    aila = by_task['AILAStatutes'][0].metric_config.additional_details
    assert sts['model_trained_on_this_dataset'] == 'true'
    assert aila['model_trained_on_this_dataset'] == 'false'


def test_source_data_pins_the_evaluation_dataset(tmp_path):
    log = logs_by_model(convert(tmp_path))[SMALL_MODEL]
    source = results_by_task(log)['AILAStatutes'][0].source_data
    assert source.hf_repo == 'mteb/AILA_statutes'
    assert source.hf_split == 'test'
    assert source.additional_details['dataset_revision']


# -- fan-out limit ---------------------------------------------------------


def test_high_fanout_task_is_excluded_by_default_with_its_count(tmp_path):
    result = convert(tmp_path)
    skips = [
        e for e in result.exclusions if 'SIB200ClusteringS2S' in e.source_ref
    ]
    assert len(skips) == 1
    assert '160' in skips[0].reason
    log = logs_by_model(result)[SMALL_MODEL]
    assert 'SIB200ClusteringS2S' not in results_by_task(log)


def test_fanout_limit_can_be_disabled(tmp_path):
    log = logs_by_model(convert(tmp_path, max_subsets_per_task=0))[SMALL_MODEL]
    assert len(results_by_task(log)['SIB200ClusteringS2S']) == 160


# -- accounting ------------------------------------------------------------


def test_run_without_model_meta_is_excluded(tmp_path):
    result = convert(tmp_path)
    assert any(
        'no-meta' in e.source_ref and 'model_meta.json' in e.reason
        for e in result.exclusions
    )


def test_run_with_no_in_scope_task_is_excluded(tmp_path):
    result = convert(tmp_path)
    assert any(
        'out-of-scope' in e.source_ref and 'scoped suites' in e.reason
        for e in result.exclusions
    )


def test_every_source_run_is_either_converted_or_accounted_for(tmp_path):
    result = convert(tmp_path)
    excluded_runs = {
        e.source_ref
        for e in result.exclusions
        if '/' not in e.source_ref.split('@')[-1]
    }
    assert result.total_records == len(result.records) + len(excluded_runs)


def test_limit_caps_the_number_of_records(tmp_path):
    assert len(convert(tmp_path, limit=1).records) == 1


def test_models_filter_restricts_the_run(tmp_path):
    result = convert(tmp_path, models={SMALL_MODEL})
    assert [o.eval_log.model_info.name for o in result.records] == [SMALL_MODEL]


# -- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ('directory', 'expected'),
    [
        ('intfloat__multilingual-e5-small', 'intfloat/multilingual-e5-small'),
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
    ],
)
def test_model_repo_from_dir(directory, expected):
    assert adapter.model_repo_from_dir(directory) == expected


def test_every_task_in_the_snapshot_has_a_publishable_metric():
    """A task whose main_score has no MetricSpec would fail every one of its rows.

    This is the check that catches a snapshot refresh introducing a metric the
    adapter has no bounds or id for.
    """
    metadata = adapter.load_task_metadata()
    unmapped = sorted(
        {
            task['main_score']
            for task in metadata['tasks'].values()
            if task['main_score'] not in adapter.METRICS
        }
    )
    assert unmapped == []


def test_legacy_spellings_name_metrics_the_adapter_can_publish():
    for declared, older in adapter.LEGACY_MAIN_SCORE_SPELLINGS.items():
        assert declared in adapter.METRICS, declared
        for name in older:
            assert name in adapter.METRICS, name


def test_every_metric_declares_bounds_containing_its_own_scale():
    for name, spec in adapter.METRICS.items():
        assert spec.min_score < spec.max_score, name
        assert spec.metric_id, name
        # A spec is either registry-backed or explicitly a proposal, never both
        # and never neither.
        assert bool(spec.registry_id) != bool(spec.proposed_id), name


def test_snapshot_tasks_carry_the_provenance_the_records_need():
    metadata = adapter.load_task_metadata()
    for name, task in metadata['tasks'].items():
        assert task['hf_repo'], name
        assert task['dataset_revision'], name
        assert task['main_score'], name


def test_suites_cover_every_task_in_the_snapshot():
    metadata = adapter.load_task_metadata()
    listed = {name for names in metadata['suites'].values() for name in names}
    assert listed == set(metadata['tasks'])
