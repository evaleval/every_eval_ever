"""Offline tests for the LM-Harmony adapter.

Fixture-based, no network: the adapter reads the results matrix through
``--input-json`` and skips registry lookups with ``--no-registry-resolve``, which
is what makes a hermetic test possible without mocking HTTP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.lm_harmony import adapter
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_logs
from every_eval_ever.validator import validate_file

SHA = 'a' * 40
MODELS = ('Qwen/Qwen2.5-7B-Instruct', 'openai-community/gpt2')

#: Two models over four tasks, one per metric family, so every bound and every
#: metric-id branch is exercised. cola carries a negative score on purpose.
SCORES: dict[str, dict[str, dict[str, float]]] = {
    'direct_eval': {
        'sciq': {MODELS[0]: 0.937, MODELS[1]: 0.751},
        'mathqa': {MODELS[0]: 0.40402010050251256, MODELS[1]: 0.2170854271356784},
        'cola': {MODELS[0]: 0.39973614, MODELS[1]: -0.0433786},
        'gsm8k': {MODELS[0]: 0.7551175132676270, MODELS[1]: 0.0075815011372251},
    },
    'train_before_test': {
        'sciq': {MODELS[0]: 0.98, MODELS[1]: 0.926},
        'mathqa': {MODELS[0]: 0.5902847571189279, MODELS[1]: 0.3266331658291457},
        'cola': {MODELS[0]: 0.63737536, MODELS[1]: 0.11902104},
        'gsm8k': {MODELS[0]: 0.8081880212282031, MODELS[1]: 0.0227445034116},
    },
}
STDERRS = {
    'direct_eval': {
        'sciq': {MODELS[0]: 0.0076870078762864, MODELS[1]: 0.0136797},
        'mathqa': {MODELS[0]: 0.0089816, MODELS[1]: 0.0075436},
        'cola': {MODELS[0]: 0.0303244, MODELS[1]: 0.0316741},
        'gsm8k': {MODELS[0]: 0.0118409, MODELS[1]: 0.0023897},
    },
    'train_before_test': {
        'sciq': {MODELS[0]: 0.0044294, MODELS[1]: 0.0082862},
        'mathqa': {MODELS[0]: 0.0089976, MODELS[1]: 0.0085876},
        'cola': {MODELS[0]: 0.0286431, MODELS[1]: 0.0324158},
        'gsm8k': {MODELS[0]: 0.0108406, MODELS[1]: 0.0041020},
    },
}


def _payload(extra_task: str | None = None) -> dict:
    payload = {
        'direct_eval': {k: dict(v) for k, v in SCORES['direct_eval'].items()},
        'train_before_test': {k: dict(v) for k, v in SCORES['train_before_test'].items()},
        'direct_eval_stderr': {k: dict(v) for k, v in STDERRS['direct_eval'].items()},
        'train_before_test_stderr': {
            k: dict(v) for k, v in STDERRS['train_before_test'].items()
        },
    }
    # The three post-cutoff corpora the adapter deliberately excludes.
    for task in adapter.UNDEFINED_TASKS:
        for block in payload:
            payload[block][task] = {MODELS[0]: 1.5}
    if extra_task:
        for block in payload:
            payload[block][extra_task] = {MODELS[0]: 0.5}
    return payload


@pytest.fixture
def payload() -> dict:
    return _payload()


def _convert(payload: dict, tmp_path: Path, **kwargs):
    """Return ``(result, flagged, output_dir)`` for a hermetic conversion."""
    output_dir = tmp_path / 'data' / adapter.COLLECTION
    result, flagged = adapter.convert(
        payload, SHA, '1750000000.0', output_dir,
        resolve_enabled=False, **kwargs,
    )
    return result, flagged, output_dir


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
def test_one_log_per_model_with_every_task_and_both_protocols(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path)
    assert len(result.records) == len(MODELS)
    for output in result.records:
        results = output.eval_log.evaluation_results
        # 4 fixture tasks x 2 protocols
        assert len(results) == 8, [r.evaluation_result_id for r in results]
        assert output.eval_log.schema_version == SCHEMA_VERSION


def test_scores_and_standard_errors_round_trip_exactly(payload, tmp_path):
    """A rescale or a float detour would show up here."""
    result, _, _ = _convert(payload, tmp_path)
    by_model = {
        o.eval_log.model_info.additional_details['source_model_string']: o.eval_log
        for o in result.records
    }
    for model in MODELS:
        for mode in adapter.MODES:
            for task in SCORES[mode]:
                got = next(
                    r for r in by_model[model].evaluation_results
                    if r.evaluation_result_id == f'lm_harmony.{task}.{mode}'
                )
                assert got.score_details.score == SCORES[mode][task][model]
                assert (
                    got.score_details.uncertainty.standard_error.value
                    == STDERRS[mode][task][model]
                )


# --------------------------------------------------------------------------- #
# the protocol split — the whole point of the source
# --------------------------------------------------------------------------- #
def test_the_two_protocols_never_share_a_metric_id(payload, tmp_path):
    """A zero-shot score and a task-trained one must not join on one metric id."""
    result, _, _ = _convert(payload, tmp_path)
    direct, trained = set(), set()
    for output in result.records:
        for r in output.eval_log.evaluation_results:
            target = trained if r.evaluation_result_id.endswith(adapter.TRAINED) else direct
            target.add(r.metric_config.metric_id)
    assert direct & trained == set()
    assert all(m.startswith(f'{adapter.SRC}.{adapter.TRAINED}.') for m in trained)


def test_direct_eval_uses_the_registry_canonical_metric_id(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path)
    ids = {
        r.evaluation_result_id: r.metric_config.metric_id
        for o in result.records for r in o.eval_log.evaluation_results
    }
    assert ids['lm_harmony.sciq.direct_eval'] == 'normalized-accuracy'
    assert ids['lm_harmony.cola.direct_eval'] == 'matthews-correlation'
    assert ids['lm_harmony.gsm8k.direct_eval'] == 'exact-match'


def test_acc_norm_is_not_folded_into_accuracy():
    """`acc_norm` is length-normalized accuracy, a different computation from
    `acc` on the same items. The registry keeps them apart and so must we."""
    assert adapter.METRICS['acc_norm'].registry_id == 'normalized-accuracy'
    assert adapter.METRICS['acc'].registry_id == 'accuracy'
    assert (
        adapter.METRICS['acc_norm'].registry_id
        != adapter.METRICS['acc'].registry_id
    )


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
def test_matthews_correlation_is_bounded_at_minus_one(payload, tmp_path):
    """cola is scored by MCC on [-1, 1]. Under [0, 1] bounds the negative score
    in the fixture is a hard gate error, so this pins the range."""
    result, _, _ = _convert(payload, tmp_path)
    negatives = [
        r for o in result.records for r in o.eval_log.evaluation_results
        if r.score_details.score < 0
    ]
    assert negatives, 'fixture should carry a negative MCC score'
    for r in negatives:
        assert r.metric_config.min_score == -1.0
        assert r.metric_config.max_score == 1.0


def test_every_score_lies_inside_its_declared_bounds(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path)
    for o in result.records:
        for r in o.eval_log.evaluation_results:
            cfg = r.metric_config
            assert cfg.min_score <= r.score_details.score <= cfg.max_score


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_evaluation_id_is_keyed_on_the_pinned_commit_not_the_clock(payload, tmp_path):
    result_a, _, _ = _convert(payload, tmp_path)
    result_b, _, _ = _convert(payload, tmp_path / 'again')
    ids_a = sorted(o.eval_log.evaluation_id for o in result_a.records)
    ids_b = sorted(o.eval_log.evaluation_id for o in result_b.records)
    assert ids_a == ids_b
    assert all(SHA in i for i in ids_a)


def test_source_data_names_the_dataset_and_not_the_results_file(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path)
    sciq = next(
        r for o in result.records for r in o.eval_log.evaluation_results
        if r.evaluation_result_id == 'lm_harmony.sciq.direct_eval'
    )
    assert sciq.source_data.hf_repo == 'allenai/sciq'
    assert sciq.source_data.hf_split == 'test'
    # the legacy spelling the harness actually asked for is kept alongside
    assert sciq.source_data.additional_details['lm_eval_dataset_path'] == 'sciq'


def test_every_task_declares_a_metric_the_metric_table_knows():
    for name, task in adapter.TASKS.items():
        assert task.metric in adapter.METRICS, name


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
def test_undefined_corpora_are_excluded_and_do_not_fail_the_run(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path)
    excluded = {e.source_ref for e in result.exclusions}
    assert excluded == {f'task={t}' for t in adapter.UNDEFINED_TASKS}
    assert result.failures == []
    result.raise_if_incomplete()  # exclusions alone must not raise


def test_an_unrecognised_task_is_a_failure_not_a_silent_skip(tmp_path):
    """A task added upstream must stop the run rather than be dropped: its
    dataset, split and metric are unknown, and guessing them is how a wrong
    number gets published."""
    result, _, _ = _convert(_payload(extra_task='brand_new_task'), tmp_path)
    reasons = [f.reason for f in result.failures]
    assert any('brand_new_task' in r for r in reasons), reasons
    with pytest.raises(Exception):
        result.raise_if_incomplete()


def test_a_missing_block_stops_the_run(tmp_path):
    broken = _payload()
    del broken['train_before_test_stderr']
    with pytest.raises(SystemExit, match='train_before_test_stderr'):
        _convert(broken, tmp_path)


# --------------------------------------------------------------------------- #
# the merge gate
# --------------------------------------------------------------------------- #
def test_published_records_pass_the_datastore_gate(payload, tmp_path):
    """Semantic checks on, at a real datastore path — what the merge gate runs.

    ``validate_file`` defaults ``run_semantic_checks=False``, so a green
    default-mode test can still hide gate errors.
    """
    result, _, output_dir = _convert(payload, tmp_path)
    paths = save_evaluation_logs(result.records)
    assert len(paths) == len(MODELS)
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


def test_records_land_under_the_model_ids_own_developer_directory(payload, tmp_path):
    result, _, output_dir = _convert(payload, tmp_path)
    paths = save_evaluation_logs(result.records)
    got = {str(p.relative_to(output_dir).parent) for p in paths}
    assert got == {'Qwen/Qwen2.5-7B-Instruct', 'openai-community/gpt2'}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_replay_without_an_explicit_revision_is_refused(tmp_path, monkeypatch):
    """Replayed bytes must name the commit they came from, or the record would
    cite whatever `main` happens to be at conversion time."""
    src = tmp_path / 'all_results.json'
    src.write_text(json.dumps(_payload()))
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    args = adapter.parse_args([
        '--input-json', str(src),
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    with pytest.raises(SystemExit, match='--revision'):
        adapter.run(args)


def test_an_unresolvable_revision_stops_the_run_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: None)
    args = adapter.parse_args([
        '--revision', 'no-such-branch',
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    with pytest.raises(SystemExit, match='could not resolve'):
        adapter.run(args)


def test_output_dir_must_be_the_collection_directory(tmp_path, monkeypatch):
    """Deriving the collection below a parent would publish somewhere other than
    the path every message names."""
    src = tmp_path / 'all_results.json'
    src.write_text(json.dumps(_payload()))
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    wrong = tmp_path / 'data' / 'not-the-collection'
    args = adapter.parse_args([
        '--input-json', str(src), '--revision', SHA,
        '--no-registry-resolve', '--output-dir', str(wrong),
    ])
    with pytest.raises(SystemExit, match=adapter.COLLECTION):
        adapter.run(args)
    assert not list(tmp_path.glob('data/**/*.json'))


def test_a_second_run_into_a_populated_directory_is_refused(tmp_path, monkeypatch):
    """Filenames are fresh uuid4s, so an unguarded re-run adds a second copy of
    every evaluation_id rather than replacing it."""
    src = tmp_path / 'all_results.json'
    src.write_text(json.dumps(_payload()))
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    argv = [
        '--input-json', str(src), '--revision', SHA, '--no-registry-resolve',
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ]
    first = adapter.run(adapter.parse_args(argv))
    assert len(first) == len(MODELS)
    with pytest.raises(SystemExit, match='already exist'):
        adapter.run(adapter.parse_args(argv))
    second = adapter.run(adapter.parse_args([*argv, '--replace-existing']))
    assert len(second) == len(MODELS)
    # replaced, not accumulated
    assert len(list((tmp_path / 'data' / adapter.COLLECTION).glob('*/*/*.json'))) == len(MODELS)


def test_emit_source_version_prints_the_commit_and_converts_nothing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(adapter, 'resolve_commit', lambda revision, **kw: SHA)
    args = adapter.parse_args([
        '--emit-source-version',
        '--output-dir', str(tmp_path / 'data' / adapter.COLLECTION),
    ])
    assert adapter.run(args) == []
    assert capsys.readouterr().out.strip() == SHA
    assert not list(tmp_path.glob('data/**/*.json'))


def test_limit_bounds_the_models_converted(payload, tmp_path):
    result, _, _ = _convert(payload, tmp_path, limit=1)
    assert len(result.records) == 1
