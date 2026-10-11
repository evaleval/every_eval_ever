"""Unit tests for the ARC Prize leaderboard adapter.

The run-log fixture under ``tests/data/arc_agi/v2/`` is three task files
(``7b5033c1``, ``58490d8a``, ``28a6681f``) from the directory
``claude-opus-4-5-20251101-thinking-16k`` of the Hugging Face dataset
``arcprize/arc_agi_v2_public_eval`` at commit
``026789c1c12a4c34580a32e84dcaf5630d7e8f31``, with each attempt's
``choices`` and ``reasoning_summary`` removed. Its ``results.json`` keeps
those three ``task_results`` entries verbatim and recomputes the run totals
from them (score 2.0 of 3 tasks, average cost per task 0.470395).
``tests/data/arc_agi/history/v2.json`` is that directory's path history at
the same commit (``git log --name-only --no-renames`` over a blobless clone,
restricted to the directory): every path committed once.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from every_eval_ever.adapters.arc_agi import adapter, run_logs
from every_eval_ever.adapters.arc_agi.run_logs import StatedFloat
from every_eval_ever.eval_types import EvaluationLog


def sample_payload() -> dict:
    return {
        'datasets': [
            {'id': 'v1_Semi_Private', 'displayName': 'ARC-AGI-1', 'url': ''},
            {'id': 'v2_Semi_Private', 'displayName': 'ARC-AGI-2', 'url': ''},
        ],
        'providers': [
            {'id': 'Human', 'displayName': 'Human', 'url': ''},
            {'id': 'ARC Prize 2024', 'displayName': 'ARC Prize 2024', 'url': ''},
            {'id': 'Anthropic', 'displayName': 'Anthropic', 'url': ''},
            {'id': 'OpenAI', 'displayName': 'OpenAI', 'url': ''},
            {'id': 'New Lab', 'displayName': 'New Lab', 'url': ''},
        ],
        'models': [
            {
                'id': '2025_human_panel',
                'displayName': 'Human Panel',
                'providerId': 'Human',
                'modelType': None,
                'modelGroup': 'Human',
                'modelReleaseDate': None,
            },
            {
                'id': 'ARChitects',
                'displayName': 'ARChitects',
                'providerId': 'ARC Prize 2024',
                'modelType': 'Custom',
                'modelGroup': 'Kaggle',
            },
            {
                'id': 'anthropic-claude-fable-5-high',
                'displayName': 'Claude Fable 5 (High)',
                'providerId': 'Anthropic',
                'modelType': 'CoT',
                'modelGroup': 'anthropic-claude-fable-5',
                'modelReleaseDate': '2026-05-01',
            },
            {
                'id': 'o4-mini',
                'displayName': 'o4-mini',
                'providerId': 'OpenAI',
                'modelType': 'CoT',
            },
            {
                'id': 'openai-o4-mini',
                'displayName': 'o4-mini',
                'providerId': 'OpenAI',
                'modelType': 'CoT',
            },
            {
                'id': 'shiny-new-model',
                'displayName': 'Shiny New Model',
                'providerId': 'New Lab',
            },
        ],
        'evaluations': [
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': '2025_human_panel',
                'score': 0.98,
                'costPerTask': 17,
                'resultsUrl': '',
                'display': True,
            },
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': 'ARChitects',
                'score': 0.535,
                'cost': 50,
                'resultsUrl': '',
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'anthropic-claude-fable-5-high',
                'score': 0.29,
                'costPerTask': 8.42,
                'resultsUrl': '',
                'display': True,
                'labelOffsetX': 12,
            },
            # Two raw ids that slugify to the same canonical OpenAI model,
            # on two different datasets.
            {
                'datasetId': 'v1_Semi_Private',
                'modelId': 'o4-mini',
                'score': 0.41,
                'costPerTask': 0.23,
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'openai-o4-mini',
                'score': 0.02,
                'costPerTask': 0.31,
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'shiny-new-model',
                'score': 0.05,
                'costPerTask': 1.0,
                'display': True,
            },
            # Hidden rows never convert.
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': 'hidden-model',
                'score': 0.99,
                'display': False,
            },
        ],
    }


def convert(payload: dict) -> dict[str, EvaluationLog]:
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    result.raise_if_incomplete()
    return {log.model_info.id: log for log, _, _ in result.records}


def test_converts_each_canonical_model_once():
    logs = convert(sample_payload())
    assert sorted(logs) == [
        'anthropic/claude-fable-5-high',
        'arcprize/2025-human-panel',
        'community/architects',
        'new-lab/shiny-new-model',
        'openai/o4-mini',
    ]


def test_developer_comes_from_provider_table():
    logs = convert(sample_payload())
    assert logs['anthropic/claude-fable-5-high'].model_info.developer == 'anthropic'
    # Kaggle-winner systems keep the historical 'community' developer.
    assert logs['community/architects'].model_info.developer == 'community'
    # The human panel keeps the historical 'arcprize' developer.
    assert logs['arcprize/2025-human-panel'].model_info.developer == 'arcprize'
    # An unmapped provider falls back to a slug of its id.
    assert logs['new-lab/shiny-new-model'].model_info.developer == 'new-lab'


def test_aliases_merge_into_one_log():
    logs = convert(sample_payload())
    log = logs['openai/o4-mini']
    aliases = json.loads(
        log.model_info.additional_details['raw_model_aliases_json']
    )
    assert aliases == ['o4-mini', 'openai-o4-mini']
    result_ids = [r.evaluation_result_id for r in log.evaluation_results]
    assert result_ids == [
        'v1_Semi_Private::score',
        'v1_Semi_Private::cost_per_task',
        'v2_Semi_Private::score',
        'v2_Semi_Private::cost_per_task',
    ]


def test_score_and_cost_results_carry_source_fields():
    logs = convert(sample_payload())
    log = logs['anthropic/claude-fable-5-high']
    score_result, cost_result = log.evaluation_results
    assert score_result.evaluation_result_id == 'v2_Semi_Private::score'
    assert score_result.score_details.score == 0.29
    assert score_result.metric_config.max_score == 1.0
    assert (
        score_result.source_data.additional_details['dataset_display_name']
        == 'ARC-AGI-2'
    )
    # Chart-layout fields stay out of the record.
    assert 'labelOffsetX' not in score_result.score_details.details

    assert cost_result.evaluation_result_id == 'v2_Semi_Private::cost_per_task'
    assert cost_result.score_details.score == 8.42
    assert cost_result.metric_config.lower_is_better is True
    # Bounds come from the largest observed value across the payload.
    assert cost_result.metric_config.max_score == 17.0


def test_cost_metric_used_when_cost_per_task_missing():
    logs = convert(sample_payload())
    log = logs['community/architects']
    result_ids = [r.evaluation_result_id for r in log.evaluation_results]
    assert result_ids == ['v1_Semi_Private::score', 'v1_Semi_Private::cost']


def test_model_metadata_from_models_table():
    logs = convert(sample_payload())
    details = logs['anthropic/claude-fable-5-high'].model_info.additional_details
    assert details['source_model_type'] == 'CoT'
    assert details['source_provider_id'] == 'Anthropic'
    assert details['model_release_date'] == '2026-05-01'
    assert (
        logs['anthropic/claude-fable-5-high'].model_info.name
        == 'Claude Fable 5 (High)'
    )


def test_unknown_model_id_is_an_accounted_failure():
    payload = sample_payload()
    payload['evaluations'].append(
        {
            'datasetId': 'v2_Semi_Private',
            'modelId': 'not-in-models-json',
            'score': 0.5,
            'display': True,
        }
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    assert len(result.failures) == 1
    assert 'not in models.json' in result.failures[0].reason
    with pytest.raises(ValueError):
        result.raise_if_incomplete()


def test_out_of_range_score_is_an_accounted_failure():
    payload = sample_payload()
    payload['evaluations'].append(
        {
            'datasetId': 'v2_Semi_Private',
            'modelId': 'o4-mini',
            'score': 1.5,
            'display': True,
        }
    )
    result = adapter.convert_logs(payload, retrieved_timestamp='123.0')
    assert len(result.failures) == 1
    assert 'proportion' in result.failures[0].reason


def test_bare_list_payload_is_rejected(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps([{'modelId': 'x'}]), encoding='utf-8')
    with pytest.raises(ValueError, match='evaluations'):
        adapter.load_payload_file(path)


def test_run_writes_records_and_replays_offline(tmp_path):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps(sample_payload()), encoding='utf-8')
    out_dir = tmp_path / 'out'
    args = adapter.parse_args(
        [
            '--input-json',
            str(payload_path),
            '--output-dir',
            str(out_dir),
            '--no-run-logs',
        ]
    )
    written = adapter.run(args)
    assert written == 5
    files = sorted(out_dir.glob('*/*/*.json'))
    assert len(files) == 5
    record = json.loads(files[0].read_text(encoding='utf-8'))
    assert record['schema_version']
    assert record['evaluation_id'].startswith('arc-agi/')


RUN_FIXTURE = Path(__file__).parent / 'data' / 'arc_agi'
RUN_ID = 'claude-opus-4-5-20251101-thinking-16k'
ALIAS_ID = 'anthropic-claude-opus-4-5-20251101-thinking-16k'
V2 = run_logs.RUN_DATASETS['v2_Public_Eval']
RETRIEVED = '1790000000.0'
FIXTURE_HISTORY = run_logs.load_history_file(
    RUN_FIXTURE / 'history' / 'v2.json', V2.revision
)


def run_log_payload(**v2_row: object) -> dict:
    row = {
        'datasetId': 'v2_Public_Eval',
        'modelId': RUN_ID,
        'score': 0.6667,
        'costPerTask': 0.47,
        'display': True,
    }
    row.update(v2_row)
    return {
        'datasets': [],
        'providers': [{'id': 'Anthropic', 'displayName': 'Anthropic'}],
        'models': [
            {'id': model_id, 'providerId': 'Anthropic'}
            for model_id in (RUN_ID, ALIAS_ID)
        ],
        'evaluations': [
            row,
            {
                'datasetId': 'v1_Public_Eval',
                'modelId': RUN_ID,
                'score': 0.6667,
                'costPerTask': 0.47,
                'display': True,
            },
            {
                'datasetId': 'v2_Semi_Private',
                'modelId': RUN_ID,
                'score': 0.6667,
                'costPerTask': 0.47,
                'display': True,
            },
        ],
    }


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / 'runs'
    shutil.copytree(RUN_FIXTURE, root)
    return root


def run_dir(root: Path, name: str = RUN_ID) -> Path:
    return root / 'v2' / name


def edit_attempts(directory: Path, edit, task_id: str | None = None) -> None:
    """Apply ``edit(metadata, task_id, attempt_key)`` to every recorded attempt."""
    for path in sorted(directory.glob('*.json')):
        if path.name == 'results.json' or task_id not in (None, path.stem):
            continue
        pairs = json.loads(path.read_text(encoding='utf-8'))
        for pair in pairs:
            for key, attempt in pair.items():
                if attempt is not None:
                    edit(attempt['metadata'], path.stem, key)
        path.write_text(json.dumps(pairs), encoding='utf-8')


def convert_with_logs(
    payload: dict,
    root: Path,
    history: dict[str, int] = FIXTURE_HISTORY,
    retrieved: str = RETRIEVED,
) -> tuple[EvaluationLog, run_logs.RunLogs]:
    logs = run_logs.RunLogs(
        {
            dataset_id: run_logs.RunLogSource(
                dataset, root / dataset.subdir, dataset.revision, history
            )
            for dataset_id, dataset in run_logs.RUN_DATASETS.items()
        }
    )
    result = adapter.convert_logs(
        payload, retrieved_timestamp=retrieved, run_logs=logs
    )
    result.raise_if_incomplete()
    [(log, _, _)] = result.records
    return log, logs


def result_by_id(log: EvaluationLog, result_id: str):
    return next(
        r for r in log.evaluation_results if r.evaluation_result_id == result_id
    )


def test_accepted_run_fills_the_score_result(runs_dir):
    log, logs = convert_with_logs(run_log_payload(), runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    config = score.generation_config
    assert config.generation_args.max_tokens == 64000
    assert config.generation_args.temperature is None
    assert config.additional_details == {
        'cap_key': 'max_tokens',
        'request_log': f'{V2.repo_id}@{V2.revision}/{RUN_ID}',
    }
    assert score.source_data.additional_details['scored_tasks'] == '3'
    assert [(o.dataset_id, o.rejection) for o in logs.outcomes] == [
        ('v2_Public_Eval', None)
    ]


def test_cost_result_and_other_sets_are_untouched(runs_dir):
    log, _ = convert_with_logs(run_log_payload(), runs_dir)
    for result_id in (
        'v2_Public_Eval::cost_per_task',
        'v1_Public_Eval::score',
        'v1_Public_Eval::cost_per_task',
        'v2_Semi_Private::score',
        'v2_Semi_Private::cost_per_task',
    ):
        result = result_by_id(log, result_id)
        assert result.generation_config is None, result_id
        assert 'scored_tasks' not in result.source_data.additional_details


def test_v2_run_never_fills_v1_even_when_v1_has_no_directory(runs_dir):
    (runs_dir / 'v1').mkdir()
    log, logs = convert_with_logs(run_log_payload(), runs_dir)
    assert result_by_id(log, 'v1_Public_Eval::score').generation_config is None
    assert {o.dataset_id for o in logs.outcomes} == {'v2_Public_Eval'}
    assert logs.unjoined == [('v1_Public_Eval', RUN_ID)]
    assert adapter.run_log_report(logs)['gate_counts'] == {
        'filled': 1,
        'no_run_directory': 1,
    }


def _set_cap(metadata, task_id, key):
    if (task_id, key) == ('28a6681f', 'attempt_2'):
        metadata['kwargs']['max_tokens'] = 32000


def _mix_cap_key(metadata, task_id, key):
    if (task_id, key) == ('28a6681f', 'attempt_2'):
        metadata['kwargs']['max_output_tokens'] = metadata['kwargs'].pop(
            'max_tokens'
        )


def _wrong_test_id(metadata, task_id, key):
    if (task_id, key) == ('58490d8a', 'attempt_1'):
        metadata['test_id'] = 'some-other-run'


@pytest.mark.parametrize(
    ('row', 'mutate', 'gate'),
    [
        ({'score': 0.6}, None, 'score_mismatch'),
        ({'costPerTask': 0.48}, None, 'cost_mismatch'),
        ({}, lambda d: edit_attempts(d, _set_cap), 'cap_conflict'),
        ({}, lambda d: edit_attempts(d, _mix_cap_key), 'cap_key_mixed'),
        ({}, lambda d: (d / 'results.json').unlink(), 'no_results_file'),
        ({}, lambda d: edit_attempts(d, _wrong_test_id), 'test_id_mismatch'),
    ],
    ids=[
        'score',
        'cost',
        'cap_conflict',
        'mixed_cap_keys',
        'no_results',
        'test_id',
    ],
)
def test_failed_gate_leaves_the_result_unfilled(runs_dir, row, mutate, gate):
    if mutate is not None:
        mutate(run_dir(runs_dir))
    log, _ = convert_with_logs(run_log_payload(**row), runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    assert score.generation_config.generation_args is None
    assert score.generation_config.additional_details == {
        'request_args_unavailable': gate,
        'request_log': f'{V2.repo_id}@{V2.revision}/{RUN_ID}',
    }
    assert 'scored_tasks' not in score.source_data.additional_details
    cost = result_by_id(log, 'v2_Public_Eval::cost_per_task')
    assert cost.generation_config is None


def test_score_is_compared_at_the_row_precision(runs_dir):
    log, _ = convert_with_logs(run_log_payload(score=0.667), runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    assert score.generation_config.generation_args.max_tokens == 64000


def test_run_ending_after_retrieval_is_rejected(runs_dir):
    log, _ = convert_with_logs(
        run_log_payload(), runs_dir, retrieved='1700000000.0'
    )
    details = result_by_id(
        log, 'v2_Public_Eval::score'
    ).generation_config.additional_details
    assert details['request_args_unavailable'] == 'run_ended_after_retrieval'


def test_rewritten_path_in_history_is_rejected(runs_dir):
    history = {f'{RUN_ID}/results.json': 1, f'{RUN_ID}/7b5033c1.json': 2}
    log, _ = convert_with_logs(run_log_payload(), runs_dir, history=history)
    details = result_by_id(
        log, 'v2_Public_Eval::score'
    ).generation_config.additional_details
    assert details['request_args_unavailable'] == 'run_rewritten'


def test_temperature_only_when_every_attempt_states_it(runs_dir):
    def set_temperature(metadata, task_id, key):
        metadata['kwargs']['temperature'] = 1.0

    edit_attempts(run_dir(runs_dir), set_temperature)
    log, _ = convert_with_logs(run_log_payload(), runs_dir)
    args = result_by_id(log, 'v2_Public_Eval::score').generation_config
    assert args.generation_args.temperature == 1.0

    def drop_one(metadata, task_id, key):
        if key == 'attempt_2':
            metadata['kwargs'].pop('temperature')

    edit_attempts(run_dir(runs_dir), drop_one, task_id='28a6681f')
    log, _ = convert_with_logs(run_log_payload(), runs_dir)
    args = result_by_id(log, 'v2_Public_Eval::score').generation_config
    assert args.generation_args.temperature is None
    assert args.generation_args.max_tokens == 64000


def test_unrecorded_attempt_rejects_the_run(runs_dir):
    path = run_dir(runs_dir) / '28a6681f.json'
    pairs = json.loads(path.read_text(encoding='utf-8'))
    pairs[0]['attempt_2'] = None
    path.write_text(json.dumps(pairs), encoding='utf-8')
    log, _ = convert_with_logs(run_log_payload(), runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    assert score.generation_config.generation_args is None
    assert (
        score.generation_config.additional_details['request_args_unavailable']
        == 'attempt_not_recorded'
    )
    assert 'scored_tasks' not in score.source_data.additional_details


@pytest.mark.parametrize(
    ('stated', 'filled'), [('0.470', True), ('0.4700', False)]
)
def test_cost_is_compared_at_its_written_precision(runs_dir, stated, filled):
    results_path = run_dir(runs_dir) / 'results.json'
    results = json.loads(results_path.read_text(encoding='utf-8'))
    results['avg_cost_per_task'] = 0.4704
    results_path.write_text(json.dumps(results), encoding='utf-8')
    log, _ = convert_with_logs(
        run_log_payload(costPerTask=StatedFloat(stated)), runs_dir
    )
    config = result_by_id(log, 'v2_Public_Eval::score').generation_config
    if filled:
        assert config.generation_args.max_tokens == 64000
    else:
        assert config.generation_args is None
        assert config.additional_details['request_args_unavailable'] == (
            'cost_mismatch'
        )


def test_payload_parsing_keeps_written_precision(tmp_path):
    texts = {
        'evaluations': '[{"modelId": "m", "score": 0.4700, "costPerTask": 1}]',
        'models': '[]',
        'providers': '[]',
        'datasets': '[]',
    }
    path = tmp_path / 'raw.json'
    path.write_text(
        adapter.combined_payload_text(texts, 'https://example.test', '1.0'),
        encoding='utf-8',
    )
    [row] = adapter.load_payload_file(path)['evaluations']
    assert row['score'].text == '0.4700'
    assert adapter.parse_payload_texts(texts)['evaluations'][0][
        'score'
    ].text == ('0.4700')


def test_join_uses_the_chosen_rows_model_id(runs_dir):
    payload = run_log_payload()
    alias_row = dict(payload['evaluations'][0], modelId=ALIAS_ID)
    payload['evaluations'].append(alias_row)
    log, _ = convert_with_logs(payload, runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    assert score.score_details.details['raw_model_id'] == RUN_ID
    assert score.generation_config.generation_args.max_tokens == 64000

    run_dir(runs_dir).rename(run_dir(runs_dir, ALIAS_ID))
    edit_attempts(
        run_dir(runs_dir, ALIAS_ID),
        lambda metadata, task_id, key: metadata.update(test_id=ALIAS_ID),
    )
    log, logs = convert_with_logs(payload, runs_dir)
    score = result_by_id(log, 'v2_Public_Eval::score')
    assert score.generation_config is None
    assert logs.outcomes == []


def test_output_is_deterministic_for_a_fixed_directory(runs_dir):
    first, _ = convert_with_logs(run_log_payload(), runs_dir)
    second, _ = convert_with_logs(run_log_payload(), runs_dir)
    assert first.model_dump_json() == second.model_dump_json()


def test_run_reads_a_local_runs_dir(tmp_path, runs_dir):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps(run_log_payload()), encoding='utf-8')
    out_dir = tmp_path / 'data' / 'arc-agi'
    args = adapter.parse_args(
        [
            '--input-json',
            str(payload_path),
            '--output-dir',
            str(out_dir),
            '--arc-runs-dir',
            str(runs_dir),
            '--arc-history-dir',
            str(runs_dir / 'history'),
        ]
    )
    assert adapter.run(args) == 1
    report = json.loads(
        (tmp_path / 'adapter_reports' / 'arc-agi_failures.json').read_text(
            encoding='utf-8'
        )
    )
    assert report['failed_record_count'] == 0
    evidence = report['run_log_evidence']
    assert evidence['sources'] == {
        'v2_Public_Eval': {'dataset': V2.repo_id, 'revision': V2.revision}
    }
    assert evidence['gate_counts'] == {'filled': 1}
    assert evidence['regressions'] == []
    assert evidence['unavailable'] == []
    assert evidence['filled'] == [
        {
            'model_id': RUN_ID,
            'dataset_id': 'v2_Public_Eval',
            'directory': RUN_ID,
            'cap_key': 'max_tokens',
            'max_tokens': 64000,
            'temperature': None,
            'scored_tasks': 3,
        }
    ]
    [path] = out_dir.glob('*/*/*.json')
    record = json.loads(path.read_text(encoding='utf-8'))
    [score] = [
        r
        for r in record['evaluation_results']
        if r['evaluation_result_id'] == 'v2_Public_Eval::score'
    ]
    assert score['generation_config']['generation_args'] == {
        'max_tokens': 64000
    }


def test_public_model_ids_cover_only_sets_with_run_logs():
    payload = run_log_payload()
    payload['evaluations'].append(
        {'datasetId': 'v2_Public_Eval', 'modelId': 'hidden', 'display': False}
    )
    assert adapter.public_model_ids(payload) == {
        'v1_Public_Eval': {RUN_ID},
        'v2_Public_Eval': {RUN_ID},
    }


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        [
            'git',
            '-c',
            'user.name=t',
            '-c',
            'user.email=t@example.com',
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def dataset_repo(tmp_path: Path) -> Path:
    if shutil.which('git') is None:
        pytest.skip('git not available')
    repo = tmp_path / 'origin'
    shutil.copytree(RUN_FIXTURE / 'v2', repo)
    other = repo / 'unrelated-run'
    other.mkdir()
    (other / 'results.json').write_text('{}', encoding='utf-8')
    _git('init', '--quiet', cwd=repo)
    _git('add', '.', cwd=repo)
    _git('commit', '--quiet', '-m', 'publish', cwd=repo)
    return repo


def test_fetch_clones_and_checks_out_wanted_dirs_at_the_pinned_revision(
    tmp_path, dataset_repo, monkeypatch
):
    revision = _git('rev-parse', 'HEAD', cwd=dataset_repo)
    path = dataset_repo / RUN_ID / '7b5033c1.json'
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    (dataset_repo / 'late.txt').write_text('later', encoding='utf-8')
    _git('add', '.', cwd=dataset_repo)
    _git('commit', '--quiet', '-m', 'later', cwd=dataset_repo)
    monkeypatch.setattr(
        run_logs.RunDataset,
        'url',
        property(lambda self: dataset_repo.as_uri()),
    )
    dataset = run_logs.RunDataset('v2_Public_Eval', 'test/v2', revision, 'v2')
    logs = run_logs.fetch_run_logs(
        {'v2_Public_Eval': {RUN_ID, 'not-published'}},
        tmp_path / 'work',
        datasets={'v2_Public_Eval': dataset},
    )
    source = logs.sources['v2_Public_Eval']
    assert source.revision == revision
    assert _git('rev-parse', 'HEAD', cwd=source.root) == revision
    assert (source.root / RUN_ID / 'results.json').is_file()
    assert not (source.root / 'unrelated-run').exists()
    assert not (source.root / 'late.txt').exists()
    assert source.history == {
        f'{RUN_ID}/{name}': 1
        for name in sorted(p.name for p in (dataset_repo / RUN_ID).iterdir())
    } | {'unrelated-run/results.json': 1}


def test_git_failure_carries_stderr(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git not available')
    with pytest.raises(RuntimeError, match='not a git repository|fatal'):
        run_logs.read_history(tmp_path, 'HEAD')


def test_local_git_checkout_supplies_revision_and_history(
    tmp_path, dataset_repo
):
    runs = tmp_path / 'local'
    runs.mkdir()
    dataset_repo.rename(runs / 'v2')
    path = runs / 'v2' / RUN_ID / '7b5033c1.json'
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    _git('commit', '--quiet', '-am', 'rewrite', cwd=runs / 'v2')
    logs = run_logs.local_run_logs(runs)
    source = logs.sources['v2_Public_Eval']
    assert source.revision == _git('rev-parse', 'HEAD', cwd=runs / 'v2')
    assert source.history[f'{RUN_ID}/7b5033c1.json'] == 2
    result = adapter.convert_logs(
        run_log_payload(), retrieved_timestamp=RETRIEVED, run_logs=logs
    )
    [(log, _, _)] = result.records
    details = result_by_id(
        log, 'v2_Public_Eval::score'
    ).generation_config.additional_details
    assert details['request_args_unavailable'] == 'run_rewritten'


def test_gate_rejection_is_reported_without_failing_the_run(
    tmp_path, runs_dir, capsys
):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(
        json.dumps(run_log_payload(score=0.6)), encoding='utf-8'
    )
    args = adapter.parse_args(
        [
            '--input-json',
            str(payload_path),
            '--output-dir',
            str(tmp_path / 'data' / 'arc-agi'),
            '--arc-runs-dir',
            str(runs_dir),
            '--arc-history-dir',
            str(runs_dir / 'history'),
        ]
    )
    assert adapter.run(args) == 1
    evidence = json.loads(
        (tmp_path / 'adapter_reports' / 'arc-agi_failures.json').read_text(
            encoding='utf-8'
        )
    )['run_log_evidence']
    assert evidence['gate_counts'] == {'score_mismatch': 1}
    assert evidence['unavailable'] == []
    assert 'ARC run-log regression: ' in capsys.readouterr().err
    assert evidence['regressions'] == [
        {
            'model_id': RUN_ID,
            'dataset_id': 'v2_Public_Eval',
            'directory': RUN_ID,
            'gate': 'score_mismatch',
            'detail': '',
        }
    ]
    assert evidence['filled'] == []


def test_plain_runs_dir_without_history_fails(tmp_path, runs_dir):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps(run_log_payload()), encoding='utf-8')
    args = adapter.parse_args(
        [
            '--input-json',
            str(payload_path),
            '--output-dir',
            str(tmp_path / 'data' / 'arc-agi'),
            '--arc-runs-dir',
            str(runs_dir),
        ]
    )
    with pytest.raises(SystemExit, match='--arc-history-dir'):
        adapter.run(args)
    assert not (tmp_path / 'data').exists()


def test_history_file_must_match_the_revision(tmp_path):
    path = tmp_path / 'v2.json'
    path.write_text(
        json.dumps({'revision': 'other', 'paths': {}}), encoding='utf-8'
    )
    with pytest.raises(run_logs.RunLogsUnavailable, match='revision'):
        run_logs.load_history_file(path, V2.revision)


def test_permanent_rejections_are_listed_as_unavailable(tmp_path, runs_dir):
    (run_dir(runs_dir) / 'results.json').unlink()
    log, logs = convert_with_logs(run_log_payload(), runs_dir)
    report = adapter.run_log_report(logs)
    assert list(report)[2:4] == ['regressions', 'unavailable']
    assert report['regressions'] == []
    assert [e['gate'] for e in report['unavailable']] == ['no_results_file']
