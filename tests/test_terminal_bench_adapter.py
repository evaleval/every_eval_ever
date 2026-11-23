import json
from pathlib import Path

from every_eval_ever.adapters.terminal_bench import adapter
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _entry(**overrides):
    entry = {
        'rank': 1,
        'agent': 'Example Agent',
        'model': 'GPT-5',
        'date': '2026-01-01',
        'agent_org': 'Example Org',
        'model_org': 'OpenAI',
        'accuracy': 50.0,
        'stderr': 2.0,
    }
    entry.update(overrides)
    return entry


def test_normalized_entries_convert_and_validate(tmp_path: Path):
    bundles = adapter.make_logs([_entry()], retrieved_timestamp='1234567890.0')
    output_dir = tmp_path / 'data' / 'terminal-bench-2.0'
    paths = adapter.export(bundles, output_dir)

    assert len(paths) == 1
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


def test_the_metric_carries_a_join_key_that_is_not_plain_accuracy():
    """A trial-averaged resolution rate must not join to registry `accuracy`.

    The metric had no `metric_id` at all, so nothing tied these scores together
    across refreshes. The registry carries no Terminal-Bench metric, so the id is
    namespaced: it claims a stable join key within this source and no global
    identity, the same shape `mmlu_pro` uses.
    """
    bundles = adapter.make_logs([_entry()], retrieved_timestamp='1234567890.0')

    metric = bundles[0][0].evaluation_results[0].metric_config
    assert metric.metric_id == 'terminal-bench-2.0.accuracy'
    # The percent scale and its bounds are the leaderboard's own and unchanged;
    # only the missing id is being filled in here.
    assert metric.metric_unit == 'percent'
    assert (metric.min_score, metric.max_score) == (0, 100)


def test_custom_leaderboard_url_is_recorded_as_source():
    leaderboard_url = 'https://example.com/terminal-bench-2'

    bundles = adapter.make_logs(
        [_entry()],
        retrieved_timestamp='1234567890.0',
        leaderboard_url=leaderboard_url,
    )

    eval_log = bundles[0][0]
    assert eval_log.evaluation_results[0].source_data.url == [leaderboard_url]


def test_rejected_entry_retains_source_provenance():
    bad_entry = _entry(model='')

    try:
        adapter.make_logs([bad_entry], retrieved_timestamp='1234567890.0')
    except SourceRecordsError as exc:
        assert exc.failures[0].source_ref == 'leaderboard row 0'
        assert exc.failures[0].source_record == bad_entry
        assert 'model' in exc.failures[0].reason
    else:
        raise AssertionError('expected invalid Terminal-Bench entry to fail')


FIXTURES = Path(__file__).parent / 'data' / 'terminal_bench'


def _parse(name: str):
    payload = json.loads((FIXTURES / name).read_text(encoding='utf-8'))
    return adapter.parse_leaderboard_payload(payload)


def test_payload_rows_become_entries_and_hidden_rows_are_excluded():
    result = _parse('leaderboard.json')

    assert result.total_records == 3
    assert not result.failures
    assert len(result.exclusions) == 1
    assert "'hidden'" in result.exclusions[0].reason
    first, second = result.records
    assert first == {
        'id': '5d127407-a570-4864-8e8a-022123748a19',
        'rank': 1,
        'agent': 'NexAU-AHE',
        'model': 'GPT-5.5',
        'date': '2026-04-23',
        'agent_org': 'china-qijizhifeng',
        'model_org': 'OpenAI',
        'reasoning_effort': None,
        'accuracy': 84.7191011236,
        'stderr': None,
        'ci95_half_width': 2.0892351283,
        # 2.0 publishes n_trials 0 on every row: unset, not zero
        'n_trials': None,
        'pass_at': {},
        'details': {},
    }
    # a row published "± N/A" carries no half-width
    assert second['ci95_half_width'] is None


def test_published_half_width_is_a_95_percent_interval(tmp_path: Path):
    entries = _parse('leaderboard.json').records

    bundles = adapter.make_logs(entries, retrieved_timestamp='1234567890.0')

    with_ci = bundles[0][0].evaluation_results[0].score_details
    interval = with_ci.uncertainty.confidence_interval
    assert with_ci.uncertainty.standard_error is None
    assert interval.confidence_level == 0.95
    assert interval.lower == round(84.7191011236 - 2.0892351283, 10)
    assert interval.upper == round(84.7191011236 + 2.0892351283, 10)
    # 2.0 documents 87 tasks x 5 trials
    assert with_ci.uncertainty.num_samples == 435
    assert bundles[1][0].evaluation_results[0].score_details.uncertainty is None
    for path in adapter.export(
        bundles, tmp_path / 'data' / 'terminal-bench-2.0'
    ):
        report = validate_file(path)
        assert report.valid, report.errors


def test_newer_versions_keep_effort_trials_and_distinct_ids(tmp_path: Path):
    spec = adapter.VERSIONS_BY_KEY['4.0']
    entries = _parse('leaderboard_4.0.json').records

    bundles = adapter.make_logs(entries, '1234567890.0', spec=spec)

    logs = [log for log, _, _ in bundles]
    # same agent and model at two efforts: two records, two ids
    assert [log.evaluation_id for log in logs] == [
        'terminal-bench-4.0/5c537be4-7fc3-449b-8bfc-ceb9061c2535',
        'terminal-bench-4.0/16db8ad5-84aa-4588-b660-1ce68c0d45e2',
    ]
    assert [
        log.model_info.additional_details['reasoning_effort'] for log in logs
    ] == ['max', 'xhigh']
    result = logs[0].evaluation_results[0]
    assert result.metric_config.metric_id == 'terminal-bench-4.0.accuracy'
    assert result.score_details.score == 58.18
    assert result.score_details.uncertainty.num_samples == 330
    assert result.score_details.details['successes'] == '192'
    # 4.0 publishes no task/trial split, so none is claimed
    command = result.generation_config.generation_args.execution_command
    assert ' -k ' not in command
    assert 'terminal-bench/terminal-bench@4.0.0' in command
    pass_results = logs[0].evaluation_results[1:]
    assert [r.metric_config.metric_name for r in pass_results] == [
        'Pass@2',
        'Pass@3',
        'Pass@4',
        'Pass@5',
    ]
    pass_at_2 = pass_results[0]
    assert pass_at_2.metric_config.metric_id == 'pass_at_k'
    assert pass_at_2.metric_config.metric_parameters == {'k': 2}
    assert pass_at_2.metric_config.max_score == 1
    assert pass_at_2.score_details.score == 0.6485
    assert pass_at_2.evaluation_result_id == (
        'terminal-bench-4.0/5c537be4-7fc3-449b-8bfc-ceb9061c2535#pass_at_2'
    )
    for path in adapter.export(bundles, tmp_path / 'data' / spec.collection):
        report = validate_file(path)
        assert report.valid, report.errors


def test_catalog_declares_every_version_collection():
    from every_eval_ever.adapters import catalog

    spec = catalog.get('terminal_bench')
    assert spec.output_scope == 'data_root'
    assert set(spec.collections) == {
        version.collection for version in adapter.VERSIONS
    }
