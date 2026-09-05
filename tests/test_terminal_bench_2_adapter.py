from pathlib import Path

from every_eval_ever.adapters.terminal_bench_2 import adapter
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
        'ci95_half_width': 2.0,
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


def _row(**overrides):
    row = {
        'rank': 1,
        'status': 'display',
        'metadata': {
            'agent_display': {'label': 'Example Agent', 'url': 'https://e.dev'},
            'agent_name': 'example',
            'agent_org': 'Example Org',
            'model_display': 'GPT-5',
            'model_names': ['gpt-5'],
            'model_org': 'OpenAI',
            'date': '2026-01-01',
        },
        'metrics': {
            'accuracy': 50.0,
            'accuracy_ci95_half_width': 2.0,
            'display_accuracy': '**50.0%** \u00b1 2.0',
        },
    }
    row.update(overrides)
    return row


def test_payload_rows_normalize_to_entries_and_keep_bad_rows():
    payload = {
        'rows': [
            _row(),
            _row(rank='second', metadata={}, metrics={}),
        ]
    }

    result = adapter.parse_leaderboard_payload(payload)

    assert result.records == [_entry()]
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == 'leaderboard row 2'


def test_a_row_the_source_withholds_is_excluded_with_its_reason():
    payload = {'rows': [_row(), _row(rank=2, status='hidden')]}

    result = adapter.parse_leaderboard_payload(payload)

    assert result.records == [_entry()]
    assert result.failures == []
    assert len(result.exclusions) == 1
    assert "'hidden'" in result.exclusions[0].reason


def test_a_leaderboard_that_displays_nothing_fails_instead_of_publishing_none():
    result = adapter.parse_leaderboard_payload(
        {'rows': [_row(status='hidden')]}
    )

    assert result.records == []
    assert len(result.exclusions) == 1
    assert 'withheld all 1 of its rows' in result.failures[0].reason


def test_an_agent_without_a_display_label_falls_back_to_its_name():
    metadata = dict(_row()['metadata'])
    del metadata['agent_display']
    metadata['model_display'] = ''

    entry = adapter.parse_leaderboard_payload(
        {'rows': [_row(metadata=metadata)]}
    ).records[0]

    assert entry['agent'] == 'example'
    assert entry['model'] == 'gpt-5'


def test_the_reported_half_width_is_recorded_as_a_confidence_interval():
    """The source publishes a 95% CI half-width, not a standard error.

    It was written into ``standard_error``, which overstates the standard error
    by roughly the 1.96 factor between the two quantities.
    """
    bundles = adapter.make_logs([_entry()], retrieved_timestamp='1234567890.0')

    uncertainty = bundles[0][0].evaluation_results[0].score_details.uncertainty
    assert uncertainty.standard_error is None
    interval = uncertainty.confidence_interval
    assert (interval.lower, interval.upper) == (48.0, 52.0)
    assert interval.confidence_level == 0.95


def test_a_row_without_a_half_width_carries_no_uncertainty():
    bundles = adapter.make_logs(
        [_entry(ci95_half_width=None)],
        retrieved_timestamp='1234567890.0',
    )

    assert bundles[0][0].evaluation_results[0].score_details.uncertainty is None
