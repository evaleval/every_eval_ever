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


def test_html_parser_uses_live_table_shape_and_keeps_bad_rows():
    html = """
    <table><tbody>
      <tr><td><input></td><td>1</td><td>Example Agent</td>
          <td>GPT-5</td><td>2026-01-01</td><td>Example Org</td>
          <td>OpenAI</td><td>50.0%± 2.0</td></tr>
      <tr><td><input></td><td>2</td><td>Bad Agent</td>
          <td>GPT-5</td><td>2026-01-01</td><td>Example Org</td>
          <td>OpenAI</td><td>unknown</td></tr>
    </tbody></table>
    """

    result = adapter.parse_leaderboard_html(html)

    assert result.records == [_entry()]
    assert len(result.failures) == 1
    assert result.failures[0].source_record['cells'][-1] == 'unknown'
