"""Unit tests for the AlpacaEval adapter."""

from unittest.mock import MagicMock, patch

import pytest

from every_eval_ever.converters.alpaca_eval.adapter import (
    LEADERBOARDS,
    AlpacaEvalAdapter,
    _fetch_csv,
    _model_name_from_row,
)

# ---------------------------------------------------------------------------
# Fixture CSV rows
# ---------------------------------------------------------------------------

_V1_ROW = {
    '': 'gpt4',
    'win_rate': '95.28',
    'standard_error': '0.68',
    'n_wins': '476',
    'n_wins_base': '24',
    'n_draws': '0',
    'n_loss': '0',
    'discrete_win_rate': '95.28',
    'avg_length': '1247',
}

_V2_ROW = {
    '': 'gpt4_turbo',
    'win_rate': '50.0',
    'standard_error': '0.0',
    'length_controlled_winrate': '55.12',
    'lc_standard_error': '0.72',
    'discrete_win_rate': '50.0',
    'avg_length': '1024',
}


def _make_csv_response(rows: list[dict]) -> MagicMock:
    import csv
    import io

    if not rows:
        text = ''
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        text = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.text = text
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# _fetch_csv
# ---------------------------------------------------------------------------


def test_fetch_csv_returns_rows():
    mock_resp = _make_csv_response([_V1_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        rows = _fetch_csv('http://example.com/fake.csv')
    assert len(rows) == 1
    assert rows[0]['win_rate'] == '95.28'


# ---------------------------------------------------------------------------
# _model_name_from_row
# ---------------------------------------------------------------------------


def test_model_name_from_unnamed_column():
    assert _model_name_from_row({'': 'my_model', 'win_rate': '50'}) == 'my_model'


def test_model_name_fallback_to_first_value():
    assert _model_name_from_row({'x': 'fallback'}) == 'fallback'


# ---------------------------------------------------------------------------
# AlpacaEvalAdapter.fetch_leaderboard — v1
# ---------------------------------------------------------------------------


def test_fetch_leaderboard_v1_produces_log():
    mock_resp = _make_csv_response([_V1_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        adapter = AlpacaEvalAdapter()
        logs = adapter.fetch_leaderboard('v1')

    assert len(logs) == 1
    log = logs[0]
    assert log.model_info.name == 'gpt4'
    assert log.model_info.developer == 'openai'
    assert log.eval_library.name == 'alpaca_eval'
    assert log.eval_library.version == '1.0'


def test_fetch_leaderboard_v1_win_rate_value():
    mock_resp = _make_csv_response([_V1_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v1')

    results = {r.evaluation_name: r for r in logs[0].evaluation_results}
    assert 'Win Rate' in results
    assert abs(results['Win Rate'].score_details.score - 95.28 / 100) < 1e-5


def test_fetch_leaderboard_v1_source_data_url_points_to_csv():
    mock_resp = _make_csv_response([_V1_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v1')

    source_url = logs[0].evaluation_results[0].source_data.url[0]
    assert source_url == LEADERBOARDS['v1']['url']


def test_fetch_leaderboard_v1_no_lc_win_rate():
    mock_resp = _make_csv_response([_V1_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v1')

    names = [r.evaluation_name for r in logs[0].evaluation_results]
    assert 'Length-Controlled Win Rate' not in names


# ---------------------------------------------------------------------------
# AlpacaEvalAdapter.fetch_leaderboard — v2
# ---------------------------------------------------------------------------


def test_fetch_leaderboard_v2_has_lc_win_rate():
    mock_resp = _make_csv_response([_V2_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v2')

    names = [r.evaluation_name for r in logs[0].evaluation_results]
    assert 'Length-Controlled Win Rate' in names


def test_fetch_leaderboard_v2_lc_win_rate_value():
    mock_resp = _make_csv_response([_V2_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v2')

    results = {r.evaluation_name: r for r in logs[0].evaluation_results}
    assert abs(results['Length-Controlled Win Rate'].score_details.score - 55.12 / 100) < 1e-5


def test_fetch_leaderboard_v2_source_data_url_points_to_csv():
    mock_resp = _make_csv_response([_V2_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v2')

    source_url = logs[0].evaluation_results[0].source_data.url[0]
    assert source_url == LEADERBOARDS['v2']['url']


# ---------------------------------------------------------------------------
# metric_id values
# ---------------------------------------------------------------------------


def test_metric_ids_are_set():
    mock_resp = _make_csv_response([_V2_ROW])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v2')

    by_name = {r.evaluation_name: r for r in logs[0].evaluation_results}
    assert by_name['Win Rate'].metric_config.metric_id == 'alpaca_eval.win_rate'
    assert by_name['Length-Controlled Win Rate'].metric_config.metric_id == 'alpaca_eval.lc_win_rate'
    assert by_name['Discrete Win Rate'].metric_config.metric_id == 'alpaca_eval.discrete_win_rate'
    assert by_name['Average Response Length'].metric_config.metric_id == 'alpaca_eval.avg_length'


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_null_model_is_skipped():
    null_row = dict(_V1_ROW)
    null_row[''] = 'NullModel'
    mock_resp = _make_csv_response([null_row])
    with patch('every_eval_ever.converters.alpaca_eval.adapter.requests.get', return_value=mock_resp):
        logs = AlpacaEvalAdapter().fetch_leaderboard('v1')
    assert logs == []


def test_unknown_version_raises():
    with pytest.raises(ValueError, match='Unknown version'):
        AlpacaEvalAdapter().fetch_leaderboard('v99')
