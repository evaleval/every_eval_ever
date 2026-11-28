"""Offline tests for the Aider Polyglot adapter."""

from pathlib import Path

import pytest

from every_eval_ever.adapters.aider_polyglot import adapter
from every_eval_ever.helpers.io import SourceRecordsError

YAML = """\
- dirname: 2025-02-25-gemini
  test_cases: 225
  model: Gemini 2.0 Pro exp-02-05
  edit_format: whole
  pass_rate_1: 20.4
  pass_rate_2: 35.6
  date: 2025-02-25
  total_cost: 0.0
- dirname: 2025-01-17-sonnet
  test_cases: 225
  model: claude-3-5-sonnet-20241022
  edit_format: diff
  pass_rate_2: 51.6
"""


def _convert(text, tmp_path):
    return adapter.convert_rows(
        adapter.parse_rows(text), tmp_path, '1', resolve=False
    )


def test_scores_protocol_and_no_invention(tmp_path: Path):
    result = _convert(YAML, tmp_path)
    assert not result.failures
    first, second = [o.eval_log for o in result.records]
    assert [r.score_details.score for r in first.evaluation_results] == [
        20.4,
        35.6,
    ]
    assert first.evaluation_results[0].metric_config.max_score == 100.0
    assert first.evaluation_results[0].generation_config.additional_details == {
        'edit_format': 'whole'
    }
    assert [r.metric_config.metric_id for r in second.evaluation_results] == [
        'pass_rate_2'
    ]
    assert first.model_info.developer == 'unattributed'
    assert first.evaluation_id != second.evaluation_id


def test_bad_rows_are_reported_not_dropped(tmp_path: Path):
    bad = YAML + (
        '- dirname: x\n  model: m\n  edit_format: diff\n  pass_rate_2: 140\n'
        '- dirname: y\n  model: m\n  edit_format: diff\n'
        '- dirname: 2025-01-17-sonnet\n  model: m\n  edit_format: diff\n'
        '  pass_rate_2: 1\n'
    )
    result = _convert(bad, tmp_path)
    assert len(result.records) == 2
    assert len(result.failures) == 3
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_shape_drift_fails_closed():
    with pytest.raises(ValueError, match='non-empty'):
        adapter.parse_rows('model: x')
