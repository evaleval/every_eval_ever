"""Offline tests for the CritPt README adapter."""

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.critpt import adapter
from every_eval_ever.helpers.io import SourceRecordsError

V1 = """\
## Leaderboard
| Model | Challenge Accuracy¹ |
| ----- | ------------------- |
| GPT-5 (high, code & web) | 12.6 |
| GPT-5 (minimal) | 0.0 |
| Gemini-3 Pro² | 9.1 |

¹ We use average accuracy over 5 runs × 70 test challenges as our primary performance metric.
² Result of Gemini 3 Pro (without tools) here is provided by **Artificial Analysis** ([source](https://artificialanalysis.ai/evaluations/critpt)).
"""
V2 = """\
## Leaderboard (v2)
| Model | Challenge Accuracy¹ |
| ----- | ------------------- |
| gpt-5 | 28.9 |
| gpt-6-astra | 83.1 |

¹ We use average accuracy over 5 runs × 70 test challenges as our primary performance metric.
"""


def test_v1_variants_attribution_and_stable_identity(tmp_path: Path, monkeypatch):
    rows = adapter.parse_rows(V1, 'v1')
    result = adapter.convert_rows(rows, 'v1', tmp_path, '1', resolve=False)
    assert result.total_records == len(result.records) == 3
    assert not result.failures
    first, second, independent = [output.eval_log for output in result.records]
    assert first.model_info.id == second.model_info.id == 'GPT-5'
    assert first.evaluation_id != second.evaluation_id
    assert first.evaluation_results[0].generation_config.additional_details == {
        'source_variant': 'high, code & web'
    }
    assert independent.model_info.name == 'Gemini-3 Pro'
    assert independent.source_metadata.additional_details['evaluator'] == 'Artificial Analysis'
    assert independent.source_metadata.additional_details['tools'] == 'without tools'
    assert independent.evaluation_results[0].score_details.score == 9.1
    monkeypatch.setattr(adapter, 'model_resolution', lambda *_: ('openai/gpt-5', {}))
    changed = adapter.convert_rows(rows, 'v1', tmp_path, '2')
    assert [o.eval_log.evaluation_id for o in changed.records] == [
        o.eval_log.evaluation_id for o in result.records
    ]


def test_v2_no_invented_metadata(tmp_path: Path):
    result = adapter.convert_rows(adapter.parse_rows(V2, 'v2'), 'v2', tmp_path, '1', resolve=False)
    log = result.records[0].eval_log
    score = log.evaluation_results[0]
    assert score.score_details.score == 28.9
    assert score.metric_config.min_score is None
    assert score.metric_config.max_score is None
    assert score.source_data.samples_number is None
    assert score.generation_config is None
    assert log.source_metadata.evaluator_relationship.value == 'other'
    assert log.model_info.additional_details['deployment_type'] == 'unknown'


def test_drift_fails_closed():
    with pytest.raises(ValueError, match='header'):
        adapter.parse_rows(V2.replace('Challenge Accuracy¹', 'Score'), 'v2')
    with pytest.raises(ValueError, match='definition'):
        adapter.parse_rows(V2.replace('5 runs', '6 runs'), 'v2')
    with pytest.raises(ValueError, match='footnote'):
        adapter.parse_rows(V1.replace('Artificial Analysis', 'Someone else'), 'v1')


def test_partial_failure_report_and_nonzero(tmp_path: Path):
    text = V2.replace('| gpt-6-astra | 83.1 |', '| bad-model | NaN |')
    source = tmp_path / 'source.md'
    source.write_text(text)
    output = tmp_path / 'data' / 'critpt-v2'
    report = tmp_path / 'report.json'
    args = adapter.parse_args([
        '--input-markdown', str(source), '--output-dir', str(output),
        '--failure-report', str(report), '--no-registry-resolve',
    ])
    with pytest.raises(SourceRecordsError):
        adapter.run(args)
    assert len(list(output.rglob('*.json'))) == 1
    data = json.loads(report.read_text())
    assert data['total_source_records'] == 2
    assert data['failed_record_count'] == 1
    assert data['failed_records'][0]['source_record']['line'] == '| bad-model | NaN |'


def test_duplicate_and_unsafe_rows_are_accounted(tmp_path: Path):
    rows = adapter.parse_rows(V2, 'v2')
    rows.extend([rows[0], {'line': '| ../bad | 10 |', 'version': 'v2'}])
    result = adapter.convert_rows(rows, 'v2', tmp_path, '1', resolve=False)
    assert len(result.records) == 2
    assert len(result.failures) == 2


def test_fetch_captures_exact_bytes(monkeypatch):
    class Response:
        content = V2.encode()

        def raise_for_status(self):
            pass

    captured = []
    monkeypatch.setattr(adapter.requests, 'get', lambda *args, **kwargs: Response())
    monkeypatch.setattr(adapter.raw_capture, 'record', lambda **kw: captured.append(kw))
    assert adapter.fetch_readme('v2') == V2
    assert captured[0]['content'] == V2.encode()
    assert captured[0]['url'] == adapter.URLS['v2']


def test_resolver_never_creates_entries(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'canonical_id': 'openai/gpt-5', 'created_new': True}

    calls = []
    monkeypatch.setattr(adapter.requests, 'post', lambda *a, **kw: calls.append(kw) or Response())
    model, provenance = adapter.model_resolution('GPT-5', True)
    assert calls[0]['json']['mode'] == 'exact'
    assert model == 'GPT-5'
    assert provenance['model_registry_strategy'] == 'no_canonical'


def test_score_revision_does_not_change_identity():
    original = adapter.convert_rows(adapter.parse_rows(V2, 'v2'), 'v2', Path('data'), '1', resolve=False)
    revised = adapter.convert_rows(adapter.parse_rows(V2.replace('28.9', '30.1'), 'v2'), 'v2', Path('data'), '2', resolve=False)
    assert original.records[0].eval_log.evaluation_id == revised.records[0].eval_log.evaluation_id
    assert revised.records[0].eval_log.evaluation_results[0].score_details.score == 30.1
