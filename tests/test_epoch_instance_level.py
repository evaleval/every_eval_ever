"""Offline tests for Epoch AI instance-level companions (Inspect logs)."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    'inspect_ai', reason='inspect extra not installed (uv sync --extra inspect)'
)

from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402

from every_eval_ever.adapters.epoch_ai import (  # noqa: E402
    adapter,
    instance_level,
)
from every_eval_ever.helpers import SourceRecordsError  # noqa: E402
from every_eval_ever.validator import validate_file  # noqa: E402

FIXTURES = Path(__file__).parent / 'fixtures' / 'epoch_ai'
INSPECT_LOG = Path(__file__).parent / 'data' / 'inspect' / 'data_arc_qwen.json'
LOG_URL = 'https://logs.example/inspect_ai_logs/qwenrun.eval'
PRIVATE_URL = 'https://logs.example/inspect_ai_logs/private.eval'
SCORE = '0.3333333333333333'  # the fixture log's choice/accuracy


def _eval_bytes(tmp_path: Path) -> bytes:
    log = read_eval_log(INSPECT_LOG)
    log.eval.model = 'ollama/qwen2.5-0.5b'
    log.samples[0].input = 'Which one? Pick a letter.'
    path = tmp_path / 'qwenrun.eval'
    write_eval_log(log, path)
    return path.read_bytes()


def _zip_with_rows(rows: list[dict[str, str]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path in sorted(FIXTURES.glob('*.csv')):
            data = path.read_text()
            if path.name == 'gpqa_diamond.csv':
                header = next(csv.reader(io.StringIO(data)))
                out = io.StringIO()
                writer = csv.DictWriter(out, header, lineterminator='\n')
                writer.writerows({**dict.fromkeys(header, ''), **row}
                                 for row in rows)
                data = data.rstrip('\n') + '\n' + out.getvalue()
            archive.writestr(path.name, data)
    return buffer.getvalue()


def _row(run_id: str, score: str, logs: str) -> dict[str, str]:
    return {'Model version': 'qwen2.5-0.5b', 'Organization': 'Alibaba',
            'mean_score': score, 'Best score (across scorers)': score,
            'Logs': logs, 'id': run_id}


class _Response:
    def __init__(self, status: int, body: bytes = b''):
        self.status_code, self.body = status, body
        self.headers = {'Content-Length': str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]


def _validate(root: Path, path: Path):
    return validate_file(
        path,
        repo_path=path.relative_to(root).as_posix(),
        available_files=frozenset(
            p.relative_to(root).as_posix()
            for p in path.parent.iterdir() if p.is_file()),
        read_repo_file=lambda f: (root / f).read_text(encoding='utf-8'),
        run_semantic_checks=True,
    )


def test_instance_level_attaches_only_logs_that_back_the_score(
    tmp_path, monkeypatch,
):
    body = _eval_bytes(tmp_path)
    monkeypatch.setattr(
        instance_level.requests, 'get',
        lambda url, **kw: _Response(200, body) if url == LOG_URL
        else _Response(403))
    archive = tmp_path / 'benchmark_data.zip'
    archive.write_bytes(_zip_with_rows([
        _row('qwenrun', SCORE, LOG_URL),
        _row('otherrun', '0.5', LOG_URL),       # log does not give 0.5
        _row('privaterun', '0.25', PRIVATE_URL),  # 403
    ]))
    out = tmp_path / 'data' / adapter.COLLECTION
    args = adapter.parse_args(['--output-dir', str(out), '--instance-level'])
    args.input = archive
    with pytest.raises(SourceRecordsError):  # the fixture has failing rows
        adapter.run(args)

    aggregates = {
        json.loads(p.read_text())['evaluation_id'].split('/')[3]: p
        for p in out.glob('*/*/*.json')}
    attached = aggregates['qwenrun']
    record = json.loads(attached.read_text())
    detailed = record['detailed_evaluation_results']
    samples = attached.with_name(f'{attached.stem}_samples.jsonl')
    assert detailed['total_rows'] == 3
    rows = [json.loads(line) for line in samples.read_text().split('\n')
            if line]
    assert {r['evaluation_id'] for r in rows} == {record['evaluation_id']}
    assert {r['evaluation_result_id'] for r in rows} == {'gpqa-diamond'}
    # escaped on disk, identical once decoded
    assert ' ' not in samples.read_text(encoding='utf-8')
    assert any(' ' in r['input']['raw'] for r in rows)
    for path in (attached, samples):
        report = _validate(tmp_path, path)
        assert report.valid, report.errors
        assert not report.warnings, report.warnings

    for key in ('otherrun', 'privaterun'):  # skipped logs cost no record
        assert json.loads(aggregates[key].read_text()).get(
            'detailed_evaluation_results') is None
    report = json.loads(
        (tmp_path / 'adapter_reports' / f'{adapter.COLLECTION}_failures.json')
        .read_text())
    reasons = [e['reason'] for e in report['excluded_records']
               if e['source_ref'].endswith('#inspect-log')]
    assert sorted(reasons) == [
        'no instance-level companion: log not downloadable (HTTP 403)',
        'no instance-level companion: no scorer in the log reports the '
        'published score 0.5',
    ]


def test_without_the_flag_no_log_is_fetched(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('fetched a log without --instance-level')

    monkeypatch.setattr(instance_level.requests, 'get', refuse)
    archive = tmp_path / 'benchmark_data.zip'
    archive.write_bytes(_zip_with_rows([_row('qwenrun', SCORE, LOG_URL)]))
    args = adapter.parse_args(
        ['--output-dir', str(tmp_path / 'data' / adapter.COLLECTION)])
    args.input = archive
    with pytest.raises(SourceRecordsError):
        adapter.run(args)
    assert not list(tmp_path.glob('data/*/*/*/*_samples.jsonl'))


def test_oversized_log_is_skipped_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(instance_level.requests, 'get',
                        lambda url, **kw: _Response(200, b'x' * 2_000_001))
    with pytest.raises(instance_level.LogUnavailable, match='over the'):
        instance_level.download_log(
            LOG_URL, tmp_path / 'x.eval', 1_000_000, 'test')
    assert not (tmp_path / 'x.eval').exists()


def _header(*metrics: dict[str, float]):
    return SimpleNamespace(results=SimpleNamespace(scores=[
        SimpleNamespace(name=f's{i}', metrics={
            k: SimpleNamespace(value=v) for k, v in m.items()})
        for i, m in enumerate(metrics)]))


@pytest.mark.parametrize(('score', 'metrics', 'expected'), [
    (0.7575757575757576, {'accuracy': 0.7575757575757576}, ['s0']),
    (0.787, {'accuracy': 0.7867494824016563}, ['s0']),  # rounded, 3 places
    (0.8, {'accuracy': 0.8372093023255814}, []),        # 1 place vouches nothing
    (0.933, {'accuracy': 0.936397147691564}, []),
    (0.015, {'stderr': 0.015}, []),                      # stderr is not a score
    (41.8, {'accuracy': 0.418}, ['s0']),                 # percent column
])
def test_matching_scorers(score, metrics, expected):
    assert instance_level.matching_scorers(
        _header(metrics), score, percent=score > 1) == expected
