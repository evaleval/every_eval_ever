"""Writer-side compression tests for the converter CLIs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from every_eval_ever import cli
from every_eval_ever import io as eee_io


# ---------------------------------------------------------------------------
# _resolve_compression: per-kind override falls back to --compress
# ---------------------------------------------------------------------------


def test_resolve_compression_uses_per_kind_override() -> None:
    args = SimpleNamespace(compress='gz', compress_aggregate='none',
                           compress_samples='zst')
    assert cli._resolve_compression(args, 'aggregate') == 'none'
    assert cli._resolve_compression(args, 'samples') == 'zst'


def test_resolve_compression_falls_back_to_compress() -> None:
    args = SimpleNamespace(compress='gz', compress_aggregate=None,
                           compress_samples=None)
    assert cli._resolve_compression(args, 'aggregate') == 'gz'
    assert cli._resolve_compression(args, 'samples') == 'gz'


def test_resolve_compression_default_none() -> None:
    args = SimpleNamespace(compress='none', compress_aggregate=None,
                           compress_samples=None)
    assert cli._resolve_compression(args, 'samples') == 'none'


def test_resolve_compression_rejects_bad_kind() -> None:
    args = SimpleNamespace(compress='gz', compress_aggregate=None,
                           compress_samples=None)
    with pytest.raises(ValueError):
        cli._resolve_compression(args, 'metadata')


# ---------------------------------------------------------------------------
# _write_log writes the right path, with the right codec, that round-trips
# ---------------------------------------------------------------------------


class _FakeSourceData:
    def __init__(self, dataset_name: str = 'fake'):
        self.dataset_name = dataset_name


class _FakeEvalResult:
    def __init__(self, dataset_name: str = 'fake'):
        self.source_data = _FakeSourceData(dataset_name)


class _FakeModelInfo:
    def __init__(self, id_: str = 'org/model'):
        self.id = id_


class _FakeLog:
    """Minimal log-shaped object that supports the bits ``_write_log`` uses."""

    def __init__(self, dataset: str = 'fake', model_id: str = 'org/model'):
        self.evaluation_results = [_FakeEvalResult(dataset)]
        self.model_info = _FakeModelInfo(model_id)
        self._dump = {'schema_version': '0.2.2', 'kind': 'aggregate',
                      'dataset': dataset, 'model': model_id}

    def model_dump(self, *, mode: str, exclude_none: bool):
        return self._dump


def test_write_log_uncompressed_default(tmp_path: Path) -> None:
    log = _FakeLog()
    out = cli._write_log(log, tmp_path, eval_uuid='aaa')
    assert out.name == 'aaa.json'
    assert out.parent == tmp_path / 'fake' / 'org' / 'model'
    assert out.exists()
    assert json.loads(out.read_text()) == log._dump


def test_write_log_gz_appends_suffix_and_compresses(tmp_path: Path) -> None:
    log = _FakeLog()
    out = cli._write_log(log, tmp_path, eval_uuid='bbb', compression='gz')
    assert out.name == 'bbb.json.gz'
    # Cannot read as plain text — must decompress.
    with gzip.open(out, 'rt', encoding='utf-8') as f:
        assert json.load(f) == log._dump


def test_write_log_preserves_indent_and_unicode(tmp_path: Path) -> None:
    log = _FakeLog()
    log._dump['note'] = 'café'  # non-ascii
    out = cli._write_log(log, tmp_path, eval_uuid='ccc', compression='bz2')
    assert out.name == 'ccc.json.bz2'
    with eee_io.open_eee_text(out, 'r') as f:
        data = json.load(f)
    assert data['note'] == 'café'


# ---------------------------------------------------------------------------
# LMEvalInstanceLevelAdapter.transform_and_save honors compression
# ---------------------------------------------------------------------------


def test_lm_eval_samples_writer_gzip(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from every_eval_ever.converters.lm_eval.instance_level_adapter import (
        LMEvalInstanceLevelAdapter,
    )

    # Write a fake lm-eval samples.jsonl input. We don't need it to be a
    # real lm-eval shape; we just need transform_samples to produce >= 1
    # record. Stub _transform_sample to short-circuit the schema work.
    fake_input = tmp_path / 'samples_fake_2025-01-01.jsonl'
    fake_input.write_text(json.dumps({'_': 1}) + '\n')

    # Build a fake instance-level log: any Pydantic-validating shape works.
    from every_eval_ever.instance_level_types import (
        Evaluation, Input, InstanceLevelEvaluationLog, InteractionType,
        Output,
    )

    def _fake_transform(self, sample, evaluation_id, model_id, task_name):
        from every_eval_ever.instance_level_types import (
            AnswerAttributionItem,
        )
        return InstanceLevelEvaluationLog(
            schema_version='0.2.2',
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_name=task_name,
            sample_id='1',
            sample_hash='a' * 64,
            interaction_type=InteractionType.single_turn,
            input=Input(raw='q', reference=[], choices=[]),
            output=Output(raw=['a'], reasoning_trace=[]),
            answer_attribution=[
                AnswerAttributionItem(
                    turn_idx=0, source='output.raw', extracted_value='a',
                    extraction_method='exact_match', is_terminal=True,
                )
            ],
            evaluation=Evaluation(score=1.0, is_correct=True),
        )

    monkeypatch.setattr(
        LMEvalInstanceLevelAdapter, '_transform_sample', _fake_transform
    )

    adapter = LMEvalInstanceLevelAdapter()
    out_dir = tmp_path / 'out'
    detailed = adapter.transform_and_save(
        samples_path=fake_input,
        evaluation_id='eid',
        model_id='org/model',
        task_name='fake',
        output_dir=out_dir,
        file_uuid='zzz',
        compression='gz',
    )

    assert detailed is not None
    out_file = Path(detailed.file_path)
    assert out_file.name == 'zzz_samples.jsonl.gz'
    # Round-trips
    with gzip.open(out_file, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    # checksum is over the on-disk (compressed) bytes
    import hashlib
    assert detailed.checksum == hashlib.sha256(out_file.read_bytes()).hexdigest()


def test_lm_eval_samples_writer_default_uncompressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from every_eval_ever.converters.lm_eval.instance_level_adapter import (
        LMEvalInstanceLevelAdapter,
    )
    from every_eval_ever.instance_level_types import (
        Evaluation, Input, InstanceLevelEvaluationLog, InteractionType,
        Output,
    )

    fake_input = tmp_path / 'samples_fake.jsonl'
    fake_input.write_text(json.dumps({'_': 1}) + '\n')

    def _fake_transform(self, sample, evaluation_id, model_id, task_name):
        from every_eval_ever.instance_level_types import (
            AnswerAttributionItem,
        )
        return InstanceLevelEvaluationLog(
            schema_version='0.2.2',
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_name=task_name,
            sample_id='1',
            sample_hash='a' * 64,
            interaction_type=InteractionType.single_turn,
            input=Input(raw='q', reference=[], choices=[]),
            output=Output(raw=['a'], reasoning_trace=[]),
            answer_attribution=[
                AnswerAttributionItem(
                    turn_idx=0, source='output.raw', extracted_value='a',
                    extraction_method='exact_match', is_terminal=True,
                )
            ],
            evaluation=Evaluation(score=1.0, is_correct=True),
        )
    monkeypatch.setattr(
        LMEvalInstanceLevelAdapter, '_transform_sample', _fake_transform
    )

    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
        samples_path=fake_input,
        evaluation_id='eid',
        model_id='org/model',
        task_name='fake',
        output_dir=tmp_path / 'out',
        file_uuid='zzz',
    )
    assert detailed is not None
    assert Path(detailed.file_path).name == 'zzz_samples.jsonl'


# ---------------------------------------------------------------------------
# CLI parser wires the flags through with sensible defaults
# ---------------------------------------------------------------------------


def test_cli_parser_defaults_to_none() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args(
        ['convert', 'lm_eval', '--log_path', '/tmp/x', '--output_dir', '/tmp/y']
    )
    assert ns.compress == 'none'
    assert ns.compress_aggregate is None
    assert ns.compress_samples is None
    assert cli._resolve_compression(ns, 'aggregate') == 'none'
    assert cli._resolve_compression(ns, 'samples') == 'none'


def test_cli_parser_compress_samples_gz() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args(
        ['convert', 'helm', '--log_path', '/tmp/x',
         '--compress-samples', 'gz']
    )
    assert ns.compress == 'none'
    assert ns.compress_samples == 'gz'
    assert cli._resolve_compression(ns, 'aggregate') == 'none'
    assert cli._resolve_compression(ns, 'samples') == 'gz'


def test_cli_parser_global_compress() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args(
        ['convert', 'inspect', '--log_path', '/tmp/x', '--compress', 'bz2']
    )
    assert ns.compress == 'bz2'
    assert cli._resolve_compression(ns, 'aggregate') == 'bz2'
    assert cli._resolve_compression(ns, 'samples') == 'bz2'


def test_cli_parser_rejects_invalid_codec() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ['convert', 'lm_eval', '--log_path', '/tmp/x',
             '--compress', 'snappy']
        )
