"""Validator tests covering compressed result files + duplicate-variant rule."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from every_eval_ever import io as eee_io
from every_eval_ever import validate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _read_fixture_aggregate() -> dict:
    """Load the existing aggregate fixture committed under tests/data/."""
    fixture = Path(__file__).parent / 'data' / '98ea850e-7019-4728-a558-8b1819ec47c2.json'
    with fixture.open(encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plain + .gz validate identically
# ---------------------------------------------------------------------------


def test_validate_aggregate_gz_matches_plain(tmp_path: Path) -> None:
    payload = _read_fixture_aggregate()

    plain = tmp_path / 'a.json'
    with plain.open('w', encoding='utf-8') as f:
        json.dump(payload, f)
    rep_plain = validate.validate_file(plain)

    gz = tmp_path / 'b.json.gz'
    with gzip.open(gz, 'wt', encoding='utf-8') as f:
        json.dump(payload, f)
    rep_gz = validate.validate_file(gz)

    # Gzipped vs. plain must produce identical validation outcomes — that's
    # the whole point of transparent compression. We don't assert
    # ``valid == True`` here because the fixture may legitimately fail
    # against the current schema; we only assert that compression doesn't
    # change the result.
    assert rep_plain.valid == rep_gz.valid
    assert rep_plain.file_type == rep_gz.file_type == 'aggregate'
    assert [e['type'] for e in rep_plain.errors] == [
        e['type'] for e in rep_gz.errors
    ]
    assert [e['msg'] for e in rep_plain.errors] == [
        e['msg'] for e in rep_gz.errors
    ]


def test_validate_samples_gz_matches_plain(tmp_path: Path) -> None:
    # Synthesize a minimal valid InstanceLevelEvaluationLog stream.
    rows = [
        {
            'schema_version': '0.2.2',
            'evaluation_id': 'fake/eval/1',
            'model_id': 'org/model',
            'evaluation_name': 'fake',
            'sample_id': str(i),
            'sample_hash': 'a' * 64,
            'interaction_type': 'single_turn',
            'input': {'raw': 'q', 'reference': [], 'choices': []},
            'output': {'raw': ['a'], 'reasoning_trace': []},
            'evaluation': {'score': 1.0, 'is_correct': True},
        }
        for i in range(3)
    ]

    plain = tmp_path / 'a_samples.jsonl'
    with plain.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    rep_plain = validate.validate_file(plain)

    gz = tmp_path / 'b_samples.jsonl.gz'
    with gzip.open(gz, 'wt', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    rep_gz = validate.validate_file(gz)

    assert rep_plain.valid == rep_gz.valid
    assert rep_plain.line_count == rep_gz.line_count == 3
    assert rep_plain.file_type == rep_gz.file_type == 'instance'


# ---------------------------------------------------------------------------
# Discovery picks up compressed files
# ---------------------------------------------------------------------------


def test_expand_paths_finds_compressed_files(tmp_path: Path) -> None:
    a_plain = tmp_path / 'sub' / 'aaa.json'
    a_plain.parent.mkdir()
    a_plain.write_text('{}')
    b_gz = tmp_path / 'sub' / 'bbb_samples.jsonl.gz'
    with gzip.open(b_gz, 'wt', encoding='utf-8') as f:
        f.write('{}\n')
    junk = tmp_path / 'sub' / 'README.md'
    junk.write_text('not eee')

    found = sorted(validate.expand_paths([str(tmp_path)]))
    assert found == [a_plain, b_gz]


# ---------------------------------------------------------------------------
# Duplicate-variant rule
# ---------------------------------------------------------------------------


def test_validate_main_fails_on_duplicate_variants(tmp_path: Path) -> None:
    folder = tmp_path / 'data' / 'bench' / 'dev' / 'model'
    folder.mkdir(parents=True)
    plain = folder / 'abc.json'
    plain.write_text('{}')  # invalid as EvaluationLog, but that's a different error
    gz = folder / 'abc.json.gz'
    with gzip.open(gz, 'wt', encoding='utf-8') as f:
        f.write('{}')

    exit_code = validate.main([str(tmp_path), '--format', 'json'])
    assert exit_code == 1


def test_duplicate_variant_reports_emit_typed_errors(tmp_path: Path) -> None:
    folder = tmp_path / 'data'
    folder.mkdir()
    (folder / 'abc.json').write_text('{}')
    with gzip.open(folder / 'abc.json.gz', 'wt', encoding='utf-8') as f:
        f.write('{}')

    paths = list(eee_io.iter_eee_results([tmp_path]))
    reports = validate._duplicate_variant_reports(paths)
    assert len(reports) == 1
    err = reports[0].errors[0]
    assert err['type'] == 'duplicate_variant'
    assert 'abc.json' in err['msg']
    assert 'abc.json.gz' in err['msg']


def test_duplicate_variant_distinct_kinds_dont_collide(tmp_path: Path) -> None:
    """Same UUID, different kind: NOT a duplicate."""
    folder = tmp_path / 'data'
    folder.mkdir()
    (folder / 'abc.json').write_text('{}')
    (folder / 'abc_samples.jsonl').write_text('{}\n')

    paths = list(eee_io.iter_eee_results([tmp_path]))
    assert validate._duplicate_variant_reports(paths) == []


# ---------------------------------------------------------------------------
# Unsupported / non-EEE filename
# ---------------------------------------------------------------------------


def test_validate_file_rejects_zip(tmp_path: Path) -> None:
    p = tmp_path / 'abc.json.zip'
    p.write_bytes(b'PK\x03\x04')
    rep = validate.validate_file(p)
    assert rep.valid is False
    assert rep.errors[0]['type'] == 'unsupported_extension'


def test_validate_file_rejects_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / 'abc.parquet'
    p.write_bytes(b'PAR1')
    rep = validate.validate_file(p)
    assert rep.valid is False
    assert rep.errors[0]['type'] == 'unsupported_extension'


# ---------------------------------------------------------------------------
# Missing optional codec surfaces as a validation error, not a crash
# ---------------------------------------------------------------------------


def test_missing_codec_surfaces_as_validation_error(tmp_path: Path,
                                                    monkeypatch) -> None:
    pytest.importorskip(  # no point if zstandard happens to be installed AND we mocked
        'pytest', reason='sanity'
    )
    # Force lz4 import to fail regardless of environment.
    import sys
    monkeypatch.setitem(sys.modules, 'lz4', None)
    monkeypatch.setitem(sys.modules, 'lz4.frame', None)

    p = tmp_path / 'abc.json.lz4'
    # Bytes don't matter — we should never reach the actual decompressor.
    p.write_bytes(b'\x00\x00\x00\x00')

    rep = validate.validate_file(p)
    assert rep.valid is False
    assert rep.errors[0]['type'] == 'codec_unavailable'
