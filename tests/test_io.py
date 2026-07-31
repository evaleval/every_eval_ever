"""Tests for ``every_eval_ever.io`` — transparent compression helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever import io as eee_io


# ---------------------------------------------------------------------------
# Suffix recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'name, expected',
    [
        ('abc.json', 'aggregate'),
        ('abc.json.gz', 'aggregate'),
        ('abc.json.zst', 'aggregate'),
        ('abc.json.bz2', 'aggregate'),
        ('abc.json.xz', 'aggregate'),
        ('abc.json.lz4', 'aggregate'),
        ('abc_samples.jsonl', 'samples'),
        ('abc_samples.jsonl.gz', 'samples'),
        ('abc_samples.jsonl.zst', 'samples'),
        ('abc_samples.jsonl.bz2', 'samples'),
        ('abc_samples.jsonl.xz', 'samples'),
        ('abc_samples.jsonl.lz4', 'samples'),
        # Lenient: bare *.jsonl (no _samples prefix) still recognized
        # as samples, since lm-eval and other converters produce these.
        ('samples_mmlu_2025-01-01.jsonl', 'samples'),
        ('whatever.jsonl', 'samples'),
        ('whatever.jsonl.gz', 'samples'),
        ('abc.txt', None),
        ('abc.parquet', None),
        ('manifest.json', 'aggregate'),  # any *.json is aggregate by shape
        ('abc.zip', None),  # zip is not a recognized stream codec
        ('abc.json.zip', None),
    ],
)
def test_is_eee_result(name: str, expected: str | None) -> None:
    assert eee_io.is_eee_result(name) == expected


@pytest.mark.parametrize(
    'name, expected_stem',
    [
        ('abc.json', 'abc'),
        ('abc.json.gz', 'abc'),
        ('a-b-c.json.zst', 'a-b-c'),
        ('abc_samples.jsonl', 'abc'),
        ('abc_samples.jsonl.gz', 'abc'),
        # Bare .jsonl: stem is the whole pre-extension filename.
        ('samples_mmlu_2025.jsonl', 'samples_mmlu_2025'),
        ('whatever.jsonl.bz2', 'whatever'),
        ('abc.txt', None),
        ('abc.json.zip', None),
    ],
)
def test_eee_uuid_stem(name: str, expected_stem: str | None) -> None:
    assert eee_io.eee_uuid_stem(name) == expected_stem


@pytest.mark.parametrize(
    'name, expected',
    [
        ('abc.json', 'none'),
        ('abc.json.gz', 'gz'),
        ('abc.json.zst', 'zst'),
        ('abc.json.bz2', 'bz2'),
        ('abc.json.xz', 'xz'),
        ('abc.json.lz4', 'lz4'),
        ('abc.txt', 'none'),
    ],
)
def test_detect_compression(name: str, expected: str) -> None:
    assert eee_io.detect_compression(name) == expected


def test_add_compression_suffix_none() -> None:
    p = Path('/x/y/abc.json')
    assert eee_io.add_compression_suffix(p, 'none') == p


@pytest.mark.parametrize('cs', ['gz', 'zst', 'bz2', 'xz', 'lz4'])
def test_add_compression_suffix(cs: str) -> None:
    p = Path('/x/y/abc.json')
    out = eee_io.add_compression_suffix(p, cs)
    assert out.name == f'abc.json.{cs}'


def test_add_compression_suffix_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        eee_io.add_compression_suffix(Path('a.json'), 'snappy')


# ---------------------------------------------------------------------------
# Round-trip per codec
# ---------------------------------------------------------------------------


def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


_CODEC_REQUIRES = {
    'none': None,
    'gz': None,  # stdlib
    'bz2': None,  # stdlib
    'xz': None,  # stdlib (lzma)
    'zst': 'zstandard',
    'lz4': 'lz4',
}


def _codec_param(cs: str) -> pytest.param:
    req = _CODEC_REQUIRES[cs]
    marks: list = []
    if req is not None and not _can_import(req):
        marks.append(pytest.mark.skip(reason=f'requires {req!r}'))
    return pytest.param(cs, marks=marks)


@pytest.mark.parametrize(
    'cs', [_codec_param(c) for c in eee_io.COMPRESSION_CHOICES]
)
def test_open_eee_text_roundtrip_aggregate(cs: str, tmp_path: Path) -> None:
    base = tmp_path / 'abc.json'
    out_path = eee_io.add_compression_suffix(base, cs)
    payload = {'schema_version': '0.2.2', 'kind': 'aggregate', 'cs': cs}

    with eee_io.open_eee_text(out_path, 'w') as f:
        json.dump(payload, f)

    with eee_io.open_eee_text(out_path, 'r') as f:
        assert json.load(f) == payload


@pytest.mark.parametrize(
    'cs', [_codec_param(c) for c in eee_io.COMPRESSION_CHOICES]
)
def test_open_eee_text_roundtrip_samples(cs: str, tmp_path: Path) -> None:
    base = tmp_path / 'abc_samples.jsonl'
    out_path = eee_io.add_compression_suffix(base, cs)
    rows = [{'i': i, 'cs': cs} for i in range(5)]

    with eee_io.open_eee_text(out_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    with eee_io.open_eee_text(out_path, 'r') as f:
        read_back = [json.loads(line) for line in f if line.strip()]
    assert read_back == rows


def test_open_eee_text_rejects_bad_mode(tmp_path: Path) -> None:
    p = tmp_path / 'abc.json'
    p.write_text('{}')
    with pytest.raises(ValueError):
        eee_io.open_eee_text(p, 'rb')


# ---------------------------------------------------------------------------
# Discovery + duplicate-variant detection
# ---------------------------------------------------------------------------


def test_iter_eee_results_finds_compressed_and_plain(tmp_path: Path) -> None:
    plain = tmp_path / 'sub' / 'a.json'
    plain.parent.mkdir()
    plain.write_text('{}')
    gz = tmp_path / 'sub' / 'b_samples.jsonl.gz'
    import gzip
    with gzip.open(gz, 'wt', encoding='utf-8') as f:
        f.write('{}\n')
    junk = tmp_path / 'sub' / 'readme.txt'
    junk.write_text('not eee')

    found = sorted(eee_io.iter_eee_results([tmp_path]))
    assert found == [plain, gz]


def test_iter_eee_results_accepts_file_root(tmp_path: Path) -> None:
    p = tmp_path / 'a.json'
    p.write_text('{}')
    assert list(eee_io.iter_eee_results([p])) == [p]


def test_find_duplicate_variants_detects_collision(tmp_path: Path) -> None:
    folder = tmp_path / 'data' / 'bench' / 'dev' / 'model'
    folder.mkdir(parents=True)
    plain = folder / 'abc.json'
    plain.write_text('{}')
    gz = folder / 'abc.json.gz'
    import gzip
    with gzip.open(gz, 'wt', encoding='utf-8') as f:
        f.write('{}')

    # Distinct kind in the same folder must NOT trigger the rule.
    samples = folder / 'abc_samples.jsonl'
    samples.write_text('{}\n')

    dups = eee_io.find_duplicate_variants(eee_io.iter_eee_results([tmp_path]))
    assert len(dups) == 1
    folder_out, stem, kind, variants = dups[0]
    assert folder_out == folder
    assert stem == 'abc'
    assert kind == 'aggregate'
    assert sorted(variants) == sorted([plain, gz])


def test_find_duplicate_variants_clean_tree(tmp_path: Path) -> None:
    folder = tmp_path / 'data'
    folder.mkdir()
    (folder / 'abc.json').write_text('{}')
    (folder / 'abc_samples.jsonl').write_text('{}\n')
    (folder / 'def.json').write_text('{}')
    assert eee_io.find_duplicate_variants(
        eee_io.iter_eee_results([tmp_path])
    ) == []
