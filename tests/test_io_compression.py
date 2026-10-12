from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from every_eval_ever import io as eee_io


@pytest.mark.parametrize(
    'name,kind,codec,stem',
    [
        ('abc.json', 'aggregate', 'none', 'abc'),
        ('abc.json.gz', 'aggregate', 'gz', 'abc'),
        ('abc.json.bz2', 'aggregate', 'bz2', 'abc'),
        ('abc.json.xz', 'aggregate', 'xz', 'abc'),
        ('abc.json.zst', 'aggregate', 'zst', 'abc'),
        ('abc.json.lz4', 'aggregate', 'lz4', 'abc'),
        ('abc_samples.jsonl.gz', 'samples', 'gz', 'abc'),
    ],
)
def test_result_shape(name, kind, codec, stem):
    assert eee_io.is_eee_result(name) == kind
    assert eee_io.detect_compression(name) == codec
    assert eee_io.eee_uuid_stem(name) == stem


@pytest.mark.parametrize('codec', ['none', 'gz', 'bz2', 'xz'])
def test_stdlib_codec_roundtrip(tmp_path: Path, codec: str):
    path = eee_io.add_compression_suffix(tmp_path / 'x.json', codec)
    payload = {'hello': 'world', 'codec': codec}
    with eee_io.open_eee_text(path, 'w') as file:
        json.dump(payload, file)
    assert json.loads(eee_io.read_eee_text(path)) == payload


def test_gzip_compression_is_deterministic():
    payload = b'deterministic checksum input\n'
    first = eee_io.compress_bytes(payload, 'gz')
    second = eee_io.compress_bytes(payload, 'gz')
    assert first == second
    assert first[4:8] == b'\x00\x00\x00\x00'
    assert first[9] == 255


def test_missing_compressed_file_stays_an_io_error(tmp_path: Path):
    path = tmp_path / 'missing.json.gz'
    with pytest.raises(FileNotFoundError):
        eee_io.read_eee_text(path)


def test_duplicate_variants(tmp_path: Path):
    plain = tmp_path / 'x.json'
    compressed = tmp_path / 'x.json.gz'
    plain.write_text('{}', encoding='utf-8')
    with gzip.open(compressed, 'wt', encoding='utf-8') as file:
        file.write('{}')
    groups = eee_io.find_duplicate_variants([plain, compressed])
    assert len(groups) == 1
    assert groups[0][1:3] == ('x', 'aggregate')


@pytest.mark.parametrize('codec', ['gz', 'bz2', 'xz'])
def test_corrupt_stdlib_stream_is_normalized(tmp_path: Path, codec: str):
    path = tmp_path / f'x.json.{codec}'
    path.write_bytes(b'not-a-valid-stream')
    with pytest.raises(eee_io.CompressedReadError):
        eee_io.read_eee_text(path)


def test_corrupt_lz4_stream_is_normalized(tmp_path: Path):
    try:
        eee_io.compress_bytes(b'codec probe', 'lz4')
    except eee_io.CodecUnavailableError:
        pytest.skip('lz4 codec extra is not installed')

    path = tmp_path / 'x.json.lz4'
    path.write_bytes(b'not-a-valid-stream')
    with pytest.raises(eee_io.CompressedReadError):
        eee_io.read_eee_text(path)


def test_zstd_reads_concatenated_frames(tmp_path: Path):
    try:
        first = eee_io.compress_bytes(b'{"row": 1}\n', 'zst')
        second = eee_io.compress_bytes(b'{"row": 2}\n', 'zst')
    except eee_io.CodecUnavailableError:
        pytest.skip('zstd codec extra is not installed')
    path = tmp_path / 'x_samples.jsonl.zst'
    path.write_bytes(first + second)
    assert [json.loads(line) for line in eee_io.iter_eee_text_lines(path)] == [
        {'row': 1},
        {'row': 2},
    ]
