from __future__ import annotations

import gzip
from pathlib import Path

from every_eval_ever.validator.validation_core import validate_file

UUID = 'f3a1c0de-4b2e-4c1a-9f6d-1b7e5a2c8d40'


def _fixture() -> Path:
    return next(
        Path('tests/data/skill_reference_conversion/data').rglob(f'{UUID}.json')
    )


def test_compressed_aggregate_matches_plain_schema_result(tmp_path: Path):
    source = _fixture()
    plain = tmp_path / 'copy.json'
    plain.write_bytes(source.read_bytes())
    compressed = tmp_path / 'copy.json.gz'
    with gzip.open(compressed, 'wb') as file:
        file.write(source.read_bytes())
    plain_report = validate_file(plain)
    compressed_report = validate_file(compressed)
    assert compressed_report.valid == plain_report.valid
    assert compressed_report.errors == plain_report.errors


def test_corrupt_compressed_stream_is_a_report(tmp_path: Path):
    path = tmp_path / 'bad.json.gz'
    path.write_bytes(b'not gzip')
    report = validate_file(path)
    assert not report.valid
    assert report.errors[0]['type'] == 'compressed_read_error'


def test_explicit_file_detects_compressed_sibling(tmp_path: Path):
    folder = tmp_path / 'data' / 'demo-source' / 'demo-org' / 'demo-model'
    folder.mkdir(parents=True)
    plain = folder / f'{UUID}.json'
    plain.write_bytes(_fixture().read_bytes())
    compressed = folder / f'{UUID}.json.gz'
    with gzip.open(compressed, 'wb') as file:
        file.write(_fixture().read_bytes())
    repo_path = f'data/demo-source/demo-org/demo-model/{UUID}.json'
    available = frozenset(
        {
            repo_path,
            repo_path + '.gz',
        }
    )
    report = validate_file(
        plain,
        repo_path=repo_path,
        available_files=available,
        run_semantic_checks=True,
    )
    assert any(error['type'] == 'duplicate_variant' for error in report.errors)
