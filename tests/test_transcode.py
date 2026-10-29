from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator
from pydantic import ValidationError

from every_eval_ever import io as eee_io
from every_eval_ever import transcode as transcode_mod
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.schema import schema_json
from every_eval_ever.transcode import (
    TranscodeError,
    prepare_transcode,
    transcode_paths,
)
from every_eval_ever.validator.json_utils import strict_json_loads
from tests.test_validation_scope import UUID, valid_aggregate, valid_sample

UUID2 = '660e8400-e29b-41d4-a716-446655440001'
UUID3 = '770e8400-e29b-41d4-a716-446655440002'


def _stored_digest(path: Path, algorithm: str = 'sha256') -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def _write_pair(
    tmp_path: Path,
    *,
    file_uuid: str = UUID,
    line_ending: bytes = b'\n',
    hash_algorithm: str | None = 'sha256',
) -> tuple[Path, Path, bytes, dict]:
    folder = tmp_path / 'data' / 'bench' / 'dev' / 'model'
    folder.mkdir(parents=True, exist_ok=True)
    sample_path = folder / f'{file_uuid}_samples.jsonl'
    sample_bytes = json.dumps(valid_sample()).encode('utf-8') + line_ending
    sample_path.write_bytes(sample_bytes)

    aggregate = valid_aggregate()
    detail: dict[str, object] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/model/{file_uuid}_samples.jsonl',
        'total_rows': 1,
    }
    if hash_algorithm is not None:
        detail['hash_algorithm'] = hash_algorithm
        detail['checksum'] = hashlib.new(
            hash_algorithm, sample_bytes
        ).hexdigest()
    aggregate['detailed_evaluation_results'] = detail
    aggregate_path = folder / f'{file_uuid}.json'
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8'
    )
    return aggregate_path, sample_path, sample_bytes, aggregate


def _read_aggregate(path: Path) -> dict:
    data = strict_json_loads(eee_io.read_eee_text(path))
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize('compression', eee_io.COMPRESSION_CHOICES)
def test_schema_accepts_every_supported_companion_compression(compression):
    data = valid_aggregate()
    suffix = eee_io.compression_suffix(compression)
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/model/{UUID}_samples.jsonl{suffix}',
    }

    EvaluationLog.model_validate(data)
    Draft7Validator(schema_json('eval.schema.json')).validate(data)


def test_schema_rejects_unknown_companion_compression():
    data = valid_aggregate()
    data['detailed_evaluation_results'] = {
        'format': 'jsonl',
        'file_path': f'data/bench/dev/model/{UUID}_samples.jsonl.zip',
    }

    with pytest.raises(ValidationError):
        EvaluationLog.model_validate(data)


def test_transcode_pair_plain_to_gzip_updates_pointer_and_checksum(tmp_path):
    aggregate, samples, sample_bytes, _ = _write_pair(tmp_path)

    [result] = transcode_paths([str(aggregate)], 'gz')

    target_aggregate = aggregate.with_name(f'{aggregate.name}.gz')
    target_samples = samples.with_name(f'{samples.name}.gz')
    assert result.aggregate_path == target_aggregate
    assert result.samples_path == target_samples
    assert not aggregate.exists()
    assert not samples.exists()
    assert target_aggregate.is_file()
    assert target_samples.is_file()
    assert eee_io.read_eee_bytes(target_samples) == sample_bytes

    data = _read_aggregate(target_aggregate)
    detail = data['detailed_evaluation_results']
    assert detail['file_path'].endswith(f'{UUID}_samples.jsonl.gz')
    assert detail['hash_algorithm'] == 'sha256'
    assert detail['checksum'] == _stored_digest(target_samples)

    plan = prepare_transcode(target_aggregate, 'gz')
    assert plan.changed is False


def test_transcode_round_trip_back_to_plain(tmp_path):
    aggregate, samples, sample_bytes, original_aggregate = _write_pair(tmp_path)
    [compressed] = transcode_paths([str(aggregate)], 'gz')

    [restored] = transcode_paths([str(compressed.aggregate_path)], 'none')

    assert restored.aggregate_path == aggregate
    assert restored.samples_path == samples
    assert aggregate.is_file()
    assert samples.is_file()
    assert eee_io.read_eee_bytes(samples) == sample_bytes
    restored_data = _read_aggregate(aggregate)
    assert restored_data == original_aggregate
    assert restored_data['detailed_evaluation_results']['checksum'] == (
        _stored_digest(samples)
    )


def test_transcode_between_compressed_codecs(tmp_path):
    aggregate, _samples, sample_bytes, _ = _write_pair(tmp_path)
    [gzip_result] = transcode_paths([str(aggregate)], 'gz')

    [xz_result] = transcode_paths([str(gzip_result.aggregate_path)], 'xz')

    assert xz_result.aggregate_path.name.endswith('.json.xz')
    assert xz_result.samples_path is not None
    assert xz_result.samples_path.name.endswith('.jsonl.xz')
    assert eee_io.read_eee_bytes(xz_result.samples_path) == sample_bytes
    assert not gzip_result.aggregate_path.exists()
    assert gzip_result.samples_path is not None
    assert not gzip_result.samples_path.exists()


def test_aggregate_only_transcode_preserves_logical_bytes(tmp_path):
    folder = tmp_path / 'data' / 'bench' / 'dev' / 'model'
    folder.mkdir(parents=True)
    aggregate = folder / f'{UUID}.json'
    raw = json.dumps(valid_aggregate(), separators=(',', ':')).encode() + b'\r\n'
    aggregate.write_bytes(raw)

    [result] = transcode_paths([str(aggregate)], 'gz')

    assert eee_io.read_eee_bytes(result.aggregate_path) == raw


def test_transcode_preserves_sample_bytes_including_crlf(tmp_path):
    aggregate, _samples, sample_bytes, _ = _write_pair(
        tmp_path, line_ending=b'\r\n'
    )

    [result] = transcode_paths([str(aggregate)], 'xz')

    assert result.samples_path is not None
    assert eee_io.read_eee_bytes(result.samples_path) == sample_bytes


def test_transcode_preserves_legacy_md5_checksum_algorithm(tmp_path):
    aggregate, _samples, _sample_bytes, _ = _write_pair(
        tmp_path, hash_algorithm='md5'
    )

    [result] = transcode_paths([str(aggregate)], 'bz2')

    assert result.samples_path is not None
    data = _read_aggregate(result.aggregate_path)
    detail = data['detailed_evaluation_results']
    assert detail['hash_algorithm'] == 'md5'
    assert detail['checksum'] == _stored_digest(result.samples_path, 'md5')


def test_transcode_preserves_absent_checksum_metadata(tmp_path):
    aggregate, _samples, _sample_bytes, _ = _write_pair(
        tmp_path, hash_algorithm=None
    )

    [result] = transcode_paths([str(aggregate)], 'gz')

    data = _read_aggregate(result.aggregate_path)
    detail = data['detailed_evaluation_results']
    assert detail['file_path'].endswith('.jsonl.gz')
    assert 'hash_algorithm' not in detail
    assert 'checksum' not in detail


def test_dry_run_preflights_without_writing(tmp_path):
    aggregate, samples, _sample_bytes, _ = _write_pair(tmp_path)
    before_aggregate = aggregate.read_bytes()
    before_samples = samples.read_bytes()

    [result] = transcode_paths([str(aggregate)], 'gz', dry_run=True)

    assert result.changed is True
    assert result.aggregate_path.name.endswith('.json.gz')
    assert aggregate.read_bytes() == before_aggregate
    assert samples.read_bytes() == before_samples
    assert not result.aggregate_path.exists()
    assert result.samples_path is not None
    assert not result.samples_path.exists()


def test_dry_run_builds_and_validates_candidates(tmp_path, monkeypatch):
    aggregate, samples, _sample_bytes, _ = _write_pair(tmp_path)
    before_aggregate = aggregate.read_bytes()
    before_samples = samples.read_bytes()

    monkeypatch.setattr(
        transcode_mod,
        '_candidate_aggregate_bytes',
        lambda plan, samples_checksum: b'not valid json\n',
    )

    with pytest.raises(TranscodeError, match='refusing to transcode invalid'):
        transcode_paths([str(aggregate)], 'gz', dry_run=True)

    assert aggregate.read_bytes() == before_aggregate
    assert samples.read_bytes() == before_samples
    assert not aggregate.with_name(f'{aggregate.name}.gz').exists()
    assert not samples.with_name(f'{samples.name}.gz').exists()
    assert not list(aggregate.parent.glob('.eee-transcode-*'))


def test_sample_input_resolves_and_transcodes_whole_record(tmp_path):
    aggregate, samples, _sample_bytes, _ = _write_pair(tmp_path)

    [result] = transcode_paths([str(samples)], 'gz')

    assert result.source_aggregate_path == aggregate
    assert result.source_samples_path == samples
    assert result.aggregate_path.name.endswith('.json.gz')
    assert result.samples_path is not None
    assert result.samples_path.name.endswith('.jsonl.gz')


def test_duplicate_physical_variant_is_rejected_without_changes(tmp_path):
    aggregate, samples, _sample_bytes, _ = _write_pair(tmp_path)
    duplicate = aggregate.with_name(f'{aggregate.name}.gz')
    duplicate.write_bytes(eee_io.compress_bytes(aggregate.read_bytes(), 'gz'))
    before_samples = samples.read_bytes()

    with pytest.raises(TranscodeError, match='multiple physical variants'):
        transcode_paths([str(aggregate)], 'xz')

    assert aggregate.is_file()
    assert duplicate.is_file()
    assert samples.read_bytes() == before_samples


def test_batch_validation_preflight_happens_before_first_mutation(tmp_path):
    aggregate1, samples1, _bytes1, _ = _write_pair(
        tmp_path, file_uuid=UUID
    )
    aggregate2, samples2, _bytes2, _ = _write_pair(
        tmp_path, file_uuid=UUID2
    )
    broken = _read_aggregate(aggregate2)
    broken['detailed_evaluation_results']['checksum'] = 'not-the-real-checksum'
    aggregate2.write_text(json.dumps(broken), encoding='utf-8')

    with pytest.raises(TranscodeError, match='checksum mismatch'):
        transcode_paths([str(tmp_path / 'data')], 'gz')

    for path in (aggregate1, samples1, aggregate2, samples2):
        assert path.is_file()
        assert not path.with_name(f'{path.name}.gz').exists()


def test_corrupt_compressed_source_is_left_untouched(tmp_path):
    folder = tmp_path / 'data' / 'bench' / 'dev' / 'model'
    folder.mkdir(parents=True)
    source = folder / f'{UUID}.json.gz'
    source.write_bytes(b'not a gzip stream')
    before = source.read_bytes()

    with pytest.raises(TranscodeError, match='could not read aggregate'):
        transcode_paths([str(source)], 'xz')

    assert source.read_bytes() == before
    assert not folder.joinpath(f'{UUID}.json.xz').exists()


def test_commit_failure_rolls_back_original_pair(tmp_path, monkeypatch):
    aggregate, samples, _sample_bytes, _ = _write_pair(tmp_path)
    before_aggregate = aggregate.read_bytes()
    before_samples = samples.read_bytes()
    real_replace = transcode_mod.os.replace
    calls = 0

    def flaky_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError('injected aggregate install failure')
        return real_replace(source, destination)

    monkeypatch.setattr(transcode_mod.os, 'replace', flaky_replace)

    with pytest.raises(TranscodeError, match='commit failed'):
        transcode_paths([str(aggregate)], 'gz')

    assert aggregate.read_bytes() == before_aggregate
    assert samples.read_bytes() == before_samples
    assert not aggregate.with_name(f'{aggregate.name}.gz').exists()
    assert not samples.with_name(f'{samples.name}.gz').exists()
    assert not list(aggregate.parent.glob('.eee-transcode-*'))



def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }


def _validate_tree(root: Path, data_root: Path) -> list[Path]:
    paths = list(eee_io.iter_eee_results([data_root]))
    repository = {
        path.relative_to(root).as_posix(): path
        for path in paths
    }

    def read_repo_file(repo_path: str) -> str:
        return eee_io.read_eee_text(repository[repo_path])

    from every_eval_ever.validator.validation_core import validate_file

    for path in paths:
        repo_path = path.relative_to(root).as_posix()
        report = validate_file(
            path,
            repo_path=repo_path,
            available_files=repository,
            read_repo_file=read_repo_file,
            run_semantic_checks=True,
        )
        assert report.valid, (path, report.errors)
    return paths


def test_directory_transcode_synthetic_datastore_tree_end_to_end(tmp_path):
    """Exercise the bulk migration workflow without relying on external data."""
    aggregate1, samples1, sample1_bytes, _ = _write_pair(
        tmp_path, file_uuid=UUID
    )
    aggregate2, samples2, sample2_bytes, _ = _write_pair(
        tmp_path, file_uuid=UUID2
    )

    folder = aggregate1.parent
    aggregate3 = folder / f'{UUID3}.json'
    aggregate3_bytes = (
        json.dumps(valid_aggregate(), separators=(',', ':')).encode('utf-8')
        + b'\r\n'
    )
    aggregate3.write_bytes(aggregate3_bytes)

    # Seed a realistic mixed-storage tree: one plain pair, one gzip pair,
    # and one bzip2 aggregate-only result.
    [pair2_gz] = transcode_paths([str(aggregate2)], 'gz')
    [aggregate3_bz2] = transcode_paths([str(aggregate3)], 'bz2')
    assert pair2_gz.samples_path is not None
    assert aggregate3_bz2.samples_path is None

    data_root = tmp_path / 'data'
    before_dry_run = _snapshot_tree(data_root)

    dry_run = transcode_paths([str(data_root)], 'xz', dry_run=True)
    assert len(dry_run) == 3
    assert all(result.changed for result in dry_run)
    assert _snapshot_tree(data_root) == before_dry_run
    assert not list(data_root.rglob('.eee-transcode-*'))

    migration = transcode_paths([str(data_root)], 'xz')
    assert len(migration) == 3
    assert all(result.changed for result in migration)

    xz_paths = _validate_tree(tmp_path, data_root)
    assert len(xz_paths) == 5
    assert {eee_io.detect_compression(path) for path in xz_paths} == {'xz'}

    sample1_xz = folder / f'{UUID}_samples.jsonl.xz'
    sample2_xz = folder / f'{UUID2}_samples.jsonl.xz'
    aggregate3_xz = folder / f'{UUID3}.json.xz'
    assert eee_io.read_eee_bytes(sample1_xz) == sample1_bytes
    assert eee_io.read_eee_bytes(sample2_xz) == sample2_bytes
    assert eee_io.read_eee_bytes(aggregate3_xz) == aggregate3_bytes

    # The same bulk operation is the inverse: migrate the entire tree back to
    # the canonical uncompressed physical representation and revalidate it.
    restored = transcode_paths([str(data_root)], 'none')
    assert len(restored) == 3
    assert all(result.changed for result in restored)

    plain_paths = _validate_tree(tmp_path, data_root)
    assert len(plain_paths) == 5
    assert {eee_io.detect_compression(path) for path in plain_paths} == {'none'}
    assert samples1.read_bytes() == sample1_bytes
    assert samples2.read_bytes() == sample2_bytes
    assert aggregate3.read_bytes() == aggregate3_bytes
    assert not list(data_root.rglob('.eee-transcode-*'))

