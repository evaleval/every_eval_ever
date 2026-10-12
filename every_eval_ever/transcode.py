"""Failure-safe transcoding of existing EEE result artifacts."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Container, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from every_eval_ever import io as eee_io
from every_eval_ever.validator.json_utils import (
    StrictJSONError,
    strict_json_loads,
)
from every_eval_ever.validator.validation_core import (
    ValidationReport,
    validate_file,
)


class TranscodeError(RuntimeError):
    """Raised when an existing result cannot be transcoded safely."""


class TranscodeRollbackError(TranscodeError):
    """Raised when a failed commit could not be rolled back completely."""


@dataclass(frozen=True)
class TranscodePlan:
    """A validated physical-representation change for one logical result."""

    aggregate_path: Path
    aggregate_repo_path: str
    target_aggregate_path: Path
    target_aggregate_repo_path: str
    samples_path: Path | None
    samples_repo_path: str | None
    target_samples_path: Path | None
    target_samples_repo_path: str | None
    compression: eee_io.Compression
    samples_hash_algorithm: str | None

    @property
    def changed(self) -> bool:
        return (
            self.aggregate_path != self.target_aggregate_path
            or self.samples_path != self.target_samples_path
        )


@dataclass(frozen=True)
class TranscodeResult:
    """The physical paths produced for one logical result."""

    source_aggregate_path: Path
    aggregate_path: Path
    source_samples_path: Path | None
    samples_path: Path | None
    changed: bool


class _RepositoryFiles(Container[str]):
    """Map repository paths used by semantic checks to local physical files."""

    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files

    def __contains__(self, repo_path: object) -> bool:
        return isinstance(repo_path, str) and repo_path in self.files

    def read_text(self, repo_path: str) -> str:
        try:
            path = self.files[repo_path]
        except KeyError as exc:
            raise OSError(f'local file is not available: {repo_path}') from exc
        return eee_io.read_eee_text(path)


def _stored_digest(path: Path, algorithm: str) -> str:
    if algorithm not in {'sha256', 'md5'}:
        raise TranscodeError(
            f'unsupported stored-byte hash algorithm: {algorithm!r}'
        )
    digest = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_logical(path: Path) -> str:
    digest = hashlib.sha256()
    for chunk in eee_io.iter_eee_binary_chunks(path):
        digest.update(chunk)
    return digest.hexdigest()


def _validation_message(report: ValidationReport) -> str:
    details = '; '.join(
        f'{error.get("loc", "(file)")}: {error.get("msg", "validation error")}'
        for error in report.errors[:5]
    )
    if len(report.errors) > 5:
        details += f'; ... {len(report.errors) - 5} more'
    return details or 'validation failed'


def _require_valid(
    path: Path,
    *,
    repo_path: str,
    repository: _RepositoryFiles,
) -> None:
    report = validate_file(
        path,
        repo_path=repo_path,
        available_files=repository,
        read_repo_file=repository.read_text,
        run_semantic_checks=True,
    )
    if not report.valid:
        raise TranscodeError(
            f'refusing to transcode invalid EEE result {path}: '
            f'{_validation_message(report)}'
        )


def _repo_path_under_data(path: Path) -> str | None:
    absolute = path.absolute()
    data_dir = next(
        (ancestor for ancestor in absolute.parents if ancestor.name == 'data'),
        None,
    )
    if data_dir is None:
        return None
    return absolute.relative_to(data_dir.parent).as_posix()


def _logical_samples_path(aggregate_path: Path) -> Path:
    logical = eee_io.strip_compression_suffix(aggregate_path)
    if not logical.name.lower().endswith('.json'):
        raise TranscodeError(f'not an aggregate EEE result: {aggregate_path}')
    stem = logical.name[: -len('.json')]
    return logical.with_name(f'{stem}_samples.jsonl')


def _logical_aggregate_path(samples_path: Path) -> Path:
    logical = eee_io.strip_compression_suffix(samples_path)
    suffix = '_samples.jsonl'
    if not logical.name.lower().endswith(suffix):
        raise TranscodeError(f'not an EEE samples result: {samples_path}')
    stem = logical.name[: -len(suffix)]
    return logical.with_name(f'{stem}.json')


def _single_variant(logical_path: Path, *, kind: str) -> Path | None:
    variants = eee_io.existing_local_variants(logical_path)
    if len(variants) > 1:
        rendered = ', '.join(str(path) for path in variants)
        raise TranscodeError(
            f'multiple physical variants exist for one logical {kind}: '
            f'{rendered}'
        )
    return variants[0] if variants else None


def _load_aggregate(path: Path) -> dict[str, Any]:
    try:
        data = strict_json_loads(eee_io.read_eee_text(path))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        StrictJSONError,
    ) as exc:
        raise TranscodeError(f'could not read aggregate {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise TranscodeError(f'aggregate {path} must contain a JSON object')
    return data


def _declared_samples_repo_path(data: dict[str, Any]) -> str | None:
    detail = data.get('detailed_evaluation_results')
    if not isinstance(detail, dict):
        return None
    path = detail.get('file_path')
    if not isinstance(path, str) or not path.strip():
        return None
    return path.strip()


def _derive_repository_paths(
    aggregate_path: Path,
    samples_path: Path | None,
    aggregate_data: dict[str, Any],
) -> tuple[str, str | None]:
    aggregate_repo = _repo_path_under_data(aggregate_path)
    samples_repo = (
        _repo_path_under_data(samples_path) if samples_path is not None else None
    )
    if aggregate_repo is not None:
        if samples_path is not None and samples_repo is None:
            logical = PurePosixPath(
                eee_io.strip_compression_suffix_text(aggregate_repo)
            )
            samples_repo = (
                logical.parent
                / f'{logical.stem}_samples.jsonl'
            ).as_posix()
            samples_repo = eee_io.add_compression_suffix_text(
                samples_repo, eee_io.detect_compression(samples_path)
            )
        return aggregate_repo, samples_repo

    declared_samples = _declared_samples_repo_path(aggregate_data)
    if declared_samples is not None:
        logical_samples = eee_io.strip_compression_suffix_text(declared_samples)
        logical_path = PurePosixPath(logical_samples)
        suffix = '_samples.jsonl'
        if logical_path.name.lower().endswith(suffix):
            aggregate_name = f'{logical_path.name[: -len(suffix)]}.json'
            logical_aggregate = (
                logical_path.parent / aggregate_name
            ).as_posix()
            aggregate_repo = eee_io.add_compression_suffix_text(
                logical_aggregate,
                eee_io.detect_compression(aggregate_path),
            )
            if samples_path is not None:
                samples_repo = eee_io.add_compression_suffix_text(
                    logical_samples,
                    eee_io.detect_compression(samples_path),
                )
            return aggregate_repo, samples_repo

    synthetic_parent = PurePosixPath('data/local/local/local')
    aggregate_repo = (synthetic_parent / aggregate_path.name).as_posix()
    samples_repo = (
        (synthetic_parent / samples_path.name).as_posix()
        if samples_path is not None
        else None
    )
    return aggregate_repo, samples_repo


def _verify_samples_checksum(
    aggregate_path: Path,
    aggregate_data: dict[str, Any],
    samples_path: Path | None,
) -> str | None:
    if samples_path is None:
        return None
    detail = aggregate_data.get('detailed_evaluation_results')
    if not isinstance(detail, dict):
        raise TranscodeError(
            f'{aggregate_path} has samples but no detailed_evaluation_results'
        )
    algorithm = detail.get('hash_algorithm')
    expected = detail.get('checksum')
    if algorithm is None and expected is None:
        return None
    if not isinstance(algorithm, str) or not isinstance(expected, str):
        raise TranscodeError(
            f'{aggregate_path} must declare hash_algorithm and checksum together'
        )
    actual = _stored_digest(samples_path, algorithm)
    if expected != actual:
        raise TranscodeError(
            f'{aggregate_path} samples checksum mismatch: declared '
            f'{expected!r}, stored bytes hash to {actual!r}'
        )
    return algorithm


def _target_repo_path(repo_path: str, compression: eee_io.Compression) -> str:
    logical = eee_io.strip_compression_suffix_text(repo_path)
    return eee_io.add_compression_suffix_text(logical, compression)


def prepare_transcode(
    aggregate_path: str | Path,
    compression: str,
) -> TranscodePlan:
    """Validate one logical result and return its transcode plan."""
    compression = eee_io.normalize_compression(compression)
    aggregate_path = Path(aggregate_path)
    if not aggregate_path.is_file():
        raise TranscodeError(f'aggregate file does not exist: {aggregate_path}')
    if eee_io.is_eee_result(aggregate_path) != 'aggregate':
        raise TranscodeError(f'not an aggregate EEE result: {aggregate_path}')

    aggregate_logical = eee_io.strip_compression_suffix(aggregate_path)
    existing_aggregate = _single_variant(
        aggregate_logical, kind='aggregate result'
    )
    if existing_aggregate != aggregate_path:
        raise TranscodeError(
            f'{aggregate_path} is not the unique physical aggregate variant'
        )

    samples_logical = _logical_samples_path(aggregate_path)
    samples_path = _single_variant(samples_logical, kind='samples result')
    aggregate_data = _load_aggregate(aggregate_path)
    aggregate_repo_path, samples_repo_path = _derive_repository_paths(
        aggregate_path, samples_path, aggregate_data
    )

    files = {aggregate_repo_path: aggregate_path}
    if samples_path is not None and samples_repo_path is not None:
        files[samples_repo_path] = samples_path
    repository = _RepositoryFiles(files)
    _require_valid(
        aggregate_path,
        repo_path=aggregate_repo_path,
        repository=repository,
    )
    if samples_path is not None:
        if samples_repo_path is None:
            raise TranscodeError(
                f'could not determine repository path for {samples_path}'
            )
        _require_valid(
            samples_path,
            repo_path=samples_repo_path,
            repository=repository,
        )
    samples_hash_algorithm = _verify_samples_checksum(
        aggregate_path, aggregate_data, samples_path
    )

    target_aggregate_path = eee_io.add_compression_suffix(
        aggregate_logical, compression
    )
    target_aggregate_repo_path = _target_repo_path(
        aggregate_repo_path, compression
    )
    if samples_path is None:
        target_samples_path = None
        target_samples_repo_path = None
    else:
        target_samples_path = eee_io.add_compression_suffix(
            samples_logical, compression
        )
        assert samples_repo_path is not None
        target_samples_repo_path = _target_repo_path(
            samples_repo_path, compression
        )

    return TranscodePlan(
        aggregate_path=aggregate_path,
        aggregate_repo_path=aggregate_repo_path,
        target_aggregate_path=target_aggregate_path,
        target_aggregate_repo_path=target_aggregate_repo_path,
        samples_path=samples_path,
        samples_repo_path=samples_repo_path,
        target_samples_path=target_samples_path,
        target_samples_repo_path=target_samples_repo_path,
        compression=compression,
        samples_hash_algorithm=samples_hash_algorithm,
    )


def _candidate_aggregate_bytes(
    plan: TranscodePlan,
    samples_checksum: str | None,
) -> bytes:
    source_bytes = eee_io.read_eee_bytes(plan.aggregate_path)
    if plan.samples_path == plan.target_samples_path:
        return source_bytes
    data = strict_json_loads(source_bytes)
    if not isinstance(data, dict):
        raise TranscodeError(
            f'aggregate {plan.aggregate_path} must contain a JSON object'
        )
    detail = data.get('detailed_evaluation_results')
    if not isinstance(detail, dict):
        raise TranscodeError(
            f'{plan.aggregate_path} is missing detailed_evaluation_results'
        )
    if plan.target_samples_repo_path is None:
        raise AssertionError('sample target path was not prepared')
    detail['file_path'] = plan.target_samples_repo_path
    if plan.samples_hash_algorithm is not None:
        if samples_checksum is None:
            raise AssertionError('sample checksum was not prepared')
        detail['hash_algorithm'] = plan.samples_hash_algorithm
        detail['checksum'] = samples_checksum
    return (
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + '\n'
    ).encode('utf-8')


def _write_sample_candidate(source: Path, target: Path) -> None:
    logical_digest = hashlib.sha256()
    with eee_io.open_eee_binary(target, 'wb') as output:
        for chunk in eee_io.iter_eee_binary_chunks(source):
            logical_digest.update(chunk)
            output.write(chunk)
    roundtrip_digest = _sha256_logical(target)
    if roundtrip_digest != logical_digest.hexdigest():
        raise TranscodeError(
            f'transcode round-trip changed the logical bytes of {source}'
        )


def _validate_candidates(
    plan: TranscodePlan,
    aggregate_candidate: Path,
    samples_candidate: Path | None,
) -> None:
    files = {plan.target_aggregate_repo_path: aggregate_candidate}
    if plan.target_samples_repo_path is not None:
        if samples_candidate is None:
            raise AssertionError('sample candidate path is missing')
        files[plan.target_samples_repo_path] = samples_candidate
    repository = _RepositoryFiles(files)
    _require_valid(
        aggregate_candidate,
        repo_path=plan.target_aggregate_repo_path,
        repository=repository,
    )
    if samples_candidate is not None and plan.target_samples_repo_path is not None:
        _require_valid(
            samples_candidate,
            repo_path=plan.target_samples_repo_path,
            repository=repository,
        )


def _commit_candidates(
    plan: TranscodePlan,
    aggregate_candidate: Path,
    samples_candidate: Path | None,
    staging_dir: Path,
) -> None:
    replacements: list[tuple[Path, Path, Path]] = []
    if plan.samples_path is not None and plan.target_samples_path is not None:
        if plan.samples_path != plan.target_samples_path:
            if samples_candidate is None:
                raise AssertionError('sample candidate was not prepared')
            replacements.append(
                (plan.samples_path, plan.target_samples_path, samples_candidate)
            )
    if (
        plan.aggregate_path != plan.target_aggregate_path
        or plan.samples_path != plan.target_samples_path
    ):
        replacements.append(
            (
                plan.aggregate_path,
                plan.target_aggregate_path,
                aggregate_candidate,
            )
        )

    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for index, (source, _target, _candidate) in enumerate(replacements):
            backup = staging_dir / f'backup-{index}-{source.name}'
            os.replace(source, backup)
            backups.append((source, backup))
        for _source, target, candidate in replacements:
            if target.exists():
                raise FileExistsError(
                    f'refusing to overwrite file that appeared during '
                    f'transcode: {target}'
                )
            os.replace(candidate, target)
            installed.append(target)
    except Exception as exc:
        rollback_errors: list[str] = []
        for target in reversed(installed):
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f'could not remove {target}: {rollback_exc}')
        for source, backup in reversed(backups):
            try:
                os.replace(backup, source)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f'could not restore {source} from {backup}: {rollback_exc}'
                )
        if rollback_errors:
            raise TranscodeRollbackError(
                'transcode commit failed and rollback was incomplete; '
                f'preserving recovery files in {staging_dir}: '
                + '; '.join(rollback_errors)
            ) from exc
        raise TranscodeError(f'transcode commit failed: {exc}') from exc


def execute_transcode(
    plan: TranscodePlan, *, dry_run: bool = False
) -> TranscodeResult:
    """Build and validate a target representation, then optionally commit it."""
    if not plan.changed:
        return TranscodeResult(
            source_aggregate_path=plan.aggregate_path,
            aggregate_path=plan.aggregate_path,
            source_samples_path=plan.samples_path,
            samples_path=plan.samples_path,
            changed=False,
        )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix='.eee-transcode-',
            dir=plan.aggregate_path.parent,
        )
    )
    preserve_staging = False
    try:
        samples_candidate: Path | None = None
        samples_checksum: str | None = None
        if plan.samples_path is not None:
            if plan.target_samples_path is None:
                raise AssertionError('sample target path is missing')
            if plan.samples_path == plan.target_samples_path:
                samples_candidate = plan.samples_path
            else:
                samples_candidate = staging_dir / plan.target_samples_path.name
                _write_sample_candidate(plan.samples_path, samples_candidate)
            if plan.samples_hash_algorithm is not None:
                samples_checksum = _stored_digest(
                    samples_candidate, plan.samples_hash_algorithm
                )

        aggregate_payload = _candidate_aggregate_bytes(
            plan, samples_checksum
        )
        aggregate_candidate = staging_dir / plan.target_aggregate_path.name
        aggregate_candidate.write_bytes(
            eee_io.compress_bytes(aggregate_payload, plan.compression)
        )
        if eee_io.read_eee_bytes(aggregate_candidate) != aggregate_payload:
            raise TranscodeError(
                f'transcode round-trip changed aggregate bytes for '
                f'{plan.aggregate_path}'
            )

        candidate_samples_for_validation = samples_candidate
        _validate_candidates(
            plan,
            aggregate_candidate,
            candidate_samples_for_validation,
        )
        if samples_candidate is not None:
            candidate_data = _load_aggregate(aggregate_candidate)
            _verify_samples_checksum(
                aggregate_candidate,
                candidate_data,
                samples_candidate,
            )

        if not dry_run:
            try:
                _commit_candidates(
                    plan,
                    aggregate_candidate,
                    (
                        samples_candidate
                        if plan.samples_path != plan.target_samples_path
                        else None
                    ),
                    staging_dir,
                )
            except TranscodeRollbackError:
                preserve_staging = True
                raise
    finally:
        if not preserve_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return TranscodeResult(
        source_aggregate_path=plan.aggregate_path,
        aggregate_path=plan.target_aggregate_path,
        source_samples_path=plan.samples_path,
        samples_path=plan.target_samples_path,
        changed=True,
    )


def _expand_input(value: str) -> list[Path]:
    matches = (
        [Path(match) for match in sorted(glob.glob(value, recursive='**' in value))]
        if glob.has_magic(value)
        else [Path(value)]
    )
    if not matches:
        raise TranscodeError(f'file pattern matched no files: {value!r}')
    result: list[Path] = []
    for path in matches:
        if path.is_dir():
            result.extend(eee_io.iter_eee_results([path]))
        elif path.is_file():
            if eee_io.is_eee_result(path) is None:
                raise TranscodeError(f'not an EEE result file: {path}')
            result.append(path)
        else:
            raise TranscodeError(f'file or directory does not exist: {path}')
    return result


def expand_transcode_inputs(values: Iterable[str]) -> list[Path]:
    """Resolve files/directories/globs to unique aggregate result paths."""
    physical: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        for path in _expand_input(value):
            identity = path.absolute()
            if identity not in seen:
                physical.append(path)
                seen.add(identity)

    duplicate_variants = eee_io.find_duplicate_variants(physical)
    if duplicate_variants:
        variants = duplicate_variants[0][3]
        raise TranscodeError(
            'multiple physical variants exist for one logical result: '
            + ', '.join(str(path) for path in variants)
        )

    aggregates: dict[Path, Path] = {}
    samples: list[Path] = []
    for path in physical:
        kind = eee_io.is_eee_result(path)
        if kind == 'aggregate':
            aggregates[eee_io.strip_compression_suffix(path)] = path
        elif kind == 'samples':
            samples.append(path)

    for sample in samples:
        aggregate_logical = _logical_aggregate_path(sample)
        aggregate = _single_variant(
            aggregate_logical, kind='aggregate result'
        )
        if aggregate is None:
            raise TranscodeError(
                f'samples file has no sibling aggregate: {sample}'
            )
        aggregates[aggregate_logical] = aggregate

    if not aggregates:
        raise TranscodeError('no aggregate EEE results were found')
    return [aggregates[key] for key in sorted(aggregates, key=str)]


def transcode_paths(
    values: Iterable[str],
    compression: str,
    *,
    dry_run: bool = False,
) -> list[TranscodeResult]:
    """Transcode selected logical records, preflighting all before mutation."""
    compression = eee_io.normalize_compression(compression)
    aggregate_paths = expand_transcode_inputs(values)

    # Fail before touching any result if the requested optional codec is absent.
    eee_io.compress_bytes(b'', compression)
    plans = [prepare_transcode(path, compression) for path in aggregate_paths]
    return [
        execute_transcode(plan, dry_run=dry_run)
        for plan in plans
    ]
