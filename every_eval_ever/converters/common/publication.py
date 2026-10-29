"""Failure-safe publication for converter aggregate and sample artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from every_eval_ever import io as eee_io
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers.io import (
    _create_parent_directories,
    _remove_empty_directories,
    datastore_output_dir,
    datastore_repo_file_path,
    require_uuid4,
)
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog


@dataclass(frozen=True)
class _PreparedArtifact:
    path: Path
    content: bytes


def _output_dir(
    base_dir: Path,
    log: EvaluationLog,
    collection_override: str | None = None,
) -> Path:
    if not log.evaluation_results:
        raise ValueError(
            'evaluation_results must contain at least one result so the '
            'output collection can be determined'
        )
    source_data = log.evaluation_results[0].source_data
    if source_data is None:
        raise ValueError(
            'evaluation_results[0].source_data is required for the output path'
        )
    return datastore_output_dir(
        base_dir,
        collection_override or source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
    )


def _reject_base_below_data(base_dir: Path, output_dir: Path) -> None:
    """Refuse a *base_dir* that is a child of a ``data`` directory.

    Every path this publisher writes is ``data/<collection>/<dev>/<model>/``, so
    a base one level inside ``data`` publishes one level too deep whatever it is
    named — the caller having passed ``data/<collection>``, the convention
    ``EvaluationLogOutput`` uses, or a directory whose name is not the
    collection being published at all. A scratch root outside ``data`` still
    publishes, even one that shares a collection's name.
    """
    if base_dir.parent.name == 'data':
        raise ValueError(
            f'base_output_dir {base_dir}/ is inside a datastore data/ '
            "directory; pass the 'data' directory itself, or this batch "
            f'publishes one level too deep at {output_dir}/'
        )


def _reject_existing_variant(path: Path) -> None:
    variants = eee_io.existing_local_variants(path)
    if variants:
        rendered = ', '.join(str(variant) for variant in variants)
        raise FileExistsError(
            'refusing to create another physical variant of one logical '
            f'EEE result: {rendered}'
        )


def _prepare_sample_artifact(
    log: EvaluationLog,
    file_uuid: str,
    output_dir: Path,
    staged_output_dir: Path | None,
    collection_override: str | None,
    compression: eee_io.Compression,
) -> _PreparedArtifact | None:
    detailed = log.detailed_evaluation_results
    if detailed is None:
        return None
    if staged_output_dir is None:
        raise ValueError(
            'staged_output_dir is required when publishing instance-level data'
        )

    expected_name = f'{file_uuid}_samples.jsonl'
    if not log.evaluation_results:
        raise ValueError(
            'evaluation_results must contain at least one result so the '
            'sample repository path can be determined'
        )
    source_data = log.evaluation_results[0].source_data
    if source_data is None:
        raise ValueError(
            'evaluation_results[0].source_data is required for the sample '
            'repository path'
        )
    expected_repo_path = datastore_repo_file_path(
        collection_override or source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
        expected_name,
    )
    if detailed.file_path != expected_repo_path:
        raise ValueError(
            'detailed_evaluation_results.file_path must match the aggregate '
            'repository path and UUID before publication: expected '
            f'{expected_repo_path!r}, got {detailed.file_path!r}'
        )
    source_path = (
        _output_dir(staged_output_dir, log, collection_override) / expected_name
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f'converter sample artifact was not staged at {source_path}'
        )

    uncompressed_content = source_path.read_bytes()
    lines = uncompressed_content.splitlines()
    if len(lines) != detailed.total_rows:
        raise ValueError(
            f'staged sample row count is {len(lines)}, expected '
            f'{detailed.total_rows}'
        )
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(
                f'staged sample artifact contains a blank row at {line_number}'
            )
        row = InstanceLevelEvaluationLog.model_validate_json(line)
        if row.evaluation_id != log.evaluation_id:
            raise ValueError(
                f'sample row {line_number} evaluation_id does not match '
                'the aggregate'
            )
        if row.model_id != log.model_info.id:
            raise ValueError(
                f'sample row {line_number} model_id does not match the '
                'aggregate'
            )

    algorithm = getattr(
        detailed.hash_algorithm, 'value', detailed.hash_algorithm
    )
    if algorithm != 'sha256':
        raise ValueError(
            f'unsupported detailed-results hash algorithm: {algorithm!r}'
        )
    staged_checksum = hashlib.sha256(uncompressed_content).hexdigest()
    if staged_checksum != detailed.checksum:
        raise ValueError(
            'staged sample checksum does not match '
            'detailed_evaluation_results.checksum'
        )

    content = eee_io.compress_bytes(uncompressed_content, compression)
    physical_name = eee_io.add_compression_suffix(
        Path(expected_name), compression
    ).name
    physical_repo_path = datastore_repo_file_path(
        collection_override or source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
        physical_name,
    )
    detailed.file_path = physical_repo_path
    detailed.checksum = hashlib.sha256(content).hexdigest()

    return _PreparedArtifact(
        path=output_dir / physical_name,
        content=content,
    )


def publish_evaluation_logs(
    logs: Iterable[EvaluationLog],
    base_output_dir: str | Path,
    file_uuids: Iterable[str],
    *,
    staged_output_dir: str | Path | None = None,
    collection_override: str | None = None,
    aggregate_compression: eee_io.Compression = eee_io.COMPRESSION_NONE,
    samples_compression: eee_io.Compression = eee_io.COMPRESSION_NONE,
) -> list[Path]:
    """Validate and atomically publish a converter batch.

    All aggregate and instance-level artifacts are prepared and preflighted
    before any destination file is created. Any publication failure removes
    only files successfully created by this call.

    ``base_output_dir`` is the **``data``** directory, not a collection
    directory: the collection comes from ``collection_override`` or from
    ``evaluation_results[0].source_data.dataset_name``, and the final path is
    ``base_output_dir/<collection>/<developer>/<model>/<uuid>.json``.
    ``EvaluationLogOutput.base_dir`` and ``default_failure_report_path`` take
    the opposite convention, ``data/<collection>``.

    Converter adapters always stage ordinary UTF-8 JSON/JSONL. Compression is
    a publication concern: this function optionally compresses the final bytes,
    updates the aggregate's companion path/checksum to describe those bytes,
    then preflights and writes the whole batch atomically.

    Raises:
        ValueError: If *base_output_dir* is a child of a ``data`` directory,
            which would publish one level too deep whatever it is named.
    """
    aggregate_compression = eee_io.normalize_compression(
        aggregate_compression
    )
    samples_compression = eee_io.normalize_compression(samples_compression)

    logs = list(logs)
    file_uuids = [require_uuid4(value) for value in file_uuids]
    if len(logs) != len(file_uuids):
        raise ValueError(
            'converter log count must match the generated UUID count'
        )

    base_output_dir = Path(base_output_dir)
    staged_root = (
        Path(staged_output_dir) if staged_output_dir is not None else None
    )
    prepared: list[_PreparedArtifact] = []
    aggregate_paths: list[Path] = []
    planned_keys: set[eee_io.ResultKey] = set()
    route_owners: dict[Path, tuple[str, str]] = {}

    for raw_log, file_uuid in zip(logs, file_uuids):
        log = EvaluationLog.model_validate(raw_log.model_dump())
        output_dir = _output_dir(base_output_dir, log, collection_override)
        _reject_base_below_data(base_output_dir, output_dir)
        source_data = log.evaluation_results[0].source_data
        route_owner = (
            collection_override or source_data.dataset_name,
            log.model_info.id,
        )
        existing_owner = route_owners.get(output_dir)
        if existing_owner is not None and existing_owner != route_owner:
            raise ValueError(
                'distinct collection/model identities resolve to the same '
                f'datastore directory {output_dir}: {existing_owner!r} and '
                f'{route_owner!r}'
            )
        route_owners[output_dir] = route_owner

        sample = _prepare_sample_artifact(
            log,
            file_uuid,
            output_dir,
            staged_root,
            collection_override,
            samples_compression,
        )
        aggregate_base = output_dir / f'{file_uuid}.json'
        aggregate_path = eee_io.add_compression_suffix(
            aggregate_base, aggregate_compression
        )
        aggregate_bytes = (
            json.dumps(
                log.model_dump(mode='json', exclude_none=True),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + '\n'
        ).encode('utf-8')
        aggregate = _PreparedArtifact(
            path=aggregate_path,
            content=eee_io.compress_bytes(
                aggregate_bytes, aggregate_compression
            ),
        )

        artifacts = [aggregate] if sample is None else [sample, aggregate]
        for artifact in artifacts:
            key = eee_io.logical_result_key(artifact.path)
            if key is None:
                raise AssertionError(f'not an EEE result path: {artifact.path}')
            if key in planned_keys:
                raise FileExistsError(
                    'batch would create multiple physical variants of one '
                    f'logical EEE result: {artifact.path}'
                )
            _reject_existing_variant(artifact.path)
            planned_keys.add(key)
            prepared.append(artifact)
        aggregate_paths.append(aggregate_path)

    created: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for artifact in prepared:
            created_dirs.extend(
                _create_parent_directories(artifact.path.parent)
            )
            with artifact.path.open('xb') as handle:
                created.append(artifact.path)
                handle.write(artifact.content)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        _remove_empty_directories(created_dirs)
        raise

    return aggregate_paths
