"""Failure-safe publication for converter aggregate and sample artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def _prepare_sample_artifact(
    log: EvaluationLog,
    file_uuid: str,
    output_dir: Path,
    staged_output_dir: Path | None,
    collection_override: str | None,
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
            'repository path and UUID: expected '
            f'{expected_repo_path!r}, got {detailed.file_path!r}'
        )
    source_path = (
        _output_dir(staged_output_dir, log, collection_override) / expected_name
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f'converter sample artifact was not staged at {source_path}'
        )

    content = source_path.read_bytes()
    lines = content.splitlines()
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
    checksum = hashlib.sha256(content).hexdigest()
    if checksum != detailed.checksum:
        raise ValueError(
            'staged sample checksum does not match '
            'detailed_evaluation_results.checksum'
        )

    return _PreparedArtifact(
        path=output_dir / expected_name,
        content=content,
    )


def publish_evaluation_logs(
    logs: Iterable[EvaluationLog],
    base_output_dir: str | Path,
    file_uuids: Iterable[str],
    *,
    staged_output_dir: str | Path | None = None,
    collection_override: str | None = None,
) -> list[Path]:
    """Validate and atomically publish a converter batch.

    All aggregate and instance-level artifacts are prepared and preflighted
    before any destination file is created. Any publication failure removes
    only files successfully created by this call.
    """

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
    planned_paths: set[Path] = set()
    route_owners: dict[Path, tuple[str, str]] = {}

    for raw_log, file_uuid in zip(logs, file_uuids):
        log = EvaluationLog.model_validate(raw_log.model_dump())
        output_dir = _output_dir(base_output_dir, log, collection_override)
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
        aggregate_path = output_dir / f'{file_uuid}.json'
        sample = _prepare_sample_artifact(
            log,
            file_uuid,
            output_dir,
            staged_root,
            collection_override,
        )
        aggregate = _PreparedArtifact(
            path=aggregate_path,
            content=(
                json.dumps(
                    log.model_dump(mode='json', exclude_none=True),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + '\n'
            ).encode('utf-8'),
        )
        artifacts = [aggregate] if sample is None else [sample, aggregate]
        for artifact in artifacts:
            if artifact.path in planned_paths or artifact.path.exists():
                raise FileExistsError(
                    f'refusing to overwrite output file {artifact.path}'
                )
            planned_paths.add(artifact.path)
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
