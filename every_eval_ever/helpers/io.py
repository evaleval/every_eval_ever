"""File I/O utilities for saving evaluation logs."""

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Generic, Iterable, TypeVar, Union

from every_eval_ever.eval_types import EvaluationLog

_INVALID_PATH_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    'CON',
    'PRN',
    'AUX',
    'NUL',
    *(f'COM{index}' for index in range(1, 10)),
    *(f'LPT{index}' for index in range(1, 10)),
}
_RecordT = TypeVar('_RecordT')


@dataclass(frozen=True)
class SourceRecordFailure:
    """One rejected upstream record and the reason it was rejected."""

    source_ref: str
    reason: str
    source_record: Any | None = None

    def model_dump(self) -> dict[str, Any]:
        result = {
            'source_ref': self.source_ref,
            'reason': self.reason,
        }
        if self.source_record is not None:
            result['source_record'] = self.source_record
        return result


@dataclass(frozen=True)
class SourceRecordExclusion:
    """One intentionally non-evaluation source record and its rationale."""

    source_ref: str
    reason: str
    source_record: Any | None = None

    def model_dump(self) -> dict[str, Any]:
        result = {
            'source_ref': self.source_ref,
            'reason': self.reason,
        }
        if self.source_record is not None:
            result['source_record'] = self.source_record
        return result


class SourceRecordsError(ValueError):
    """Raised after a conversion encounters one or more source issues."""

    def __init__(
        self,
        source_name: str,
        total_records: int,
        failures: list[SourceRecordFailure],
    ) -> None:
        self.source_name = source_name
        self.total_records = total_records
        self.failures = failures

        preview = '; '.join(
            f'{failure.source_ref}: {failure.reason}'
            for failure in failures[:10]
        )
        if len(failures) > 10:
            preview += f'; ... {len(failures) - 10} more'
        super().__init__(
            f'{source_name}: encountered {len(failures)} conversion issue(s) '
            f'across {total_records} source record(s) ({preview})'
        )

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-ready failure report for logs or a raw-data ledger."""
        return {
            'source_name': self.source_name,
            'total_records': self.total_records,
            'failed_records': [
                failure.model_dump() for failure in self.failures
            ],
        }


@dataclass(frozen=True)
class SourceConversionResult(Generic[_RecordT]):
    """Successful conversions plus any rejected upstream records."""

    source_name: str
    total_records: int
    records: list[_RecordT]
    failures: list[SourceRecordFailure]
    exclusions: list[SourceRecordExclusion] = field(default_factory=list)

    def raise_if_incomplete(self) -> None:
        """Signal partial conversion after callers preserve valid outputs."""
        raise_for_failed_records(
            self.source_name,
            self.total_records,
            self.failures,
        )

    def failure_report(self) -> dict[str, Any]:
        """Return a strict JSON-ready partial-conversion report."""
        return {
            'source_name': self.source_name,
            'total_source_records': self.total_records,
            'converted_records': len(self.records),
            'failed_record_count': len(self.failures),
            'excluded_record_count': len(self.exclusions),
            'failed_records': [
                failure.model_dump() for failure in self.failures
            ],
            'excluded_records': [
                exclusion.model_dump() for exclusion in self.exclusions
            ],
        }


@dataclass(frozen=True)
class EvaluationLogOutput:
    """One evaluation log and its explicit datastore destination."""

    eval_log: EvaluationLog
    base_dir: Union[str, Path]
    developer: str
    model_name: str

    def __post_init__(self) -> None:
        """Normalize explicit routing before an output enters a batch.

        Model identities may contain additional slash-separated namespaces,
        but the datastore permits exactly one model directory. Flatten those
        namespaces for the filesystem while leaving ``eval_log.model_info``
        unchanged. Doing this at construction time also keeps a bad route
        inside an adapter's per-record error boundary instead of failing the
        whole batch during publication.
        """
        object.__setattr__(
            self,
            'developer',
            _required_path_component(self.developer, 'model developer'),
        )
        object.__setattr__(
            self,
            'model_name',
            _flatten_path_components(self.model_name, 'model name'),
        )


@dataclass(frozen=True)
class _PreparedEvaluationLog:
    path: Path
    json_text: str


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename/directory name.

    Replaces characters that are invalid on common filesystems.

    Args:
        name: The string to sanitize

    Returns:
        Sanitized string safe for filesystem use
    """
    # Replace characters invalid on Windows/Unix filesystems
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def require_identity(value: str | None, field_name: str) -> str:
    """Return a non-placeholder identity value or raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} is required')
    value = value.strip()
    if value.lower() == 'unknown':
        raise ValueError(f'{field_name} must be known')
    return value


def require_finite_number(value: Any, field_name: str) -> float:
    """Parse one required numeric source value and reject NaN/infinity."""
    if isinstance(value, bool):
        raise ValueError(f'{field_name} must be numeric; got {value!r}')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{field_name} must be numeric; got {value!r}'
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f'{field_name} must be finite; got {value!r}')
    return number


def _required_path_component(value: str | None, field_name: str) -> str:
    """Validate one explicit datastore path component."""
    value = require_identity(value, field_name)
    # Colons are common in model/version identifiers but are not valid in
    # Windows filenames. Preserve the identity in the JSON; only make its
    # output directory portable.
    value = value.replace(':', '_')
    if (
        value in {'.', '..'}
        or '/' in value
        or _INVALID_PATH_CHARS.search(value)
        or value.endswith(('.', ' '))
        or value.split('.', 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f'{field_name} is not a safe single datastore path component: '
            f'{value!r}'
        )
    if value == 'data':
        raise ValueError(
            f'{field_name} cannot use the reserved datastore name "data"'
        )
    return value


def _flatten_path_components(value: str | None, field_name: str) -> str:
    """Flatten a slash-separated identity into one safe path component."""
    value = require_identity(value, field_name)
    parts = value.split('/')
    if any(not part for part in parts):
        raise ValueError(f'invalid {field_name}: {value!r}')
    return '_'.join(
        _required_path_component(part, field_name) for part in parts
    )


def datastore_path_components(
    collection: str | None,
    model_id: str | None,
    developer: str | None = None,
) -> tuple[str, str, str]:
    """Resolve the exact collection/developer/model datastore components.

    Slash-separated collection and model identifiers are flattened with
    underscores. Empty, unknown, or otherwise unsafe identity components are
    rejected instead of being replaced with placeholders.
    """

    if not isinstance(collection, str):
        raise ValueError('collection is required for the datastore path')
    collection_parts = collection.strip().split('/')
    if not collection_parts or any(not part for part in collection_parts):
        raise ValueError(f'invalid collection identity: {collection!r}')
    collection_name = _flatten_path_components(collection, 'collection')

    if not isinstance(model_id, str):
        raise ValueError('model_info.id is required for the datastore path')
    model_parts = model_id.strip().split('/')
    if not model_parts or any(not part for part in model_parts):
        raise ValueError(f'invalid model_info.id: {model_id!r}')

    if len(model_parts) >= 2:
        developer_name = _required_path_component(
            model_parts[0], 'model developer'
        )
        model_name = _flatten_path_components(
            '/'.join(model_parts[1:]), 'model name'
        )
    else:
        developer_name = _required_path_component(
            developer, 'model_info.developer'
        )
        model_name = _required_path_component(model_parts[0], 'model name')

    return collection_name, developer_name, model_name


def datastore_output_dir(
    base_dir: Union[str, Path],
    collection: str | None,
    model_id: str | None,
    developer: str | None = None,
) -> Path:
    """Return the one allowed output directory for an evaluation log."""
    collection_name, developer_name, model_name = datastore_path_components(
        collection,
        model_id,
        developer,
    )
    return Path(base_dir) / collection_name / developer_name / model_name


def datastore_repo_file_path(
    collection: str | None,
    model_id: str | None,
    developer: str | None,
    filename: str | None,
) -> str:
    """Return a canonical repository-relative path for one datastore file."""
    collection_name, developer_name, model_name = datastore_path_components(
        collection,
        model_id,
        developer,
    )
    filename = _required_path_component(filename, 'datastore filename')
    return PurePosixPath(
        'data',
        collection_name,
        developer_name,
        model_name,
        filename,
    ).as_posix()


def require_uuid4(value: str | None, field_name: str = 'file UUID') -> str:
    """Return a canonical UUIDv4 string or raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} is required')
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid {field_name}: {value!r}') from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError(f'{field_name} must be UUIDv4: {value!r}')
    return str(parsed)


def _create_parent_directories(parent: Path) -> list[Path]:
    """Create missing parents individually and return only those we created."""
    missing = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent

    created = []
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            created.append(directory)
    return created


def _remove_empty_directories(directories: Iterable[Path]) -> None:
    """Remove directories created by a failed publication, if still empty."""
    for directory in reversed(list(dict.fromkeys(directories))):
        try:
            directory.rmdir()
        except OSError:
            pass


def raise_for_failed_records(
    source_name: str,
    total_records: int,
    failures: Iterable[SourceRecordFailure | tuple[int, str]],
) -> None:
    """Fail without dropping records, retaining source failure provenance."""
    normalized = [
        failure
        if isinstance(failure, SourceRecordFailure)
        else SourceRecordFailure(
            source_ref=f'row {failure[0]}',
            reason=failure[1],
        )
        for failure in failures
    ]
    if not normalized:
        return
    raise SourceRecordsError(source_name, total_records, normalized)


def default_failure_report_path(
    output_dir: Union[str, Path],
) -> Path:
    """Place adapter failure provenance outside the validated data tree.

    A report is not an ``EvaluationLog`` and must never be placed under
    ``data/<collection>/...`` where a PR validator could mistake it for one.
    For a normal ``data/<collection>`` output this returns
    ``adapter_reports/<collection>_failures.json``.
    """
    output_path = Path(output_dir)
    data_root = next(
        (path for path in output_path.parents if path.name == 'data'),
        None,
    )
    if data_root is not None:
        report_root = data_root.parent / 'adapter_reports'
        report_stem = sanitize_filename(
            '__'.join(output_path.relative_to(data_root).parts)
        )
    else:
        report_root = output_path.parent / 'adapter_reports'
        report_stem = sanitize_filename(output_path.name)
    return report_root / f'{report_stem}_failures.json'


def save_failure_report(
    result: SourceConversionResult[Any],
    path: Union[str, Path],
) -> Path:
    """Persist rejected source records and reasons as strict JSON."""
    report_path = Path(path)
    report_text = (
        json.dumps(
            _strict_provenance_value(result.failure_report()),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + '\n'
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding='utf-8')
    return report_path


def _strict_provenance_value(value: Any) -> Any:
    """Convert raw source data to strict JSON without concealing non-finite values.

    Evaluation records reject non-finite values. Provenance must retain the
    fact that an upstream value was ``NaN`` or infinity while still producing
    valid JSON, so those values are represented by an explicit tagged object.
    Other unsupported values are an error rather than an implicit ``str()``
    fallback.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        label = (
            'NaN'
            if math.isnan(value)
            else 'Infinity'
            if value > 0
            else '-Infinity'
        )
        return {'__eee_nonfinite_float__': label}
    if isinstance(value, list):
        return [_strict_provenance_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('provenance object keys must be strings')
        return {
            key: _strict_provenance_value(item) for key, item in value.items()
        }
    raise TypeError(
        f'provenance values must be JSON-compatible; got {type(value).__name__}'
    )


def generate_output_path(
    base_dir: Union[str, Path],
    developer: str,
    model_name: str,
) -> Path:
    """
    Generate the output directory path for an evaluation log.

    Creates the standard structure: {base_dir}/{developer}/{model_name}/

    Args:
        base_dir: Base output directory (e.g., "data/helm_lite")
        developer: Developer/organization name
        model_name: Model name (without developer prefix)

    Returns:
        Path object for the output directory
    """
    developer = _required_path_component(developer, 'model developer')
    model_name = _required_path_component(model_name, 'model name')

    return Path(base_dir) / developer / model_name


def save_evaluation_log(
    eval_log: EvaluationLog,
    base_dir: Union[str, Path],
    developer: str,
    model_name: str,
) -> Path:
    """
    Save an evaluation log to the standard directory structure.

    Creates: {base_dir}/{developer}/{model_name}/{uuid}.json

    Args:
        eval_log: The EvaluationLog to save
        base_dir: Base output directory (e.g., "data/helm_lite")
        developer: Developer/organization name
        model_name: Model name (without developer prefix)

    Returns:
        Path to the saved file

    Example:
        >>> save_evaluation_log(log, "data/helm_lite", "anthropic", "claude-3-opus")
        PosixPath('data/helm_lite/anthropic/claude-3-opus/a1b2c3d4-....json')
    """
    return save_evaluation_logs(
        [
            EvaluationLogOutput(
                eval_log=eval_log,
                base_dir=base_dir,
                developer=developer,
                model_name=model_name,
            )
        ]
    )[0]


def _prepare_evaluation_logs(
    outputs: Iterable[EvaluationLogOutput],
) -> list[_PreparedEvaluationLog]:
    """Validate every output path and JSON body before writing any file."""
    prepared = []
    paths = set()
    route_owners: dict[Path, tuple[str, str]] = {}
    for output in outputs:
        base_dir = Path(output.base_dir)
        collection_identity = base_dir.name
        collection = _required_path_component(base_dir.name, 'collection')
        base_dir = base_dir.with_name(collection)
        # Revalidate generated/dataclass-constructed values at the publication
        # boundary so all schema validators run before the first write.
        validated = EvaluationLog.model_validate(output.eval_log.model_dump())
        dir_path = generate_output_path(
            base_dir,
            output.developer,
            output.model_name,
        )
        route_owner = (collection_identity, validated.model_info.id)
        existing_owner = route_owners.get(dir_path)
        if existing_owner is not None and existing_owner != route_owner:
            raise ValueError(
                'distinct collection/model identities resolve to the same '
                f'datastore directory {dir_path}: {existing_owner!r} and '
                f'{route_owner!r}'
            )
        route_owners[dir_path] = route_owner
        path = dir_path / f'{uuid.uuid4()}.json'
        if path in paths or path.exists():
            raise FileExistsError(f'refusing to overwrite output file {path}')
        paths.add(path)

        json_text = json.dumps(
            validated.model_dump(mode='json', exclude_none=True),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        prepared.append(
            _PreparedEvaluationLog(path=path, json_text=json_text + '\n')
        )
    return prepared


def save_evaluation_logs(
    outputs: Iterable[EvaluationLogOutput],
) -> list[Path]:
    """Validate a batch first, then publish it without partial files.

    Conversion, schema, path, and JSON serialization failures are all detected
    before the first file is created. If a later filesystem write fails, files
    created by this call are removed before the error is re-raised.
    """
    prepared = _prepare_evaluation_logs(outputs)
    created: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for output in prepared:
            created_dirs.extend(_create_parent_directories(output.path.parent))
            with output.path.open('x', encoding='utf-8') as file:
                created.append(output.path)
                file.write(output.json_text)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        _remove_empty_directories(created_dirs)
        raise
    return [output.path for output in prepared]
