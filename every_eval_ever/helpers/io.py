"""File I/O utilities for saving evaluation logs."""

import json
import re
import uuid
from pathlib import Path
from typing import Union

from every_eval_ever.eval_types import EvaluationLog

_INVALID_PATH_CHARS = re.compile(r'[<>:"\\|?*]')


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
    ):
        raise ValueError(
            f'{field_name} is not a safe single datastore path component: '
            f'{value!r}'
        )
    return value


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
    collection_name = '_'.join(
        _required_path_component(part, 'collection')
        for part in collection_parts
    )

    if not isinstance(model_id, str):
        raise ValueError('model_info.id is required for the datastore path')
    model_parts = model_id.strip().split('/')
    if not model_parts or any(not part for part in model_parts):
        raise ValueError(f'invalid model_info.id: {model_id!r}')

    if len(model_parts) >= 2:
        developer_name = _required_path_component(
            model_parts[0], 'model developer'
        )
        model_name = '_'.join(
            _required_path_component(part, 'model name')
            for part in model_parts[1:]
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


def require_uuid4(value: str | None, field_name: str = 'file UUID') -> str:
    """Return a canonical UUIDv4 string or raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} is required')
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid {field_name}: {value!r}') from exc
    if parsed.version != 4:
        raise ValueError(f'{field_name} must be UUIDv4: {value!r}')
    return str(parsed)


def raise_for_failed_records(
    source_name: str,
    total_records: int,
    failures: list[tuple[int, str]],
) -> None:
    """Fail a conversion with an explicit rejected-record count."""
    if not failures:
        return
    preview = '; '.join(
        f'row {index}: {reason}' for index, reason in failures[:10]
    )
    if len(failures) > 10:
        preview += f'; ... {len(failures) - 10} more'
    raise ValueError(
        f'{source_name}: failed to convert {len(failures)} of '
        f'{total_records} source records ({preview})'
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
    dir_path = generate_output_path(base_dir, developer, model_name)
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = f'{uuid.uuid4()}.json'
    filepath = dir_path / filename

    json_str = json.dumps(
        eval_log.model_dump(mode='json', exclude_none=True),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    filepath.write_text(json_str + '\n', encoding='utf-8')

    return filepath
