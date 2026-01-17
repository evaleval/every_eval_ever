"""File I/O utilities for saving evaluation logs."""

import re
import uuid
from pathlib import Path
from typing import Union

from eval_types import EvaluationLog


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
    return re.sub(r'[<>:"/\\|?*]', "_", name)


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
    developer = sanitize_filename(developer)
    model_name = sanitize_filename(model_name)

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

    filename = f"{uuid.uuid4()}.json"
    filepath = dir_path / filename

    json_str = eval_log.model_dump_json(indent=2, exclude_none=True)
    filepath.write_text(json_str)

    return filepath
