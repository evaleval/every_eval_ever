"""Helpers for loading bundled JSON schemas."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def schema_text(name: str = "eval.schema.json") -> str:
    with schema_path(name) as path:
        return path.read_text(encoding="utf-8")


def schema_json(name: str = "eval.schema.json") -> dict[str, Any]:
    return json.loads(schema_text(name))


class _SchemaPathContext:
    """Context manager for a bundled schema file.

    If the package is installed on disk, this usually yields the existing schema
    path directly. For zipped or other non-filesystem imports, it materializes the
    resource as a temporary file for the lifetime of the context.
    """

    def __init__(self, name: str = "eval.schema.json") -> None:
        resource = resources.files("every_eval_ever.schemas").joinpath(name)
        self._context = resources.as_file(resource)

    def __enter__(self) -> Path:
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.__exit__(exc_type, exc_value, traceback)
        return None


def schema_path(name: str = "eval.schema.json") -> _SchemaPathContext:
    return _SchemaPathContext(name)
