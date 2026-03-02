"""Helpers for loading bundled JSON schemas."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def schema_text(name: str = "eval.schema.json") -> str:
    return resources.files("every_eval_ever.schemas").joinpath(name).read_text(encoding="utf-8")


def schema_json(name: str = "eval.schema.json") -> dict[str, Any]:
    return json.loads(schema_text(name))


def schema_path(name: str = "eval.schema.json") -> Path:
    return Path(resources.files("every_eval_ever.schemas").joinpath(name))
