"""every_eval_ever public package API."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["eval_types", "instance_level_types"]


def __getattr__(name: str) -> Any:
    if name in {"eval_types", "instance_level_types"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
