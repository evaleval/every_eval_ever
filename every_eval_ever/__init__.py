"""every_eval_ever public package API."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ['dedup', 'eval_types', 'instance_level_types', 'validation_core']


def __getattr__(name: str) -> Any:
    if name in {
        'dedup',
        'eval_types',
        'instance_level_types',
        'validation_core',
    }:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
