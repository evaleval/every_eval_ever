"""Strict JSON parsing shared by validation and duplicate detection."""

from __future__ import annotations

import json
from typing import Any


class StrictJSONError(ValueError):
    """Raised for JSON extensions or ambiguous objects we do not accept."""


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f'non-finite JSON number {value!r} is not allowed')


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f'duplicate JSON object key {key!r}')
        result[key] = value
    return result


def strict_json_loads(content: str | bytes) -> Any:
    """Parse standards-compliant, unambiguous JSON."""
    return json.loads(
        content,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
