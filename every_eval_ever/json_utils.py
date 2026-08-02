"""Strict JSON parsing shared by aggregate and instance validation."""

from __future__ import annotations

import json
import math
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


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise StrictJSONError(
            f'non-finite JSON number {value!r} is not allowed'
        )
    return number


def strict_json_loads(content: str | bytes) -> Any:
    """Parse standards-compliant JSON and reject duplicate object keys."""
    return json.loads(
        content,
        parse_constant=_reject_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_keys,
    )
