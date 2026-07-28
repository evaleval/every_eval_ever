"""Strict JSON parsing shared by aggregate and instance validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class StrictJSONError(ValueError):
    """Raised for JSON extensions or ambiguous objects we do not accept."""


class StrictJSONReader(Protocol):
    """Backend contract for strict, hook-capable JSON readers."""

    def loads(
        self,
        content: str | bytes,
        *,
        parse_constant: Callable[[str], None],
        object_pairs_hook: Callable[[list[tuple[str, Any]]], dict[str, Any]],
    ) -> Any: ...


@dataclass(frozen=True)
class StdlibJSONReader:
    """Standard-library implementation of the strict reader contract."""

    def loads(
        self,
        content: str | bytes,
        *,
        parse_constant: Callable[[str], None],
        object_pairs_hook: Callable[[list[tuple[str, Any]]], dict[str, Any]],
    ) -> Any:
        return json.loads(
            content,
            parse_constant=parse_constant,
            object_pairs_hook=object_pairs_hook,
        )


STDLIB_JSON_READER = StdlibJSONReader()


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f'non-finite JSON number {value!r} is not allowed')


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f'duplicate JSON object key {key!r}')
        result[key] = value
    return result


def strict_json_loads(
    content: str | bytes, *, reader: StrictJSONReader = STDLIB_JSON_READER
) -> Any:
    """Parse unambiguous JSON with an explicit strict reader backend.

    Alternative readers are opt-in and must implement the hook contract. A
    missing or incompatible accelerated reader raises normally; this function
    never silently falls back to a different parser.
    """
    return reader.loads(
        content,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
