"""Source-backed provenance for SWE-bench Verified submissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

SWE_BENCH_VERIFIED_METHOD_URL = 'https://github.com/swe-bench/experiments'


@dataclass(frozen=True)
class SWEBenchVerifiedProvenance:
    developer: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _source_boolean(value: Any) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == 'true':
            return True
        if normalized == 'false':
            return False
    raise ValueError('open_source_model must be true, false, or blank')


def swe_bench_verified_provenance(
    developer: str | None,
    open_source_model: Any,
) -> SWEBenchVerifiedProvenance:
    """Use the submission's explicit model-availability declaration."""
    normalized_developer = (developer or '').strip().casefold() or UNKNOWN
    source_open = _source_boolean(open_source_model)
    if source_open is True:
        return SWEBenchVerifiedProvenance(
            normalized_developer,
            UNKNOWN,
            OPEN_WEIGHTS,
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
        )
    if source_open is False:
        platform = (
            normalized_developer if normalized_developer != UNKNOWN else UNKNOWN
        )
        return SWEBenchVerifiedProvenance(
            normalized_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            platform,
            UNKNOWN,
            UNKNOWN,
        )
    return SWEBenchVerifiedProvenance(
        normalized_developer,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )


__all__ = [
    'SWE_BENCH_VERIFIED_METHOD_URL',
    'SWEBenchVerifiedProvenance',
    'swe_bench_verified_provenance',
]
