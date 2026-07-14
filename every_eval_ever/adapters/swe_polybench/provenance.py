"""Reviewed submission provenance for SWE-PolyBench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwePolybenchProvenance:
    developer: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = 'unknown'
    inference_engine_version: str = 'unknown'


PROVENANCE = {
    'openai/gpt-5': SwePolybenchProvenance(
        developer='openai',
        deployment_type='externally_managed',
        model_availability='closed_weights',
        inference_platform='openai',
    ),
    # These source rows identify agent products, not their underlying model.
    # Atlassian documents Rovo Dev as a hosted product using third-party LLMs.
    'unknown/atlassian-rovo-dev': SwePolybenchProvenance(
        developer='atlassian',
        deployment_type='externally_managed',
        model_availability='unknown',
        inference_platform='atlassian',
    ),
    'unknown/iswe_agent': SwePolybenchProvenance(
        developer='amazon',
        deployment_type='unknown',
        model_availability='unknown',
        inference_platform='unknown',
    ),
}


def swe_polybench_provenance(model_id: str) -> SwePolybenchProvenance:
    key = model_id.strip().casefold()
    try:
        return PROVENANCE[key]
    except KeyError as exc:
        raise ValueError(
            f'unreviewed SWE-PolyBench model or agent id: {model_id!r}'
        ) from exc


__all__ = ['SwePolybenchProvenance', 'swe_polybench_provenance']
