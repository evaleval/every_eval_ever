"""Reviewed model provenance for Exgentic leaderboard exports."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_PLATFORMS = {
    'anthropic/claude-opus-4-5': 'anthropic',
    'google/gemini-3-pro-preview': 'google',
    'openai/gpt-5.2-2025-12-11': 'openai',
}


@dataclass(frozen=True)
class ExgenticProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = 'unknown'
    inference_engine_version: str = 'unknown'


def exgentic_provenance(model_id: str) -> ExgenticProvenance:
    normalized = model_id.strip().casefold()
    try:
        platform = MODEL_PLATFORMS[normalized]
    except KeyError as exc:
        raise ValueError(f'unreviewed Exgentic model id: {model_id!r}') from exc
    return ExgenticProvenance(
        deployment_type='externally_managed',
        model_availability='closed_weights',
        inference_platform=platform,
    )


__all__ = ['ExgenticProvenance', 'exgentic_provenance']
