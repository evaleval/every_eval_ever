"""Reviewed model provenance for CocoaBench aggregate releases."""

from __future__ import annotations

from dataclasses import dataclass

OPEN_MODEL_IDS = frozenset(
    {
        'moonshotai/kimi-k2.5',
        'qwen/qwen3.5-397b-a17b',
    }
)
CLOSED_MODEL_IDS = frozenset(
    {
        'anthropic/claude-code',
        'anthropic/claude-sonnet-4.6-high',
        'google/gemini-3.1-pro',
        'google/gemini-flash-3.0',
        'openai/chatgpt-agent',
        'openai/codex',
        'openai/deep-research',
        'openai/gpt-5.4-high',
    }
)
ALIASES = {
    'qwen/qwen3.5-397b-a13b': 'qwen/qwen3.5-397b-a17b',
}


@dataclass(frozen=True)
class CocoaBenchProvenance:
    canonical_model_id: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = 'unknown'
    inference_engine_version: str = 'unknown'


def cocoabench_provenance(model_id: str) -> CocoaBenchProvenance:
    normalized = model_id.strip().casefold()
    canonical = ALIASES.get(normalized, normalized)
    if canonical in OPEN_MODEL_IDS:
        availability = 'open_weights'
    elif canonical in CLOSED_MODEL_IDS:
        availability = 'closed_weights'
    else:
        raise ValueError(f'unreviewed CocoaBench model id: {model_id!r}')
    return CocoaBenchProvenance(
        canonical_model_id=canonical,
        deployment_type='externally_managed',
        model_availability=availability,
        inference_platform=canonical.partition('/')[0],
    )


__all__ = ['CocoaBenchProvenance', 'cocoabench_provenance']
