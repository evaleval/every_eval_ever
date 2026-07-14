"""Reviewed model provenance for the Scale SEAL HLE leaderboard."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

OPEN_MODEL_IDS = frozenset(
    {
        'meta/llama-4-maverick',
        'moonshotai/kimi-k2',
        'moonshotai/kimi-k2.5',
        'zhipu-ai/glm-4p5',
        'zhipu-ai/glm-4p5-air',
    }
)

EXACT_CLOSED_MODEL_IDS = frozenset(
    {
        'meta/muse-spark',
        'mistralai/mistral-medium-3',
    }
)


@dataclass(frozen=True)
class HleProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = UNKNOWN
    inference_engine_version: str = UNKNOWN


def _closed_family_platform(model_id: str) -> str | None:
    developer, separator, leaf = model_id.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid HLE model id: {model_id!r}')
    if developer == 'amazon' and leaf.startswith('nova-'):
        return 'amazon'
    if developer == 'anthropic' and leaf.startswith('claude-'):
        return 'anthropic'
    if developer == 'google' and leaf.startswith('gemini-'):
        return 'google'
    if developer == 'openai' and (
        leaf.startswith('gpt-')
        or (leaf.startswith('o') and len(leaf) > 1 and leaf[1].isdigit())
    ):
        return 'openai'
    return None


def hle_provenance(model_id: str) -> HleProvenance:
    """Classify a reviewed HLE identifier without guessing serving details."""
    normalized = model_id.strip().casefold()
    if normalized in OPEN_MODEL_IDS:
        return HleProvenance(UNKNOWN, OPEN_WEIGHTS, UNKNOWN)

    platform = _closed_family_platform(normalized)
    if normalized in EXACT_CLOSED_MODEL_IDS:
        platform = normalized.partition('/')[0]
    if platform is not None:
        return HleProvenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, platform)
    raise ValueError(f'unreviewed HLE model id: {model_id!r}')


__all__ = ['HleProvenance', 'hle_provenance']
