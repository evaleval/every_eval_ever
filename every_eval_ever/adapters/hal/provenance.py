"""Reviewed model provenance for Holistic Agent Leaderboard exports."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

OPEN_MODEL_IDS = frozenset(
    {
        'deepseek/deepseek-r1',
        'deepseek/deepseek-v3',
        'deepseek/deepseek-v3.1',
        'openai/gpt-oss-120b',
    }
)


@dataclass(frozen=True)
class HalProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = UNKNOWN
    inference_engine_version: str = UNKNOWN


def _closed_platform(model_id: str) -> str | None:
    developer, separator, leaf = model_id.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid HAL model id: {model_id!r}')
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


def hal_provenance(model_id: str) -> HalProvenance:
    """Classify a HAL model while preserving undocumented open deployment."""
    normalized = model_id.strip().casefold()
    if normalized in OPEN_MODEL_IDS:
        return HalProvenance(UNKNOWN, OPEN_WEIGHTS, UNKNOWN)
    platform = _closed_platform(normalized)
    if platform is not None:
        return HalProvenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, platform)
    raise ValueError(f'unreviewed HAL model id: {model_id!r}')


__all__ = ['HalProvenance', 'hal_provenance']
