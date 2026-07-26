"""Source-backed execution and availability rules for ARC-AGI systems."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

ARC_AGI_METHOD_URL = 'https://arcprize.org/leaderboard'


@dataclass(frozen=True)
class ArcAgiProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def arc_agi_provenance(model_id: str) -> ArcAgiProvenance:
    """Classify one ARC Prize entry without treating solvers as weight models."""
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        return ArcAgiProvenance(UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN)

    routed_platform = None
    if 'openrouter' in leaf:
        routed_platform = 'openrouter'
    elif 'together' in leaf:
        routed_platform = 'together'
    elif 'bedrock' in leaf:
        routed_platform = 'aws'

    if developer in {'openai', 'anthropic', 'google', 'xai'}:
        platform = routed_platform or developer
        return ArcAgiProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            platform,
            UNKNOWN,
            UNKNOWN,
        )

    if developer == 'mistralai':
        availability = (
            OPEN_WEIGHTS
            if leaf.startswith('magistral-small-')
            else CLOSED_WEIGHTS
        )
    elif developer in {
        'deepseek',
        'meta',
        'minimax',
        'moonshotai',
        'qwen',
        'zhipu',
    }:
        availability = OPEN_WEIGHTS
    else:
        availability = UNKNOWN

    if routed_platform is not None:
        deployment = EXTERNALLY_MANAGED
        platform = routed_platform
    elif availability == CLOSED_WEIGHTS:
        deployment = EXTERNALLY_MANAGED
        platform = 'mistral'
    else:
        deployment = UNKNOWN
        platform = UNKNOWN
    return ArcAgiProvenance(
        deployment, availability, platform, UNKNOWN, UNKNOWN
    )


__all__ = ['ARC_AGI_METHOD_URL', 'ArcAgiProvenance', 'arc_agi_provenance']
