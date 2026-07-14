"""Reviewed model provenance for Mercor evaluation exports."""

from __future__ import annotations

import re
from dataclasses import dataclass

OPEN_MODEL_IDS = frozenset(
    {
        'minimax/minimax 2 5',
        'moonshot/kimi k2 thinking',
        'moonshot/kimi k2 5',
        'openai/gpt oss 120b',
        'zhipu/glm 4 6',
        'zhipu/glm 4 7',
    }
)
UNKNOWN_MODEL_IDS = frozenset({'applied-compute/applied compute small'})


@dataclass(frozen=True)
class MercorProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = 'unknown'
    inference_engine_version: str = 'unknown'


def _identity_key(model_id: str) -> tuple[str, str, str]:
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid Mercor model id: {model_id!r}')
    # Mercor's historical exports use display names (``GPT 5.2``), while the
    # live adapter emits path-safe slugs (``gpt-5.2``). Punctuation is not
    # semantic model identity here, so compare a delimiter-neutral exact key.
    leaf_key = re.sub(r'[^a-z0-9]+', ' ', leaf).strip()
    return developer, leaf_key, f'{developer}/{leaf_key}'


def mercor_provenance(
    model_id: str, source_platform: str = 'unknown'
) -> MercorProvenance:
    developer, leaf, normalized = _identity_key(model_id)
    platform = source_platform.strip().casefold() or 'unknown'
    if normalized in OPEN_MODEL_IDS:
        deployment = (
            'externally_managed' if platform != 'unknown' else 'unknown'
        )
        return MercorProvenance(deployment, 'open_weights', platform)
    if normalized in UNKNOWN_MODEL_IDS:
        return MercorProvenance('unknown', 'unknown', platform)
    if (
        (developer == 'anthropic' and leaf.startswith(('opus ', 'sonnet ')))
        or (developer == 'google' and leaf.startswith('gemini '))
        or (
            developer == 'openai'
            and (
                leaf.startswith('gpt ')
                or (leaf.startswith('o') and len(leaf) > 1 and leaf[1].isdigit())
            )
        )
        or (developer == 'xai' and leaf.startswith('grok '))
    ):
        return MercorProvenance(
            'externally_managed',
            'closed_weights',
            platform if platform != 'unknown' else developer,
        )
    raise ValueError(f'unreviewed Mercor model id: {model_id!r}')


__all__ = ['MercorProvenance', 'mercor_provenance']
