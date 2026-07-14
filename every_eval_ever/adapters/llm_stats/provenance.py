"""Source-backed execution and availability rules for LLM Stats models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

LLM_STATS_METHOD_URL = 'https://llm-stats.com/developer'


@dataclass(frozen=True)
class LLMStatsProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _explicit_open_source(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == 'true':
            return True
        if normalized == 'false':
            return False
    raise ValueError('is_open_source must be a boolean when provided')


def _closed_provider(developer: str, leaf: str) -> str | None:
    if developer == 'anthropic' and leaf.startswith('claude-'):
        return 'anthropic'
    if developer == 'amazon' and leaf.startswith('nova-'):
        return 'aws'
    if developer == 'baidu' and leaf.startswith('ernie-'):
        return 'baidu'
    if developer == 'bytedance' and leaf.startswith('seed-'):
        return 'bytedance'
    if developer == 'google' and leaf.startswith('gemini-'):
        return 'google'
    if developer == 'inceptionlabs' and leaf == 'mercury-2':
        return 'inceptionlabs'
    if developer == 'meta' and leaf.startswith('muse-'):
        return 'meta'
    if developer == 'openai' and not leaf.startswith('gpt-oss-'):
        if leaf.startswith(('gpt-', 'o1-', 'o3-', 'o4-')):
            return 'openai'
    if developer == 'xai' and leaf.startswith('grok-'):
        return 'xai'
    if developer == 'qwen' and leaf in {'qwen3-max', 'qwen3.6-plus'}:
        return 'alibaba'
    if developer == 'xiaomi' and leaf in {'mimo-v2-omni', 'mimo-v2-pro'}:
        return 'xiaomi'
    if developer == 'zai-org' and leaf == 'glm-5v-turbo':
        return 'zai'
    if developer == 'mistral' and leaf == 'magistral-medium':
        return 'mistral'
    if developer == 'deepseek' and leaf in {
        'deepseek-v4-flash-max',
        'deepseek-v4-pro-max',
    }:
        return 'deepseek'
    return None


def _reviewed_availability(developer: str, leaf: str) -> str:
    if developer == 'mistral' and leaf in {
        'mistral-large-latest',
        'mistral-small-latest',
    }:
        return UNKNOWN
    if developer == 'moonshotai' and leaf == 'kimi-k1.5':
        return UNKNOWN

    open_family_rules = (
        ('ai21', ('jamba-',)),
        ('cohere', ('command-r-',)),
        ('deepseek', ('deepseek-',)),
        ('google', ('gemma-', 'medgemma-')),
        ('ibm', ('granite-',)),
        ('lg', ('k-exaone-',)),
        ('meituan', ('longcat-',)),
        ('meta', ('llama-',)),
        ('microsoft', ('phi-',)),
        ('minimax', ('minimax-',)),
        ('mistral', ('codestral-', 'ministral-', 'mistral-', 'pixtral-')),
        ('moonshotai', ('kimi-k2',)),
        ('nous-research', ('hermes-',)),
        ('nvidia', ('llama-', 'nemotron-', 'nvidia-nemotron-')),
        ('openbmb', ('minicpm-',)),
        ('qwen', ('qwen', 'qvq-', 'qwq-')),
        ('sarvamai', ('sarvam-',)),
        ('stepfun', ('step-', 'step3-')),
        ('xiaomi', ('mimo-v2-flash',)),
        ('zai-org', ('glm-',)),
    )
    if developer == 'openai' and leaf.startswith('gpt-oss-'):
        return OPEN_WEIGHTS
    if any(
        developer == expected and leaf.startswith(prefixes)
        for expected, prefixes in open_family_rules
    ):
        return OPEN_WEIGHTS
    return UNKNOWN


def llm_stats_provenance(
    model_id: str,
    is_open_source: Any = None,
) -> LLMStatsProvenance:
    """Classify a model without treating score attribution as deployment."""
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        return LLMStatsProvenance(
            UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN
        )

    explicit = _explicit_open_source(is_open_source)
    if explicit is not None:
        availability = OPEN_WEIGHTS if explicit else CLOSED_WEIGHTS
        deployment = EXTERNALLY_MANAGED if not explicit else UNKNOWN
        platform = developer if not explicit else UNKNOWN
        return LLMStatsProvenance(
            deployment, availability, platform, UNKNOWN, UNKNOWN
        )

    provider = _closed_provider(developer, leaf)
    if provider is not None:
        return LLMStatsProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            provider,
            UNKNOWN,
            UNKNOWN,
        )

    return LLMStatsProvenance(
        UNKNOWN,
        _reviewed_availability(developer, leaf),
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )


__all__ = [
    'LLM_STATS_METHOD_URL',
    'LLMStatsProvenance',
    'llm_stats_provenance',
]
