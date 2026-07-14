"""Reviewed source aliases and model provenance for SciArena."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

SCI_ARENA_MODEL_DEVELOPERS = {
    'o3': 'openai',
    'Claude-4.1-Opus': 'anthropic',
    'GPT-5': 'openai',
    'Gemini-3-Pro-Preview': 'google',
    'GPT-5.1': 'openai',
    'Claude-4-Opus': 'anthropic',
    'GPT-5-mini': 'openai',
    'Gemini-2.5-Pro': 'google',
    'Grok-4': 'xai',
    'Deepseek-R1-0528': 'deepseek',
    'GPT-OSS-120B': 'openai',
    'Qwen3-235B-A22B-Thinking-2507': 'qwen',
    'o4-mini': 'openai',
    'Claude-4-Sonnet': 'anthropic',
    'Qwen3-235B-A22B-2507': 'qwen',
    'GPT-4.1': 'openai',
    'GPT-4.1-mini': 'openai',
    'Qwen3-30B-A3B-Instruct-2507': 'qwen',
    'Gemini-2.5-Pro-Preview': 'google',
    'GLM-4.5': 'zhipu',
    'Deepseek-R1': 'deepseek',
    'Deepseek-V3': 'deepseek',
    'Qwen3-235B-A22B': 'qwen',
    'Kimi-K2': 'moonshotai',
    'Grok-3': 'xai',
    'QwQ-32B': 'qwen',
    'Claude-3-7-Sonnet': 'anthropic',
    'Gemini-2.5-Flash': 'google',
    'Olmo-3.1-32B-Instruct': 'allenai',
    'Qwen3-32B': 'qwen',
    'Gemini-2.5-Flash-Preview': 'google',
    'GPT-OSS-20B': 'openai',
    'GPT-5-nano': 'openai',
    'Mistral-Small-3.1': 'mistralai',
    'Mistral-Medium-3': 'mistralai',
    'Minimax-M1': 'minimax',
    'Llama-4-Maverick': 'meta',
    'Llama-4-Scout': 'meta',
}

OPEN_MODEL_IDS = frozenset(
    {
        'allenai/olmo-3.1-32b-instruct',
        'deepseek/deepseek-r1',
        'deepseek/deepseek-r1-0528',
        'deepseek/deepseek-v3',
        'meta/llama-4-maverick',
        'meta/llama-4-scout',
        'minimax/minimax-m1',
        'mistralai/mistral-small-3.1',
        'moonshotai/kimi-k2',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        'qwen/qwen3-235b-a22b',
        'qwen/qwen3-235b-a22b-2507',
        'qwen/qwen3-235b-a22b-thinking-2507',
        'qwen/qwen3-30b-a3b-instruct-2507',
        'qwen/qwen3-32b',
        'qwen/qwq-32b',
        'zhipu/glm-4.5',
    }
)

EXACT_CLOSED_MODEL_IDS = frozenset({'mistralai/mistral-medium-3'})


@dataclass(frozen=True)
class SciArenaProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str = UNKNOWN
    inference_engine_version: str = UNKNOWN


def sci_arena_developer(raw_model_id: str) -> str:
    try:
        return SCI_ARENA_MODEL_DEVELOPERS[raw_model_id]
    except KeyError as exc:
        raise ValueError(
            f'unreviewed SciArena source model: {raw_model_id!r}'
        ) from exc


def _closed_platform(model_id: str) -> str | None:
    developer, separator, leaf = model_id.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid SciArena model id: {model_id!r}')
    if developer == 'anthropic' and leaf.startswith('claude-'):
        return 'anthropic'
    if developer == 'google' and leaf.startswith('gemini-'):
        return 'google'
    if developer == 'openai' and (
        leaf.startswith('gpt-')
        or (leaf.startswith('o') and len(leaf) > 1 and leaf[1].isdigit())
    ):
        return 'openai'
    if developer == 'xai' and leaf.startswith('grok-'):
        return 'xai'
    return None


def sci_arena_provenance(model_id: str) -> SciArenaProvenance:
    normalized = model_id.strip().casefold()
    if normalized in OPEN_MODEL_IDS:
        return SciArenaProvenance(UNKNOWN, OPEN_WEIGHTS, UNKNOWN)
    platform = _closed_platform(normalized)
    if normalized in EXACT_CLOSED_MODEL_IDS:
        platform = normalized.partition('/')[0]
    if platform is not None:
        return SciArenaProvenance(
            EXTERNALLY_MANAGED, CLOSED_WEIGHTS, platform
        )
    raise ValueError(f'unreviewed SciArena model id: {model_id!r}')


__all__ = [
    'SCI_ARENA_MODEL_DEVELOPERS',
    'SciArenaProvenance',
    'sci_arena_developer',
    'sci_arena_provenance',
]
