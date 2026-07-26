"""Source-backed execution and availability rules for MMLU-Pro models."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

MMLU_PRO_METHOD_URL = 'https://github.com/TIGER-AI-Lab/MMLU-Pro'

_OPEN_UNKNOWN_MODELS = frozenset(
    {
        'unknown/ernie-4.5-21b-a3b-base',
        'unknown/ernie-4.5-300b-a47b',
        'unknown/ernie-4.5-300b-a47b-base',
        'unknown/general-reasoner-14b',
        'unknown/general-reasoner-7b',
        'unknown/hunyuan-a13b',
        'unknown/hunyuan-large',
        'unknown/intern-s1',
        'unknown/internmath-20b-plus',
        'unknown/internmath-7b-plus',
        'unknown/k2.5-1t-a32b',
        'unknown/llada',
        'unknown/llemma-7b',
        'unknown/longcat-flash-chat',
        'unknown/magnum-72b-v1',
        'unknown/mathstral-7b-v0.1',
        'unknown/mimo-7b-base',
        'unknown/mimo-7b-rl',
        'unknown/ministral-8b-instruct-2410',
        'unknown/neo-7b',
        'unknown/neo-7b-instruct',
        'unknown/openchat-3.5-8b',
        'unknown/qwq-32b',
        'unknown/qwq-32b-preview',
        'unknown/reka-3',
        'unknown/seed-oss-36b-instruct',
        'unknown/skythought-t1',
        'unknown/zephyr-7b-beta',
    }
)

_CLOSED_UNKNOWN_PROVIDERS = {
    'unknown/doubao-1.5-pro': 'bytedance',
    'unknown/hunyuanturbos': 'tencent',
    'unknown/hunyuan-t1': 'tencent',
    'unknown/seed-thinking-v1.5': 'bytedance',
    'unknown/seed1.6-base': 'bytedance',
    'unknown/seed1.6-thinking': 'bytedance',
    'unknown/seed2.0-lite': 'bytedance',
    'unknown/seed2.0-mini': 'bytedance',
    'unknown/seed2.0-pro': 'bytedance',
}


@dataclass(frozen=True)
class MMLUProProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _provider_and_availability(model_id: str) -> tuple[str, str]:
    developer, _, leaf = model_id.partition('/')

    if developer == 'anthropic' and leaf.startswith('claude-'):
        return 'anthropic', CLOSED_WEIGHTS
    if developer == 'google' and leaf.startswith('gemini-'):
        return 'google', CLOSED_WEIGHTS
    if developer.startswith('gemini-'):
        return 'google', CLOSED_WEIGHTS
    if developer in {'xai', 'grok'} and leaf.startswith('grok'):
        return 'xai', CLOSED_WEIGHTS
    if developer == 'mistral' and leaf.endswith('-latest'):
        return 'mistral', CLOSED_WEIGHTS
    if developer == 'openai':
        if leaf.startswith('seed1.6-'):
            return 'bytedance', CLOSED_WEIGHTS
        if leaf.startswith('gpt-oss-'):
            return UNKNOWN, OPEN_WEIGHTS
        if leaf.startswith(('gpt-', 'o1-', 'o3-', 'o4-')):
            return 'openai', CLOSED_WEIGHTS
    if developer == 'alibaba' and ('max' in leaf or 'turbo' in leaf):
        return 'alibaba', CLOSED_WEIGHTS
    if developer == '01-ai' and leaf in {'yi-large', 'yi-lightning'}:
        return '01-ai', CLOSED_WEIGHTS
    if model_id == 'meta/llama4-behemoth':
        return 'meta', CLOSED_WEIGHTS
    if model_id in _CLOSED_UNKNOWN_PROVIDERS:
        return _CLOSED_UNKNOWN_PROVIDERS[model_id], CLOSED_WEIGHTS

    open_family_rules = (
        ('01-ai', ('yi-',)),
        ('abacus-ai', ('llama',)),
        ('ai21', ('jamba-',)),
        ('alibaba', ('qwen',)),
        ('cohere', ('aya-', 'c4ai-command-', 'cohere-aya-')),
        ('deepseek', ('deepseek',)),
        ('google', ('gemma-',)),
        ('ibm', ('granite-',)),
        ('lg-ai', ('exaone-',)),
        ('meta', ('llama-', 'llama3-', 'llama4-', 'higgs-', 'reflection-')),
        ('meta-llama', ('llama-',)),
        ('microsoft', ('phi-', 'phi3-')),
        ('mistralai', ('mistral-', 'mixtral-')),
        ('moonshotai', ('kimi-',)),
        ('nexusflow', ('athene-',)),
        ('qwen', ('qwen-',)),
        ('shanghai-ai-lab', ('internlm-', 'internlm3-')),
        ('tiger-lab', ('mammoth-', 'mammoth2-')),
        ('wizardlm', ('wizardlm-',)),
        ('zhipu-ai', ('glm-',)),
    )
    if any(
        developer == expected and leaf.startswith(prefixes)
        for expected, prefixes in open_family_rules
    ):
        return UNKNOWN, OPEN_WEIGHTS

    if developer.casefold() == 'newenai' and leaf.startswith('newenai-phi4-'):
        return UNKNOWN, OPEN_WEIGHTS
    if developer.casefold().startswith('seed-oss-36b-base(w'):
        return UNKNOWN, OPEN_WEIGHTS
    if model_id in _OPEN_UNKNOWN_MODELS:
        return UNKNOWN, OPEN_WEIGHTS
    if model_id.startswith('unknown/minimax-'):
        return UNKNOWN, OPEN_WEIGHTS
    if model_id.startswith('unknown/nemotron-'):
        return UNKNOWN, OPEN_WEIGHTS
    if model_id.startswith('unknown/smollm'):
        return UNKNOWN, OPEN_WEIGHTS

    return UNKNOWN, UNKNOWN


def mmlu_pro_provenance(
    model_id: str,
    inference_platform: str | None = None,
) -> MMLUProProvenance:
    """Classify one model while preserving source-published provider data."""
    normalized = model_id.strip().casefold()
    platform = (inference_platform or '').strip().casefold() or UNKNOWN
    if normalized.count('/') != 1 or normalized.endswith('/'):
        return MMLUProProvenance(UNKNOWN, UNKNOWN, platform, UNKNOWN, UNKNOWN)

    provider, availability = _provider_and_availability(normalized)
    if platform != UNKNOWN:
        deployment = EXTERNALLY_MANAGED
    elif availability == CLOSED_WEIGHTS:
        deployment = EXTERNALLY_MANAGED
        platform = provider
    else:
        deployment = UNKNOWN

    return MMLUProProvenance(
        deployment, availability, platform, UNKNOWN, UNKNOWN
    )


__all__ = [
    'MMLU_PRO_METHOD_URL',
    'MMLUProProvenance',
    'mmlu_pro_provenance',
]
