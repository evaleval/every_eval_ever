"""Source-backed execution and availability rules for RewardBench models."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
SELF_DEPLOYED = 'self_deployed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

REWARDBENCH_METHOD_URL = 'https://github.com/allenai/reward-bench'

# The official RewardBench generative implementation routes these identifiers
# through provider APIs. The three Turbo checkpoints and Gemma route through
# Together while retaining open model weights.
TOGETHER_MODEL_IDS = frozenset(
    {
        'google/gemma-2-27b-it',
        'meta-llama/meta-llama-3.1-405b-instruct-turbo',
        'meta-llama/meta-llama-3.1-70b-instruct-turbo',
        'meta-llama/meta-llama-3.1-8b-instruct-turbo',
    }
)

KNOWN_PLACEHOLDER_MODEL_IDS = frozenset({'my_model/'})
_NON_GENERATIVE_MODEL_TYPES = frozenset(
    {'custom classifier', 'dpo', 'seq. classifier'}
)
_COHERE_PRIVATE_JUDGES = frozenset(
    {'cohere march 2024', 'cohere may 2024'}
)


@dataclass(frozen=True)
class RewardBenchProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def is_known_placeholder_model(model_id: str) -> bool:
    """Return whether RewardBench published a non-model placeholder row."""
    return model_id.strip().casefold() in KNOWN_PLACEHOLDER_MODEL_IDS


def _api_provider(model_id: str) -> str | None:
    normalized = model_id.strip().casefold()
    leaf = normalized.rsplit('/', 1)[-1]
    if normalized.startswith('openai/') or leaf.startswith(('gpt-', 'o1-')):
        return 'openai'
    if normalized.startswith('anthropic/') or leaf.startswith('claude-'):
        return 'anthropic'
    if normalized.startswith('google/') and leaf.startswith('gemini-'):
        return 'google'
    if normalized == 'gemini-1.5-flash-8b':
        return 'google'
    if normalized.startswith('poll/'):
        return 'multiple_api_providers'
    return None


def _is_unpublished_or_ambiguous(model_id: str) -> bool:
    normalized = model_id.strip().casefold()
    if normalized in _COHERE_PRIVATE_JUDGES:
        return True
    if '...' in normalized:
        return True
    if normalized.startswith('allenai/open_instruct_dev-'):
        return True
    if normalized.startswith('ai2/') and normalized.endswith(
        ('.json', '.jsonl')
    ):
        return True
    return normalized.count('/') != 1 or normalized.endswith('/')


def rewardbench_provenance(
    model_id: str,
    model_type: str | None,
) -> RewardBenchProvenance:
    """Classify one source model without guessing an undocumented endpoint."""
    normalized = model_id.strip().casefold()
    normalized_type = (model_type or '').strip().casefold()
    provider = _api_provider(normalized)

    if provider is not None:
        return RewardBenchProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            provider,
            UNKNOWN,
            UNKNOWN,
        )

    if normalized in TOGETHER_MODEL_IDS:
        return RewardBenchProvenance(
            EXTERNALLY_MANAGED,
            OPEN_WEIGHTS,
            'together',
            UNKNOWN,
            UNKNOWN,
        )

    availability = (
        UNKNOWN
        if _is_unpublished_or_ambiguous(normalized)
        else OPEN_WEIGHTS
    )

    # RewardBench documents local Transformers inference for classifier and
    # DPO models. Private Cohere submissions are not reproducible on Ai2's
    # infrastructure, so their deployment remains unknown.
    if normalized_type in _NON_GENERATIVE_MODEL_TYPES:
        if normalized in _COHERE_PRIVATE_JUDGES:
            return RewardBenchProvenance(
                UNKNOWN, availability, UNKNOWN, UNKNOWN, UNKNOWN
            )
        return RewardBenchProvenance(
            SELF_DEPLOYED,
            availability,
            UNKNOWN,
            'transformers',
            UNKNOWN,
        )

    # The frozen leaderboard does not retain whether an arbitrary open
    # generative model used local vLLM or an optional provider endpoint.
    return RewardBenchProvenance(
        UNKNOWN, availability, UNKNOWN, UNKNOWN, UNKNOWN
    )


__all__ = [
    'KNOWN_PLACEHOLDER_MODEL_IDS',
    'REWARDBENCH_METHOD_URL',
    'RewardBenchProvenance',
    'is_known_placeholder_model',
    'rewardbench_provenance',
]
