"""Shared, source-backed Artificial Analysis model provenance."""

from __future__ import annotations

from every_eval_ever.adapters.artificial_analysis.registry import (
    CLOSED_WEIGHTS,
    EXTERNALLY_MANAGED,
    OPEN_WEIGHTS,
    UNKNOWN_AVAILABILITY,
    UNKNOWN_DEPLOYMENT,
    is_api_only_closed_weight,
    is_verified_open_weight,
)
from every_eval_ever.adapters.artificial_analysis.registry import (
    INFERENCE_ENGINE_NAME as INFERENCE_ENGINE_NAME,
)
from every_eval_ever.adapters.artificial_analysis.registry import (
    INFERENCE_ENGINE_VERSION as INFERENCE_ENGINE_VERSION,
)
from every_eval_ever.adapters.artificial_analysis.registry import (
    INFERENCE_PLATFORM as INFERENCE_PLATFORM,
)
from every_eval_ever.adapters.artificial_analysis.registry import (
    MODEL_AVAILABILITY_SOURCES as MODEL_AVAILABILITY_SOURCES,
)


def _is_family(model_slug: str, family: str) -> bool:
    return model_slug == family or model_slug.startswith(f'{family}-')


def model_provenance(
    creator_slug: str | None, model_slug: str | None
) -> tuple[str, str]:
    """Return ``(deployment_type, model_availability)``.

    Classification is deliberately model-family-specific. An unmatched model,
    including another model from a recognized creator, remains unknown.
    """
    creator = creator_slug.strip().casefold() if creator_slug else ''
    model = model_slug.strip().casefold() if model_slug else ''

    if is_verified_open_weight(creator, model):
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS
    if is_api_only_closed_weight(creator, model):
        return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'openai':
        if any(
            _is_family(model, family)
            for family in ('gpt-oss-120b', 'gpt-oss-20b')
        ):
            return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS
        if any(
            _is_family(model, family)
            for family in (
                'gpt-3-5',
                'gpt-35',
                'gpt-4',
                'gpt-4o',
                'gpt-5',
                'o1',
                'o3',
                'o4',
            )
        ):
            return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'google':
        if any(
            _is_family(model, family)
            for family in ('gemma-3', 'gemma-3n', 'gemma-4')
        ):
            return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS
        if any(
            _is_family(model, family)
            for family in ('gemini-1', 'gemini-2', 'gemini-3', 'palm-2')
        ):
            return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'meta' and (
        model == 'llama-65b'
        or any(
            _is_family(model, family)
            for family in ('llama-2', 'llama-3', 'llama-4')
        )
    ):
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'xai':
        if model == 'grok-1':
            return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS
        if any(
            _is_family(model, family)
            for family in ('grok-2', 'grok-3', 'grok-4')
        ):
            return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'alibaba':
        if _is_family(model, 'qwen3-8b-instruct'):
            return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS
        if _is_family(model, 'qwen-turbo'):
            return EXTERNALLY_MANAGED, CLOSED_WEIGHTS
        if model == 'qwen-2-5-max' or model.startswith('qwen3-max'):
            return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'minimax' and model in {
        'minimax-m1-40k',
        'minimax-m1-80k',
        'minimax-m2',
        'minimax-m2-1',
        'minimax-m2-5',
        'minimax-m2-7',
    }:
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'cohere' and model in {
        'command-a',
        'command-r-plus-04-2024',
    }:
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'arcee' and model == 'trinity-large-thinking':
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'baidu' and model == 'ernie-5-0-thinking-preview':
        return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'azure' and model in {'phi-4', 'phi-4-mini'}:
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'nous-research' and model.startswith(
        ('hermes-', 'deephermes-')
    ):
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'ai2':
        if model.startswith(('olmo-', 'molmo')) or _is_family(
            model, 'tulu3-405b'
        ):
            return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'meta' and _is_family(model, 'muse-spark'):
        return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'bytedance_seed' and model == 'seed-oss-36b-instruct':
        return UNKNOWN_DEPLOYMENT, OPEN_WEIGHTS

    if creator == 'anthropic' and model.startswith('claude-'):
        return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    if creator == 'aws' and model.startswith('nova-'):
        return EXTERNALLY_MANAGED, CLOSED_WEIGHTS

    return UNKNOWN_DEPLOYMENT, UNKNOWN_AVAILABILITY
