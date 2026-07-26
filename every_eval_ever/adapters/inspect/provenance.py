"""Reviewed model and execution provenance for Inspect exports."""

from __future__ import annotations

from dataclasses import dataclass

OPEN_MODEL_IDS = frozenset(
    {
        'deepseek/deepseek-chat',
        'deepseek/deepseek-r1',
        'meta-llama/llama-3.2-90b-vision-instruct',
        'meta-llama/llama-3.3-70b-instruct',
        'mistralai/codestral-2501',
        'qwen/qwen-2.5-coder-32b-instruct',
        'qwen/qwen-2.5-vl-72b-instruct',
        'qwen/qwen2.5-3b-instruct',
    }
)
MUTABLE_UNKNOWN_MODEL_IDS = frozenset(
    {
        'mistral/mistral-large-latest',
        'mistral/mistral-small-latest',
    }
)
LOCAL_ENGINES = frozenset(
    {'hf', 'llama-cpp-python', 'llamacpp', 'ollama', 'sglang', 'vllm'}
)


@dataclass(frozen=True)
class InspectProvenance:
    model_id: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _canonical_model_id(model_id: str) -> str:
    normalized = model_id.strip().casefold()
    duplicate_prefix = 'anthropic/anthropic/'
    if normalized.startswith(duplicate_prefix):
        return f'anthropic/{model_id.strip()[len(duplicate_prefix) :]}'
    return model_id.strip()


def _closed_api_family(normalized: str) -> bool:
    developer, separator, leaf = normalized.partition('/')
    if not separator:
        return False
    return (
        (developer == 'anthropic' and leaf.startswith('claude-'))
        or (developer == 'google' and leaf.startswith('gemini-'))
        or (
            developer == 'openai'
            and leaf.startswith(('gpt-', 'o1-', 'o3-', 'o4-'))
            and not leaf.startswith('gpt-oss-')
        )
        or (developer in {'grok', 'xai'} and leaf.startswith('grok-'))
    )


def inspect_provenance(
    model_id: str,
    inference_platform: str | None,
    inference_engine_name: str | None,
) -> InspectProvenance:
    canonical_id = _canonical_model_id(model_id)
    normalized = canonical_id.casefold()
    platform = (inference_platform or '').strip().casefold()
    engine = (inference_engine_name or '').strip().casefold()
    if platform in {'none', 'unknown'}:
        platform = ''
    if engine in {'none', 'unknown'}:
        engine = ''
    if platform and engine:
        raise ValueError(
            f'Inspect execution cannot be both provider-hosted and local: {model_id!r}'
        )
    if engine and engine not in LOCAL_ENGINES:
        raise ValueError(f'unreviewed Inspect inference engine: {engine!r}')

    if normalized in OPEN_MODEL_IDS:
        availability = 'open_weights'
    elif normalized in MUTABLE_UNKNOWN_MODEL_IDS:
        availability = 'unknown'
    elif _closed_api_family(normalized):
        availability = 'closed_weights'
    elif engine:
        # Local execution is proven, but a generic engine path alone does not
        # prove that the model's weights are publicly redistributable.
        availability = 'unknown'
    else:
        raise ValueError(f'unreviewed Inspect model id: {model_id!r}')

    if platform:
        deployment = 'externally_managed'
    elif engine:
        deployment = 'self_deployed'
    elif availability == 'closed_weights':
        deployment = 'externally_managed'
        platform = canonical_id.partition('/')[0].casefold()
    else:
        deployment = 'unknown'

    return InspectProvenance(
        model_id=canonical_id,
        deployment_type=deployment,
        model_availability=availability,
        inference_platform=platform or 'unknown',
        inference_engine_name=engine or 'unknown',
        inference_engine_version='unknown',
    )


__all__ = ['InspectProvenance', 'inspect_provenance']
