"""Reviewed model aliases and execution provenance for MT-Bench."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
SELF_DEPLOYED = 'self_deployed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

MT_BENCH_DEVELOPER_OVERRIDES: dict[str, str] = {
    'alpaca-13b': 'stanford',
    'baize-v2-13b': 'project-baize',
    'chatglm-6b': 'thudm',
    'claude-instant-v1': 'anthropic',
    'claude-v1': 'anthropic',
    'dolly-v2-12b': 'databricks',
    'falcon-40b-instruct': 'tiiuae',
    'fastchat-t5-3b': 'lmsys',
    'gpt-3.5-turbo': 'openai',
    'gpt-4': 'openai',
    'gpt4all-13b-snoozy': 'nomic-ai',
    'guanaco-33b': 'timdettmers',
    'guanaco-65b': 'timdettmers',
    'h2ogpt-oasst-open-llama-13b': 'h2oai',
    'koala-13b': 'young-geng',
    'llama-13b': 'meta',
    'Llama-2-7b-chat': 'meta',
    'Llama-2-13b-chat': 'meta',
    'Llama-2-70b-chat': 'meta',
    'mpt-7b-chat': 'mosaicml',
    'mpt-30b-chat': 'mosaicml',
    'mpt-30b-instruct': 'mosaicml',
    'nous-hermes-13b': 'nousresearch',
    'oasst-sft-4-pythia-12b': 'openassistant',
    'oasst-sft-7-llama-30b': 'openassistant',
    'palm-2-chat-bison-001': 'google',
    'rwkv-4-raven-14b': 'rwkv',
    'stablelm-tuned-alpha-7b': 'stabilityai',
    'tulu-30b': 'allenai',
    'vicuna-7b-v1.3': 'lmsys',
    'vicuna-13b-v1.3': 'lmsys',
    'vicuna-33b-v1.3': 'lmsys',
    'wizardlm-13b': 'wizardlm',
    'wizardlm-30b': 'wizardlm',
}

CLOSED_MODEL_IDS = frozenset(
    {
        'anthropic/claude-instant-v1',
        'anthropic/claude-v1',
        'google/palm-2-chat-bison-001',
        'openai/gpt-3.5-turbo',
        'openai/gpt-4',
    }
)

OPEN_MODEL_IDS = frozenset(
    {
        f'{developer}/{raw_name}'.casefold()
        for raw_name, developer in MT_BENCH_DEVELOPER_OVERRIDES.items()
        if f'{developer}/{raw_name}'.casefold() not in CLOSED_MODEL_IDS
    }
)


@dataclass(frozen=True)
class MtBenchProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str = UNKNOWN


def mt_bench_provenance(model_id: str) -> MtBenchProvenance:
    normalized = model_id.strip().casefold()
    if normalized in OPEN_MODEL_IDS:
        return MtBenchProvenance(
            SELF_DEPLOYED, OPEN_WEIGHTS, UNKNOWN, 'fastchat'
        )
    if normalized in CLOSED_MODEL_IDS:
        return MtBenchProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            normalized.partition('/')[0],
            UNKNOWN,
        )
    raise ValueError(f'unreviewed MT-Bench model id: {model_id!r}')


__all__ = [
    'MT_BENCH_DEVELOPER_OVERRIDES',
    'MtBenchProvenance',
    'mt_bench_provenance',
]
