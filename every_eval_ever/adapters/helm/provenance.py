"""Reviewed model provenance shared by the HELM leaderboard adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

HELM_METHOD_URL = 'https://crfm.stanford.edu/helm/'

MODEL_AVAILABILITY_SOURCES = {
    'ai21_jamba': 'https://huggingface.co/ai21labs/models',
    'cohere_command_r': 'https://huggingface.co/CohereLabs/models',
    'mistral': 'https://docs.mistral.ai/models/overview',
    'writer': 'https://huggingface.co/Writer/models',
    'yalm': 'https://huggingface.co/yandex/yalm-100b',
}

_MISTRAL_OPEN = frozenset(
    {
        'mistral-v0.1-7b',
        'mistral-7b-instruct-v0.1',
        'mistral-7b-instruct-v0.3',
        'mistral-7b-v0.1',
        'mistral-small-2501',
        'mistral-small-2503',
        'mixtral-8x22b',
        'mixtral-8x22b-instruct-v0.1',
        'mixtral-8x7b-32kseqlen',
        'mixtral-8x7b-instruct-v0.1',
        'open-mistral-nemo-2407',
    }
)
_MISTRAL_CLOSED = frozenset(
    {
        'mistral-large-2402',
        'mistral-large-2407',
        'mistral-large-2411',
        'mistral-medium-2312',
        'mistral-small-2402',
    }
)
_WRITER_OPEN = frozenset(
    {'instructpalmyra-30b', 'palmyra-fin', 'palmyra-med'}
)
_WRITER_CLOSED = frozenset(
    {
        'palmyra-x-004',
        'palmyra-x-v2',
        'palmyra-x-v3',
        'palmyra-x5',
    }
)


@dataclass(frozen=True)
class HelmProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _provenance(
    deployment: str, availability: str, platform: str = UNKNOWN
) -> HelmProvenance:
    return HelmProvenance(deployment, availability, platform, UNKNOWN, UNKNOWN)


def helm_provenance(model_id: str) -> HelmProvenance:
    """Classify a reviewed HELM identifier; reject unreviewed additions."""
    normalized = model_id.strip().casefold()
    if normalized == 'anthropic-lm-v4-s3-52b':
        return _provenance(UNKNOWN, UNKNOWN)
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'unreviewed HELM model id: {model_id!r}')

    if developer in {'alephalpha', 'aleph-alpha', 'amazon', 'anthropic'}:
        return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == '01-ai':
        if leaf == 'yi-large-preview':
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
        if leaf.startswith('yi-'):
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
    if developer == 'ai21':
        if leaf.startswith('jamba'):
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf.startswith(('j1-', 'j2-', 'jurassic-')):
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'cohere':
        if leaf in {'command-r', 'command-r-plus'}:
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf.startswith(('cohere-', 'command')):
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'google':
        if leaf.casefold().startswith(('t5-', 'ul2-', 'gemma-')):
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf.casefold().startswith(
            ('gemini-', 'text-bison', 'text-unicorn', 'palmyra-x')
        ):
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'microsoft':
        if leaf.casefold().startswith('phi-'):
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf.casefold().startswith('tnlg-v2-'):
            return _provenance(UNKNOWN, CLOSED_WEIGHTS)
    if developer == 'mistralai':
        if leaf in _MISTRAL_OPEN:
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf in _MISTRAL_CLOSED:
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'openai':
        if leaf.casefold().startswith(('gpt-j-', 'gpt-neox-', 'gpt-oss-')):
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf.casefold().startswith(
            (
                'ada-',
                'babbage-',
                'curie-',
                'davinci-',
                'gpt-3',
                'gpt-4',
                'gpt-5',
                'o1-',
                'o3-',
                'o4-',
                'text-',
            )
        ):
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'writer':
        if leaf in _WRITER_OPEN:
            return _provenance(UNKNOWN, OPEN_WEIGHTS)
        if leaf in _WRITER_CLOSED:
            return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'upstage' and leaf == 'solar-pro-241126':
        return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)
    if developer == 'xai' and leaf.startswith('grok-'):
        return _provenance(EXTERNALLY_MANAGED, CLOSED_WEIGHTS, developer)

    open_developers = {
        'allenai',
        'bigscience',
        'databricks',
        'deepseek-ai',
        'eleutherai',
        'ibm',
        'lmsys',
        'marin-community',
        'meta',
        'moonshotai',
        'mosaicml',
        'qwen',
        'snowflake',
        'stanford',
        'tiiuae',
        'together',
        'yandex',
        'zai-org',
        'zhipu-ai',
    }
    if developer in open_developers:
        if developer in {'meta', 'qwen'} and (
            leaf.endswith('-turbo') or leaf.endswith('-tput')
        ):
            return _provenance(
                EXTERNALLY_MANAGED, OPEN_WEIGHTS, 'together_ai'
            )
        return _provenance(UNKNOWN, OPEN_WEIGHTS)

    raise ValueError(f'unreviewed HELM model id: {model_id!r}')


def helm_metric_identity(
    collection: str, evaluation_name: str, metric_name: str | None
) -> tuple[str, str]:
    """Return a deterministic metric ID and a guaranteed non-blank name."""
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError('HELM collection must be non-blank text')
    if not isinstance(evaluation_name, str) or not evaluation_name.strip():
        raise ValueError('HELM evaluation name must be non-blank text')
    resolved_name = (
        metric_name.strip()
        if isinstance(metric_name, str) and metric_name.strip()
        else evaluation_name.strip()
    )

    def slug(value: str) -> str:
        normalized = re.sub(r'[^a-z0-9]+', '_', value.casefold()).strip('_')
        if not normalized:
            raise ValueError(f'HELM metric component has no identity: {value!r}')
        return normalized

    metric_id = '.'.join(
        (slug(collection), slug(evaluation_name), slug(resolved_name))
    )
    return metric_id, resolved_name
