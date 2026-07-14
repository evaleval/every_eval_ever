"""Source-backed model provenance and metric IDs for AlpacaEval."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

ALPACA_EVAL_METHOD_URL = 'https://github.com/tatsu-lab/alpaca_eval'

ALPACA_EVAL_METRICS = {
    'Win Rate': 'alpaca_eval.win_rate',
    'Length-Controlled Win Rate': 'alpaca_eval.lc_win_rate',
    'Discrete Win Rate': 'alpaca_eval.discrete_win_rate',
    'Average Response Length': 'alpaca_eval.avg_length',
}


@dataclass(frozen=True)
class AlpacaEvalProvenance:
    developer: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _closed_provider(developer: str, leaf: str) -> str | None:
    if developer == 'openai':
        return 'openai'
    if developer == 'anthropic' and not leaf.startswith('claude2-alpaca-'):
        return 'anthropic'
    if developer == 'google' and leaf.startswith(('gemini-', 'palm-', 'bard')):
        return 'google'
    if developer == '01-ai' and leaf == 'yi-large-preview':
        return '01-ai'
    if developer == 'mistralai' and leaf in {
        'mistral-medium',
        'mistral-large-2402',
    }:
        return 'mistral'
    if developer.casefold() == 'cohereforai' and leaf == 'cohere':
        return 'cohere'
    return None


def alpaca_eval_provenance(
    model_id: str,
    developer: str | None,
) -> AlpacaEvalProvenance:
    """Classify one leaderboard model without guessing its local runtime."""
    normalized = model_id.strip().casefold()
    id_developer, separator, leaf = normalized.partition('/')
    normalized_developer = (developer or '').strip().casefold()
    if separator:
        normalized_developer = normalized_developer or id_developer
    else:
        leaf = normalized
    normalized_developer = normalized_developer or UNKNOWN

    provider = _closed_provider(normalized_developer, leaf)
    if provider is not None:
        return AlpacaEvalProvenance(
            normalized_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            provider,
            UNKNOWN,
            UNKNOWN,
        )

    structured_open_developers = {
        '01-ai',
        'allenai',
        'deepseek-ai',
        'huggingfaceh4',
        'lmsys',
        'meta-llama',
        'microsoft',
        'mistralai',
        'openchat',
        'qwen',
        'stanford',
        'tiiuae',
        'timdettmers',
        'wizardlm',
        'xwin-lm',
    }
    if normalized_developer == 'google' and not leaf.startswith('gemini-'):
        availability = OPEN_WEIGHTS
    elif normalized_developer in structured_open_developers:
        availability = OPEN_WEIGHTS
    elif normalized_developer == 'anthropic' and leaf.startswith(
        'claude2-alpaca-'
    ):
        availability = UNKNOWN
    else:
        unscoped_open_prefixes = (
            'airoboros-',
            'baichuan-',
            'baize-',
            'chatglm',
            'conifer-',
            'dbrx-',
            'deita-',
            'internlm',
            'merlinite-',
            'minichat-',
            'minotaur-',
            'nanbeige',
            'nous-hermes-',
            'oasst-',
            'openencoder',
            'opencoder',
            'platolm-',
            'pythia-',
            'starling-',
            'storm-',
            'ultralm-',
        )
        availability = (
            OPEN_WEIGHTS
            if leaf.startswith(unscoped_open_prefixes)
            else UNKNOWN
        )

    together_routed = (
        'together' in leaf
        or leaf.endswith('-turbo')
        or '-turbo-' in leaf
    )
    if together_routed:
        return AlpacaEvalProvenance(
            normalized_developer,
            EXTERNALLY_MANAGED,
            availability,
            'together',
            UNKNOWN,
            UNKNOWN,
        )
    return AlpacaEvalProvenance(
        normalized_developer,
        UNKNOWN,
        availability,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )


__all__ = [
    'ALPACA_EVAL_METHOD_URL',
    'ALPACA_EVAL_METRICS',
    'AlpacaEvalProvenance',
    'alpaca_eval_provenance',
]
