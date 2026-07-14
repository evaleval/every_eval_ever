"""Source-backed execution and availability rules for OpenEval models."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

OPENEVAL_METHOD_URL = 'https://github.com/open-eval/OpenEval'


@dataclass(frozen=True)
class OpenEvalProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def _matches_family(
    developer: str,
    leaf: str,
    expected_developer: str,
    prefixes: tuple[str, ...],
) -> bool:
    return developer == expected_developer and leaf.startswith(prefixes)


def openeval_provenance(model_id: str) -> OpenEvalProvenance:
    """Classify one OpenEval model without guessing undocumented execution."""
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        return OpenEvalProvenance(
            UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN
        )

    if _matches_family(
        developer, leaf, 'openai', ('gpt-', 'o1', 'o3', 'o4')
    ):
        return OpenEvalProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'openai',
            UNKNOWN,
            UNKNOWN,
        )
    if _matches_family(developer, leaf, 'xai', ('grok-',)):
        return OpenEvalProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'xai',
            UNKNOWN,
            UNKNOWN,
        )

    open_families = (
        ('alibaba', ('qwen-',)),
        ('deepseek', ('deepseek-',)),
        ('eleutherai', ('pythia-',)),
        ('google', ('gemma-',)),
        ('lmsys', ('vicuna-',)),
        ('meta', ('llama-',)),
        ('microsoft', ('phi-',)),
        ('mistralai', ('mistral-', 'mixtral-')),
        ('moonshotai', ('kimi-',)),
        ('mosaicml', ('mpt-',)),
        ('stanford', ('alpaca-',)),
        ('tiiuae', ('falcon-',)),
        ('together', ('redpajama-',)),
        ('unknown', ('stablelm-',)),
    )
    if any(
        _matches_family(developer, leaf, expected, prefixes)
        for expected, prefixes in open_families
    ):
        return OpenEvalProvenance(
            UNKNOWN, OPEN_WEIGHTS, UNKNOWN, UNKNOWN, UNKNOWN
        )

    return OpenEvalProvenance(UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN)


__all__ = [
    'OPENEVAL_METHOD_URL',
    'OpenEvalProvenance',
    'openeval_provenance',
]
