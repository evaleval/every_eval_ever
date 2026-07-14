"""Source-backed model provenance for Multi-SWE-Bench submissions."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

MULTI_SWE_BENCH_METHOD_URL = (
    'https://github.com/multi-swe-bench/experiments'
)


@dataclass(frozen=True)
class MultiSWEBenchProvenance:
    model_id: str
    developer: str
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def multi_swe_bench_provenance(
    model_id: str,
) -> MultiSWEBenchProvenance:
    """Classify the small reviewed model set used by the leaderboard."""
    raw = model_id.strip()
    normalized = raw.casefold()
    developer, separator, leaf = normalized.partition('/')
    raw_leaf = raw.partition('/')[2]
    if not separator or not developer or not leaf:
        return MultiSWEBenchProvenance(
            raw, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN
        )

    canonical_id = raw
    canonical_developer = developer
    if developer == 'unknown' and leaf.startswith('doubao-'):
        canonical_developer = 'bytedance'
        canonical_id = f'bytedance/{raw_leaf}'
    elif developer == 'unknown' and leaf == 'codearts-minimax-m2.5':
        canonical_developer = 'minimax'
        canonical_id = f'minimax/{raw_leaf}'

    if canonical_developer == 'anthropic' and leaf.startswith('claude-'):
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'anthropic',
            UNKNOWN,
            UNKNOWN,
        )
    if canonical_developer == 'google' and leaf.startswith('gemini-'):
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'google',
            UNKNOWN,
            UNKNOWN,
        )
    if canonical_developer == 'openai' and leaf.startswith(
        ('gpt-', 'openai-o1', 'openai-o3')
    ):
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'openai',
            UNKNOWN,
            UNKNOWN,
        )
    if canonical_developer == 'bytedance' and leaf.startswith('doubao-'):
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            'bytedance',
            UNKNOWN,
            UNKNOWN,
        )
    if canonical_developer == 'minimax' and leaf == 'codearts-minimax-m2.5':
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            EXTERNALLY_MANAGED,
            OPEN_WEIGHTS,
            'codearts',
            UNKNOWN,
            UNKNOWN,
        )

    open_families = (
        ('alibaba', ('qwen',)),
        ('deepseek', ('deepseek-',)),
        ('meta', ('llama-',)),
    )
    if any(
        canonical_developer == expected and leaf.startswith(prefixes)
        for expected, prefixes in open_families
    ):
        return MultiSWEBenchProvenance(
            canonical_id,
            canonical_developer,
            UNKNOWN,
            OPEN_WEIGHTS,
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
        )

    return MultiSWEBenchProvenance(
        canonical_id,
        canonical_developer or UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )


__all__ = [
    'MULTI_SWE_BENCH_METHOD_URL',
    'MultiSWEBenchProvenance',
    'multi_swe_bench_provenance',
]
