"""Conservative model provenance for Terminal-Bench 2.0 entries."""

from __future__ import annotations

from dataclasses import dataclass

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

TERMINAL_BENCH_METHOD_URL = (
    'https://www.tbench.ai/leaderboard/terminal-bench/2.0'
)

MODEL_AVAILABILITY_SOURCES = {
    'qwen3_coder': 'https://huggingface.co/collections/Qwen/qwen3-coder',
    'deepseek': 'https://huggingface.co/deepseek-ai/models',
    'minimax_m2': 'https://huggingface.co/collections/MiniMaxAI/minimax-m2',
    'kimi_k2': 'https://huggingface.co/collections/moonshotai/kimi-k2',
    'gpt_oss': 'https://huggingface.co/openai/models',
    'glm': 'https://huggingface.co/zai-org/models',
}


@dataclass(frozen=True)
class TerminalBenchProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def terminal_bench_provenance(model_id: str) -> TerminalBenchProvenance:
    """Classify known model families and preserve unknowns conservatively."""
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid Terminal-Bench model id: {model_id!r}')

    closed = (
        (developer == 'anthropic' and leaf.startswith('claude-'))
        or (developer == 'google' and leaf.startswith('gemini-'))
        or (
            developer == 'openai'
            and leaf.startswith('gpt-')
            and not leaf.startswith('gpt-oss-')
        )
        or (developer == 'xai' and leaf.startswith('grok-'))
    )
    if closed:
        return TerminalBenchProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            developer,
            UNKNOWN,
            UNKNOWN,
        )

    open_weight = (
        (developer == 'alibaba' and leaf == 'qwen-3-coder-480b')
        or (developer == 'deepseek' and leaf == 'deepseek-v3.2')
        or (
            developer == 'minimax'
            and leaf in {'minimax-m2', 'minimax-m2.1', 'minimax-m2.5'}
        )
        or (developer == 'moonshot-ai' and leaf.startswith('kimi-k2'))
        or (developer == 'openai' and leaf.startswith('gpt-oss-'))
        or (developer == 'zhipu-ai' and leaf in {'glm-4.6', 'glm-4.7', 'glm-5'})
    )
    if open_weight:
        return TerminalBenchProvenance(
            UNKNOWN,
            OPEN_WEIGHTS,
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
        )

    if normalized == 'multiple/multiple':
        return TerminalBenchProvenance(
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
            UNKNOWN,
        )

    return TerminalBenchProvenance(
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )
