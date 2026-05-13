"""HELM metric filtering helpers."""

from __future__ import annotations

from typing import Optional


# HELM emits both benchmark metrics and bookkeeping telemetry in stats.json /
# per_instance_stats.json. In this PR, only benchmark-quality metrics become
# EEE aggregate/detail metric rows. Bookkeeping can be mapped to token_usage,
# performance, metadata, or additional_details in a future follow-up.
CORE_METRIC_PREFIXES: tuple[str, ...] = (
    'exact_match',
    'quasi_exact_match',
    'prefix_exact_match',
    'quasi_prefix_exact_match',
    'classification_micro_f1',
    'classification_macro_f1',
    'f1_score',
    'rouge_l',
    'bleu_',
    'ifeval_strict_accuracy',
    'chain_of_thought_correctness',
    'math_equiv',
    'math_equiv_chain_of_thought',
)


def is_core_metric(metric_name: Optional[str]) -> bool:
    """Return True when a HELM stat should become an EEE metric row."""
    return bool(metric_name) and any(
        metric_name.startswith(prefix) for prefix in CORE_METRIC_PREFIXES
    )
