"""Utility functions for the lm-eval adapter."""

from pathlib import Path
from typing import Dict, Optional


def evaluation_result_id(metric_name: str, filter_name: str) -> str:
    """Build the join key shared by aggregate results and instance rows.

    lm-eval reports a metric once per filter, so a metric name alone is not
    unique within a task: `exact_match` under `flexible-extract` and under no
    filter are separate results.
    """
    if not filter_name or filter_name == 'none':
        return metric_name
    return f'{metric_name}:{filter_name}'


# How a metric's standard error was computed follows from its *aggregation*, not
# from its name: `stderr_for_metric` in lm_eval/api/metrics.py bootstraps these
# aggregations, gives `mean` and `acc_all` an analytic standard error
# (`sample_stddev / sqrt(n)`), and computes none at all for anything else.
BOOTSTRAP_AGGREGATIONS: frozenset[str] = frozenset(
    {
        'bleu',
        'chrf',
        'f1_score',
        'matthews_corrcoef',
        'median',
        'nanmean',
        'perplexity',
        'ter',
    }
)
ANALYTIC_AGGREGATIONS: frozenset[str] = frozenset({'acc_all', 'mean'})

# lm-eval resamples these three at `min(bootstrap_iters, 100)`, whatever the run
# configured, so the configured value is not the number that was used.
CAPPED_BOOTSTRAP_METRICS: frozenset[str] = frozenset({'bleu', 'chrf', 'ter'})
BOOTSTRAP_ITERS_CAP = 100


def aggregations_by_metric(task_config: Dict) -> Dict[str, str]:
    """The aggregation each metric of a task was reduced with, where stated.

    lm-eval resolves an unstated aggregation from its own registry at load time
    and dumps the task's config as written, so a metric absent here has an
    aggregation we cannot read off the log.
    """
    aggregations: Dict[str, str] = {}
    for entry in task_config.get('metric_list') or []:
        if not isinstance(entry, dict):
            continue
        metric, aggregation = entry.get('metric'), entry.get('aggregation')
        if isinstance(metric, str) and isinstance(aggregation, str):
            aggregations[metric] = aggregation
    return aggregations


def standard_error_method(aggregation: Optional[str]) -> Optional[str]:
    """How lm-eval computed the standard error of a metric aggregated this way."""
    if aggregation in BOOTSTRAP_AGGREGATIONS:
        return 'bootstrap'
    if aggregation in ANALYTIC_AGGREGATIONS:
        return 'analytic'
    return None


def bootstrap_resamples(
    metric_name: str,
    aggregation: Optional[str],
    configured_iters: Optional[int],
) -> Optional[int]:
    """How many resamples went into a bootstrapped standard error."""
    if configured_iters is None or aggregation not in BOOTSTRAP_AGGREGATIONS:
        return None
    if metric_name in CAPPED_BOOTSTRAP_METRICS:
        return min(configured_iters, BOOTSTRAP_ITERS_CAP)
    return configured_iters


def parse_model_args(model_args: str | None) -> Dict[str, str]:
    """Parse lm-eval model_args string (comma-separated key=value pairs).

    Handles the common format: "pretrained=EleutherAI/pythia-160m,dtype=float16"
    """
    if not model_args or not isinstance(model_args, str):
        return {}
    result = {}
    for part in model_args.split(','):
        if '=' in part:
            key, value = part.split('=', 1)
            result[key.strip()] = value.strip()
        elif result:
            # Continuation of previous value that contained a comma
            last_key = list(result.keys())[-1]
            result[last_key] += ',' + part
    return result


def find_samples_file(output_dir: Path, task_name: str) -> Optional[Path]:
    """Find the samples JSONL file for a given task in the output directory.

    lm-eval writes samples as: samples_<task_name>_<datetime>.jsonl
    """
    pattern = f'samples_{task_name}_*.jsonl'
    matches = sorted(output_dir.glob(pattern))
    if matches:
        return matches[-1]  # Most recent
    return None


# Maps lm-eval config.model values to every_eval_ever inference_platform
MODEL_TYPE_TO_INFERENCE_PLATFORM = {
    'openai-completions': 'openai',
    'openai-chat-completions': 'openai',
    'anthropic': 'anthropic',
    'anthropic-chat': 'anthropic',
    'together': 'together',
}

# Maps lm-eval config.model values to inference engine names
MODEL_TYPE_TO_INFERENCE_ENGINE = {
    'hf': 'transformers',
    'vllm': 'vllm',
    'gguf': 'llama.cpp',
}

# The registry's canonical slug for this harness, which is what namespaces the
# metric ids it reports that the registry does not carry.
LM_EVAL_HARNESS_ID = 'lm-evaluation-harness'

# Bounds for the metrics lm-eval spells its own way, layered over
# converters/common/metrics.py::SHARED_METRIC_BOUNDS. lm-eval's `bleu`, `chrf`
# and `ter` are sacrebleu's, which report 0-100 rather than the 0-1 of nltk's
# sentence_bleu; `ter` has no ceiling, since a hypothesis can need more edits
# than the reference has tokens.
LM_EVAL_METRIC_BOUNDS = {
    'mc1': (0.0, 1.0),
    'mc2': (0.0, 1.0),
    'bleu': (0.0, 100.0),
    'chrf': (0.0, 100.0),
    'rouge1': (0.0, 1.0),
    'rouge2': (0.0, 1.0),
    'rougeL': (0.0, 1.0),
    'rougeLsum': (0.0, 1.0),
    'ter': (0.0, float('inf')),
    'word_perplexity': (1.0, float('inf')),
    'byte_perplexity': (1.0, float('inf')),
    # v0.3's spelling of the same metric; see CANONICAL_METRIC_IDS.
    'ppl': (1.0, float('inf')),
    'perplexity': (1.0, float('inf')),
    'bits_per_byte': (0.0, float('inf')),
}
