"""Utility functions for the lighteval adapter."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

STDERR_SUFFIX = '_stderr'

# lighteval synthesises two kinds of row into the same `results` mapping it uses
# for measured tasks: a per-parent-task mean under "<parent>:_average|<fewshot>"
# and a mean over everything under the literal key "all".
# (MetricsLogger.aggregate in lighteval/logging/info_loggers.py.)
SUITE_AVERAGE_KEY = 'all'
SUBTASK_AVERAGE_SUFFIX = ':_average'

# Dropped from model_info.additional_details. lighteval dumps the whole model
# config into the results file, and LiteLLMModelConfig.api_key is a plain str,
# so a converted record would otherwise carry a live credential.
SECRET_MODEL_CONFIG_KEYS = frozenset(
    {
        'access_token',
        'api_key',
        'api_token',
        'auth_token',
        'credentials',
        'hf_token',
        'password',
        'secret',
        'token',
    }
)

_DATE_ID_PATTERN = re.compile(
    r'^results_(?P<date>\d{4}-\d{2}-\d{2}T\d{2})-(?P<minute>\d{2})-'
    r'(?P<second>\d{2}(?:\.\d+)?)$'
)

# Known metric bounds: metric_name -> (min_score, max_score).
# Names are lighteval's own metric_name values (lighteval/metrics/metrics.py).
# Infinite bounds are serialized as the JSON strings "Infinity"/"-Infinity".
KNOWN_METRIC_BOUNDS = {
    'acc': (0.0, 1.0),
    'bits_per_byte': (0.0, float('inf')),
    'bleu': (0.0, 100.0),
    'bleu_1': (0.0, 100.0),
    'bleu_4': (0.0, 100.0),
    'byte_perplexity': (1.0, float('inf')),
    'chrf': (0.0, 100.0),
    'chrf++': (0.0, 100.0),
    'em': (0.0, 1.0),
    'extractive_match': (0.0, 1.0),
    'f1': (0.0, 1.0),
    'loglikelihood_f1': (0.0, 1.0),
    'mcc': (-1.0, 1.0),
    'mf1': (0.0, 1.0),
    'mrr': (0.0, 1.0),
    'perplexity': (1.0, float('inf')),
    'ppl': (1.0, float('inf')),
    'recall': (0.0, 1.0),
    'rouge1': (0.0, 1.0),
    'rouge2': (0.0, 1.0),
    'rougeL': (0.0, 1.0),
    'rougeLsum': (0.0, 1.0),
    'summarization_coverage': (0.0, 1.0),
    'ter': (0.0, float('inf')),
    'truthfulqa_mc1': (0.0, 1.0),
    'word_perplexity': (1.0, float('inf')),
}


def is_derived_aggregate_key(task_key: str) -> bool:
    """Report whether a `results` key was averaged by lighteval, not measured."""
    if task_key == SUITE_AVERAGE_KEY:
        return True
    return task_key.split('|')[0].endswith(SUBTASK_AVERAGE_SUFFIX)


def split_task_key(task_key: str) -> tuple[str, Optional[int]]:
    """Split a `results` key into its task name and few-shot count.

    lighteval builds these as f'{task_name}|{num_fewshots}'; the task name
    itself may contain ':' for a subset, as in 'mmlu:abstract_algebra|5'.
    """
    name, separator, fewshot = task_key.rpartition('|')
    if not separator:
        return task_key, None
    try:
        return name, int(fewshot)
    except ValueError:
        return task_key, None


def is_finite_number(value: Any) -> bool:
    """Report whether a value is a real number that JSON can round-trip."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def find_metric_spec(
    task_config: Dict[str, Any], metric_name: str
) -> Optional[Dict[str, Any]]:
    """Find the configured metric that produced a given `results` entry.

    Grouped metrics declare metric_name as a list, so a spec can own several
    entries.
    """
    for spec in task_config.get('metrics') or []:
        if not isinstance(spec, dict):
            continue
        declared = spec.get('metric_name')
        if isinstance(declared, str):
            declared = [declared]
        if isinstance(declared, list) and metric_name in declared:
            return spec
    return None


def higher_is_better_for(
    metric_spec: Optional[Dict[str, Any]], metric_name: str
) -> Optional[bool]:
    """Read a metric's direction, returning None when the run does not state it."""
    if metric_spec is None:
        return None
    declared = metric_spec.get('higher_is_better')
    if isinstance(declared, bool):
        return declared
    if isinstance(declared, dict):
        value = declared.get(metric_name)
        return value if isinstance(value, bool) else None
    return None


def stderr_method_for(
    metric_spec: Optional[Dict[str, Any]], metric_name: str
) -> Optional[str]:
    """Name the estimator lighteval used for a metric's standard error.

    Mirrors get_stderr_function in lighteval/metrics/utils/stderr.py, which
    picks the analytic mean_stderr when the corpus-level aggregation's name
    contains 'mean' and bootstraps otherwise. Returns None when the results
    file does not record which aggregation ran.
    """
    if metric_spec is None:
        return None
    aggregation = metric_spec.get('corpus_level_fn')
    if isinstance(aggregation, dict):
        aggregation = aggregation.get(metric_name)
    if not isinstance(aggregation, str) or not aggregation:
        return None
    return 'analytic' if 'mean' in aggregation else 'bootstrap'


def parse_results_file_timestamp(file_path: Path) -> Optional[str]:
    """Recover the wall-clock stamp lighteval encodes in a results filename.

    lighteval names results files f'results_{date_id}.json' where date_id is
    datetime.now().isoformat() with ':' replaced by '-'. This is the only
    wall-clock time in a run: config_general's start_time and end_time come
    from time.perf_counter(), whose origin is undefined.
    """
    match = _DATE_ID_PATTERN.match(Path(file_path).stem)
    if match is None:
        return None
    return (
        f'{match.group("date")}:{match.group("minute")}:{match.group("second")}'
    )


def flatten_model_config(
    model_config: Any,
) -> tuple[Dict[str, str], List[str]]:
    """Stringify a dumped lighteval model config for additional_details.

    Returns the flattened values and the names of any credential-bearing keys
    that were dropped.
    """
    if not isinstance(model_config, dict):
        return {}, []

    flattened: Dict[str, str] = {}
    redacted: List[str] = []
    for key, value in model_config.items():
        if value is None:
            continue
        if key.lower() in SECRET_MODEL_CONFIG_KEYS:
            redacted.append(key)
            continue
        if isinstance(value, str):
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value, sort_keys=True, default=str)
    return flattened, sorted(redacted)
