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
        'auth',
        'auth_token',
        'credentials',
        'hf_token',
        'inference_server_auth',
        'key',
        'password',
        'secret',
        'token',
    }
)

# Sentinel for "no value supplied", since None is itself a value a caller may pass.
_UNSET = object()

# Suffixes that make a key credential-bearing whatever the provider prefix.
# Nested `env_vars` mappings are where these actually appear: OPENAI_API_KEY,
# AWS_SECRET_ACCESS_KEY, HUGGING_FACE_HUB_TOKEN. Anchoring on the suffix keeps
# `tokenizer` and `max_tokens` out of the redaction set.
#
# `_auth` was added after lighteval's own fix (huggingface/lighteval#1326) excluded
# BOTH `api_key` and `inference_server_auth`. We knew about the first and not the
# second, which is the point: this list is a guess at someone else's field names, so
# it should be widened whenever upstream tells us one we missed.
_SECRET_KEY_SUFFIXES = (
    '_key',
    '_token',
    '_secret',
    '_password',
    '_credentials',
    '_auth',
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


DETAILS_DIR_NAME = 'details'

# How far above a results file to look for the sibling `details/` tree.
# results_path_template lets an operator move the results directory but not the
# details one, so the two are not always siblings.
_DETAILS_SEARCH_DEPTH = 6


def results_file_date_id(file_path: Path) -> Optional[str]:
    """Recover the raw date_id lighteval stamps into a results filename.

    This is the filename form, ':' replaced by '-', which is also the name of
    the details subdirectory and part of every details filename. Use
    parse_results_file_timestamp for the ISO-8601 form.
    """
    stem = Path(file_path).stem
    if _DATE_ID_PATTERN.match(stem) is None:
        return None
    return stem[len('results_') :]


def details_file_name(task_key: str, date_id: str) -> str:
    """Name the per-sample parquet lighteval writes for one task of one run."""
    return f'details_{task_key}_{date_id}.parquet'


def find_details_file(
    results_path: Path,
    task_key: str,
    model_name: Optional[str] = None,
) -> Optional[Path]:
    """Locate the per-sample parquet belonging to one task of one run.

    lighteval writes details to
    `<output_dir>/details/<model_name>/<date_id>/details_<task_key>_<date_id>.parquet`
    while the results file it belongs to sits under `<output_dir>/results/...`,
    so the two are found by walking up to the shared output directory. Returns
    None when the run was made without `save_details`.
    """
    results_path = Path(results_path)
    date_id = results_file_date_id(results_path)
    if date_id is None:
        return None
    expected = details_file_name(task_key, date_id)

    directory = results_path.resolve().parent
    for _ in range(_DETAILS_SEARCH_DEPTH):
        details_root = directory / DETAILS_DIR_NAME
        if details_root.is_dir():
            # A model subtree first: a details root can hold several models,
            # and matching the run's own model keeps a same-named task from a
            # different model out of this evaluation's samples.
            roots = []
            if model_name:
                model_root = details_root / model_name.strip('/')
                if model_root.is_dir():
                    roots.append(model_root)
            roots.append(details_root)
            for root in roots:
                # Compared by name rather than globbed: a task key carries '|'
                # and ':', and glob would read a '[' in a task name as a
                # character class.
                for candidate in sorted(root.rglob('*.parquet')):
                    if candidate.name == expected:
                        return candidate
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


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


# lighteval metric names whose cross-source identity is unambiguous. Only names
# that mean the same thing in every harness belong here — this is the join key,
# so a wrong entry silently merges two different measurements, which is worse
# than leaving one unjoined.
#
# Everything else gets a stable `lighteval/<name>` id: namespaced rather than
# guessed, and never bare. `metric_id_source` records which route was taken, so
# a later pass can resolve the namespaced ones against the eval-card-registry
# without having to re-derive which were canonical to begin with.
CANONICAL_METRIC_IDS = {
    'acc': 'accuracy',
    'acc_norm': 'accuracy_normalized',
    'bleu': 'bleu',
    'chrf': 'chrf',
    'exact_match': 'exact_match',
    'f1': 'f1',
    'mcc': 'matthews_correlation',
    'perplexity': 'perplexity',
    'rouge1': 'rouge1',
    'rouge2': 'rouge2',
    'rougeL': 'rougeL',
    'ter': 'ter',
    'word_perplexity': 'word_perplexity',
}

METRIC_ID_NAMESPACE = 'lighteval'


def resolve_metric_id(metric_name: str) -> tuple[str, str]:
    """Return (metric_id, how_it_was_derived) for a lighteval metric name.

    `metric_id` is the cross-source join key and must always be set, so this
    never returns None. Canonical names come from the table above; anything
    else is namespaced under `lighteval/` rather than being invented, and the
    second element says which happened.
    """
    canonical = CANONICAL_METRIC_IDS.get(metric_name)
    if canonical is not None:
        return canonical, 'canonical'
    return f'{METRIC_ID_NAMESPACE}/{metric_name}', 'namespaced_unresolved'


def _is_secret_key(key: Any, value: Any = _UNSET) -> bool:
    """True if a config key names a credential.

    Exact names cover the common cases; the suffix rule catches the
    provider-prefixed forms that show up inside nested `env_vars` mappings,
    such as OPENAI_API_KEY or AWS_SECRET_ACCESS_KEY.

    Matching on suffixes rather than substrings is deliberate: 'token' as a
    substring would also redact `tokenizer`, and 'tokens' would take
    `max_tokens`. Both are ordinary evaluation config worth keeping, and
    neither ends in `_token`.
    """
    lowered = str(key).lower()
    if lowered in SECRET_MODEL_CONFIG_KEYS:
        # An exact name redacts whatever it holds. If a field called `api_key`
        # carries something odd, that is still not worth publishing.
        return True
    if not lowered.endswith(_SECRET_KEY_SUFFIXES):
        return False
    # A suffix is a weaker signal than a name, so require a value a credential
    # could be. `requires_auth: True` ends in `_auth` but a boolean cannot carry
    # a secret, and redacting it would delete provenance to no benefit. Anything
    # not positively known to be harmless still redacts -- a missed credential in
    # a published record is unrecoverable, while an over-redacted setting is not.
    if value is _UNSET:
        return True
    return not isinstance(value, (bool, int, float))


def _sanitize_config_value(
    value: Any, path: List[str], redacted: List[str]
) -> Any:
    """Strip credential-bearing keys from nested mappings and sequences.

    lighteval dumps the whole model config, and a nested `env_vars` mapping can
    carry a live provider token. Filtering only the top level left those values
    to be serialized wholesale into `additional_details`, which is published.
    """
    if isinstance(value, dict):
        cleaned: Dict[Any, Any] = {}
        for key, item in value.items():
            child_path = path + [str(key)]
            if _is_secret_key(key, item):
                redacted.append('.'.join(child_path))
                continue
            cleaned[key] = _sanitize_config_value(item, child_path, redacted)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_config_value(item, path + [str(index)], redacted)
            for index, item in enumerate(value)
        ]
    return value


def flatten_model_config(
    model_config: Any,
) -> tuple[Dict[str, str], List[str]]:
    """Stringify a dumped lighteval model config for additional_details.

    Returns the flattened values and the dotted paths of any credential-bearing
    keys that were dropped, nested ones included.
    """
    if not isinstance(model_config, dict):
        return {}, []

    flattened: Dict[str, str] = {}
    redacted: List[str] = []
    for key, value in model_config.items():
        if value is None:
            continue
        if _is_secret_key(key, value):
            redacted.append(str(key))
            continue
        if isinstance(value, str):
            flattened[key] = value
        else:
            cleaned = _sanitize_config_value(value, [str(key)], redacted)
            flattened[key] = json.dumps(cleaned, sort_keys=True, default=str)
    return flattened, sorted(redacted)
