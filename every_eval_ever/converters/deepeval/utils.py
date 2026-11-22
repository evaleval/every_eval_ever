"""Shared helpers for the DeepEval converter."""

import re
from datetime import datetime, timezone

_RUN_FILE_PATTERN = re.compile(r'test_run_(\d{8})_(\d{6})')


def metric_key(metric_name: str) -> str:
    """Normalize a DeepEval metric display name to its EEE metric name.

    DeepEval reports metrics by display name ('Exact Match',
    'Tool Correctness'); EEE metric names use the snake_case spelling
    harnesses report programmatically ('exact_match'), as lm-eval and
    HELM do. The verbatim display name is preserved on the result's
    metric_config additional_details.
    """
    key = re.sub(r'[^a-z0-9]+', '_', metric_name.strip().lower())
    return key.strip('_') or 'metric'


def evaluation_result_id(metric_name: str) -> str:
    """Build the join key shared by aggregate results and instance rows."""
    return metric_key(metric_name)


def parse_run_timestamp(file_name: str) -> int | None:
    """Epoch seconds from a DeepEval export file name, when it carries one.

    DeepEval names results-folder exports test_run_<YYYYMMDD>_<HHMMSS>.json
    after the run's local time, and the file records no timezone. The
    name's time is therefore read as UTC: the epoch is nominal, but the
    same export keys identically on any machine, which is what makes
    re-conversion idempotent in a shared datastore. The rolling snapshot
    (.deepeval/.latest_run_full.json) carries no timestamp and returns
    None.
    """
    match = _RUN_FILE_PATTERN.search(file_name)
    if match is None:
        return None
    try:
        moment = datetime.strptime(
            f'{match.group(1)}_{match.group(2)}', '%Y%m%d_%H%M%S'
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(moment.timestamp())


def is_int(value: object) -> bool:
    """True for genuine ints; JSON booleans are not counts or indices."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: object) -> bool:
    """True for genuine ints/floats; JSON booleans are not scores."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
