"""Shared constants and helpers for the SimpleAudit adapter."""

from __future__ import annotations

# The registry's canonical slug for this harness, which is what namespaces the
# metric ids it reports that the registry does not carry.
SIMPLEAUDIT_HARNESS_ID = 'simpleaudit'

# Bounds for SimpleAudit's own metrics, layered over
# converters/common/metrics.py::SHARED_METRIC_BOUNDS. The safety score is
# always 0-100 by construction of the engine's penalty weighting.
SIMPLEAUDIT_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    'safety_score': (0.0, 100.0),
}

# Scenario severities as written by SimpleAudit core (the engine's Judge verdict)
# and Studio (its export endpoint). `pass` means the auditor found no issue;
# `ERROR` and `ungraded` are *not* safety findings — they mean the scenario
# could not be scored — so they sit outside the severity scale entirely.
SEVERITY_LEVEL_NAMES: tuple[str, ...] = (
    'critical',
    'high',
    'medium',
    'low',
    'pass',
)

# Severities a consumer must not read as "the model was safe" or "unsafe":
# they were never judged.
NON_GRADED_SEVERITIES: frozenset[str] = frozenset({'ERROR', 'ungraded'})


def evaluation_result_id(metric_name: str) -> str:
    """The join key shared by aggregate results and instance-level rows.

    A run reports each of its metrics once, so the metric name alone is
    unique within an `evaluation_id`.
    """
    return metric_name


# Datastore records must live under a known developer folder, but single-run
# SimpleAudit files carry no model name at all. This synthetic namespace keeps
# such runs publishable without pretending to identify a real organization;
# the record says in model_info.additional_details that the id is a fallback.
UNIDENTIFIED_MODEL_ID = 'simpleaudit-unknown/model'


def resolve_model_id(
    provenance_target: str | None,
    cli_override: str | None,
    repeated_runs_key: str | None,
) -> tuple[str, str]:
    """`(model_id, source)` for a run's model identity.

    Precedence: CLI override > Studio provenance > repeated-experiment runs
    key > the synthetic unknown id. The returned source label is recorded on
    `model_info` so a reader can tell which rung won.
    """
    if cli_override:
        return cli_override, 'cli --model_id'
    if provenance_target:
        return provenance_target, 'studio_provenance.target_model'
    if repeated_runs_key:
        return repeated_runs_key, 'repeated_experiment.runs key'
    return UNIDENTIFIED_MODEL_ID, 'unidentified (no model name in source)'
