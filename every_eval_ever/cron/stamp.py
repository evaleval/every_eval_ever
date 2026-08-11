"""Mark generated records as cron-produced, before they are published.

The cron stamps records after the adapter wrote them rather than asking each
adapter to do it, so an adapter behaves identically whether a person or the
schedule invoked it. Both facts a later fix needs — that a record arrived
automatically, and when — live in ``source_metadata.additional_details``, which
the schema defines as a string map.

The stamp never overwrites what an adapter reported.

``model_info.additional_details.deployment_type`` and ``model_availability``
are inferred axes a leaderboard almost never states. ``ModelInfo`` already
defaults both to ``unknown`` (see ``every_eval_ever.post_codegen``), so the
cron does not need to fill them — it names the ones that came out ``unknown``
under :data:`UNKNOWN_FIELDS_KEY`, which is what lets a later pass find the
records still needing a real value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from every_eval_ever.eval_types import EvaluationLog

TYPE_OF_ADDITION_KEY = 'type_of_addition'
CRON_ADDITION_TYPE = 'cron'
RUN_DATE_KEY = 'cron_run_date'
ADAPTER_KEY = 'cron_adapter'
RUN_URL_KEY = 'cron_run_url'
UNKNOWN_FIELDS_KEY = 'cron_unknown_inferred_fields'

UNKNOWN = 'unknown'
#: Axes that are almost never available to an adapter directly. Both are
#: required by the schema and both admit 'unknown'.
INFERRED_MODEL_FIELDS = ('deployment_type', 'model_availability')


class StampError(Exception):
    """Raised when a record cannot be stamped."""


class StampConflict(StampError):
    """Raised when a record already carries a different addition type."""


@dataclass
class StampSummary:
    """What stamping did across a set of files."""

    stamped: int = 0
    #: How many records came out ``unknown`` on each inferred axis.
    unknown_inferred: dict[str, int] = field(default_factory=dict)
    paths: list[Path] = field(default_factory=list)

    def record(self, path: Path, unknown: list[str]) -> None:
        self.stamped += 1
        self.paths.append(path)
        for name in unknown:
            self.unknown_inferred[name] = self.unknown_inferred.get(name, 0) + 1


def stamp_payload(
    payload: dict[str, Any],
    *,
    adapter: str,
    run_date: str,
    run_url: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``payload`` stamped, plus the inferred axes that are ``unknown``.

    Raises:
        StampConflict: if the record already declares a different
            ``type_of_addition``, which would mean the adapter has its own
            meaning for the key.
    """
    stamped = json.loads(json.dumps(payload))
    source_metadata = stamped.setdefault('source_metadata', {})
    details = source_metadata.setdefault('additional_details', {})

    existing = details.get(TYPE_OF_ADDITION_KEY)
    if existing is not None and existing != CRON_ADDITION_TYPE:
        raise StampConflict(
            f'record declares {TYPE_OF_ADDITION_KEY}={existing!r}; refusing to '
            f'relabel it as {CRON_ADDITION_TYPE!r}'
        )

    details[TYPE_OF_ADDITION_KEY] = CRON_ADDITION_TYPE
    details[RUN_DATE_KEY] = run_date
    details[ADAPTER_KEY] = adapter
    if run_url:
        details[RUN_URL_KEY] = run_url

    unknown = _note_unknown_inferred_fields(stamped)
    if unknown:
        details[UNKNOWN_FIELDS_KEY] = ','.join(unknown)

    return stamped, unknown


def _note_unknown_inferred_fields(payload: dict[str, Any]) -> list[str]:
    """Return the inferred axes that are ``unknown``, filling any that are absent.

    ``ModelInfo`` normally fills these before a record is ever written; the
    ``setdefault`` here only matters for a record that reached the cron without
    passing through the model.
    """
    model_info = payload.setdefault('model_info', {})
    details = model_info.setdefault('additional_details', {})
    unknown = []
    for name in INFERRED_MODEL_FIELDS:
        if not details.get(name):
            details[name] = UNKNOWN
        if details[name] == UNKNOWN:
            unknown.append(name)
    return unknown


def _serialize(payload: dict[str, Any]) -> str:
    """Serialize exactly as ``helpers.io`` does, so the stamp is the only diff."""
    validated = EvaluationLog.model_validate(payload)
    return (
        json.dumps(
            validated.model_dump(mode='json', exclude_none=True),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + '\n'
    )


def stamp_file(
    path: str | Path,
    *,
    adapter: str,
    run_date: str,
    run_url: str | None = None,
) -> list[str]:
    """Stamp one aggregate record in place, returning its ``unknown`` axes."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    stamped, unknown = stamp_payload(
        payload,
        adapter=adapter,
        run_date=run_date,
        run_url=run_url,
    )
    try:
        text = _serialize(stamped)
    except ValidationError as error:
        # Name the file: the useful fact is which record the adapter got wrong.
        raise StampError(
            f'{path} is not a valid EvaluationLog, so it cannot be stamped or '
            f'published: {error}'
        ) from error
    path.write_text(text, encoding='utf-8')
    return unknown


def aggregate_records(root: str | Path) -> list[Path]:
    """Return the aggregate records under a ``data/`` tree, sample files aside.

    Instance-level ``*_samples.jsonl`` companions carry no ``source_metadata``;
    they are reachable from the aggregate that references them, which is what
    the stamp goes on.
    """
    data_root = Path(root)
    return sorted(
        path
        for path in data_root.glob('*/*/*/*.json')
        if not path.name.endswith('_samples.json')
    )


def stamp_tree(
    root: str | Path,
    *,
    adapter: str,
    run_date: str,
    run_url: str | None = None,
) -> StampSummary:
    """Stamp every aggregate record under a ``data/`` tree."""
    summary = StampSummary()
    for path in aggregate_records(root):
        unknown = stamp_file(
            path,
            adapter=adapter,
            run_date=run_date,
            run_url=run_url,
        )
        summary.record(path, unknown)
    return summary
