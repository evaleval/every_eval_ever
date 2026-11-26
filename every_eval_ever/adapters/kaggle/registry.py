"""Canonicalize a Kaggle model slug against the hosted eval-card-registry.

``model_info.id`` is what the datastore is queried by, so one model spelled two
ways is two models to every reader. The registry
(https://github.com/evaleval/eval-card-registry) is the authority on that
spelling, and publishes a no-auth resolver so an adapter does not need a clone.

This never raises: the caller opts out for an offline run, and a request that
still fails after retries, or an unusable reply, returns the id the source gave
with provenance saying why (``unreachable`` / ``unresolved``). The reply is
parsed inside the same boundary that catches the request. What to do with an
unverified id is the caller's decision.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

#: Hosted eval-card-registry resolver (public HF Space, no auth).
RESOLVER_URL = "https://evaleval-entity-registry.hf.space/api/v1/resolve"

#: Pauses before retrying a request that failed; a malformed reply is final.
RETRY_DELAYS = (2.0, 8.0)

#: After this many lookups in a row get no answer, lookups fail at once for a
#: pause instead of each spending its retries, so a registry outage reports in
#: minutes rather than running the sweep into its job timeout.
BREAKER_THRESHOLD = 3
BREAKER_PAUSE_SECONDS = 60.0
_unanswered_in_a_row = 0
_paused_until = 0.0

#: Below this, the resolver's alias is treated as unverified and flagged for
#: review rather than trusted as canonical.
RESOLVE_CONFIDENCE_FLOOR = 0.9

#: ``model_id_resolution`` values that mean "this id is not the registry's".
_UNRESOLVED = frozenset({"unreachable", "invalid_reply", "unresolved"})

#: Values that mean the registry gave no usable answer at all: unlike a
#: definitive ``unresolved``, these say nothing about whether the id is known.
UNANSWERED = frozenset({"unreachable", "invalid_reply"})


class _InvalidReply(ValueError):
    """A 200 whose body is not a resolver answer."""


def _canonical(payload: Any) -> tuple[Optional[str], str]:
    """Read one resolver reply.

    Returns ``(canonical id, strategy)``, or ``(None, strategy)`` when the
    registry answered that it does not know the value (``no_match``). Raises
    :class:`_InvalidReply` for a 200 carrying a list, ``null`` or an object
    without a strategy: a protocol change, which is neither a confident answer
    nor a miss.
    """
    if not isinstance(payload, dict):
        raise _InvalidReply(
            f"malformed reply: expected an object, got {type(payload).__name__}"
        )
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise _InvalidReply("malformed reply: no resolution strategy")
    canonical = payload.get("canonical_id")
    if not isinstance(canonical, str) or not canonical.strip():
        # The resolver's own "I do not know this one" (strategy ``no_match``).
        return None, strategy.strip()
    return canonical.strip(), strategy.strip()


def _post(raw_repo: str, timeout: float) -> requests.Response:
    for delay in (*RETRY_DELAYS, None):
        try:
            response = requests.post(
                RESOLVER_URL,
                # `exact` is the one mode that never creates a draft canonical
                # (helpers/eval_card_registry.py); for every Kaggle slug it
                # answered as the default mode did (153/153, November 2026).
                json={"raw_value": raw_repo, "entity_type": "model", "mode": "exact"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException:
            if delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")  # the loop returns or raises


def resolve_model_id(
    raw_repo: str, *, enabled: bool = True, timeout: float = 15.0
) -> tuple[str, dict[str, Any]]:
    """Canonicalize ``raw_repo``, returning ``(model_id, provenance)``.

    ``provenance['model_id_resolution']`` is one of ``offline`` (opted out),
    ``unreachable`` (the request failed), ``invalid_reply`` (a 200 that is not
    a resolver answer), ``unresolved`` (answered, but named no canonical id) or
    ``registry`` (canonical). Only the last means the returned id is the
    registry's; the others return ``raw_repo`` unchanged.
    """
    global _unanswered_in_a_row, _paused_until
    if not enabled:
        return raw_repo, {"model_id_resolution": "offline"}
    if time.monotonic() < _paused_until:
        return raw_repo, {
            "model_id_resolution": "unreachable",
            "model_id_resolution_error": "skipped: the registry failed repeatedly",
        }
    try:
        response = _post(raw_repo, timeout)
        payload = response.json()
        canonical, strategy = _canonical(payload)
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        _unanswered_in_a_row += 1
        if _unanswered_in_a_row >= BREAKER_THRESHOLD:
            _unanswered_in_a_row = 0
            _paused_until = time.monotonic() + BREAKER_PAUSE_SECONDS
        resolution = (
            "invalid_reply" if isinstance(exc, _InvalidReply) else "unreachable"
        )
        return raw_repo, {
            "model_id_resolution": resolution,
            "model_id_resolution_error": str(exc)[:200],
        }
    _unanswered_in_a_row = 0
    if canonical is None:
        return raw_repo, {
            "model_id_resolution": "unresolved",
            "model_id_resolution_strategy": strategy,
        }
    # `payload` is a dict here: _canonical returns a canonical id only for one.
    return canonical, {
        "model_id_resolution": "registry",
        "model_id_resolution_strategy": strategy,
        "model_id_resolution_confidence": payload.get("confidence"),
        "model_id_created_new": payload.get("created_new"),
        "model_id_review_status": payload.get("review_status"),
    }


def resolved_canonically(provenance: Optional[dict[str, Any]]) -> bool:
    """Whether the returned id is the registry's canonical one."""
    if not provenance:
        return False
    return provenance.get("model_id_resolution") == "registry"


def needs_review(provenance: Optional[dict[str, Any]]) -> bool:
    """True when a resolved id is not a confident, already-reviewed canonical.

    An unreachable resolver, a reply naming no canonical id, a freshly
    auto-created draft, a non-``reviewed`` status, or confidence below the
    floor. (``offline`` is a property of the run, reported once by the caller
    rather than per model.)
    """
    if not provenance:
        return False
    if provenance.get("model_id_resolution") in _UNRESOLVED:
        return True
    if provenance.get("model_id_created_new"):
        return True
    status = provenance.get("model_id_review_status")
    if status not in (None, "reviewed"):
        return True
    confidence = provenance.get("model_id_resolution_confidence")
    return (
        isinstance(confidence, (int, float))
        and confidence < RESOLVE_CONFIDENCE_FLOOR
    )
