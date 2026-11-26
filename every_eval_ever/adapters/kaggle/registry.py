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

#: Below this, the resolver's alias is treated as unverified and flagged for
#: review rather than trusted as canonical.
RESOLVE_CONFIDENCE_FLOOR = 0.9

#: ``model_id_resolution`` values that mean "this id is not the registry's".
_UNRESOLVED = frozenset({"unreachable", "unresolved"})


def _canonical(payload: Any) -> tuple[Optional[str], str]:
    """Read one resolver reply.

    Returns ``(canonical id, strategy)``, or ``(None, reason)`` when the reply
    names no canonical id — whether because the registry does not know the
    value (``no_match``) or because the reply is not one this code understands.
    A 200 carrying a list, ``null`` or an object without a strategy is a
    protocol change, and must not read as a confident answer.
    """
    if not isinstance(payload, dict):
        return None, (
            f"malformed reply: expected an object, got {type(payload).__name__}"
        )
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        return None, "malformed reply: no resolution strategy"
    canonical = payload.get("canonical_id")
    if not isinstance(canonical, str) or not canonical.strip():
        # The resolver's own "I do not know this one" (strategy ``no_match``)
        # arrives this way, so it is reported as the reason rather than as an
        # error.
        return None, strategy.strip()
    return canonical.strip(), strategy.strip()


def _post(raw_repo: str, timeout: float) -> requests.Response:
    for delay in (*RETRY_DELAYS, None):
        try:
            response = requests.post(
                RESOLVER_URL,
                json={"raw_value": raw_repo, "entity_type": "model"},
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
    ``unreachable`` (the request failed), ``unresolved`` (answered, but named no
    canonical id) or ``registry`` (canonical). Only the last means the returned
    id is the registry's; the others return ``raw_repo`` unchanged.
    """
    if not enabled:
        return raw_repo, {"model_id_resolution": "offline"}
    try:
        response = _post(raw_repo, timeout)
        payload = response.json()
        canonical, strategy = _canonical(payload)
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        return raw_repo, {
            "model_id_resolution": "unreachable",
            "model_id_resolution_error": str(exc)[:200],
        }
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
