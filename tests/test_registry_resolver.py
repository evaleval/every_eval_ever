"""The registry resolver promises never to raise, so prove it.

`model_info.id` is what the datastore is queried by, so this helper sits in
front of every record an adapter writes. Its contract is that a bad answer
degrades to the source's own id, marked unverified — never an exception that
aborts a sweep, and never a malformed reply mistaken for a confident one.
"""

from __future__ import annotations

import pytest
import requests

from every_eval_ever.helpers import registry


class _FakeResponse:
    def __init__(self, payload=None, status=200, raises=None):
        self._payload = payload
        self._status = status
        self._raises = raises

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"{self._status} error")

    def json(self):
        if self._raises is not None:
            raise self._raises
        return self._payload


def _post(monkeypatch, response):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(registry.requests, "post", fake_post)
    return calls


def test_a_canonical_answer_is_used_with_its_provenance(monkeypatch):
    calls = _post(
        monkeypatch,
        _FakeResponse(
            {
                "canonical_id": "google/gemma-3-27b-it",
                "strategy": "exact",
                "confidence": 1.0,
                "created_new": False,
                "review_status": "reviewed",
            }
        ),
    )
    model_id, provenance = registry.resolve_model_id("gemma-3-27b-it")
    assert model_id == "google/gemma-3-27b-it"
    assert provenance["model_id_resolution"] == "registry"
    assert provenance["model_id_resolution_strategy"] == "exact"
    assert registry.resolved_canonically(provenance) is True
    assert registry.needs_review(provenance) is False
    assert calls[0][0] == registry.RESOLVER_URL
    assert calls[0][1]["json"] == {
        "raw_value": "gemma-3-27b-it",
        "entity_type": "model",
    }


def test_opting_out_touches_no_network(monkeypatch):
    calls = _post(monkeypatch, _FakeResponse({}))
    model_id, provenance = registry.resolve_model_id("acme/x", enabled=False)
    assert model_id == "acme/x"
    assert provenance == {"model_id_resolution": "offline"}
    assert calls == []


# ---------------------------------------------------------------------------
# A 200 that is not a usable answer must not read as a confident one.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        [],
        ["google/gemma"],
        None,
        "google/gemma",
        42,
        {},
        {"canonical_id": "google/gemma"},
        {"canonical_id": "google/gemma", "strategy": ""},
        {"canonical_id": "google/gemma", "strategy": None},
        {"canonical_id": "google/gemma", "strategy": 7},
        {"strategy": "exact"},
        {"strategy": "exact", "canonical_id": ""},
        {"strategy": "exact", "canonical_id": "   "},
        {"strategy": "exact", "canonical_id": None},
        {"strategy": "exact", "canonical_id": 7},
    ],
)
def test_a_malformed_200_falls_back_instead_of_raising(monkeypatch, payload):
    _post(monkeypatch, _FakeResponse(payload))
    model_id, provenance = registry.resolve_model_id("gemma-3-27b-it")
    assert model_id == "gemma-3-27b-it"
    assert registry.resolved_canonically(provenance) is False
    assert registry.needs_review(provenance) is True


def test_no_match_is_reported_as_the_reason(monkeypatch):
    _post(
        monkeypatch,
        _FakeResponse(
            {"canonical_id": None, "strategy": "no_match", "confidence": 0.0}
        ),
    )
    model_id, provenance = registry.resolve_model_id("gpt-5.9-unreleased")
    assert model_id == "gpt-5.9-unreleased"
    assert provenance["model_id_resolution"] == "unresolved"
    assert provenance["model_id_resolution_strategy"] == "no_match"


def test_a_malformed_reply_says_so_rather_than_naming_a_strategy(monkeypatch):
    _post(monkeypatch, _FakeResponse([]))
    _, provenance = registry.resolve_model_id("x")
    assert "malformed reply" in provenance["model_id_resolution_strategy"]


@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse(status=500),
        _FakeResponse(status=404),
        _FakeResponse(raises=ValueError("not json")),
        requests.ConnectionError("no route to host"),
        requests.Timeout("too slow"),
    ],
)
def test_a_failed_request_falls_back_instead_of_raising(monkeypatch, response):
    _post(monkeypatch, response)
    model_id, provenance = registry.resolve_model_id("gemma-3-27b-it")
    assert model_id == "gemma-3-27b-it"
    assert provenance["model_id_resolution"] == "unreachable"
    assert "model_id_resolution_error" in provenance
    assert registry.needs_review(provenance) is True


def test_the_error_detail_is_bounded(monkeypatch):
    _post(monkeypatch, requests.ConnectionError("x" * 5000))
    _, provenance = registry.resolve_model_id("y")
    assert len(provenance["model_id_resolution_error"]) <= 200


# ---------------------------------------------------------------------------
# needs_review
# ---------------------------------------------------------------------------
def test_a_confident_reviewed_canonical_needs_no_review():
    assert (
        registry.needs_review(
            {
                "model_id_resolution": "registry",
                "model_id_resolution_strategy": "exact",
                "model_id_review_status": "reviewed",
                "model_id_resolution_confidence": 1.0,
            }
        )
        is False
    )


@pytest.mark.parametrize(
    "provenance",
    [
        {"model_id_resolution": "unreachable"},
        {"model_id_resolution": "unresolved"},
        {"model_id_created_new": True},
        {"model_id_review_status": "draft"},
        {"model_id_resolution_confidence": 0.5},
    ],
)
def test_an_unverified_id_is_flagged(provenance):
    assert registry.needs_review(provenance) is True


def test_an_offline_run_is_not_flagged_per_model():
    # The caller reports opting out once, rather than on every record.
    assert registry.needs_review({"model_id_resolution": "offline"}) is False


def test_no_provenance_is_not_a_review_signal():
    assert registry.needs_review(None) is False
    assert registry.needs_review({}) is False
    assert registry.resolved_canonically(None) is False
    assert registry.resolved_canonically({}) is False
