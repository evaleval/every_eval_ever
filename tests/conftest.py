"""Global pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def allow_production_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authorize production repo references during offline test runs.

    Many existing unit tests mock the Hub and use the default datastore
    repository ('evaleval/EEE_datastore'). Setting this prevents the production
    guardrail from interfering with existing offline tests.
    """
    monkeypatch.setenv('EEE_ALLOW_PRODUCTION', 'true')
