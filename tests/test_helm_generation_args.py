"""Tests for HELM adapter generation args extraction.

Verifies that falsy-but-valid values like temperature=0 are preserved,
not silently replaced by adapter defaults.
"""

import pytest

pytest.importorskip(
    'helm', reason='crfm-helm not installed; install with: uv sync --extra helm'
)

from types import SimpleNamespace

from every_eval_ever.converters.helm.adapter import HELMAdapter


def _make_request_state(temperature=None, max_tokens=None, top_p=None, top_k=None):
    """Build a minimal mock RequestState with the given request-level values."""
    request = SimpleNamespace(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k_per_token=top_k,
    )
    return SimpleNamespace(
        request=request,
        result=None,  # no completions → extract_reasoning returns None
    )


def _make_adapter_spec(temperature=None, max_tokens=None, top_p=None, top_k=None):
    """Build a minimal mock AdapterSpec with the given fallback values."""
    return SimpleNamespace(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k_per_token=top_k,
    )


class TestExtractGenerationArgsFalsyValues:
    """Verify that 0 is treated as a real value, not as missing."""

    def test_temperature_zero_is_preserved(self):
        adapter = HELMAdapter()
        request_state = _make_request_state(temperature=0)
        adapter_spec = _make_adapter_spec(temperature=0.7)

        args = adapter._extract_generation_args(adapter_spec, request_state)

        assert args.temperature == 0, (
            f'temperature=0 was replaced by adapter default {args.temperature}'
        )

    def test_max_tokens_low_value_is_preserved(self):
        """max_tokens has a schema constraint of ge=1, so test with 1 not 0."""
        adapter = HELMAdapter()
        request_state = _make_request_state(max_tokens=1)
        adapter_spec = _make_adapter_spec(max_tokens=100)

        args = adapter._extract_generation_args(adapter_spec, request_state)

        assert args.max_tokens == 1, (
            f'max_tokens=1 was replaced by adapter default {args.max_tokens}'
        )

    def test_top_p_zero_is_preserved(self):
        adapter = HELMAdapter()
        request_state = _make_request_state(top_p=0)
        adapter_spec = _make_adapter_spec(top_p=0.9)

        args = adapter._extract_generation_args(adapter_spec, request_state)

        assert args.top_p == 0, (
            f'top_p=0 was replaced by adapter default {args.top_p}'
        )

    def test_top_k_zero_is_preserved(self):
        adapter = HELMAdapter()
        request_state = _make_request_state(top_k=0)
        adapter_spec = _make_adapter_spec(top_k=50)

        args = adapter._extract_generation_args(adapter_spec, request_state)

        assert args.top_k == 0, (
            f'top_k=0 was replaced by adapter default {args.top_k}'
        )

    def test_none_falls_back_to_adapter_spec(self):
        """When request value is None, the adapter default should be used."""
        adapter = HELMAdapter()
        request_state = _make_request_state(
            temperature=None, max_tokens=None, top_p=None, top_k=None
        )
        adapter_spec = _make_adapter_spec(
            temperature=0.7, max_tokens=100, top_p=0.9, top_k=50
        )

        args = adapter._extract_generation_args(adapter_spec, request_state)

        assert args.temperature == 0.7
        assert args.max_tokens == 100
        assert args.top_p == 0.9
        assert args.top_k == 50
