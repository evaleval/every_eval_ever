"""Offline cover for the registry verifier.

The resolution pass needs a checkout of ``eval-card-registry``, but ``normalize``,
``registry_index`` and ``bounds_conflicts`` do not — an inline seed exercises all
three. ``normalize`` is the one most easily broken by a well-meaning edit, since
dropping ``+`` from its character class silently maps ``chrf`` onto the registry's
``chrF++``, which is a different metric.
"""

from __future__ import annotations

import pytest

pytest.importorskip('yaml')

from tools import verify_metric_ids as verifier  # noqa: E402

SEED = """
metrics:
- id: bleu
  display_name: bleu
  aliases:
  - spBLEU
  min_score: 0.0
  max_score: 1.0
- id: chrf-plus-plus
  display_name: chrF++
  min_score: 0.0
  max_score: 1.0
- id: perplexity
  min_score: 1.0
  max_score: null
"""


@pytest.fixture
def seed(tmp_path):
    path = tmp_path / 'metrics.yaml'
    path.write_text(SEED)
    return path


def test_normalize_keeps_the_plus_that_separates_two_metrics():
    assert verifier.normalize('chrF++') != verifier.normalize('chrF')
    assert verifier.normalize('chrF++') == 'chrf++'


def test_normalize_folds_case_and_separators():
    assert verifier.normalize('Exact Match') == verifier.normalize('exact_match')


def test_registry_index_maps_every_surface_form(seed):
    index = verifier.registry_index(seed)
    assert index[verifier.normalize('spBLEU')] == {'bleu'}
    assert index[verifier.normalize('chrF++')] == {'chrf-plus-plus'}
    assert verifier.normalize('chrf') not in index


def test_registry_bounds_reports_an_absent_ceiling_as_none(seed):
    assert verifier.registry_bounds(seed)['perplexity'] == (1.0, None)


def test_a_declared_scale_contradicting_its_entry_is_reported(seed, monkeypatch):
    monkeypatch.setitem(verifier.CANONICAL_METRIC_IDS, 'bleu', 'bleu')
    monkeypatch.setitem(verifier.LM_EVAL_METRIC_BOUNDS, 'bleu', (0.0, 100.0))
    found = verifier.bounds_conflicts(seed)
    assert any('bleu -> bleu' in row and '100.0' in row for row in found)


def test_an_entry_declaring_no_ceiling_constrains_nothing(seed, monkeypatch):
    monkeypatch.setitem(
        verifier.CANONICAL_METRIC_IDS, 'perplexity', 'perplexity'
    )
    monkeypatch.setitem(
        verifier.LM_EVAL_METRIC_BOUNDS, 'perplexity', (1.0, float('inf'))
    )
    assert not [r for r in verifier.bounds_conflicts(seed) if 'perplexity' in r]


def test_an_unmapped_name_is_not_compared(seed, monkeypatch):
    """A namespaced metric cites no entry, so it cannot contradict one."""
    monkeypatch.delitem(verifier.CANONICAL_METRIC_IDS, 'bleu', raising=False)
    monkeypatch.setitem(verifier.LM_EVAL_METRIC_BOUNDS, 'bleu', (0.0, 100.0))
    assert not [r for r in verifier.bounds_conflicts(seed) if 'bleu' in r]
