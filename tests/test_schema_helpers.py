"""The schema helpers refuse to guess a metric's scale.

`make_metric_config` used to default `min_score=0.0, max_score=1.0` and build
`MetricConfig` before assigning them, so the documented default path raised
rather than producing a record — the helper could not emit a continuous metric at
all. Fixing the crash without removing the default would have turned a broken
helper into a silent one, since `[0, 1]` is a claim about the scale a score was
computed on.
"""

from __future__ import annotations

import pytest

from every_eval_ever.eval_types import ScoreType
from every_eval_ever.helpers.schema import (
    make_evaluation_result,
    make_metric_config,
)


def test_a_continuous_metric_needs_both_bounds():
    with pytest.raises(ValueError, match='needs min_score and max_score'):
        make_metric_config('Accuracy on the thing')


def test_stated_bounds_reach_the_config():
    config = make_metric_config('Accuracy', min_score=0.0, max_score=100.0)
    assert (config.min_score, config.max_score) == (0.0, 100.0)
    assert config.score_type is ScoreType.continuous


def test_a_quantity_with_no_ceiling_is_expressible():
    config = make_metric_config(
        'Cost in USD',
        lower_is_better=True,
        min_score=0.0,
        max_score=float('inf'),
    )
    assert config.max_score == float('inf')


def test_a_level_based_metric_needs_no_bounds():
    config = make_metric_config(
        'Risk level',
        score_type=ScoreType.levels,
        level_names=['low', 'medium', 'high'],
        has_unknown_level=False,
    )
    assert config.level_names == ['low', 'medium', 'high']
    assert config.min_score is None


def test_the_result_helper_carries_the_same_requirement():
    """It refuses before it reaches the part of itself that is stale.

    `make_evaluation_result` never supplies `source_data`, which
    `EvaluationResult` requires, so it cannot build a result even with bounds
    given. That is a separate defect; this only pins that the missing-bounds
    refusal happens first, and names the message so the caller is told what to
    supply rather than shown a schema error.
    """
    with pytest.raises(ValueError, match='needs min_score and max_score'):
        make_evaluation_result('MMLU', 73.4, 'Accuracy')
