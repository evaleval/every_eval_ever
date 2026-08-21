"""What `converters/common/metrics.py` promises to every converter that uses it.

The three harness converters each have their own tests for the records they
produce; these pin the shared rules those records rely on, so a change here shows
up as one failure with a name rather than as three unrelated ones.
"""

from every_eval_ever.converters.common.metrics import (
    CANONICAL_METRIC_IDS,
    METRIC_ID_REGISTRY_REVISION,
    count_unknown_bounds,
    metric_bounds_fields,
    metric_config_fields,
)
from every_eval_ever.converters.helm.metrics import (
    HELM_HARNESS_ID,
    HELM_METRIC_BOUNDS,
    metric_bounds_name,
    metric_parameters,
)
from every_eval_ever.converters.inspect.supplemental_eval_details import (
    SupplementalMetricConfig,
)
from every_eval_ever.converters.inspect.utils import (
    INSPECT_HARNESS_ID,
    SYNTHETIC_METRIC_CONFIG_FIELDS,
)
from every_eval_ever.converters.lm_eval.utils import (
    LM_EVAL_HARNESS_ID,
    LM_EVAL_METRIC_BOUNDS,
)
from every_eval_ever.eval_types import MetricConfig, ScoreType


def _config(metric_name: str, bounds_table=None) -> MetricConfig:
    """Build the MetricConfig a converter would build for this metric."""
    return MetricConfig(
        evaluation_description=metric_name,
        metric_name=metric_name,
        **metric_bounds_fields(metric_name, bounds_table),
    )


def _identified(
    metric_name: str,
    *,
    harness: str = 'test_harness',
    bounds_table=None,
    lookup_name: str | None = None,
    metric_parameters=None,
) -> MetricConfig:
    """Build the MetricConfig a converter builds once identity is included."""
    return MetricConfig(
        evaluation_description=metric_name,
        metric_name=metric_name,
        **metric_config_fields(
            metric_name,
            harness=harness,
            bounds_table=bounds_table,
            lookup_name=lookup_name,
            metric_parameters=metric_parameters,
        ),
    )


def test_a_known_metric_gets_its_range_and_its_direction():
    config = _config('accuracy')

    assert (config.min_score, config.max_score) == (0.0, 1.0)
    assert config.score_type == ScoreType.continuous
    assert config.lower_is_better is False
    assert config.additional_details is None


def test_an_unknown_metric_gets_no_range_at_all():
    """`min_score`/`max_score` are nullable, so "not provided" is available and true.

    [0, 1] on a metric whose scale we have not checked is not.
    """
    config = _config('semantic_similarity_v3')

    assert config.min_score is None
    assert config.max_score is None
    assert config.additional_details == {
        'polarity': 'unknown',
        'bounds_status': 'unknown',
    }


def test_a_metric_whose_definition_fixes_its_direction_says_so():
    assert _config('word_perplexity').lower_is_better is True
    assert _config('exact_match').lower_is_better is False


def test_an_unknown_direction_metric_marks_its_fallback_a_guess():
    """`lower_is_better` is required, so an unrecognized metric serializes False.

    A `polarity: unknown` marker keeps that from reading as an asserted
    higher-is-better, the same way `bounds_status` keeps a missing range from
    reading as [0, 1]. A metric whose direction its definition fixes, or one
    for which direction does not apply, is stated rather than hedged.
    """
    unknown = _config('semantic_similarity_v3')
    assert unknown.lower_is_better is False
    assert unknown.additional_details['polarity'] == 'unknown'

    # Known direction, still without a resolved range: no hedge.
    known = _config('word_perplexity')
    assert known.lower_is_better is True
    assert 'polarity' not in (known.additional_details or {})

    # A recognized higher-is-better metric is a claim, not a guess.
    accuracy = _config('accuracy')
    assert accuracy.lower_is_better is False
    assert accuracy.additional_details is None


def test_a_dispersion_metric_claims_no_direction():
    """`lower_is_better` is required, so `False` is what an inapplicable direction
    serializes as, and the marker is what carries the caveat."""
    config = _config('std')

    assert (config.min_score, config.max_score) == (0.0, float('inf'))
    assert config.additional_details == {'polarity': 'not_applicable'}


def test_the_same_metric_name_can_mean_two_scales():
    """lm-eval's `bleu` is sacrebleu's 0-100; HELM's `bleu_1` is nltk's 0-1.

    This is why the bounds are layered per harness rather than kept in one table
    keyed by a bare metric name.
    """
    lm_eval_bleu = _config('bleu', LM_EVAL_METRIC_BOUNDS)
    helm_bleu = _config('bleu_1', HELM_METRIC_BOUNDS)

    assert (lm_eval_bleu.min_score, lm_eval_bleu.max_score) == (0.0, 100.0)
    assert (helm_bleu.min_score, helm_bleu.max_score) == (0.0, 1.0)


def test_a_multi_class_brier_score_is_allowed_its_full_range():
    """A Brier score over `n` classes runs to 2.0, not to 1.0.

    lm-eval computes `mean(sum((softmax(lls) - one_hot(gold)) ** 2))`, so a model
    putting all its mass on one wrong class scores 2.0. Declaring `[0, 1]` makes
    the validator warn that a legitimate score is out of range, which the
    converter suite treats as a failure.
    """
    config = _config('brier_score')

    assert (config.min_score, config.max_score) == (0.0, 2.0)
    assert config.lower_is_better is True


def test_a_harness_table_can_override_the_shared_bounds():
    shared = _config('accuracy')
    overridden = _config('accuracy', {'accuracy': (0.0, 100.0)})

    assert shared.max_score == 1.0
    assert overridden.max_score == 100.0


def test_helm_at_k_suffix_does_not_cost_a_metric_its_bounds():
    assert metric_bounds_name('exact_match@5') == 'exact_match'
    assert metric_bounds_name('exact_match') == 'exact_match'

    config = _config(
        metric_bounds_name('quasi_exact_match@5'), HELM_METRIC_BOUNDS
    )
    assert (config.min_score, config.max_score) == (0.0, 1.0)


def test_unknown_bounds_are_countable_for_the_record_they_end_up_in():
    configs = [_config('accuracy'), _config('vibes'), _config('more_vibes')]

    assert count_unknown_bounds(configs) == 2


def test_a_registry_metric_is_published_under_its_canonical_id():
    config = _identified('accuracy')

    assert config.metric_id == 'accuracy'
    assert config.metric_kind == 'accuracy'
    assert config.metric_unit == 'proportion'
    assert config.additional_details == {
        'metric_id_registry_revision': METRIC_ID_REGISTRY_REVISION
    }


def test_a_metric_the_registry_lacks_is_namespaced_rather_than_invented():
    """A bare id claims a global identity the registry has not granted.

    The namespaced form still joins within the harness, and the marker is what
    lists the metrics owed a registry entry.
    """
    config = _identified('quasi_exact_match', harness='helm')

    assert config.metric_id == 'helm.quasi_exact_match'
    assert config.additional_details['metric_id_status'] == 'unregistered'
    # Dated, so a reader can tell whether the entry has been added since.
    assert config.additional_details['metric_id_registry_revision'] == (
        METRIC_ID_REGISTRY_REVISION
    )


def test_two_spellings_of_one_metric_join_on_the_same_id():
    """The whole point of the field: same quantity, same id, whatever it was called."""
    assert _identified('em').metric_id == _identified('exact_match').metric_id
    assert _identified('f1_score').metric_id == _identified('f1').metric_id
    assert _identified('rouge_l').metric_id == _identified('rougeL').metric_id


def test_a_parameter_in_the_name_keeps_the_bounds_but_not_the_id():
    """`exact_match@5` is exact match over the best of five completions.

    That is a different and higher quantity than `exact_match`, and the registry
    gives such metrics a slug of their own (`pass-at-1`, `recall-at-5`), so
    sharing `exact-match` would average the two in exactly the join this field
    exists for. The scale is unaffected, so the bounds still resolve.
    """
    config = _identified(
        'exact_match@5',
        harness=HELM_HARNESS_ID,
        bounds_table=HELM_METRIC_BOUNDS,
        lookup_name=metric_bounds_name('exact_match@5'),
        metric_parameters=metric_parameters('exact_match@5'),
    )

    assert config.metric_id == 'helm.exact_match@5'
    assert (config.min_score, config.max_score) == (0.0, 1.0)
    assert config.metric_kind == 'accuracy'
    assert config.metric_parameters == {'k': 5}


def test_the_unit_follows_the_scale_the_harness_actually_reported():
    """lm-eval's `bleu` is sacrebleu's 0-100, HELM's `bleu_1` nltk's 0-1.

    Both are BLEU and both say so in `metric_kind`; the unit is what keeps a
    consumer from averaging 31.4 with 0.314.
    """
    lm_eval_bleu = _identified(
        'bleu', harness=LM_EVAL_HARNESS_ID, bounds_table=LM_EVAL_METRIC_BOUNDS
    )
    helm_bleu = _identified(
        'bleu_1', harness=HELM_HARNESS_ID, bounds_table=HELM_METRIC_BOUNDS
    )

    assert lm_eval_bleu.metric_unit == 'percent'
    assert helm_bleu.metric_unit == 'proportion'
    assert lm_eval_bleu.metric_kind == helm_bleu.metric_kind == 'text_overlap'


def test_a_metric_with_no_resolved_range_claims_no_unit():
    config = _identified('semantic_similarity_v3')

    assert config.metric_unit is None
    assert config.additional_details['bounds_status'] == 'unknown'
    assert config.additional_details['metric_id_status'] == 'unregistered'


def test_identity_does_not_cost_a_dispersion_metric_its_caveat():
    config = _identified('std')

    assert config.additional_details['polarity'] == 'not_applicable'
    assert config.metric_kind == 'dispersion'


def test_the_namespace_is_the_slug_the_registry_knows_the_harness_by():
    """Renaming one of these rewrites every id that converter has ever published.

    `lm-evaluation-harness` and `helm` are the registry's own harness slugs.
    `inspect_ai` is not: the registry carries no entry for Inspect yet, so this is
    the name the converter already publishes as `eval_library.name`, and it should
    become the registry slug once that entry exists.
    """
    assert LM_EVAL_HARNESS_ID == 'lm-evaluation-harness'
    assert HELM_HARNESS_ID == 'helm'
    assert INSPECT_HARNESS_ID == 'inspect_ai'


def test_a_derived_field_can_be_corrected_by_a_caller():
    """Inspect drops a supplied `metric_config` field that it does not synthesize.

    Every field resolved from a table here is therefore listed as synthetic, or a
    caller could see a wrong id without being able to replace it. A listed field
    the strict supplement model forbids is the same dead end from the other side
    -- the allowlist waves it through but the caller can never supply it -- so
    every overridable field must also be a `SupplementalMetricConfig` field.
    """
    assert {
        'metric_id',
        'metric_kind',
        'metric_unit',
        'metric_parameters',
    } <= SYNTHETIC_METRIC_CONFIG_FIELDS
    assert SYNTHETIC_METRIC_CONFIG_FIELDS <= set(
        SupplementalMetricConfig.model_fields
    )


def test_every_canonical_id_is_shaped_like_a_registry_slug():
    """Cheap guard against a hand-written id that no registry entry can match.

    The registry's slugs are lowercase and hyphen-separated; a dot would make one
    indistinguishable from the `<harness>.<name>` fallback.
    """
    for name, metric_id in CANONICAL_METRIC_IDS.items():
        assert metric_id == metric_id.lower(), name
        assert '.' not in metric_id, name
        assert '_' not in metric_id, name
