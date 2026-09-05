"""What a metric's name tells us about its identity, its range and its polarity.

Upstream harnesses report a metric's value and almost never its range, so the
range has to come from the metric's definition. The names below mean the same
thing in every harness we convert, so their bounds are shared; anything whose
scale is harness-specific belongs in that converter's own table, layered on top
of `SHARED_METRIC_BOUNDS`. `bleu` is the cautionary example: sacrebleu (lm-eval)
reports 0-100 while nltk's `sentence_bleu` (HELM's `bleu_1`/`bleu_4`) reports
0-1, so the bare name cannot carry a range.

A metric that is in no table gets no bounds at all and a `bounds_status` marker,
because `min_score`/`max_score` are nullable and "not provided" is true, while
[0, 1] on an unbounded metric is not.

`metric_id` is the field a consumer joins on across sources, so a metric that
means the same thing in two harnesses has to arrive under the same id. That id
is the eval-card-registry's canonical metric slug, resolved once by hand into
`CANONICAL_METRIC_IDS` rather than over the network, because a converter has to
run offline and give the same answer every time. Anything the registry does not
carry gets `<harness>.<name>`, which is a stable join key within the harness and
claims no global identity; `metric_id_status` marks those so the set needing a
registry entry can be listed. Re-check the map against a registry checkout with
`uv run python -m tools.verify_metric_ids --seed <registry>/seed/metrics.yaml`.
"""

from __future__ import annotations

from every_eval_ever.eval_types import ScoreType

# Infinite bounds are serialized as the JSON strings "Infinity"/"-Infinity".
SHARED_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    'accuracy': (0.0, 1.0),
    'acc': (0.0, 1.0),
    'acc_norm': (0.0, 1.0),
    'em': (0.0, 1.0),
    'exact_match': (0.0, 1.0),
    'f1': (0.0, 1.0),
    'f1_score': (0.0, 1.0),
    'precision': (0.0, 1.0),
    'recall': (0.0, 1.0),
    'mcc': (-1.0, 1.0),
    # Not [0, 1]: the multi-class Brier score sums the squared error over every
    # class, so all the mass on one wrong class costs 2.0.
    'brier_score': (0.0, 2.0),
    # Dispersion of a score distribution, not a score: non-negative, and
    # unbounded above unless the underlying metric is bounded.
    'std': (0.0, float('inf')),
    'stddev': (0.0, float('inf')),
    'stderr': (0.0, float('inf')),
    'bootstrap_stderr': (0.0, float('inf')),
    'var': (0.0, float('inf')),
}

# Metrics whose definition fixes the direction, so no harness has to say so.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        'bits_per_byte',
        'brier_score',
        'byte_perplexity',
        'calibration_error',
        'cer',
        'ece',
        'perplexity',
        'ter',
        'wer',
        'word_perplexity',
    }
)

# Metrics that summarize the spread of a score distribution. "Better" does not
# apply to them, and the schema has fields for them on the score they describe
# (`uncertainty.standard_error`, `uncertainty.standard_deviation`), so a
# converter should prefer routing them there over emitting them as scores.
DISPERSION_METRICS: frozenset[str] = frozenset(
    {'std', 'stddev', 'stderr', 'bootstrap_stderr', 'var'}
)

# The eval-card-registry commit these ids were resolved against, on that repo's
# main branch so anyone can check out the state that produced them. Bump it with
# the map.
METRIC_ID_REGISTRY_REVISION = '8b83e9c'

# Harness metric name -> canonical registry metric id, matched case- and
# separator-insensitively against each entry's id, display_name and aliases.
# Only names the registry actually carries appear here; the rest are namespaced.
CANONICAL_METRIC_IDS: dict[str, str] = {
    'acc': 'accuracy',
    # Length-normalized accuracy is a different computation from `acc` on the same
    # items, and the registry keeps them apart. It is carried as
    # `normalized-accuracy` there, with `acc_norm` already among its aliases; the
    # hosted resolver reports no match only because the live Space lags the seed,
    # which is why this looked unregistered.
    'acc_norm': 'normalized-accuracy',
    'accuracy': 'accuracy',
    'bleu': 'bleu',
    'bleu_1': 'bleu-1',
    'bleu_4': 'bleu-4',
    'cer': 'cer',
    'em': 'exact-match',
    'exact_match': 'exact-match',
    'f1': 'f1',
    'f1_score': 'f1',
    # lm-eval v0.3 reports perplexity under `ppl` (v0.4 renamed it), so both
    # spellings reach the one registry metric rather than fragmenting by format.
    'ppl': 'perplexity',
    'perplexity': 'perplexity',
    'precision': 'precision',
    'recall': 'recall',
    'rouge1': 'rouge-1',
    'rouge2': 'rouge-2',
    'rougeL': 'rouge-l',
    'rouge_l': 'rouge-l',
    'wer': 'wer',
}

# The family a metric aggregates safely within. Coarse on purpose: it says two
# numbers are the same kind of quantity, not that they are comparable, which is
# `evaluation_name`'s business. A metric whose family we would have to invent is
# absent rather than guessed.
METRIC_KINDS: dict[str, str] = {
    'acc': 'accuracy',
    'acc_norm': 'accuracy',
    'accuracy': 'accuracy',
    'chain_of_thought_correctness': 'accuracy',
    'em': 'accuracy',
    'exact_match': 'accuracy',
    'ifeval_strict_accuracy': 'accuracy',
    'math_equiv': 'accuracy',
    'math_equiv_chain_of_thought': 'accuracy',
    'mc1': 'accuracy',
    'prefix_exact_match': 'accuracy',
    'quasi_exact_match': 'accuracy',
    'quasi_prefix_exact_match': 'accuracy',
    'classification_macro_f1': 'f1',
    'classification_micro_f1': 'f1',
    'f1': 'f1',
    'f1_score': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'mcc': 'correlation',
    'brier_score': 'brier_score',
    'calibration_error': 'calibration_error',
    'ece': 'calibration_error',
    'bleu': 'text_overlap',
    'bleu_1': 'text_overlap',
    'bleu_4': 'text_overlap',
    'chrf': 'text_overlap',
    'rouge1': 'text_overlap',
    'rouge2': 'text_overlap',
    'rougeL': 'text_overlap',
    'rougeLsum': 'text_overlap',
    'rouge_l': 'text_overlap',
    'cer': 'edit_distance',
    'ter': 'edit_distance',
    'wer': 'edit_distance',
    'byte_perplexity': 'perplexity',
    'perplexity': 'perplexity',
    'word_perplexity': 'perplexity',
    'bits_per_byte': 'compression_rate',
    'std': 'dispersion',
    'stddev': 'dispersion',
    'stderr': 'dispersion',
    'var': 'dispersion',
    'bootstrap_stderr': 'dispersion',
}

# A unit that does not follow from the metric's resolved range. Everything on
# exactly [0, 1] is a proportion and everything on exactly [0, 100] a percent,
# which is derived rather than listed, so the same name reports the unit of
# whichever harness table won: lm-eval's sacrebleu `bleu` is a percent and
# HELM's nltk `bleu_1` a proportion, from their bounds alone. Listed here are
# the metrics whose scale their bounds cannot show, such as a sacrebleu metric
# that is percent-scaled but may exceed 100.
METRIC_UNITS: dict[str, str] = {
    'bits_per_byte': 'bits_per_byte',
    'chrf': 'percent',
    'ter': 'percent',
}

_PROPORTION_BOUNDS = (0.0, 1.0)
_PERCENT_BOUNDS = (0.0, 100.0)

_UNKNOWN_BOUNDS = {'bounds_status': 'unknown'}
_NO_POLARITY = {'polarity': 'not_applicable'}
_UNKNOWN_POLARITY = {'polarity': 'unknown'}
_UNREGISTERED_ID = {'metric_id_status': 'unregistered'}
# Both a resolved id and an unresolved one are claims about a registry state, so
# each says which state it was: an `unregistered` marker with no revision beside
# it does not tell a reader whether the entry has since been added.
_ID_REVISION = {'metric_id_registry_revision': METRIC_ID_REGISTRY_REVISION}


def metric_bounds_fields(
    metric_name: str | None,
    bounds_table: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """The `MetricConfig` fields that describe one metric's range and direction.

    Spread into a `MetricConfig(...)` call. `bounds_table` overrides and extends
    the shared bounds for harnesses that spell a metric differently or report it
    on a different scale.
    """
    table = (
        {**SHARED_METRIC_BOUNDS, **bounds_table}
        if bounds_table
        else SHARED_METRIC_BOUNDS
    )
    name = metric_name or ''
    bounds = table.get(name)
    details = dict(_NO_POLARITY) if name in DISPERSION_METRICS else {}

    if bounds is None:
        # `lower_is_better` is required, so an unrecognized metric defaults to
        # False; for one whose direction we cannot resolve that default is a
        # guess a ranking consumer would misread as an asserted "higher is
        # better", so it is marked like an unresolved range. A known direction
        # (LOWER_IS_BETTER) or an inapplicable one (dispersion) keeps its answer.
        if name not in LOWER_IS_BETTER and name not in DISPERSION_METRICS:
            details = {**details, **_UNKNOWN_POLARITY}
        return {
            'lower_is_better': name in LOWER_IS_BETTER,
            'additional_details': {**details, **_UNKNOWN_BOUNDS},
        }
    return {
        'lower_is_better': name in LOWER_IS_BETTER,
        'score_type': ScoreType.continuous,
        'min_score': bounds[0],
        'max_score': bounds[1],
        'additional_details': details or None,
    }


def metric_unit(
    metric_name: str | None, bounds: tuple[float, float] | None
) -> str | None:
    """The unit a metric's values are expressed in, where its scale states it."""
    listed = METRIC_UNITS.get(metric_name or '')
    if listed:
        return listed
    if bounds == _PROPORTION_BOUNDS:
        return 'proportion'
    if bounds == _PERCENT_BOUNDS:
        return 'percent'
    return None


def metric_config_fields(
    metric_name: str | None,
    *,
    harness: str,
    bounds_table: dict[str, tuple[float, float]] | None = None,
    lookup_name: str | None = None,
    metric_parameters: dict[str, str | float | bool | None] | None = None,
) -> dict[str, object]:
    """The `MetricConfig` fields that follow from one metric's name.

    Spread into a `MetricConfig(...)` call alongside the fields only the harness
    knows. `lookup_name` is the name the tables are keyed on, for a harness that
    decorates the reported name with something that does not change the metric
    (HELM's `exact_match@5`); `metric_name` is still what gets published.
    """
    reported = metric_name or ''
    name = lookup_name if lookup_name is not None else reported
    table = (
        {**SHARED_METRIC_BOUNDS, **bounds_table}
        if bounds_table
        else SHARED_METRIC_BOUNDS
    )
    bounds = table.get(name)
    # A parameter the harness spelled into the name leaves the metric's scale
    # alone but not its identity: HELM's `exact_match@5` is exact match over the
    # best of five completions, which is a different and higher quantity than
    # `exact_match`, and the registry gives such a metric a slug of its own
    # (`pass-at-1`, `recall-at-5`). So the range comes from the undecorated name
    # while the id may not, or a join on `exact-match` would average the two.
    canonical = CANONICAL_METRIC_IDS.get(name) if name == reported else None

    fields = metric_bounds_fields(name, bounds_table)
    details = dict(fields.pop('additional_details') or {})
    if reported:
        details.update(_ID_REVISION)
        if canonical is None:
            details.update(_UNREGISTERED_ID)

    return {
        **fields,
        'metric_id': (canonical or f'{harness}.{reported}')
        if reported
        else None,
        'metric_kind': METRIC_KINDS.get(name),
        'metric_unit': metric_unit(name, bounds),
        'metric_parameters': metric_parameters or None,
        'additional_details': details or None,
    }


def count_unknown_bounds(metric_configs) -> int:
    """How many of these metric configs have no known range."""
    return sum(
        config.additional_details is not None
        and config.additional_details.get('bounds_status') == 'unknown'
        for config in metric_configs
    )
