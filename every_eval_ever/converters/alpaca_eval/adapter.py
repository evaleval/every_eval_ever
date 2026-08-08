"""Adapter for converting AlpacaEval leaderboard CSVs to every_eval_ever format.

The leaderboard CSVs hold the scores; everything that makes those scores
interpretable — the judge and its prompt, the baseline, the harness version and
each entry's real model identity and generation settings — lives in sibling
files of the same upstream repository. This adapter reads all of it at one
pinned git ref (see :mod:`.upstream`) and resolves model identity from the
upstream model configs (see :mod:`.identity`) instead of guessing from slugs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.alpaca_eval import identity as identity_mod
from every_eval_ever.converters.alpaca_eval.upstream import (
    DEFAULT_UPSTREAM_REF,
    UPSTREAM_REPO,
    UPSTREAM_URL,
    LeaderboardSnapshot,
    UpstreamSnapshot,
    annotator_config_path,
    blob_url,
    model_config_path,
    model_prompt_path,
    populate_snapshot,
    raw_url,
)
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    JudgeConfig,
    LlmScoring,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    SourceType,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import eval_card_registry as registry_mod
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
)

# ---------------------------------------------------------------------------
# The evaluated dataset
# ---------------------------------------------------------------------------

#: Both leaderboards score the same 805 instructions; the configs differ only in
#: whose outputs are the baseline (verified against the published HF dataset).
HF_DATASET_REPO = 'tatsu-lab/alpaca_eval'
HF_DATASET_SPLIT = 'eval'
HF_DATASET_SAMPLES = 805

#: Who runs the leaderboard. The same organization also has models on it, so
#: this is compared against each row's developer rather than assumed distinct.
EVALUATOR_ORG = 'tatsu-lab'
EVALUATOR_ORG_NAME = 'Tatsu Lab (Stanford University)'

# ---------------------------------------------------------------------------
# Leaderboard configurations
# ---------------------------------------------------------------------------

LEADERBOARDS: Dict[str, Dict[str, Any]] = {
    'v1': {
        'source_name': 'AlpacaEval 1.0 Leaderboard',
        'leaderboard_version': '1.0',
        'collection': 'alpaca_eval_v1',
        #: Name looked up in the eval-card-registry, and the local fallback used
        #: when it has no canonical benchmark for this leaderboard.
        'benchmark_query': 'AlpacaEval 1.0',
        'evaluation_name': 'alpaca_eval.v1',
        'csv_path': (
            'src/alpaca_eval/leaderboards/data_AlpacaEval/'
            'alpaca_eval_gpt4_leaderboard.csv'
        ),
        'annotator': 'alpaca_eval_gpt4',
        'baseline': 'text_davinci_003',
        'hf_config': 'alpaca_eval',
        'preference_rule': (
            "the judge ranks the two outputs and the preferred one counts as a "
            "full win (ties count as half)"
        ),
        # Back-compat aliases used by the CLI and by older callers.
        'version': '1.0',
    },
    'v2': {
        'source_name': 'AlpacaEval 2.0 Leaderboard',
        'leaderboard_version': '2.0',
        'collection': 'alpaca_eval_v2',
        'benchmark_query': 'AlpacaEval 2.0',
        'evaluation_name': 'alpaca_eval.v2',
        'csv_path': (
            'src/alpaca_eval/leaderboards/data_AlpacaEval_2/'
            'weighted_alpaca_eval_gpt4_turbo_leaderboard.csv'
        ),
        'annotator': 'weighted_alpaca_eval_gpt4_turbo',
        'baseline': 'gpt4_turbo',
        'hf_config': 'alpaca_eval_gpt4_baseline',
        'preference_rule': (
            "each comparison is weighted by the judge's token probability of "
            'preferring the model, so a single preference contributes a value '
            'in [0, 1] rather than 0 or 1'
        ),
        'version': '2.0',
    },
}

for _version, _cfg in LEADERBOARDS.items():
    #: Absolute URL of the leaderboard CSV at the default pinned ref.
    _cfg['url'] = raw_url(_cfg['csv_path'])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetricSpec:
    """One leaderboard column, and how it becomes an ``EvaluationResult``.

    Bounds are not declared here. The eval-card-registry is what settles the
    scale a metric is published on — its canonical ``win-rate`` is declared on
    ``[0, 100]``, the scale the CSV already uses — so the fields below are only
    the fallback for a column the registry has no canonical for. See
    :func:`_resolve_metrics`.
    """

    column: str
    #: Query passed to the registry, and the local id used if it has no answer.
    metric_name: str
    metric_kind: str
    metric_unit: str
    description: str
    #: Upper bound to declare when the registry has no canonical entry.
    #: ``float('inf')`` for an unbounded column: a mean character count has no
    #: maximum, and the schema encodes that as the string ``"Infinity"`` on the
    #: wire (see ``MetricConfig.validate_bound_wire_type``) rather than as an
    #: invented ceiling.
    fallback_max: float = 100.0
    #: Largest value the raw CSV cell can take, used to rescale onto the
    #: declared bounds if the registry ever publishes this metric on a
    #: different scale. ``None`` for an unbounded column.
    source_max: Optional[float] = 100.0
    se_column: Optional[str] = None
    judge_scored: bool = True


_WIN_RATE_METRICS = (
    _MetricSpec(
        column='win_rate',
        metric_name='win_rate',
        metric_kind='win_rate',
        metric_unit='percent',
        description=(
            'Share of the {samples} AlpacaEval instructions on which the '
            '{annotator} judge preferred this model\'s output over the '
            '{baseline} baseline: {preference_rule}.'
        ),
        se_column='standard_error',
    ),
    _MetricSpec(
        column='length_controlled_winrate',
        metric_name='length_controlled_win_rate',
        metric_kind='win_rate',
        metric_unit='percent',
        description=(
            'Length-controlled win rate against the {baseline} baseline: a '
            'logistic regression of the {annotator} judge\'s preferences on '
            'model identity, instruction difficulty and output-length '
            'difference, evaluated at zero length difference, so the score '
            'estimates the win rate the model would obtain if its outputs '
            'were as long as the baseline\'s (Dubois et al., 2024).'
        ),
        se_column='lc_standard_error',
    ),
    _MetricSpec(
        column='discrete_win_rate',
        metric_name='discrete_win_rate',
        metric_kind='win_rate',
        metric_unit='percent',
        description=(
            'Win rate against the {baseline} baseline computed from '
            'binarized preferences: each of the {annotator} judge\'s '
            'preferences is rounded to a win, a loss or a tie before '
            'averaging, so no partial credit is carried over.'
        ),
    ),
)

_AVG_LENGTH_METRIC = _MetricSpec(
    column='avg_length',
    metric_name='avg_length',
    metric_kind='length',
    metric_unit='characters',
    description=(
        'Mean length of the model\'s outputs in **characters** (upstream '
        'computes `model_outputs["output"].str.len().mean()`). Reported for '
        'length-bias context, not as a quality score.'
    ),
    fallback_max=float('inf'),
    source_max=None,
    judge_scored=False,
)

METRIC_SPECS = _WIN_RATE_METRICS + (_AVG_LENGTH_METRIC,)

#: Namespace for a metric the registry has no canonical entry for. Prefixed so
#: it cannot be mistaken for a registry id, which is a bare kebab-case slug.
LOCAL_METRIC_PREFIX = 'alpaca_eval'


@dataclass(frozen=True)
class _ResolvedMetric:
    """A metric spec whose identity and bounds the registry has settled."""

    spec: _MetricSpec
    metric_id: str
    min_score: float
    max_score: float
    lower_is_better: bool
    resolution: registry_mod.Resolution

    @property
    def scale(self) -> float:
        """Divide a raw CSV cell by this to land on the declared bounds.

        ``1.0`` for every metric as the registry currently stands, because its
        ``win-rate`` is declared on ``[0, 100]`` and that is what the CSV holds.
        The division exists so that a registry that later publishes win rate as
        a proportion moves the scores with it instead of putting ``95.3`` in a
        field bounded at ``1.0``.
        """
        if self.spec.source_max is None:
            return 1.0
        if not math.isfinite(self.max_score) or self.max_score <= 0:
            return 1.0
        return self.spec.source_max / self.max_score


def _resolve_metrics(
    registry: registry_mod.Registry,
) -> Tuple[_ResolvedMetric, ...]:
    """Attach a canonical id and bounds to every metric this adapter emits.

    A column the registry cannot place keeps a local ``alpaca_eval.*`` id and
    this adapter's fallback bounds, and says so through ``metric_registry_*`` in
    ``additional_details``, rather than being dropped — three of the four
    AlpacaEval columns have no canonical yet, including the length-controlled
    win rate the benchmark is known for.
    """
    resolved = []
    for spec in METRIC_SPECS:
        resolution = registry.metric(spec.column)
        record = resolution.record if resolution.resolved else {}
        min_score = record.get('min_score')
        max_score = record.get('max_score')
        resolved.append(
            _ResolvedMetric(
                spec=spec,
                metric_id=(
                    resolution.canonical_id
                    or f'{LOCAL_METRIC_PREFIX}.{spec.metric_name}'
                ),
                min_score=0.0 if min_score is None else float(min_score),
                max_score=(
                    spec.fallback_max if max_score is None else float(max_score)
                ),
                lower_is_better=bool(record.get('lower_is_better') or False),
                resolution=resolution,
            )
        )
    return tuple(resolved)

#: Raw comparison counts kept alongside the win-rate scores.
_COUNT_COLUMNS = ('n_wins', 'n_wins_base', 'n_draws', 'n_total')

_MODEL_COLUMNS = ('', 'Unnamed: 0', 'model', 'Model')

#: Placeholder entry from the "Cheating on leaderboards" study: a constant
#: string submitted to show the judge can be gamed, not a model.
_NULL_MODEL_RE = re.compile(r'null.?model', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> Optional[float]:
    """Parse a CSV cell as a finite number, or ``None``.

    ``float`` accepts ``'nan'``, ``'inf'`` and ``'1e999'``. None of them is a
    score, and a non-finite one reaches the published record as either a bound
    violation or a bare ``NaN`` token that is not valid JSON.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> Optional[int]:
    number = _to_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _metric_cell_failure(
    metrics: Tuple['_ResolvedMetric', ...], row: Dict[str, str]
) -> Optional[str]:
    """Say why a row's metric cells cannot be published, or ``None``.

    An absent cell is not a problem — a leaderboard version with no
    ``discrete_win_rate`` column simply publishes fewer results. A *populated*
    cell that is not a finite number inside its declared bounds is, and the row
    loop drops that row with a reason rather than publishing the headline metric
    silently missing (``'n/a'`` parses as nothing).

    The cells that qualify a score answer to the same rule, because parsing them
    leniently makes a populated one look absent: an unreadable standard error
    publishes as no uncertainty at all, and an unreadable or non-positive
    ``n_total`` as the dataset's 805 — a denominator the row never claimed.
    """
    primary = metrics[0]
    if not _cell(row, primary.spec.column):
        return f'missing {primary.spec.column}'
    for metric in metrics:
        raw = _cell(row, metric.spec.column)
        if not raw:
            continue
        value = _to_float(raw)
        if value is None:
            return f'{metric.spec.column} is not a finite number: {raw!r}'
        score = value / metric.scale if metric.scale != 1.0 else value
        if not metric.min_score <= score <= metric.max_score:
            return (
                f'{metric.spec.column} of {raw!r} is outside the '
                f'[{metric.min_score}, {metric.max_score}] the registry '
                f'declares for {metric.metric_id}'
            )
        # Checked with its metric: a standard error whose score is absent never
        # reaches a record, so it cannot be a reason to drop the row.
        error_raw = (
            _cell(row, metric.spec.se_column) if metric.spec.se_column else ''
        )
        if error_raw:
            error = _to_float(error_raw)
            if error is None or error < 0:
                return (
                    f'{metric.spec.se_column} is not a usable standard error: '
                    f'{error_raw!r}'
                )
    denominator = _cell(row, 'n_total')
    samples = _to_int(denominator)
    if denominator and (samples is None or samples <= 0):
        return f'n_total is not a positive count: {denominator!r}'
    return None


def _cell(row: Dict[str, str], column: str) -> str:
    value = row.get(column)
    return value.strip() if isinstance(value, str) else ''


def model_slug_from_row(row: Dict[str, str]) -> str:
    """Extract the leaderboard slug from a CSV row (first/unnamed column)."""
    for key in _MODEL_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fallback = next(iter(row.values()), '')
    return fallback.strip() if isinstance(fallback, str) else ''


def model_slugs(rows: Iterable[Dict[str, str]]) -> List[str]:
    """Return the leaderboard slugs of *rows*, in order, without blanks."""
    return [slug for slug in (model_slug_from_row(r) for r in rows) if slug]


def _stringify(values: Dict[str, Any]) -> Dict[str, str]:
    """Coerce a details mapping to the schema's ``dict[str, str]``."""
    out: Dict[str, str] = {}
    for key, value in values.items():
        if value is None or value == '':
            continue
        out[key] = (
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True, default=str)
        )
    return out


# ---------------------------------------------------------------------------
# Judge / metric construction
# ---------------------------------------------------------------------------


def _judge_model_info(board: LeaderboardSnapshot, annotator: str) -> ModelInfo:
    """Build the judge's ``ModelInfo`` from the upstream annotator config."""
    judge_identity = identity_mod.resolve_identity(
        annotator, board.annotator_config
    )
    judge_name = (
        identity_mod.upstream_model_name(board.annotator_config) or annotator
    )
    if judge_identity is None:  # pragma: no cover - defensive
        return ModelInfo(
            name=judge_name,
            id=judge_name,
            additional_details=_stringify(
                {'deployment_type': 'unknown', 'model_availability': 'unknown'}
            ),
        )
    return ModelInfo(
        name=judge_name,
        id=judge_identity.model_id,
        developer=judge_identity.developer,
        inference_platform=judge_identity.inference_platform,
        additional_details=_stringify(
            {
                'deployment_type': judge_identity.deployment_type,
                'model_availability': judge_identity.model_availability,
                'annotator_config': annotator,
            }
        ),
    )


def _llm_scoring(
    board: LeaderboardSnapshot, cfg: Dict[str, Any], ref: str
) -> LlmScoring:
    """Build ``llm_scoring`` for the judge-scored metrics of one leaderboard.

    ``aggregation_method`` is deliberately left unset: there is one judge per
    comparison, and neither upstream rule (v1's ranking parser, v2's
    probability-weighted logprob parser) is one of the schema's enum values.
    The real rule is recorded in ``additional_details`` instead.
    """
    annotator = cfg['annotator']
    kwargs = identity_mod.completions_kwargs(board.annotator_config)
    annotator_path = annotator_config_path(annotator)
    judge_details = {
        'annotator_config': annotator,
        'annotator_config_url': blob_url(annotator_path, ref),
        'completion_parser': board.annotator_config.get(
            'fn_completion_parser'
        ),
        'completion_parser_kwargs': board.annotator_config.get(
            'completion_parser_kwargs'
        ),
        'max_tokens': kwargs.get('max_tokens'),
        'top_logprobs': kwargs.get('top_logprobs'),
    }
    return LlmScoring(
        judges=[
            JudgeConfig(
                model_info=_judge_model_info(board, annotator),
                temperature=_to_float(kwargs.get('temperature')),
                additional_details=_stringify(judge_details),
            )
        ],
        input_prompt=board.judge_prompt,
        additional_details=_stringify(
            {
                'baseline_model': cfg['baseline'],
                'preference_rule': cfg['preference_rule'],
                'prompt_template_path': board.judge_prompt_path,
                'prompt_template_url': blob_url(board.judge_prompt_path, ref),
            }
        ),
    )


def _metric_config(
    metric: _ResolvedMetric,
    cfg: Dict[str, Any],
    benchmark: registry_mod.Resolution,
    llm_scoring: Optional[LlmScoring],
    samples: int,
) -> MetricConfig:
    """Build ``metric_config`` for one metric of one leaderboard row.

    Args:
        samples: How many instructions this row's score is actually over. Not
            always the dataset's 805 — a row whose ``n_total`` is lower was
            judged on fewer, and a description that claims 805 anyway misstates
            the denominator of a published win rate.
    """
    spec = metric.spec
    return MetricConfig(
        evaluation_description=spec.description.format(
            samples=samples,
            annotator=cfg['annotator'],
            baseline=cfg['baseline'],
            preference_rule=cfg['preference_rule'],
        ),
        metric_id=metric.metric_id,
        metric_name=spec.metric_name,
        metric_kind=spec.metric_kind,
        metric_unit=spec.metric_unit,
        metric_parameters={
            'baseline_model': cfg['baseline'],
            'annotator': cfg['annotator'],
            'leaderboard_version': cfg['leaderboard_version'],
        },
        lower_is_better=metric.lower_is_better,
        score_type=ScoreType.continuous,
        min_score=metric.min_score,
        max_score=metric.max_score,
        llm_scoring=llm_scoring if spec.judge_scored else None,
        additional_details=_stringify(
            {
                'source_column': spec.column,
                'source_scale': spec.metric_unit,
                # 1.0 unless the registry moves the metric's bounds; recorded so
                # a reader can tell a rescaled score from a verbatim one.
                'score_scale_divisor': metric.scale,
                'direction': (
                    'higher_is_better'
                    if spec.judge_scored
                    else 'not a quality metric: neither direction is better'
                ),
                **metric.resolution.provenance('metric'),
                **benchmark.provenance('benchmark'),
            }
        ),
    )


def _score_details(
    metric: _ResolvedMetric, row: Dict[str, str], value: float
) -> ScoreDetails:
    spec = metric.spec
    scale = metric.scale
    score = value / scale if scale != 1.0 else value
    standard_error = (
        _to_float(_cell(row, spec.se_column)) if spec.se_column else None
    )
    n_total = _to_int(_cell(row, 'n_total'))
    uncertainty = None
    if spec.judge_scored and (standard_error is not None or n_total):
        uncertainty = Uncertainty(
            standard_error=(
                StandardError(
                    value=standard_error / scale,
                    # Upstream reports pandas' `.sem()` of the per-instruction
                    # preferences (times 100), i.e. an analytic standard error
                    # of the mean — not a bootstrap estimate.
                    method='analytic',
                )
                if standard_error is not None
                else None
            ),
            num_samples=n_total,
        )
    details = {f'source_{spec.column}': value}
    if spec.judge_scored:
        # Verbatim: these are provenance and are stringified either way, so
        # parsing them could only drop a value it failed to read.
        details.update(
            {column: _cell(row, column) for column in _COUNT_COLUMNS}
        )
    return ScoreDetails(
        score=score, uncertainty=uncertainty, details=_stringify(details)
    )


def _source_data(cfg: Dict[str, Any], ref: str) -> SourceDataHf:
    return SourceDataHf(
        dataset_name=cfg['collection'],
        source_type='hf_dataset',
        hf_repo=HF_DATASET_REPO,
        hf_split=HF_DATASET_SPLIT,
        samples_number=HF_DATASET_SAMPLES,
        additional_details=_stringify(
            {
                'hf_config': cfg['hf_config'],
                'baseline_model': cfg['baseline'],
                'leaderboard_csv_url': raw_url(cfg['csv_path'], ref),
            }
        ),
    )


def _evaluation_results(
    row: Dict[str, str],
    cfg: Dict[str, Any],
    board: LeaderboardSnapshot,
    ref: str,
    evaluation_id: str,
    generation_config: Optional[GenerationConfig],
    metrics: Tuple[_ResolvedMetric, ...],
    benchmark: registry_mod.Resolution,
    evaluation_name: str,
) -> List[EvaluationResult]:
    """Build every result a leaderboard row supports (missing columns skipped).

    Every populated cell here has already passed :func:`_metric_cell_failure`,
    so a skipped column is an absent one, never a rejected value.
    """
    llm_scoring = _llm_scoring(board, cfg, ref)
    source_data = _source_data(cfg, ref)
    # Only an absent n_total falls back: a populated one that is not a positive
    # count fails the row above, so 805 is never put in place of a real value.
    samples = _to_int(_cell(row, 'n_total')) or HF_DATASET_SAMPLES
    results = []
    for metric in metrics:
        value = _to_float(_cell(row, metric.spec.column))
        if value is None:
            continue
        results.append(
            EvaluationResult(
                # Keyed on the local metric name, never on the resolved
                # canonical: a canonical id can be renamed or merged upstream,
                # and an identifier that moves with it is not an identifier.
                evaluation_result_id=(
                    f'{evaluation_id}/{metric.spec.metric_name}'
                ),
                evaluation_name=evaluation_name,
                metric_config=_metric_config(
                    metric, cfg, benchmark, llm_scoring, samples
                ),
                score_details=_score_details(metric, row, value),
                source_data=source_data,
                generation_config=generation_config,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Model info / generation config
# ---------------------------------------------------------------------------

_MAX_TOKEN_KEYS = ('max_tokens', 'max_new_tokens')
_TYPED_GENERATION_KEYS = frozenset(
    {'temperature', 'top_p', 'top_k', 'model_name', *_MAX_TOKEN_KEYS}
)


def _model_info(
    resolved: identity_mod.ModelIdentity,
    ref: str,
    config_missing: Optional[str],
    developer: registry_mod.Resolution,
) -> ModelInfo:
    """Assemble ``model_info``, with the registry naming the organization.

    ``id`` is the repo id that resolves on HuggingFace today; ``developer``
    carries the registry's canonical organization id, which is a different string
    by design (``meta-llama/…`` published by ``meta``, ``Qwen/…`` by
    ``alibaba``), so the id's prefix is kept in ``additional_details`` rather than
    overwritten. A renamed repo can disagree on the *organization* too
    (``WizardLMTeam/…`` published by ``wizardlm``), and
    ``model_id_as_referenced`` then records the spelling the source used.

    ``raw_model_id`` and ``raw_model_namespace`` are what the source called this
    model, under the names the model-registry proposal reserves, so the registry
    can be re-resolved against them without refetching upstream.
    """
    engine = (
        InferenceEngine(name=resolved.inference_engine)
        if resolved.inference_engine
        else None
    )
    return ModelInfo(
        name=resolved.slug,
        id=resolved.model_id,
        developer=developer.canonical_id or resolved.developer,
        inference_platform=resolved.inference_platform,
        inference_engine=engine,
        additional_details=_stringify(
            {
                # Both axes are validated enums; the library would otherwise
                # silently fill them with 'unknown'.
                'deployment_type': resolved.deployment_type,
                'model_availability': resolved.model_availability,
                'deployment_evidence': resolved.deployment_evidence,
                'identity_source': resolved.identity_source,
                'raw_model_id': resolved.model_id,
                'raw_model_namespace': resolved.model_id.split('/')[0],
                **developer.provenance('developer'),
                'leaderboard_slug': resolved.slug,
                'model_id_as_referenced': resolved.model_id_as_referenced,
                'pretty_name': resolved.pretty_name,
                'upstream_model_name': resolved.upstream_model_name,
                'upstream_config_url': blob_url(
                    model_config_path(resolved.slug), ref
                ),
                'model_reference_link': resolved.reference_link,
                'link_evidence': resolved.link_evidence,
                'upstream_config_status': config_missing,
            }
        ),
    )


def _evaluator_relationship(
    developer: registry_mod.Resolution,
    evaluator: registry_mod.Resolution,
) -> EvaluatorRelationship:
    """Say whether the organization that ran the eval also built the model.

    Tatsu Lab's own models are on Tatsu Lab's leaderboard — ``alpaca-7b`` and the
    two AlpacaFarm PPO checkpoints — so a blanket ``third_party`` would claim
    independent evaluation for entries where there is none.

    Canonical ids decide it wherever the registry has both, so two spellings of
    one organization cannot read as two. ``tatsu-lab`` has no canonical entry
    today, hence the normalized-spelling fallback.
    """
    left = developer.canonical_id or developer.raw_value
    right = evaluator.canonical_id or evaluator.raw_value
    if developer.canonical_id and evaluator.canonical_id:
        same = left == right
    else:
        same = bool(registry_mod.normalize(left)) and (
            registry_mod.normalize(left) == registry_mod.normalize(right)
        )
    return (
        EvaluatorRelationship.first_party
        if same
        else EvaluatorRelationship.third_party
    )


def _generation_config(
    config: Optional[Dict[str, Any]],
    ref: str,
    prompts: Optional[Dict[str, str]] = None,
    missing_prompts: Optional[Dict[str, str]] = None,
) -> Optional[GenerationConfig]:
    """Map the upstream completion kwargs onto ``generation_config``.

    Args:
        config: The model's upstream ``configs.yaml`` body.
        ref: Pinned upstream revision, for the provenance URLs.
        prompts: ``models_configs``-relative template path -> verbatim text, as
            recorded on the snapshot. ``prompt_template`` carries the text, so a
            reader can see the prompt without refetching upstream; the path and
            URL stay in ``additional_details``.
        missing_prompts: Same keys, with why the text is absent. A snapshot saved
            before prompt text was recorded has neither dict, and leaves the
            typed value unset with ``prompt_template_status`` saying so.
    """
    if not config:
        return None
    kwargs = identity_mod.completions_kwargs(config)
    max_tokens = None
    for key in _MAX_TOKEN_KEYS:
        candidate = _to_int(kwargs.get(key))
        if candidate is not None and candidate >= 1:
            max_tokens = candidate
            break
    template_path = config.get('prompt_template')
    template_path = (
        template_path if isinstance(template_path, str) else None
    )
    prompt_text = (prompts or {}).get(template_path) if template_path else None
    extra = {
        key: value
        for key, value in kwargs.items()
        if key not in _TYPED_GENERATION_KEYS
    }
    args = GenerationArgs(
        temperature=_to_float(kwargs.get('temperature')),
        top_p=_to_float(kwargs.get('top_p')),
        top_k=_to_float(kwargs.get('top_k')),
        max_tokens=max_tokens,
        prompt_template=prompt_text,
    )
    details = {
        'fn_completions': identity_mod.completions_fn(config),
        'upstream_completions_kwargs': extra or None,
    }
    if template_path:
        # The path distinguishes same-model variants (e.g. the `_concise` /
        # `_verbose` entries) that share prompt text, so keep it alongside.
        details['prompt_template_path'] = template_path
        details['prompt_template_url'] = blob_url(
            model_prompt_path(template_path), ref
        )
        if prompt_text is None:
            details['prompt_template_status'] = (missing_prompts or {}).get(
                template_path, 'not recorded in this snapshot'
            )
    return GenerationConfig(
        generation_args=args, additional_details=_stringify(details)
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AlpacaEvalAdapter:
    """Converts AlpacaEval leaderboard CSV rows into EvaluationLog objects."""

    def __init__(
        self,
        ref: str = DEFAULT_UPSTREAM_REF,
        snapshot: Optional[UpstreamSnapshot] = None,
        max_workers: int = 8,
        registry: Optional[registry_mod.Registry] = None,
    ) -> None:
        """
        Args:
            ref: Upstream git ref to convert from. Pinning a commit keeps
                ``evaluation_id`` stable across reruns.
            snapshot: Pre-fetched upstream payload (``--input-json``). When
                given, the adapter performs no network access and *ref* comes
                from the snapshot.
            max_workers: Parallel connections used for per-model configs.
            registry: eval-card-registry resolver for organization, metric and
                benchmark ids. Defaults to the vendored offline snapshot, so a
                conversion needs no registry access and cannot write to one.
        """
        self.snapshot = snapshot if snapshot is not None else UpstreamSnapshot(ref=ref)
        self.ref = self.snapshot.ref
        self.offline = snapshot is not None
        self.max_workers = max_workers
        self.registry = (
            registry if registry is not None else registry_mod.Registry()
        )

    # -- upstream access ----------------------------------------------------

    def _board(self, version: str) -> LeaderboardSnapshot:
        board = self.snapshot.leaderboards.get(version)
        if board is not None:
            return board
        if self.offline:
            raise ValueError(
                f'snapshot has no {version!r} leaderboard '
                f'(has: {sorted(self.snapshot.leaderboards) or "none"})'
            )
        populate_snapshot(
            self.snapshot,
            {version: LEADERBOARDS[version]},
            model_slugs,
            self.max_workers,
        )
        return self.snapshot.leaderboards[version]

    # -- conversion ---------------------------------------------------------

    def fetch_leaderboard(self, version: str = 'v2') -> List[EvaluationLog]:
        """Fetch a complete leaderboard or raise with row provenance."""
        result = self.fetch_leaderboard_result(version)
        result.raise_if_incomplete()
        return result.records

    def fetch_leaderboard_result(
        self, version: str = 'v2'
    ) -> SourceConversionResult[EvaluationLog]:
        """Fetch a leaderboard while retaining valid and rejected rows.

        Args:
            version: Leaderboard version — 'v1' (AlpacaEval 1.0) or
                     'v2' (AlpacaEval 2.0, weighted LC win rate).

        Returns:
            Valid logs plus provenance for rejected and excluded rows.
        """
        if version not in LEADERBOARDS:
            raise ValueError(
                f'Unknown version {version!r}. Choose from: '
                + ', '.join(LEADERBOARDS)
            )
        cfg = LEADERBOARDS[version]
        board = self._board(version)
        ref = self.snapshot.ref
        retrieved_ts = get_current_unix_timestamp()
        # Built over every config in the snapshot, so a row with a miscased
        # repo id can borrow a sibling entry's HuggingFace link as evidence.
        casing = identity_mod.canonical_repo_casing(
            self.snapshot.model_configs.values()
        )

        # Registry lookups that are the same for every row in the leaderboard.
        metrics = _resolve_metrics(self.registry)
        benchmark = self.registry.benchmark(cfg['benchmark_query'])
        harness = self.registry.harness('alpaca_eval')
        evaluator_org = self.registry.org(EVALUATOR_ORG)
        # A resolved benchmark id is what makes these records joinable with
        # other sources' AlpacaEval records, so it names the evaluation when the
        # registry has one. AlpacaEval 1.0 has no canonical entry yet, so it
        # keeps the local name and says so through `benchmark_registry_strategy`.
        evaluation_name = benchmark.canonical_id or cfg['evaluation_name']

        logs: List[EvaluationLog] = []
        failures: List[SourceRecordFailure] = []
        exclusions: List[SourceRecordExclusion] = []

        for row_number, row in enumerate(board.rows, start=2):
            source_ref = f'CSV row {row_number}'
            slug = model_slug_from_row(row)
            if not slug:
                failures.append(
                    SourceRecordFailure(
                        source_ref=source_ref,
                        reason='missing model name',
                        source_record=row,
                    )
                )
                continue
            source_ref = f'{source_ref} ({slug!r})'

            if _NULL_MODEL_RE.fullmatch(slug):
                exclusions.append(
                    SourceRecordExclusion(
                        source_ref=source_ref,
                        reason=(
                            'NullModel is a constant-string placeholder from '
                            'the leaderboard-gaming study, not a model'
                        ),
                        source_record=row,
                    )
                )
                continue

            cell_failure = _metric_cell_failure(metrics, row)
            if cell_failure:
                failures.append(
                    SourceRecordFailure(
                        source_ref=source_ref,
                        reason=cell_failure,
                        source_record=row,
                    )
                )
                continue

            model_config = self.snapshot.model_configs.get(slug)
            config_missing = self.snapshot.missing_model_configs.get(slug)
            resolved = identity_mod.resolve_identity(
                slug, model_config, casing
            )
            if resolved is None:
                failures.append(
                    SourceRecordFailure(
                        source_ref=source_ref,
                        reason=(
                            'cannot determine model identity from '
                            f'{model_config_path(slug)} '
                            f'({config_missing or "no usable evidence"})'
                        ),
                        source_record=row,
                    )
                )
                continue

            # Stable and rerun-idempotent: the leaderboard row plus the pinned
            # upstream revision. Same-model variants (`_concise`, `_verbose`,
            # `-best-of-16`, `_gamed`) legitimately share a model id, so the
            # slug is what keeps their records distinct.
            evaluation_id = f'{cfg["collection"]}/{slug}@{ref[:12]}'

            results = _evaluation_results(
                row,
                cfg,
                board,
                ref,
                evaluation_id,
                _generation_config(
                    model_config,
                    ref,
                    self.snapshot.model_prompts,
                    self.snapshot.missing_model_prompts,
                ),
                metrics,
                benchmark,
                evaluation_name,
            )
            if not results:
                failures.append(
                    SourceRecordFailure(
                        source_ref=source_ref,
                        reason='no usable evaluation metrics',
                        source_record=row,
                    )
                )
                continue

            # The id prefix is the organization as the source spells it; the
            # registry says which organization that is.
            developer = self.registry.org(resolved.developer)
            logs.append(
                EvaluationLog(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=evaluation_id,
                    retrieved_timestamp=retrieved_ts,
                    # evaluation_timestamp is deliberately unset: the CSVs carry
                    # no per-row evaluation date, and the only proxy available
                    # (the snapshot date) would misdate 2023 submissions.
                    eval_library=EvalLibrary(
                        name='alpaca_eval',
                        version=self.snapshot.package_version,
                        additional_details=_stringify(
                            {
                                'leaderboard_version': cfg[
                                    'leaderboard_version'
                                ],
                                'annotator': cfg['annotator'],
                                'baseline_model': cfg['baseline'],
                                'repository': UPSTREAM_URL,
                                'upstream_ref': ref,
                                **harness.provenance('harness'),
                            }
                        ),
                    ),
                    source_metadata=SourceMetadata(
                        source_name=cfg['source_name'],
                        source_type=SourceType.documentation,
                        source_organization_name=EVALUATOR_ORG_NAME,
                        source_organization_url=UPSTREAM_URL,
                        evaluator_relationship=_evaluator_relationship(
                            developer, evaluator_org
                        ),
                        additional_details=_stringify(
                            {
                                'leaderboard_mode': _cell(row, 'mode'),
                                'leaderboard_csv_url': raw_url(
                                    cfg['csv_path'], ref
                                ),
                                'upstream_repository': UPSTREAM_REPO,
                                'upstream_ref': ref,
                            }
                        ),
                    ),
                    model_info=_model_info(
                        resolved, ref, config_missing, developer
                    ),
                    evaluation_results=results,
                )
            )

        return SourceConversionResult(
            source_name=f'AlpacaEval {version}',
            total_records=len(board.rows),
            records=logs,
            failures=failures,
            exclusions=exclusions,
        )
