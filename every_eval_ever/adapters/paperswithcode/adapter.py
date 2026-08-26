#!/usr/bin/env python3
"""Convert Papers with Code evaluation results into Every Eval Ever records.

Data source:
- HF bucket ``huggingface/paperswithcode-backups`` -> nightly PostgreSQL custom-format
  dumps under ``postgres/*.dump`` (pg_dump ``-Fc``). Read with ``pgdumplib``
  (pure-python; no PostgreSQL server needed).

The relevant tables are:
- ``evaluations``  -> one row per (paper, task, dataset, model) leaderboard entry,
                      with a ``metrics`` jsonb of ``{metric_name: value}``.
- ``datasets``     -> the benchmark the eval ran on (source_data).
- ``tasks``        -> the task/category the benchmark belongs to.
- ``metrics``      -> metric definitions incl. ``direction`` (lower/higher_is_better).
- ``papers``       -> provenance (arXiv id / source url).

Shape (see the eee-dataset-conversion skill, reference/fields.md #shape):
- ``source_type = documentation`` -- PwC aggregates reported numbers; no raw outputs.
- aggregate ``.json`` only -- there is no per-item data.
- grain = one ``EvaluationLog`` per model; each ``evaluation_results[]`` entry is
  one (evaluation row x metric-in-jsonb) pair.

Usage:
    # from a dump already on disk (no network):
    uv run python -m every_eval_ever.adapters.paperswithcode.adapter \
        --dump /tmp/pwc-raw/paperswithcode_hf_20260716_031511.dump \
        --dataset-slug eth3d-relative --dataset-slug re10k-2-view \
        --output-dir /tmp/eee-pwc

    # download the latest dump from the HF bucket first:
    uv run python -m every_eval_ever.adapters.paperswithcode.adapter --output-dir data/paperswithcode

Then validate:
    python -m every_eval_ever validate /tmp/eee-pwc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    get_developer,
    raw_capture,
    require_identity,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)

SRC = 'paperswithcode'
PWC_SITE = 'https://paperswithcode.com'
DEFAULT_BUCKET = 'huggingface/paperswithcode-backups'
DEFAULT_OUTPUT_DIR = 'data/paperswithcode'

# A small slice that exercises every field decision:
#   eth3d-relative -> hf_dataset source, open + closed models, higher & lower
#                     is-better metrics, multi-metric rows.
#   re10k-2-view   -> url source, an unbounded metric (PSNR) and a lower-is-better
#                     metric (LPIPS), multi-metric rows.
SAMPLE_DATASET_SLUGS = ('eth3d-relative', 're10k-2-view')

# Vendored snapshot of the eval-card-registry's canonical metrics (bounds +
# direction + score_type), so bounds come from the registry without needing it
# installed at runtime. Regenerate with refresh_registry_snapshot.py.
SNAPSHOT_PATH = Path(__file__).with_name('registry_snapshot.json')

HF_MODEL_RE = re.compile(
    r'https?://huggingface\.co/(?!datasets/|spaces/|collections/)([^/\s]+)/([^/?#\s]+)'
)
HF_DATASET_RE = re.compile(
    r'https?://huggingface\.co/datasets/([^/\s]+/[^/?#\s]+)'
)


# ---------------------------------------------------------------------------
# String / value helpers
# ---------------------------------------------------------------------------


def stringify(value: Any) -> str:
    """Coerce a scalar/collection to a string for a `dict[str, str]` field."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    """Drop None values and stringify the rest (EEE string-maps forbid non-str)."""
    return {k: stringify(v) for k, v in details.items() if v is not None}


def coerce_bool(value: Any) -> bool | None:
    """Postgres dumps booleans as 't'/'f'; normalise to real bools."""
    if isinstance(value, bool):
        return value
    if value in ('t', 'true', 'True', '1'):
        return True
    if value in ('f', 'false', 'False', '0'):
        return False
    return None


def slugify(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw).replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    raw = re.sub(r'-{2,}', '-', raw).strip('-')
    return raw or fallback


def snake(value: Any, fallback: str = 'unknown') -> str:
    return slugify(value, fallback).replace('-', '_').replace('.', '_')


def _to_float(text: str) -> float | None:
    """Parse a numeric string tolerating '%' and European decimal commas.

    PwC stores metric values as free text: '95.2', '95,2'/'0,991'/'97,345'
    (decimal comma), '1,234.5'/'1,234,567' (thousands separator), '30%'. A
    *single* comma reads as a decimal separator regardless of how many digits
    follow it, because a thousands-grouped metric score is vanishingly rare and
    3+ decimal places are routine; a comma alongside a '.', or several commas, is
    a thousands separator and stripped. Non-finite inputs ('NaN', 'Infinity',
    'inf') are rejected -- a score must be a finite real number, not a bound.
    """
    s = str(text).strip().rstrip('%').strip()
    if not s:
        return None
    if (',' in s and '.' in s) or s.count(',') > 1:
        # thousands: 1,234.5 -> 1234.5 ; 1,234,567 -> 1234567
        s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')  # decimal comma: 97,3 -> 97.3 ; 0,991 -> 0.991
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_metric_value(raw: Any) -> tuple[float | None, str | None]:
    """Return (score, uncertainty_text).

    A 'mean +/- sd' value yields the numeric mean plus the raw right-hand token
    as *text*. PwC's '±' does not identify the spread as a standard error, a
    standard deviation, or a CI half-width, so the caller keeps it verbatim rather
    than coercing it into a typed Uncertainty.
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    unc_text: str | None = None
    for sep in ('±', '+/-', '+-'):
        if sep in s:
            left, _, right = s.partition(sep)
            s = left.strip()
            unc_text = right.strip() or None
            break
    return _to_float(s), unc_text


def dedupe(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return out


# ---------------------------------------------------------------------------
# Field builders (pure -- operate on plain dicts, so tests need no DB / network)
# ---------------------------------------------------------------------------


def model_identity(
    model_name: Any, hf_model_url: Any
) -> tuple[str, str, str, str]:
    """Return (model_id, developer, model_slug, display_name).

    Prefers the HF url, which names the publishing org directly and keeps
    HF-true casing. Otherwise the developer comes from the shared helper, and a
    name the helper does not know raises: PwC publishes a bare method name with
    no owning organization for much of its data, and a placeholder developer
    would both assert an untrue identity and route unrelated models into one
    ``unknown/`` directory. Extend ``helpers.get_developer`` to admit a model.

    Effort/mode tiers baked into PwC model names (e.g. 'GPT-5.5 Pro (xhigh)')
    are preserved in the slug rather than stripped -- collapsing them is the
    eval-card-registry's job.
    """
    display = str(model_name or '').strip()
    if not display:
        raise ValueError('PwC row has no model name')
    if hf_model_url:
        m = HF_MODEL_RE.match(str(hf_model_url).strip())
        if m:
            dev, mdl = m.group(1), m.group(2)
            return f'{dev}/{mdl}', dev, mdl, display
    dev = slugify(
        require_identity(
            get_developer(display),
            f'developer for PwC model {display!r} (no HF url; not in '
            'helpers.get_developer)',
        ),
        '',
    )
    mdl = slugify(display, '')
    if not dev or not mdl:
        raise ValueError(f'PwC model name has no usable slug: {display!r}')
    return f'{dev}/{mdl}', dev, mdl, display


def extract_hf_dataset_repo(hf_url: str) -> str | None:
    m = HF_DATASET_RE.match(str(hf_url).strip())
    return m.group(1) if m else None


def dataset_details(dataset: dict[str, Any]) -> dict[str, str]:
    return stringify_details(
        {
            'raw_dataset_id': dataset.get('id'),
            'pwc_dataset_slug': dataset.get('slug'),
            'pwc_dataset_url': f'{PWC_SITE}/dataset/{dataset.get("slug")}'
            if dataset.get('slug')
            else None,
            'license_name': dataset.get('license_name'),
            'license_url': dataset.get('license_url'),
            'introduced_year': dataset.get('introduced_year'),
            'paper_url': dataset.get('paper_url'),
        }
    )


def build_source_data(dataset: dict[str, Any]):
    """The DATASET the eval ran on. hf_dataset if an HF url exists, else url,
    else (never, in practice -- see the skill friction report) private/other."""
    name = dataset.get('name') or dataset.get('slug') or 'unknown'
    details = dataset_details(dataset)
    hf_url = dataset.get('hf_url')
    if hf_url:
        repo = extract_hf_dataset_repo(hf_url)
        if repo:
            return SourceDataHf(
                dataset_name=name,
                source_type='hf_dataset',
                hf_repo=repo,
                additional_details=details,
            )
    urls = dedupe(
        [
            dataset.get('url'),
            dataset.get('homepage'),
            dataset.get('paper_url'),
        ]
    )
    if dataset.get('slug'):
        urls.append(f'{PWC_SITE}/dataset/{dataset["slug"]}')
    urls = dedupe(urls)
    if urls:
        return SourceDataUrl(
            dataset_name=name,
            source_type='url',
            url=urls,
            additional_details=details,
        )
    return SourceDataPrivate(
        dataset_name=name, source_type='other', additional_details=details
    )


def _finite_bounds(lo: float, hi: float) -> tuple[float, float]:
    """Ensure a usable, VALID finite [min, max] (hi strictly > lo)."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Scale reconciliation
#
# The canonical target scale is a property of the metric and comes from the
# registry bounds; it is never inferred here. What IS inferred, once per
# (dataset, metric) leaderboard, is how that board's reported numbers map onto
# it -- `uniform`, `mixed` or `anomaly`, decided in log10 space with mass-aware
# gates. METRIC_MAINTENANCE.md sec 6 documents the three outcomes and what each
# means for a maintainer.
#
# Two invariants hold throughout: the raw value is always retained
# (score_details.raw_value) and every non-identity decision is flagged. A value
# is corrected only when it is impossible under the metric's declared range AND
# a single power-of-100 factor uniquely resolves it -- fix when sure, flag
# otherwise.
# ---------------------------------------------------------------------------

# The only rescales we ever apply: identity, percent->proportion (x1/100), and
# proportion->percent (x100). A value maps to canonical by multiplying by one.
_RESCALE_FACTORS: tuple[float, ...] = (1.0, 0.01, 100.0)
# Minimum empty valley (in log10 decades) between two clusters before a board is
# called genuinely two-scaled -- ~3/4 of an order of magnitude with NO value in
# it. A smooth heavy tail has no such gap; a percent/proportion mix (a full x100
# == 2 decades apart) clears it easily. This gap, not a raw ratio, is what stays
# reliable as N grows (a 3k-row board WILL contain both small and large values).
_LOG_GAP_MIN = 0.75
# ...and the two cluster medians must themselves be ~an order of magnitude apart.
_LOG_CLUSTER_SEP_MIN = 1.3
# A single non-identity factor is the board's uniform scale only if it places
# (nearly) this fraction of values in range while identity fails on the centre.
_UNIFORM_FIT_MIN = 0.9
# Relative tolerance for "out of range enough to matter" -- keeps rounding noise
# at a boundary (a 1.001 on a [0,1] board) from being treated as a scale error.
_SCALE_REL_TOL = 1e-3


def _range_abstol(bound: float) -> float:
    return max(abs(bound) * _SCALE_REL_TOL, 1e-9)


def _in_canonical_range(v: float, lo: float, hi: float) -> bool:
    """True if v is within [lo, hi] (inclusive, small tolerance). An infinite
    bound is open on that side (so unbounded error metrics never go 'out')."""
    if math.isfinite(lo) and v < lo - _range_abstol(lo):
        return False
    if math.isfinite(hi) and v > hi + _range_abstol(hi):
        return False
    return True


def _unique_nonidentity_factor(v: float, lo: float, hi: float) -> float | None:
    """The single x100 // /100 factor that brings an out-of-range v into range,
    or None if none does or the choice is ambiguous (>1 works). Uniqueness is
    the 'we are sure' bar for a per-row fix."""
    fits = [f for f in (0.01, 100.0) if _in_canonical_range(v * f, lo, hi)]
    return fits[0] if len(fits) == 1 else None


def _mass_floor(n: int) -> int:
    """Minimum size for a value cluster to count as a real second scale rather
    than stray noise: max(3, 5% of N) (an HDBSCAN-style min_cluster_size floor).
    A lone off-scale value never clears it."""
    return max(3, math.ceil(0.05 * n))


def _best_factor(cluster: list[float], lo: float, hi: float) -> float | None:
    """The factor placing the most of a cluster in range (identity wins ties);
    None if none place any in range."""
    best_f, best_fit = None, -1
    for f in _RESCALE_FACTORS:
        fit = sum(_in_canonical_range(v * f, lo, hi) for v in cluster)
        if fit > best_fit or (fit == best_fit and f == 1.0):
            best_f, best_fit = f, fit
    return best_f if best_fit and best_fit > 0 else None


@dataclass(frozen=True)
class GroupScale:
    """How one (dataset, metric) leaderboard maps onto the canonical scale.

    mode:
      'uniform' -- multiply the whole group by ``factor`` (1.0 == already
                   canonical); out-of-range stragglers below the mass floor may
                   be fixed per-row.
      'mixed'   -- two log-separated clusters; a value uses ``high_factor`` when
                   log10(value) > ``split_log`` else ``low_factor``.
      'anomaly' -- systematic scale mismatch; do NOT rescale, flag the group.
    """

    mode: str
    n: int
    factor: float = 1.0
    split_log: float = 0.0
    low_factor: float = 1.0
    high_factor: float = 1.0
    reason: str | None = None

    @property
    def allow_row_fix(self) -> bool:
        return self.mode in ('uniform', 'mixed')


def _detect_mixture(
    values: list[float], lo: float, hi: float, floor: int
) -> GroupScale | None:
    """Two genuinely-different scales on one board: an empty ~decade valley in
    log10 space separating two clusters that each clear the mass floor."""
    pos = sorted(v for v in values if v > 0)
    if len(pos) < 2 * floor:
        return None
    logs = [math.log10(v) for v in pos]
    gap, k = max((logs[i + 1] - logs[i], i) for i in range(len(logs) - 1))
    low, high = pos[: k + 1], pos[k + 1 :]
    if gap < _LOG_GAP_MIN or min(len(low), len(high)) < floor:
        return None
    low_logs = [math.log10(v) for v in low]
    high_logs = [math.log10(v) for v in high]
    if (
        statistics.median(high_logs) - statistics.median(low_logs)
        < _LOG_CLUSTER_SEP_MIN
    ):
        return None
    low_f, high_f = _best_factor(low, lo, hi), _best_factor(high, lo, hi)
    if low_f is None or high_f is None or low_f == high_f:
        return None  # not resolvable as two distinct canonical scales
    return GroupScale(
        mode='mixed',
        n=len(values),
        split_log=(logs[k] + logs[k + 1]) / 2.0,
        low_factor=low_f,
        high_factor=high_f,
    )


def analyze_group(values: list[float], lo: float, hi: float) -> GroupScale:
    """Classify a leaderboard's reporting scale relative to canonical [lo, hi]."""
    n = len(values)
    out = [v for v in values if not _in_canonical_range(v, lo, hi)]
    if not out:
        # Everything already sits in range: a clean board, an unbounded error
        # metric, or a metric correctly registered on its natural scale.
        return GroupScale(mode='uniform', n=n, factor=1.0)

    floor = _mass_floor(n)

    # (1) Genuinely mixed -- checked first so an all-one-scale board is not
    # mistaken for a rescale (dividing anything small by 100 trivially "fits").
    mixed = _detect_mixture(values, lo, hi, floor)
    if mixed is not None:
        return mixed

    # (2) Uniform off-scale: the whole board reported on one wrong scale. The
    # robust centre itself must be out of range at identity but rescued by a
    # single factor (a factor that merely fits small values is not enough).
    center = statistics.median(values)
    if not _in_canonical_range(center, lo, hi):
        best = [
            f
            for f in (0.01, 100.0)
            if _in_canonical_range(center * f, lo, hi)
            and sum(_in_canonical_range(v * f, lo, hi) for v in values) / n
            >= _UNIFORM_FIT_MIN
        ]
        if len(best) == 1:
            return GroupScale(mode='uniform', n=n, factor=best[0])

    # (3) A few strays below the mass floor, each uniquely fixable -> keep the
    # board as-is and let the per-row fix handle them.
    if len(out) < floor and all(
        _unique_nonidentity_factor(v, lo, hi) is not None for v in out
    ):
        return GroupScale(mode='uniform', n=n, factor=1.0)

    # (4) A substantial minority is off-scale with no consistent rescale: the
    # registered scale is likely wrong. Flag the group, touch nothing.
    return GroupScale(mode='anomaly', n=n, reason='group_scale_mismatch')


def reconcile_scale(
    score: float,
    lo: float,
    hi: float,
    resolved: bool,
    group_scale: GroupScale | None = None,
) -> tuple[float, dict[str, Any]]:
    """Map one source value onto the canonical [lo, hi] scale using its group's
    decision (``analyze_group``). Returns ``(mapped_score, detail)``; ``detail``
    is empty when nothing was decided. ``canonical_rescale_factor`` is the
    multiplier applied to the raw value to reach canonical (0.01 ==
    percent->proportion, 100.0 == the reverse). ``rescale_basis`` names WHY
    (``group_uniform`` / ``group_mixed`` / ``per_row``). The raw value is
    preserved by the caller regardless. With no group context, a singleton
    group is assumed (per-row fix still applies to a value impossible under its
    own declared range)."""
    detail: dict[str, Any] = {}
    if not resolved:
        return score, detail
    gs = (
        group_scale
        if group_scale is not None
        else GroupScale(mode='uniform', n=1)
    )

    if gs.mode == 'mixed' and score > 0:
        cand = (
            gs.high_factor
            if math.log10(score) > gs.split_log
            else gs.low_factor
        )
    else:
        cand = gs.factor  # 1.0 for uniform-canonical and for anomaly groups
    scaled = score * cand

    if _in_canonical_range(scaled, lo, hi):
        score = scaled
        if cand != 1.0:
            detail['canonical_rescale_factor'] = cand
            detail['rescale_basis'] = (
                'group_mixed' if gs.mode == 'mixed' else 'group_uniform'
            )
    elif gs.allow_row_fix:
        g = _unique_nonidentity_factor(scaled, lo, hi)
        if g is not None:
            score = scaled * g
            detail['canonical_rescale_factor'] = cand * g
            detail['rescale_basis'] = 'per_row'
        else:
            # out of range and no unique fix -> not sure, keep raw + flag
            detail['scale_anomaly'] = 'score_outside_canonical_range'
    else:  # anomaly group: never rescale, just flag (keep raw)
        detail['scale_anomaly'] = gs.reason or 'group_scale_mismatch'

    if detail:
        detail['scale_group_mode'] = gs.mode
        if gs.n > 1:
            detail['scale_group_n'] = gs.n
    return score, detail


def _normalize_metric_key(name: Any) -> str:
    """Mirror the registry's `normalized` matcher: drop case + all separators."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _metadata_kind_confidence(meta: Any) -> tuple[str | None, str | None]:
    """Pull (kind, confidence) out of a registry metric's `metadata` field.

    The seed stores it as a JSON *string* (e.g. '{"kind": "real", "confidence":
    "high"}'); the snapshot may keep it as that string or as a parsed object.
    Best-effort: a malformed/absent value yields (None, None), never raises.
    """
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return None, None
    if not isinstance(meta, dict):
        return None, None
    kind = meta.get('kind')
    confidence = meta.get('confidence')
    return (
        str(kind) if kind is not None else None,
        str(confidence) if confidence is not None else None,
    )


@dataclass(frozen=True)
class ResolvedMetric:
    metric_id: str
    metric_kind: str
    lower_is_better: bool
    score_type: str
    min_score: float
    max_score: float
    resolved: bool  # True = came from the registry snapshot
    detail: dict[str, Any]


class MetricResolver:
    """Resolve a PwC metric name to canonical bounds/direction from the vendored
    registry snapshot (Tier 1). Unknown metrics are recorded and, unless the
    caller opts into a fallback, cause the run to fail closed (Tier 2); with
    ``allow_unresolved`` they get an observed-range proxy (Tier 3).

    Unbounded canonical bounds are ``null`` in the registry snapshot; they are
    emitted as ``+/-inf``, which serializes to the JSON string
    ``"Infinity"``/``"-Infinity"``. ``null`` means "not provided", never
    unbounded.
    """

    def __init__(
        self,
        pwc_directions: dict[str, str] | None = None,
        snapshot_path: str | Path = SNAPSHOT_PATH,
    ) -> None:
        data = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
        self._by_id = {m['id']: m for m in data['metrics']}
        # The exact registry commit these bounds came from (surfaced per metric so
        # a downstream consumer can re-check a value if the registry moves).
        self.registry_revision = (data.get('_meta') or {}).get(
            'registry_revision'
        )
        # Two indices, EXACT-first. A spelling (id/display_name/alias) maps to the
        # set of canonical ids that use it. Exact (case-insensitive) match wins;
        # only if that misses do we fall back to the normalized key (case +
        # separators dropped). A key that maps to >1 id is an unresolvable
        # COLLISION -- we refuse to silently pick one, because a first-seen-wins
        # index lets 'CLIP-IQA'/'CLIPIQA+' steal each other's id.
        self._exact: dict[str, set[str]] = defaultdict(set)
        self._norm: dict[str, set[str]] = defaultdict(set)
        for m in data['metrics']:
            for sp in (
                m['id'],
                m.get('display_name'),
                *(m.get('aliases') or []),
            ):
                if sp:
                    self._exact[str(sp).strip().casefold()].add(m['id'])
                    self._norm[_normalize_metric_key(sp)].add(m['id'])
        self.pwc_directions = pwc_directions or {}
        # raw metric name -> set of dataset slugs it was seen on (for the report)
        self.unresolved: dict[str, set[str]] = {}
        # raw metric name -> (why, candidate_ids): 'unknown' (not in registry) vs
        # 'ambiguous_*' (a collision we would not guess through)
        self.unresolved_reason: dict[str, tuple[str, tuple[str, ...]]] = {}
        # metric name -> count of results emitted with an unbounded (inf) bound
        self.unbounded_emitted: dict[str, int] = {}
        # metric name -> set of dataset slugs emitted with a direction that could
        # be resolved from NEITHER the registry NOR the PwC source (an imperfection
        # the strict gate refuses; see _direction)
        self.direction_unknown: dict[str, set[str]] = {}
        # metric name -> set of dataset slugs where a score could not be reconciled
        # onto the canonical scale (scale_anomaly). Filled by build_results from
        # reconcile_scale's detail; the strict gate refuses these too.
        self.scale_anomalies: dict[str, set[str]] = {}
        # Reconciliations that SUCCEEDED: reported for transparency, never fatal.
        #   scale_corrected -- per-row fixes of values impossible under the
        #                      declared range (kept raw + flagged).
        #   scale_rescaled  -- whole-group technical rescales (percent<->prop).
        self.scale_corrected: dict[str, set[str]] = {}
        self.scale_rescaled: dict[str, set[str]] = {}

    def _direction(
        self, registry_dir: bool | None, metric_name: str
    ) -> tuple[bool, str]:
        """Resolve ``lower_is_better`` by a source-priority chain.

        The registry leaves ``lower_is_better`` ``null`` where direction is a
        property of the use rather than the metric (a refusal rate), so a ``null``
        there is not coerced to ``False`` -- that would assert higher-is-better.
        It falls back to PwC's own per-metric ``direction``, and only if that is
        absent too is the direction unknown. The schema requires a bool, so an
        unknown direction is emitted as ``False`` and tagged, never silently
        wrong: the strict gate fails on it, best-effort keeps it flagged.

        Returns ``(lower_is_better, source)`` with source in
        {``registry``, ``pwc_source``, ``unknown``}.
        """
        if registry_dir is not None:
            return bool(registry_dir), 'registry'
        pwc = self.pwc_directions.get(metric_name)
        if pwc == 'lower_is_better':
            return True, 'pwc_source'
        if pwc is not None:  # any other non-null PwC value == higher_is_better
            return False, 'pwc_source'
        return False, 'unknown'

    def _match(
        self, metric_name: str
    ) -> tuple[str | None, str, tuple[str, ...]]:
        """Return (canonical_id, match_tier, candidate_ids).

        canonical_id is None when the name is unresolvable -- either 'unknown'
        (no candidate) or an 'ambiguous_*' collision (>1 candidate). Exact match
        is tried before the lossy normalized match so distinct-but-similar names
        ('CLIP-IQA' vs 'CLIPIQA+') resolve to their own ids.
        """
        exact = self._exact.get(str(metric_name).strip().casefold())
        if exact:
            if len(exact) == 1:
                return next(iter(exact)), 'exact', ()
            return None, 'ambiguous_exact', tuple(sorted(exact))
        nrm = self._norm.get(_normalize_metric_key(metric_name))
        if nrm:
            if len(nrm) == 1:
                return next(iter(nrm)), 'normalized', ()
            return None, 'ambiguous_normalized', tuple(sorted(nrm))
        return None, 'unknown', ()

    def bounds_for(self, metric_name: str) -> tuple[float, float] | None:
        """Canonical (lo, hi) for a name, or None if it does not resolve to a
        single registry entry. Side-effect free (does NOT record unresolved /
        direction state) -- used to analyse a group's reporting scale up front.
        Null registry bounds become +/-inf, exactly as ``resolve``."""
        cid, _, _ = self._match(metric_name)
        if cid is None:
            return None
        entry = self._by_id[cid]
        lo, hi = entry.get('min_score'), entry.get('max_score')
        lo = float('-inf') if lo is None else float(lo)
        hi = float('inf') if hi is None else float(hi)
        return lo, hi

    def resolve(
        self,
        metric_name: str,
        obs_range: tuple[float, float],
        dataset_slug: str | None = None,
    ) -> ResolvedMetric:
        obs_min, obs_max = obs_range
        canonical_id, tier, candidates = self._match(metric_name)
        if canonical_id is not None:
            entry = self._by_id[canonical_id]
            score_type = entry.get('score_type') or 'continuous'
            detail: dict[str, Any] = {
                'bound_source': 'registry',
                'canonical_metric_id': entry['id'],
                'match_tier': tier,
                'bound_registry_revision': self.registry_revision,
            }
            # Surface the canonical entry's vetting status/confidence rather than
            # hard-rejecting un-reviewed metrics, so a conversion does not wait
            # on the registry's review queue while consumers can still see that
            # a bound is only `draft`.
            if entry.get('review_status'):
                detail['canonical_review_status'] = entry.get('review_status')
            kind, confidence = _metadata_kind_confidence(entry.get('metadata'))
            if kind:
                detail['canonical_metric_kind_flag'] = kind
            if confidence:
                detail['canonical_confidence'] = confidence
            lo, hi = entry.get('min_score'), entry.get('max_score')
            # null bound in the registry == unbounded -> +/-inf, which the schema
            # serializes as the JSON string "Infinity"/"-Infinity".
            if lo is None:
                lo = float('-inf')
                detail['canonical_min'] = 'unbounded'
            if hi is None:
                hi = float('inf')
                detail['canonical_max'] = 'unbounded'
            if lo == float('-inf') or hi == float('inf'):
                self.unbounded_emitted[metric_name] = (
                    self.unbounded_emitted.get(metric_name, 0) + 1
                )
            lib, dir_source = self._direction(
                entry.get('lower_is_better'), metric_name
            )
            detail['direction_source'] = dir_source
            if dir_source == 'unknown':
                self.direction_unknown.setdefault(metric_name, set())
                if dataset_slug:
                    self.direction_unknown[metric_name].add(dataset_slug)
            return ResolvedMetric(
                metric_id=entry['id'],
                metric_kind=entry['id'],
                lower_is_better=lib,
                score_type=score_type,
                min_score=float(lo),
                max_score=float(hi),
                resolved=True,
                detail=detail,
            )
        # Unresolved: an unknown metric OR an ambiguous collision. Both fail closed
        # by default and are salvageable with --allow-unresolved (observed-range
        # bounds), so a collision is handled by the SAME gate as any un-vetted
        # metric -- no separate flag.
        self.unresolved.setdefault(metric_name, set())
        if dataset_slug:
            self.unresolved[metric_name].add(dataset_slug)
        self.unresolved_reason[metric_name] = (tier, candidates)
        lo, hi = _finite_bounds(min(0.0, obs_min), obs_max)
        lib, dir_source = self._direction(None, metric_name)
        detail = {
            'bound_source': 'observed_unresolved',
            'match_tier': tier,
            'pwc_metric_direction': self.pwc_directions.get(metric_name),
            'direction_source': dir_source,
        }
        # NB: an unresolved metric is already caught by the unresolved gate, so its
        # unknown direction is not *also* tracked in direction_unknown (no double
        # count); the direction_source flag is still recorded for transparency.
        if candidates:
            detail['collision_candidates'] = list(candidates)
        return ResolvedMetric(
            metric_id=f'{SRC}.{snake(metric_name)}',
            metric_kind=snake(metric_name),
            lower_is_better=lib,
            score_type='continuous',
            min_score=lo,
            max_score=hi,
            resolved=False,
            detail=detail,
        )


# The unit names the CANONICAL scale -- the one the emitted score is on after
# reconciliation -- not the scale PwC declared for the source number. Any other
# bound pair ([0,inf) errors, [-1,1] correlations, [1,5] MOS) has no unit name,
# and an unresolved metric has no canonical contract; both leave it unset. PwC's
# declaration is kept verbatim as `pwc_scale` in additional_details.
_BOUNDS_TO_UNIT = {
    (0.0, 1.0): 'proportion',
    (0.0, 100.0): 'percent',
}


def _metric_unit_from_bounds(resolved: ResolvedMetric) -> str | None:
    if not resolved.resolved:
        return None
    return _BOUNDS_TO_UNIT.get((resolved.min_score, resolved.max_score))


def build_metric_config(
    metric_name: str,
    resolved: ResolvedMetric,
    obs_range: tuple[float, float],
    metric_meta: dict[str, Any] | None,
) -> MetricConfig:
    meta = metric_meta or {}
    return MetricConfig(
        evaluation_description=meta.get('description'),
        metric_id=resolved.metric_id,
        metric_name=metric_name,
        metric_kind=resolved.metric_kind,
        # The unit of the canonical scale the score is emitted on, so it stays
        # true after a rescale; PwC's declaration is kept as `pwc_scale`.
        metric_unit=_metric_unit_from_bounds(resolved),
        lower_is_better=resolved.lower_is_better,
        score_type=ScoreType(resolved.score_type),
        min_score=resolved.min_score,
        max_score=resolved.max_score,
        additional_details=stringify_details(
            {
                **resolved.detail,
                'observed_min': obs_range[0],
                'observed_max': obs_range[1],
                'pwc_metric_full_name': meta.get('full_name'),
                'pwc_scale': meta.get('scale'),
            }
        ),
    )


def score_details(
    ev: dict[str, Any],
    raw_value: Any,
    score: float,
    uncertainty_text: str | None,
    dataset: dict[str, Any],
    paper: dict[str, Any] | None,
    scale_detail: dict[str, Any] | None = None,
) -> ScoreDetails:
    paper = paper or {}
    arxiv_id = paper.get('arxiv_id')
    return ScoreDetails(
        score=score,
        # PwC's '±' spread does not declare itself a standard error, standard
        # deviation, or CI half-width, so we do NOT assert a typed Uncertainty
        # (which would misrepresent it). The reported spread is kept verbatim in
        # `reported_uncertainty` (and within `raw_value`) for downstream
        # interpretation.
        uncertainty=None,
        details=stringify_details(
            {
                **(scale_detail or {}),
                'raw_value': raw_value,
                'reported_uncertainty': uncertainty_text,
                'pwc_evaluation_id': ev.get('id'),
                'best_rank': ev.get('best_rank'),
                'best_metric': ev.get('best_metric'),
                'harness': _clean_harness(ev.get('harness')),
                'uses_additional_data': coerce_bool(
                    ev.get('uses_additional_data')
                ),
                'external': coerce_bool(ev.get('external')),
                'external_source_url': ev.get('external_source_url'),
                'source_url': ev.get('source_url'),
                'paper_arxiv_url': f'https://arxiv.org/abs/{arxiv_id}'
                if arxiv_id
                else None,
                'paper_title': paper.get('title'),
                'paper_source_url': paper.get('source_url'),
            }
        ),
    )


def _clean_harness(harness: Any) -> str | None:
    """PwC 'harness' is often an agent scaffold or 'Not reported' -- not a classic
    eval library. Normalise obvious non-values to None."""
    if not harness:
        return None
    text = str(harness).strip()
    if text.lower() in ('not reported', 'none', 'n/a', 'unknown', 'pool'):
        return None
    return text


def build_results(
    ev: dict[str, Any],
    dataset: dict[str, Any],
    task: dict[str, Any] | None,
    resolver: MetricResolver,
    metric_ranges: dict[str, tuple[float, float]],
    metric_meta: dict[str, dict[str, Any]],
    paper: dict[str, Any] | None,
    group_scales: dict[tuple[Any, str], GroupScale] | None = None,
) -> list[EvaluationResult]:
    """Fan one evaluation row out to one EvaluationResult per jsonb metric."""
    try:
        metrics = json.loads(ev['metrics']) if ev.get('metrics') else {}
    except (TypeError, ValueError):
        metrics = {}
    if not isinstance(metrics, dict) or not metrics:
        return []

    group_scales = group_scales or {}
    ds_slug = dataset.get('slug') or slugify(dataset.get('name'))
    task_slug = (task or {}).get('slug') or 'unknown-task'
    eval_name = f'{SRC}.{snake(task_slug)}.{snake(ds_slug)}'
    src_data = build_source_data(dataset)
    ts = ev.get('evaluated_on') or (
        str(ev.get('created_at') or '')[:10] or None
    )

    results: list[EvaluationResult] = []
    for mname, raw in metrics.items():
        score, unc_text = parse_metric_value(raw)
        if score is None:
            continue
        obs_range = metric_ranges.get(mname, (score, score))
        resolved = resolver.resolve(mname, obs_range, ds_slug)
        # The reporting scale is decided once per (dataset, metric) leaderboard
        # (see analyze_group), then applied to this single value.
        group_scale = group_scales.get((ev.get('dataset_id'), mname))
        score, scale_detail = reconcile_scale(
            score,
            resolved.min_score,
            resolved.max_score,
            resolved.resolved,
            group_scale,
        )
        if 'scale_anomaly' in scale_detail:
            resolver.scale_anomalies.setdefault(mname, set()).add(ds_slug)
        elif scale_detail.get('rescale_basis') == 'per_row':
            resolver.scale_corrected.setdefault(mname, set()).add(ds_slug)
        elif 'rescale_basis' in scale_detail:  # group_uniform / group_mixed
            resolver.scale_rescaled.setdefault(mname, set()).add(ds_slug)
        results.append(
            EvaluationResult(
                evaluation_result_id=f'{SRC}.{ev.get("id")}.{snake(mname)}',
                evaluation_name=eval_name,
                source_data=src_data,
                evaluation_timestamp=str(ts) if ts else None,
                metric_config=build_metric_config(
                    mname, resolved, obs_range, metric_meta.get(mname)
                ),
                score_details=score_details(
                    ev, raw, score, unc_text, dataset, paper, scale_detail
                ),
            )
        )
    return results


def build_source_metadata(
    dump_version: str,
    source_bucket: str | None = None,
    dump_file: str | None = None,
) -> SourceMetadata:
    # Provenance reflects the ACTUAL source of this run: the HF bucket only when
    # the dump was fetched from one, plus the dump file name. Naming the default
    # bucket on a `--dump` or custom-bucket run would assert provenance that did
    # not hold.
    details: dict[str, Any] = {
        'source_role': 'aggregator',
        'dump_version': dump_version,
        'note': (
            'Scores aggregated by Papers with Code from papers and external '
            'leaderboards; not re-run by this adapter.'
        ),
    }
    if source_bucket:
        details['source_bucket'] = source_bucket
    if dump_file:
        details['source_dump_file'] = dump_file
    return SourceMetadata(
        source_name='Papers with Code',
        source_type='documentation',
        source_organization_name='Papers with Code',
        source_organization_url=PWC_SITE,
        # A leaderboard aggregating reported numbers is third_party wrt the model
        # developer, even when a score was self-reported to it (see fields.md).
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details=stringify_details(details),
    )


def canonical_spelling(
    spellings: set[tuple[str, str, str]],
) -> tuple[str, str, str]:
    """Pick one ``(model_id, developer, model_slug)`` for one model.

    PwC spells the same model differently from row to row and only some rows
    carry the HF url that settles it -- ``moonshotai/Kimi-K2.5`` on one,
    ``kimi-k2.5`` on the next. Each spelling taken at face value becomes its own
    log, splitting the model's results between two records whose paths differ
    only in case: two directories on Linux, and on a case-insensitive filesystem
    one whose name depends on which row was read first.

    HF's spelling wins, and is recognisable without re-reading the row -- the
    fallback slug is always lowercased, so an uppercase letter means the url.
    Sorting settles the rest (some PwC urls disagree in case), so the choice
    never depends on row order.
    """
    from_hf = sorted(s for s in spellings if s[0] != s[0].lower())
    return (from_hf or sorted(spellings))[0]


def model_availability(open_flags: set[bool | None], repo_linked: bool) -> str:
    """Map PwC's ``is_open`` flags for one model onto the schema's axis.

    ``is_open`` is recorded per leaderboard row, so a model with several rows
    carries several flags. Only an unambiguous set names the model: an empty one
    (no row said), or rows that contradict each other, is ``unknown`` rather than
    a coin toss on whichever row happened to be read first.

    A ``not open`` flag on a model whose rows link an HF model repo is the same
    kind of disagreement -- the source names a home for the weights it says are
    unavailable -- so that is ``unknown`` too, rather than a record whose own two
    adjacent fields contradict each other.
    """
    stated = {flag for flag in open_flags if flag is not None}
    if len(stated) != 1:
        return 'unknown'
    if stated.pop():
        return 'open_weights'
    return 'unknown' if repo_linked else 'closed_weights'


def build_model_info(
    model_id: str,
    developer: str,
    display_name: str,
    ev: dict[str, Any],
    open_flags: set[bool | None],
    repo_linked: bool,
) -> ModelInfo:
    return ModelInfo(
        name=display_name,
        id=model_id,
        developer=developer,
        additional_details=stringify_details(
            {
                'raw_model_name': display_name,
                'hf_model_url': ev.get('hf_model_url'),
                'num_parameters': ev.get('num_parameters'),
                # `is_open` is a claim about the weights, so it names
                # model_availability. deployment_type keeps the schema's
                # `unknown`: PwC records what the model is, never how the party
                # reporting the number reached it.
                'model_availability': model_availability(open_flags, repo_linked),
            }
        ),
    )


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str


def _partition_out_of_range(
    results: list[EvaluationResult],
) -> tuple[list[EvaluationResult], list[EvaluationResult]]:
    """Split results into (publishable, out of range) on their own bounds.

    The declared ``[min_score, max_score]`` must contain the score -- the
    datastore's semantic check rejects the record otherwise. Reconciliation can
    still leave a score outside it: a ``scale_anomaly`` keeps the raw value on
    purpose, and a boundary overrun within ``_SCALE_REL_TOL`` is tolerated for
    scale classification. Those cells belong in the failure report, not the data.
    """
    keep: list[EvaluationResult] = []
    out: list[EvaluationResult] = []
    for res in results:
        cfg = res.metric_config
        score = res.score_details.score
        in_range = cfg.min_score <= score <= cfg.max_score
        (keep if in_range else out).append(res)
    return keep, out


def build_logs(
    evaluations: Iterable[dict[str, Any]],
    datasets_by_id: dict[Any, dict[str, Any]],
    tasks_by_id: dict[Any, dict[str, Any]],
    resolver: MetricResolver,
    metric_ranges: dict[str, tuple[float, float]],
    metric_meta: dict[str, dict[str, Any]],
    papers_by_id: dict[Any, dict[str, Any]],
    dump_version: str,
    retrieved_ts: str,
    source_bucket: str | None = None,
    dump_file: str | None = None,
    group_scales: dict[tuple[Any, str], GroupScale] | None = None,
) -> SourceConversionResult[LogBundle]:
    """Group evaluation rows by canonical model id into one log per model.

    A row that cannot be represented -- no usable metric, or a model whose
    developer cannot be established -- is recorded as a failure with its source
    reference rather than dropped, so the caller can report every omission and
    exit non-zero.
    """
    # Keyed on the case-folded model id, so rows that spell one model two ways
    # land in one log rather than one each (see `canonical_spelling`).
    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    harnesses: dict[str, set[str]] = defaultdict(set)
    open_flags: dict[str, set[bool | None]] = defaultdict(set)
    repo_linked: dict[str, bool] = defaultdict(bool)
    spellings: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    first_row: dict[str, tuple[str, dict[str, Any]]] = {}
    failures: list[SourceRecordFailure] = []
    total = 0

    for ev in evaluations:
        total += 1
        source_ref = f'evaluations.id={ev.get("id")}'
        dataset = datasets_by_id.get(ev.get('dataset_id'))
        if dataset is None:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=(
                        f'dataset_id {ev.get("dataset_id")!r} is not in the '
                        'dump'
                    ),
                    source_record=ev,
                )
            )
            continue
        task = tasks_by_id.get(ev.get('task_id'))
        paper = papers_by_id.get(ev.get('paper_id'))
        results = build_results(
            ev,
            dataset,
            task,
            resolver,
            metric_ranges,
            metric_meta,
            paper,
            group_scales,
        )
        if not results:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='no metric on this row converted to a valid result',
                    source_record=ev,
                )
            )
            continue
        results, out_of_range = _partition_out_of_range(results)
        for res in out_of_range:
            cfg = res.metric_config
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{source_ref} metric={cfg.metric_name}',
                    reason=(
                        f'score {res.score_details.score!r} is outside the '
                        f'canonical range [{cfg.min_score}, {cfg.max_score}] '
                        f'for {cfg.metric_id!r}; the reporting scale could not '
                        'be reconciled, so the value is not published'
                    ),
                    source_record=ev,
                )
            )
        if not results:
            continue
        try:
            model_id, developer, model_slug, display = model_identity(
                ev.get('model_name'), ev.get('hf_model_url')
            )
        except ValueError as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=str(exc),
                    source_record=ev,
                )
            )
            continue
        key = model_id.lower()
        groups[key].extend(results)
        harness = _clean_harness(ev.get('harness'))
        if harness:
            harnesses[key].add(harness)
        open_flags[key].add(coerce_bool(ev.get('is_open')))
        repo_linked[key] |= bool(ev.get('hf_model_url'))
        spellings[key].add((model_id, developer, model_slug))
        # The remaining model fields repeat across a model's rows; take the first
        # row of the spelling that wins, so name and url agree with the id.
        first_row.setdefault(model_id, (display, ev))

    bundles: list[LogBundle] = []
    for key, results in sorted(groups.items()):
        model_id, developer, model_slug = canonical_spelling(spellings[key])
        display, model_row = first_row[model_id]
        # eval_library is reserved for the eval *harness* (inspect_ai/lm-eval/helm).
        # PwC's `harness` column is usually an agent scaffold (SWE-agent, OpenHands)
        # or "Not reported" -- NOT a harness -- so eval_library stays 'unknown' and
        # any scaffold is recorded in additional_details instead.
        harness_set = harnesses.get(key, set())
        eval_lib_details = (
            {'pwc_harness': ', '.join(sorted(harness_set))}
            if harness_set
            else None
        )
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            # STABLE anchor: model id + dump version -> idempotent per dump, never `now`.
            evaluation_id=f'{SRC}/{model_id.replace("/", "_")}/{dump_version}',
            retrieved_timestamp=retrieved_ts,
            source_metadata=build_source_metadata(
                dump_version, source_bucket, dump_file
            ),
            eval_library=EvalLibrary(
                name='unknown',
                version='unknown',
                additional_details=eval_lib_details,
            ),
            model_info=build_model_info(
                model_id,
                developer,
                display,
                model_row,
                open_flags[key],
                repo_linked[key],
            ),
            evaluation_results=sorted(
                results, key=lambda r: r.evaluation_result_id or ''
            ),
        )
        bundles.append(
            LogBundle(log=log, developer=developer, model=model_slug)
        )
    return SourceConversionResult(
        source_name='Papers with Code',
        total_records=total,
        records=bundles,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Dump IO (pgdumplib) -- kept out of the pure builders so tests need no DB
# ---------------------------------------------------------------------------


def _parse_columns(create_defn: str) -> list[str]:
    """Extract column names (in order) from a CREATE TABLE statement."""
    body = create_defn.split('(', 1)[1]
    cols: list[str] = []
    for line in body.splitlines():
        line = line.strip().rstrip(',')
        if not line or line.startswith('CONSTRAINT') or line.startswith(')'):
            continue
        cols.append(line.split()[0])
    return cols


def load_dump(dump_path: str | Path):
    import pgdumplib

    return pgdumplib.load(str(dump_path))


def _columns_for(dump, table: str) -> list[str]:
    for e in dump.entries:
        if e.desc == 'TABLE' and e.tag == table:
            return _parse_columns(e.defn)
    raise KeyError(f'table public.{table} not found in dump')


def table_rows(dump, table: str) -> Iterator[dict[str, Any]]:
    cols = _columns_for(dump, table)
    for row in dump.table_data('public', table):
        yield dict(zip(cols, row))


def _iter_metric_values(metrics_json: Any) -> Iterator[tuple[str, float]]:
    try:
        metrics = json.loads(metrics_json) if metrics_json else {}
    except (TypeError, ValueError):
        return
    if not isinstance(metrics, dict):
        return
    for name, raw in metrics.items():
        val, _ = parse_metric_value(raw)
        if val is not None:
            yield name, val


def scan_evaluations(
    dump,
    dataset_ids: set[Any] | None,
    limit: int | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[float, float]],
    dict[tuple[Any, str], list[float]],
]:
    """Single pass over ``evaluations``. Accumulate, over the WHOLE dump,
    per-metric observed ranges (stable, slice-independent bounds for the
    unresolved fallback) AND, per (dataset, metric) leaderboard within the
    selected slice, the full list of reported values used to infer that group's
    reporting scale (see ``analyze_group``). The group values ignore ``--limit``
    -- the limit caps how many rows are EMITTED, not the scale evidence for a
    leaderboard."""
    ranges: dict[str, list[float]] = defaultdict(
        lambda: [float('inf'), float('-inf')]
    )
    group_values: dict[tuple[Any, str], list[float]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for ev in table_rows(dump, 'evaluations'):
        vals = list(_iter_metric_values(ev.get('metrics')))
        for name, val in vals:
            r = ranges[name]
            r[0] = min(r[0], val)
            r[1] = max(r[1], val)
        if dataset_ids is not None and ev.get('dataset_id') not in dataset_ids:
            continue
        for name, val in vals:
            group_values[(ev.get('dataset_id'), name)].append(val)
        if limit is not None and len(selected) >= limit:
            continue
        selected.append(ev)
    metric_ranges = {k: (lo, hi) for k, (lo, hi) in ranges.items()}
    return selected, metric_ranges, dict(group_values)


def build_group_scales(
    group_values: dict[tuple[Any, str], list[float]],
    resolver: MetricResolver,
) -> dict[tuple[Any, str], GroupScale]:
    """Decide each (dataset, metric) leaderboard's reporting scale once, from
    its full value list and the metric's canonical registry bounds. Unresolved
    metrics get no entry (they are never rescaled -- their bounds are observed,
    already on the source scale)."""
    bounds_cache: dict[str, tuple[float, float] | None] = {}
    scales: dict[tuple[Any, str], GroupScale] = {}
    for (ds_id, name), vals in group_values.items():
        if name not in bounds_cache:
            bounds_cache[name] = resolver.bounds_for(name)
        bounds = bounds_cache[name]
        if bounds is None or not vals:
            continue
        scales[(ds_id, name)] = analyze_group(vals, *bounds)
    return scales


def read_papers_subset(dump, paper_ids: set[Any]) -> dict[Any, dict[str, Any]]:
    if not paper_ids:
        return {}
    want = {str(p) for p in paper_ids}
    out: dict[Any, dict[str, Any]] = {}
    for row in table_rows(dump, 'papers'):
        if str(row.get('id')) in want:
            out[row['id']] = {
                'arxiv_id': row.get('arxiv_id'),
                'title': row.get('title'),
                'source_url': row.get('source_url'),
            }
    return out


def dump_version_from_path(dump_path: str | Path) -> str:
    # Keep any sub-day suffix (e.g. `_HHMMSS`) so two dumps on the same date get
    # distinct versions, hence distinct evaluation_ids. A date-only filename still
    # yields just the date, so the common one-dump-per-day case is unchanged.
    m = re.search(r'(\d{8})(_\d+)?', Path(dump_path).name)
    if not m:
        return Path(dump_path).stem
    return m.group(1) + (m.group(2) or '')


# ---------------------------------------------------------------------------
# HF bucket download
# ---------------------------------------------------------------------------


# The HF *bucket* API (`list_bucket_tree` / `download_bucket_files`) exists only
# in `huggingface_hub>=1.0`, above the range this repo pins. Auto-download is
# therefore an optional capability: the import is lazy and only this one code
# path needs it, so `--dump` (a dump already on disk) keeps working under the
# pinned range.
def _require_bucket_api():
    """Return an ``HfApi``, or exit with a clear remedy if the bucket API is absent.

    The two bucket methods land together in ``huggingface_hub>=1.0``; feature-
    detecting ``list_bucket_tree`` is more robust than parsing a version string
    (and lets the test suite substitute a fake ``HfApi``).
    """
    from huggingface_hub import HfApi

    if not hasattr(HfApi, 'list_bucket_tree'):
        try:
            from importlib.metadata import version

            installed = version('huggingface_hub')
        except Exception:
            installed = 'unknown'
        raise SystemExit(
            'auto-download from the HF bucket needs huggingface_hub>=1.0 for '
            f'the bucket API, but {installed} is installed. Either install a '
            "1.x build here (`pip install 'huggingface_hub>=1.0'`), or pass "
            '--dump <path> to convert a dump already on disk (that path needs '
            'only pgdumplib, no bucket API).'
        )
    return HfApi()


def latest_dump_remote_path(bucket: str, prefix: str = 'postgres') -> str:
    api = _require_bucket_api()
    # The dumps live under `postgres/` in the bucket; list that subtree
    # RECURSIVELY. A non-recursive top-level listing returns the `postgres` dir
    # entry (not the nested `.dump` files) and silently finds nothing.
    dumps = [
        f.path
        for f in api.list_bucket_tree(bucket, prefix=prefix, recursive=True)
        if getattr(f, 'path', '').endswith('.dump')
    ]
    if not dumps:
        raise SystemExit(
            f'no .dump files found under {prefix!r} in bucket {bucket}'
        )
    return sorted(dumps)[-1]


def download_dump(bucket: str, remote_path: str, dest_dir: Path) -> Path:
    api = _require_bucket_api()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(remote_path).name
    if local.exists():
        print(f'reusing cached dump {local}')
        return local
    print(f'downloading {bucket}:{remote_path} -> {local}')
    api.download_bucket_files(
        bucket, [(remote_path, str(local))], raise_on_missing_files=True
    )
    return local


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def retrieved_ts_from_dump(dump_version: str) -> str:
    """Deterministic Unix-epoch ``retrieved_timestamp`` derived from the dump date.

    ``dump_version`` is the dump's ``YYYYMMDD...`` stamp. Pinning the retrieved
    timestamp to the dump — rather than ``time.time()`` at conversion time — makes
    a re-run over the SAME dump byte-identical, so regenerating the datastore does
    not churn every record's timestamp.
    Falls back to the raw version string if it does not start with a parseable
    date. The schema constrains ``retrieved_timestamp`` only to a string
    documented as Unix epoch, so a date-derived epoch is valid.
    """
    try:
        dt = datetime.strptime(str(dump_version)[:8], '%Y%m%d')
    except (ValueError, TypeError):
        return str(dump_version)
    return str(dt.replace(tzinfo=timezone.utc).timestamp())


def existing_records(output_dir: str | Path) -> list[Path]:
    """List the records already under ``output_dir``, newest run aside."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob('*.json'))


def _record_evaluation_id(path: Path) -> str | None:
    """Return a record's ``evaluation_id``, or ``None`` if it cannot be read."""
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('evaluation_id')
    except (OSError, ValueError):
        return None


def remove_superseded_records(
    stale: Iterable[Path],
    keep: set[Path],
    superseded_ids: set[str],
    root: str | Path,
) -> int:
    """Delete a previous run's copies of the evaluations THIS run rewrote.

    Each file is named by a fresh ``uuid4``, so re-running the same selection
    would otherwise pile up records that differ only by filename. A stale file is
    removed only when it is not one we just wrote and its ``evaluation_id`` is one
    this run re-emitted; records for evaluations outside this run's selection --
    other datasets already in an existing export, say -- are left untouched, so a
    partial run cannot wipe them. Deletion happens only after publication
    succeeds, so a failure leaves the previous run's output intact. Returns the
    number of records removed.
    """
    root = Path(root).resolve()
    removed = 0
    emptied = set()
    for path in stale:
        if path in keep or not path.exists():
            continue
        if _record_evaluation_id(path) not in superseded_ids:
            continue
        path.unlink()
        emptied.add(path.parent.resolve())
        removed += 1
    for directory in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
        # Prune only the model/developer directories this call emptied, never
        # the output root itself or anything above it.
        while directory != root and root in directory.parents:
            if any(directory.iterdir()):
                break
            directory.rmdir()
            directory = directory.parent
    return removed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Convert Papers with Code dumps to Every Eval Ever format.'
    )
    ap.add_argument(
        '--dump',
        type=Path,
        help='Path to a local .dump file. If omitted, a dump is fetched from '
        'the HF bucket.',
    )
    ap.add_argument(
        '--bucket',
        default=DEFAULT_BUCKET,
        help=f'HF bucket to fetch the dump from (default: {DEFAULT_BUCKET}).',
    )
    ap.add_argument(
        '--remote-path',
        help='Specific postgres/*.dump path in the bucket (default: latest).',
    )
    ap.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('/tmp/pwc-raw'),
        help='Where to download the dump (default: /tmp/pwc-raw).',
    )
    ap.add_argument(
        '--dataset-slug',
        action='append',
        dest='dataset_slugs',
        help='Restrict to this dataset slug (repeatable). Defaults to a small '
        'representative sample; pass --all to convert everything.',
    )
    ap.add_argument(
        '--all',
        action='store_true',
        help='Convert every dataset (overrides the default sample slice).',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Cap the number of evaluation rows emitted (after filtering).',
    )
    ap.add_argument(
        '--allow-unresolved',
        action='store_true',
        help='Narrow relaxation: tolerate ONLY unresolved/ambiguous metrics '
        '(emit with observed-range bounds, labelled), while STILL failing on '
        'other imperfections (unknown direction, scale anomaly). Without it the '
        'run fails closed on unresolved metrics so CI never ships un-vetted bounds.',
    )
    ap.add_argument(
        '--best-effort',
        action='store_true',
        help='Emit as much data as possible: an imperfection (unresolved metric, '
        'unknown direction, scale anomaly) is flagged in the output instead of '
        'aborting the run. The default is strict -- ANY imperfection aborts '
        'non-zero, giving CI a clean-or-fail signal; use --best-effort for '
        'exploratory runs or to keep collecting data while fixes are batched. '
        'Neither mode publishes an invalid record: a score that cannot be placed '
        'on its canonical scale, and a row with no usable metric or no '
        'establishable developer, are omitted and listed in the failure report, '
        'so a partial conversion still exits non-zero. Imperfections are always '
        'reported.',
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    ap.add_argument(
        '--emit-source-version',
        action='store_true',
        help='Print the current dump version (its date stamp) and exit without '
        'converting. Resolves the latest dump by listing the bucket, without '
        'downloading it; a --dump path names that file instead. The scheduler '
        'uses this to skip a run whose dump has not rolled over.',
    )
    return ap.parse_args(argv)


# --- metric-family triage table --------------------------------------------
# Sorts the unresolved report into names that belong to a standard family (a
# quick registry add) and bespoke composites that need a paper read, so a
# maintainer can tell which is which at a glance.
#
# These patterns are a triage HINT, never an auto-registration rule: a name does
# not determine a bound (PIE-Bench reports LPIPS x10^3; a 'BD-Rate (PSNR ...)' is
# a signed rate, not a PSNR). METRIC_MAINTENANCE.md sec 3-4 is the procedure.
# Order matters -- the first match wins, so specific families precede generic
# ones.
_FAMILY_TABLE: tuple[tuple[str, str, str], ...] = (
    # (family, suggested bound/direction, regex tried against the metric name)
    (
        'bd',
        'null..null, lower better (signed %)',
        r'\bbd[-\s]?rate\b|bjontegaard',
    ),
    (
        'pose-error',
        '[0, inf), lower better (mm)',
        r'\b(?:pa-|n-|b-|f-|fb-|lh-|rh-)?mp?[jv]pe\b|\bmpvpe\b|\bn?m[jv]e\b|\bmrpe\b|\bpve\b|\btepe\b|end-?point\s*error',
    ),
    (
        'mos',
        '[1, 5], higher better',
        r'\bmos\b|dnsmos|plcmos|utmos|visqol|nisqa',
    ),
    ('pesq', '[-0.5, 4.5], higher better', r'\bpesq\b'),
    ('stoi', '[0, 1], higher better', r'\be?stoi\b'),
    ('mcd', '[0, inf), lower better', r'\bmcd\b|mel[-\s]?cepstral'),
    ('psnr', '[0, inf), higher better (dB)', r'\bpsnr\b'),
    ('bitrate', '[0, inf), lower better', r'\bb(?:pp|psp)\b|bits?[-\s]?per'),
    (
        'spec-loss',
        '[0, inf), lower better',
        r'\b(?:mel|stft|f0)[-\s]?(?:loss|rmse|dist|distance|l1)\b|spectral\s*(?:loss|convergence)',
    ),
    (
        'gen-dist',
        '[0, inf), lower better',
        r'\br?f[iv]d\b|\bkid\b|\bfvd\b|inception\s*distance',
    ),
    (
        'dist-error',
        '[0, inf), lower better',
        r'\b(?:r?mse|mae|l1|l2)\b|\bmean\s*(?:squared|absolute)\s*error',
    ),
    (
        'rate',
        '[0, 1], higher better (percent boards rescaled)',
        r'\bacc(?:uracy)?\b|\bf1\b|\bap\d*\b|\bm?ap\b|\bau[rp]?[oc]c?\b|\bauc\b|\brecall\b|\bprecision\b|\biou\b|\br@?\d|\br\d@|success\s*rate|\bols\b|\bpro\b|\bem\b',
    ),
)
_FAMILY_PATTERNS = tuple(
    (fam, hint, re.compile(rx, re.I)) for fam, hint, rx in _FAMILY_TABLE
)


def classify_metric_family(name: str) -> tuple[str, str] | None:
    """Return (family, bound_hint) for a metric name that looks like a recurring
    standard family, else None (bespoke -> needs a paper read). First match wins.
    Advisory only: bounds must be confirmed before registering (see the note on
    _FAMILY_TABLE and METRIC_MAINTENANCE.md)."""
    for fam, hint, rx in _FAMILY_PATTERNS:
        if rx.search(name):
            return fam, hint
    return None


def _report_unresolved(
    unresolved: dict[str, set[str]],
    reasons: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> str:
    reasons = reasons or {}
    recurring: list[str] = []  # look like a known family -> quick registry add
    bespoke: list[str] = []  # need a paper read -> auto-handled meanwhile
    ambiguous: list[str] = []
    for name, ds in sorted(unresolved.items()):
        why, candidates = reasons.get(name, ('unknown', ()))
        if why.startswith('ambiguous'):
            ambiguous.append(
                f'  - {name!r} (on {sorted(ds)}) — AMBIGUOUS: matches '
                f'{list(candidates)}'
            )
            continue
        fam = classify_metric_family(name)
        if fam is not None:
            recurring.append(
                f'  - {name!r} (on {sorted(ds)}) — family {fam[0]}: {fam[1]}'
            )
        else:
            bespoke.append(f'  - {name!r} (on {sorted(ds)})')

    blocks = [
        f'{len(unresolved)} metric(s) do not resolve in the registry snapshot '
        f'({SNAPSHOT_PATH.name}). Triage below; to register, edit '
        'eval-card-registry seed/metrics.yaml (see its `registry-entity-aliases` '
        'skill) and refresh the snapshot. Full runbook: METRIC_MAINTENANCE.md.'
    ]
    if recurring:
        blocks.append(
            f'RECURRING — look like a standard family ({len(recurring)}): register in '
            'eval-card-registry seed/metrics.yaml with the family bound/direction '
            'shown (CONFIRM it against the definition first), then refresh the '
            'snapshot (refresh_registry_snapshot.py) and re-run:\n'
            + '\n'.join(recurring)
        )
    if bespoke:
        blocks.append(
            f'BESPOKE — no family match, need a read of the defining paper ({len(bespoke)}): '
            "open each metric's paper (paper_url is kept in the source rows / prior "
            'output), deduce the bound; if the paper is not enough, register '
            'name-only with null bounds (or leave unresolved). --allow-unresolved '
            'emits them now with observed-range bounds (labelled bound_source) so '
            'data keeps flowing while you work through them:\n'
            + '\n'.join(bespoke)
        )
    if ambiguous:
        blocks.append(
            f'AMBIGUOUS — match more than one canonical id ({len(ambiguous)}): a '
            'duplicate alias/display_name in the registry, NOT a missing entry. Fix '
            'the collision in seed/metrics.yaml (remove or uniquify the offending '
            'alias) and refresh the snapshot:\n' + '\n'.join(ambiguous)
        )
    return '\n\n'.join(blocks)


def _summarize_class(title: str, items: dict[str, set[str]]) -> str:
    lines = [
        f'  - {name!r} (on {sorted(ds)})' for name, ds in sorted(items.items())
    ]
    return f'{len(items)} {title}:\n' + '\n'.join(lines)


def _imperfection_report(resolver: MetricResolver) -> str:
    """Human-readable summary of every imperfection in a run, across all classes,
    printed regardless of run mode (the "noisy" reporting the modes never turn
    off). Empty string when the run was perfect."""
    blocks = []
    if resolver.unresolved:
        blocks.append(
            _report_unresolved(resolver.unresolved, resolver.unresolved_reason)
        )
    if resolver.direction_unknown:
        blocks.append(
            _summarize_class(
                'metric(s) emitted with UNKNOWN direction (no registry direction '
                'and no PwC source direction; lower_is_better defaulted to False, '
                'flagged direction_source=unknown)',
                resolver.direction_unknown,
            )
        )
    if resolver.scale_anomalies:
        blocks.append(
            _summarize_class(
                'metric(s) with a SCALE ANOMALY (value(s) outside the canonical '
                'range with no consistent rescale, OR a whole group whose '
                'registered scale looks wrong -- group_scale_mismatch). Never '
                'guessed: the affected value is NOT published, it is listed in '
                'the failure report. A group_scale_mismatch usually means the '
                'metric is registered on the wrong scale: re-register it on its '
                'natural scale (METRIC_MAINTENANCE.md sec 4.3), refresh, re-run',
                resolver.scale_anomalies,
            )
        )
    # Informational only -- successful reconciliations, never fatal. They are the
    # data being "scaled the right way"; raw values are retained in every case.
    if resolver.scale_rescaled:
        blocks.append(
            _summarize_class(
                'metric(s) UNIFORMLY RESCALED to canonical (whole-group '
                'percent<->proportion; rescale_basis=group_uniform/group_mixed; '
                'informational, raw kept)',
                resolver.scale_rescaled,
            )
        )
    if resolver.scale_corrected:
        blocks.append(
            _summarize_class(
                'metric(s) with PER-ROW scale fixes (value(s) impossible under '
                'the declared range, uniquely resolved by x100//100; '
                'rescale_basis=per_row; informational, raw kept)',
                resolver.scale_corrected,
            )
        )
    return '\n\n'.join(blocks)


def run(args: argparse.Namespace) -> int:
    if getattr(args, 'emit_source_version', False):
        # The dump's date stamp is this source's version, the way a commit is
        # git's. Resolving it means only listing the bucket (no multi-GB
        # download), so the scheduler can decide to skip cheaply. A local --dump
        # names that file's stamp; otherwise the latest remote dump names it.
        if args.dump is not None:
            version = dump_version_from_path(args.dump)
        else:
            remote = args.remote_path or latest_dump_remote_path(args.bucket)
            version = dump_version_from_path(remote)
        print(version)
        return 0

    if args.dump is not None:
        dump_path = args.dump
        source_bucket: str | None = (
            None  # local dump -> no bucket provenance claim
        )
        pointer_kind = 'local_dump'
        source_reference = str(Path(dump_path).resolve())
    else:
        remote = args.remote_path or latest_dump_remote_path(args.bucket)
        dump_path = download_dump(args.bucket, remote, args.raw_dir)
        source_bucket = args.bucket
        pointer_kind = 'hf_bucket_dump'
        source_reference = f'{args.bucket}:{remote}'

    dump_version = dump_version_from_path(dump_path)
    dump_file = Path(dump_path).name
    retrieved_ts = retrieved_ts_from_dump(dump_version)

    # The dump is content-addressed by its date-stamped filename and stays put
    # in the bucket, so a pointer at that revision is the whole provenance; the
    # multi-GB payload itself would blow the sink's size caps and buys nothing.
    raw_capture.record_pointer(
        kind=pointer_kind,
        reference=source_reference,
        revision=dump_version,
        url=f'{PWC_SITE}',
        label='Papers with Code postgres dump',
        revision_required=True,
    )

    print(f'loading dump {dump_path} ...')
    dump = load_dump(dump_path)

    datasets_by_id = {d['id']: d for d in table_rows(dump, 'datasets')}
    tasks_by_id = {t['id']: t for t in table_rows(dump, 'tasks')}
    metric_dir: dict[str, str] = {}
    metric_meta: dict[str, dict[str, Any]] = {}
    for m in table_rows(dump, 'metrics'):
        metric_dir[m['name']] = m.get('direction')
        metric_meta[m['name']] = m

    if args.all:
        dataset_ids: set[Any] | None = None
    else:
        slugs = set(args.dataset_slugs or SAMPLE_DATASET_SLUGS)
        dataset_ids = {
            d['id'] for d in datasets_by_id.values() if d.get('slug') in slugs
        }
        missing = slugs - {datasets_by_id[i].get('slug') for i in dataset_ids}
        if missing:
            print(f'warning: dataset slug(s) not found: {sorted(missing)}')
        # An EXPLICIT selection that matches nothing is a user error (e.g. a
        # typo'd slug), not an empty-but-successful run: fail loudly rather than
        # writing zero records and exiting 0. A partial match keeps the warning
        # above and proceeds.
        if args.dataset_slugs and not dataset_ids:
            raise SystemExit(
                'ERROR: none of the requested --dataset-slug value(s) matched a '
                f'dataset in the dump: {sorted(slugs)}'
            )

    selected, metric_ranges, group_values = scan_evaluations(
        dump, dataset_ids, args.limit
    )
    print(f'selected {len(selected)} evaluation row(s)')

    paper_ids = {ev.get('paper_id') for ev in selected if ev.get('paper_id')}
    papers_by_id = read_papers_subset(dump, paper_ids)

    resolver = MetricResolver(pwc_directions=metric_dir)
    # Decide each leaderboard's reporting scale once, from its full value list
    # and the metric's canonical registry bounds (see analyze_group).
    group_scales = build_group_scales(group_values, resolver)
    conversion = build_logs(
        selected,
        datasets_by_id,
        tasks_by_id,
        resolver,
        metric_ranges,
        metric_meta,
        papers_by_id,
        dump_version,
        retrieved_ts,
        source_bucket=source_bucket,
        dump_file=dump_file,
        group_scales=group_scales,
    )
    bundles = conversion.records

    # --- Imperfection gate -------------------------------------------------
    # Two run modes, one report. The report ("noisy" output) is ALWAYS printed
    # when anything was imperfect, in either mode — modes decide whether to
    # ABORT, never whether to speak. Do this before publishing so an aborted run
    # leaves any prior output intact.
    report = _imperfection_report(resolver)
    if report:
        print(report, file=sys.stderr)
    # Which imperfection classes are fatal in strict (default) mode. Unresolved
    # is separately relaxable via --allow-unresolved (the narrow escape hatch);
    # direction_unknown and scale_anomaly are only waived by --best-effort.
    fatal = []
    if resolver.unresolved and not args.allow_unresolved:
        fatal.append(f'{len(resolver.unresolved)} unresolved metric(s)')
    if resolver.direction_unknown:
        fatal.append(
            f'{len(resolver.direction_unknown)} metric(s) with unknown direction'
        )
    if resolver.scale_anomalies:
        fatal.append(
            f'{len(resolver.scale_anomalies)} metric(s) with a scale anomaly'
        )
    if fatal and not args.best_effort:
        raise SystemExit(
            'ERROR: strict mode aborted — ' + '; '.join(fatal) + '. Fix these, '
            'or re-run with --best-effort to emit everything anyway (each '
            'imperfection stays flagged in the output), or --allow-unresolved '
            'to tolerate only the unresolved class. See the report above.'
        )
    if fatal and args.best_effort:
        print(
            'best-effort: emitting despite '
            + '; '.join(fatal)
            + ' (flagged in the output; a score that could not be placed on '
            'its canonical scale is omitted and listed in the failure report).',
            file=sys.stderr,
        )

    if not bundles:
        raise SystemExit(
            f'ERROR: none of the {conversion.total_records} selected '
            f'evaluation row(s) converted to a record. '
            + (
                f'First failure: {conversion.failures[0].reason}'
                if conversion.failures
                else 'No failures were recorded either — the selection was empty.'
            )
        )
    # Provenance for the omitted rows lands before publication, so a publication
    # failure cannot take the record of what was dropped with it.
    if conversion.failures or conversion.exclusions:
        report_path = save_failure_report(
            conversion, default_failure_report_path(args.output_dir)
        )
        print(
            f'{len(conversion.failures)} of {conversion.total_records} row(s) '
            f'did not convert; see {report_path}',
            file=sys.stderr,
        )

    # Replace, don't accumulate: uuid4 filenames mean a re-run would otherwise
    # pile up duplicate records. The whole batch is validated and written before
    # any record of the previous run is removed, so a failure here leaves that
    # run's output intact rather than a half-replaced directory. Supersession is
    # scoped to the evaluation ids this run re-emitted, so a partial selection
    # (e.g. the default two-dataset sample) never deletes the rest of an export.
    stale = existing_records(args.output_dir)
    written = save_evaluation_logs(
        [
            EvaluationLogOutput(
                eval_log=bundle.log,
                base_dir=args.output_dir,
                developer=bundle.developer,
                model_name=bundle.model,
            )
            for bundle in bundles
        ]
    )
    superseded_ids = {bundle.log.evaluation_id for bundle in bundles}
    removed = remove_superseded_records(
        stale, set(written), superseded_ids, args.output_dir
    )
    if removed:
        print(f'removed {removed} superseded record(s) from the previous run')
    total_results = sum(len(b.log.evaluation_results) for b in bundles)
    print(
        f'wrote {len(bundles)} model log(s), {total_results} result(s) '
        f'-> {args.output_dir}'
    )
    if resolver.unresolved:
        print(
            f'WARNING: {len(resolver.unresolved)} metric(s) used observed-range '
            f'fallback (--allow-unresolved): '
            f'{sorted(resolver.unresolved)}. Upstream them to the registry.'
        )
    if resolver.unbounded_emitted:
        print(
            f'NOTE: {sum(resolver.unbounded_emitted.values())} result(s) for '
            f'metric(s) {sorted(resolver.unbounded_emitted)} emitted with '
            f'unbounded (inf) bounds, serialized as "Infinity".'
        )
    # Every valid record is published first; only then does the run signal that
    # the conversion was partial.
    conversion.raise_if_incomplete()
    return len(bundles)


if __name__ == '__main__':
    run(parse_args())
    # then validate:  python -m every_eval_ever validate <output-dir>
