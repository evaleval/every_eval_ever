#!/usr/bin/env python3
"""Convert the BenchPress score matrix into Every Eval Ever records.

BenchPress (``microsoft/benchpress-score-matrix``) is an *aggregator*: it
re-reports model scores scraped from provider blogs, tech reports, model cards,
leaderboards and third-party aggregators, each cell carrying its own citation
(``reference_url``) and provenance (``source_type``). It is handled like the
``llm_stats`` adapter: ``source_type=documentation``, ``source_role=aggregator``,
and output logs are split by ``evaluator_relationship`` (derived per score).

Data source & updates
----------------------
BenchPress publishes its freshness manifest as ``metadata.json`` ("Export counts,
source commit, and matrix construction metadata" per the dataset README). This
adapter reads it as the version anchor: ``generated_at_utc`` becomes the record
``retrieved_timestamp``, and ``source_git_commit`` / ``generated_at_utc`` are
recorded on every record so a consumer can tell which BenchPress snapshot it came
from and re-run when the manifest changes. The four files are read at one pinned
dataset commit, which is recorded alongside them; ``--revision`` replays an
earlier snapshot.

Run
---
    uv run python -m every_eval_ever.adapters.benchpress.adapter --output-dir /tmp/eee-benchpress
    uv run python -m every_eval_ever validate '/tmp/eee-benchpress/*/*/*.json'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from pydantic import ValidationError

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_csv,
    fetch_json,
    require_finite_number,
    save_evaluation_logs,
    save_failure_report,
)

HF_REPO = 'microsoft/benchpress-score-matrix'
ATTRIBUTION_URL = f'https://huggingface.co/datasets/{HF_REPO}'
PAPER_URL = 'https://arxiv.org/abs/2606.24020'
DEFAULT_OUTPUT_DIR = 'data/benchpress'

# BenchPress score.audit_status -> whether BenchPress itself accepts the row.
# `dropped`, `needs_review` and `flagged` rows are excluded from its canonical
# matrix, so they are excluded here too unless --include-unaccepted is passed.
ACCEPTED_AUDIT_STATUSES = frozenset({'verified', 'verified_third_party'})

# source_types whose evaluator is independent of the scored model's provider,
# whatever else the citation contains.
INDEPENDENT_SOURCE_TYPES = frozenset({
    'leaderboard', 'third_party', 'third_party_aggregator', 'academic_paper',
})

# source_types a provider writes about its own models. The type alone says only
# what KIND of document a citation is; who published it has to come from the
# citation itself -- see relationship_from_score.
PROVIDER_AUTHORED_SOURCE_TYPES = frozenset({
    'official_blog', 'official_paper', 'model_card', 'tech_report',
})

# metric_type -> EEE metric_unit.
METRIC_UNIT = {
    'pct': 'percent', 'elo': 'points', 'rating': 'points', 'dollars': 'usd',
    'index': 'points', 'raw': 'points', 'bleu': 'points', 'wer': 'proportion',
}

# The metric's TRUE mathematical bounds; +/-inf where unbounded. A benchmark's
# declared `range` overrides these. Bounds are the one place EEE allows an
# unbounded value: MetricConfig serializes inf as the JSON string "Infinity".
INF = float('inf')
METRIC_BOUNDS = {
    'pct': (0.0, 100.0),     # bounded percentage
    'bleu': (0.0, 100.0),    # 0-100
    'wer': (0.0, INF),       # 0 floor; can exceed 1.0 (insertions)
    'dollars': (0.0, INF),   # cost: 0 floor, unbounded above
    'elo': (-INF, INF),      # ratings unbounded both ways
    'rating': (-INF, INF),
    'index': (-INF, INF),
    'raw': (-INF, INF),
}

# Recognized eval FRAMEWORKS that may appear in the free-text harness field.
RECOGNIZED_HARNESS = {
    'lm-evaluation-harness': 'lm-evaluation-harness', 'lm-eval': 'lm-evaluation-harness',
    'lm_eval': 'lm-evaluation-harness', 'olmes': 'OLMES', 'simple-evals': 'simple-evals',
    'opencompass': 'OpenCompass', 'mistral-eval': 'mistral-eval',
    'inspect_ai': 'inspect_ai', 'inspect': 'inspect_ai', 'helm': 'helm',
}

# tools tokens that map cleanly to EEE agentic tool entries.
_TOOL_TOKENS = {
    'code': [{'name': 'code'}], 'web': [{'name': 'web'}], 'file': [{'name': 'file'}],
    'all': [{'name': 'code'}, {'name': 'web'}, {'name': 'file'}],
}


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _str_map(pairs: dict) -> dict:
    """additional_details / details require str values; drop empties."""
    out = {}
    for key, value in pairs.items():
        if value is None or value == '':
            continue
        out[key] = value if isinstance(value, str) else json.dumps(value)
    return out


_OPEN_WEIGHTS_TRUE = frozenset({'true', 't', 'yes', 'y', '1', 'open', 'open_weights'})
_OPEN_WEIGHTS_FALSE = frozenset({'false', 'f', 'no', 'n', '0', 'closed', 'closed_weights'})


def _model_availability(open_weights: Any) -> str:
    """BenchPress's ``open_weights`` flag -> the schema's model_availability enum.

    Only a clear yes/no is mapped; an absent or unrecognized value stays
    ``unknown`` rather than being guessed in either direction.
    """
    value = _clean(open_weights)
    if value is None:
        return 'unknown'
    token = str(value).strip().lower()
    if token in _OPEN_WEIGHTS_TRUE:
        return 'open_weights'
    if token in _OPEN_WEIGHTS_FALSE:
        return 'closed_weights'
    return 'unknown'


def _slug(text: Any) -> str:
    return ''.join(c if c.isalnum() else '-' for c in str(text).strip().lower()).strip('-')


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).netloc or None


def _iso_to_epoch_str(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return repr(dt.timestamp())


def _clean(value):
    if isinstance(value, str):
        s = value.strip()
        return None if s == '' or s.lower() in {'nan', 'none', 'null'} else s
    return value


def _optional_number(value):
    """A descriptive numeric column, kept as written when it will not parse.

    These reach the record through _str_map, which stringifies either way, so a
    value the source wrote as text is preserved rather than lost -- and no score
    depends on it, so it is not worth ending the export over.
    """
    value = _clean(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _json_obj(value):
    value = _clean(value)
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        return {}


# --------------------------------------------------------------------------- #
# fetch / load  (the public CSV mirror + metadata.json freshness manifest)
# --------------------------------------------------------------------------- #

def _parse_models(rows: list[dict]) -> list[dict]:
    return [{
        'id': r['model_id'], 'name': r.get('model_name') or r['model_id'],
        'provider': _clean(r.get('provider')),
        'release_date': _clean(r.get('release_date')),
        'params_total_M': _optional_number(r.get('params_total_M')),
        'params_active_M': _optional_number(r.get('params_active_M')),
        'architecture': _clean(r.get('architecture')),
        'is_reasoning': _clean(r.get('is_reasoning')),
        'open_weights': _clean(r.get('open_weights')),
    } for r in rows]


def _parse_benchmarks(rows: list[dict]) -> list[dict]:
    return [{
        'id': r['benchmark_id'], 'name': r.get('benchmark_name') or r['benchmark_id'],
        'category': _clean(r.get('category')), 'metric': _clean(r.get('metric')),
        'num_problems': _optional_number(r.get('num_problems')),
        'source_url': _clean(r.get('source_url')),
        'canonical_setting': _json_obj(r.get('canonical_setting_json')),
    } for r in rows]


def _parse_scores(rows: list[dict]) -> list[dict]:
    # The score stays as the source wrote it: it is parsed per row in make_logs,
    # where a value that will not parse is that row's failure rather than the end
    # of the run. The id columns stay strict -- one missing from the header is a
    # structural mismatch, not one bad row. audit_status is strict for the same
    # reason: it gates accept/exclude, so a vanished column would silently exclude
    # every row and still exit 0 (an empty successful run) instead of failing loud.
    return [{
        'model_id': r['model_id'], 'benchmark_id': r['benchmark_id'],
        'score': _clean(r.get('score')),
        'reference_url': _clean(r.get('reference_url')),
        'source_type': _clean(r.get('source_type')),
        'audit_status': _clean(r['audit_status']),
        'matches_canonical': _clean(r.get('matches_canonical')),
        'reported_setting': _json_obj(r.get('reported_setting_json')),
        'notes': _clean(r.get('notes')),
        'n_candidates': _clean(r.get('n_candidates')),
    } for r in rows]


_FULL_SHA = re.compile(r'[0-9a-f]{40}')


def resolve_revision(reference: str | None = None) -> str:
    """The commit SHA for ``reference`` (default: the dataset's current tip).

    A branch or tag names whatever it points at now, so reading the four files at
    one would still mix revisions, and the SHA recorded on every record would be a
    ref that can move afterwards. A full SHA is already immutable and is taken as
    given, so pinning one costs no request.
    """
    if reference and _FULL_SHA.fullmatch(reference):
        return reference
    suffix = f'/revision/{quote(reference, safe="")}' if reference else ''
    info = fetch_json(f'https://huggingface.co/api/datasets/{HF_REPO}{suffix}')
    sha = info.get('sha')
    if not sha:
        raise RuntimeError(
            f'{HF_REPO} returned no commit sha for '
            f'{reference or "its current revision"}; pass --revision <sha> to '
            'pin one'
        )
    return sha


def fetch_payload(revision: str | None = None) -> dict[str, Any]:
    """Fetch the BenchPress CSV mirror + metadata.json at one pinned commit.

    ``main`` moves, and the four files are four requests, so reading them at a
    branch tip can mix revisions. Whatever is asked for -- the default tip, a
    branch, a tag -- resolves to one commit first, and every file is read at that
    commit; ``--revision`` reproduces an earlier snapshot.
    """
    revision = resolve_revision(revision)
    base = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}'
    metadata = fetch_json(f'{base}/metadata.json')
    return {
        'models': _parse_models(fetch_csv(f'{base}/data/models.csv')),
        'benchmarks': _parse_benchmarks(fetch_csv(f'{base}/data/benchmarks.csv')),
        'scores': _parse_scores(fetch_csv(f'{base}/data/scores_all.csv')),
        'metadata': {**metadata, 'dataset_revision': revision},
    }


def load_payload(input_json: Path) -> dict[str, Any]:
    """Replay a saved payload (already-parsed lists, as fetch_payload returns)."""
    data = json.loads(Path(input_json).read_text(encoding='utf-8'))
    return {key: data.get(key, [] if key != 'metadata' else {})
            for key in ('models', 'benchmarks', 'scores', 'metadata')}


# --------------------------------------------------------------------------- #
# record construction
# --------------------------------------------------------------------------- #

def _provider_publishes(url: str | None, provider: str | None) -> bool:
    """Whether a citation is hosted on a domain that carries the provider's name.

    Where a citation lives is the only publisher evidence this export offers. A
    hostname label equal to the provider's name -- ``openai.com``,
    ``cdn.amazon.science``, ``moonshotai.github.io`` -- names the provider as
    publisher, because a domain has an owner. A shared host names nobody,
    whatever its path says: the org in ``huggingface.co/Qwen/...`` sits in the
    path, and ``storage.googleapis.com`` serves anyone's bucket.
    """
    provider = _slug(provider or '').replace('-', '')
    host = _domain(url)
    if not provider or not host:
        return False
    labels = [label.replace('-', '') for label in host.lower().split('.')]
    return provider in set(labels) or provider == ''.join(labels)


def relationship_from_score(score: dict, model: dict) -> str:
    """Who evaluated the model, as far as the BenchPress export can establish it.

    ``source_type`` says what KIND of document a citation is -- a model card, a
    blog post, a tech report -- not who published it, and provider-authored
    documents routinely tabulate competitors' scores, which BenchPress scrapes
    too: on the current snapshot a Google ``gemini-2.5-flash`` score is cited to
    Qwen's model card. So ``first_party`` takes both a provider-authored type and
    a citation on the provider's own domain. Independent types are
    ``third_party``, and a document whose publisher the export does not identify
    is ``other`` rather than a guess in either direction.
    """
    source_type = score.get('source_type') or ''
    if source_type in INDEPENDENT_SOURCE_TYPES:
        return 'third_party'
    if source_type in PROVIDER_AUTHORED_SOURCE_TYPES and _provider_publishes(
            score.get('reference_url'), model.get('provider')):
        return 'first_party'
    return 'other'


def normalize_model_info(model: dict) -> tuple[ModelInfo, str, str]:
    """Return (ModelInfo, org_slug, model_slug). id = ``<org>/<benchpress slug>``;
    the registry resolves this to a canonical id downstream."""
    slug = model['id']
    provider = model.get('provider') or 'unknown'
    org = _slug(provider) or 'unknown'
    info = ModelInfo(
        name=model.get('name') or slug,
        id=f'{org}/{slug}',
        developer=provider,
        additional_details=_str_map({
            # deployment_type + model_availability are required in
            # additional_details (schema + validator's model-deployment check).
            # BenchPress records no serving platform, so deployment_type is
            # unknown; model_availability is derived from the open_weights flag,
            # whose raw value is kept alongside it.
            'deployment_type': 'unknown',
            'model_availability': _model_availability(model.get('open_weights')),
            'benchpress_model_id': slug,
            'release_date': model.get('release_date'),
            'params_total_M': model.get('params_total_M'),
            'params_active_M': model.get('params_active_M'),
            'architecture': model.get('architecture'),
            'is_reasoning': model.get('is_reasoning'),
            'open_weights': model.get('open_weights'),
        }),
    )
    return info, org, slug


def metric_bounds(benchmark: dict) -> tuple[float, float, str]:
    """(min_score, max_score, bound_strategy) = the metric's TRUE bounds.

    A benchmark's declared ``range`` wins; otherwise the per-family bounds in
    METRIC_BOUNDS (with +/-inf where the metric is genuinely unbounded);
    otherwise fully unbounded.
    """
    cs = benchmark.get('canonical_setting') or {}
    rng = cs.get('range')
    if (isinstance(rng, (list, tuple)) and len(rng) == 2
            and all(isinstance(x, (int, float)) for x in rng)):
        return float(rng[0]), float(rng[1]), 'declared_range'
    metric_type = cs.get('metric_type')
    if metric_type in METRIC_BOUNDS:
        lo, hi = METRIC_BOUNDS[metric_type]
        return lo, hi, 'metric_family_bounds'
    return -INF, INF, 'unbounded_default'


def _within_bounds(score: float, bounds: MetricConfig) -> bool:
    """Whether a score sits inside the bounds the record itself declares.

    BenchPress's ``canonical_setting.range`` is the benchmark's documented scale,
    but the export mixes scales inside one benchmark -- ``mt_bench_101`` declares
    1-10 and carries values up to 90.2. The validator rejects such a record, so
    the disagreement is reported as a failed source row rather than guessed at.
    """
    return bounds.min_score <= score <= bounds.max_score


def _generation_config(reported: dict) -> GenerationConfig | None:
    args: dict[str, Any] = {}
    temperature = reported.get('temperature')
    if isinstance(temperature, (int, float)):
        args['temperature'] = temperature
    mode = reported.get('mode')
    if mode == 'thinking':
        args['reasoning'] = True
    elif mode == 'non-thinking':
        args['reasoning'] = False
    tools = reported.get('tools')
    if tools in _TOOL_TOKENS:
        args['agentic_eval_config'] = {'available_tools': _TOOL_TOKENS[tools]}
    details = _str_map({
        'effort': reported.get('effort'), 'context': reported.get('context'),
        'prompt_style': reported.get('prompt_style'), 'mode': mode,
        'system_type': reported.get('system_type'),
        'temperature_raw': temperature if isinstance(temperature, str) else None,
        'tools_raw': tools if (tools and tools not in _TOOL_TOKENS) else None,
    })
    if not args and not details:
        return None
    cfg: dict[str, Any] = {}
    if args:
        cfg['generation_args'] = args
    if details:
        cfg['additional_details'] = details
    return GenerationConfig(**cfg)


def make_evaluation_result(score: dict, benchmark: dict) -> EvaluationResult | None:
    value = score.get('score')
    if value is None:
        return None
    # Parsed here, not when the CSV is read: inside make_logs's per-row boundary a
    # value that will not parse is attributed to the row it came from, instead of
    # ending the run before any row has been recorded.
    value = require_finite_number(value, f'{_score_ref(score)} score')
    cs = benchmark.get('canonical_setting') or {}
    reported = score.get('reported_setting') or {}
    metric_type = cs.get('metric_type')
    bslug = _slug(benchmark['id'])
    lo, hi, bound_strategy = metric_bounds(benchmark)

    ref_url = score.get('reference_url')
    dataset_url = benchmark.get('source_url')
    urls = [u for u in (ref_url, dataset_url) if u] or [ATTRIBUTION_URL]

    harness = reported.get('harness')
    harness_canon = None
    if harness:
        low = str(harness).lower()
        harness_canon = next((c for tok, c in RECOGNIZED_HARNESS.items() if tok in low), None)

    return EvaluationResult(
        evaluation_result_id=bslug,
        evaluation_name=f'benchpress.{bslug}',
        source_data=SourceDataUrl(
            dataset_name=benchmark.get('name') or benchmark['id'],
            source_type='url',
            url=urls,
            additional_details=_str_map({
                'source_role': 'aggregator',
                'reported_by': _domain(ref_url),
                'reference_url': ref_url,
                'dataset_url': dataset_url if (ref_url and dataset_url) else None,
                'num_problems': benchmark.get('num_problems'),
                'benchmark_version': cs.get('version'),
                'multimodal_input': cs.get('multimodal_input'),
                'benchmark_category': benchmark.get('category'),
            }) or None,
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f'{benchmark.get("name") or benchmark["id"]} score reported via BenchPress.'),
            metric_id=f'benchpress.{bslug}.score',
            metric_name=benchmark.get('metric') or (metric_type or 'score'),
            metric_kind=metric_type or 'score',
            metric_unit=METRIC_UNIT.get(metric_type, 'points'),
            lower_is_better=(cs.get('higher_is_better') is False),
            score_type=ScoreType.continuous,
            min_score=lo,
            max_score=hi,
            additional_details=_str_map({
                'bound_strategy': bound_strategy,
                'benchpress_metric_type': metric_type,
                'benchpress_harness': harness,
                'eval_framework': harness_canon,
                'sampling': reported.get('sampling'),
                'judge': reported.get('judge'),
                'benchmark_notes': cs.get('notes'),
            }) or None,
        ),
        score_details=ScoreDetails(
            score=value,
            details=_str_map({
                'benchpress_source_type': score.get('source_type'),
                'reference_url': ref_url,
                'audit_status': score.get('audit_status'),
                'matches_canonical': score.get('matches_canonical'),
                'n_candidates': score.get('n_candidates'),
                'notes': score.get('notes'),
            }) or None,
        ),
        generation_config=_generation_config(reported),
    )


def source_metadata(relationship: str, version: dict) -> SourceMetadata:
    """`version` carries the BenchPress freshness manifest (metadata.json), so
    each record records the snapshot it came from (update tracking)."""
    return SourceMetadata(
        source_name=f'BenchPress Score Matrix: {relationship} scores',
        source_type='documentation',
        source_organization_name='BenchPress',
        source_organization_url=ATTRIBUTION_URL,
        evaluator_relationship=EvaluatorRelationship(relationship),
        additional_details=_str_map({
            'benchpress_publisher': 'Microsoft',
            'dataset_url': ATTRIBUTION_URL,
            'paper_url': PAPER_URL,
            'attribution_required': 'true',
            'source_role': 'aggregator',
            # freshness/version anchor from metadata.json (BenchPress's documented
            # update manifest) -> lets consumers detect a new snapshot.
            'benchpress_source_git_commit': version.get('source_git_commit'),
            'benchpress_generated_at_utc': version.get('generated_at_utc'),
            'benchpress_source_data_dirty': version.get('source_data_dirty'),
            # The manifest can lag the CSVs, so the commit every file was read
            # at is recorded too -- that is what --revision reproduces.
            'benchpress_dataset_revision': version.get('dataset_revision'),
        }),
    )


def _score_ref(score: dict) -> str:
    return f'{score.get("model_id")}/{score.get("benchmark_id")}'


def make_logs(payload: dict[str, Any],
              retrieved_timestamp: str | None = None,
              include_unaccepted: bool = False,
              ) -> SourceConversionResult[LogBundle]:
    models = {m['id']: m for m in payload['models']}
    benchmarks = {b['id']: b for b in payload['benchmarks']}
    version = payload.get('metadata') or {}

    timestamp = retrieved_timestamp
    if timestamp is None and version.get('generated_at_utc'):
        timestamp = _iso_to_epoch_str(version['generated_at_utc'])
    timestamp = timestamp or str(time.time())

    # The retrieved_timestamp comes from the manifest, which can lag the CSVs, so
    # two content-differing snapshots can share one timestamp. The dataset commit
    # is immutable per snapshot, so it anchors the evaluation_id's uniqueness.
    revision = version.get('dataset_revision') or 'unknown-revision'

    exclusions: list[SourceRecordExclusion] = []
    failures: list[SourceRecordFailure] = []
    # (developer, model, relationship) -> benchmark id -> result. Keyed by the
    # join key so a second, different result for one benchmark is reported
    # instead of quietly losing to the first.
    groups: dict[
        tuple[str, str, str], dict[str | None, EvaluationResult]
    ] = defaultdict(dict)
    model_infos: dict[tuple[str, str, str], ModelInfo] = {}
    for score in payload['scores']:
        audit_status = score.get('audit_status') or 'missing'
        if not include_unaccepted and audit_status not in ACCEPTED_AUDIT_STATUSES:
            exclusions.append(SourceRecordExclusion(
                source_ref=_score_ref(score),
                reason=(f'BenchPress audit_status={audit_status!r} is outside its '
                        'own accepted set; pass --include-unaccepted to export it'),
                source_record=score,
            ))
            continue
        model = models.get(score['model_id'])
        benchmark = benchmarks.get(score['benchmark_id'])
        try:
            result = (make_evaluation_result(score, benchmark)
                      if model is not None and benchmark is not None else None)
            identity = None if result is None else normalize_model_info(model)
        except (ValueError, TypeError, ValidationError) as exc:
            # Only what an unusable source value raises: a number that will not
            # parse, and a field the schema rejects. Anything else is a bug in
            # this adapter and stays visible instead of being filed as bad data.
            failures.append(SourceRecordFailure(
                source_ref=_score_ref(score),
                reason=f'{type(exc).__name__}: {exc}',
                source_record=score,
            ))
            continue
        if result is None:
            failures.append(SourceRecordFailure(
                source_ref=_score_ref(score),
                reason='no score, or the model/benchmark id is not in this export',
                source_record=score,
            ))
            continue
        bounds = result.metric_config
        if not _within_bounds(result.score_details.score, bounds):
            failures.append(SourceRecordFailure(
                source_ref=_score_ref(score),
                reason=(f'score {result.score_details.score} is outside the '
                        f'benchmark\'s declared range '
                        f'[{bounds.min_score}, {bounds.max_score}], so the two '
                        'disagree about the scale and neither can be trusted'),
                source_record=score,
            ))
            continue
        model_info, org, slug = identity
        relationship = relationship_from_score(score, model)
        key = (org, slug, relationship)
        kept = groups[key]
        previous = kept.get(result.evaluation_result_id)
        if previous is not None:
            if previous.model_dump() == result.model_dump():
                # The same cell reported twice: nothing new to convert, but the
                # row is still accounted as an exclusion so converted + failed +
                # excluded reconciles with total_records.
                exclusions.append(SourceRecordExclusion(
                    source_ref=_score_ref(score),
                    reason=(f'exact duplicate of {result.evaluation_result_id} '
                            f'already reported for this model as {relationship}'),
                    source_record=score,
                ))
                continue
            failures.append(SourceRecordFailure(
                source_ref=_score_ref(score),
                reason=(f'{result.evaluation_result_id} is already reported for '
                        f'this model as {relationship}, citing '
                        f'{previous.source_data.url[0]}; evaluation_result_id is the '
                        'join key for instance records, so one log cannot carry '
                        'two different results for one benchmark'),
                source_record=score,
            ))
            continue
        kept[result.evaluation_result_id] = result
        model_infos[key] = model_info

    bundles: list[LogBundle] = []
    for (org, slug, relationship), results in sorted(groups.items()):
        model_info = model_infos[(org, slug, relationship)]
        sanitized = model_info.id.replace('/', '_')
        ordered = sorted(results.values(),
                         key=lambda r: r.evaluation_result_id or '')
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=f'benchpress/{relationship}/{sanitized}/{timestamp}/{revision}',
            retrieved_timestamp=timestamp,
            source_metadata=source_metadata(relationship, version),
            eval_library=EvalLibrary(name='BenchPress', version='unknown'),
            model_info=model_info,
            evaluation_results=ordered,
        )
        bundles.append(LogBundle(log=log, developer=org, model=slug))
    return SourceConversionResult(
        source_name='BenchPress score matrix',
        total_records=len(payload['scores']),
        records=bundles,
        failures=failures,
        exclusions=exclusions,
    )


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def write_conversion_report(result: SourceConversionResult[LogBundle],
                            output_dir: Path) -> Path:
    """Persist this run's accounting, replacing any previous run's copy in one step.

    Called on every run, a clean one included, and before publication.
    """
    final = default_failure_report_path(output_dir)
    staged = save_failure_report(result, final.with_name(final.name + '.tmp'))
    # Staged then renamed: an interrupted write must not truncate the report
    # a complete run left behind.
    os.replace(staged, final)
    return final


def export_logs(bundles: list[LogBundle], output_dir: Path) -> list[Path]:
    """Publish every log in one batch, so a late failure leaves no partial tree."""
    return save_evaluation_logs([
        EvaluationLogOutput(
            eval_log=bundle.log,
            base_dir=output_dir,
            developer=bundle.developer,
            model_name=bundle.model,
        )
        for bundle in bundles
    ])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert the BenchPress score matrix to EEE.')
    parser.add_argument('--input-json', type=Path, default=None,
                        help='Replay a saved payload offline instead of fetching from HF.')
    parser.add_argument('--save-raw-json', type=Path, default=None,
                        help='Write the fetched payload here (must be OUTSIDE --output-dir).')
    parser.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--retrieved-timestamp', default=None,
                        help='Override the epoch timestamp (default: metadata.generated_at_utc).')
    parser.add_argument('--revision', default=None,
                        help='Dataset commit to read (default: the current one).')
    parser.add_argument('--include-unaccepted', action='store_true',
                        help='Also export scores BenchPress marks dropped, '
                             'needs_review or flagged (excluded by default).')
    return parser.parse_args(argv)


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def run(args: argparse.Namespace) -> int:
    if args.save_raw_json is not None and _is_subpath(args.save_raw_json, args.output_dir):
        raise SystemExit('--save-raw-json must point outside --output-dir.')
    payload = (load_payload(args.input_json) if args.input_json
               else fetch_payload(args.revision))
    if args.save_raw_json is not None:
        args.save_raw_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_raw_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    result = make_logs(
        payload,
        retrieved_timestamp=args.retrieved_timestamp,
        include_unaccepted=args.include_unaccepted,
    )
    report = write_conversion_report(result, args.output_dir)
    paths = export_logs(result.records, args.output_dir)
    for path in paths:
        print(path)
    print(f'{result.total_records} source scores -> {len(paths)} log(s); '
          f'{len(result.exclusions)} excluded, {len(result.failures)} failed')
    print(f'Conversion accounting: {report}')
    # Only failures fail the run: an excluded row is BenchPress's own decision.
    # The report itemizes both.
    result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} BenchPress model log(s).')
