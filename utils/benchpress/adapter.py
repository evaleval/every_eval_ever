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
from and re-run when the manifest changes.

Run
---
    uv run python -m utils.benchpress.adapter --output-dir /tmp/eee-benchpress
    uv run python -m every_eval_ever validate /tmp/eee-benchpress
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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
    fetch_csv,
    fetch_json,
    generate_output_path,
)

HF_REPO = 'microsoft/benchpress-score-matrix'
HF_BASE = f'https://huggingface.co/datasets/{HF_REPO}/raw/main'
ATTRIBUTION_URL = f'https://huggingface.co/datasets/{HF_REPO}'
PAPER_URL = 'https://arxiv.org/abs/2606.24020'
DEFAULT_OUTPUT_DIR = 'data/benchpress'

# BenchPress score.source_type -> EEE evaluator_relationship (derived per score).
# Provider-authored sources are first_party; independent ones third_party;
# blank/unknown -> other.
RELATIONSHIP_BY_SOURCE_TYPE = {
    'official_blog': 'first_party',
    'tech_report': 'first_party',
    'official_paper': 'first_party',
    'model_card': 'first_party',
    'leaderboard': 'third_party',
    'third_party': 'third_party',
    'third_party_aggregator': 'third_party',
    'academic_paper': 'third_party',
}

# metric_type -> EEE metric_unit.
METRIC_UNIT = {
    'pct': 'percent', 'elo': 'points', 'rating': 'points', 'dollars': 'usd',
    'index': 'points', 'raw': 'points', 'bleu': 'points', 'wer': 'proportion',
}

# The metric's TRUE mathematical bounds; +/-inf where unbounded. A benchmark's
# declared `range` overrides these. inf is serialized as the JSON `Infinity`
# token (see _write_log) which EEE's loader (json.loads + pydantic) reads back as
# float('inf').
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


def _slug(text: Any) -> str:
    return ''.join(c if c.isalnum() else '-' for c in str(text).strip().lower()).strip('-')


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlparse
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


def _to_float(value):
    value = _clean(value)
    return None if value is None else float(value)


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
        'params_total_M': _to_float(r.get('params_total_M')),
        'params_active_M': _to_float(r.get('params_active_M')),
        'architecture': _clean(r.get('architecture')),
        'is_reasoning': _clean(r.get('is_reasoning')),
        'open_weights': _clean(r.get('open_weights')),
    } for r in rows]


def _parse_benchmarks(rows: list[dict]) -> list[dict]:
    return [{
        'id': r['benchmark_id'], 'name': r.get('benchmark_name') or r['benchmark_id'],
        'category': _clean(r.get('category')), 'metric': _clean(r.get('metric')),
        'num_problems': _to_float(r.get('num_problems')),
        'source_url': _clean(r.get('source_url')),
        'canonical_setting': _json_obj(r.get('canonical_setting_json')),
    } for r in rows]


def _parse_scores(rows: list[dict]) -> list[dict]:
    return [{
        'model_id': r['model_id'], 'benchmark_id': r['benchmark_id'],
        'score': _to_float(r.get('score')),
        'reference_url': _clean(r.get('reference_url')),
        'source_type': _clean(r.get('source_type')),
        'audit_status': _clean(r.get('audit_status')),
        'matches_canonical': _clean(r.get('matches_canonical')),
        'reported_setting': _json_obj(r.get('reported_setting_json')),
        'notes': _clean(r.get('notes')),
        'n_candidates': _clean(r.get('n_candidates')),
    } for r in rows]


def fetch_payload() -> dict[str, Any]:
    """Fetch the live BenchPress CSV mirror + metadata.json from HuggingFace."""
    try:
        metadata = fetch_json(f'{HF_BASE}/metadata.json')
    except Exception:  # noqa: BLE001 - metadata is best-effort (version anchor only)
        metadata = {}
    return {
        'models': _parse_models(fetch_csv(f'{HF_BASE}/data/models.csv')),
        'benchmarks': _parse_benchmarks(fetch_csv(f'{HF_BASE}/data/benchmarks.csv')),
        'scores': _parse_scores(fetch_csv(f'{HF_BASE}/data/scores_all.csv')),
        'metadata': metadata,
    }


def load_payload(input_json: Path) -> dict[str, Any]:
    """Replay a saved payload (already-parsed lists, as fetch_payload returns)."""
    data = json.loads(Path(input_json).read_text(encoding='utf-8'))
    return {key: data.get(key, [] if key != 'metadata' else {})
            for key in ('models', 'benchmarks', 'scores', 'metadata')}


# --------------------------------------------------------------------------- #
# record construction
# --------------------------------------------------------------------------- #

def relationship_from_score(score: dict) -> str:
    return RELATIONSHIP_BY_SOURCE_TYPE.get(score.get('source_type') or '', 'other')


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
            'benchpress_model_id': slug,
            'release_date': model.get('release_date'),
            'params_total_M': model.get('params_total_M'),
            'params_active_M': model.get('params_active_M'),
            'architecture': model.get('architecture'),
            'is_reasoning': model.get('is_reasoning'),
            'open_weights': model.get('open_weights'),
        }) or None,
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
        }),
    )


def make_logs(payload: dict[str, Any],
              retrieved_timestamp: str | None = None) -> list[LogBundle]:
    models = {m['id']: m for m in payload['models']}
    benchmarks = {b['id']: b for b in payload['benchmarks']}
    version = payload.get('metadata') or {}

    timestamp = retrieved_timestamp
    if timestamp is None and version.get('generated_at_utc'):
        timestamp = _iso_to_epoch_str(version['generated_at_utc'])
    timestamp = timestamp or str(time.time())

    groups: dict[tuple[str, str, str], list[EvaluationResult]] = defaultdict(list)
    model_infos: dict[tuple[str, str, str], ModelInfo] = {}
    for score in payload['scores']:
        model = models.get(score['model_id'])
        benchmark = benchmarks.get(score['benchmark_id'])
        if model is None or benchmark is None or score.get('score') is None:
            continue
        result = make_evaluation_result(score, benchmark)
        if result is None:
            continue
        model_info, org, slug = normalize_model_info(model)
        key = (org, slug, relationship_from_score(score))
        groups[key].append(result)
        model_infos[key] = model_info

    bundles: list[LogBundle] = []
    for (org, slug, relationship), results in sorted(groups.items()):
        model_info = model_infos[(org, slug, relationship)]
        sanitized = model_info.id.replace('/', '_')
        # de-dup by result id (one benchmark appears at most once per log)
        seen: set[str] = set()
        deduped: list[EvaluationResult] = []
        for result in sorted(results, key=lambda r: r.evaluation_result_id or ''):
            if result.evaluation_result_id in seen:
                continue
            seen.add(result.evaluation_result_id or '')
            deduped.append(result)
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=f'benchpress/{relationship}/{sanitized}/{timestamp}',
            retrieved_timestamp=timestamp,
            source_metadata=source_metadata(relationship, version),
            eval_library=EvalLibrary(name='BenchPress', version='unknown'),
            model_info=model_info,
            evaluation_results=deduped,
        )
        bundles.append(LogBundle(log=log, developer=org, model=slug))
    return bundles


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f'Not JSON serializable: {type(obj).__name__}')


def write_log(log: EvaluationLog, base_dir: Path, developer: str, model: str) -> Path:
    """Write one log as ``<base>/<dev>/<model>/<uuid>.json``.

    Mirrors helpers.save_evaluation_log's layout, but serializes with
    ``allow_nan=True`` so unbounded metric bounds (``float('inf')``) are written
    as the ``Infinity`` token. EEE's loader (json.loads + pydantic) reads those
    back as ``float('inf')``; pydantic's ``model_dump_json`` would instead null
    them, which is why this adapter writes the JSON itself.
    """
    dir_path = generate_output_path(base_dir, developer, model)
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / f'{uuid.uuid4()}.json'
    data = log.model_dump(mode='python', exclude_none=True)
    filepath.write_text(
        json.dumps(data, indent=2, allow_nan=True, default=_json_default),
        encoding='utf-8',
    )
    return filepath


def export_logs(bundles: list[LogBundle], output_dir: Path) -> list[Path]:
    return [write_log(b.log, output_dir, b.developer, b.model) for b in bundles]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert the BenchPress score matrix to EEE.')
    parser.add_argument('--input-json', type=Path, default=None,
                        help='Replay a saved payload offline instead of fetching from HF.')
    parser.add_argument('--save-raw-json', type=Path, default=None,
                        help='Write the fetched payload here (must be OUTSIDE --output-dir).')
    parser.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--retrieved-timestamp', default=None,
                        help='Override the epoch timestamp (default: metadata.generated_at_utc).')
    return parser.parse_args()


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def run(args: argparse.Namespace) -> int:
    if args.save_raw_json is not None and _is_subpath(args.save_raw_json, args.output_dir):
        raise SystemExit('--save-raw-json must point outside --output-dir.')
    payload = load_payload(args.input_json) if args.input_json else fetch_payload()
    if args.save_raw_json is not None:
        args.save_raw_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_raw_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    bundles = make_logs(payload, retrieved_timestamp=args.retrieved_timestamp)
    paths = export_logs(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} BenchPress model log(s).')
