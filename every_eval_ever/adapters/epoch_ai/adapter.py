#!/usr/bin/env python3
"""Convert Epoch AI's Benchmarking Hub data into Every Eval Ever records.

Epoch AI publishes the data behind its Benchmarking Hub and the Epoch
Capabilities Index as ``benchmark_data.zip``: one CSV per benchmark plus
``benchmark_metadata.csv`` (which CSV and which column is each benchmark's
headline score, and on what scale) and ``model_metadata.csv``. Each CSV row is
one (model version, benchmark) result. Rows in the CSVs not named ``*_external``
are results Epoch ran itself; ``*_external`` rows are results Epoch compiled
from a leaderboard, paper or announcement and carry their own ``Source``,
``Source link`` and ``Notes``.

One log is written per converted source row, so every row keeps its own
provenance. The headline score column is the one ``benchmark_metadata.csv``
names; the score is recorded on the scale Epoch gives it. ECI values
(``epoch_capabilities_index/``) are not converted: this adapter is about the
benchmark scores.

Licensing (https://epoch.ai/benchmarks/use-this-data#licensing): Epoch AI's
own data is CC BY 4.0, and data Epoch sourced from external projects "retains
its original licensing". Epoch-run records carry ``license: CC-BY-4.0``;
external records carry no blanket licence, only Epoch's contribution licence
and a note pointing at the original source. Epoch's attribution is on every
record.

Archive-level facts (the zip's checksum and snapshot date) are not written into
the records: the cron's raw capture keeps the zip and its sha256, and keeping
them out of the records lets an unchanged result fingerprint identically when
unrelated archive content changes.

Run
---
    uv run python -m every_eval_ever.adapters.epoch_ai.adapter --output-dir /tmp/epoch-ai/data
    uv run python -m every_eval_ever validate '/tmp/epoch-ai/data/epoch-ai-benchmarking/*/*/*.json'
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import requests

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    require_finite_number,
    save_failure_report,
)
from every_eval_ever.helpers.eval_card_registry import (
    Registry,
    hf_namespace_of,
)
from every_eval_ever.helpers.io import datastore_path_components

SRC = 'epoch-ai'
COLLECTION = 'epoch-ai-benchmarking'
DATA_URL = 'https://epoch.ai/data/benchmark_data.zip'
HUB_URL = 'https://epoch.ai/benchmarks'
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
LICENSE_TERMS_URL = 'https://epoch.ai/benchmarks/use-this-data#licensing'
EXTERNAL_LICENSE_NOTE = (
    'Compiled by Epoch AI from the external source named in epoch_source; '
    'external data retains its original licensing. Epoch AI\'s own '
    'contribution is CC BY 4.0.'
)
# https://epoch.ai/benchmarks/about: "Our internal evaluations are powered by
# Inspect". The version is not published.
EPOCH_RUN_LIBRARY = 'inspect_ai'
USER_AGENT = 'every-eval-ever-epoch-adapter'
CITATION = (
    "Epoch AI, 'Capabilities & benchmarking'. Published online at epoch.ai. "
    "Retrieved from 'https://epoch.ai/benchmarks' [online resource]."
)

NON_BENCHMARK_TABLES = frozenset({
    'benchmark_metadata.csv', 'model_metadata.csv',
    'epoch_capabilities_index/eci_scores.csv',
    'epoch_capabilities_index/edi_scores.csv',
    'epoch_capabilities_index/processed_data_for_eci.csv',
})

# Columns that say where an external score came from, or where Epoch's own
# run logs are. They go on the log's source_metadata, not on the score.
PROVENANCE_COLUMNS = (
    'Source', 'Source link', 'Source Link', 'Source link (site from table)',
    'Notes', 'Notes (details)', 'Log viewer', 'Logs',
)
# Columns that describe the model rather than this result.
MODEL_COLUMNS = (
    'Model version', 'Organization', 'Country', 'Release date',
    'Training compute (FLOP)', 'Training compute notes',
)

# Leading path segments of a model version that name the serving provider,
# not the model's namespace (``chutes/Qwen3-8B``,
# ``accounts/fireworks/models/glm-4p6``).
INFERENCE_PROVIDERS = frozenset(
    {'chutes', 'fireworks', 'deepinfra', 'hfendpoint', 'together',
     'openrouter', 'groq'}
)

# Collaborative models whose published repository is not under the first
# organization Epoch lists. base model -> (model id namespace, the Epoch
# organization that is the developer). Each namespace was checked against the
# model's Hugging Face repository; the developer is one of the organizations
# Epoch lists for the model.
MODEL_ID_OVERRIDES = {
    'bloom': ('bigscience', 'BigScience'),
    'starcoder2-3b': ('bigcode', 'BigCode'),
    'starcoder2-7b': ('bigcode', 'BigCode'),
    'starcoder2-15b': ('bigcode', 'BigCode'),
    'vicuna-13b-v1.3': ('lmsys', 'Large Model Systems Organization'),
    'Eurus-2-7B-PRIME': ('PRIME-RL', 'Tsinghua University'),
    # Epoch lists Google before Google DeepMind for this one Gemini version
    # only; the developer is Google DeepMind, as for every other Gemini model.
    'gemini-2.5-deep-think-2025-08-01-webapp': (None, 'Google DeepMind'),
}

# Score columns whose meaning is a global metric (schema: "use a canonical
# global id when applicable"). column -> (metric_id, metric_kind, parameters).
# Every other column keeps a benchmark-namespaced id.
GLOBAL_METRICS = {
    'Accuracy': ('accuracy', 'accuracy', None),
    'Overall accuracy': ('accuracy', 'accuracy', None),
    'EM': ('exact_match', 'exact_match', None),
    'Pass@1 score': ('pass_at_k', 'pass_rate', {'k': 1}),
}

# Suffixes Epoch appends to a model version for the reasoning setting. They
# are a variant of one model, so they come off the model id and are kept in
# additional_details and in evaluation_id.
EFFORT_SUFFIXES = frozenset(
    {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'unknown'}
)
_BUDGET_SUFFIX = re.compile(r'^\d+[Kk]$')

# model_metadata.csv `accessibility` -> (model_availability, deployment_type).
# Open weights say nothing about how Epoch served them, hence `unknown`.
ACCESSIBILITY = {
    'API access': ('closed_weights', 'externally_managed'),
    'Hosted access (no API)': ('closed_weights', 'externally_managed'),
    'Limited access': ('closed_weights', 'unknown'),
    'Unreleased': ('closed_weights', 'unknown'),
    'Open weights (unrestricted)': ('open_weights', 'unknown'),
    'Open weights (restricted use)': ('open_weights', 'unknown'),
    'Open weights (non-commercial)': ('open_weights', 'unknown'),
}


class Dataset:
    """The tables inside one ``benchmark_data.zip`` snapshot."""

    def __init__(self, tables: dict[str, list[dict[str, str]]],
                 snapshot: str, sha256: str, origin: str) -> None:
        self.tables = tables
        self.snapshot = snapshot
        self.sha256 = sha256
        self.origin = origin


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def load_zip(content: bytes, origin: str) -> Dataset:
    """Parse every CSV in the archive; snapshot = newest member date."""
    tables: dict[str, list[dict[str, str]]] = {}
    newest = ''
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.csv'):
                continue
            newest = max(newest, '%04d-%02d-%02d' % info.date_time[:3])
            text = archive.read(info).decode('utf-8-sig')
            tables[info.filename] = list(csv.DictReader(io.StringIO(text)))
    return Dataset(tables, newest, hashlib.sha256(content).hexdigest(), origin)


def fetch_zip(url: str = DATA_URL) -> bytes:
    response = requests.get(
        url, timeout=120, headers={'User-Agent': USER_AGENT}
    )
    response.raise_for_status()
    raw_capture.record(
        url=url, content=response.content,
        content_type=response.headers.get('Content-Type'),
    )
    return response.content


def split_variant(model_version: str) -> tuple[str, str | None]:
    """Return (base model, reasoning-setting suffix or None)."""
    base, sep, suffix = model_version.rpartition('_')
    if sep and base and (
        suffix in EFFORT_SUFFIXES or _BUDGET_SUFFIX.fullmatch(suffix)
    ):
        return base, suffix
    return model_version, None


def _alias_key(name: str) -> str:
    """Spelling-insensitive key: case, separators, and Fireworks' ``4p6``."""
    key = re.sub(r'[\s_]+', '-', name.strip().lower())
    return re.sub(r'(?<=\d)p(?=\d)', '.', key)


def _model_groups(dataset: Dataset) -> dict[str, set[str]]:
    """Epoch model group -> the base model versions (variant suffix removed)."""
    groups: dict[str, set[str]] = {}
    for row in dataset.tables.get('model_metadata.csv', []):
        version, group = _clean(row.get('model_version')), _clean(
            row.get('model_group'))
        if version and group:
            groups.setdefault(group, set()).add(split_variant(version)[0])
    return groups


def canonical_model_name(
    base: str, group_versions: set[str],
) -> tuple[str, dict[str, str]]:
    """Return (model name for the id, provenance of how it was chosen).

    A version qualified with a path (``openai/gpt-oss-120b``,
    ``chutes/Qwen3-8B``, ``accounts/fireworks/models/glm-4p6``) keeps only its
    last segment; a leading serving provider is recorded. The name is then
    resolved, through Epoch's own model group, to the group's unqualified
    spelling of the same model where one exists (``kimi-k2p5`` ->
    ``kimi-k2.5``), else to a qualified sibling spelled as the normalized name.
    """
    details: dict[str, str] = {}
    segments = [s for s in base.split('/') if s]
    name = segments[-1] if segments else base
    if len(segments) > 1:
        leading = segments[:-1]
        details['epoch_version_prefix'] = '/'.join(leading)
        provider = next(
            (s.lower() for s in leading if s.lower() in INFERENCE_PROVIDERS),
            None,
        )
        if provider:
            details['inference_provider'] = provider
    key = _alias_key(name)
    plain = sorted(
        v for v in group_versions if '/' not in v and _alias_key(v) == key
    )
    qualified = sorted(
        v.rsplit('/', 1)[-1] for v in group_versions
        if '/' in v and v.rsplit('/', 1)[-1].lower() == key
    )
    chosen = plain[0] if plain else qualified[0] if qualified else name
    if chosen != base:
        details['model_name_resolution'] = (
            'epoch_model_group' if chosen != name else 'version_path'
        )
    return chosen, details


def _model_metadata(dataset: Dataset) -> dict[str, dict[str, str]]:
    """First non-blank group/accessibility/organization per model version."""
    out: dict[str, dict[str, str]] = {}
    for row in dataset.tables.get('model_metadata.csv', []):
        version = _clean(row.get('model_version'))
        if version is None:
            continue
        entry = out.setdefault(version, {})
        for key in ('model_group', 'organization', 'accessibility'):
            if key not in entry and _clean(row.get(key)):
                entry[key] = _clean(row[key])
    # Epoch states a model version's organization in whichever table has it;
    # a blank in one row is filled from another row of the same version.
    for name, rows in sorted(dataset.tables.items()):
        for row in rows:
            version = _clean(row.get('Model version'))
            org = _clean(row.get('Organization'))
            if version and org:
                out.setdefault(version, {}).setdefault('organization', org)
    return out


def _resolve_org(registry: Registry, name: str):
    """Resolve an organization name, retrying without a parenthetical or a
    trailing corporate word (``Z.ai (Zhipu AI)``, ``Xiaomi Corp``)."""
    queries = [name]
    for query in (
        re.sub(r'\s*\(.*?\)', '', name).strip(),
        re.sub(r'\s+(Corp|Inc|AI|Labs)$', '', name).strip(),
    ):
        if query and query not in queries:
            queries.append(query)
    resolution = registry.org(name)
    for query in queries[1:]:
        if resolution.resolved:
            break
        resolution = registry.org(query)
    return resolution


def build_model_info(
    row: dict[str, str], meta: dict[str, str], registry: Registry,
    groups: dict[str, set[str]] | None = None,
) -> tuple[ModelInfo, str, str, str | None]:
    """Return (ModelInfo, developer prefix, model slug, variant suffix).

    Raises ValueError when the row names no model or no organization.
    """
    version = _clean(row.get('Model version'))
    if version is None:
        raise ValueError(
            'no Model version: Epoch has not attached this row to a model'
        )
    organization = _clean(row.get('Organization')) or meta.get('organization')
    if organization is None:
        raise ValueError(
            f'no Organization for model version {version!r} in the row or '
            'in model_metadata.csv'
        )
    organizations = [o.strip() for o in organization.split(',') if o.strip()]
    raw_base, variant = split_variant(version)
    base, name_details = canonical_model_name(
        raw_base, (groups or {}).get(meta.get('model_group', ''), set())
    )
    availability, deployment = ACCESSIBILITY.get(
        meta.get('accessibility', ''), ('unknown', 'unknown')
    )
    namespace, developer = MODEL_ID_OVERRIDES.get(base, (None, None))
    if developer is not None and developer not in organizations:
        raise ValueError(
            f'model id override for {base!r} names {developer!r}, which Epoch '
            f'does not list among {organizations}'
        )
    primary = developer or organizations[0]
    resolution = _resolve_org(registry, primary)
    org_id = resolution.canonical_id or _slug(primary)
    prefix = org_id
    if availability == 'open_weights':
        prefix = hf_namespace_of(org_id) or org_id
    if namespace is not None:
        prefix = namespace
    datastore_path_components(COLLECTION, f'{prefix}/{base}')  # raises on an unusable identity
    details = {
        'deployment_type': deployment,
        'model_availability': availability,
        'epoch_model_version': version,
        'epoch_organization': organization,
        'org_registry_id': resolution.canonical_id,
        'org_registry_strategy': resolution.strategy,
        'org_registry_review_status': resolution.review_status,
        'epoch_accessibility': meta.get('accessibility'),
        'epoch_model_group': meta.get('model_group'),
        'epoch_release_date': _clean(row.get('Release date')),
        'epoch_country': _clean(row.get('Country')),
        'epoch_training_compute_flop': _clean(
            row.get('Training compute (FLOP)')),
        'epoch_training_compute_notes': _clean(
            row.get('Training compute notes')),
        'model_id_strategy': (
            'reviewed_override' if developer or namespace
            else 'organization_namespace'
        ),
        **name_details,
    }
    if len(organizations) > 1:
        details['epoch_organizations'] = json.dumps(organizations)
    if variant is not None:
        details['epoch_variant_suffix'] = variant
        key = 'reasoning_effort' if variant in EFFORT_SUFFIXES else 'thinking_budget'
        details[key] = variant
    info = ModelInfo(
        name=base,
        id=f'{prefix}/{base}',
        developer=org_id,
        additional_details={
            k: v for k, v in details.items() if v is not None
        },
    )
    return info, prefix, base, variant


def _benchmark_specs(dataset: Dataset) -> list[dict[str, str]]:
    return [
        row for row in dataset.tables['benchmark_metadata.csv']
        if _clean(row.get('source_file'))
    ]


def _row_key(row: dict[str, str], has_id: bool) -> str:
    if has_id:
        return str(row['id'])
    payload = '\x1f'.join(f'{k}={row[k]}' for k in sorted(row))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def make_log(
    row: dict[str, str], spec: dict[str, str], row_key: str,
    meta: dict[str, str], registry: Registry, dataset: Dataset,
    retrieved: str, groups: dict[str, set[str]] | None = None,
) -> tuple[EvaluationLog, str, str]:
    benchmark = spec['benchmark']
    bslug = _slug(benchmark)
    column = spec['score_column']
    raw_score = _clean(row.get(column))
    if raw_score is None:
        raise ValueError(f'blank {column!r} (the headline score column)')
    score = require_finite_number(raw_score, f'{column} score')
    info, prefix, base, variant = build_model_info(
        row, meta, registry, groups)

    scale = float(spec['scale'])
    # Epoch's `scale` multiplies a score onto 0-1; 0.01 means the column is
    # already a percentage. Keep the column's own scale.
    percent = scale == 0.01
    if scale not in (1.0, 0.01):
        raise ValueError(f'unrecognised scale {spec["scale"]!r}')
    lo, hi = (0.0, 100.0) if percent else (0.0, 1.0)
    external = spec['source_file'].endswith('_external.csv')
    origin = 'external' if external else 'epoch_run'

    skip = set(PROVENANCE_COLUMNS) | set(MODEL_COLUMNS) | {column, 'id'}
    details = {k: v.strip() for k, v in row.items()
               if k and k not in skip and _clean(v) is not None}
    provenance = {
        f'epoch_{_slug(k).replace("-", "_")}': row[k].strip()
        for k in PROVENANCE_COLUMNS if _clean(row.get(k)) is not None
    }

    metric_id, metric_kind, metric_parameters = GLOBAL_METRICS.get(
        column, (f'{SRC}.{bslug}.score', 'score', None)
    )
    result = EvaluationResult(
        evaluation_result_id=bslug,
        evaluation_name=f'{SRC}.{bslug}',
        source_data=SourceDataUrl(
            dataset_name=benchmark,
            source_type='url',
            url=[HUB_URL],
            additional_details={
                k: v for k, v in {
                    'epoch_in_eci': spec.get('in_eci'),
                    'epoch_random_baseline': _clean(spec.get('random_baseline')),
                    'epoch_score_ceiling': _clean(spec.get('score_ceiling')),
                    'epoch_benchmark_release_date': _clean(spec.get('release_date')),
                    'epoch_superseded_by': _clean(spec.get('superseded_by')),
                }.items() if v is not None
            } or None,
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f'{benchmark}: Epoch AI headline score, column {column!r} of '
                f'{spec["source_file"]}.'
            ),
            metric_id=metric_id,
            metric_name=column,
            metric_kind=metric_kind,
            metric_parameters=metric_parameters,
            metric_unit='percent' if percent else 'proportion',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=lo,
            max_score=hi,
            additional_details={
                'epoch_scale': spec['scale'],
                'epoch_source_file': spec['source_file'],
            },
        ),
        score_details=ScoreDetails(score=score, details=details or None),
    )
    if external:
        license_fields = {
            'license_note': EXTERNAL_LICENSE_NOTE,
            'epoch_contribution_license': 'CC-BY-4.0',
            'license_terms_url': LICENSE_TERMS_URL,
        }
    else:
        license_fields = {
            'license': 'CC-BY-4.0',
            'license_url': LICENSE_URL,
            'license_terms_url': LICENSE_TERMS_URL,
        }
    # Where this run fetched the archive, when that was a URL (a local
    # --input path is not provenance anyone else can follow).
    retrieval = (
        {'retrieved_from': dataset.origin}
        if dataset.origin.startswith(('http://', 'https://')) else {}
    )
    source = SourceMetadata(
        source_name='Epoch AI Benchmarking Hub',
        source_type='documentation',
        source_organization_name='Epoch AI',
        source_organization_url=HUB_URL,
        evaluator_relationship=(
            EvaluatorRelationship.other if external
            else EvaluatorRelationship.third_party
        ),
        additional_details={
            'epoch_score_origin': origin,
            'epoch_source_file': spec['source_file'],
            **license_fields,
            'attribution': CITATION,
            'dataset_url': DATA_URL,
            **retrieval,
            **provenance,
        },
    )
    version = re.sub(r'[^A-Za-z0-9._-]+', '-', _clean(row['Model version']))
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # Raw Epoch identity, never the resolved ids. The snapshot is in the
        # key because Epoch edits rows in place.
        evaluation_id=(
            f'{SRC}/{bslug}/{version}/{row_key}/{dataset.snapshot}'
        ),
        retrieved_timestamp=retrieved,
        source_metadata=source,
        eval_library=EvalLibrary(
            name='unknown' if external else EPOCH_RUN_LIBRARY,
            version='unknown',
        ),
        model_info=info,
        evaluation_results=[result],
    )
    return log, prefix, base


def convert(dataset: Dataset, retrieved: str,
            registry: Registry | None = None) -> SourceConversionResult:
    """Convert every row of every benchmark that names a headline column."""
    registry = registry or Registry()
    models = _model_metadata(dataset)
    groups = _model_groups(dataset)
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    logs: list[EvaluationLog] = []
    total = 0
    seen: dict[str, str] = {}
    named = {spec['source_file'] for spec in _benchmark_specs(dataset)}
    for row in dataset.tables['benchmark_metadata.csv']:
        if not _clean(row.get('source_file')):
            exclusions.append(SourceRecordExclusion(
                source_ref=f'benchmark_metadata.csv:{row["benchmark"]}',
                reason='Epoch names no source file for this benchmark',
                source_record=row,
            ))
    for name, rows in sorted(dataset.tables.items()):
        if name not in named and name not in NON_BENCHMARK_TABLES:
            exclusions.append(SourceRecordExclusion(
                source_ref=name,
                reason='benchmark_metadata.csv names no score column for '
                       'this file, so no headline score is declared',
                source_record={'rows': len(rows)},
            ))
    for spec in _benchmark_specs(dataset):
        rows = dataset.tables.get(spec['source_file'])
        if rows is None:
            failures.append(SourceRecordFailure(
                source_ref=spec['source_file'],
                reason='benchmark_metadata.csv names a file the archive lacks',
                source_record=spec,
            ))
            continue
        has_id = bool(rows) and 'id' in rows[0] and len(
            {r['id'] for r in rows}
        ) == len(rows) and all(_clean(r['id']) for r in rows)
        for index, row in enumerate(rows, start=2):  # CSV line numbers
            total += 1
            ref = f'{spec["source_file"]}:{index}'
            try:
                version = _clean(row.get('Model version'))
                log, _, _ = make_log(
                    row, spec, _row_key(row, has_id),
                    models.get(version or '', {}), registry, dataset,
                    retrieved, groups,
                )
                log = EvaluationLog.model_validate(log.model_dump())
                # Epoch lists some results twice under different row ids;
                # the datastore rejects records that differ only in id. Every
                # source column is carried on the log, so equal content means
                # the rows differ in nothing but the id.
                content = log.model_dump(
                    mode='json', exclude={'evaluation_id'}
                )
                fingerprint = json.dumps(content, sort_keys=True)
                if fingerprint in seen:
                    exclusions.append(SourceRecordExclusion(
                        source_ref=ref,
                        reason='exact duplicate of '
                               f'{seen[fingerprint]} (same model, score and '
                               'provenance; only the Epoch row id differs)',
                        source_record=row,
                    ))
                    continue
                seen[fingerprint] = ref
                logs.append(log)
            except (ValueError, TypeError) as exc:
                failures.append(SourceRecordFailure(
                    source_ref=ref, reason=f'{type(exc).__name__}: {exc}',
                    source_record=row,
                ))
    return SourceConversionResult(
        source_name='Epoch AI Benchmarking Hub', total_records=total,
        records=logs, failures=failures, exclusions=exclusions,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Epoch AI benchmark_data.zip to EEE.')
    parser.add_argument('--input', type=Path, default=None,
                        help='Local benchmark_data.zip (default: fetch --url).')
    parser.add_argument('--url', default=DATA_URL,
                        help='Where to fetch benchmark_data.zip from.')
    # Outside the checkout by default; the path must end in `data`.
    parser.add_argument('--output-dir', type=Path,
                        default=Path(f'/tmp/{SRC}-smoke/data/{COLLECTION}'),
                        help='The data/<collection> directory.')
    parser.add_argument('--failure-report', type=Path, default=None)
    parser.add_argument('--no-registry-resolve', action='store_true',
                        help='Skip organization resolution (ids unverified).')
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.input is not None:
        content, origin = args.input.read_bytes(), str(args.input)
    else:
        content, origin = fetch_zip(args.url), args.url
    dataset = load_zip(content, origin)
    registry = Registry(enabled=not args.no_registry_resolve)
    result = convert(dataset, str(time.time()), registry)
    report = save_failure_report(
        result,
        args.failure_report or default_failure_report_path(args.output_dir),
    )
    paths = publish_evaluation_logs(
        result.records,
        base_output_dir=args.output_dir.parent,
        file_uuids=[str(uuid.uuid4()) for _ in result.records],
        collection_override=args.output_dir.name,
    )
    print(f'{result.total_records} source rows -> {len(paths)} logs, '
          f'{len(result.failures)} failed, {len(result.exclusions)} '
          f'files/benchmarks excluded -> {args.output_dir}')
    print(f'failure report: {report}')
    result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    run(parse_args())
