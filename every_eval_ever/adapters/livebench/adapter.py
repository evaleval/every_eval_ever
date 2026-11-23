"""Convert the LiveBench leaderboard releases to the EvalEval schema format.

Data source (https://livebench.ai), read from the site's own repository,
https://github.com/LiveBench/livebench.github.io:
- ``src/App.js`` lists the leaderboard releases (``YYYY-MM-DD``).
- ``public/table_<release>.csv`` holds each model's score per task.
- ``public/categories_<release>.json`` groups the tasks into categories.
- ``src/Table/modelLinks.js`` names each model's organization, display name
  and link; a model the site has no entry for has no stated organization.

The site shows a category's score as the mean of its tasks' scores and the
overall score as the mean of the category scores (``src/Table/Averaging.js``);
this adapter publishes the task scores and those two means, each naming its
aggregation level so a consumer takes one level, not several.

Usage:
    uv run python -m every_eval_ever.adapters.livebench.adapter
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

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
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.eval_card_registry import (
    REGISTRY_BASE_URL,
    Registry,
)
from every_eval_ever.helpers.fetch import fetch_text
from every_eval_ever.helpers.io import datastore_path_components

COLLECTION = 'livebench'
OUTPUT_DIR = f'data/{COLLECTION}'
SITE_URL = 'https://livebench.ai'
SITE_REPO = 'https://github.com/LiveBench/livebench.github.io'
RAW_BASE = (
    'https://raw.githubusercontent.com/LiveBench/livebench.github.io/main'
)
#: The benchmark's questions, per category.
DATASET_URL = 'https://huggingface.co/livebench'

#: Category labels the site uses that are abbreviations.
CATEGORY_SLUGS = {'IF': 'instruction_following'}

#: Registry resolutions of LiveBench model names, pinned so a record's model
#: id does not depend on the registry being reachable at convert time.
#: Refresh with ``--refresh-registry-map``.
REGISTRY_MAP = Path(__file__).with_name('registry_snapshot.json')
REGISTRY_TIMEOUT = 120

_RELEASE = re.compile(r"setSelectedDate\('(\d{4}-\d{2}-\d{2})'\)")


# -- source files -------------------------------------------------------------


def release_file(release: str, kind: str, suffix: str) -> str:
    return f'{RAW_BASE}/public/{kind}_{release.replace("-", "_")}.{suffix}'


def parse_releases(app_js: str) -> list[str]:
    """The release dates the site's release picker offers, oldest first."""
    releases = sorted(set(_RELEASE.findall(app_js)))
    if not releases:
        raise ValueError('no releases found in the site App.js')
    return releases


def parse_model_links(model_links_js: str) -> dict[str, dict[str, Any]]:
    """Return the site's model metadata, variants resolved like ``getModelInfo``.

    ``modelLinks.js`` is a JavaScript object literal: bare keys, double-quoted
    strings, trailing commas. It is rewritten to JSON rather than evaluated.
    """
    start = model_links_js.index('{')
    end = model_links_js.index('};', start) + 1
    body = model_links_js[start:end]
    # quote bare keys, only where a key can start (after { or ,), so ':' in
    # a quoted URL is never touched
    body = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', body)
    body = re.sub(r',(\s*[}\]])', r'\1', body)
    links = json.loads(body)
    lookup: dict[str, dict[str, Any]] = {}
    for base, info in links.items():
        lookup[base] = {k: v for k, v in info.items() if k != 'variants'}
        for variant in info.get('variants') or []:
            merged = dict(lookup[base])
            merged.update({k: v for k, v in variant.items() if k != 'rawName'})
            lookup[variant['rawName']] = merged
    return lookup


def parse_table(table_csv: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(table_csv)))
    if not rows or 'model' not in rows[0]:
        raise ValueError('release table has no model column')
    return rows


def category_slug(label: str) -> str:
    return CATEGORY_SLUGS.get(
        label, re.sub(r'\W+', '_', label.lower()).strip('_')
    )


# -- registry --------------------------------------------------------------------


def load_registry_map(path: Path = REGISTRY_MAP) -> dict[str, dict[str, Any]]:
    """The pinned registry resolutions, keyed by LiveBench model name."""
    return json.loads(path.read_text(encoding='utf-8'))['models']


def namespace_casing(
    registry_models: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    """One spelling per model-id namespace, keyed by its lowercase form.

    The registry spells some namespaces two ways (``Qwen/Qwen3-32B`` beside
    ``qwen/qwen3.6-plus``, ``Anthropic/claude-3-opus-20240229`` beside
    ``anthropic/claude-sonnet-3.7``), which would file one publisher under two
    datastore directories. The spelling most pinned canonical ids use wins;
    a tie goes to the lowercase one.
    """
    counts: dict[str, dict[str, int]] = {}
    for entry in (registry_models or {}).values():
        namespace = entry['canonical_id'].split('/', 1)[0]
        spellings = counts.setdefault(namespace.lower(), {})
        spellings[namespace] = spellings.get(namespace, 0) + 1
    return {
        key: min(spellings, key=lambda s: (-spellings[s], s != s.lower(), s))
        for key, spellings in counts.items()
    }


def _with_namespace_casing(model_id: str, casing: dict[str, str]) -> str:
    namespace, sep, rest = model_id.partition('/')
    return casing.get(namespace.lower(), namespace) + sep + rest


def refresh_registry_map(
    names: list[str], base_url: str = REGISTRY_BASE_URL
) -> dict[str, Any]:
    """Resolve model names in the registry and return a map to pin.

    Uses the side-effect-free ``exact`` mode. A resolution is kept only when
    it names an existing ``<namespace>/<name>`` canonical; its organization
    is read from the model's registry record.
    """
    base_url = base_url.rstrip('/')
    response = requests.post(
        f'{base_url}/api/v1/resolve/batch',
        json=[
            {'raw_value': name, 'entity_type': 'model', 'mode': 'exact'}
            for name in names
        ],
        timeout=REGISTRY_TIMEOUT,
    )
    response.raise_for_status()
    models: dict[str, dict[str, Any]] = {}
    for resolution in response.json():
        canonical_id = resolution.get('canonical_id')
        # exact mode must never create a canonical, and a flat id names no
        # datastore directory of its own
        if (
            not canonical_id
            or resolution.get('created_new')
            or '/' not in canonical_id
        ):
            continue
        record = requests.get(
            f'{base_url}/api/v1/models/{quote(canonical_id, safe="")}',
            timeout=REGISTRY_TIMEOUT,
        )
        record.raise_for_status()
        record = record.json()
        models[resolution['raw_value']] = {
            'canonical_id': canonical_id,
            'strategy': resolution.get('strategy'),
            'confidence': resolution.get('confidence'),
            'review_status': resolution.get('review_status'),
            'developer': record.get('developer'),
            'org_id': record.get('org_id'),
            'open_weights': record.get('open_weights'),
        }
    return {
        '_meta': {
            'endpoint': f'{base_url}/api/v1/resolve/batch',
            'mode': 'exact',
            'refreshed': time.strftime('%Y-%m-%d', time.gmtime()),
            'n_queried': len(names),
            'n_resolved': len(models),
        },
        'models': dict(sorted(models.items())),
    }


# -- conversion ----------------------------------------------------------------


def _score(value: str | None) -> float | None:
    """A task cell as the site reads it (``parseFloat``); blank is absent."""
    try:
        number = float(value) if value not in (None, '') else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _result(
    release: str,
    name: str,
    score: float,
    level: str,
    description: str,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_result_id=f'{COLLECTION}/{release}/{name}',
        evaluation_name=f'{COLLECTION}/{name}',
        source_data=SourceDataUrl(
            dataset_name=COLLECTION,
            source_type='url',
            url=[DATASET_URL],
            additional_details={'release': release},
        ),
        metric_config=MetricConfig(
            evaluation_description=description,
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=100,
        ),
        score_details=ScoreDetails(
            score=score,
            details={'aggregation_level': level},
        ),
    )


def convert_row(
    row: dict[str, str],
    release: str,
    categories: dict[str, list[str]],
    model_links: dict[str, dict[str, Any]],
    registry: Registry,
    retrieved_timestamp: str,
    registry_models: dict[str, dict[str, Any]] | None = None,
) -> tuple[EvaluationLog, str, str]:
    """One table row -> one EvaluationLog and its datastore directories.

    ``model_info.id`` is the model's registry canonical id where the pinned
    map resolves it, else ``<registry org id>/<LiveBench name>``, its
    namespace spelled as ``namespace_casing`` settles. The organization is the
    site's own (``modelLinks.js``); where the site gives none, the resolved
    registry model's organization is used.
    """
    model = (row.get('model') or '').strip()
    if not model:
        raise ValueError('row has no model name')
    info = model_links.get(model) or {}
    resolved = (registry_models or {}).get(model)
    organization = info.get('organization')
    details: dict[str, Any] = {}
    if organization:
        org = registry.org(organization)
        org_id = org.canonical_id or re.sub(r'\W+', '-', organization.lower())
        details['livebench_organization'] = organization
        details.update(org.provenance('developer'))
        open_weights = info.get('openweight')
    elif resolved and resolved.get('developer') and resolved.get('org_id'):
        organization = resolved['developer']
        details['developer_registry_id'] = resolved['org_id']
        open_weights = resolved.get('open_weights')
    else:
        raise ValueError(
            f'{model}: the site names no organization for this model (no '
            'modelLinks.js entry) and the pinned registry map has no model '
            'with an organization for it'
        )
    if resolved:
        model_id = resolved['canonical_id']
        details.update(
            {
                'model_registry_id': resolved['canonical_id'],
                'model_registry_strategy': resolved.get('strategy'),
                'model_registry_confidence': resolved.get('confidence'),
                'model_registry_review_status': resolved.get('review_status'),
            }
        )
    else:
        model_id = f'{org_id}/{model}'
    model_id = _with_namespace_casing(
        model_id, namespace_casing(registry_models)
    )

    results = []
    category_scores = []
    for label, tasks in categories.items():
        slug = category_slug(label)
        task_scores = []
        for task in tasks:
            score = _score(row.get(task))
            if score is None:
                continue
            task_scores.append(score)
            results.append(
                _result(
                    release,
                    f'{slug}/{task}',
                    score,
                    'task',
                    f'LiveBench {release} task {task} ({label})',
                )
            )
        category_score = _mean(task_scores)
        category_scores.append(category_score)
        if category_score is not None:
            results.append(
                _result(
                    release,
                    slug,
                    category_score,
                    'category',
                    f'LiveBench {release} {label}: mean of its task scores',
                )
            )
    if category_scores and None not in category_scores:
        results.insert(
            0,
            _result(
                release,
                'overall',
                _mean(category_scores),
                'overall',
                f'LiveBench {release} overall: mean of the category scores',
            ),
        )
    if not results:
        raise ValueError(f'{model}: no task scores in the release table')

    details['deployment_type'] = 'unknown'
    # an open-weight mark is a claim; its absence is not the opposite claim
    details['model_availability'] = (
        'open_weights' if open_weights is True else 'unknown'
    )
    for key, field in (
        ('displayName', 'livebench_display_name'),
        ('url', 'livebench_model_url'),
        ('version', 'livebench_model_version'),
        ('reasoner', 'livebench_reasoner'),
        ('note', 'livebench_note'),
    ):
        if info.get(key) is not None:
            details[field] = str(info[key])

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'{COLLECTION}/{release}/{model}',
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name=f'LiveBench {release}',
            source_type='documentation',
            source_organization_name='LiveBench',
            source_organization_url=SITE_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'release': release,
                'source_repository': SITE_REPO,
                'source_file': release_file(release, 'table', 'csv'),
                'categories_file': release_file(release, 'categories', 'json'),
            },
        ),
        eval_library=EvalLibrary(
            name='livebench',
            version=release,
            additional_details={
                'github': 'https://github.com/LiveBench/LiveBench'
            },
        ),
        model_info=ModelInfo(
            name=model,
            id=model_id,
            developer=organization,
            additional_details={
                k: str(v) for k, v in details.items() if v is not None
            },
        ),
        evaluation_results=results,
    )
    _, developer_dir, model_dir = datastore_path_components(
        COLLECTION, model_id
    )
    return log, developer_dir, model_dir


def convert_release(
    release: str,
    table_csv: str,
    categories: dict[str, list[str]],
    model_links: dict[str, dict[str, Any]],
    registry: Registry,
    retrieved_timestamp: str,
    registry_models: dict[str, dict[str, Any]] | None = None,
) -> SourceConversionResult[tuple[EvaluationLog, str, str]]:
    rows = parse_table(table_csv)
    bundles = []
    failures: list[SourceRecordFailure] = []
    for index, row in enumerate(rows):
        try:
            bundles.append(
                convert_row(
                    row,
                    release,
                    categories,
                    model_links,
                    registry,
                    retrieved_timestamp,
                    registry_models,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{release_file(release, "table", "csv")} row {index + 1}',
                    reason=str(exc),
                    source_record=row,
                )
            )
    return SourceConversionResult(
        source_name=f'LiveBench {release}',
        total_records=len(rows),
        records=bundles,
        failures=failures,
    )


def export(
    bundles: list[tuple[EvaluationLog, str, str]], output_dir: Path
) -> list[Path]:
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=log,
            base_dir=output_dir,
            developer=developer,
            model_name=model_name,
        )
        for log, developer, model_name in bundles
    )


# -- CLI -------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and convert the LiveBench leaderboard releases.'
    )
    parser.add_argument(
        '--release',
        action='append',
        help='Convert only this release (YYYY-MM-DD); repeatable. '
        'Default: every release the site lists.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Collection directory (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--no-registry-resolve',
        action='store_true',
        help=(
            'Do not use the eval-card-registry: no organization resolution '
            'and no pinned model map.'
        ),
    )
    parser.add_argument(
        '--refresh-registry-map',
        action='store_true',
        help=(
            'Resolve every model in the selected releases against the live '
            f'registry, rewrite {REGISTRY_MAP.name}, and convert nothing.'
        ),
    )
    return parser.parse_args(argv)


def _fetch_release(release: str) -> tuple[str, dict[str, list[str]]]:
    table_csv = fetch_text(release_file(release, 'table', 'csv'))
    categories = json.loads(
        fetch_text(release_file(release, 'categories', 'json'))
    )
    return table_csv, categories


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    releases = args.release or parse_releases(
        fetch_text(f'{RAW_BASE}/src/App.js')
    )
    if args.refresh_registry_map:
        names = sorted(
            {
                row['model']
                for release in releases
                for row in parse_table(_fetch_release(release)[0])
            }
        )
        pinned = refresh_registry_map(names)
        REGISTRY_MAP.write_text(
            json.dumps(pinned, indent=2) + '\n', encoding='utf-8'
        )
        print(
            f'{pinned["_meta"]["n_resolved"]} of {len(names)} model(s) '
            f'resolved; wrote {REGISTRY_MAP}'
        )
        return 0

    model_links = parse_model_links(
        fetch_text(f'{RAW_BASE}/src/Table/modelLinks.js')
    )
    registry = Registry(enabled=not args.no_registry_resolve)
    registry_models = None if args.no_registry_resolve else load_registry_map()
    retrieved_timestamp = str(time.time())
    results = []
    for release in releases:
        try:
            table_csv, categories = _fetch_release(release)
            result = convert_release(
                release,
                table_csv,
                categories,
                model_links,
                registry,
                retrieved_timestamp,
                registry_models,
            )
        except Exception as exc:
            result = SourceConversionResult(
                source_name=f'LiveBench {release}',
                total_records=1,
                records=[],
                failures=[
                    SourceRecordFailure(
                        source_ref=release_file(release, 'table', 'csv'),
                        reason=f'could not read release {release}: {exc}',
                    )
                ],
            )
        paths = export(result.records, args.output_dir)
        print(
            f'LiveBench {release}: {len(paths)} of '
            f'{result.total_records} row(s) written'
        )
        results.append(result)

    combined = SourceConversionResult(
        source_name='LiveBench',
        total_records=sum(r.total_records for r in results),
        records=[b for r in results for b in r.records],
        failures=[f for r in results for f in r.failures],
    )
    if combined.failures:
        report = save_failure_report(
            combined, default_failure_report_path(args.output_dir)
        )
        print(f'{len(combined.failures)} row(s) not converted: {report}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
