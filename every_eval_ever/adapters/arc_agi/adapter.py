#!/usr/bin/env python3
"""Convert the ARC Prize leaderboard into Every Eval Ever records.

Data source:
- Public leaderboard page: https://arcprize.org/leaderboard
  The page renders from three JSON files under ``/media/data/``:
  ``evaluations.json`` (one row per model x dataset), ``models.json``
  (model id -> provider, type, group) and ``providers.json`` (provider
  id -> display name). ``datasets.json`` names the ARC-AGI-1/2/3 splits.
- The older ``/media/data/leaderboard/evaluations.json`` endpoint is gone
  (404 as of 2026-08-12); this adapter fetches the current files.

Each evaluation row has shape:
    {
      "datasetId": "v2_Semi_Private",
      "modelId": "anthropic-claude-fable-5-high",
      "score": 0.29,                # proportion correct, 0..1
      "costPerTask": 8.42,          # USD; some rows carry "cost" instead
      "resultsUrl": "",
      "display": true
    }

Rows are grouped by canonical (developer, model): several raw model ids
can alias one canonical model, and the merged aliases are preserved in
the record details. For every canonical model the adapter emits one
``EvaluationLog`` with a ``score`` result per dataset, plus a
``cost_per_task`` (or ``cost``) result when the source reports one.

Usage:
    uv run python -m every_eval_ever.adapters.arc_agi.adapter \\
        --output-dir data/arc-agi
    uv run python -m every_eval_ever.adapters.arc_agi.adapter \\
        --input-json /tmp/arc_agi_payload.json --output-dir /tmp/arc-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    SourceType,
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
from every_eval_ever.helpers.fetch import fetch_json

SOURCE_NAME = 'ARC Prize leaderboard'
SOURCE_ORGANIZATION = 'ARC Prize Foundation'
SOURCE_ORGANIZATION_URL = 'https://arcprize.org'
LEADERBOARD_URL = 'https://arcprize.org/leaderboard'
DEFAULT_BASE_URL = 'https://arcprize.org/media/data'
DEFAULT_OUTPUT_DIR = 'data/arc-agi'

# The leaderboard's provider ids, mapped to the developer slugs this
# datastore already uses for the same organizations. Unlisted providers
# fall back to a slug of the provider id, so a new lab appearing upstream
# converts without a code change.
PROVIDER_TO_DEVELOPER = {
    'Human': 'arcprize',
    'ARC Prize 2024': 'community',
    'ARC Prize 2025': 'community',
    'Anthropic': 'anthropic',
    'Google': 'google',
    'OpenAI': 'openai',
    'Meta': 'meta',
    'DeepSeek': 'deepseek',
    'xAI': 'xai',
    'Mistral': 'mistralai',
    'Alibaba': 'alibaba',
    'Moonshot AI': 'moonshotai',
    'Minimax': 'minimax',
    'Z.ai': 'zhipu-ai',
    'Thinking Machines': 'thinking-machines',
}

# Chart-layout fields on evaluation rows that carry no evaluation content.
PRESENTATION_KEYS = {'display', 'displayLabel', 'labelOffsetX', 'labelOffsetY'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert the ARC Prize leaderboard to EEE records.',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help=(
            'Read a saved combined payload (as written by --save-raw-json) '
            'instead of fetching the leaderboard data live.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help=(
            'Directory URL holding evaluations.json, models.json, '
            f'providers.json and datasets.json (default: {DEFAULT_BASE_URL}).'
        ),
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help=(
            'After fetching, write the combined payload to this path so '
            'future runs can replay offline with --input-json.'
        ),
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected source rows and reasons here. Defaults beside '
            '--output-dir when any row fails.'
        ),
    )
    return parser.parse_args(argv)


def fetch_payload(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip('/')
    return {
        'evaluations': fetch_json(f'{base}/evaluations.json'),
        'models': fetch_json(f'{base}/models.json'),
        'providers': fetch_json(f'{base}/providers.json'),
        'datasets': fetch_json(f'{base}/datasets.json'),
    }


def load_payload_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or 'evaluations' not in payload:
        raise ValueError(
            f'{path}: expected a combined payload with an "evaluations" key '
            '(as written by --save-raw-json), got '
            f'{type(payload).__name__}'
        )
    return payload


def slugify_model_name(raw_model_id: str, developer_name: str) -> str:
    s = raw_model_id.strip().lower()
    s = s.replace('_', '-')
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'[^a-z0-9.\-]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')

    prefix = developer_name + '-'
    if s.startswith(prefix):
        s = s[len(prefix):]

    return s


def resolve_developer(provider_id: str) -> str:
    mapped = PROVIDER_TO_DEVELOPER.get(provider_id)
    if mapped is not None:
        return mapped
    return slugify_model_name(provider_id, developer_name='')


def choose_primary_raw_model_id(
    rows_for_canonical: list[dict], developer_name: str
) -> str:
    aliases = sorted({row['modelId'] for row in rows_for_canonical})
    prefix = developer_name + '-'
    aliases.sort(
        key=lambda raw: (
            raw.lower().replace('_', '-').startswith(prefix),
            len(raw),
            raw.lower(),
        )
    )
    return aliases[0]


def choose_best_row(rows: list[dict], developer_name: str) -> dict:
    prefix = developer_name + '-'
    return sorted(
        rows,
        key=lambda row: (
            row['modelId'].lower().replace('_', '-').startswith(prefix),
            len(row['modelId']),
            row['modelId'].lower(),
        ),
    )[0]


def compute_metric_bounds(rows: list[dict]) -> dict[str, dict[str, float]]:
    cost_per_task_values = [
        float(row['costPerTask'])
        for row in rows
        if row.get('costPerTask') is not None
    ]
    cost_values = [
        float(row['cost']) for row in rows if row.get('cost') is not None
    ]

    bounds = {'score': {'min_score': 0.0, 'max_score': 1.0}}
    if cost_per_task_values:
        bounds['cost_per_task'] = {
            'min_score': 0.0,
            'max_score': max(cost_per_task_values),
        }
    if cost_values:
        bounds['cost'] = {'min_score': 0.0, 'max_score': max(cost_values)}
    return bounds


def make_source_data(dataset_id: str, dataset_display_name: str | None) -> SourceDataUrl:
    details = {'dataset_id': dataset_id}
    if dataset_display_name:
        details['dataset_display_name'] = dataset_display_name
    return SourceDataUrl(
        dataset_name='ARC Prize evaluations leaderboard JSON',
        source_type='url',
        url=[f'{DEFAULT_BASE_URL}/evaluations.json', LEADERBOARD_URL],
        additional_details=details,
    )


def stringify_details(row: dict, exclude_keys: set[str]) -> dict[str, str]:
    details = {}
    for k, v in row.items():
        if k in exclude_keys or k in PRESENTATION_KEYS or v is None:
            continue
        details[k] = str(v)
    return details


def _cost_result(
    row: dict,
    metric_id: str,
    raw_field: str,
    dataset_id: str,
    dataset_display_name: str | None,
    aliases_for_dataset: list[str],
    metric_bounds: dict[str, dict[str, float]],
) -> EvaluationResult:
    metric_name = 'Cost per task' if metric_id == 'cost_per_task' else 'Cost'
    return EvaluationResult(
        evaluation_result_id=f'{dataset_id}::{metric_id}',
        evaluation_name=dataset_id,
        source_data=make_source_data(dataset_id, dataset_display_name),
        metric_config=MetricConfig(
            metric_id=metric_id,
            metric_name=metric_name,
            metric_kind='cost',
            metric_unit='usd',
            lower_is_better=True,
            score_type=ScoreType.continuous,
            **metric_bounds[metric_id],
            additional_details={'raw_metric_field': raw_field},
        ),
        score_details=ScoreDetails(
            score=float(row[raw_field]),
            details={
                **stringify_details(row, exclude_keys={raw_field, 'modelId'}),
                'raw_model_id': row['modelId'],
                'raw_model_aliases_json': json.dumps(aliases_for_dataset),
            },
        ),
    )


def make_results(
    rows_for_canonical: list[dict],
    developer_name: str,
    metric_bounds: dict[str, dict[str, float]],
    dataset_names: dict[str, str],
) -> list[EvaluationResult]:
    results = []
    by_dataset = defaultdict(list)
    for row in rows_for_canonical:
        by_dataset[row['datasetId']].append(row)

    for dataset_id in sorted(by_dataset):
        row = choose_best_row(by_dataset[dataset_id], developer_name)
        aliases_for_dataset = sorted(
            {r['modelId'] for r in by_dataset[dataset_id]}
        )
        dataset_display_name = dataset_names.get(dataset_id)

        results.append(
            EvaluationResult(
                evaluation_result_id=f'{dataset_id}::score',
                evaluation_name=dataset_id,
                source_data=make_source_data(dataset_id, dataset_display_name),
                metric_config=MetricConfig(
                    metric_id='score',
                    metric_name='ARC score',
                    metric_kind='accuracy',
                    metric_unit='proportion',
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    **metric_bounds['score'],
                    additional_details={'raw_metric_field': 'score'},
                ),
                score_details=ScoreDetails(
                    score=float(row['score']),
                    details={
                        **stringify_details(
                            row, exclude_keys={'score', 'modelId'}
                        ),
                        'raw_model_id': row['modelId'],
                        'raw_model_aliases_json': json.dumps(
                            aliases_for_dataset
                        ),
                    },
                ),
            )
        )

        if row.get('costPerTask') is not None:
            results.append(
                _cost_result(
                    row,
                    'cost_per_task',
                    'costPerTask',
                    dataset_id,
                    dataset_display_name,
                    aliases_for_dataset,
                    metric_bounds,
                )
            )
        elif row.get('cost') is not None:
            results.append(
                _cost_result(
                    row,
                    'cost',
                    'cost',
                    dataset_id,
                    dataset_display_name,
                    aliases_for_dataset,
                    metric_bounds,
                )
            )

    return results


def make_log(
    rows_for_canonical: list[dict],
    developer_name: str,
    model_name: str,
    metric_bounds: dict[str, dict[str, float]],
    retrieved_timestamp: str,
    model_entry: dict | None,
    provider_entry: dict | None,
    dataset_names: dict[str, str],
) -> EvaluationLog:
    primary_raw_model_id = choose_primary_raw_model_id(
        rows_for_canonical, developer_name
    )
    all_aliases = sorted({row['modelId'] for row in rows_for_canonical})

    model_details: dict[str, str] = {
        'raw_model_id': primary_raw_model_id,
        'raw_model_aliases_json': json.dumps(all_aliases),
    }
    if model_entry is not None:
        for source_key, detail_key in (
            ('displayName', 'source_display_name'),
            ('providerId', 'source_provider_id'),
            ('modelType', 'source_model_type'),
            ('modelGroup', 'source_model_group'),
            ('modelReleaseDate', 'model_release_date'),
            ('paperUrl', 'paper_url'),
            ('codeUrl', 'code_url'),
        ):
            value = model_entry.get(source_key)
            if value:
                model_details[detail_key] = str(value)
    if provider_entry is not None and provider_entry.get('displayName'):
        model_details['source_provider_display_name'] = str(
            provider_entry['displayName']
        )

    display_name = (model_entry or {}).get('displayName')

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'arc-agi/{developer_name}/{model_name}/{retrieved_timestamp}'
        ),
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type=SourceType.documentation,
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=SOURCE_ORGANIZATION_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'leaderboard_url': LEADERBOARD_URL,
                'evaluations_url': f'{DEFAULT_BASE_URL}/evaluations.json',
                'models_url': f'{DEFAULT_BASE_URL}/models.json',
                'providers_url': f'{DEFAULT_BASE_URL}/providers.json',
                'filtered_to_display_true': 'True',
            },
        ),
        eval_library=EvalLibrary(
            name='ARC Prize leaderboard', version='unknown'
        ),
        model_info=ModelInfo(
            name=str(display_name or primary_raw_model_id),
            id=f'{developer_name}/{model_name}',
            developer=developer_name,
            additional_details=model_details,
        ),
        evaluation_results=make_results(
            rows_for_canonical, developer_name, metric_bounds, dataset_names
        ),
    )


def _valid_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value))


def convert_logs(
    payload: dict[str, Any],
    retrieved_timestamp: str | None = None,
) -> SourceConversionResult[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    models = {m['id']: m for m in payload.get('models') or []}
    providers = {p['id']: p for p in payload.get('providers') or []}
    dataset_names = {
        d['id']: d['displayName']
        for d in payload.get('datasets') or []
        if d.get('displayName')
    }

    rows = [
        row
        for row in payload['evaluations']
        if isinstance(row, dict) and row.get('display') is True
    ]

    failures: list[SourceRecordFailure] = []
    resolved: list[tuple[dict, str, str]] = []
    for index, row in enumerate(rows):
        source_ref = f'evaluations.json display row {index}'
        model_id = row.get('modelId')
        if not model_id or not isinstance(model_id, str):
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='missing modelId',
                    source_record=row,
                )
            )
            continue
        if not row.get('datasetId'):
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='missing datasetId',
                    source_record=row,
                )
            )
            continue
        score = row.get('score')
        if not _valid_number(score) or not 0.0 <= float(score) <= 1.0:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=f'score is not a proportion in [0, 1]: {score!r}',
                    source_record=row,
                )
            )
            continue
        bad_cost = next(
            (
                field
                for field in ('costPerTask', 'cost')
                if row.get(field) is not None
                and (not _valid_number(row[field]) or float(row[field]) < 0)
            ),
            None,
        )
        if bad_cost is not None:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=(
                        f'{bad_cost} is not a non-negative number: '
                        f'{row[bad_cost]!r}'
                    ),
                    source_record=row,
                )
            )
            continue
        model_entry = models.get(model_id)
        if model_entry is None:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=f'modelId {model_id!r} not in models.json',
                    source_record=row,
                )
            )
            continue
        provider_id = model_entry.get('providerId')
        if not provider_id:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=f'model {model_id!r} has no providerId',
                    source_record=row,
                )
            )
            continue
        developer_name = resolve_developer(provider_id)
        model_name = slugify_model_name(model_id, developer_name)
        if not model_name:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=f'modelId {model_id!r} slugifies to nothing',
                    source_record=row,
                )
            )
            continue
        resolved.append((row, developer_name, model_name))

    metric_bounds = compute_metric_bounds([row for row, _, _ in resolved])

    by_canonical: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row, developer_name, model_name in resolved:
        by_canonical[(developer_name, model_name)].append(row)

    bundles = []
    for (developer_name, model_name), rows_for_canonical in sorted(
        by_canonical.items()
    ):
        primary_raw_model_id = choose_primary_raw_model_id(
            rows_for_canonical, developer_name
        )
        model_entry = models.get(primary_raw_model_id)
        provider_entry = providers.get((model_entry or {}).get('providerId'))
        log = make_log(
            rows_for_canonical,
            developer_name,
            model_name,
            metric_bounds,
            timestamp,
            model_entry,
            provider_entry,
            dataset_names,
        )
        bundles.append((log, developer_name, model_name))

    if not bundles and not failures:
        raise ValueError('ARC Prize: converted 0 source records')
    return SourceConversionResult(
        source_name=SOURCE_NAME,
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


def run(args: argparse.Namespace) -> int:
    if args.input_json is not None:
        payload = load_payload_file(args.input_json)
    else:
        payload = fetch_payload(args.base_url)
        if args.save_raw_json is not None:
            args.save_raw_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_raw_json.write_text(
                json.dumps(
                    {
                        'source_base_url': args.base_url,
                        'fetched_at': str(time.time()),
                        **payload,
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding='utf-8',
            )

    result = convert_logs(payload)
    paths = export(result.records, args.output_dir)
    for path in paths:
        print(path)
    if result.failures:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} ARC Prize model log(s).')
