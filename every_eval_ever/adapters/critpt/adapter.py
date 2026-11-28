"""Convert the CritPt v1 and v2 README leaderboards."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import requests

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    SourceDataHf,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    raw_capture,
    require_finite_number,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.eval_card_registry import Registry
from every_eval_ever.helpers.io import (
    datastore_path_components,
    default_failure_report_path,
)

URLS = {
    'v1': 'https://raw.githubusercontent.com/CritPt-Benchmark/CritPt/17c2545c302762d2f2d644d923ea4c301605cb08/README.md',
    'v2': 'https://raw.githubusercontent.com/CritPt-Benchmark/CritPt/main/README.md',
}
REGISTRY_URL = 'https://evaleval-entity-registry.hf.space/api/v1/resolve'
DATASET = 'CritPt-Benchmark/CritPt'
FOOTNOTE = '²'


def fetch_readme(version: str) -> str:
    """Fetch and capture the exact source bytes."""
    response = requests.get(URLS[version], timeout=30)
    response.raise_for_status()
    raw_capture.record(
        url=URLS[version], content=response.content,
        content_type='text/markdown', label=f'critpt-{version}',
    )
    return response.content.decode('utf-8')


def parse_rows(text: str, version: str) -> list[dict[str, str]]:
    """Extract every row of the expected leaderboard table, failing on drift."""
    heading = 'Leaderboard' if version == 'v1' else 'Leaderboard (v2)'
    section = re.search(
        rf'^## {re.escape(heading)}\s*$([\s\S]*?)(?=^## |\Z)',
        text, re.MULTILINE,
    )
    if not section:
        raise ValueError(f'{version} leaderboard section not found')
    if 'average accuracy over 5 runs × 70 test challenges' not in section.group(1):
        raise ValueError('unsupported or missing challenge accuracy definition')
    lines = section.group(1).splitlines()
    for i, line in enumerate(lines):
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if cells == ['Model', 'Challenge Accuracy¹']:
            break
    else:
        raise ValueError('leaderboard table header not found')
    if i + 1 >= len(lines) or not re.fullmatch(
        r'\s*\|?\s*:?-+:?\s*\|\s*:?-+:?\s*\|?\s*', lines[i + 1]
    ):
        raise ValueError('invalid leaderboard table separator')
    rows = []
    for line in lines[i + 2:]:
        if not line.strip():
            break
        rows.append({'line': line, 'version': version})
    if not rows:
        raise ValueError('leaderboard table is empty')
    if version == 'v1' and any(FOOTNOTE in row['line'] for row in rows):
        if not re.search(
            r'² Result of Gemini 3 Pro \(without tools\).*?'
            r'provided by \*\*Artificial Analysis\*\*',
            section.group(1), re.DOTALL,
        ):
            raise ValueError('v1 independent-result footnote is missing or changed')
    return rows


def model_resolution(name: str, enabled: bool) -> tuple[str, dict[str, str]]:
    """Resolve without creating registry entries; retain the raw name on failure."""
    if not enabled:
        return name, {'model_registry_strategy': 'registry_disabled'}
    try:
        response = requests.post(
            REGISTRY_URL,
            json={'raw_value': name, 'entity_type': 'model', 'mode': 'exact'},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data.get('canonical_id') and not data.get('created_new'):
            return data['canonical_id'], {
                'model_registry_strategy': str(data.get('strategy') or 'live_exact'),
                'model_registry_review_status': str(data.get('review_status') or 'unknown'),
                'model_registry_confidence': str(data.get('confidence') or 'unknown'),
            }
        return name, {'model_registry_strategy': 'no_canonical'}
    except (requests.RequestException, ValueError, TypeError):
        return name, {'model_registry_strategy': 'registry_unavailable'}


def convert_rows(
    rows: list[dict[str, str]], version: str, output_dir: Path,
    retrieved: str, *, resolve: bool = True,
) -> SourceConversionResult:
    """Convert each source row independently and report all rejected rows."""
    outputs, failures = [], []
    registry = Registry(enabled=resolve)
    benchmark = registry.benchmark(f'critpt-{version}')
    metric = registry.metric('accuracy')
    seen: set[str] = set()
    cache: dict[str, tuple[str, dict[str, str]]] = {}
    for index, row in enumerate(rows, 1):
        try:
            if not row['line'].strip().startswith('|') or not row['line'].strip().endswith('|'):
                raise ValueError('invalid markdown table row')
            parts = [part.strip() for part in row['line'].strip().strip('|').split('|')]
            if len(parts) != 2 or not parts[0]:
                raise ValueError('expected model and score cells')
            label, raw_score = parts
            match = re.fullmatch(r'(.+?)(?:\s*\(([^()]*)\))?(²)?', label)
            if not match:
                raise ValueError('invalid model label')
            model, variant, footnote = match.groups()
            model = model.strip()
            if not model or label in seen:
                raise ValueError('empty or duplicate model label')
            seen.add(label)
            score = require_finite_number(raw_score, 'challenge accuracy')
            if score < 0:
                raise ValueError('negative challenge accuracy')
            if model not in cache:
                cache[model] = model_resolution(model, resolve)
            model_id, provenance = cache[model]
            # A flat unresolved source name has no known developer.
            developer = model_id.split('/')[0] if '/' in model_id else 'unattributed'
            _, dev_dir, model_dir = datastore_path_components(
                f'critpt-{version}', model_id, developer
            )
            identity = hashlib.sha256(f'{version}\0{label}'.encode('utf-8')).hexdigest()
            independent = version == 'v1' and model == 'Gemini-3 Pro' and footnote == FOOTNOTE
            details = {
                'leaderboard_url': URLS[version], 'source_model_label': label,
                'reported_by': 'CritPt maintainers (GitHub README)',
                'benchmark_version': version,
                'metric_definition': 'Average accuracy over 5 runs × 70 test challenges',
                'benchmark_registry_strategy': benchmark.strategy,
            }
            if independent:
                details['evaluator'] = 'Artificial Analysis'
                details['evaluator_url'] = 'https://artificialanalysis.ai/evaluations/critpt'
                details['tools'] = 'without tools'
            generation = GenerationConfig(additional_details={'source_variant': variant}) if variant else None
            result = EvaluationResult(
                evaluation_name=benchmark.canonical_id or f'critpt-{version}',
                source_data=SourceDataHf(
                    dataset_name=f'critpt-{version}', source_type='hf_dataset',
                    hf_repo=DATASET,
                    additional_details={
                        'benchmark_version': version,
                        'dataset_revision': 'Current HF dataset; not verified to match this leaderboard version',
                    },
                ),
                metric_config=MetricConfig(
                    metric_id=metric.canonical_id or 'accuracy',
                    metric_name='Challenge Accuracy', metric_kind='accuracy',
                    lower_is_better=False,
                    additional_details={
                        'bounds_status': 'unknown',
                        'registry_strategy': metric.strategy,
                    },
                ),
                score_details=ScoreDetails(score=score, details={'source_value': raw_score}),
                generation_config=generation,
            )
            log = EvaluationLog(
                schema_version=SCHEMA_VERSION, evaluation_id=f'critpt/{version}/{identity}',
                retrieved_timestamp=retrieved,
                source_metadata=SourceMetadata(
                    source_name=f'CritPt leaderboard {version}',
                    source_type='documentation', source_organization_name='CritPt',
                    evaluator_relationship='third_party' if independent else 'other',
                    additional_details=details,
                ),
                eval_library=EvalLibrary(name='unknown', version='unknown'),
                model_info=ModelInfo(
                    name=model, id=model_id, developer=developer,
                    additional_details=provenance,
                ),
                evaluation_results=[result],
            )
            outputs.append(EvaluationLogOutput(
                eval_log=log, base_dir=output_dir,
                developer=dev_dir, model_name=model_dir,
            ))
        except Exception as exc:
            failures.append(SourceRecordFailure(
                source_ref=f'{version} row {index}', reason=str(exc), source_record=row,
            ))
    return SourceConversionResult(
        source_name=f'critpt-{version}', total_records=len(rows),
        records=outputs, failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=URLS, default='v2')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--input-markdown', type=Path)
    parser.add_argument('--failure-report', type=Path)
    parser.add_argument('--no-registry-resolve', action='store_true')
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    text = args.input_markdown.read_text(encoding='utf-8') if args.input_markdown else fetch_readme(args.version)
    rows = parse_rows(text, args.version)
    output_dir = args.output_dir or Path('data') / f'critpt-{args.version}'
    result = convert_rows(rows, args.version, output_dir, str(time.time()), resolve=not args.no_registry_resolve)
    paths = save_evaluation_logs(result.records)
    print(f'{result.source_name}: {result.total_records} rows, {len(paths)} records, {len(result.failures)} failures')
    if result.failures:
        report = save_failure_report(result, args.failure_report or default_failure_report_path(output_dir))
        print(f'failure report: {report}')
        result.raise_if_incomplete()
    return paths


if __name__ == '__main__':
    run(parse_args())
