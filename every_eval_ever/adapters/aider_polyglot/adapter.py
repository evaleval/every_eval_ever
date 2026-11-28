"""Convert Aider's published Polyglot leaderboard YAML.

Usage:
    uv run python -m every_eval_ever.adapters.aider_polyglot.adapter
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    SourceDataUrl,
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

COLLECTION = 'aider-polyglot'
YAML_URL = (
    'https://raw.githubusercontent.com/Aider-AI/aider/main/'
    'aider/website/_data/polyglot_leaderboard.yml'
)
LEADERBOARD_URL = 'https://aider.chat/docs/leaderboards/'
# Aider reports pass rates as percentages.
METRICS = (
    ('pass_rate_1', 'Pass rate (first attempt)', 'pass_rate_1'),
    ('pass_rate_2', 'Pass rate (after one retry)', 'pass_rate_2'),
)


def fetch_yaml() -> bytes:
    """Fetch and capture the exact source bytes."""
    response = requests.get(YAML_URL, timeout=30)
    response.raise_for_status()
    raw_capture.record(
        url=YAML_URL,
        content=response.content,
        content_type='application/yaml',
        label=COLLECTION,
    )
    return response.content


def parse_rows(content: bytes | str) -> list[Any]:
    """Load the leaderboard list, failing on a changed top-level shape."""
    rows = yaml.safe_load(content)
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected a non-empty YAML list of leaderboard rows')
    return rows


def _required_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'missing {key}')
    return value.strip()


def _detail(value: Any) -> str:
    return str(value)


def convert_rows(
    rows: list[Any],
    output_dir: Path,
    retrieved: str,
    *,
    resolve: bool = True,
) -> SourceConversionResult:
    """Convert each row independently and report every rejected row."""
    outputs, failures = [], []
    registry = Registry(enabled=resolve)
    benchmark = registry.benchmark('aider-polyglot')
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError('row is not a mapping')
            run_id = _required_str(row, 'dirname')
            if run_id in seen:
                raise ValueError(f'duplicate dirname {run_id}')
            seen.add(run_id)
            model = _required_str(row, 'model')
            edit_format = _required_str(row, 'edit_format')
            # A flat source name has no known developer.
            _, dev_dir, model_dir = datastore_path_components(
                COLLECTION, model, 'unattributed'
            )
            test_cases = row.get('test_cases')
            source_data = SourceDataUrl(
                dataset_name='Aider Polyglot',
                source_type='url',
                url=[YAML_URL, LEADERBOARD_URL],
                additional_details={'test_cases': _detail(test_cases)}
                if test_cases is not None
                else None,
            )
            run_details = {
                k: _detail(row[k])
                for k in (
                    'edit_format',
                    'command',
                    'commit_hash',
                    'versions',
                    'date',
                    'total_cost',
                    'seconds_per_case',
                )
                if row.get(k) is not None
            }
            results = []
            for key, metric_name, metric_id in METRICS:
                raw = row.get(key)
                if raw is None:
                    continue
                score = require_finite_number(raw, key)
                if not 0 <= score <= 100:
                    raise ValueError(f'{key} outside 0-100: {score}')
                results.append(
                    EvaluationResult(
                        evaluation_name=benchmark.canonical_id
                        or 'aider-polyglot',
                        source_data=source_data,
                        metric_config=MetricConfig(
                            metric_id=metric_id,
                            metric_name=metric_name,
                            metric_kind='pass_rate',
                            metric_unit='percent',
                            lower_is_better=False,
                            score_type='continuous',
                            min_score=0.0,
                            max_score=100.0,
                        ),
                        score_details=ScoreDetails(
                            score=score, details={'source_field': key}
                        ),
                        generation_config=GenerationConfig(
                            additional_details={'edit_format': edit_format}
                        ),
                    )
                )
            if not results:
                raise ValueError('no pass_rate_1 or pass_rate_2 score')
            identity = hashlib.sha256(run_id.encode('utf-8')).hexdigest()
            log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=f'{COLLECTION}/{identity}',
                retrieved_timestamp=retrieved,
                source_metadata=SourceMetadata(
                    source_name='Aider Polyglot leaderboard',
                    source_type='documentation',
                    source_organization_name='Aider',
                    evaluator_relationship='third_party',
                    additional_details={
                        'leaderboard_url': LEADERBOARD_URL,
                        'source_run_dirname': run_id,
                        'benchmark_registry_strategy': benchmark.strategy,
                        **run_details,
                    },
                ),
                eval_library=EvalLibrary(name='aider', version='unknown'),
                model_info=ModelInfo(
                    name=model,
                    id=model,
                    developer='unattributed',
                    additional_details={
                        'deployment_type': 'unknown',
                        'model_availability': 'unknown',
                    },
                ),
                evaluation_results=results,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=log,
                    base_dir=output_dir,
                    developer=dev_dir,
                    model_name=model_dir,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'row {index}',
                    reason=str(exc),
                    source_record=row,
                )
            )
    return SourceConversionResult(
        source_name=COLLECTION,
        total_records=len(rows),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--input-yaml', type=Path)
    parser.add_argument('--failure-report', type=Path)
    parser.add_argument('--no-registry-resolve', action='store_true')
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    content = args.input_yaml.read_bytes() if args.input_yaml else fetch_yaml()
    rows = parse_rows(content)
    output_dir = args.output_dir or Path('data') / COLLECTION
    result = convert_rows(
        rows,
        output_dir,
        str(time.time()),
        resolve=not args.no_registry_resolve,
    )
    paths = save_evaluation_logs(result.records)
    print(
        f'{result.source_name}: {result.total_records} rows, '
        f'{len(paths)} records, {len(result.failures)} failures'
    )
    if result.failures:
        report = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(output_dir),
        )
        print(f'failure report: {report}')
        result.raise_if_incomplete()
    return paths


if __name__ == '__main__':
    run(parse_args())
