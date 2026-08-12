"""
Script to fetch RewardBench and RewardBench 2 leaderboard results
from HuggingFace and convert them to the EvalEval schema format.

Data sources:
- RewardBench v1: CSV from HuggingFace Space (leaderboard/final-rbv1-data.csv)
- RewardBench v2: JSON files from allenai/reward-bench-2-results dataset (eval-set/{org}/{model}.json)

Usage:
    uv run python -m every_eval_ever.adapters.rewardbench.adapter
    uv run python -m every_eval_ever.adapters.rewardbench.adapter \
        --output-dir /tmp/smoke/data/reward-bench
"""

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
    get_developer,
    get_model_id,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

# Data source URLs
REWARDBENCH_V1_CSV = 'https://huggingface.co/spaces/allenai/reward-bench/resolve/main/leaderboard/final-rbv1-data.csv'
REWARDBENCH_V2_TREE_API = 'https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/eval-set'
REWARDBENCH_V2_FILE_BASE = 'https://huggingface.co/datasets/allenai/reward-bench-2-results/resolve/main/eval-set'

OUTPUT_DIR = Path('data/reward-bench')

# RewardBench v1 source data (shared across all v1 evaluation results)
V1_SOURCE_DATA = SourceDataHf(
    dataset_name='RewardBench',
    source_type='hf_dataset',
    hf_repo='allenai/reward-bench',
)

# RewardBench v2 source data (shared across all v2 evaluation results)
V2_SOURCE_DATA = SourceDataHf(
    dataset_name='RewardBench 2',
    source_type='hf_dataset',
    hf_repo='allenai/reward-bench-2-results',
)

# Source metadata (shared)
V1_SOURCE_METADATA = SourceMetadata(
    source_name='RewardBench',
    source_type='documentation',
    source_organization_name='Allen Institute for AI',
    source_organization_url='https://allenai.org',
    evaluator_relationship=EvaluatorRelationship.third_party,
)

V2_SOURCE_METADATA = SourceMetadata(
    source_name='RewardBench 2',
    source_type='documentation',
    source_organization_name='Allen Institute for AI',
    source_organization_url='https://allenai.org',
    evaluator_relationship=EvaluatorRelationship.third_party,
)

# RewardBench v1 metrics with descriptions
V1_METRICS = {
    'Score': 'Overall RewardBench Score',
    'Chat': 'Chat accuracy - includes easy chat subsets',
    'Chat Hard': 'Chat Hard accuracy - includes hard chat subsets',
    'Safety': 'Safety accuracy - includes safety subsets',
    'Reasoning': 'Reasoning accuracy - includes code and math subsets',
    'Prior Sets (0.5 weight)': 'Prior Sets score (weighted 0.5) - includes test sets',
}

# RewardBench v2 metrics with descriptions
V2_METRICS = [
    ('Factuality', 'Factuality score - measures factual accuracy'),
    ('Precise IF', 'Precise Instruction Following score'),
    ('Math', 'Math score - measures mathematical reasoning'),
    ('Safety', 'Safety score - measures safety awareness'),
    ('Focus', 'Focus score - measures response focus'),
    ('Ties', 'Ties score - ability to identify tie cases'),
]


def _make_eval_result(
    name: str,
    score: float,
    description: str,
    source_data: SourceDataHf,
) -> EvaluationResult:
    """Create an EvaluationResult for a continuous 0-1 metric."""
    return EvaluationResult(
        evaluation_name=name,
        source_data=source_data,
        metric_config=MetricConfig(
            evaluation_description=description,
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(score=round(score, 4)),
    )


def _make_model_info(
    model_name: str,
    developer: str,
    additional_details: Optional[Dict[str, Any]] = None,
) -> ModelInfo:
    """Create ModelInfo without setting inference_platform."""
    model_id = get_model_id(model_name, developer)
    return ModelInfo(
        name=model_name,
        id=model_id,
        developer=developer,
        additional_details=additional_details,
    )


def extract_model_name_from_html(html_string: str) -> str:
    """Extract the model name from an HTML anchor tag."""
    pattern = r'>([^<]+)<'
    match = re.search(pattern, html_string)
    if match:
        name = match.group(1).strip()
        name = re.sub(r'\s*[\*⚠️]+$', '', name).strip()
        return name
    return re.sub(r'\s*[\*⚠️]+$', '', html_string).strip()


def extract_hf_model_id_from_html(html_string: str) -> str | None:
    """Return an explicit Hugging Face org/model reference when present."""
    match = re.search(r'href=["\']([^"\']+)["\']', html_string, re.IGNORECASE)
    if match is None:
        return None

    href = match.group(1).strip()
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc:
        if (
            parsed.scheme not in {'http', 'https', ''}
            or parsed.hostname is None
            or parsed.hostname.lower()
            not in {'huggingface.co', 'www.huggingface.co'}
        ):
            return None

    parts = [part for part in parsed.path.split('/') if part]
    if parts[:1] == ['models']:
        parts = parts[1:]
    if len(parts) != 2 or parts[0].lower() in {'spaces', 'datasets'}:
        return None
    return '/'.join(parts)


def parse_score(value: str) -> Optional[float]:
    """Parse a score string, normalizing 0-100 scores to 0-1."""
    if not value or not value.strip():
        return None
    try:
        score = float(value)
        # RewardBench v1 scores are typically 0-100, normalize to 0-1
        if score > 1:
            score = score / 100.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid RewardBench score: {value!r}') from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f'RewardBench score must be between 0 and 1, got {score!r}'
        )
    return score


def _output_for_log(
    eval_log: EvaluationLog,
    output_dir: Path | str,
) -> EvaluationLogOutput:
    model_id = require_identity(eval_log.model_info.id, 'RewardBench model id')
    if '/' not in model_id:
        raise ValueError(
            f'RewardBench model id must be developer/model: {model_id!r}'
        )
    developer, model = model_id.split('/', 1)
    return EvaluationLogOutput(
        eval_log=eval_log,
        base_dir=output_dir,
        developer=developer,
        model_name=model,
    )


def convert_rewardbench_v1_rows(
    rows: list[dict],
    retrieved_timestamp: str,
    output_dir: Path | str = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert valid v1 models and retain all rejected source fragments."""
    outputs = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []

    if not rows:
        failures.append(
            SourceRecordFailure(
                source_ref='RewardBench v1 leaderboard',
                reason='source contained zero model rows',
            )
        )

    for row_index, row in enumerate(rows):
        row_ref = f'RewardBench v1 CSV row {row_index + 2}'
        failure_count_before = len(failures)
        try:
            model_html = row.get('Model', '')
            display_name = extract_model_name_from_html(model_html)
            if display_name.lower() == 'random':
                exclusions.append(
                    SourceRecordExclusion(
                        source_ref=row_ref,
                        reason=(
                            'published random baseline is not a model '
                            'evaluation'
                        ),
                        source_record=row,
                    )
                )
                continue
            require_identity(display_name, 'RewardBench model name')

            hf_model_id = extract_hf_model_id_from_html(model_html)
            if hf_model_id is not None:
                developer, model_name = hf_model_id.split('/', 1)
            else:
                model_name = display_name
                developer = require_identity(
                    get_developer(model_name),
                    'RewardBench model developer',
                )

            eval_results: List[EvaluationResult] = []
            for metric_name, description in V1_METRICS.items():
                raw_score = row.get(metric_name, '')
                try:
                    score = parse_score(raw_score)
                    if score is not None:
                        eval_results.append(
                            _make_eval_result(
                                name=metric_name,
                                score=score,
                                description=description,
                                source_data=V1_SOURCE_DATA,
                            )
                        )
                except ValueError as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=f'{row_ref} metric {metric_name!r}',
                            reason=str(exc),
                            source_record={
                                'model': model_html,
                                'metric': metric_name,
                                'value': raw_score,
                            },
                        )
                    )

            if not eval_results:
                raise ValueError('model has no usable RewardBench metrics')

            model_type = row.get('Model Type', '')
            details = {}
            if model_type:
                details['model_type'] = model_type
            if display_name != model_name:
                details['display_name'] = display_name
            model_info = _make_model_info(
                model_name=model_name,
                developer=developer,
                additional_details=details or None,
            )
            evaluation_id = (
                'reward-bench/'
                f'{model_info.id.replace("/", "_")}/{retrieved_timestamp}'
            )
            eval_log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_metadata=V1_SOURCE_METADATA,
                eval_library=EvalLibrary(name='unknown', version='unknown'),
                model_info=model_info,
                evaluation_results=eval_results,
            )
            outputs.append(_output_for_log(eval_log, output_dir))
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=(
                        f'no output written: {exc}'
                        if len(failures) > failure_count_before
                        else str(exc)
                    ),
                    source_record=row,
                )
            )

    return SourceConversionResult(
        source_name='RewardBench v1',
        total_records=len(rows),
        records=outputs,
        failures=failures,
        exclusions=exclusions,
    )


def fetch_rewardbench_v1(
    retrieved_timestamp: str,
    output_dir: Path | str = OUTPUT_DIR,
) -> int:
    """Fetch and process RewardBench v1 results from the CSV file."""
    print('Fetching RewardBench v1 CSV...')

    result = convert_rewardbench_v1_rows(
        fetch_csv(REWARDBENCH_V1_CSV),
        retrieved_timestamp,
        output_dir,
    )
    return _publish_result(result, output_dir)


def collect_rewardbench_v2(
    retrieved_timestamp: str,
    output_dir: Path | str = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Fetch v2 source records, retaining fetch and conversion failures."""
    print('Fetching RewardBench v2 model list...')

    orgs = fetch_json(REWARDBENCH_V2_TREE_API)
    outputs = []
    failures: list[SourceRecordFailure] = []
    total_records = 0

    for org_item in orgs:
        if org_item['type'] != 'directory':
            continue

        org_path = org_item['path']
        org_name = org_path.split('/')[-1]
        print(f'  Processing organization: {org_name}')

        # Get models for this org
        org_tree_url = f'https://huggingface.co/api/datasets/allenai/reward-bench-2-results/tree/main/{org_path}'
        try:
            model_files = fetch_json(org_tree_url)
        except Exception as exc:
            total_records += 1
            failures.append(
                SourceRecordFailure(
                    source_ref=f'RewardBench v2 organization {org_path!r}',
                    reason=f'failed to fetch organization tree: {exc}',
                    source_record=org_item,
                )
            )
            continue

        for model_file in model_files:
            if model_file['type'] != 'file' or not model_file['path'].endswith(
                '.json'
            ):
                continue

            total_records += 1
            model_path = model_file['path']
            model_url = f'{REWARDBENCH_V2_FILE_BASE}/{"/".join(model_path.split("/")[1:])}'

            try:
                model_data = fetch_json(model_url)
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'RewardBench v2 file {model_path!r}',
                        reason=f'failed to fetch model result: {exc}',
                        source_record=model_file,
                    )
                )
                continue

            source_ref = f'RewardBench v2 file {model_path!r}'
            failure_count_before = len(failures)
            try:
                raw_model_name = require_identity(
                    model_data.get('model'),
                    'RewardBench model name',
                )
                if '/' in raw_model_name:
                    developer, model_name = raw_model_name.split('/', 1)
                else:
                    developer = require_identity(
                        org_name,
                        'RewardBench model developer',
                    )
                    model_name = raw_model_name
                model_type = model_data.get('model_type', '')

                eval_results: List[EvaluationResult] = []
                scores_for_average = []

                for metric_name, description in V2_METRICS:
                    raw_score = model_data.get(metric_name)
                    if raw_score is None:
                        continue
                    try:
                        score = float(raw_score)
                        if not 0.0 <= score <= 1.0:
                            raise ValueError(
                                f'score must be between 0 and 1, got {score!r}'
                            )
                        scores_for_average.append(score)
                        eval_results.append(
                            _make_eval_result(
                                name=metric_name,
                                score=score,
                                description=description,
                                source_data=V2_SOURCE_DATA,
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        failures.append(
                            SourceRecordFailure(
                                source_ref=(
                                    f'{source_ref} metric {metric_name!r}'
                                ),
                                reason=str(exc),
                                source_record={
                                    'model': raw_model_name,
                                    'metric': metric_name,
                                    'value': raw_score,
                                },
                            )
                        )

                if not eval_results:
                    raise ValueError(
                        'model has no usable RewardBench 2 metrics'
                    )

                mean_score = sum(scores_for_average) / len(scores_for_average)
                eval_results.insert(
                    0,
                    _make_eval_result(
                        name='Score',
                        score=mean_score,
                        description=(
                            'Overall RewardBench 2 Score (mean of all metrics)'
                        ),
                        source_data=V2_SOURCE_DATA,
                    ),
                )

                model_info = _make_model_info(
                    model_name=model_name,
                    developer=developer,
                    additional_details={'model_type': model_type}
                    if model_type
                    else None,
                )
                evaluation_id = (
                    'reward-bench-2/'
                    f'{model_info.id.replace("/", "_")}/'
                    f'{retrieved_timestamp}'
                )
                eval_log = EvaluationLog(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=evaluation_id,
                    retrieved_timestamp=retrieved_timestamp,
                    source_metadata=V2_SOURCE_METADATA,
                    eval_library=EvalLibrary(
                        name='unknown',
                        version='unknown',
                    ),
                    model_info=model_info,
                    evaluation_results=eval_results,
                )
                outputs.append(_output_for_log(eval_log, output_dir))
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=source_ref,
                        reason=(
                            f'no output written: {exc}'
                            if len(failures) > failure_count_before
                            else str(exc)
                        ),
                        source_record=model_data,
                    )
                )

    if total_records == 0 and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref='RewardBench v2 leaderboard',
                reason='source contained zero model result files',
            )
        )
    return SourceConversionResult(
        source_name='RewardBench v2',
        total_records=total_records,
        records=outputs,
        failures=failures,
    )


def _publish_result(
    result: SourceConversionResult[EvaluationLogOutput],
    output_dir: Path | str,
) -> int:
    """Publish valid outputs and a non-schema provenance report."""
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(f'Saved: {path}')
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            default_failure_report_path(output_dir),
        )
        print(f'Provenance report: {report_path}')
    result.raise_if_incomplete()
    return len(paths)


def fetch_rewardbench_v2(
    retrieved_timestamp: str,
    output_dir: Path | str = OUTPUT_DIR,
) -> int:
    """Fetch, process, and publish RewardBench v2 results."""
    return _publish_result(
        collect_rewardbench_v2(retrieved_timestamp, output_dir),
        output_dir,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert the RewardBench v1 and v2 leaderboards to EEE records.'
        )
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Datastore collection directory (default: {OUTPUT_DIR}).',
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    """Main function to fetch and process RewardBench results."""
    args = parse_args(argv)
    output_dir = args.output_dir
    retrieved_timestamp = str(time.time())

    print('=' * 60)
    print('Fetching RewardBench v1 results...')
    print('=' * 60)

    v1 = convert_rewardbench_v1_rows(
        fetch_csv(REWARDBENCH_V1_CSV),
        retrieved_timestamp,
        output_dir,
    )
    print(f'\nConverted {len(v1.records)} models from RewardBench v1')

    print('\n' + '=' * 60)
    print('Fetching RewardBench v2 results...')
    print('=' * 60)

    v2 = collect_rewardbench_v2(retrieved_timestamp, output_dir)
    print(f'\nConverted {len(v2.records)} models from RewardBench v2')

    combined = SourceConversionResult(
        source_name='RewardBench v1 and v2',
        total_records=v1.total_records + v2.total_records,
        records=[*v1.records, *v2.records],
        failures=[*v1.failures, *v2.failures],
        exclusions=[*v1.exclusions, *v2.exclusions],
    )
    count = _publish_result(combined, output_dir)
    print(f'\nPublished {count} RewardBench models')

    print('\n' + '=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == '__main__':
    main()
