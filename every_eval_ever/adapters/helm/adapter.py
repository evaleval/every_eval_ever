"""
Script to convert HELM leaderboard data to the EvalEval schema format.

Supports multiple HELM variants:
- HELM_Capabilities
- HELM_Lite
- HELM_Classic
- HELM_Instruct
- HELM_MMLU

Usage:
    uv run python -m every_eval_ever.adapters.helm.adapter --leaderboard_name HELM_Lite --source_data_url <url>
"""

import json
import math
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    get_developer,
    make_model_info,
    make_source_metadata,
    require_finite_number,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

HELM_PROJECT_METADATA_URL = (
    'https://crfm.stanford.edu/helm/project_metadata.json'
)


def parse_args():
    """Parse CLI arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--leaderboard_name',
        type=str,
        default='HELM_Capabilities',
        choices=[
            'HELM_Capabilities',
            'HELM_Lite',
            'HELM_Classic',
            'HELM_Instruct',
            'HELM_MMLU',
            'HELM_AIR_Bench',
            'HELM_Safety',
        ],
        help='HELM leaderboard name',
    )
    parser.add_argument(
        '--leaderboard_version',
        type=str,
        default='latest',
        help='Version of the HELM leaderboard to use; defaults to the latest version',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='helm',
        help='Name of the evaluation library (e.g. helm, lm_eval, inspect_ai)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library',
    )
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--failure-report')
    return parser.parse_args()


def clean_model_name(model_name: str) -> str:
    """Remove parentheses from model name."""
    return model_name.replace('(', '').replace(')', '')


def extract_generation_config(run_specs: List[str]) -> Dict[str, Any]:
    """Extract generation configuration from HELM run spec strings."""
    generation_config: Dict[str, Any] = defaultdict(list)

    for run_spec in run_specs:
        _, args_str = run_spec.split(':', 1)
        args = args_str.split(',')

        for arg in args:
            key, value = arg.split('=')
            if key == 'model':
                continue
            generation_config[key].append(value)

    # Collapse values if they are identical
    for key, values in list(generation_config.items()):
        if isinstance(values, list) and len(set(values)) == 1:
            values = values[0]

        generation_config[key] = json.dumps(values)

    return dict(generation_config)


def extract_model_info_from_row(
    row: List[Dict[str, Any]], model_name: str
) -> Tuple[ModelInfo, str]:
    """Extract model metadata from leaderboard row."""
    run_spec_names = next(
        (cell['run_spec_names'] for cell in row if 'run_spec_names' in cell),
        None,
    )

    if '(' in model_name and ')' in model_name:
        model_name = clean_model_name(model_name)

    if not run_spec_names:
        developer = get_developer(model_name)
        if developer == 'unknown':
            model_id = model_name.replace(' ', '-')
        else:
            model_id = f'{developer}/{model_name.replace(" ", "-")}'
    else:
        spec = run_spec_names[0]
        args = spec.split(':', 1)[1].split(',')

        model_details = next(
            (arg.split('=', 1)[1] for arg in args if arg.startswith('model=')),
            '',
        )

        developer = model_details.split('_')[0]
        model_id = model_details.replace('_', '/')

    if developer == 'unknown':
        developer = get_developer(model_name)

    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        inference_platform='unknown',
    )
    model_info.id = model_id

    return model_info


def find_column_ranges(
    tab_rows: List[List[Dict[str, Any]]], num_columns: int | None = None
):
    """Determine min/max values for each metric column."""
    if num_columns is None:
        num_columns = max(
            (len(row) - 1 for row in tab_rows if isinstance(row, list)),
            default=0,
        )
    mins = [0.0] * num_columns
    maxs = [0.0] * num_columns

    for row in tab_rows:
        if not isinstance(row, list):
            continue
        for idx, cell in enumerate(row[1 : num_columns + 1]):
            if not isinstance(cell, dict) or cell.get('value') is None:
                continue
            try:
                value = require_finite_number(cell['value'], 'HELM score')
            except ValueError:
                continue
            mins[idx] = min(mins[idx], value)
            maxs[idx] = max(maxs[idx], value)

    return mins, maxs


def convert(
    leaderboard_name: str,
    leaderboard_data: List[Dict[str, Any]],
    eval_library_name: str = 'helm',
    eval_library_version: str = 'unknown',
    source_data_url: str = 'unknown',
    output_dir: str = 'data',
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert HELM leaderboard data into unified evaluation logs."""
    retrieved_timestamp = str(time.time())

    model_infos: Dict[str, ModelInfo] = {}
    model_ids: Dict[str, str] = {}
    model_results: Dict[str, Dict[str, EvaluationResult]] = defaultdict(dict)
    failures = []
    total_rows = 0

    for tab_index, tab in enumerate(leaderboard_data, start=1):
        if not isinstance(tab, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'HELM tab {tab_index}',
                    reason='tab must be an object',
                    source_record=tab,
                )
            )
            continue
        tab_name = tab.get('title')
        headers = tab.get('header')
        rows = tab.get('rows')
        if (
            not isinstance(tab_name, str)
            or not isinstance(headers, list)
            or len(headers) < 2
            or not isinstance(rows, list)
        ):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'HELM tab {tab_index}',
                    reason=(
                        'tab requires a string title, at least two headers, '
                        'and a rows list'
                    ),
                    source_record=tab,
                )
            )
            continue

        mins, maxs = find_column_ranges(rows, len(headers) - 1)

        for row_index, row in enumerate(rows, start=1):
            total_rows += 1
            row_ref = f'HELM tab {tab_name!r} row {row_index}'
            if not isinstance(row, list) or not row:
                failures.append(
                    SourceRecordFailure(
                        source_ref=row_ref,
                        reason='row must be a non-empty list of cells',
                        source_record=row,
                    )
                )
                continue
            first_cell = row[0]
            if not isinstance(first_cell, dict):
                failures.append(
                    SourceRecordFailure(
                        source_ref=row_ref,
                        reason='model cell must be an object',
                        source_record=row,
                    )
                )
                continue
            try:
                model_name = require_identity(
                    first_cell.get('value'), 'HELM model name'
                )
                if model_name not in model_infos:
                    model_info = extract_model_info_from_row(row, model_name)
                    model_infos[model_name] = model_info
                    model_ids[model_name] = model_info.id
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=row_ref,
                        reason=str(exc),
                        source_record=row,
                    )
                )
                continue

            for col_idx, header in enumerate(headers[1:]):
                cell_ref = f'{row_ref} column {col_idx + 2}'
                if col_idx + 1 >= len(row):
                    failures.append(
                        SourceRecordFailure(
                            source_ref=cell_ref,
                            reason='row is missing the metric cell',
                            source_record=row,
                        )
                    )
                    continue
                cell = row[col_idx + 1]
                if not isinstance(header, dict) or not isinstance(cell, dict):
                    failures.append(
                        SourceRecordFailure(
                            source_ref=cell_ref,
                            reason='header and metric cell must be objects',
                            source_record={'header': header, 'cell': cell},
                        )
                    )
                    continue
                raw_score = cell.get('value')
                if raw_score is None:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=cell_ref,
                            reason='metric score is missing',
                            source_record={'header': header, 'cell': cell},
                        )
                    )
                    continue
                try:
                    score = require_finite_number(raw_score, 'HELM score')
                    full_eval_name = require_identity(
                        header.get('value'), 'HELM metric header'
                    )
                    is_air_category = (
                        leaderboard_name == 'helm_air_bench'
                        and tab_name.startswith('AIR')
                        and tab_name.endswith('categories')
                    )
                    short_name = (
                        full_eval_name.split()[0]
                        if '-' in full_eval_name and not is_air_category
                        else full_eval_name
                    )
                    is_new_metric = (
                        tab_name.lower() == 'accuracy'
                        or short_name not in model_results[model_name]
                        or 'instruct' in leaderboard_name.lower()
                        or is_air_category
                    )

                    if full_eval_name.lower().startswith('mean'):
                        metric_name = None
                        dataset_name = leaderboard_name
                        evaluation_name = full_eval_name
                    elif is_air_category:
                        dataset_name = full_eval_name
                        evaluation_name = dataset_name
                        metric_name = 'Refusal Rate'
                    else:
                        if ' - ' not in full_eval_name:
                            raise ValueError(
                                'HELM metric header must use '
                                "'<dataset> - <metric>' format"
                            )
                        dataset_name, metric_name = full_eval_name.split(
                            ' - ', 1
                        )
                        evaluation_name = dataset_name

                    evaluation_description = (
                        f'{metric_name} on {dataset_name}'
                        if metric_name
                        else header.get('description', '')
                    )
                    if is_new_metric:
                        metric_config = MetricConfig(
                            evaluation_description=evaluation_description,
                            metric_name=metric_name,
                            lower_is_better=header.get(
                                'lower_is_better', False
                            ),
                            min_score=(
                                0.0
                                if mins[col_idx] >= 0
                                else math.floor(mins[col_idx])
                            ),
                            max_score=(
                                1.0
                                if maxs[col_idx] <= 1
                                else math.ceil(maxs[col_idx])
                            ),
                            score_type=ScoreType.continuous,
                        )
                        source_dataset_name = (
                            leaderboard_name
                            if leaderboard_name.lower()
                            in ['helm_mmlu', 'helm_air_bench']
                            else dataset_name
                        )
                        source_data = SourceDataUrl(
                            dataset_name=source_dataset_name,
                            source_type='url',
                            url=[source_data_url],
                        )
                        generation_details = (
                            extract_generation_config(
                                cell.get('run_spec_names', [])
                            )
                            if cell.get('run_spec_names')
                            else {}
                        )
                        model_results[model_name][short_name] = (
                            EvaluationResult(
                                evaluation_name=evaluation_name,
                                source_data=source_data,
                                metric_config=metric_config,
                                score_details=ScoreDetails(
                                    score=round(score, 3),
                                    details={
                                        'description': str(
                                            cell.get('description', '')
                                        ),
                                        'tab': str(tab_name),
                                    },
                                ),
                                generation_config=GenerationConfig(
                                    additional_details=generation_details
                                ),
                            )
                        )
                    else:
                        existing = model_results[model_name][short_name]
                        detail_key = (
                            full_eval_name
                            if full_eval_name != existing.evaluation_name
                            else f'{full_eval_name} - {tab_name}'
                        )
                        if existing.score_details.details is None:
                            existing.score_details.details = {}
                        existing.score_details.details[detail_key] = json.dumps(
                            {
                                'description': str(cell.get('description', '')),
                                'tab': tab_name,
                                'score': str(raw_score),
                            }
                        )
                except Exception as exc:
                    failures.append(
                        SourceRecordFailure(
                            source_ref=cell_ref,
                            reason=str(exc),
                            source_record={'header': header, 'cell': cell},
                        )
                    )

    outputs = []
    for model_name, results_by_metric in model_results.items():
        try:
            model_info = model_infos[model_name]
            model_id = require_identity(model_ids[model_name], 'HELM model id')

            evaluation_id = (
                f'{leaderboard_name}/'
                f'{model_id.replace("/", "_")}/'
                f'{retrieved_timestamp}'
            )

            eval_log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_timestamp,
                source_metadata=make_source_metadata(
                    source_name=leaderboard_name,
                    organization_name='crfm',
                    evaluator_relationship=EvaluatorRelationship.third_party,
                ),
                eval_library=EvalLibrary(
                    name=eval_library_name,
                    version=eval_library_version,
                ),
                model_info=model_info,
                evaluation_results=list(results_by_metric.values()),
            )

            if '/' in model_id:
                developer, model = model_id.split('/', 1)
            else:
                developer = require_identity(
                    model_info.developer, 'HELM model developer'
                )
                model = model_id
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=Path(output_dir) / leaderboard_name,
                    developer=developer,
                    model_name=model,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'HELM model {model_name!r}',
                    reason=str(exc),
                    source_record={'model_name': model_name},
                )
            )
    if not outputs and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref=f'HELM leaderboard {leaderboard_name}',
                reason='leaderboard contained zero usable model scores',
            )
        )
    return SourceConversionResult(
        source_name=f'HELM leaderboard {leaderboard_name}',
        total_records=total_rows,
        records=outputs,
        failures=failures,
    )


def get_leaderboard_versions(leaderboard_id: str) -> List[str]:
    """Return a list of published versions for the leaderboard"""
    project_metadata = fetch_json(HELM_PROJECT_METADATA_URL)
    project = ''
    for project in project_metadata:
        if project['id'] == leaderboard_id:
            return project['releases']
    raise ValueError(
        f'Leaderboard ID {leaderboard_id} not found in HELM project metadata at {HELM_PROJECT_METADATA_URL}'
    )


def get_source_data_url(leaderboard_id: str, leaderboard_version: str) -> str:
    """Return the URL of the JSON file containing the results table of the primary group of the leaderboard"""
    leaderboard_versions = get_leaderboard_versions(leaderboard_id)
    if not leaderboard_versions:
        raise ValueError(f'No versions found for leaderboard {leaderboard_id}')
    if leaderboard_version == 'latest':
        leaderboard_version = leaderboard_versions[0]
    if leaderboard_version not in leaderboard_versions:
        raise ValueError(
            f'Version {leaderboard_version} for leaderboard {leaderboard_id} not found; available versions: {leaderboard_versions}'
        )

    groups_table = fetch_json(
        f'https://storage.googleapis.com/crfm-helm-public/{leaderboard_id}/benchmark_output/releases/{leaderboard_version}/groups.json'
    )
    # This is un ugly hack to get the first group's ID.
    # Unfortunately, this is actually how the offical HELM code does it.
    # See: https://github.com/stanford-crfm/helm/blob/v0.5.14/helm-frontend/src/routes/Leaderboard.tsx#L44-L56
    first_group_name = groups_table[0]['rows'][0][0]['href'].removeprefix(
        '?group='
    )
    return f'https://storage.googleapis.com/crfm-helm-public/{leaderboard_id}/benchmark_output/releases/{leaderboard_version}/groups/{first_group_name}.json'


def main():
    args = parse_args()

    leaderboard_name = args.leaderboard_name.lower()

    if not leaderboard_name.startswith('helm_'):
        raise ValueError('leaderboard_name must start with helm_')
    leaderboard_id = leaderboard_name.removeprefix('helm_').replace('_', '-')
    source_data_url = get_source_data_url(
        leaderboard_id, args.leaderboard_version
    )

    print(
        f'Fetching {leaderboard_name} {args.leaderboard_version} data from {source_data_url}'
    )
    leaderboard_data = fetch_json(source_data_url)

    result = convert(
        leaderboard_name=leaderboard_name,
        leaderboard_data=leaderboard_data,
        eval_library_name=args.eval_library_name,
        eval_library_version=args.eval_library_version,
        source_data_url=source_data_url,
        output_dir=args.output_dir,
    )
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(f'Saved: {path}')
    if result.failures:
        report_path = save_failure_report(
            result,
            args.failure_report
            or default_failure_report_path(
                Path(args.output_dir) / leaderboard_name
            ),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()

    print('Done!')


if __name__ == '__main__':
    main()
