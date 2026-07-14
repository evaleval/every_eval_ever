"""
Script to convert HELM leaderboard data to the EvalEval schema format.

Supports multiple HELM variants:
- HELM_Capabilities
- HELM_Lite
- HELM_Classic
- HELM_Instruct
- HELM_MMLU

Usage:
    uv run python -m utils.helm.adapter --leaderboard_name HELM_Lite --source_data_url <url>
"""

import json
import math
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List

from every_eval_ever.adapters.helm.provenance import (
    helm_metric_identity,
    helm_provenance,
)
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationConfig,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    fetch_json,
    get_developer,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)

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
) -> ModelInfo:
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
    provenance = helm_provenance(model_id)
    model_info.inference_platform = provenance.inference_platform
    model_info.inference_engine = InferenceEngine(
        name=provenance.inference_engine_name,
        version=provenance.inference_engine_version,
    )
    details = dict(model_info.additional_details or {})
    details.update(
        {
            'deployment_type': provenance.deployment_type,
            'model_availability': provenance.model_availability,
        }
    )
    model_info.additional_details = details

    return model_info


def find_column_ranges(tab_rows: List[List[Dict[str, Any]]]):
    """Determine min/max values for each metric column."""
    num_columns = len(tab_rows[0]) - 1
    mins = [0.0] * num_columns
    maxs = [0.0] * num_columns

    for row in tab_rows:
        for idx, cell in enumerate(row[1:], start=0):
            value = cell.get('value', 0)
            if value is not None:
                mins[idx] = min(mins[idx], value)
                maxs[idx] = max(maxs[idx], value)

    return mins, maxs


def convert(
    leaderboard_name: str,
    leaderboard_data: List[Dict[str, Any]],
    eval_library_name: str = 'helm',
    eval_library_version: str = 'unknown',
    source_data_url: str = 'unknown',
):
    """Convert HELM leaderboard data into unified evaluation logs."""
    retrieved_timestamp = str(time.time())

    model_infos: Dict[str, ModelInfo] = {}
    model_ids: Dict[str, str] = {}
    model_results: Dict[str, Dict[str, EvaluationResult]] = defaultdict(dict)
    skipped_results: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for tab in leaderboard_data:
        tab_name = tab.get('title')
        headers = tab.get('header')
        rows = tab.get('rows')

        mins, maxs = find_column_ranges(rows)

        for row in rows:
            model_name = row[0].get('value')

            if model_name not in model_infos:
                model_info = extract_model_info_from_row(row, model_name)
                model_infos[model_name] = model_info
                model_ids[model_name] = model_info.id

            for col_idx, (header, cell) in enumerate(zip(headers[1:], row[1:])):
                if cell.get('value') is None:
                    skipped_results[model_name].append(
                        {
                            'evaluation': str(header.get('value', 'unknown')),
                            'tab': str(tab_name),
                            'description': str(cell.get('description', '')),
                            'reason': 'source_value_missing',
                        }
                    )
                    continue
                # The "HELM level K category" tables in HELM AIR-Bench need special handling.
                # The column headers look like "AIRBench 2024 - Security Risks".
                # For for this example, the dataset name should be "AIRBench 2024 - Security Risks"
                # and the metric name should be "Refusal Rate".
                # This differs from other HELM tables, in which the column headers
                # are in the format "dataset_name - metric_name" (e.g. "MMLU - EM")
                # This boolean indicates whether the special handling is needed.
                is_helm_air_bench_category_table = (
                    leaderboard_name == 'helm_air_bench'
                    and tab_name.startswith('AIR')
                    and tab_name.endswith('categories')
                )

                full_eval_name = header.get('value')
                short_name = (
                    full_eval_name.split()[0]
                    if '-' in full_eval_name
                    and not is_helm_air_bench_category_table
                    else full_eval_name
                )

                is_new_metric = (
                    tab_name.lower() == 'accuracy'
                    or short_name not in model_results[model_name]
                    or 'instruct' in leaderboard_name.lower()
                    or is_helm_air_bench_category_table
                )

                if full_eval_name.lower().startswith('mean'):
                    metric_name = None
                    dataset_name = leaderboard_name
                    evaluation_name = full_eval_name
                elif is_helm_air_bench_category_table:
                    dataset_name = full_eval_name
                    evaluation_name = dataset_name
                    metric_name = 'Refusal Rate'
                else:
                    dataset_name, metric_name = full_eval_name.split(' - ', 1)
                    evaluation_name = dataset_name

                if metric_name:
                    evaluation_description = f'{metric_name} on {dataset_name}'
                else:
                    evaluation_description = header.get('description', '')

                if is_new_metric:
                    metric_id, resolved_metric_name = helm_metric_identity(
                        leaderboard_name,
                        evaluation_name,
                        metric_name,
                    )
                    metric_config = MetricConfig(
                        evaluation_description=evaluation_description,
                        metric_id=metric_id,
                        metric_name=resolved_metric_name,
                        lower_is_better=header.get('lower_is_better', False),
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

                    generation_config = (
                        extract_generation_config(
                            cell.get('run_spec_names', [])
                        )
                        if cell.get('run_spec_names')
                        else {}
                    )

                    model_results[model_name][short_name] = EvaluationResult(
                        evaluation_name=evaluation_name,
                        source_data=source_data,
                        metric_config=metric_config,
                        score_details=ScoreDetails(
                            score=round(cell['value'], 3),
                            details={
                                'description': str(cell.get('description', '')),
                                'tab': str(tab_name),
                            },
                        ),
                        generation_config=GenerationConfig(
                            additional_details=generation_config
                        ),
                    )
                else:
                    # Add extra score details under the same metric
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
                            'score': str(cell.get('value', '')),
                        }
                    )

    # Save evaluation logs
    for model_name, results_by_metric in model_results.items():
        model_info = model_infos[model_name]
        model_id = model_ids[model_name]

        evaluation_id = (
            f'{leaderboard_name}/'
            f'{model_id.replace("/", "_")}/'
            f'{retrieved_timestamp}'
        )

        source_metadata = make_source_metadata(
            source_name=leaderboard_name,
            organization_name='crfm',
            evaluator_relationship=EvaluatorRelationship.third_party,
        )
        model_skipped = skipped_results.get(model_name, [])
        if model_skipped:
            details = dict(source_metadata.additional_details or {})
            details['skipped_evaluation_results'] = json.dumps(
                model_skipped, ensure_ascii=False, sort_keys=True
            )
            source_metadata.additional_details = details
            print(
                f'WARNING: {leaderboard_name} {model_name}: preserved '
                f'{len(model_skipped)} missing source cells outside '
                'evaluation_results',
                file=sys.stderr,
            )

        eval_log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=source_metadata,
            eval_library=EvalLibrary(
                name=eval_library_name,
                version=eval_library_version,
            ),
            model_info=model_info,
            evaluation_results=list(results_by_metric.values()),
        )

        # Determine output path
        if model_info.developer == 'unknown':
            developer = model_id
            model = model_id
        else:
            if '/' in model_id:
                developer, model = model_id.split('/', 1)
            else:
                developer = model_info.developer
                model = model_id

        filepath = save_evaluation_log(
            eval_log,
            f'data/{leaderboard_name}',
            developer,
            model,
        )
        print(f'Saved: {filepath}')

    skipped_count = sum(len(items) for items in skipped_results.values())
    if skipped_count:
        print(
            f'WARNING: {leaderboard_name}: preserved {skipped_count} missing '
            'source cells outside evaluation_results',
            file=sys.stderr,
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

    convert(
        leaderboard_name=leaderboard_name,
        leaderboard_data=leaderboard_data,
        eval_library_name=args.eval_library_name,
        eval_library_version=args.eval_library_version,
        source_data_url=source_data_url,
    )

    print('Done!')


if __name__ == '__main__':
    main()
