import json
import time
import uuid
import requests
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    EvaluationResult,
    EvaluationSource,
    EvaluationSourceType,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceMetadata
)

def download_leaderboard(url):
    response = requests.get(url)
    response.raise_for_status()

    return response.json()

def save_to_file(unified_eval_log: EvaluationLog, output_filepath: str) -> bool:
    try:
        json_str = unified_eval_log.model_dump_json(indent=2, exclude_none=True)

        with open(output_filepath, 'w') as json_file:
            json_file.write(json_str)

        print(f'Unified eval log was successfully saved to {output_filepath} file.')
    except Exception as e:
        print(f"Problem with saving unified eval log to file: {e}")
        raise e
    
def extract_generation_config_from_run_specs(run_specs: List[str]) -> Dict[str, Any]:
    generation_config = defaultdict(list)

    for run_spec in run_specs:
        task, args_str = tuple(run_spec.split(':', 1))
        args = args_str.split(',')
        
        for arg in args:
            key, value = tuple(arg.split('='))
            if key == 'model':
                continue
                
            generation_config[key].append(value)
    
    for key, values in generation_config.items():
        if len(set(values)) == 1:
            generation_config[key] = values[0]
    
    return generation_config

def extract_model_info(row: List) -> ModelInfo:
    model_name = row[0].get('value')
    run_spec_names = next(
        (r["run_spec_names"] for r in row if "run_spec_names" in r),
        None
    )
    spec = run_spec_names[0]
    args = spec.split(':', 1)[1].split(',')

    model_details = next(
        (arg.split('=', 1)[1] for arg in args if arg.startswith('model=')),
        ''
    )
    
    developer = model_details.split('_')[0]
    model_id = model_details.replace('_', '/')

    return ModelInfo(
        name=model_name,
        id=model_id,
        developer=developer,
        inference_platform='unknown'
    )
    
def extract_efficiency_stats(efficiency_row: List) -> Tuple[float, List[float]]:
    mean_win_rate = efficiency_row[1].get('value')
    eval_times_for_benchmarks = [
        stats.get('value') for stats in efficiency_row
    ]
    return mean_win_rate, eval_times_for_benchmarks

def prepare_score_details(
    acc_stats: Dict[str, Any], 
    eff_stats: Dict[str, Any], 
    column_idx: int,
    leaderboard_name: str
) -> ScoreDetails:
    details = {
        'accuracy_description': acc_stats.get('description'),
        'efficiency_description': eff_stats.get('description')
    }

    if column_idx == 0 and leaderboard_name == 'helm_capabilities': # mean_score stats
        details['mean_eval_time'] = round(eff_stats.get('value'), 3)
    elif column_idx == 0 and leaderboard_name == 'helm_lite': # mean_win_rate stats
        details['eval_time_mean_win_rate'] = round(eff_stats.get('value'), 3)
    else:
        details['eval_time'] = round(eff_stats.get('value'), 3)
    
    return ScoreDetails(
        score=round(acc_stats.get('value'), 3),
        details=details
    )

def convert(leaderboard_name, leaderboard_data, evaluation_source, source_data):
    '''
    Script for conversion data from leaderboards: HELM Capabilities, HELM Lite.
    '''
    accuracy_tab_data = leaderboard_data[0]
    accuracy_rows = accuracy_tab_data.get('rows')
    headers = accuracy_tab_data.get('header')
    eval_names = [header.get('value') for header in headers[1:]]

    efficiency_rows = leaderboard_data[1].get('rows') if len(leaderboard_data) > 1 and leaderboard_data[1].get('title') == 'Efficiency' else []

    metrics = [
        MetricConfig(
            evaluation_description=header.get('description') or None,
            lower_is_better=header.get('lower_is_better') or False,
            min_score=0,
            max_score=1,
            score_type=ScoreType.continuous
        )
        for header in headers[1:]
    ]

    evaluation_logs = {}

    for acc_row, eff_row in list(zip(accuracy_rows[:10], efficiency_rows[:10])):
        model_info = extract_model_info(acc_row)
        retrieved_timestamp = str(time.time())
        evaluation_id=f'{leaderboard_name}/{model_info.id.replace('/', '_')}/{retrieved_timestamp}'

        evaluation_results = []
        for column_idx, (acc_per_column, eff_per_column) in enumerate(zip(acc_row[1:], eff_row[1:])):
            if not acc_per_column.get('value'):
                continue

            generation_config = extract_generation_config_from_run_specs(acc_per_column.get('run_spec_names')) if acc_per_column.get('run_spec_names') else {}

            evaluation_results.append(
                EvaluationResult(
                    evaluation_name=eval_names[column_idx - 1],
                    metric_config=metrics[column_idx - 1],
                    score_details=prepare_score_details(
                        acc_per_column,
                        eff_per_column,
                        column_idx,
                        leaderboard_name
                    ),
                    generation_config=generation_config
                )
            )

        eval_log = EvaluationLog(
            schema_version='0.0.1',
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_organization_name='crfm',
                evaluator_relationship=EvaluatorRelationship.third_party
            ),
            model_info=model_info,
            source_data=source_data,
            evaluation_source=evaluation_source,
            evaluation_results=evaluation_results
        )

        evaluation_logs[evaluation_id] = eval_log

        log_filename = f'{uuid.uuid4()}.json'
        model_dev, model_name = tuple(model_info.id.split('/'))
        dirpath = f'data/{leaderboard_name}/{model_dev}/{model_name}'

        os.makedirs(dirpath, exist_ok=True)

        save_to_file(eval_log, f'{dirpath}/{log_filename}')


if __name__ == '__main__':
    leaderboard_name = 'HELM_Capabilities' # 'HELM_Lite'
    leaderboard_name = leaderboard_name.lower()
    source_data = [
        'https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.12.0/groups/core_scenarios.json'
        # 'https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json'
    ]

    os.makedirs(f'data/{leaderboard_name}', exist_ok=True)

    leaderboard_data = download_leaderboard(source_data[0])

    evaluation_source = EvaluationSource(
        evaluation_source_name=leaderboard_name,
        evaluation_source_type=EvaluationSourceType.leaderboard
    )

    convert(leaderboard_name, leaderboard_data, evaluation_source, source_data)
