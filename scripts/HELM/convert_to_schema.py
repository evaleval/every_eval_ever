import math
import os
import requests
import time
import uuid

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    EvaluationResult,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceMetadata
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--leaderboard_name', type=str, default='HELM_Capabilities', choices=['HELM_Capabilities', 'HELM_Lite', 'HELM_Classic', 'HELM_Instruct', 'HELM_MMLU'])
    parser.add_argument('--source_data_url', type=str, default='https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.12.0/groups/core_scenarios.json')

    args = parser.parse_args()
    return args


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


def extract_model_info(row: List, model_name: str) -> ModelInfo:
    run_spec_names = next(
        (r["run_spec_names"] for r in row if "run_spec_names" in r),
        None
    )

    if not run_spec_names:
        developer = 'unknown'
        model_id = model_name.replace(' ', '-')
    else:
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


def check_min_and_max_values_in_columns(tab_rows):
    ''''
    Check if we can assign range [0, 1] for metric score.
    '''
    number_of_columns = len(tab_rows[0]) - 1
    mins = [0] * number_of_columns
    maxs = [0] * number_of_columns
    for row in tab_rows:
        for col_id, el in enumerate(row):
            if not col_id:
                continue

            mins[col_id - 1] = min(mins[col_id - 1], el.get('value', 0))
            maxs[col_id - 1] = max(maxs[col_id - 1], el.get('value', 0))

    return mins, maxs

def convert(leaderboard_name, leaderboard_data, source_name, source_type, source_data):
    '''
    Script for conversion data from leaderboards: HELM Capabilities, HELM Lite.
    '''
    retrieved_timestamp = str(time.time())
    model_to_results = defaultdict(list)
    model_infos = {}

    tabs_headers = [tab.get('header') for tab in leaderboard_data]
    tabs_rows = [tab.get('rows') for tab in leaderboard_data]

    for idx, (headers, tab_rows) in enumerate(zip(tabs_headers, tabs_rows)):
        tab_name = leaderboard_data[idx].get('title')
        mins, maxs = check_min_and_max_values_in_columns(tab_rows)

        for rows in tab_rows:
            model_name = rows[0].get('value')
            if model_name not in model_infos.keys():
                model_infos[model_name] = extract_model_info(rows, model_name)

            for col_id, (header, row) in enumerate(zip(headers[1:], rows[1:])):
                eval_name = header.get('value')
                metric_config = MetricConfig(
                    evaluation_description=header.get('description') or None,
                    lower_is_better=header.get('lower_is_better') or False,
                    min_score=0 if mins[col_id] >= 0 else math.floor(mins[col_id]),
                    max_score=1 if maxs[col_id] <= 1 else math.ceil(maxs[col_id]),
                    score_type=ScoreType.continuous
                )

                score = row.get('value')

                generation_config = extract_generation_config_from_run_specs(
                    row.get('run_spec_names')
                ) if row.get('run_spec_names') else {}

                model_to_results[model_name].append(
                    EvaluationResult(
                        evaluation_name=eval_name,
                        metric_config=metric_config,
                        score_details=ScoreDetails(
                            score=round(score, 3) if score else -1,
                            details={
                                "description": row.get('description', None),
                                "tab": tab_name
                            }
                        ),
                        generation_config=generation_config
                    )
                )


    for model_name, evaluation_results in model_to_results.items():
        model_info = model_infos.get(model_name)
        evaluation_id=f'{leaderboard_name}/{model_info.id.replace('/', '_')}/{retrieved_timestamp}'
        eval_log = EvaluationLog(
            schema_version='0.0.1',
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name=source_name,
                source_type=source_type,
                source_organization_name='crfm',
                evaluator_relationship=EvaluatorRelationship.third_party
            ),
            model_info=model_info,
            source_data=source_data,
            evaluation_results=evaluation_results
        )

        log_filename = f'{uuid.uuid4()}.json'
        if model_info.developer == 'unknown':
            model_name = model_info.id
            dirpath = f'data/{leaderboard_name}/{model_name}'
        else:
            model_dev, model_name = tuple(model_info.id.split('/'))
            dirpath = f'data/{leaderboard_name}/{model_dev}/{model_name}'

        os.makedirs(dirpath, exist_ok=True)
        save_to_file(eval_log, f'{dirpath}/{log_filename}')


if __name__ == '__main__':
    args = parse_args()

    leaderboard_name = args.leaderboard_name.lower()
    source_data = [args.source_data_url]

    os.makedirs(f'data/{leaderboard_name}', exist_ok=True)

    leaderboard_data = download_leaderboard(source_data[0])

    convert(leaderboard_name, leaderboard_data, leaderboard_name, 'documentation', source_data)
    
    # uv run python3 -m scripts.HELM.convert_to_schema --leaderboard_name HELM_Instruct --source_data_url https://storage.googleapis.com/crfm-helm-public/instruct/benchmark_output/releases/v1.0.0/groups/instruction_following.json    # https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.12.0/groups/core_scenarios.json
    # https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json
    # https://storage.googleapis.com/crfm-helm-public/benchmark_output/releases/v0.4.0/groups/core_scenarios.json
    # https://storage.googleapis.com/crfm-helm-public/instruct/benchmark_output/releases/v1.0.0/groups/instruction_following.json
    # https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/releases/v1.13.0/groups/mmlu_subjects.json