from __future__ import annotations
from argparse import ArgumentParser
import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from scripts.eval_converters.inspect.adapter import InspectAIAdapter
from eval_types import (
    EvaluatorRelationship,
    EvaluationLog
)
from instance_level_types import InstanceLevelEvaluationLog

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_path', type=str, default='tests/data/inspect/data.json')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--source_organization_name', type=str, default='unknown', help='Orgnization which pushed evaluation to the every-eval-ever.')
    parser.add_argument('--evaluator_relationship', type=str, default='third_party', help='Relationship of evaluation author to the model', choices=['first_party', 'third_party', 'collaborative', 'other'])
    parser.add_argument('--source_organization_url', type=str, default=None)
    parser.add_argument('--source_organization_logo_url', type=str, default=None)


    args = parser.parse_args()
    return args


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class InspectEvalLogConverter:
    def __init__(self, log_path: str | Path, output_dir: str = 'unified_schema/inspect_ai'):
        '''
        InspectAI generates log file for an evaluation.
        '''
        self.log_path = Path(log_path)
        self.is_log_path_directory = self.log_path.is_dir()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(
        self, metadata_args: Dict[str, Any] = None
    ) -> Union[
        Tuple[EvaluationLog, InstanceLevelEvaluationLog],
        List[Tuple[EvaluationLog, InstanceLevelEvaluationLog]]
    ]:
        if self.is_log_path_directory:
            return InspectAIAdapter().transform_from_directory(
                self.log_path, 
                metadata_args=metadata_args
            )
        else:
            return InspectAIAdapter().transform_from_file(
                self.log_path, 
                metadata_args=metadata_args
            )

    def save_to_file(
        self, 
        unified_eval_log: EvaluationLog, 
        output_filedir: str, 
        output_filepath: str
    ) -> bool:
        try:
            json_str = unified_eval_log.model_dump_json(indent=4, exclude_none=True)

            unified_eval_log_dir = Path(f'{self.output_dir}/{output_filedir}')
            unified_eval_log_dir.mkdir(parents=True, exist_ok=True)

            with open(f'{unified_eval_log_dir}/{output_filepath}', 'w') as json_file:
                json_file.write(json_str)

            print(f'Unified eval log was successfully saved to {output_filepath} file.')
        except Exception as e:
            print(f"Problem with saving unified eval log to file: {e}")
            raise e

def save_evaluation_log(
    unified_output: EvaluationLog,
    inspect_converter: InspectEvalLogConverter
) -> bool:
    try:
        model_developer, model_name = unified_output.model_info.id.split('/')
        filedir = f'{unified_output.evaluation_results[0].source_data.dataset_name}/{model_developer}/{model_name}'
        filename = f'{str(uuid.uuid4())}.json'
        inspect_converter.save_to_file(unified_output, filedir, filename)
        return True
    except Exception as e:
        print(f'Failed to save eval log {unified_output.evaluation_id} to file.\n{str(e)}')
        return False


if __name__ == '__main__':
    args = parse_args()

    inspect_converter = InspectEvalLogConverter(
        log_path=args.log_path,
        output_dir=args.output_dir
    )
    
    metadata_args = {
        'source_organization_name': args.source_organization_name,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'evaluator_relationship': EvaluatorRelationship(args.evaluator_relationship)
    }

    unified_output = inspect_converter.convert_to_unified_schema(metadata_args)
    
    if unified_output:
        if isinstance(unified_output, List):
            for single_unified_output in unified_output:
                save_evaluation_log(
                    single_unified_output,
                    inspect_converter
                )
        else:
            save_evaluation_log(
                unified_output,
                inspect_converter
            )
    else:
        print("Missing unified schema result!")