from __future__ import annotations
from argparse import ArgumentParser
import uuid
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from scripts.eval_converters.helm.adapter import HELMAdapter
from eval_types import (
    EvaluatorRelationship,
    EvaluationLog
)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_path', type=str, default='tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2', help="Path to directory with single evaluaion or multiple evaluations to convert")
    parser.add_argument('--output_dir', type=str, default='data/helm_local_evals')
    parser.add_argument('--source_organization_name', type=str, default='Unknown', help='Orgnization which pushed evaluation to the evalHub.')
    parser.add_argument('--evaluator_relationship', type=str, default='other', help='Relationship of evaluation author to the model', choices=['first_party', 'third_party', 'collaborative', 'other'])
    parser.add_argument('--source_organization_url', type=str, default=None)
    parser.add_argument('--source_organization_logo_url', type=str, default=None)


    args = parser.parse_args()
    return args


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class HELMEvalLogConverter:
    def __init__(self, log_path: str | Path, output_dir: str = 'unified_schema/helm'):
        '''
        HELM generates log file for an evaluation.
        '''
        self.log_path = Path(log_path)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(self, metadata_args: Dict[str, Any] = None) -> Union[EvaluationLog, List[EvaluationLog]]:
        return HELMAdapter().transform_from_directory(self.log_path, metadata_args=metadata_args)

    def save_to_file(self, unified_eval_log: EvaluationLog, output_filedir: str, output_filepath: str) -> bool:
        try:
            json_str = unified_eval_log.model_dump_json(indent=2)

            unified_eval_log_dir = Path(f'{self.output_dir}/{output_filedir}')
            unified_eval_log_dir.mkdir(parents=True, exist_ok=True)

            with open(f'{unified_eval_log_dir}/{output_filepath}', 'w') as json_file:
                json_file.write(json_str)

            print(f'Unified eval log was successfully saved to {output_filepath} file.')
        except Exception as e:
            print(f"Problem with saving unified eval log to file: {e}")
            raise e


if __name__ == '__main__':
    args = parse_args()

    helm_converter = HELMEvalLogConverter(
        log_path=args.log_path,
        output_dir=args.output_dir
    )
    
    metadata_args = {
        'source_organization_name': args.source_organization_name,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'evaluator_relationship': EvaluatorRelationship(args.evaluator_relationship)
    }

    unified_output = helm_converter.convert_to_unified_schema(metadata_args)

    if unified_output and isinstance(unified_output, EvaluationLog):
        model_developer, model_name = unified_output.model_info.id.split('/')
        filedir = f'{unified_output.source_data.dataset_name}/{model_developer}/{model_name}'
        filename = f'{str(uuid.uuid4())}.json'
        helm_converter.save_to_file(unified_output, filedir, filename)

    elif unified_output and isinstance(unified_output, List):
        for single_unified_output in unified_output:
            model_developer, model_name = single_unified_output.model_info.id.split('/')
            filedir = f'{single_unified_output.source_data.dataset_name}/{model_developer}/{model_name}'
            filename = f'{str(uuid.uuid4())}.json'
            helm_converter.save_to_file(single_unified_output, filedir, filename)
    else:
        print("Missing unified schema result!")