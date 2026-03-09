from __future__ import annotations
from argparse import ArgumentParser
import json
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from inspect_ai.log import list_eval_logs
from eval_converters.inspect.adapter import InspectAIAdapter
from eval_types import (
    EvaluatorRelationship,
    EvaluationLog
)
from instance_level_types import InstanceLevelEvaluationLog

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--log_path', type=str, default='tests/data/inspect/data.json', help='Inspect evalaution log file with extension eval or json.')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--source_organization_name', type=str, default='unknown', help='Orgnization which pushed evaluation to the every-eval-ever.')
    parser.add_argument('--evaluator_relationship', type=str, default='third_party', help='Relationship of evaluation author to the model', choices=['first_party', 'third_party', 'collaborative', 'other'])
    parser.add_argument('--source_organization_url', type=str, default=None)
    parser.add_argument('--source_organization_logo_url', type=str, default=None)
    parser.add_argument('--eval_library_name', type=str, default='inspect_ai', help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)')
    parser.add_argument('--eval_library_version', type=str, default='unknown', help='Version of the evaluation library. It should be extracted in the adapter if available in the evaluation log.')


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
        self, 
        metadata_args: Dict[str, Any] = None,
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

            unified_eval_path = f'{unified_eval_log_dir}/{output_filepath}'
            with open(unified_eval_path, 'w') as json_file:
                json_file.write(json_str)

            logger.info(
                "Unified eval log was successfully saved to %s path.",
                unified_eval_path,
            )
        except Exception as e:
            logger.exception("Problem with saving unified eval log to file: %s", e)
            raise e

def save_evaluation_log(
    unified_output: EvaluationLog,
    inspect_converter: InspectEvalLogConverter,
    file_uuid: str
) -> bool:
    try:
        model_developer, model_name = unified_output.model_info.id.split('/')
        filedir = f'{unified_output.evaluation_results[0].source_data.dataset_name}/{model_developer}/{model_name}'
        filename = f'{file_uuid}.json'
        inspect_converter.save_to_file(unified_output, filedir, filename)
        return True
    except Exception as e:
        logger.error(
            "Failed to save eval log %s to file. %s",
            unified_output.evaluation_id,
            str(e),
        )
        return False

def extract_file_uuid_from_output(unified_output: EvaluationLog) -> str | None:
    detailed = unified_output.detailed_evaluation_results
    if detailed and detailed.file_path:
        stem = Path(detailed.file_path).stem
        suffix = "_samples"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    inspect_converter = InspectEvalLogConverter(
        log_path=args.log_path,
        output_dir=args.output_dir
    )

    base_metadata_args = {
        'source_organization_name': args.source_organization_name,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'evaluator_relationship': EvaluatorRelationship(args.evaluator_relationship),
        'parent_eval_output_dir': args.output_dir,
        'eval_library_name': args.eval_library_name,
        'eval_library_version': args.eval_library_version,
    }

    if inspect_converter.is_log_path_directory:
        log_paths: List[Path] = list_eval_logs(inspect_converter.log_path.absolute().as_posix())
        if not log_paths:
            logger.warning("Missing evaluations logs to convert!")
        else:
            file_uuids = [str(uuid.uuid4()) for _ in log_paths]
            metadata_args = {
                **base_metadata_args,
                "file_uuids": file_uuids,
            }
            unified_output = inspect_converter.convert_to_unified_schema(metadata_args)
            if unified_output and isinstance(unified_output, List):
                for idx, single_unified_output in enumerate(unified_output):
                    file_uuid = file_uuids[idx] if idx < len(file_uuids) else None
                    if not file_uuid:
                        file_uuid = extract_file_uuid_from_output(single_unified_output)
                    if not file_uuid:
                        file_uuid = str(uuid.uuid4())
                        logger.warning(
                            "Missing UUID for output %s; generated %s for aggregate save.",
                            single_unified_output.evaluation_id,
                            file_uuid,
                        )
                    save_evaluation_log(
                        single_unified_output,
                        inspect_converter,
                        file_uuid,
                    )
            else:
                logger.warning("Missing unified schema result!")
    else:
        file_uuid = str(uuid.uuid4())
        metadata_args = {
            **base_metadata_args,
            'file_uuid': file_uuid,
        }

        unified_output = inspect_converter.convert_to_unified_schema(metadata_args)
        
        if unified_output:
            save_evaluation_log(
                unified_output,
                inspect_converter,
                file_uuid,
            )
        else:
            logger.warning("Missing unified schema result!")
