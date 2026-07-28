from __future__ import annotations

import json
import logging
import uuid
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    from inspect_ai.log import list_eval_logs

    from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
    from every_eval_ever.converters.inspect.supplemental_eval_details import (
        SupplementalEvalDetails,
    )
except ImportError as exc:
    raise SystemExit(
        "The 'inspect-ai' package is required to use the Inspect AI converter.\n"
        'Install it with: uv sync --extra inspect'
    ) from exc

from every_eval_ever.eval_types import EvaluationLog, EvaluatorRelationship
from every_eval_ever.helpers.io import datastore_output_dir, require_uuid4
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--log_path',
        type=str,
        default='tests/data/inspect/data.json',
        help='Inspect evalaution log file with extension eval or json.',
    )
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument(
        '--source_organization_name',
        type=str,
        default='unknown',
        help='Orgnization which pushed evaluation to the every-eval-ever.',
    )
    parser.add_argument(
        '--evaluator_relationship',
        type=str,
        default='third_party',
        help='Relationship of evaluation author to the model',
        choices=['first_party', 'third_party', 'collaborative', 'other'],
    )
    parser.add_argument('--source_organization_url', type=str, default=None)
    parser.add_argument(
        '--source_organization_logo_url', type=str, default=None
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='inspect_ai',
        help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library. It should be extracted in the adapter if available in the evaluation log.',
    )
    parser.add_argument(
        '--supplemental_eval_details_path',
        type=str,
        default=None,
        help=(
            'Path to JSON file containing supplemental evaluation details to fill '
            'missing fields in converted output.'
        ),
    )

    args = parser.parse_args()
    return args


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class InspectEvalLogConverter:
    def __init__(
        self,
        log_path: str | Path,
        output_dir: str = 'unified_schema/inspect_ai',
    ):
        """
        InspectAI generates log file for an evaluation.
        """
        self.log_path = Path(log_path)
        self.is_log_path_directory = self.log_path.is_dir()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(
        self,
        metadata_args: Dict[str, Any] = None,
    ) -> Union[
        Tuple[EvaluationLog, InstanceLevelEvaluationLog],
        List[Tuple[EvaluationLog, InstanceLevelEvaluationLog]],
    ]:
        if self.is_log_path_directory:
            return InspectAIAdapter().transform_from_directory(
                self.log_path, metadata_args=metadata_args
            )
        else:
            return InspectAIAdapter().transform_from_file(
                self.log_path, metadata_args=metadata_args
            )

    def save_to_file(
        self,
        unified_eval_log: EvaluationLog,
        output_filedir: str,
        output_filepath: str,
    ) -> bool:
        json_str = json.dumps(
            unified_eval_log.model_dump(mode='json', exclude_none=True),
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        unified_eval_log_dir = self.output_dir / output_filedir
        unified_eval_log_dir.mkdir(parents=True, exist_ok=True)
        unified_eval_path = unified_eval_log_dir / output_filepath
        unified_eval_path.write_text(json_str + '\n', encoding='utf-8')
        logger.info(
            'Unified eval log was successfully saved to %s path.',
            unified_eval_path,
        )
        return True


def save_evaluation_log(
    unified_output: EvaluationLog,
    inspect_converter: InspectEvalLogConverter,
    file_uuid: str,
) -> bool:
    file_uuid = require_uuid4(file_uuid)
    if not unified_output.evaluation_results:
        raise ValueError('Inspect output contains no evaluation results')
    output_dir = datastore_output_dir(
        inspect_converter.output_dir,
        unified_output.evaluation_results[0].source_data.dataset_name,
        unified_output.model_info.id,
        unified_output.model_info.developer,
    )
    filedir = output_dir.relative_to(inspect_converter.output_dir).as_posix()
    filename = f'{file_uuid}.json'
    return inspect_converter.save_to_file(unified_output, filedir, filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    inspect_converter = InspectEvalLogConverter(
        log_path=args.log_path, output_dir=args.output_dir
    )

    base_metadata_args = {
        'source_organization_name': args.source_organization_name,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'evaluator_relationship': EvaluatorRelationship(
            args.evaluator_relationship
        ),
        'parent_eval_output_dir': args.output_dir,
        'eval_library_name': args.eval_library_name,
        'eval_library_version': args.eval_library_version,
    }
    if args.supplemental_eval_details_path:
        with open(
            args.supplemental_eval_details_path, 'r', encoding='utf-8'
        ) as f:
            base_metadata_args['supplemental_eval_details'] = (
                SupplementalEvalDetails.model_validate(json.load(f))
            )

    if inspect_converter.is_log_path_directory:
        log_paths: List[Path] = sorted(
            list_eval_logs(
                inspect_converter.log_path.absolute().as_posix()
            ),
            key=lambda path: path.name,
        )
        if not log_paths:
            raise ValueError(
                f'No Inspect evaluation logs found in '
                f'{inspect_converter.log_path}'
            )
        else:
            file_uuids = [str(uuid.uuid4()) for _ in log_paths]
            metadata_args = {
                **base_metadata_args,
                'file_uuids': file_uuids,
            }
            unified_output = inspect_converter.convert_to_unified_schema(
                metadata_args
            )
            if unified_output and isinstance(unified_output, List):
                if len(unified_output) != len(file_uuids):
                    raise RuntimeError(
                        'Inspect conversion produced a different number of '
                        'logs than the generated UUID list.'
                    )
                for single_unified_output, file_uuid in zip(
                    unified_output, file_uuids
                ):
                    save_evaluation_log(
                        single_unified_output,
                        inspect_converter,
                        file_uuid,
                    )
            else:
                raise ValueError('Inspect conversion produced no logs')
    else:
        file_uuid = str(uuid.uuid4())
        metadata_args = {
            **base_metadata_args,
            'file_uuid': file_uuid,
        }

        unified_output = inspect_converter.convert_to_unified_schema(
            metadata_args
        )

        if unified_output:
            save_evaluation_log(
                unified_output,
                inspect_converter,
                file_uuid,
            )
        else:
            raise ValueError('Inspect conversion produced no log')
