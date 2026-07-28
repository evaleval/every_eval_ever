from __future__ import annotations

import json
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    from every_eval_ever.converters.helm.adapter import HELMAdapter
except ImportError as exc:
    raise SystemExit(
        "The 'crfm-helm' package is required to use the HELM converter.\n"
        'Install it with: uv sync --extra helm'
    ) from exc

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers.io import datastore_output_dir, require_uuid4


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--log_path',
        type=str,
        default='tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
        help='Path to directory with single evaluaion or multiple evaluations to convert',
    )
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument(
        '--source_organization_name',
        type=str,
        help='Orgnization which pushed evaluation.',
    )
    parser.add_argument(
        '--evaluator_relationship',
        type=str,
        default='other',
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
        default='helm',
        help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library',
    )

    args = parser.parse_args()
    return args


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class HELMEvalLogConverter:
    def __init__(
        self, log_path: str | Path, output_dir: str = 'unified_schema/helm'
    ):
        """
        HELM generates log file for an evaluation.
        """
        self.log_path = Path(log_path)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_unified_schema(
        self, metadata_args: Dict[str, Any] = None
    ) -> Union[EvaluationLog, List[EvaluationLog]]:
        return HELMAdapter().transform_from_directory(
            self.log_path,
            metadata_args=metadata_args,
            output_path=str(self.output_dir),
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
        print(
            f'Unified eval log was successfully saved to {output_filepath} file.'
        )
        return True


def save_evaluation_log(
    unified_output: EvaluationLog,
    helm_converter: HELMEvalLogConverter,
    file_uuid: str,
) -> bool:
    file_uuid = require_uuid4(file_uuid)
    if not unified_output.evaluation_results:
        raise ValueError('HELM output contains no evaluation results')
    output_dir = datastore_output_dir(
        helm_converter.output_dir,
        unified_output.evaluation_results[0].source_data.dataset_name,
        unified_output.model_info.id,
        unified_output.model_info.developer,
    )
    filedir = output_dir.relative_to(helm_converter.output_dir).as_posix()
    filename = f'{file_uuid}.json'
    return helm_converter.save_to_file(unified_output, filedir, filename)


def main() -> int:
    args = parse_args()
    from every_eval_ever.cli import _cmd_convert_helm

    return _cmd_convert_helm(args)


if __name__ == '__main__':
    raise SystemExit(main())
