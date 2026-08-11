from __future__ import annotations

from argparse import ArgumentParser
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


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--log_path',
        type=str,
        default='tests/data/helm/mmlu-subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
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

def main() -> int:
    args = parse_args()
    from every_eval_ever.cli import _cmd_convert_helm

    return _cmd_convert_helm(args)


if __name__ == '__main__':
    raise SystemExit(main())
