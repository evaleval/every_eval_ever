from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
except ImportError as exc:
    raise SystemExit(
        "The 'inspect-ai' package is required to use the Inspect AI converter.\n"
        'Install it with: uv sync --extra inspect'
    ) from exc

from every_eval_ever.eval_types import EvaluationLog
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

def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    from every_eval_ever.cli import _cmd_convert_inspect

    return _cmd_convert_inspect(args)


if __name__ == '__main__':
    raise SystemExit(main())
