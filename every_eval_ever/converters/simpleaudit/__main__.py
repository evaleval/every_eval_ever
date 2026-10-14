"""CLI for converting SimpleAudit output to every_eval_ever format."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Convert SimpleAudit audit results (single-run or '
            'repeated-experiment JSON) to every_eval_ever format'
        )
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to a SimpleAudit results JSON file or directory of them',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for converted files',
    )
    parser.add_argument(
        '--source_organization_name',
        type=str,
        default='',
        help='Name of the organization that ran the evaluation',
    )
    parser.add_argument(
        '--evaluator_relationship',
        type=str,
        default='third_party',
        choices=['first_party', 'third_party', 'collaborative', 'other'],
        help='Relationship of the evaluator to the model',
    )
    parser.add_argument(
        '--source_organization_url',
        type=str,
        default=None,
        help='URL of the source organization',
    )
    parser.add_argument(
        '--source_organization_logo_url',
        type=str,
        default=None,
        help='Logo of the source organization',
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default=None,
        help=(
            'Model identifier for runs that do not name their model. Single-run '
            'files carry no model name unless studio_provenance.target_model is '
            'present; repeated-experiment files use their runs key.'
        ),
    )
    parser.add_argument(
        '--include_samples',
        action='store_true',
        help='Include instance-level per-scenario data as a sidecar JSONL',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='simpleaudit',
        help='Name of the evaluation library',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library',
    )

    args = parser.parse_args()
    from every_eval_ever.cli import _cmd_convert_simpleaudit

    return _cmd_convert_simpleaudit(args)


if __name__ == '__main__':
    raise SystemExit(main())
