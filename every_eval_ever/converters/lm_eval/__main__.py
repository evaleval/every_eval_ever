"""CLI for converting lm-evaluation-harness output to every_eval_ever format."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Convert lm-evaluation-harness output to every_eval_ever format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to results JSON file or directory containing results files',
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
        default='first_party',
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
        '--include_samples',
        action='store_true',
        help='Include instance-level sample data (requires --log_samples in original eval)',
    )
    parser.add_argument(
        '--inference_engine',
        type=str,
        default=None,
        help="Override inference engine name (e.g. 'vllm', 'transformers'). "
        'Auto-detected from model type when possible.',
    )
    parser.add_argument(
        '--inference_engine_version',
        type=str,
        default=None,
        help="Inference engine version (e.g. '0.6.0'). "
        'Not available from lm-eval logs, so must be provided manually.',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='lm_eval',
        help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library. It should be extracted in the adapter if available in the evaluation log.',
    )

    args = parser.parse_args()
    from every_eval_ever.cli import _cmd_convert_lm_eval

    return _cmd_convert_lm_eval(args)


if __name__ == '__main__':
    raise SystemExit(main())
