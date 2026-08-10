"""CLI for converting lighteval output to every_eval_ever format."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Convert lighteval output to every_eval_ever format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to a results JSON file or a directory containing results files',
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
        '--include_details',
        '--include-details',
        action='store_true',
        help='Also convert lighteval details parquet into instance-level '
        'output. Needs a run made with save_details and the lighteval extra '
        'installed.',
    )
    parser.add_argument(
        '--inference_platform',
        type=str,
        default=None,
        help="Inference platform (e.g. 'together', 'openai'). Read from the "
        'model config for LiteLLM and inference-provider runs; must be '
        'provided manually otherwise.',
    )
    parser.add_argument(
        '--inference_engine',
        type=str,
        default=None,
        help="Inference engine name (e.g. 'vllm', 'transformers'). lighteval "
        'dumps its model config without a backend discriminator, so this '
        'cannot be read from the logs.',
    )
    parser.add_argument(
        '--inference_engine_version',
        type=str,
        default=None,
        help="Inference engine version (e.g. '0.6.0'). "
        'Not available from lighteval logs, so must be provided manually.',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='lighteval',
        help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library. lighteval records a git SHA '
        "rather than a version, and writes '?' outside a git checkout.",
    )

    args = parser.parse_args()
    from every_eval_ever.cli import _cmd_convert_lighteval

    return _cmd_convert_lighteval(args)


if __name__ == '__main__':
    raise SystemExit(main())
