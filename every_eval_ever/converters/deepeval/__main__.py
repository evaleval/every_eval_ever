"""CLI for converting DeepEval output to every_eval_ever format."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Convert DeepEval output to every_eval_ever format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to a DeepEval test-run JSON file or a directory '
        'containing test_run_*.json exports',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for converted files',
    )
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='HF-format id of the model under test (e.g. '
        'meta-llama/Llama-3.1-8B). DeepEval logs do not record it, so '
        'it must be provided manually.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Display name of the model under test. Defaults to --model_id.',
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
        '--eval_library_name',
        type=str,
        default='deepeval',
        help='Name of the evaluation library (e.g. inspect_ai, lm_eval, helm)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library. It should be extracted in the adapter if available in the evaluation log.',
    )

    args = parser.parse_args()
    from every_eval_ever.cli import _cmd_convert_deepeval

    return _cmd_convert_deepeval(args)


if __name__ == '__main__':
    raise SystemExit(main())
