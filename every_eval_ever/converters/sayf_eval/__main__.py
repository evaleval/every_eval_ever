"""CLI for converting sayf-eval results records to every_eval_ever format."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Convert sayf-eval results records to every_eval_ever format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to a sayf-eval results record JSON, or a run output directory '
        '(searched recursively for results_*.json).',
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
        default='unknown',
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
        '--collection_prefix',
        type=str,
        default='sayf-eval-',
        help='Prefix for the per-task datastore collection '
        '(data/<prefix><task>/...). Upstream dataset names are kept in source_data.',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='sayf-eval',
        help='Name of the evaluation library (recorded in eval_library.name)',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Fallback eval library version (the record embeds its own version).',
    )

    args = parser.parse_args()
    from every_eval_ever.cli import _cmd_convert_sayf_eval

    return _cmd_convert_sayf_eval(args)


if __name__ == '__main__':
    raise SystemExit(main())
