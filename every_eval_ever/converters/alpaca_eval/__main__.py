"""CLI for converting AlpacaEval leaderboard data to every_eval_ever format."""

import argparse

from .adapter import LEADERBOARDS


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Fetch AlpacaEval leaderboard data from GitHub and convert it '
            'to Every Eval Ever schema JSON files.'
        )
    )
    parser.add_argument(
        '--version',
        choices=list(LEADERBOARDS.keys()),
        default=None,
        help=(
            'Which leaderboard to convert. '
            'Omit to convert all versions (default).'
        ),
    )
    parser.add_argument(
        '--output_dir',
        default='data',
        help='Base output directory (default: data).',
    )
    args = parser.parse_args()
    args.source_organization_name = 'unknown'
    args.evaluator_relationship = 'third_party'
    args.source_organization_url = None
    args.source_organization_logo_url = None
    args.eval_library_name = 'alpaca_eval'
    args.eval_library_version = 'unknown'

    from every_eval_ever.cli import _cmd_convert_alpaca_eval

    return _cmd_convert_alpaca_eval(args)


if __name__ == '__main__':
    raise SystemExit(main())
