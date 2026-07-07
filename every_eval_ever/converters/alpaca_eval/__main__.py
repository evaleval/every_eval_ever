"""CLI for converting AlpacaEval leaderboard data to every_eval_ever format."""

import argparse
import json
import sys
import uuid
from pathlib import Path

from .adapter import LEADERBOARDS, AlpacaEvalAdapter


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

    adapter = AlpacaEvalAdapter()
    versions = [args.version] if args.version else list(LEADERBOARDS.keys())
    output_dir = Path(args.output_dir)

    total = 0
    for version in versions:
        cfg_name = LEADERBOARDS[version]['source_name']
        print(f'\n=== {cfg_name} ===')
        try:
            logs = adapter.fetch_leaderboard(version)
        except Exception as exc:
            print(f'  ERROR: {exc}', file=sys.stderr)
            continue

        benchmark_key = f'alpaca_eval_{version}'

        for log in logs:
            parts = log.model_info.id.split('/', 1)
            developer = parts[0] if len(parts) == 2 else 'unknown'
            model_name = parts[1] if len(parts) == 2 else log.model_info.id

            out_dir = output_dir / benchmark_key / developer / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f'{uuid.uuid4()}.json'

            with out_file.open('w', encoding='utf-8') as f:
                json.dump(
                    log.model_dump(mode='json', exclude_none=True), f, indent=2
                )
            print(f'  {out_file}')
            total += 1

    print(f'\nConverted {total} model evaluation(s).')


if __name__ == '__main__':
    main()
