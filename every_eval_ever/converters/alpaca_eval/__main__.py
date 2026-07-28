"""CLI for converting AlpacaEval leaderboard data to every_eval_ever format."""

import argparse
import json
import uuid
from pathlib import Path

from every_eval_ever.helpers.io import datastore_output_dir

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
        logs = adapter.fetch_leaderboard(version)
        if not logs:
            raise ValueError(
                f'AlpacaEval conversion produced no logs for {version}'
            )

        benchmark_key = f'alpaca_eval_{version}'

        for log in logs:
            out_dir = datastore_output_dir(
                output_dir,
                benchmark_key,
                log.model_info.id,
                log.model_info.developer,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f'{uuid.uuid4()}.json'
            serialized = json.dumps(
                log.model_dump(mode='json', exclude_none=True),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            out_file.write_text(serialized + '\n', encoding='utf-8')
            print(f'  {out_file}')
            total += 1

    print(f'\nConverted {total} model evaluation(s).')


if __name__ == '__main__':
    main()
