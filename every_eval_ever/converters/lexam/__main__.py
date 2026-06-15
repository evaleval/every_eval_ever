"""CLI for converting LEXam leaderboard data to every_eval_ever format."""

import argparse
import json
import sys
import uuid
from pathlib import Path

from .adapter import LEXamAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Fetch LEXam leaderboard data from GitHub and convert it '
            'to Every Eval Ever schema JSON files.'
        )
    )
    parser.add_argument(
        '--output_dir',
        default='data',
        help='Base output directory (default: data).',
    )
    args = parser.parse_args()

    adapter = LEXamAdapter()
    output_dir = Path(args.output_dir)

    try:
        logs = adapter.fetch_leaderboard()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc

    for log in logs:
        parts = log.model_info.id.split('/', 1)
        developer = parts[0] if len(parts) == 2 else 'unknown'
        model_name = parts[1] if len(parts) == 2 else log.model_info.id

        out_dir = output_dir / 'lexam' / developer / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'{uuid.uuid4()}.json'

        with out_file.open('w', encoding='utf-8') as file:
            json.dump(
                log.model_dump(mode='json', exclude_none=True),
                file,
                indent=2,
            )
        print(out_file)

    print(f'\nConverted {len(logs)} model evaluation(s).')


if __name__ == '__main__':
    main()
