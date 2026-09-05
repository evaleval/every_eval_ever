"""``python -m every_eval_ever.converters.alpaca_eval`` — the shared CLI.

Every option comes from ``every_eval_ever.cli``'s ``convert alpaca_eval``
parser, so this entry point cannot fall behind it.
"""

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    from every_eval_ever.cli import main as cli_main

    if argv is None:
        argv = sys.argv[1:]
    return cli_main(['convert', 'alpaca_eval', *argv])


if __name__ == '__main__':
    raise SystemExit(main())
