"""Compatibility wrapper for the package-local validator."""

from every_eval_ever.validate import *  # noqa: F401,F403
from every_eval_ever.validate import main


if __name__ == "__main__":
    raise SystemExit(main())
