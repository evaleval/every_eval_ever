"""Compatibility wrapper for the package-local duplicate checker."""

from every_eval_ever.check_duplicate_entries import *  # noqa: F401,F403
from every_eval_ever.check_duplicate_entries import main


if __name__ == "__main__":
    raise SystemExit(main())
