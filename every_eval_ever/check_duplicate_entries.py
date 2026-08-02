"""Compatibility wrapper for the canonical duplicate checker package."""

from .validator.check_duplicate_entries import *  # noqa: F403
from .validator.check_duplicate_entries import main

if __name__ == '__main__':
    raise SystemExit(main())
