"""Compatibility wrapper for :mod:`every_eval_ever.validator.validate`."""

from .validator import validate as _implementation
from .validator.validate import *  # noqa: F403
from .validator.validate import main

__all__ = _implementation.__all__


if __name__ == '__main__':
    raise SystemExit(main())
