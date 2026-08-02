"""Canonical module entry point for the validation CLI."""

from .validate import main

__all__ = ['main']


if __name__ == '__main__':
    raise SystemExit(main())
