"""Canonical validation package for Every Eval Ever files."""

from .json_utils import StrictJSONError, strict_json_loads
from .validate import *  # noqa: F403
from .validate import __all__ as _validate_all

__all__ = [
    *_validate_all,
    'StrictJSONError',
    'strict_json_loads',
]
