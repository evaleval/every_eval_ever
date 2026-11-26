"""Timestamp conversion tests for the shared converter helpers."""

import os
import time

import pytest

from every_eval_ever.converters.common.utils import (
    convert_timestamp_to_unix_format,
)


def test_naive_timestamp_is_read_as_utc():
    """A stamp without an offset is UTC, not the converting machine's local time."""
    assert convert_timestamp_to_unix_format('2026-02-07T11:26:57') == '1770463617.0'
    assert convert_timestamp_to_unix_format('2026-02-07T11:26:57+00:00') == '1770463617.0'


@pytest.mark.skipif(not hasattr(time, 'tzset'), reason='TZ switching requires time.tzset')
def test_naive_timestamp_does_not_depend_on_the_local_timezone():
    """The same naive stamp converts to the same instant in every timezone."""
    original = os.environ.get('TZ')
    results = []
    try:
        for zone in ('America/New_York', 'Asia/Tokyo'):
            os.environ['TZ'] = zone
            time.tzset()
            results.append(convert_timestamp_to_unix_format('2026-02-07T11:26:57'))
    finally:
        if original is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = original
        time.tzset()

    assert results == ['1770463617.0', '1770463617.0']
