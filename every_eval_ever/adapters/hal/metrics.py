"""Stable metric identities for HAL benchmark records."""

from __future__ import annotations

import re


def hal_metric_identity(
    benchmark_slug: str, evaluation_name: str
) -> tuple[str, str]:
    """Return a deterministic metric ID/name or reject an unknown shape."""
    slug = benchmark_slug.strip().casefold()
    name = evaluation_name.strip()
    if not slug or not name:
        raise ValueError('HAL benchmark slug and evaluation name are required')
    level = re.search(r'level\s*([123])', name, re.IGNORECASE)
    if level is not None:
        if slug != 'gaia':
            raise ValueError(f'HAL level metric outside GAIA: {name!r}')
        number = level.group(1)
        return (
            f'hal.gaia.level{number}_accuracy',
            f'GAIA Level {number} Accuracy',
        )
    return f'hal.{slug}.accuracy', f'{name} Accuracy'


__all__ = ['hal_metric_identity']
