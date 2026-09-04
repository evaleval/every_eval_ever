"""Check the converters' canonical metric ids against an eval-card-registry checkout.

`CANONICAL_METRIC_IDS` in `converters/common/metrics.py` is resolved by hand so a
converter needs no network and gives the same answer every run. This re-does that
resolution against a registry `seed/metrics.yaml` and reports three things: a
mapped id the registry no longer carries, a name that now resolves but is still
namespaced, and a name that resolves ambiguously (two canonical ids claim it,
which is a registry-side collision rather than anything to fix here).

    uv run python -m tools.verify_metric_ids --seed ../eval-card-registry/seed/metrics.yaml

Exits non-zero when anything disagrees, so it can gate a registry bump. Not part
of the test suite: it needs a checkout of a different repository.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from every_eval_ever.converters.common.metrics import (
    CANONICAL_METRIC_IDS,
    DISPERSION_METRICS,
    LOWER_IS_BETTER,
    METRIC_ID_REGISTRY_REVISION,
    METRIC_KINDS,
    METRIC_UNITS,
    SHARED_METRIC_BOUNDS,
)
from every_eval_ever.converters.helm.metrics import (
    HELM_HARNESS_ID,
    HELM_METRIC_BOUNDS,
)
from every_eval_ever.converters.inspect.utils import INSPECT_HARNESS_ID
from every_eval_ever.converters.lm_eval.utils import (
    LM_EVAL_HARNESS_ID,
    LM_EVAL_METRIC_BOUNDS,
)

# Every table a converter looks a metric name up in. A name in any of them is a
# name we can publish, so it is a name that needs an id — including one that has
# bounds but no `metric_kind`, which is how `mc2` escaped this check.
LOOKUP_TABLES: tuple[dict, ...] = (
    CANONICAL_METRIC_IDS,
    METRIC_KINDS,
    METRIC_UNITS,
    SHARED_METRIC_BOUNDS,
    HELM_METRIC_BOUNDS,
    LM_EVAL_METRIC_BOUNDS,
)


def normalize(surface: str) -> str:
    """Fold case and separators, keeping the characters that name a metric.

    `+` stays: chrF and chrF++ are different metrics, and dropping it makes the
    registry's `chrF++` answer to a plain `chrF`. This is deliberately stricter
    than the registry's own resolver — a verification pass should under-claim
    matches rather than invent them.
    """
    return re.sub(r'[^a-z0-9+]', '', str(surface).lower())


def registry_index(seed_path: Path) -> dict[str, set[str]]:
    """Every surface form in the registry, mapped to the canonical ids using it."""
    import yaml

    loaded = yaml.safe_load(seed_path.read_text(encoding='utf-8'))
    entries = (
        loaded['metrics']
        if isinstance(loaded, dict) and 'metrics' in loaded
        else loaded
    )

    index: dict[str, set[str]] = {}
    for entry in entries:
        surfaces = [entry['id'], entry.get('display_name')]
        surfaces.extend(entry.get('aliases') or [])
        for surface in surfaces:
            if surface:
                index.setdefault(normalize(surface), set()).add(entry['id'])
    return index


def registry_bounds(seed_path: Path) -> dict[str, tuple[float | None, float | None]]:
    """Each canonical id's declared bounds, or ``None`` where it declares none.

    An entry with no ``max_score`` asserts nothing about scale, which is the
    correct state for a metric whose range depends on the implementation that
    computed it. 117 of the registry's entries are in that state.
    """
    import yaml

    loaded = yaml.safe_load(seed_path.read_text(encoding='utf-8'))
    entries = (
        loaded['metrics']
        if isinstance(loaded, dict) and 'metrics' in loaded
        else loaded
    )
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for entry in entries:
        bounds[entry['id']] = (
            entry.get('min_score'),
            entry.get('max_score'),
        )
    return bounds


def bounds_conflicts(seed_path: Path) -> list[str]:
    """Names whose declared scale contradicts the entry they are published under.

    ``fields.md`` takes the bounds from the resolved registry entry when the
    metric resolves. A record that cites a canonical id while declaring a
    different range tells a consumer two things at once, and the consumer that
    normalizes from the entry gets the wrong number.
    """
    entry_bounds = registry_bounds(seed_path)
    #: Every table a converter declares a scale in, so the check covers all
    #: three rather than only the harness that happened to surface it.
    declared_bounds: dict[str, tuple] = {}
    for table in (
        SHARED_METRIC_BOUNDS,
        HELM_METRIC_BOUNDS,
        LM_EVAL_METRIC_BOUNDS,
    ):
        declared_bounds.update(table)
    found: list[str] = []
    for name, declared in sorted(declared_bounds.items()):
        canonical = CANONICAL_METRIC_IDS.get(name)
        if canonical is None:
            continue
        entry = entry_bounds.get(canonical)
        if entry is None:
            continue
        for position, index in (('min', 0), ('max', 1)):
            ours, theirs = declared[index], entry[index]
            if theirs is None or ours is None:
                continue
            if math.isinf(float(ours)) or float(ours) == float(theirs):
                continue
            found.append(
                f'{name} -> {canonical}: declares {position}_score '
                f'{ours}, entry says {theirs}'
            )
    return found


def harness_report(harnesses_path: Path) -> str:
    """Which converters' namespaces the registry knows as harnesses.

    A namespace the registry does not carry still works as a join key inside that
    harness; it just says so, rather than implying the registry blessed it.
    """
    if not harnesses_path.exists():
        return f'{harnesses_path.name} not found beside the metric seed'

    import yaml

    loaded = yaml.safe_load(harnesses_path.read_text(encoding='utf-8'))
    entries = (
        loaded['harnesses']
        if isinstance(loaded, dict) and 'harnesses' in loaded
        else loaded
    )
    known = {normalize(entry['id']) for entry in entries}
    return ', '.join(
        f'{slug} {"ok" if normalize(slug) in known else "NOT IN REGISTRY"}'
        for slug in (
            LM_EVAL_HARNESS_ID,
            HELM_HARNESS_ID,
            INSPECT_HARNESS_ID,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--seed',
        type=Path,
        required=True,
        help="path to the registry's seed/metrics.yaml",
    )
    args = parser.parse_args(argv)

    harnesses = args.seed.parent / 'harnesses.yaml'
    index = registry_index(args.seed)
    # Every name the converters can look up, mapped or not. LOWER_IS_BETTER is a
    # set rather than a table, so it is unioned separately.
    known_names = set(LOWER_IS_BETTER).union(*LOOKUP_TABLES)

    stale: list[str] = []
    ambiguous: list[str] = []
    newly_resolvable: list[str] = []
    conflicting = bounds_conflicts(args.seed)

    for name in sorted(known_names):
        hits = index.get(normalize(name), set())
        mapped = CANONICAL_METRIC_IDS.get(name)
        if len(hits) > 1:
            ambiguous.append(f'{name} -> {sorted(hits)}')
        if mapped is None and hits:
            newly_resolvable.append(f'{name} -> {sorted(hits)[0]}')
        elif mapped is not None and mapped not in hits:
            stale.append(
                f'{name} -> {mapped} (registry now says {sorted(hits) or "nothing"})'
            )

    # Dispersion belongs in `score_details.uncertainty`, not in the registry as a
    # metric of its own, so its absence is not counted as a gap -- but a converter
    # can still publish one namespaced (Inspect does for `var` and lone dispersion
    # metrics), so list them separately rather than dropping them from the report.
    dispersion_unregistered = sorted(
        (known_names & DISPERSION_METRICS) - set(CANONICAL_METRIC_IDS)
    )
    unregistered = sorted(
        known_names - set(CANONICAL_METRIC_IDS) - DISPERSION_METRICS
    )

    print(f'registry seed: {args.seed}')
    print(
        f"ids resolved against the map's pinned revision "
        f'{METRIC_ID_REGISTRY_REVISION} (the seed file above is read as given; '
        f'its own checkout revision is not verified)'
    )
    print(
        f'{len(CANONICAL_METRIC_IDS)} mapped, {len(known_names)} names checked'
    )
    for label, rows in (
        ('MAPPED ID NO LONGER IN THE REGISTRY', stale),
        ('NOW RESOLVABLE, STILL NAMESPACED', newly_resolvable),
        ('AMBIGUOUS IN THE REGISTRY', ambiguous),
        ('DECLARED SCALE CONTRADICTS THE ENTRY', conflicting),
    ):
        print(f'\n{label}: {len(rows)}')
        for row in rows:
            print(f'  {row}')

    # Not a failure: these are the entries to propose upstream, and the converters
    # publish them namespaced in the meantime.
    print(f'\nNAMESPACED, WANTING A REGISTRY ENTRY: {len(unregistered)}')
    for name in unregistered:
        print(f'  {name}')
    print(
        f'\nDISPERSION, NAMESPACED WHEN PUBLISHED '
        f'(belongs in uncertainty, not a registry gap): '
        f'{len(dispersion_unregistered)}'
    )
    for name in dispersion_unregistered:
        print(f'  {name}')
    print(f'\nHARNESS SLUGS: {harness_report(harnesses)}')

    return (
        1
        if stale or newly_resolvable or ambiguous or conflicting
        else 0
    )


if __name__ == '__main__':
    sys.exit(main())
