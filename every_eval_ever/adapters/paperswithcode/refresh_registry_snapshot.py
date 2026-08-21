#!/usr/bin/env python3
"""Regenerate the vendored canonical-metric snapshot from the eval-card-registry.

The adapter resolves metric bounds/direction against a committed snapshot of the
registry's canonical metrics (``registry_snapshot.json``) so it needs no registry
install at runtime. Run this (with the registry repo checked out) whenever the
registry's ``seed/metrics.yaml`` changes:

    python -m every_eval_ever.adapters.paperswithcode.refresh_registry_snapshot \
        --seed ../eval-card-registry/seed/metrics.yaml

A snapshot is authoritative-at-snapshot-time; new metrics added to the registry
only resolve here after a refresh (until then the adapter fails closed on them).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

DEFAULT_SEED = Path('../eval-card-registry/seed/metrics.yaml')
SNAPSHOT = Path(__file__).with_name('registry_snapshot.json')


def _registry_revision(seed_path: Path) -> str | None:
    """Best-effort git revision of the registry the seed was read from.

    Recorded in the snapshot ``_meta`` (and surfaced per resolved metric by the
    adapter) so a downstream consumer can pin exactly which registry commit a
    bound came from. A ``-dirty`` suffix flags an uncommitted working tree.
    Returns None if the seed is not inside a git checkout.
    """
    repo = seed_path.resolve().parent
    try:
        sha = subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    if not sha:
        return None
    status = subprocess.run(
        ['git', '-C', str(repo), 'status', '--porcelain'],
        capture_output=True,
        text=True,
    )
    return sha + ('-dirty' if status.stdout.strip() else '')


def build_snapshot(seed_path: Path) -> dict:
    rows = yaml.safe_load(seed_path.read_text(encoding='utf-8'))
    metrics = [
        {
            'id': r['id'],
            'display_name': r.get('display_name'),
            'aliases': r.get('aliases') or [],
            'score_type': r.get('score_type'),
            'lower_is_better': r.get('lower_is_better'),
            'min_score': r.get('min_score'),
            'max_score': r.get('max_score'),
            # Vetting status + confidence/kind travel with the bound so the adapter
            # can surface (never silently trust) a still-`draft` canonical entry.
            'review_status': r.get('review_status'),
            'metadata': r.get('metadata'),
        }
        for r in rows
    ]
    meta = {
        'source': 'eval-card-registry seed/metrics.yaml',
        'note': (
            'Vendored snapshot of canonical metric entries. Regenerate with '
            'refresh_registry_snapshot.py. Do not edit by hand.'
        ),
        'count': len(metrics),
    }
    revision = _registry_revision(seed_path)
    if revision:
        meta['registry_revision'] = revision
    return {'_meta': meta, 'metrics': metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, default=DEFAULT_SEED)
    ap.add_argument('--out', type=Path, default=SNAPSHOT)
    args = ap.parse_args()
    snap = build_snapshot(args.seed)
    args.out.write_text(
        json.dumps(snap, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    rev = snap['_meta'].get(
        'registry_revision', '(unknown — not a git checkout)'
    )
    print(
        f'wrote {args.out} with {snap["_meta"]["count"]} canonical metrics '
        f'@ registry {rev}'
    )


if __name__ == '__main__':
    main()
