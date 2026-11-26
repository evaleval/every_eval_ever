#!/usr/bin/env python3
"""Regenerate the vendored model-id snapshot from the eval-card-registry.

The adapter takes ``model_info.id`` from a committed snapshot
(``registry_snapshot.json``) so it needs no registry install at runtime. Run
this with the registry repo checked out and seeded locally
(``uv run eval-entity-registry seed --local`` there), from an environment that
has the registry's ``eval_entity_resolver`` package:

    python -m every_eval_ever.adapters.smb_bench.refresh_registry_snapshot \
        --registry ../eval-card-registry --input-dir <dataset checkout>

Each model is resolved with the registry's own resolver: first on the model id
that was served, then on that id under the organization the benchmark's own
model label resolves to, then on the label. A match is kept when the registry
matched it exactly or after normalization and the canonical id names its
organization; anything else stays unresolved and the adapter publishes that
model under the source's spelling, marked unverified.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from every_eval_ever.adapters.smb_bench.adapter import (
    DEFAULT_REVISION,
    HF_REPO,
    MODELS_FILE,
    _read_parquet,
)

SNAPSHOT = Path(__file__).with_name('registry_snapshot.json')
ACCEPTED_STRATEGIES = ('exact', 'normalized')


def _git_revision(repo: Path) -> str:
    sha = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ['git', '-C', str(repo), 'status', '--porcelain', '--', 'seed'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f'{sha}-dirty' if dirty else sha


def _candidates(resolver, row: dict) -> list[tuple[str, str]]:
    """Spellings to try for one model, most specific first.

    The served id comes first. A served id that carries no organization is
    then tried under the organization the benchmark's own label resolves to,
    which keeps a versioned id (``muse-ember-1.0``) from falling back to the
    unversioned label. The label itself is last.
    """
    served, label = row['model_id'], row['model']
    candidates = [(served, 'model_id')]
    if '/' not in served:
        by_label = resolver.resolve(raw_value=label, entity_type='model')
        if by_label.canonical_id and '/' in by_label.canonical_id:
            prefix = by_label.canonical_id.split('/')[0]
            candidates.append(
                (f'{prefix}/{served}', "model_id under the label's org")
            )
    candidates.append((label, 'model'))
    return candidates


def resolve_models(resolver, models: list[dict]) -> dict[str, dict]:
    resolved = {}
    for row in sorted(models, key=lambda item: item['model']):
        entry = {
            'served_model_id': row['model_id'],
            'canonical_id': None,
            'strategy': 'no_match',
        }
        for raw, field in _candidates(resolver, row):
            result = resolver.resolve(raw_value=raw, entity_type='model')
            canonical = result.canonical_id
            if (
                canonical
                and '/' in canonical
                and result.strategy in ACCEPTED_STRATEGIES
            ):
                org = resolver.resolve(
                    raw_value=canonical.split('/')[0], entity_type='org'
                )
                entry.update(
                    canonical_id=canonical,
                    strategy=result.strategy,
                    confidence=result.confidence,
                    review_status=result.review_status,
                    matched_value=raw,
                    matched_field=field,
                    org_id=org.canonical_id,
                    open_weights=result.open_weights,
                )
                break
        resolved[row['model']] = entry
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    args = parser.parse_args(argv)

    from eval_entity_resolver import Resolver, ResolverConfig

    resolver = Resolver.from_parquet(
        str(args.registry / 'fixtures'), config=ResolverConfig(threshold=0.85)
    )
    models, _ = _read_parquet(args.input_dir / MODELS_FILE)
    resolved = resolve_models(resolver, models)
    snapshot = {
        '_meta': {
            'source': (
                'eval-card-registry seed, resolved with eval_entity_resolver '
                'on a local `seed --local` build'
            ),
            'note': (
                'Vendored snapshot of the registry ids for the models in '
                'SMB Bench. Regenerate with refresh_registry_snapshot.py. '
                'Do not edit by hand.'
            ),
            'registry_revision': _git_revision(args.registry),
            'retrieved_date': datetime.date.today().isoformat(),
            'dataset': f'{HF_REPO}@{args.revision}',
            'count': len(resolved),
            'unresolved': sorted(
                label
                for label, entry in resolved.items()
                if not entry['canonical_id']
            ),
        },
        'models': resolved,
    }
    SNAPSHOT.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f'wrote {SNAPSHOT} ({len(resolved)} models)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
