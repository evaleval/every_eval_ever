"""Refresh the vendored eval-card-registry snapshot this repo resolves against.

Run by a maintainer, not by tests or CI::

    uv run python -m every_eval_ever.tools.refresh_eval_card_registry

This reads the **read-only list endpoints** of the registry
(``https://evaleval-entity-registry.hf.space``) and writes the snapshot that
:mod:`every_eval_ever.helpers.eval_card_registry` loads offline. GETs only: the
write-capable ``POST /api/v1/resolve`` is never called here.

The snapshot is **derived**, not a verbatim mirror: it is the vocabulary a
consumer needs, keyed the way a consumer looks things up. Organizations come out
whole, because any source can publish any organization; metrics, benchmarks and
harnesses are keyed by the spellings the converters ask about
(:data:`METRIC_QUERIES` and friends), so a query the registry has no canonical
for is recorded as a known gap rather than mistaken for a stale snapshot.

``--check`` verifies the committed snapshot still matches the registry without
writing anything, so drift can be caught in review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from every_eval_ever.helpers.eval_card_registry import (
    SNAPSHOT_PATH,
    normalize,
    snapshot_gaps,
)

REGISTRY_BASE_URL = 'https://evaleval-entity-registry.hf.space'
REQUEST_TIMEOUT = 180

#: Metric names the converters look up, in the spelling their source uses. The
#: registry matches punctuation-insensitively, so one spelling per distinct name
#: is enough; a source adding a name it needs adds it here. All four are
#: AlpacaEval's leaderboard columns today.
METRIC_QUERIES = (
    'win_rate',
    'length_controlled_winrate',
    'discrete_win_rate',
    'avg_length',
)

#: Benchmark names the converters look up. AlpacaEval's two leaderboards.
BENCHMARK_QUERIES = ('AlpacaEval 1.0', 'AlpacaEval 2.0')

#: Harness names the converters look up.
HARNESS_QUERIES = ('alpaca_eval',)


def _get(endpoint: str, base_url: str, **params: Any) -> List[Dict[str, Any]]:
    """Read one registry list endpoint and return its records."""
    response = requests.get(
        f'{base_url}/api/v1/{endpoint}', params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f'expected a JSON array from {endpoint}, '
            f'got {type(payload).__name__}'
        )
    return payload


def _pick(
    records: Iterable[Dict[str, Any]], queries: Iterable[str]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Match each query against canonical ids, display names and aliases.

    Matching is punctuation- and case-insensitive (:func:`normalize`), which is
    how ``win_rate`` reaches the canonical ``win-rate``. A query that matches
    nothing maps to ``None`` and is kept in the snapshot: an explicit "the
    registry has no canonical for this" is what lets the adapter mark the metric
    unverified instead of silently attaching a wrong id.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        keys = [record.get('id'), record.get('display_name')]
        keys.extend(record.get('aliases') or [])
        for key in keys:
            if isinstance(key, str) and normalize(key):
                index.setdefault(normalize(key), record)
    return {query: index.get(normalize(query)) for query in queries}


def _metric_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the fields a score depends on, plus how vetted the entry is.

    ``min_score``/``max_score`` are the reason to consult the registry at all —
    they settle the scale a win rate is published on — and ``review_status``
    travels with them so the adapter can surface (never silently trust) a
    still-``draft`` bound.
    """
    return {
        'id': record['id'],
        'display_name': record.get('display_name'),
        'score_type': record.get('score_type'),
        'lower_is_better': record.get('lower_is_better'),
        'min_score': record.get('min_score'),
        'max_score': record.get('max_score'),
        'review_status': record.get('review_status'),
    }


def _named_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'id': record['id'],
        'display_name': record.get('display_name'),
        'review_status': record.get('review_status'),
    }


def _one_owner(claims: Dict[str, set]) -> Dict[str, str]:
    """Keep only the spellings exactly one organization claims."""
    return {
        key: next(iter(owners))
        for key, owners in claims.items()
        if len(owners) == 1
    }


def org_identity_spellings(orgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """``lowercased spelling -> canonical org id``, punctuation intact.

    :func:`org_identities` keyed by the spelling the registry records rather than
    by its punctuation-stripped form, which is the stronger of the two tiers
    :meth:`Registry.org` tries. Case is still folded: the registry aims for
    HuggingFace-true casing and HuggingFace is not consistent, so ``Qwen`` and
    ``qwen`` are one identifier. Same one-owner-wins rule.
    """
    claims: Dict[str, set] = {}
    for record in orgs:
        org_id = record.get('id')
        if not isinstance(org_id, str) or not org_id.strip():
            continue
        for spelling in (org_id, record.get('hf_org')):
            if isinstance(spelling, str) and spelling.strip():
                claims.setdefault(spelling.strip().lower(), set()).add(
                    org_id.strip()
                )
    return _one_owner(claims)


def org_alias_spellings(
    org_aliases: List[Dict[str, Any]], identities: Dict[str, str]
) -> Dict[str, str]:
    """``lowercased confirmed alias -> canonical org id``, punctuation intact.

    The case-preserving counterpart of :func:`org_second_names`, dropping toward
    silence for the same reasons: confirmed only, one organization per spelling,
    and an identity wins over an alias. Unlike that function it keeps an alias
    restating its own organization's name, since the question here is "which
    organization is this" rather than "is this a *second* name".
    """
    listed = set(identities.values())
    claims: Dict[str, set] = {}
    for record in org_aliases:
        raw, canonical = record.get('raw_value'), record.get('canonical_id')
        if record.get('status') != 'confirmed':
            continue
        if not isinstance(raw, str) or canonical not in listed:
            continue
        key = raw.strip().lower()
        if not key or key in identities:
            continue
        claims.setdefault(key, set()).add(canonical)
    return _one_owner(claims)


def org_identities(orgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """``normalized spelling -> canonical org id`` for every name of record.

    ``hf_org`` is how a canonical org id reaches the HuggingFace namespace models
    are published under (``meta`` -> ``meta-llama``, ``alibaba`` -> ``qwen``),
    and both spellings are real identities for the same organization.

    A normalized spelling resolves only when **one** organization owns it, so a
    spelling two canonical ids collapse onto is dropped rather than awarded to
    whichever sorts first — four exist today, three of them registry entries that
    are really model names (``Gemini-3-Pro(11``) and one a genuine pair of
    punctuation twins (``DeepAuto-AI``/``deepautoai``). Neither id is stranded:
    an exact canonical id resolves by name, before normalization.

    Canonical ids win over namespaces, so a namespace colliding with some
    organization's id names that organization.
    """
    claimants: Dict[str, set] = {}
    for record in orgs:
        org_id = record.get('id')
        if isinstance(org_id, str) and normalize(org_id):
            claimants.setdefault(normalize(org_id), set()).add(org_id)

    identities: Dict[str, str] = _one_owner(claimants)
    for record in sorted(orgs, key=lambda record: str(record.get('id', ''))):
        namespace, org_id = record.get('hf_org'), record.get('id')
        if isinstance(namespace, str) and isinstance(org_id, str):
            key = normalize(namespace)
            if key and key not in claimants:
                identities.setdefault(key, org_id)
    return identities


def org_hf_namespaces(orgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """``canonical org id -> hf_org``, the namespace that org publishes under.

    The reverse of the namespace tier in :func:`org_identities`. A vendor's own
    website names the organization (``ai.meta.com`` -> ``meta``) while a model
    repo lives under the namespace (``meta-llama``), so a source that offers
    only a website needs this direction to reach a repo id that resolves.

    Recorded only where the registry declares one, so an org with no HuggingFace
    presence stays absent rather than mapping to itself.
    """
    return {
        record['id']: record['hf_org']
        for record in orgs
        if isinstance(record.get('id'), str)
        and isinstance(record.get('hf_org'), str)
        and record['hf_org']
    }


def org_second_names(
    org_aliases: List[Dict[str, Any]], identities: Dict[str, str]
) -> Dict[str, str]:
    """``normalized alias -> canonical org id``, for genuinely other names.

    Four rules, and every one of them drops toward silence — this vocabulary
    decides a *published* developer id and holds back a validator warning, so a
    wrong entry is worse than a missing one:

    - **Confirmed only.** An unconfirmed alias is the registry's guess.
    - **The alias must point at an organization the registry lists.** An alias
      naming a canonical id that no longer exists is stale, not a second name.
    - **An identity wins.** A spelling that normalizes onto a canonical id or a
      recorded namespace is dropped, whether it lands on *its own* organization
      (``Mistral AI`` for ``mistralai`` — no information) or on a different one.
      The second case is the one that matters: the registry has ``ai21-labs`` as
      its own canonical id while confirming ``AI21 Labs`` as an alias of
      ``ai21``, so keeping the alias would claim a spelling another organization
      already answers to. Six publishers are in that position today
      (``ai21``/``ai21-labs``, ``ibm``/``ibm-granite``,
      ``inception``/``inceptionlabs``, ``LGAI-EXAONE``/``lg-ai``,
      ``internlm``/``shanghai-ai-lab``, ``LiquidAI``/``liquid``). Until the
      registry settles which id is primary, saying nothing cannot be wrong.
    - **One organization per normalized spelling.** Aliases that disagree about
      the organization are dropped as ambiguous rather than resolved by
      whichever the endpoint returned first.
    """
    listed = set(identities.values())
    by_key: Dict[str, Tuple[str, str]] = {}
    ambiguous = set()
    for record in org_aliases:
        raw, canonical = record.get('raw_value'), record.get('canonical_id')
        if record.get('status') != 'confirmed':
            continue
        if not isinstance(raw, str) or canonical not in listed:
            continue
        key = normalize(raw)
        if not key or key in identities:
            continue
        seen = by_key.get(key)
        if seen is not None:
            if seen[1] != canonical:
                ambiguous.add(key)
            # One spelling per normalized form; the lexicographically smallest
            # is arbitrary but stable, so a refresh diffs cleanly.
            if seen[0] <= raw.strip():
                continue
        by_key[key] = (raw.strip(), canonical)
    return {
        key: canonical
        for key, (_, canonical) in by_key.items()
        if key not in ambiguous
    }


def org_review_status(orgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """``canonical org id -> review_status``, for ids that declare one.

    Travels with the vocabulary so a consumer can tell a ``reviewed``
    organization from one the registry auto-created as a ``draft``.
    """
    return {
        record['id']: record['review_status']
        for record in orgs
        if isinstance(record.get('id'), str) and record.get('review_status')
    }


def build_snapshot(base_url: str = REGISTRY_BASE_URL) -> Dict[str, Any]:
    """Derive the offline vocabulary from the registry's list endpoints."""
    orgs = _get('orgs', base_url)
    aliases = _get('aliases', base_url, entity_type='org')
    metrics = _get('metrics', base_url)
    benchmarks = _get('benchmarks', base_url)
    harnesses = _get('harnesses', base_url)

    identities = org_identities(orgs)
    second_names = org_second_names(aliases, identities)
    review_status = org_review_status(orgs)
    spellings = org_identity_spellings(orgs)
    hf_namespaces = org_hf_namespaces(orgs)
    alias_spellings = org_alias_spellings(aliases, spellings)

    return {
        '_meta': {
            'source': f'{base_url}/api/v1 read-only list endpoints',
            'endpoints': [
                'orgs',
                'aliases?entity_type=org',
                'metrics',
                'benchmarks',
                'harnesses',
            ],
            'note': (
                'Vendored snapshot of eval-card-registry canonical entries. '
                'Regenerate with '
                'python -m every_eval_ever.tools.refresh_eval_card_registry. '
                'Do not edit by hand. Derived, not a verbatim mirror: org '
                'spellings are recorded twice, once case-folded and once also '
                'punctuation-stripped, so a consumer can tell a recorded '
                'identifier from a spelling that only collapses onto one; an '
                'alias is dropped when it restates an identity or points at '
                'two organizations. '
                'Authoritative at snapshot time: entries added to the '
                'registry later resolve here only after a refresh.'
            ),
            'retrieved_date': datetime.now(timezone.utc)
            .date()
            .isoformat(),
            'counts': {
                'orgs': len(orgs),
                'org_aliases_confirmed': len(second_names),
                'org_identities': len(identities),
                'org_identity_spellings': len(spellings),
                'org_hf_namespaces': len(hf_namespaces),
                'org_alias_spellings': len(alias_spellings),
                'metrics': len(metrics),
                'benchmarks': len(benchmarks),
                'harnesses': len(harnesses),
            },
        },
        'org_identity_spellings': dict(sorted(spellings.items())),
        'org_alias_spellings': dict(sorted(alias_spellings.items())),
        'org_identities': dict(sorted(identities.items())),
        'org_aliases': dict(sorted(second_names.items())),
        'org_review_status': dict(sorted(review_status.items())),
        'org_hf_namespaces': dict(sorted(hf_namespaces.items())),
        'metrics': {
            query: (_metric_entry(record) if record else None)
            for query, record in _pick(metrics, METRIC_QUERIES).items()
        },
        'benchmarks': {
            query: (_named_entry(record) if record else None)
            for query, record in _pick(benchmarks, BENCHMARK_QUERIES).items()
        },
        'harnesses': {
            query: (_named_entry(record) if record else None)
            for query, record in _pick(harnesses, HARNESS_QUERIES).items()
        },
    }


def _serialize(snapshot: Dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=False) + '\n'


def _comparable(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """The snapshot minus fields that change on every run."""
    trimmed = dict(snapshot)
    meta = dict(trimmed.get('_meta') or {})
    meta.pop('retrieved_date', None)
    trimmed['_meta'] = meta
    return trimmed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=REGISTRY_BASE_URL)
    parser.add_argument('--output', type=Path, default=SNAPSHOT_PATH)
    parser.add_argument(
        '--check',
        action='store_true',
        help='compare against the committed snapshot without writing',
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot(args.base_url)
    if args.check:
        if not args.output.exists():
            print(f'{args.output} does not exist', file=sys.stderr)
            return 1
        committed = json.loads(args.output.read_text(encoding='utf-8'))
        if _comparable(committed) == _comparable(snapshot):
            print(f'{args.output.name} is up to date')
            return 0
        print(
            f'{args.output.name} differs from the registry; rerun without '
            '--check to refresh',
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_serialize(snapshot), encoding='utf-8')
    counts = snapshot['_meta']['counts']
    print(f'wrote {args.output} ({counts})')
    gaps = snapshot_gaps(snapshot)
    if gaps:
        print('no canonical entry for: ' + ', '.join(gaps))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
