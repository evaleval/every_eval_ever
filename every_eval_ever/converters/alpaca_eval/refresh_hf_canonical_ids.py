"""Refresh the vendored map of HuggingFace repo renames this adapter follows.

Run by a maintainer, not by tests or CI::

    uv run python -m every_eval_ever.converters.alpaca_eval.refresh_hf_canonical_ids

AlpacaEval's leaderboards are from 2023-2024 and repo ids have moved since:
``WizardLM`` was renamed to ``WizardLMTeam``, ``THUDM`` to ``zai-org``,
``cognitivecomputations`` to ``dphn``, and Meta dropped the ``Meta-`` prefix
from the Llama 3.1 repos. HuggingFace still answers under the old id with a
``307`` redirect, so nothing *looks* broken — but the datastore already holds
records under the current id from other sources, and a stale id silently fails
to join with them.

This script asks HuggingFace which id each published repo id resolves to and
writes the differences to :data:`~identity.HF_CANONICAL_PATH`, which
:func:`~identity.hf_canonical_ids` loads offline. Only ``GET
/api/models/<id>`` is used — reading is not authenticated, so the map records
public metadata and nothing else.

Response classes, handled differently:

* **200** — the returned ``id`` is authoritative. Recorded when it differs from
  the id we asked about.
* **3xx** — ``requests`` follows the redirect, so a rename lands as a 200 under
  the new id.
* **401** — HuggingFace conflates *gated* with *nonexistent* here
  (``databricks/dbrx-instruct`` is real but gated; ``meta/llama-2-70b-chat-hf``
  never existed). Neither yields a canonical id, so both are counted as
  unverifiable and left untouched rather than guessed at. Resolving them would
  need a token, which would make this script's output depend on whose token ran
  it.
* **anything else** — a ``429``, a ``5xx`` or an unfollowable redirect is not an
  answer. The map is rebuilt from scratch on every run, so writing one from an
  incomplete sweep would delete renames this file already records, silently
  returning the adapter to stale ids. Such a sweep leaves the file untouched and
  exits non-zero instead.

``--check`` compares against the committed map without writing, so a rename that
happened after the last refresh surfaces in review instead of silently ageing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from every_eval_ever.converters.alpaca_eval.adapter import (
    LEADERBOARDS,
    model_slug_from_row,
    model_slugs,
)
from every_eval_ever.converters.alpaca_eval.identity import (
    HF_CANONICAL_PATH,
    HF_GROUNDED_SOURCES,
    canonical_repo_casing,
    resolve_identity,
)
from every_eval_ever.converters.alpaca_eval.upstream import (
    UpstreamSnapshot,
    populate_snapshot,
)

HF_API_URL = 'https://huggingface.co/api/models/{repo_id}'
REQUEST_TIMEOUT = 60

#: Statuses that answer the question this script asks. ``200`` carries the
#: canonical id; ``401`` is HuggingFace's answer for a gated or nonexistent
#: repo. Any other status means the sweep never got an answer for that id.
ANSWERED_STATUSES = frozenset({200, 401})


def published_repo_ids(snapshot: UpstreamSnapshot) -> List[str]:
    """Return every HuggingFace repo id this adapter would publish, sorted.

    Resolution runs with the rename map switched off, so the keys of the map
    this script writes are the ids *as the source spells them* — which is what
    :func:`~identity.resolve_identity` looks up.
    """
    casing = canonical_repo_casing(snapshot.model_configs.values())
    repo_ids = set()
    for board in snapshot.leaderboards.values():
        for row in board.rows:
            slug = model_slug_from_row(row)
            if not slug:
                continue
            resolved = resolve_identity(
                slug,
                snapshot.model_configs.get(slug),
                casing,
                hf_canonical={},
            )
            if resolved and resolved.identity_source in HF_GROUNDED_SOURCES:
                repo_ids.add(resolved.model_id)
    return sorted(repo_ids)


def hf_canonical_id(repo_id: str) -> Tuple[int, Optional[str]]:
    """Return ``(status, current id)`` for one repo id.

    The id is ``None`` whenever HuggingFace does not hand one back — a gated or
    nonexistent repo (both ``401``), or anything else non-200.
    """
    response = requests.get(
        HF_API_URL.format(repo_id=repo_id), timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        return response.status_code, None
    payload = response.json()
    current = payload.get('id') if isinstance(payload, dict) else None
    return response.status_code, current if isinstance(current, str) else None


def build_map(
    snapshot: UpstreamSnapshot,
) -> Tuple[Dict[str, Any], List[Tuple[str, int]]]:
    """Sweep every published repo id: the vendorable payload, and what failed.

    The second element pairs each id HuggingFace did not answer for with the
    status it returned instead. A caller with a non-empty list holds a payload
    built from an incomplete sweep and must not publish it over the committed
    map.
    """
    repo_ids = published_repo_ids(snapshot)
    renames: Dict[str, str] = {}
    statuses: Counter = Counter()
    unverifiable: List[str] = []
    unanswered: List[Tuple[str, int]] = []
    for repo_id in repo_ids:
        status, current = hf_canonical_id(repo_id)
        statuses[status] += 1
        if status not in ANSWERED_STATUSES:
            unanswered.append((repo_id, status))
            continue
        if current is None:
            unverifiable.append(repo_id)
            continue
        if current != repo_id:
            renames[repo_id] = current
    return {
        '_meta': {
            'source': HF_API_URL.format(repo_id='<id>'),
            'note': (
                'Vendored map of HuggingFace repo renames, keyed by the id the '
                'AlpacaEval source references. Regenerate with '
                'refresh_hf_canonical_ids.py. Do not edit by hand. Only ids '
                'HuggingFace confirmed are listed: a gated or nonexistent repo '
                'answers 401 and is left as the source spells it.'
            ),
            'upstream_ref': snapshot.ref,
            'retrieved_date': datetime.now(timezone.utc).date().isoformat(),
            'counts': {
                'checked': len(repo_ids),
                'renamed': len(renames),
                'unverifiable': len(unverifiable),
                'by_status': dict(sorted(statuses.items())),
            },
            'unverifiable_repo_ids': unverifiable,
        },
        'renamed_repos': dict(sorted(renames.items())),
    }, unanswered


def _serialize(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + '\n'


def _comparable(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload minus fields that change on every run.

    ``retrieved_date`` always moves. ``unverifiable_repo_ids`` and the status
    histogram move whenever HuggingFace gates or ungates a repo, which says
    nothing about the ids this adapter publishes — the map itself is what
    ``--check`` guards.
    """
    meta = dict(payload.get('_meta') or {})
    meta.pop('retrieved_date', None)
    meta.pop('unverifiable_repo_ids', None)
    counts = dict(meta.get('counts') or {})
    counts.pop('unverifiable', None)
    counts.pop('by_status', None)
    meta['counts'] = counts
    return {**payload, '_meta': meta}


def _load_snapshot(path: Optional[Path]) -> UpstreamSnapshot:
    if path:
        return UpstreamSnapshot.from_payload(
            json.loads(path.read_text(encoding='utf-8'))
        )
    return populate_snapshot(UpstreamSnapshot(), LEADERBOARDS, model_slugs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HF_CANONICAL_PATH)
    parser.add_argument(
        '--upstream-snapshot',
        type=Path,
        default=None,
        help=(
            'read the upstream snapshot from a file written by '
            'UpstreamSnapshot.to_payload instead of fetching it from GitHub'
        ),
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='compare against the committed map without writing',
    )
    args = parser.parse_args(argv)

    payload, unanswered = build_map(_load_snapshot(args.upstream_snapshot))
    if unanswered:
        listed = ', '.join(
            f'{repo_id} ({status})' for repo_id, status in unanswered[:10]
        )
        if len(unanswered) > 10:
            listed += f', ... {len(unanswered) - 10} more'
        # Neither writing nor comparing: a sweep with holes in it cannot say
        # whether a rename is gone or was simply never asked about.
        print(
            f'HuggingFace did not answer for {len(unanswered)} repo id(s): '
            f'{listed}. Leaving {args.output.name} as it is; rerun when '
            'HuggingFace answers.',
            file=sys.stderr,
        )
        return 1
    if args.check:
        if not args.output.exists():
            print(f'{args.output} does not exist', file=sys.stderr)
            return 1
        committed = json.loads(args.output.read_text(encoding='utf-8'))
        if _comparable(committed) == _comparable(payload):
            print(f'{args.output.name} is up to date')
            return 0
        print(
            f'{args.output.name} differs from HuggingFace; rerun without '
            '--check to refresh',
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_serialize(payload), encoding='utf-8')
    counts = payload['_meta']['counts']
    print(f'wrote {args.output} ({counts})')
    for repo_id, current in payload['renamed_repos'].items():
        print(f'  {repo_id} -> {current}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
