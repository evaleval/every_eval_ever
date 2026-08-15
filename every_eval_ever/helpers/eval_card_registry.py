"""Canonical ids from the eval-card-registry, resolved offline.

The registry (``https://evaleval-entity-registry.hf.space``) is the shared
canonicalization service for EEE. This module answers four questions from it:

- ``model_info.developer`` — the canonical **organization** id. The registry
  records the organization and the HuggingFace namespace its models are
  published under as two identities for one organization (``meta`` and
  ``meta-llama``, ``alibaba`` and ``qwen``), so ``model_info.id`` keeps the
  namespace, which is the repo id that resolves, and ``developer`` carries the
  canonical org id.
- whether a publisher name is a **second name** for an organization already in
  the registry (``Mistral`` for ``mistralai``) — see :func:`second_name_of`.
  The datastore gives each publisher one directory, so one publisher under two
  names is two directories and neither listing is complete.
- the **metric** id and the score bounds declared with it.
- the **benchmark** and **harness** ids.

Resolution reads a vendored snapshot of the registry's read-only list
endpoints (:data:`SNAPSHOT_PATH`), not ``POST /api/v1/resolve``, which defaults
to ``mode="resolve"`` and **auto-creates a draft canonical** for any value it
cannot place. ``--registry-live`` opts into a live check in ``mode="exact"``,
the one mode that creates nothing; a live failure or an unresolved value falls
back to the source-derived spelling marked unverified, never fatally.

Refresh the snapshot with
``python -m every_eval_ever.tools.refresh_eval_card_registry``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

REGISTRY_BASE_URL = 'https://evaleval-entity-registry.hf.space'

SNAPSHOT_NAME = 'eval_card_registry.json'

#: Where :mod:`every_eval_ever.tools.refresh_eval_card_registry` writes the
#: snapshot in a source checkout. Reading goes through :func:`load_snapshot` so
#: an installed or zipped package works too.
SNAPSHOT_PATH = Path(__file__).resolve().parent / 'data' / SNAPSHOT_NAME

#: Only this mode resolves without creating a draft canonical.
_SIDE_EFFECT_FREE_MODE = 'exact'

_LIVE_TIMEOUT = 30

#: Snapshot section -> registry entity type. Spelled out because the entity type
#: is not the section name minus an ``s`` (``harnesses`` -> ``harness``).
_SECTION_ENTITY_TYPES = {
    'metrics': 'metric',
    'benchmarks': 'benchmark',
    'harnesses': 'harness',
}


def normalize(value: str) -> str:
    """Collapse a name to its punctuation-insensitive identity.

    ``win_rate``, ``Win Rate`` and ``win-rate`` normalize alike; so do
    ``moonshot-ai``, ``Moonshot AI`` and ``moonshotai``. This is the weakest of
    the org tiers :meth:`Registry.org` tries, since it discards punctuation the
    registry may be recording deliberately.
    """
    if not isinstance(value, str):
        return ''
    return re.sub(r'[^a-z0-9]+', '', value.strip().lower())


def load_snapshot() -> Dict[str, Any]:
    """Return the bundled registry snapshot as parsed JSON."""
    resource = resources.files('every_eval_ever.helpers').joinpath(
        'data', SNAPSHOT_NAME
    )
    return json.loads(resource.read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _snapshot() -> Dict[str, Any]:
    return load_snapshot()


def snapshot_meta() -> Dict[str, Any]:
    """Provenance of the vendored snapshot: endpoints, date, counts."""
    return dict(_snapshot().get('_meta') or {})


class Resolution(NamedTuple):
    """One canonical id, and how much weight it deserves.

    ``strategy`` and ``review_status`` are published alongside the value they
    produced, so a record distinguishes a ``reviewed`` canonical from a ``draft``
    one and either from a source-derived fallback.
    """

    raw_value: str
    entity_type: str
    canonical_id: Optional[str]
    review_status: Optional[str]
    #: ``snapshot_exact`` | ``snapshot_identifier``
    #: | ``snapshot_alias_identifier`` | ``snapshot_normalized``
    #: | ``snapshot_alias_normalized`` (see :meth:`Registry.org`) | ``snapshot``
    #: | ``live_exact`` | ``no_canonical`` | ``registry_disabled``
    #: | ``registry_unavailable``.
    strategy: str
    record: Dict[str, Any] = {}

    @property
    def resolved(self) -> bool:
        return self.canonical_id is not None

    @property
    def reviewed(self) -> bool:
        """True when a human has vetted this entry in the registry."""
        return self.review_status == 'reviewed'

    def provenance(self, prefix: str) -> Dict[str, Optional[str]]:
        """Fields to publish in ``additional_details`` for this resolution."""
        return {
            f'{prefix}_registry_id': self.canonical_id,
            f'{prefix}_registry_strategy': self.strategy,
            f'{prefix}_registry_review_status': self.review_status,
        }


def _unresolved(raw_value: str, entity_type: str, strategy: str) -> Resolution:
    return Resolution(
        raw_value=raw_value,
        entity_type=entity_type,
        canonical_id=None,
        review_status=None,
        strategy=strategy,
    )


class Registry:
    """Offline resolver over the vendored snapshot, with an opt-in live check.

    Args:
        enabled: When False every lookup returns unresolved with strategy
            ``registry_disabled``, so ``--no-registry-resolve`` produces records
            that are explicit about having had no registry opinion rather than
            records that quietly look source-derived.
        live: Additionally consult ``POST /api/v1/resolve/batch`` in
            ``mode="exact"`` for values the snapshot cannot place. Never fatal.
        base_url: Registry base URL, for tests and staging deployments.
    """

    def __init__(
        self,
        enabled: bool = True,
        live: bool = False,
        base_url: str = REGISTRY_BASE_URL,
    ) -> None:
        self.enabled = enabled
        self.live = live and enabled
        self.base_url = base_url
        #: Values the live endpoint was asked about, so a run can report it.
        self.live_queries = 0
        self.live_hits = 0
        self.live_error: Optional[str] = None
        self._live_cache: Dict[
            Tuple[str, str], Tuple[Optional[Dict[str, Any]], Optional[str]]
        ] = {}

    # -- organizations ----------------------------------------------------

    def org(self, slug: str) -> Resolution:
        """Resolve a HuggingFace namespace or org name to a canonical org id.

        Five tiers, tried in order of how much of the spelling had to be thrown
        away to get a match, with ``strategy`` recording which one answered:

        ``snapshot_exact``
            The value **is** a canonical id. Tried first because two ids can
            collapse to one normalized spelling (``DeepAuto-AI`` and
            ``deepautoai``), and such a spelling is left unowned.
        ``snapshot_identifier`` / ``snapshot_alias_identifier``
            A spelling the registry records — a canonical id, a ``hf_org``
            namespace, or a confirmed alias — matched case-insensitively.
        ``snapshot_normalized`` / ``snapshot_alias_normalized``
            The same, but only after punctuation was discarded as well:
            ``meta-llama`` is a namespace Meta declares, while ``metallama``
            merely collapses onto one.
        """
        if not self.enabled:
            return _unresolved(slug, 'org', 'registry_disabled')
        snapshot = _snapshot()
        exact = slug.strip() if isinstance(slug, str) else slug
        if exact in snapshot['org_review_status']:
            return Resolution(
                raw_value=slug,
                entity_type='org',
                canonical_id=exact,
                review_status=snapshot['org_review_status'][exact],
                strategy='snapshot_exact',
            )
        # A snapshot predating the identifier sections still resolves, by the
        # normalized tiers alone.
        folded = exact.lower() if isinstance(exact, str) else ''
        key = normalize(slug)
        for section, lookup, strategy in (
            ('org_identity_spellings', folded, 'snapshot_identifier'),
            ('org_alias_spellings', folded, 'snapshot_alias_identifier'),
            ('org_identities', key, 'snapshot_normalized'),
            ('org_aliases', key, 'snapshot_alias_normalized'),
        ):
            canonical = (snapshot.get(section) or {}).get(lookup)
            if canonical is not None:
                return Resolution(
                    raw_value=slug,
                    entity_type='org',
                    canonical_id=canonical,
                    review_status=snapshot['org_review_status'].get(canonical),
                    strategy=strategy,
                )
        return self._live('org', slug)

    # -- metrics / benchmarks / harnesses ---------------------------------

    def metric(self, name: str) -> Resolution:
        """Resolve a leaderboard column name to a canonical metric entry.

        The entry carries ``min_score``/``max_score``/``lower_is_better``, so
        the registry decides the scale a score is published on.
        """
        return self._keyed('metrics', 'metric', name)

    def benchmark(self, name: str) -> Resolution:
        return self._keyed('benchmarks', 'benchmark', name)

    def harness(self, name: str) -> Resolution:
        return self._keyed('harnesses', 'harness', name)

    def _keyed(self, section: str, entity_type: str, name: str) -> Resolution:
        """Look up a value the snapshot stores under its query spelling.

        A missing key was never asked about. A ``None`` entry is a gap the
        refresh did ask about and record — but that answer is only as current as
        the snapshot, so live mode asks again rather than making a canonical
        added since the refresh wait for the next one. Offline, both are the
        same ``no_canonical``.
        """
        if not self.enabled:
            return _unresolved(name, entity_type, 'registry_disabled')
        entries = _snapshot()[section]
        if name not in entries:
            return self._live(entity_type, name)
        entry = entries[name]
        if entry is None:
            return self._live(entity_type, name)
        return Resolution(
            raw_value=name,
            entity_type=entity_type,
            canonical_id=entry['id'],
            review_status=entry.get('review_status'),
            strategy='snapshot',
            record=entry,
        )

    # -- opt-in live path -------------------------------------------------

    def _live(self, entity_type: str, raw_value: str) -> Resolution:
        """Ask the registry directly, in the mode that creates nothing."""
        if not self.live:
            return _unresolved(raw_value, entity_type, 'no_canonical')
        payload, error = self._live_lookup(entity_type, raw_value)
        if payload is None:
            # This lookup's own error, not `live_error`, which is a sticky
            # run-level aggregate.
            strategy = 'registry_unavailable' if error else 'no_canonical'
            return _unresolved(raw_value, entity_type, strategy)
        return Resolution(
            raw_value=raw_value,
            entity_type=entity_type,
            canonical_id=payload['canonical_id'],
            review_status=payload.get('review_status'),
            strategy='live_exact',
            record=payload,
        )

    def _live_lookup(
        self, entity_type: str, raw_value: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return ``(payload, error)`` for one lookup — a hit, a miss, or a fault.

        A miss and a fault are both ``payload is None``, so they are cached and
        reported separately.
        """
        cache_key = (entity_type, raw_value)
        if cache_key in self._live_cache:
            return self._live_cache[cache_key]
        result = None
        error_text = None
        try:
            import requests

            self.live_queries += 1
            response = requests.post(
                f'{self.base_url}/api/v1/resolve',
                json={
                    'raw_value': raw_value,
                    'entity_type': entity_type,
                    'mode': _SIDE_EFFECT_FREE_MODE,
                    'source_config': 'alpaca_eval',
                },
                timeout=_LIVE_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            # Belt and braces: `mode="exact"` must never create a canonical, so
            # a response claiming it did is a contract change, not a resolution.
            if payload.get('canonical_id') and not payload.get('created_new'):
                result = payload
                self.live_hits += 1
        except Exception as error:  # never fatal — provenance, not data
            error_text = f'{type(error).__name__}: {error}'
            self.live_error = error_text
        self._live_cache[cache_key] = (result, error_text)
        return result, error_text

    # -- reporting --------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """A summary of what the registry contributed, for the run report."""
        meta = snapshot_meta()
        return {
            'enabled': self.enabled,
            'live': self.live,
            'snapshot_date': meta.get('retrieved_date'),
            'snapshot_counts': meta.get('counts'),
            'live_queries': self.live_queries,
            'live_hits': self.live_hits,
            'live_error': self.live_error,
        }


def snapshot_gaps(snapshot: Dict[str, Any]) -> List[str]:
    """``entity_type:query`` for every query *snapshot* has no canonical for."""
    return [
        f'{entity_type}:{query}'
        for section, entity_type in _SECTION_ENTITY_TYPES.items()
        for query, entry in snapshot[section].items()
        if entry is None
    ]


def gaps() -> List[str]:
    """Queries the vendored snapshot records as having no canonical entry.

    Fixing one means minting a canonical, which is a PR to the registry rather
    than something an adapter can do.
    """
    return snapshot_gaps(_snapshot())


def iter_org_identities() -> Iterable[Tuple[str, str]]:
    """(normalized spelling, canonical org id) pairs, for tests and tooling."""
    return _snapshot()['org_identities'].items()


def second_name_of(slug: str) -> Optional[str]:
    """Return the canonical org id when ``slug`` is a *second name* for it.

    A second name is a confirmed alias that is a genuinely **different** name
    for an organization the registry already knows: ``Mistral`` for
    ``mistralai``, ``AI2`` for ``allenai``, or a model family such as ``glm``
    used where its publisher belongs. Two names for one publisher split it
    across two datastore directories.

    ``None`` where a name is not evidence of a split: a canonical id, a
    HuggingFace namespace the registry records for one (``meta-llama`` is Meta),
    and a spelling the registry has never seen. Unlike :meth:`Registry.org`,
    an identity therefore gets ``None`` rather than itself.
    """
    if not isinstance(slug, str):
        return None
    key = normalize(slug)
    if not key:
        return None
    snapshot = _snapshot()
    # Redundant against a snapshot the builder wrote, which already drops these;
    # kept so a hand-edited snapshot cannot turn a canonical id into a second
    # name, since callers use the answer to warn a contributor.
    if slug.strip() in snapshot['org_review_status']:
        return None
    if key in snapshot['org_identities']:
        return None
    return snapshot['org_aliases'].get(key)
