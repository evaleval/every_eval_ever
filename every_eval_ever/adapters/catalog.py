"""The set of adapters automation may run, and how to invoke each one.

Automation needs three things an adapter module cannot state about itself:
which datastore collections it is allowed to write, the exact argv that makes
it write somewhere other than the checkout, and how long it may take. Parsing
that back out of each adapter's ``argparse`` block is what an earlier
scheduled-ingestion attempt did, and it silently mis-read adapters whose CLI
conventions differed. This module is the declaration instead, and
``tests/test_adapter_catalog.py`` checks every entry against the adapter's own
parser, so an entry cannot drift from the code it describes.

Every adapter package under ``every_eval_ever/adapters/`` must appear here or
in :data:`LEGACY_ADAPTERS`. An adapter that cannot be scheduled is listed with
``runnable=False`` and a reason, so it stays visible.

Named catalog rather than registry because "the registry" already means
``eval-card-registry`` throughout this project's docs and contribution flow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

OutputScope = Literal['collection', 'data_root']
Cadence = Literal['daily', 'weekly']

_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_MODULE_PATH = re.compile(r'^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$')
#: Every registered module must live here. Enforced over ``ADAPTERS`` by the
#: catalog test rather than by ``AdapterSpec``, so a test double can exist.
ADAPTER_MODULE_PREFIX = 'every_eval_ever.adapters.'

#: Adapter packages deliberately excluded from automation. Their upstream
#: sources are no longer usable for an active refresh; see the "Legacy
#: integrations" section of ``every_eval_ever/adapters/README.md``.
LEGACY_ADAPTERS = frozenset({'livecodebenchpro'})

#: Minutes a scheduled job gets on top of its adapter's own budget.
#: ``timeout_minutes`` bounds the adapter subprocess, and the job around it
#: also checks out the repository, installs the environment, uploads the raw
#: snapshot and commits records. Giving the job the adapter's figure meant an
#: adapter that used its full budget left nothing for the publishing, and a
#: job cancelled there loses the ledger entry for what it had already
#: uploaded. Sized for the slow case of the surrounding steps rather than the
#: usual one, since the cost of it being too large is a late cancellation and
#: the cost of it being too small is a torn publication.
JOB_TIMEOUT_BUFFER_MINUTES = 15


@dataclass(frozen=True)
class AdapterSpec:
    """One schedulable unit of adapter work.

    ``key`` is the unit's stable identity: it names the scheduled job, the
    per-adapter datastore pull request, and the raw-snapshot and state paths.
    One adapter module may contribute several units when it takes a
    source-selecting argument (``helm`` has one per leaderboard).
    """

    key: str
    module: str
    collections: tuple[str, ...]
    output_arg: str = '--output-dir'
    output_scope: OutputScope = 'collection'
    extra_args: tuple[str, ...] = ()
    cadence: Cadence = 'daily'
    weekday: int | None = None
    timeout_minutes: int = 20
    runnable: bool = True
    unrunnable_reason: str | None = None
    required_env: tuple[str, ...] = ()
    with_packages: tuple[str, ...] = ()
    allow_partial: bool = True
    #: Whether a scheduled run must leave a raw-capture manifest behind.
    #: Every adapter that fetches live sources does, through the shared fetch
    #: helpers or its own ``raw_capture`` calls, so for them a missing
    #: manifest means the evidence trail is broken, not that there was
    #: nothing to keep. Set ``False`` only for adapters that convert local
    #: files and have nothing to snapshot, so the exemption is a reviewed
    #: catalog fact rather than whatever the run happened to write.
    captures_raw: bool = True
    #: Whether an adapter exit of 75 (``EX_TEMPFAIL``) is reported as the
    #: source being unavailable rather than as a failed job. For sources
    #: known to go down for stretches, a nightly red job says nothing new;
    #: the run stays green and the report says the source was down. The
    #: adapter must exit 75 deliberately and stage nothing for it to apply,
    #: so a crash cannot dress itself up as an outage. Grant it only while
    #: an outage is expected, and take it back once the source is stable,
    #: so a real regression goes red again.
    allow_source_outage: bool = False
    notes: str = ''

    def __post_init__(self) -> None:
        if not _SAFE_COMPONENT.fullmatch(self.key):
            raise ValueError(f'adapter key is not a safe slug: {self.key!r}')
        # That a registered module is an in-tree adapter is checked over
        # ADAPTERS in tests/test_adapter_catalog.py, not here: this type is
        # also how a test stands one in.
        if not _MODULE_PATH.fullmatch(self.module):
            raise ValueError(
                f'{self.key}: module must be a dotted module path, got '
                f'{self.module!r}'
            )
        if not self.collections:
            raise ValueError(f'{self.key}: at least one collection is required')
        for collection in self.collections:
            if not _SAFE_COMPONENT.fullmatch(collection):
                raise ValueError(
                    f'{self.key}: collection is not a safe datastore path '
                    f'component: {collection!r}'
                )
        if len(set(self.collections)) != len(self.collections):
            raise ValueError(f'{self.key}: duplicate collection names')
        if self.output_scope == 'collection' and len(self.collections) != 1:
            raise ValueError(
                f'{self.key}: output_scope "collection" needs exactly one '
                'collection; use "data_root" for a multi-collection adapter'
            )
        if self.runnable == (self.unrunnable_reason is not None):
            raise ValueError(
                f'{self.key}: set unrunnable_reason if and only if '
                'runnable is False'
            )
        if self.cadence == 'weekly' and self.weekday is None:
            raise ValueError(f'{self.key}: weekly cadence requires a weekday')
        if self.cadence == 'daily' and self.weekday is not None:
            raise ValueError(f'{self.key}: daily cadence must not set weekday')
        if self.weekday is not None and not 0 <= self.weekday <= 6:
            raise ValueError(
                f'{self.key}: weekday must be 0 (Monday) to 6 (Sunday), got '
                f'{self.weekday}'
            )
        if self.timeout_minutes <= 0:
            raise ValueError(f'{self.key}: timeout_minutes must be positive')

    @property
    def package(self) -> str:
        """Return the adapter package directory name under ``adapters/``."""
        return self.module.split('.')[-2]

    @property
    def job_timeout_minutes(self) -> int:
        """Return the wall clock the whole scheduled job may take.

        The adapter's own budget plus :data:`JOB_TIMEOUT_BUFFER_MINUTES` for
        the work either side of it. Kept here rather than as arithmetic in the
        workflow so that one place decides how long a unit may take and a test
        can hold it to being longer than the subprocess it contains.
        """
        return self.timeout_minutes + JOB_TIMEOUT_BUFFER_MINUTES

    def output_dir(self, data_root: Path | str) -> Path:
        """Return the value to pass to ``output_arg`` for a staging root.

        ``data_root`` is the staging tree's ``data`` directory. Adapters
        disagree about which level their ``--output-dir`` means, so the
        catalog records it rather than the caller guessing.
        """
        data_root = Path(data_root)
        if self.output_scope == 'data_root':
            return data_root
        return data_root / self.collections[0]

    def argv(self, data_root: Path | str) -> list[str]:
        """Return the adapter arguments that stage output under ``data_root``."""
        return [
            self.output_arg,
            str(self.output_dir(data_root)),
            *self.extra_args,
        ]

    def runs_on(self, run_date: date) -> bool:
        """Return whether this unit is scheduled on ``run_date``."""
        if not self.runnable:
            return False
        if self.cadence == 'daily':
            return True
        return run_date.weekday() == self.weekday


def _helm(key: str, leaderboard: str, weekday: int) -> AdapterSpec:
    """Build one HELM leaderboard unit.

    The adapter lowercases ``--leaderboard_name`` and uses the result as the
    collection directory, so the collection is derived from the same string
    rather than repeated by hand.
    """
    return AdapterSpec(
        key=key,
        module='every_eval_ever.adapters.helm.adapter',
        collections=(leaderboard.lower(),),
        output_scope='data_root',
        extra_args=('--leaderboard_name', leaderboard),
        cadence='weekly',
        weekday=weekday,
        timeout_minutes=30,
        runnable=False,
        unrunnable_reason=(
            'the HELM leaderboards are effectively static (paused '
            '2026-08-12); a weekly refresh refetches unchanged data'
        ),
        notes=(
            'The HELM API still serves, so a manual run works; flip '
            'runnable back on if Stanford resumes publishing new results.'
        ),
    )


ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        key='arc_agi',
        module='every_eval_ever.adapters.arc_agi.adapter',
        collections=('arc-agi',),
        notes=(
            'Repointed 2026-08-12 to the JSON files behind '
            'arcprize.org/leaderboard; the old '
            '/media/data/leaderboard/evaluations.json endpoint is gone.'
        ),
    ),
    AdapterSpec(
        key='artificial_analysis',
        module='every_eval_ever.adapters.artificial_analysis.adapter',
        collections=('artificial-analysis-llms',),
        required_env=('ARTIFICIAL_ANALYSIS_API_KEY',),
    ),
    AdapterSpec(
        key='exgentic',
        module='every_eval_ever.adapters.exgentic.adapter',
        collections=('exgentic',),
        extra_args=('--from-hf',),
        with_packages=('datasets',),
        notes=(
            'Reads Exgentic/results, the successor to the retired '
            'open-agent-leaderboard-results dataset (repointed 2026-08-12).'
        ),
    ),
    AdapterSpec(
        key='global_mmlu_lite',
        module='every_eval_ever.adapters.global_mmlu_lite.adapter',
        collections=('global-mmlu-lite',),
    ),
    AdapterSpec(
        key='hal',
        module='every_eval_ever.adapters.hal.adapter',
        collections=(
            'hal-assistantbench',
            'hal-corebench-hard',
            'hal-gaia',
            'hal-online-mind2web',
            'hal-scicode',
            'hal-scienceagentbench',
            'hal-swebench-verified-mini',
            'hal-taubench-airline',
            'hal-usaco',
        ),
        output_scope='data_root',
        extra_args=('--benchmark', 'all'),
        timeout_minutes=30,
    ),
    _helm('helm_capabilities', 'HELM_Capabilities', weekday=0),
    _helm('helm_lite', 'HELM_Lite', weekday=1),
    _helm('helm_classic', 'HELM_Classic', weekday=2),
    _helm('helm_instruct', 'HELM_Instruct', weekday=3),
    _helm('helm_mmlu', 'HELM_MMLU', weekday=4),
    AdapterSpec(
        key='hfopenllm_v2',
        module='every_eval_ever.adapters.hfopenllm_v2.adapter',
        collections=('hfopenllm_v2',),
        cadence='weekly',
        weekday=4,
        timeout_minutes=45,
        runnable=False,
        unrunnable_reason=(
            'the Open LLM Leaderboard is no longer maintained upstream '
            '(space discussion 1135); the archive is frozen, so a scheduled '
            'refresh has nothing new to fetch'
        ),
        notes=(
            'The adapter still works for a one-off manual conversion of the '
            'frozen archive (around 4,576 models).'
        ),
    ),
    AdapterSpec(
        key='hle',
        module='every_eval_ever.adapters.hle.adapter',
        collections=('hle',),
    ),
    AdapterSpec(
        key='lexam',
        module='every_eval_ever.adapters.lexam.adapter',
        collections=('lexam',),
        output_scope='data_root',
    ),
    AdapterSpec(
        key='llm_stats',
        module='every_eval_ever.adapters.llm_stats.adapter',
        collections=('llm-stats',),
        required_env=('LLM_STATS_API_KEY',),
        timeout_minutes=30,
    ),
    AdapterSpec(
        key='mercor_eval',
        module='every_eval_ever.adapters.mercor_eval.adapter',
        # One collection per Mercor benchmark slug. A new benchmark on their
        # API shows up as a StagingError naming the undeclared collection,
        # which makes adding it here a decision rather than a surprise.
        collections=('apex-agents',),
        output_scope='data_root',
        required_env=('MERCOR_EVAL_API_EVALEVAL_KEY',),
        runnable=False,
        unrunnable_reason=(
            'the Mercor Exports API is broken upstream as of 2026-08-12; '
            'paused until it serves data again'
        ),
        notes=(
            'Flip runnable back on once Mercor is stable. The adapter itself '
            'is healthy: it exits 75 on an unreachable API and 1 on a '
            'rejected key, so a smoke run distinguishes the two.'
        ),
    ),
    AdapterSpec(
        key='mmlu_pro',
        module='every_eval_ever.adapters.mmlu_pro.adapter',
        collections=('mmlu-pro',),
    ),
    AdapterSpec(
        key='mt_bench',
        module='every_eval_ever.adapters.mt_bench.adapter',
        collections=('mt-bench',),
    ),
    AdapterSpec(
        key='multi_swe_bench',
        module='every_eval_ever.adapters.multi_swe_bench.adapter',
        collections=('multi-swe-bench-leaderboard',),
        cadence='weekly',
        weekday=2,
        timeout_minutes=45,
        notes='Clones the upstream submission repository.',
    ),
    AdapterSpec(
        key='openeval',
        module='every_eval_ever.adapters.openeval.adapter',
        collections=('openeval',),
        timeout_minutes=45,
    ),
    AdapterSpec(
        key='rewardbench',
        module='every_eval_ever.adapters.rewardbench.adapter',
        collections=('reward-bench',),
        cadence='weekly',
        weekday=3,
        timeout_minutes=30,
        runnable=False,
        unrunnable_reason=(
            'the RewardBench leaderboard has not been updated in a while '
            '(paused 2026-08-12); a weekly refresh refetches unchanged data'
        ),
        notes=(
            'The source still serves, so a manual run works; flip runnable '
            'back on if upstream resumes publishing new results.'
        ),
    ),
    AdapterSpec(
        key='swe_bench_verified',
        module='every_eval_ever.adapters.swe_bench_verified.adapter',
        collections=('swe-bench-verified-leaderboard',),
        cadence='weekly',
        weekday=0,
        timeout_minutes=30,
        with_packages=('datasets',),
    ),
    AdapterSpec(
        key='swe_polybench',
        module='every_eval_ever.adapters.swe_polybench.adapter',
        collections=('swe-polybench-leaderboard',),
        cadence='weekly',
        weekday=1,
        timeout_minutes=30,
        with_packages=('datasets',),
        notes='Clones the upstream submission repository.',
    ),
    AdapterSpec(
        key='terminal_bench_2',
        module='every_eval_ever.adapters.terminal_bench_2.adapter',
        collections=('terminal-bench-2.0',),
    ),
    AdapterSpec(
        key='vals_ai',
        module='every_eval_ever.adapters.vals_ai.adapter',
        collections=('vals-ai',),
        timeout_minutes=30,
    ),
    AdapterSpec(
        key='vectara_hallucination_leaderboard',
        module=(
            'every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter'
        ),
        collections=('vectara-hallucination-leaderboard',),
        cadence='weekly',
        weekday=5,
        timeout_minutes=30,
        notes='Pinned to SOURCE_COMMIT; only changes when the pin is bumped.',
    ),
    # Registered but not schedulable. Each needs a local input file because it
    # has no live fetch path; automation would have nothing to hand it.
    AdapterSpec(
        key='bfcl',
        module='every_eval_ever.adapters.bfcl.adapter',
        collections=('bfcl',),
        output_scope='data_root',
        runnable=False,
        unrunnable_reason='requires --input-csv; no live fetch path',
        captures_raw=False,
    ),
    AdapterSpec(
        key='cocoabench',
        module='every_eval_ever.adapters.cocoabench.adapter',
        collections=('cocoabench',),
        runnable=False,
        unrunnable_reason='requires --csv; no live fetch path',
        captures_raw=False,
    ),
    AdapterSpec(
        key='sciarena',
        module='every_eval_ever.adapters.sciarena.adapter',
        collections=('sciarena',),
        output_scope='data_root',
        runnable=False,
        unrunnable_reason='requires --input-json; no live fetch path',
        captures_raw=False,
    ),
)

BY_KEY: dict[str, AdapterSpec] = {spec.key: spec for spec in ADAPTERS}

if len(BY_KEY) != len(ADAPTERS):
    raise RuntimeError('duplicate adapter key in ADAPTERS')


class UnknownAdapterError(KeyError):
    """Raised when a requested adapter key is not registered."""


def get(key: str) -> AdapterSpec:
    """Return one registered adapter, or raise with the available keys."""
    try:
        return BY_KEY[key]
    except KeyError:
        raise UnknownAdapterError(
            f'unknown adapter {key!r}; registered adapters: '
            f'{", ".join(sorted(BY_KEY))}'
        ) from None


def runnable_adapters() -> tuple[AdapterSpec, ...]:
    """Return every adapter automation is allowed to run."""
    return tuple(spec for spec in ADAPTERS if spec.runnable)


def scheduled_for(
    run_date: date,
    *,
    available_env: set[str] | None = None,
) -> tuple[AdapterSpec, ...]:
    """Return the adapters due on ``run_date``.

    ``available_env`` names the credentials the caller actually holds. Passing
    it filters out adapters whose credentials are absent; leaving it ``None``
    keeps them, so a caller that wants to report a missing credential as its
    own outcome can still see the adapter.
    """
    due = tuple(spec for spec in ADAPTERS if spec.runs_on(run_date))
    if available_env is None:
        return due
    return tuple(
        spec
        for spec in due
        if all(name in available_env for name in spec.required_env)
    )


def registered_packages() -> frozenset[str]:
    """Return every adapter package directory the catalog accounts for."""
    return frozenset(spec.package for spec in ADAPTERS) | LEGACY_ADAPTERS


__all__ = [
    'ADAPTERS',
    'ADAPTER_MODULE_PREFIX',
    'BY_KEY',
    'JOB_TIMEOUT_BUFFER_MINUTES',
    'LEGACY_ADAPTERS',
    'AdapterSpec',
    'Cadence',
    'OutputScope',
    'UnknownAdapterError',
    'get',
    'registered_packages',
    'runnable_adapters',
    'scheduled_for',
]
