"""Which adapters the daily refresh may run, and how to run each one.

Adapter CLIs are not uniform: some need a hand-supplied input file, some are
pinned to a source commit, some fetch several leaderboards one invocation at a
time. This module records those differences once so the runner and the workflow
do not each re-derive them, and so adding an adapter to the cron is a data
change rather than a code change.

Every adapter directory must appear either in :data:`CRON_ADAPTERS` or in
:data:`EXCLUDED_ADAPTERS` with a reason. ``tests/test_cron_schedule.py``
enforces that, so a newly added adapter cannot quietly sit outside the cron.

This is the cron's own schedule. It is unrelated to the canonical entity ids in
the `eval-card-registry <https://github.com/evaleval/eval-card-registry>`_.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ADAPTERS_PACKAGE = 'every_eval_ever.adapters'

# Placeholder the runner substitutes with the run's raw-capture directory.
RAW_DIR_PLACEHOLDER = '{raw_dir}'


class RawPolicy(str, Enum):
    """Where a run's raw source data comes from.

    The cron archives payloads it scrapes and deliberately does not re-archive
    data that upstream already stores immutably.
    """

    #: Fetched through ``helpers.fetch``, so the shared capture hook archives it.
    VIA_FETCH_HELPERS = 'via_fetch_helpers'
    #: Archived by the adapter's own ``--save-raw-*`` flag.
    VIA_ADAPTER_FLAG = 'via_adapter_flag'
    #: Already immutable upstream (a HF dataset revision, a git commit).
    UPSTREAM_VERSIONED = 'upstream_versioned'
    #: Not archived: the adapter calls an HTTP client directly and has no flag.
    NOT_CAPTURED = 'not_captured'


@dataclass(frozen=True)
class CronAdapter:
    """One unit of cron work: one adapter, one datastore pull request."""

    name: str
    raw_policy: RawPolicy
    #: One argv tuple per invocation. An adapter that exposes a leaderboard as
    #: a required choice needs one entry per leaderboard; their records share a
    #: single pull request because they share an adapter.
    runs: tuple[tuple[str, ...], ...] = ((),)
    #: Extra argv appended to every run, with RAW_DIR_PLACEHOLDER substituted.
    raw_args: tuple[str, ...] = ()
    #: Environment variables the source requires. An enabled adapter missing
    #: one fails its run visibly rather than being quietly dropped.
    requires_env: tuple[str, ...] = ()
    #: True only when the source itself needs an authenticated Hugging Face
    #: read. The child then receives EEE_SOURCE_HF_TOKEN — a separate,
    #: least-privilege read token — as HF_TOKEN. The cron's own write-capable
    #: token is never forwarded to adapter code.
    source_hf_token: bool = False
    enabled: bool = True
    notes: str = ''

    @property
    def module(self) -> str:
        return f'{ADAPTERS_PACKAGE}.{self.name}.adapter'

    def argv_for(self, run: tuple[str, ...], raw_dir: Path | str) -> list[str]:
        """Return the full argv for one invocation."""
        raw_args = [
            argument.replace(RAW_DIR_PLACEHOLDER, str(raw_dir))
            for argument in self.raw_args
        ]
        return ['-m', self.module, *run, *raw_args]

    def missing_env(self, environment: dict[str, str]) -> list[str]:
        """Return the required environment variables that are not set."""
        return [
            name
            for name in self.requires_env
            if not (environment.get(name) or '').strip()
        ]


CRON_ADAPTERS: tuple[CronAdapter, ...] = (
    CronAdapter(
        name='artificial_analysis',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=(
            '--save-raw-json',
            f'{RAW_DIR_PLACEHOLDER}/artificial-analysis.json',
        ),
        requires_env=('ARTIFICIAL_ANALYSIS_API_KEY',),
    ),
    CronAdapter(
        name='exgentic',
        raw_policy=RawPolicy.UPSTREAM_VERSIONED,
        runs=(('--from-hf',),),
        notes='Reads a HuggingFace dataset; upstream keeps the revisions.',
    ),
    CronAdapter(
        name='global_mmlu_lite',
        raw_policy=RawPolicy.VIA_FETCH_HELPERS,
    ),
    CronAdapter(
        name='hal',
        raw_policy=RawPolicy.NOT_CAPTURED,
        notes=(
            'Calls requests directly and exposes no raw-dump flag, so this '
            'run is gated on its output fingerprint rather than raw data.'
        ),
    ),
    CronAdapter(
        name='helm',
        raw_policy=RawPolicy.VIA_FETCH_HELPERS,
        runs=(
            ('--leaderboard_name', 'HELM_Capabilities'),
            ('--leaderboard_name', 'HELM_Lite'),
            ('--leaderboard_name', 'HELM_Classic'),
            ('--leaderboard_name', 'HELM_Instruct'),
            ('--leaderboard_name', 'HELM_MMLU'),
        ),
        notes='One invocation per leaderboard; all five share one PR.',
    ),
    CronAdapter(
        name='hfopenllm_v2',
        raw_policy=RawPolicy.VIA_FETCH_HELPERS,
        enabled=False,
        notes=(
            'Emits one record per leaderboard model (>4500 per run). Until '
            'record-level de-duplication exists, a daily refresh would add '
            'thousands of near-duplicate files to its PR every time the '
            'leaderboard moves. Enable this entry once de-duplication lands.'
        ),
    ),
    CronAdapter(
        name='hle',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-json', f'{RAW_DIR_PLACEHOLDER}/hle.json'),
    ),
    CronAdapter(
        name='lexam',
        raw_policy=RawPolicy.NOT_CAPTURED,
        notes=(
            'Scrapes HTML with requests directly and exposes --input-html but '
            'no --save-raw-html, so nothing is archived for it yet.'
        ),
    ),
    CronAdapter(
        name='llm_stats',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-json', f'{RAW_DIR_PLACEHOLDER}/llm-stats.json'),
        requires_env=('LLM_STATS_API_KEY',),
    ),
    CronAdapter(
        name='mmlu_pro',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-csv', f'{RAW_DIR_PLACEHOLDER}/mmlu-pro.csv'),
    ),
    CronAdapter(
        name='mt_bench',
        raw_policy=RawPolicy.NOT_CAPTURED,
        notes=(
            'Streams the judgment JSONL with requests directly and has no '
            'raw-dump flag. The source is a finished artifact, so the output '
            'fingerprint should hold steady and skip most days.'
        ),
    ),
    CronAdapter(
        name='multi_swe_bench',
        raw_policy=RawPolicy.UPSTREAM_VERSIONED,
        notes='Clones the upstream experiments repo; git holds the history.',
    ),
    CronAdapter(
        name='openeval',
        raw_policy=RawPolicy.UPSTREAM_VERSIONED,
        notes='Downloads from a HuggingFace dataset revision.',
    ),
    CronAdapter(
        name='rewardbench',
        raw_policy=RawPolicy.VIA_FETCH_HELPERS,
    ),
    CronAdapter(
        name='swe_bench_verified',
        raw_policy=RawPolicy.UPSTREAM_VERSIONED,
        notes='Reads the upstream experiments repo and a HF dataset.',
    ),
    CronAdapter(
        name='swe_polybench',
        raw_policy=RawPolicy.UPSTREAM_VERSIONED,
        notes='Reads the upstream experiments repo and a HF dataset.',
    ),
    CronAdapter(
        name='terminal_bench_2',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=(
            '--save-raw-html',
            f'{RAW_DIR_PLACEHOLDER}/terminal-bench-2.html',
        ),
    ),
    CronAdapter(
        name='vals_ai',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-json', f'{RAW_DIR_PLACEHOLDER}/vals-ai.json'),
        notes='Saves the normalized payload rather than the raw HTML.',
    ),
)


#: Adapter directories the cron must not run, and why.
EXCLUDED_ADAPTERS: dict[str, str] = {
    'arc_agi': 'Legacy: upstream source is no longer usable for a refresh.',
    'bfcl': 'Needs a hand-supplied --input-csv; no live fetch.',
    'cocoabench': 'Needs a hand-supplied --csv; no live fetch.',
    'livecodebenchpro': 'Legacy: upstream source is no longer usable.',
    'mercor_eval': 'Legacy: the API currently returns an empty response.',
    'sciarena': 'Needs a hand-supplied --input-json; no live fetch.',
    'vectara_hallucination_leaderboard': (
        'Pinned to SOURCE_COMMIT in code, so a refresh is a code change and a '
        'daily run could only re-emit identical records.'
    ),
}


_BY_NAME = {adapter.name: adapter for adapter in CRON_ADAPTERS}


def get_adapter(name: str) -> CronAdapter:
    """Return the registry entry for ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        excluded = EXCLUDED_ADAPTERS.get(name)
        if excluded:
            raise KeyError(
                f'{name!r} is excluded from the cron: {excluded}'
            ) from None
        known = ', '.join(sorted(_BY_NAME))
        raise KeyError(
            f'unknown cron adapter {name!r}; registered adapters: {known}'
        ) from None


def scheduled_adapters(
    environment: dict[str, str],
) -> tuple[list[CronAdapter], list[tuple[CronAdapter, str]]]:
    """Split the adapters into scheduled ones and quietly skipped ones.

    Every *enabled* adapter is scheduled, credentialed or not: an enabled
    adapter with a missing credential must fail its own job visibly, not
    vanish from the matrix behind a green run. Only adapters deliberately
    declared disabled are skipped quietly, with their reason.
    """
    del environment  # Credentials are checked (and failed) per run.
    runnable = [adapter for adapter in CRON_ADAPTERS if adapter.enabled]
    skipped = [
        (adapter, adapter.notes or 'disabled')
        for adapter in CRON_ADAPTERS
        if not adapter.enabled
    ]
    return runnable, skipped


def adapter_directories() -> set[str]:
    """Return the adapter package directories present in the checkout."""
    package_root = Path(__file__).resolve().parent.parent / 'adapters'
    return {
        entry.name
        for entry in package_root.iterdir()
        if entry.is_dir()
        and not entry.name.startswith('_')
        and (entry / 'adapter.py').is_file()
    }
