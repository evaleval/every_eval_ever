"""The adapter catalog must keep describing the adapters it points at.

An earlier scheduled-ingestion attempt discovered adapter arguments by
regex-scanning adapter source, and mis-read the adapters whose CLI conventions
differed. The catalog replaces that guess with a declaration, so these tests
check the declaration against the real parsers rather than trusting it.
"""

from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path

import pytest

from every_eval_ever.adapters import catalog

ADAPTERS_DIR = Path(catalog.__file__).resolve().parent
RUNNABLE = catalog.runnable_adapters()
# 2026-08-10 is a Monday, so an offset in days is also the weekday index.
_MONDAY = date(2026, 8, 10)


def adapter_packages() -> set[str]:
    """Return every adapter package directory that ships an adapter module."""
    return {
        path.parent.name
        for path in ADAPTERS_DIR.glob('*/adapter.py')
        if path.parent.name != '__pycache__'
    }


def test_every_adapter_package_is_accounted_for() -> None:
    """A new adapter must be registered or explicitly marked legacy.

    Without this, adding an adapter silently leaves it out of automation and
    nothing says so.
    """
    unaccounted = sorted(adapter_packages() - catalog.registered_packages())

    assert not unaccounted, (
        'these adapter packages are neither registered in '
        'every_eval_ever/adapters/catalog.py nor listed in '
        f'catalog.LEGACY_ADAPTERS: {", ".join(unaccounted)}'
    )


def test_registry_does_not_name_missing_adapters() -> None:
    packages = adapter_packages()
    missing = sorted(catalog.registered_packages() - packages)

    assert not missing, (
        'the catalog names adapter packages that do not exist: '
        f'{", ".join(missing)}'
    )


def test_legacy_adapters_are_not_also_registered() -> None:
    registered = {spec.package for spec in catalog.ADAPTERS}
    overlap = sorted(registered & catalog.LEGACY_ADAPTERS)

    assert not overlap, (
        f'adapters cannot be both registered and legacy: {", ".join(overlap)}'
    )


@pytest.mark.parametrize('spec', catalog.ADAPTERS, ids=lambda spec: spec.key)
def test_registered_module_is_an_in_tree_adapter(
    spec: catalog.AdapterSpec,
) -> None:
    assert spec.module.startswith(catalog.ADAPTER_MODULE_PREFIX)
    assert spec.module.endswith('.adapter')


@pytest.mark.parametrize('spec', catalog.ADAPTERS, ids=lambda spec: spec.key)
def test_registered_module_imports(spec: catalog.AdapterSpec) -> None:
    importlib.import_module(spec.module)


@pytest.mark.parametrize('spec', RUNNABLE, ids=lambda spec: spec.key)
def test_runnable_adapter_accepts_its_registered_arguments(
    spec: catalog.AdapterSpec, tmp_path: Path
) -> None:
    """The recorded argv must parse against the adapter's own parser.

    This is the check that keeps a catalog entry honest when someone renames
    a flag or changes a ``choices`` list.
    """
    module = importlib.import_module(spec.module)
    parse_args = getattr(module, 'parse_args', None)
    assert callable(parse_args), (
        f'{spec.key}: {spec.module} must expose '
        'parse_args(argv) so automation can be checked against the real '
        'parser'
    )

    data_root = tmp_path / 'data'
    args = parse_args(spec.argv(data_root))

    assert Path(args.output_dir) == spec.output_dir(data_root)


@pytest.mark.parametrize('spec', RUNNABLE, ids=lambda spec: spec.key)
def test_runnable_adapter_output_dir_stays_inside_the_staging_tree(
    spec: catalog.AdapterSpec, tmp_path: Path
) -> None:
    """Automation must never be able to write into the checkout."""
    data_root = tmp_path / 'data'
    output_dir = spec.output_dir(data_root)

    assert output_dir.is_relative_to(data_root)
    if spec.output_scope == 'collection':
        assert output_dir.name == spec.collections[0]


def test_collections_are_unique_across_adapters() -> None:
    """Two adapters writing one collection would collide in the datastore."""
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for spec in catalog.ADAPTERS:
        for collection in spec.collections:
            owner = seen.setdefault(collection, spec.key)
            if owner != spec.key:
                collisions.append(f'{collection}: {owner} and {spec.key}')

    assert not collisions, (
        f'collections claimed by more than one adapter: {collisions}'
    )


def test_get_reports_the_available_keys() -> None:
    with pytest.raises(catalog.UnknownAdapterError) as excinfo:
        catalog.get('not-an-adapter')

    assert 'hle' in str(excinfo.value)


def test_daily_adapters_run_every_day() -> None:
    daily = [spec for spec in RUNNABLE if spec.cadence == 'daily']
    assert daily

    for day in range(7):
        due = catalog.scheduled_for(_MONDAY + _timedelta(day))
        for spec in daily:
            assert spec in due


def test_weekly_adapters_run_on_exactly_one_weekday() -> None:
    weekly = [spec for spec in RUNNABLE if spec.cadence == 'weekly']
    assert weekly

    for spec in weekly:
        scheduled_days = [
            day for day in range(7) if spec.runs_on(_MONDAY + _timedelta(day))
        ]
        assert scheduled_days == [spec.weekday]


def test_unrunnable_adapters_are_never_scheduled() -> None:
    unrunnable = [spec for spec in catalog.ADAPTERS if not spec.runnable]
    assert unrunnable

    for day in range(7):
        due = catalog.scheduled_for(_MONDAY + _timedelta(day))
        for spec in unrunnable:
            assert spec not in due
            assert spec.unrunnable_reason


def test_scheduled_for_filters_on_available_credentials() -> None:
    credentialed = [spec for spec in RUNNABLE if spec.required_env]
    assert credentialed, 'expected at least one adapter to need a credential'
    spec = credentialed[0]
    run_date = _first_scheduled_date(spec)

    assert spec in catalog.scheduled_for(run_date)
    assert spec not in catalog.scheduled_for(run_date, available_env=set())
    assert spec in catalog.scheduled_for(
        run_date, available_env=set(spec.required_env)
    )


def test_a_partial_credential_set_still_excludes_the_adapter() -> None:
    """Holding one of two required keys is not enough to schedule a run."""
    credentialed = [spec for spec in RUNNABLE if spec.required_env]
    every_key = {name for spec in credentialed for name in spec.required_env}
    assert len(every_key) >= 2, 'expected distinct credentials to compare'

    spec = credentialed[0]
    run_date = _first_scheduled_date(spec)
    partial = every_key - set(spec.required_env)

    assert spec not in catalog.scheduled_for(run_date, available_env=partial)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'key': 'bad key'}, 'safe slug'),
        ({'module': 'not a module'}, 'dotted module path'),
        ({'collections': ()}, 'at least one collection'),
        ({'collections': ('..',)}, 'safe datastore path component'),
        ({'collections': ('a', 'b')}, 'exactly one collection'),
        ({'runnable': False}, 'unrunnable_reason'),
        ({'cadence': 'weekly'}, 'requires a weekday'),
        ({'weekday': 2}, 'must not set weekday'),
        ({'timeout_minutes': 0}, 'must be positive'),
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict, message: str) -> None:
    base = {
        'key': 'demo',
        'module': 'every_eval_ever.adapters.hle.adapter',
        'collections': ('hle',),
    }

    with pytest.raises(ValueError, match=message):
        catalog.AdapterSpec(**(base | kwargs))


def test_every_job_gets_more_time_than_the_adapter_it_runs() -> None:
    """The subprocess budget is not the job budget.

    A job that is cancelled after its adapter finished but before its records
    are recorded leaves the datastore ahead of the ledger, so the surrounding
    steps get their own room.
    """
    assert catalog.JOB_TIMEOUT_BUFFER_MINUTES > 0
    for spec in catalog.ADAPTERS:
        assert spec.job_timeout_minutes > spec.timeout_minutes


def _timedelta(days: int):
    from datetime import timedelta

    return timedelta(days=days)


def _first_scheduled_date(spec: catalog.AdapterSpec) -> date:
    for day in range(7):
        run_date = _MONDAY + _timedelta(day)
        if spec.runs_on(run_date):
            return run_date
    raise AssertionError(f'{spec.key} is never scheduled')
