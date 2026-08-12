"""The cron schedule must stay in step with the adapters that exist."""

from __future__ import annotations

from pathlib import Path

import pytest

from every_eval_ever.cron.schedule import (
    CRON_ADAPTERS,
    EXCLUDED_ADAPTERS,
    RAW_DIR_PLACEHOLDER,
    CronAdapter,
    RawPolicy,
    adapter_directories,
    get_adapter,
    scheduled_adapters,
)

ADAPTERS_ROOT = (
    Path(__file__).resolve().parent.parent / 'every_eval_ever' / 'adapters'
)


def test_every_adapter_is_either_scheduled_or_excluded_with_a_reason():
    accounted = {adapter.name for adapter in CRON_ADAPTERS} | set(
        EXCLUDED_ADAPTERS
    )
    unaccounted = adapter_directories() - accounted
    assert not unaccounted, (
        'these adapters are neither scheduled nor excluded: '
        f'{sorted(unaccounted)}. Add them to CRON_ADAPTERS, or to '
        'EXCLUDED_ADAPTERS with the reason a daily refresh cannot run them.'
    )


def test_schedule_and_exclusions_do_not_overlap():
    scheduled = {adapter.name for adapter in CRON_ADAPTERS}
    assert not scheduled & set(EXCLUDED_ADAPTERS)


def test_scheduled_and_excluded_names_exist_as_adapters():
    known = adapter_directories()
    named = {adapter.name for adapter in CRON_ADAPTERS} | set(EXCLUDED_ADAPTERS)
    assert not named - known, f'no such adapter: {sorted(named - known)}'


def test_every_exclusion_states_a_reason():
    for name, reason in EXCLUDED_ADAPTERS.items():
        assert reason.strip(), f'{name} is excluded without a reason'


def test_adapter_names_are_unique():
    names = [adapter.name for adapter in CRON_ADAPTERS]
    assert len(names) == len(set(names))


def test_every_adapter_module_resolves_to_a_file():
    for adapter in CRON_ADAPTERS:
        module_path = ADAPTERS_ROOT / adapter.name / 'adapter.py'
        assert module_path.is_file(), f'{adapter.module} does not exist'


def test_flag_archived_adapters_declare_raw_arguments():
    for adapter in CRON_ADAPTERS:
        declares_flag = adapter.raw_policy is RawPolicy.VIA_ADAPTER_FLAG
        assert declares_flag == bool(adapter.raw_args), (
            f'{adapter.name}: raw policy {adapter.raw_policy.value} and '
            f'raw_args {adapter.raw_args} disagree'
        )
        for argument in adapter.raw_args:
            if argument.startswith('--'):
                continue
            assert RAW_DIR_PLACEHOLDER in argument, (
                f'{adapter.name} writes raw data to a fixed path {argument!r}; '
                f'use {RAW_DIR_PLACEHOLDER} so it lands in the run directory'
            )


def test_every_adapter_has_at_least_one_invocation():
    for adapter in CRON_ADAPTERS:
        assert adapter.runs, f'{adapter.name} declares no invocation'


def test_disabled_adapters_explain_themselves():
    for adapter in CRON_ADAPTERS:
        if not adapter.enabled:
            assert adapter.notes.strip(), (
                f'{adapter.name} is disabled without saying why'
            )


def test_argv_substitutes_the_raw_directory():
    adapter = CronAdapter(
        name='vals_ai',
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-json', f'{RAW_DIR_PLACEHOLDER}/vals-ai.json'),
    )
    argv = adapter.argv_for(('--benchmark', 'finance'), '/tmp/run/raw')
    assert argv == [
        '-m',
        'every_eval_ever.adapters.vals_ai.adapter',
        '--benchmark',
        'finance',
        '--save-raw-json',
        '/tmp/run/raw/vals-ai.json',
    ]


def test_unknown_adapter_names_the_registered_ones():
    with pytest.raises(KeyError, match='unknown cron adapter'):
        get_adapter('not_an_adapter')


def test_excluded_adapter_reports_why_it_is_excluded():
    name = next(iter(EXCLUDED_ADAPTERS))
    with pytest.raises(KeyError, match='excluded from the cron'):
        get_adapter(name)


def test_uncredentialed_enabled_adapters_stay_scheduled():
    # They must fail their own job visibly, not vanish from the matrix behind
    # a green run — so scheduling ignores credentials entirely.
    runnable, _ = scheduled_adapters({})
    needs_credentials = {
        adapter.name
        for adapter in CRON_ADAPTERS
        if adapter.requires_env and adapter.enabled
    }
    assert needs_credentials
    assert needs_credentials <= {adapter.name for adapter in runnable}


def test_blank_credentials_count_as_missing():
    adapter = next(
        item for item in CRON_ADAPTERS if item.requires_env and item.enabled
    )
    assert adapter.missing_env({name: '   ' for name in adapter.requires_env})


def test_disabled_adapters_are_never_scheduled():
    disabled = {
        adapter.name for adapter in CRON_ADAPTERS if not adapter.enabled
    }
    runnable, skipped = scheduled_adapters(
        {
            name: 'token'
            for adapter in CRON_ADAPTERS
            for name in adapter.requires_env
        }
    )
    assert not disabled & {adapter.name for adapter in runnable}
    assert disabled <= {adapter.name for adapter, _ in skipped}
