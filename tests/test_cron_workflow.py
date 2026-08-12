"""The daily workflow must isolate adapters from each other.

These assertions are the reason the cron can be trusted to keep going: one
source failing, hanging, or producing nothing must not cost the others their
refresh. They are checked here because a YAML regression is silent otherwise —
it only shows up as a day of missing data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent
    / '.github'
    / 'workflows'
    / 'adapter_cron.yml'
)

#: GitHub cancels any job still running after six hours.
JOB_CEILING_MINUTES = 360


@pytest.fixture(scope='module')
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def refresh_job(workflow: dict) -> dict:
    return workflow['jobs']['refresh']


def test_the_workflow_runs_daily(workflow: dict):
    # PyYAML reads a bare `on:` key as the boolean True.
    triggers = workflow[True]
    schedules = triggers['schedule']
    assert len(schedules) == 1
    minute, hour, day, month, weekday = schedules[0]['cron'].split()
    assert (day, month, weekday) == ('*', '*', '*'), 'not a daily schedule'
    assert minute.isdigit() and hour.isdigit()


def test_the_workflow_can_be_run_by_hand_for_one_adapter(workflow: dict):
    inputs = workflow[True]['workflow_dispatch']['inputs']
    assert 'adapter' in inputs
    assert 'dry_run' in inputs


def test_one_job_per_adapter(refresh_job: dict):
    matrix = refresh_job['strategy']['matrix']
    assert list(matrix) == ['adapter']
    assert 'needs.plan.outputs.adapters' in matrix['adapter']


def test_a_failing_adapter_does_not_cancel_the_others(refresh_job: dict):
    # Without this, the first failing source aborts every adapter still queued.
    assert refresh_job['strategy']['fail-fast'] is False


def test_every_adapter_job_has_its_own_bounded_timeout(refresh_job: dict):
    timeout = refresh_job['timeout-minutes']
    assert 0 < timeout < JOB_CEILING_MINUTES, (
        'a per-job timeout below the six-hour ceiling is what stops a hung '
        'adapter from burning a runner and colliding with the next run'
    )


def test_the_daily_run_is_not_serialised_behind_one_adapter(refresh_job: dict):
    # max-parallel may be absent (full parallelism); it must not be 1, which
    # would queue every adapter behind the slowest one before it.
    assert refresh_job['strategy'].get('max-parallel', 2) > 1


def test_the_summary_is_uploaded_even_when_the_refresh_fails(
    refresh_job: dict,
):
    upload = next(
        step
        for step in refresh_job['steps']
        if str(step.get('uses', '')).startswith('actions/upload-artifact')
    )
    assert upload['if'] == 'always()'
    assert 'summary.json' in upload['with']['path']


def test_no_raw_data_reaches_a_public_artifact(refresh_job: dict):
    # Artifacts on a public repository are downloadable by anyone signed in.
    # Raw bodies AND adapter failure reports (which embed raw source rows)
    # belong solely in the private raw dataset.
    for step in refresh_job['steps']:
        if str(step.get('uses', '')).startswith('actions/upload-artifact'):
            paths = step['with']['path']
            assert '/raw' not in paths
            assert 'adapter_reports' not in paths


def test_two_publishes_of_one_adapter_never_run_at_once(refresh_job: dict):
    # Scoped to the job and keyed by adapter: a workflow-wide group would
    # serialize unrelated adapters AND silently replace an older pending run
    # (GitHub keeps one pending item per group).
    assert 'concurrency' not in refresh_job.get('strategy', {})
    concurrency = refresh_job['concurrency']
    assert 'matrix.adapter' in concurrency['group']
    assert 'concurrency' not in refresh_job.get('workflow', {})
    # Cancelling mid-refresh could abandon a half-committed pull request.
    assert concurrency['cancel-in-progress'] is False


def test_dry_runs_do_not_queue_behind_publishes(refresh_job: dict):
    assert 'dry_run' in refresh_job['concurrency']['group']


def test_no_workflow_wide_concurrency_group(workflow: dict):
    assert 'concurrency' not in workflow


def test_the_workflow_asks_for_no_write_access_to_the_repository(
    workflow: dict,
):
    assert workflow['permissions'] == {'contents': 'read'}


def test_the_plan_job_feeds_the_matrix(workflow: dict):
    plan = workflow['jobs']['plan']
    assert 'adapters' in plan['outputs']
    assert workflow['jobs']['refresh']['needs'] == 'plan'


def _refresh_step(refresh_job: dict) -> dict:
    return next(
        step
        for step in refresh_job['steps']
        if 'every_eval_ever.cron run' in step.get('run', '')
    )


def test_a_no_op_refresh_does_not_fail_the_job(refresh_job: dict):
    # The runner exits 3 when the source had not moved. The default shell is
    # `bash -e`, which would abort the step before that code could be read, so
    # errexit has to be off around the call.
    script = _refresh_step(refresh_job)['run']
    assert 'set +e' in script
    assert 'code=$?' in script
    assert '-ne 3' in script


def test_a_usage_error_fails_the_job(refresh_job: dict):
    # Argparse exits 2 on a bad flag. If 2 were the nothing-new code, a flag
    # typo would silently disable the whole cron while every job stayed green.
    from every_eval_ever.cron import runner

    assert runner.EXIT_NOTHING_NEW != 2
    script = _refresh_step(refresh_job)['run']
    assert str(runner.EXIT_NOTHING_NEW) in script
    assert '-ne 2' not in script


def test_no_secrets_at_workflow_scope(workflow: dict):
    # Workflow-level env hands every secret to checkout, setup, install and
    # artifact steps that have no business seeing them.
    for name, value in workflow.get('env', {}).items():
        assert 'secrets.' not in str(value), (
            f'workflow-level env {name} references a secret; scope it to the '
            'command steps that use it'
        )


def test_every_declared_adapter_credential_reaches_the_command_steps(
    workflow: dict,
):
    # requires_env in the schedule and the step env blocks drift silently: a
    # secret named in one but not the other means an adapter is planned and
    # then skipped forever as 'missing environment'.
    from every_eval_ever.cron.schedule import CRON_ADAPTERS

    plan_steps = workflow['jobs']['plan']['steps']
    preflight_env = next(
        step for step in plan_steps if 'cron preflight' in step.get('run', '')
    )['env']
    refresh_env = _refresh_step(workflow['jobs']['refresh'])['env']
    for adapter in CRON_ADAPTERS:
        for name in adapter.requires_env:
            assert name in preflight_env, (
                f'{adapter.name} requires {name}; preflight must see it to '
                'report credential coverage'
            )
            assert name in refresh_env, (
                f'{adapter.name} requires {name}; the refresh step must '
                'receive it (scoped to matrix.adapter)'
            )
            # Scoped: only the matching matrix job receives the key.
            assert 'matrix.adapter ==' in refresh_env[name]


def test_secrets_are_scoped_away_from_non_command_steps(workflow: dict):
    for job_name, job in workflow['jobs'].items():
        for step in job['steps']:
            runs_cron = 'every_eval_ever' in step.get('run', '')
            if runs_cron:
                continue
            for name, value in step.get('env', {}).items():
                assert 'secrets.' not in str(value), (
                    f'{job_name} step {step.get("name", step.get("uses"))} '
                    f'receives secret env {name} but runs no cron command'
                )


def test_a_partial_refresh_is_annotated(refresh_job: dict):
    scripts = ' '.join(step.get('run', '') for step in refresh_job['steps'])
    assert '::warning' in scripts


def test_credentials_are_checked_before_any_adapter_runs(workflow: dict):
    # The refresh fails closed, so a token that cannot store raw data must be
    # reported once up front rather than by every adapter failing at the end.
    steps = workflow['jobs']['plan']['steps']
    names = [step.get('name', '') for step in steps]
    preflight = next(
        index
        for index, step in enumerate(steps)
        if 'preflight' in step.get('run', '')
    )
    plan = next(
        index for index, step in enumerate(steps) if step.get('id') == 'plan'
    )
    assert preflight < plan, f'preflight must run before planning: {names}'
    # The preflight command step itself carries the token, scoped to it.
    assert 'HF_TOKEN' in steps[preflight]['env']
