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


def test_raw_data_is_uploaded_even_when_the_refresh_fails(refresh_job: dict):
    upload = next(
        step
        for step in refresh_job['steps']
        if str(step.get('uses', '')).startswith('actions/upload-artifact')
    )
    assert upload['if'] == 'always()'
    paths = upload['with']['path']
    assert 'raw/' in paths
    assert 'adapter_reports/' in paths
    assert 'summary.json' in paths


def test_two_refreshes_never_run_at_once(workflow: dict):
    concurrency = workflow['concurrency']
    assert concurrency['group']
    # Cancelling mid-refresh could abandon a half-committed pull request.
    assert concurrency['cancel-in-progress'] is False


def test_the_workflow_asks_for_no_write_access_to_the_repository(
    workflow: dict,
):
    assert workflow['permissions'] == {'contents': 'read'}


def test_the_plan_job_feeds_the_matrix(workflow: dict):
    plan = workflow['jobs']['plan']
    assert 'adapters' in plan['outputs']
    assert workflow['jobs']['refresh']['needs'] == 'plan'


def test_a_no_op_refresh_does_not_fail_the_job(refresh_job: dict):
    # The runner exits 2 when the source had not moved. The default shell is
    # `bash -e`, which would abort the step before that code could be read, so
    # errexit has to be off around the call.
    step = next(
        step for step in refresh_job['steps'] if step.get('id') == 'refresh'
    )
    script = step['run']
    assert 'set +e' in script
    assert 'code=$?' in script


def test_a_partial_refresh_is_annotated(refresh_job: dict):
    scripts = ' '.join(step.get('run', '') for step in refresh_job['steps'])
    assert '::warning' in scripts
