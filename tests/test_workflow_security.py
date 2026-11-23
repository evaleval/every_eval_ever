"""Security invariants for GitHub Actions workflows."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

from every_eval_ever.adapters import catalog

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / '.github' / 'workflows'
PACKAGE_MANAGER = re.compile(
    r'(^|[;&|()\s])(uv|pip|pipx|npm|npx|pnpm|yarn)([;&|()\s]|$)'
)


def _workflow(path: str) -> dict:
    return yaml.safe_load((WORKFLOW_DIR / path).read_text(encoding='utf-8'))


def _requirement_name(requirement: str) -> str:
    return re.split(r'[<>=!~;\[]', requirement, maxsplit=1)[0].strip()


def _shell_code(run: str) -> str:
    return '\n'.join(
        line for line in run.splitlines() if not line.lstrip().startswith('#')
    )


def test_secret_bearing_run_steps_do_not_invoke_package_managers() -> None:
    """Dependency downloads/build hooks must finish before secrets exist."""
    failures = []
    for path in sorted(WORKFLOW_DIR.glob('*.yml')):
        workflow = yaml.safe_load(path.read_text(encoding='utf-8'))
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                env = json.dumps(step.get('env', {}), sort_keys=True)
                run = step.get('run', '') or ''
                if 'secrets.' not in env or not run:
                    continue
                if PACKAGE_MANAGER.search(_shell_code(run)):
                    failures.append(
                        f'{path.name}:{job_name}:{step.get("name", "unnamed")}'
                    )
    assert not failures, (
        'package manager invoked while GitHub secrets are in the step '
        'environment:\n' + '\n'.join(failures)
    )


def test_cron_optional_dependencies_are_locked_before_ingest() -> None:
    """Every scheduled adapter dependency is in the locked cron group."""
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    cron_requirements = pyproject['dependency-groups']['cron']
    cron_packages = {_requirement_name(item) for item in cron_requirements}
    required = {
        package
        for spec in catalog.ADAPTERS
        if spec.runnable
        for package in spec.with_packages
    }
    assert required <= cron_packages

    uv_lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    project = next(
        package
        for package in uv_lock['package']
        if package['name'] == 'every-eval-ever'
    )
    locked = {
        item['name'] for item in project['dev-dependencies']['cron']
    }
    assert cron_packages <= locked


def test_adapter_ingest_uses_prepared_interpreter_directly() -> None:
    """The credential-bearing ingest step must not ask uv to prepare code."""
    workflow = _workflow('adapter_cron.yml')
    run_steps = workflow['jobs']['run']['steps']
    install = next(step for step in run_steps if step.get('name') == 'Install dependencies')
    ingest = next(step for step in run_steps if step.get('name') == 'Ingest')

    assert 'uv sync --locked --group cron' in install['run']
    assert 'secrets.' not in json.dumps(install.get('env', {}))
    assert 'secrets.' in json.dumps(ingest['env'])
    assert '"${UV_PROJECT_ENVIRONMENT}/bin/python"' in ingest['run']
    assert PACKAGE_MANAGER.search(_shell_code(ingest['run'])) is None
