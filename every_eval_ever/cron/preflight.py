"""Check a refresh's credentials before it does any work.

The cron fails closed: if raw payloads cannot be stored, nothing is published.
That is the right order, but it means a token without the access it needs turns
into every adapter failing at the last step. These checks run first, on every
scheduled run, so a credential problem is reported once and up front rather than
fifteen jobs deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from every_eval_ever.cron.archive import DEFAULT_RAW_REPO_ID
from every_eval_ever.cron.publish import DEFAULT_REPO_ID
from every_eval_ever.cron.schedule import CRON_ADAPTERS


@dataclass
class Check:
    """One thing that has to be true before a refresh can work."""

    name: str
    ok: bool
    detail: str
    #: False when the refresh can still run usefully without it.
    required: bool = True

    def render(self) -> str:
        mark = 'PASS' if self.ok else ('FAIL' if self.required else 'WARN')
        return f'{mark}  {self.name}: {self.detail}'


def check_token(api: HfApi) -> Check:
    """Confirm the Hugging Face token resolves to an identity that can write.

    A read-only token would sail through planning and fail every adapter at
    its publish step, so a token whose role is definitively read-only fails
    here. A fine-grained token whose scopes cannot be interpreted passes with
    its role reported — only the datastore commit itself can prove those.
    """
    try:
        identity = api.whoami()
    except Exception as error:
        return Check(
            'hugging face token',
            False,
            f'no usable token: {error}. Set HF_TOKEN.',
        )
    name = identity.get('name') or identity.get('fullname') or 'unknown'
    role = (identity.get('auth') or {}).get('accessToken', {}).get('role')
    if role == 'read':
        return Check(
            'hugging face token',
            False,
            f'authenticated as {name}, but the token is read-only; publishing '
            'and raw archiving both need write access',
        )
    return Check(
        'hugging face token',
        True,
        f'authenticated as {name}' + (f' (role: {role})' if role else ''),
    )


def check_datastore(api: HfApi, repo_id: str) -> Check:
    """Confirm the datastore is reachable.

    Reachable is not the same as writable — only a commit proves that — so this
    catches a wrong repo id or an unauthorised token, not a read-only one.
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type='dataset')
    except Exception as error:
        return Check('datastore', False, f'cannot read {repo_id}: {error}')
    return Check('datastore', True, f'{repo_id} is reachable')


def check_raw_dataset(
    api: HfApi, repo_id: str, *, create: bool = True
) -> Check:
    """Confirm the private raw dataset exists and is private.

    Creating it here rather than mid-refresh means a token that cannot create
    repositories is reported before any adapter has run. A dataset that exists
    but is *public* is a hard failure: raw source payloads are archived there
    on the promise of privacy, and visibility is a deliberate human decision —
    the cron reports it rather than flipping it. (The archive itself re-checks
    privacy before every commit, so this failing closed is defence in depth,
    not the only line.)
    """
    try:
        info = api.repo_info(repo_id=repo_id, repo_type='dataset')
    except RepositoryNotFoundError:
        info = None
    except Exception as error:
        # A transient error is not evidence the repo is missing; creating (or
        # blessing) anything on that basis could mask a public dataset.
        return Check(
            'raw dataset',
            False,
            f'could not read {repo_id}: {error}',
        )

    if info is not None:
        if not getattr(info, 'private', False):
            return Check(
                'raw dataset',
                False,
                f'{repo_id} exists but is PUBLIC; raw source data must stay '
                'private. Make it private (or point at another repo) — the '
                'cron will not change visibility itself, and archiving '
                'refuses to write to a public dataset.',
            )
        return Check('raw dataset', True, f'{repo_id} exists and is private')

    if not create:
        return Check('raw dataset', False, f'{repo_id} does not exist')

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type='dataset',
            private=True,
            exist_ok=True,
        )
        created = api.repo_info(repo_id=repo_id, repo_type='dataset')
    except Exception as error:
        return Check(
            'raw dataset',
            False,
            f'{repo_id} does not exist and could not be created: {error}. '
            'Create it by hand as a private dataset, or point '
            '--raw-repo-id at one that exists.',
        )
    if not getattr(created, 'private', False):
        return Check(
            'raw dataset',
            False,
            f'{repo_id} was created but reads back as PUBLIC; refusing to '
            'treat it as a raw-data destination.',
        )
    return Check('raw dataset', True, f'created {repo_id} as a private dataset')


def check_adapter_credentials(environment: dict[str, str]) -> list[Check]:
    """Report which adapters are held back by a missing credential.

    Asks each adapter directly (``missing_env``) instead of parsing the prose
    reason ``scheduled_adapters`` formats — rewording a message must never make
    a credential problem invisible.
    """
    runnable = []
    held_back: list[Check] = []
    for adapter in CRON_ADAPTERS:
        if not adapter.enabled:
            continue
        missing = adapter.missing_env(environment)
        if missing:
            held_back.append(
                Check(
                    f'credential for {adapter.name}',
                    False,
                    f'missing environment: {", ".join(missing)}',
                    required=False,
                )
            )
        else:
            runnable.append(adapter)
    return [
        Check(
            'scheduled adapters',
            bool(runnable),
            f'{len(runnable)} will run: '
            + ', '.join(adapter.name for adapter in runnable),
        ),
        *held_back,
    ]


def run_preflight(
    *,
    environment: dict[str, str],
    repo_id: str = DEFAULT_REPO_ID,
    raw_repo_id: str = DEFAULT_RAW_REPO_ID,
    create_raw: bool = True,
    api: HfApi | None = None,
) -> list[Check]:
    """Run every pre-refresh check and return the results in order."""
    api = api or HfApi(token=environment.get('HF_TOKEN'))
    token_check = check_token(api)
    checks = [token_check]
    if token_check.ok:
        checks.append(check_datastore(api, repo_id))
        checks.append(check_raw_dataset(api, raw_repo_id, create=create_raw))
    checks.extend(check_adapter_credentials(environment))
    return checks


def render(checks: list[Check]) -> str:
    """Render the checks as lines, in the order they ran."""
    return '\n'.join(check.render() for check in checks)


def failed(checks: list[Check]) -> list[Check]:
    """Return the checks that must pass and did not."""
    return [check for check in checks if check.required and not check.ok]


def write_markdown(checks: list[Check], path: str | Path) -> None:
    """Append a checklist to a file, for a workflow step summary."""
    lines = ['## Preflight', '']
    for check in checks:
        icon = '✅' if check.ok else ('❌' if check.required else '⚠️')
        lines.append(f'- {icon} **{check.name}** — {check.detail}')
    Path(path).open('a', encoding='utf-8').write('\n'.join(lines) + '\n')
