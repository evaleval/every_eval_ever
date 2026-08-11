"""Send a refresh to the datastore as one pull request per adapter.

Each adapter owns one open pull request, identified by its title. A refresh
commits onto that request when it is open and opens it when it is not, because
opening a fresh request per round is the largest documented source of review
churn in this datastore's history — see the conversion skill's
``reference/datastore-submission.md``.

Commits are batched: a single commit carrying thousands of files can time out
server-side while the client sees an error, leaving a half-submitted request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

DEFAULT_REPO_ID = 'evaleval/EEE_datastore'
DEFAULT_FILES_PER_COMMIT = 300

#: Marks the pull request a given adapter's cron output belongs to.
PR_TITLE_TEMPLATE = '[Submission] Daily cron refresh: {adapter}'

_logger = logging.getLogger(__name__)


class PublishError(Exception):
    """Raised when a refresh cannot be published."""


@dataclass
class PublishPlan:
    """The files a refresh would send, and where."""

    repo_id: str
    adapter: str
    files: list[Path] = field(default_factory=list)
    existing_pr: int | None = None

    @property
    def title(self) -> str:
        return PR_TITLE_TEMPLATE.format(adapter=self.adapter)


@dataclass
class PublishResult:
    """What publishing did."""

    pr_url: str | None
    pr_number: int | None
    files: int
    commits: int
    reused_existing_pr: bool


def pr_title(adapter: str) -> str:
    """Return the pull request title that identifies an adapter's cron output."""
    return PR_TITLE_TEMPLATE.format(adapter=adapter)


def collect_files(data_root: str | Path) -> list[Path]:
    """Return every record file to upload, aggregates and sample companions."""
    root = Path(data_root)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob('*/*/*/*')
        if path.is_file() and path.suffix in {'.json', '.jsonl'}
    )


def find_open_pr(
    api: HfApi,
    repo_id: str,
    adapter: str,
) -> int | None:
    """Return the number of this adapter's open cron pull request, if any."""
    title = pr_title(adapter)
    for discussion in api.get_repo_discussions(
        repo_id=repo_id,
        repo_type='dataset',
        discussion_type='pull_request',
        discussion_status='open',
    ):
        if discussion.title.strip() == title:
            return discussion.num
    return None


def _pr_number_from_url(pr_url: str | None) -> int | None:
    if not pr_url:
        return None
    tail = pr_url.rstrip('/').rsplit('/', 1)[-1]
    return int(tail) if tail.isdigit() else None


def _batched(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def publish(
    data_root: str | Path,
    *,
    adapter: str,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    commit_description: str = '',
    files_per_commit: int = DEFAULT_FILES_PER_COMMIT,
    api: HfApi | None = None,
) -> PublishResult:
    """Commit every record under ``data_root`` to this adapter's pull request."""
    files = collect_files(data_root)
    if not files:
        raise PublishError(f'no record files found under {data_root}')

    root = Path(data_root)
    api = api or HfApi(token=token)
    existing = find_open_pr(api, repo_id, adapter)
    batches = _batched(files, max(1, files_per_commit))
    total = len(batches)

    pr_number = existing
    pr_url = None
    for index, batch in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(
                path_in_repo=f'data/{path.relative_to(root).as_posix()}',
                path_or_fileobj=str(path),
            )
            for path in batch
        ]
        suffix = f' ({index}/{total})' if total > 1 else ''
        if pr_number is None:
            info = api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=operations,
                commit_message=f'{pr_title(adapter)}{suffix}',
                commit_description=commit_description,
                create_pr=True,
            )
            pr_url = info.pr_url
            pr_number = _pr_number_from_url(pr_url)
            if pr_number is None:
                raise PublishError(
                    'Hugging Face did not return a pull request URL for the '
                    f'first commit; {len(batches) - 1} batches were not sent'
                )
        else:
            info = api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=operations,
                commit_message=f'{pr_title(adapter)}{suffix}',
                commit_description=commit_description,
                revision=f'refs/pr/{pr_number}',
            )
            pr_url = pr_url or getattr(info, 'pr_url', None)
        _logger.info('committed batch %d/%d to PR %s', index, total, pr_number)

    if pr_url is None and pr_number is not None:
        pr_url = (
            f'https://huggingface.co/datasets/{repo_id}/discussions/{pr_number}'
        )

    return PublishResult(
        pr_url=pr_url,
        pr_number=pr_number,
        files=len(files),
        commits=total,
        reused_existing_pr=existing is not None,
    )


def plan(
    data_root: str | Path,
    *,
    adapter: str,
    repo_id: str = DEFAULT_REPO_ID,
    api: HfApi | None = None,
) -> PublishPlan:
    """Describe what :func:`publish` would send, without sending it."""
    existing = None
    if api is not None:
        existing = find_open_pr(api, repo_id, adapter)
    return PublishPlan(
        repo_id=repo_id,
        adapter=adapter,
        files=collect_files(data_root),
        existing_pr=existing,
    )
