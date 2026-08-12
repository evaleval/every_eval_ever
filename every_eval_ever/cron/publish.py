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

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

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


def repo_paths(data_root: str | Path) -> list[str]:
    """Return the datastore paths a publish of ``data_root`` would add."""
    root = Path(data_root)
    return [
        f'data/{path.relative_to(root).as_posix()}'
        for path in collect_files(root)
    ]


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
    """Return the number of this adapter's open cron pull request, if any.

    Matched by title *and* author: the datastore is public, so anyone can open
    a pull request with any title, and committing onto a stranger's PR would
    hand them control of where the cron's records land. Only PRs opened by the
    account the cron authenticates as are candidates.
    """
    title = pr_title(adapter)
    author = api.whoami().get('name')
    for discussion in api.get_repo_discussions(
        repo_id=repo_id,
        repo_type='dataset',
        discussion_type='pull_request',
        discussion_status='open',
        author=author,
    ):
        if discussion.title.strip() == title:
            return discussion.num
    return None


def _pr_number_from_url(pr_url: str | None) -> int | None:
    if not pr_url:
        return None
    tail = pr_url.rstrip('/').rsplit('/', 1)[-1]
    return int(tail) if tail.isdigit() else None


def _paths_on_revision(
    api: HfApi, repo_id: str, revision: str, paths: list[str]
) -> list[str]:
    """Return which of ``paths`` exist at ``revision``, in one call.

    Deleting a path that does not exist fails the whole commit, so stale paths
    are filtered against the pull request's actual tree first.
    """
    try:
        found = api.get_paths_info(
            repo_id=repo_id,
            paths=paths,
            repo_type='dataset',
            revision=revision,
        )
    except Exception as error:
        raise PublishError(
            f'could not inspect {revision} of {repo_id} to reconcile an '
            f'incomplete attempt: {error}'
        ) from error
    return [entry.path for entry in found]


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
    stale_paths: list[str] | None = None,
    api: HfApi | None = None,
) -> PublishResult:
    """Commit every record under ``data_root`` to this adapter's pull request.

    ``stale_paths`` names datastore paths a previous *incomplete* attempt
    committed before failing; whichever of them exist on the adopted pull
    request are deleted in the first commit, so a retry replaces the broken
    attempt instead of stacking a second copy of the set beside it.
    """
    files = collect_files(data_root)
    if not files:
        raise PublishError(f'no record files found under {data_root}')

    root = Path(data_root)
    api = api or HfApi(token=token)
    existing = find_open_pr(api, repo_id, adapter)
    batches = _batched(files, max(1, files_per_commit))
    total = len(batches)

    removals: list[CommitOperationDelete] = []
    if stale_paths and existing is not None:
        removals = [
            CommitOperationDelete(path_in_repo=path)
            for path in _paths_on_revision(
                api, repo_id, f'refs/pr/{existing}', stale_paths
            )
        ]
        if removals:
            _logger.info(
                'removing %d file(s) left by an incomplete attempt from '
                'PR %d before republishing',
                len(removals),
                existing,
            )

    pr_number = existing
    pr_url = None
    for index, batch in enumerate(batches, start=1):
        operations: list[CommitOperationAdd | CommitOperationDelete] = [
            CommitOperationAdd(
                path_in_repo=f'data/{path.relative_to(root).as_posix()}',
                path_or_fileobj=str(path),
            )
            for path in batch
        ]
        if index == 1 and removals:
            operations = [*removals, *operations]
        suffix = f' ({index}/{total})' if total > 1 else ''
        if pr_number is None:
            # The Hub titles a created PR from this commit's message, and
            # find_open_pr matches that title exactly tomorrow — so the
            # PR-creating commit must carry the bare title, never the batch
            # suffix. The suffix still marks the follow-up commits, which are
            # ordinary commits on the PR ref.
            description = commit_description
            if total > 1:
                description = (
                    f'Batch 1/{total}; the set is incomplete until the last '
                    f'batch lands.\n\n{commit_description}'
                )
            info = api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=operations,
                commit_message=pr_title(adapter),
                commit_description=description,
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
