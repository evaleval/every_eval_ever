"""Send one adapter's records to that adapter's own datastore pull request.

The rule the ticket asks for is one pull request per adapter, reused across
runs. Getting that wrong is expensive in both directions: opening a fresh
pull request every day buries reviewers, and guessing at which existing one
to reuse can push a scrape into somebody else's submission. So the pull
request is remembered by number, re-checked before use, and identified by a
marker the cron itself wrote — never by "the newest open one".

An ambiguous match is an error. There is no safe guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from huggingface_hub import CommitOperationAdd

DEFAULT_DATASTORE_REPO = 'evaleval/EEE_datastore'
#: Machine-readable line in the pull request body identifying its adapter.
MARKER_PREFIX = 'eee-cron-adapter:'
#: A single commit of thousands of files can 504 with the commit still
#: landing server-side, so batches are kept small.
DEFAULT_BATCH_SIZE = 300


class SubmissionError(RuntimeError):
    """Raised when a submission cannot proceed safely."""


class AmbiguousPullRequestError(SubmissionError):
    """Raised when more than one open pull request claims one adapter."""


def marker(adapter: str) -> str:
    return f'{MARKER_PREFIX} {adapter}'


def pull_request_title(adapter: str) -> str:
    return f'[Submission] cron: {adapter} — automated ingestion'


def _title_matches(title: str, adapter: str) -> bool:
    return bool(re.search(rf'\bcron:\s*{re.escape(adapter)}\b', title or ''))


@dataclass(frozen=True)
class PullRequest:
    """The datastore pull request one adapter publishes into."""

    number: int
    url: str
    revision: str
    title: str


def _discussion_number(discussion: Any) -> int | None:
    value = getattr(discussion, 'num', None)
    if value is None:
        url = getattr(discussion, 'url', '') or ''
        if '/discussions/' in url:
            value = url.rsplit('/discussions/', 1)[-1].strip('/')
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _discussion_url(repo_id: str, discussion: Any, number: int) -> str:
    url = getattr(discussion, 'url', None)
    if isinstance(url, str) and url:
        return url
    return f'https://huggingface.co/datasets/{repo_id}/discussions/{number}'


def _discussion_revision(discussion: Any, number: int) -> str:
    revision = getattr(discussion, 'git_reference', None)
    if isinstance(revision, str) and revision:
        return revision
    return f'refs/pr/{number}'


def _is_open_pull_request(discussion: Any) -> bool:
    return (
        getattr(discussion, 'is_pull_request', False)
        and getattr(discussion, 'status', None) == 'open'
    )


class DatastoreSubmitter:
    """Find or open one adapter's pull request, and upload into it."""

    def __init__(
        self,
        api: Any,
        *,
        repo_id: str = DEFAULT_DATASTORE_REPO,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not repo_id:
            raise SubmissionError('a datastore repository id is required')
        self.api = api
        self.repo_id = repo_id
        self.batch_size = batch_size

    def _open_pull_requests(self) -> list[Any]:
        try:
            return list(
                self.api.get_repo_discussions(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    discussion_type='pull_request',
                    discussion_status='open',
                )
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not list open pull requests on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

    def resolve_known(self, adapter: str, number: int) -> PullRequest | None:
        """Return the remembered pull request if it is still usable.

        Usable means: still open, still a pull request, and still carrying
        this adapter's marker. A merged, closed, or re-titled discussion is
        treated as gone, so the next run opens a fresh one rather than
        pushing into something a reviewer has finished with.
        """
        for discussion in self._open_pull_requests():
            if _discussion_number(discussion) != number:
                continue
            if not _is_open_pull_request(discussion):
                return None
            title = getattr(discussion, 'title', '') or ''
            if not _title_matches(title, adapter):
                return None
            return PullRequest(
                number=number,
                url=_discussion_url(self.repo_id, discussion, number),
                revision=_discussion_revision(discussion, number),
                title=title,
            )
        return None

    def find_by_marker(self, adapter: str) -> PullRequest | None:
        """Find this adapter's pull request when no number is remembered.

        Matching is on the exact ``cron: <adapter>`` marker the cron writes.
        Two matches is an error rather than a choice: picking one would mean
        appending a scrape to a pull request nobody expected it in.
        """
        matches = []
        for discussion in self._open_pull_requests():
            if not _is_open_pull_request(discussion):
                continue
            title = getattr(discussion, 'title', '') or ''
            if not _title_matches(title, adapter):
                continue
            number = _discussion_number(discussion)
            if number is None:
                continue
            matches.append(
                PullRequest(
                    number=number,
                    url=_discussion_url(self.repo_id, discussion, number),
                    revision=_discussion_revision(discussion, number),
                    title=title,
                )
            )
        if not matches:
            return None
        if len(matches) > 1:
            raise AmbiguousPullRequestError(
                f'{len(matches)} open pull requests on {self.repo_id} claim '
                f'adapter {adapter!r}: '
                f'{", ".join(str(match.number) for match in matches)}. '
                'Close the duplicates, or set the intended number in '
                f'state/{adapter}.json, and re-run.'
            )
        return matches[0]

    def open_pull_request(
        self,
        adapter: str,
        *,
        operations: Sequence[CommitOperationAdd],
        description: str,
    ) -> PullRequest:
        """Open this adapter's pull request with its first batch of records."""
        try:
            commit = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type='dataset',
                operations=list(operations),
                commit_message=pull_request_title(adapter),
                commit_description=description,
                create_pr=True,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not open a pull request on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

        url = getattr(commit, 'pr_url', None)
        number = getattr(commit, 'pr_num', None)
        if number is None and isinstance(url, str) and '/discussions/' in url:
            try:
                number = int(url.rsplit('/discussions/', 1)[-1].strip('/'))
            except ValueError:
                number = None
        if number is None:
            raise SubmissionError(
                'the Hub accepted the commit but did not report a pull '
                'request number; check the repository before re-running so '
                'a second pull request is not opened'
            )
        revision = getattr(commit, 'pr_revision', None) or f'refs/pr/{number}'
        return PullRequest(
            number=number,
            url=url
            or f'https://huggingface.co/datasets/{self.repo_id}/discussions/{number}',
            revision=revision,
            title=pull_request_title(adapter),
        )

    def upload(
        self,
        pull_request: PullRequest,
        *,
        operations: Sequence[CommitOperationAdd],
        message: str,
    ) -> list[Any]:
        """Add records to an existing pull request, in bounded batches."""
        commits = []
        batches = list(_chunks(operations, self.batch_size))
        for index, batch in enumerate(batches, start=1):
            suffix = f' ({index}/{len(batches)})' if len(batches) > 1 else ''
            try:
                commits.append(
                    self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type='dataset',
                        revision=pull_request.revision,
                        operations=list(batch),
                        commit_message=f'{message}{suffix}',
                    )
                )
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise SubmissionError(
                    f'could not add records to {pull_request.url}: '
                    f'{type(exc).__name__}: {exc}'
                ) from exc
        return commits


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def upload_operations(upload_dir: Any) -> list[CommitOperationAdd]:
    """Return one commit operation per file in a prepared upload tree."""
    from pathlib import Path

    upload_dir = Path(upload_dir)
    operations = []
    for path in sorted(upload_dir.rglob('*')):
        if path.is_file():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path.relative_to(upload_dir).as_posix(),
                    path_or_fileobj=str(path),
                )
            )
    return operations


def pull_request_description(
    adapter: str,
    *,
    coverage_line: str,
    run_date: str,
    status: str,
    run_url: str | None = None,
    raw_reference: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    """Compose the body a reviewer reads, including the machine marker."""
    lines = [
        f'Automated daily ingestion for the `{adapter}` adapter.',
        '',
        f'- **Run date**: {run_date}',
        f'- **Status**: {status}',
        f'- **Coverage**: {coverage_line}',
    ]
    if raw_reference:
        lines.append(f'- **Raw source snapshot**: `{raw_reference}`')
    if run_url:
        lines.append(f'- **Workflow run**: {run_url}')
    lines += [
        '',
        'Every record carries `type_of_addition: cron` with the run date, '
        'adapter and workflow run in `source_metadata.additional_details`, '
        'so this batch can be found and corrected later.',
        '',
        'Adapter code lives in '
        '[`every_eval_ever`](https://github.com/evaleval/every_eval_ever) '
        'under `every_eval_ever/adapters/`; this pull request carries data '
        'only.',
    ]
    if notes:
        lines += ['', '### Notes', *[f'- {note}' for note in notes]]
    lines += ['', marker(adapter)]
    return '\n'.join(lines)


__all__ = [
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DATASTORE_REPO',
    'MARKER_PREFIX',
    'AmbiguousPullRequestError',
    'DatastoreSubmitter',
    'PullRequest',
    'SubmissionError',
    'marker',
    'pull_request_description',
    'pull_request_title',
    'upload_operations',
]
