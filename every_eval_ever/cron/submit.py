"""Commit one adapter's records straight to the datastore.

Cron output used to go up as one pull request per adapter, reused across
runs, waiting on a human merge. The review that matters happens before
publication: nothing reaches this module that the packaged validator has not
passed, records are de-duplicated against the per-adapter ledger, and every
record names its run in `source_metadata.additional_details`. The pull
request added only a click to that, so passing runs now commit directly to
the datastore's default branch, one commit series per adapter run.

The pull-request machinery that remains, finding a pull request by author
and marker and asking what became of it, exists to settle what the retired
flow left behind: pending fingerprints waiting on a pull request a reviewer
may yet merge or close, and in-flight batches that were headed for one. It
can go once no adapter's state names a pull request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from huggingface_hub import CommitOperationAdd

DEFAULT_DATASTORE_REPO = 'evaleval/EEE_datastore'
#: Machine-readable line the retired flow wrote into a pull request body to
#: identify its adapter. Still how a leftover pull request is recognised.
MARKER_PREFIX = 'eee-cron-adapter:'
#: A single commit of thousands of files can 504 with the commit still
#: landing server-side, so batches are kept small.
DEFAULT_BATCH_SIZE = 300
#: How an instance-level sidecar is named after its aggregate. The two are
#: one record and are committed together.
SAMPLES_SUFFIX = '_samples.jsonl'


class SubmissionError(RuntimeError):
    """Raised when a submission cannot proceed safely."""


class AmbiguousPullRequestError(SubmissionError):
    """Raised when more than one open pull request claims one adapter."""


class PartialSubmissionError(SubmissionError):
    """Raised when some batches landed and a later one did not.

    Carries what actually reached the datastore. A caller that discards this
    and retries from scratch republishes the landed records under fresh
    UUID paths, which is the duplicate this exists to prevent.

    ``unresolved_paths`` names the one batch that is neither: its commit
    errored and the datastore could not be read to arbitrate, so the records
    may or may not have landed. A caller must keep them in flight rather
    than treat them as absent, because re-uploading them blind is the same
    duplicate by another route.
    """

    def __init__(
        self,
        message: str,
        *,
        committed_paths: Sequence[str],
        unresolved_paths: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.committed_paths = tuple(committed_paths)
        self.unresolved_paths = tuple(unresolved_paths)


def marker(adapter: str) -> str:
    return f'{MARKER_PREFIX} {adapter}'


def _opening_comment(details: Any) -> Any | None:
    """Return the comment a pull request was opened with, if it is readable."""
    for event in getattr(details, 'events', None) or ():
        content = getattr(event, 'content', None)
        if isinstance(content, str) and content:
            return event
    return None


def _first_comment(details: Any) -> str:
    """Return the body a pull request was opened with, if it is readable."""
    comment = _opening_comment(details)
    content = getattr(comment, 'content', None) if comment else None
    return content if isinstance(content, str) else ''


@dataclass(frozen=True)
class PullRequest:
    """A datastore pull request the retired flow published into."""

    number: int
    url: str
    revision: str
    title: str


@dataclass(frozen=True)
class Submission:
    """What one publish attempt actually put in the datastore."""

    committed_paths: tuple[str, ...]


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
    """Commit an adapter's records to the datastore's default branch."""

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
        self._identity: dict[str, Any] | None = None

    @property
    def repo_url(self) -> str:
        return f'https://huggingface.co/datasets/{self.repo_id}'

    def _resolve_identity(self) -> dict[str, Any]:
        """Return who this token is, asked once and cached."""
        if self._identity is not None:
            return self._identity
        try:
            identity = self.api.whoami()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                'could not resolve the account this token publishes as: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        if not isinstance(identity, dict) or not identity.get('name'):
            raise SubmissionError(
                'the Hub reported no username for this token; refusing to '
                'settle pull requests without knowing who opened them'
            )
        self._identity = identity
        return identity

    @property
    def author(self) -> str:
        """Return the account this run publishes as.

        Nothing this cron did not open counts when settling leftover pull
        requests. A token without a resolvable identity is an error rather
        than an empty filter, because an empty filter is how every open pull
        request on a public datastore becomes adoptable.
        """
        return self._resolve_identity()['name']

    def ensure_writable(self) -> None:
        """Check what can be checked before an adapter spends an hour.

        Both failures this catches are already caught, at the publish step, an
        adapter run later. Catching them here costs two requests and turns
        "scraped a leaderboard for forty-five minutes, then could not commit"
        into a job that fails in seconds and says which setting is wrong.

        A role the Hub does not report is not treated as read-only. Only a
        commit proves a fine-grained token's scopes, so an uninterpretable
        role passes here and fails later if it has to.
        """
        identity = self._resolve_identity()
        auth = identity.get('auth')
        token = auth.get('accessToken') if isinstance(auth, dict) else None
        role = token.get('role') if isinstance(token, dict) else None
        if role == 'read':
            raise SubmissionError(
                f'the token authenticates as {identity["name"]} but is '
                f'read-only. Committing to {self.repo_id} and writing the '
                'raw store both need write access.'
            )
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type='dataset')
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not reach the datastore {self.repo_id}: '
                f'{type(exc).__name__}: {exc}. Check EEE_DATASTORE_REPO_ID '
                'and that this token may read it.'
            ) from exc

    # -- publishing -------------------------------------------------------

    def publish(
        self,
        *,
        operations: Sequence[CommitOperationAdd],
        description: str,
        message: str,
    ) -> Submission:
        """Commit records to the datastore, in bounded batches.

        Batches are whole records (see :func:`_batches`), so every commit
        either publishes a record completely or not at all. Each commit
        carries ``description``, so the provenance a reviewer used to read
        in a pull request body is on the commits themselves.

        A commit that errored after the Hub accepted it is adopted rather
        than repeated: the datastore's file listing arbitrates, and a batch
        proven present counts as committed and the upload carries on.

        A failure after something landed raises
        :class:`PartialSubmissionError` carrying what did, so the caller can
        record it and retry the remainder instead of publishing it twice.
        """
        batches = _batches(operations, self.batch_size)
        committed: list[str] = []
        total = len(batches)
        for index, batch in enumerate(batches, start=1):
            suffix = f' ({index}/{total})' if total > 1 else ''
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    operations=list(batch),
                    commit_message=f'{message}{suffix}',
                    commit_description=description,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                landed = self._landed_anyway(batch)
                if landed:
                    # The Hub accepted the commit and only the reply was
                    # lost, so this batch is in the datastore and the upload
                    # carries on. Stopping here instead would end a run
                    # whose every batch landed as a failure nothing retries,
                    # because its records are all accounted for.
                    committed.extend(landed)
                    continue
                unresolved: list[str] = []
                if landed is None:
                    unresolved = [
                        operation.path_in_repo for operation in batch
                    ]
                    hint = (
                        ' Whether the failing batch landed could not be '
                        'checked either; if the error was a timeout whose '
                        'commit went through, a retry would duplicate its '
                        f'records, so inspect {self.repo_url} before '
                        're-running.'
                    )
                else:
                    hint = ''
                raise PartialSubmissionError(
                    f'could not commit records to {self.repo_id}: '
                    f'{type(exc).__name__}: {exc}.{hint}',
                    committed_paths=committed,
                    unresolved_paths=unresolved,
                ) from exc
            committed.extend(operation.path_in_repo for operation in batch)
        return Submission(committed_paths=tuple(committed))

    def _landed_anyway(
        self, batch: Sequence[CommitOperationAdd]
    ) -> list[str] | None:
        """Return the failed batch's paths if its commit landed regardless.

        A ``create_commit`` can time out after the Hub has accepted the
        commit (see :data:`DEFAULT_BATCH_SIZE`), so the error alone does not
        say the batch is absent. The datastore is the arbiter: a Hub commit
        is atomic, so either every path in the batch is there or none is.
        ``None`` means the datastore could not be read, which the caller
        reports rather than resolves.
        """
        paths = [operation.path_in_repo for operation in batch]
        present = self.paths_present(paths)
        if present is None:
            return None
        return paths if len(present) == len(paths) else []

    def paths_present(
        self, paths: Sequence[str], *, revision: str | None = None
    ) -> set[str] | None:
        """Return which of ``paths`` exist at ``revision``, or ``None``.

        ``revision`` defaults to the datastore's default branch, where runs
        publish; settling a leftover pull request passes its ref instead.
        ``None`` means the question could not be asked, which callers report
        rather than resolve: an empty answer and an unanswerable one mean
        opposite things about whether records were published.
        """
        try:
            files = set(
                self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    revision=revision,
                )
            )
        except Exception:  # noqa: BLE001 - the caller decides what it means
            return None
        return {path for path in paths if path in files}

    # -- settling what the retired pull-request flow left behind ----------

    def _open_pull_requests(self) -> list[Any]:
        """Return the open pull requests this account opened, and no others."""
        author = self.author
        try:
            discussions = list(
                self.api.get_repo_discussions(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    discussion_type='pull_request',
                    discussion_status='open',
                    author=author,
                )
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not list open pull requests on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        # The server-side filter is the cheap part. This is the check that has
        # to hold, so a Hub that ignores the parameter cannot widen the set.
        return [
            discussion
            for discussion in discussions
            if getattr(discussion, 'author', None) == author
        ]

    def carries_marker(self, number: int, adapter: str) -> bool:
        """Return whether a pull request body claims this adapter.

        Which adapter a pull request belongs to is the ``eee-cron-adapter``
        line the retired flow wrote into the body, not the title. A title is
        display metadata: anyone can edit it to something that looks like
        ours, and a reviewer renaming ours does not hand it to somebody
        else. A body that cannot be read is an error rather than a "no",
        because silently treating it as unowned would mis-settle whatever is
        still waiting on that pull request.

        Callers reach this only for pull requests :attr:`author` opened, so
        the marker answers "which adapter", not "is this ours".
        """
        try:
            details = self.api.get_discussion_details(
                repo_id=self.repo_id,
                repo_type='dataset',
                discussion_num=number,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not read pull request {number} on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        return marker(adapter) in _first_comment(details)

    def pull_request_status(self, number: int) -> str:
        """Return ``open``, ``merged`` or ``closed`` for a pull request.

        Merged and closed are different verdicts for the ledger. Merged means
        the records a pull request carried are in the datastore, so their
        fingerprints may be kept forever. Closed without merging means a
        reviewer rejected them, so the same fingerprints must be forgotten:
        kept, they would filter the unchanged records out of every later run
        and the resubmission could never happen.

        An unanswerable status is an error rather than a guess, because both
        wrong guesses lose data: "merged" buries rejected records for good,
        and "closed" republishes accepted ones.
        """
        try:
            details = self.api.get_discussion_details(
                repo_id=self.repo_id,
                repo_type='dataset',
                discussion_num=number,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not read pull request {number} on {self.repo_id} to '
                f'settle its records: {type(exc).__name__}: {exc}'
            ) from exc
        if not getattr(details, 'is_pull_request', False):
            raise SubmissionError(
                f'discussion {number} on {self.repo_id} is not a pull '
                'request, but the ledger says records were committed to it; '
                'refusing to guess what happened to them'
            )
        status = getattr(details, 'status', None)
        if status == 'draft':
            return 'open'
        if status not in ('open', 'merged', 'closed'):
            raise SubmissionError(
                f'pull request {number} on {self.repo_id} reports '
                f'unrecognised status {status!r}'
            )
        return status

    def resolve_known(self, adapter: str, number: int) -> PullRequest | None:
        """Return the remembered pull request if it still answers for itself.

        That means: opened by this account, still open, still a pull
        request, and still carrying this adapter's marker in its body. A
        merged, closed, or repurposed discussion returns ``None``, as does a
        remembered number that now points at somebody else's pull request.
        """
        for discussion in self._open_pull_requests():
            if _discussion_number(discussion) != number:
                continue
            if not _is_open_pull_request(discussion):
                return None
            if not self.carries_marker(number, adapter):
                return None
            return PullRequest(
                number=number,
                url=_discussion_url(self.repo_id, discussion, number),
                revision=_discussion_revision(discussion, number),
                title=getattr(discussion, 'title', '') or '',
            )
        return None

    def find_by_marker(self, adapter: str) -> PullRequest | None:
        """Find an adapter's leftover pull request when no number is known.

        Every open pull request this account opened is checked for the
        marker, including ones whose title no longer looks like ours,
        because a title edit must not strand a pull request records are
        waiting on. Two matches is an error rather than a choice: picking
        one would settle records against a pull request nobody sent them to.
        """
        matches = []
        for discussion in self._open_pull_requests():
            if not _is_open_pull_request(discussion):
                continue
            number = _discussion_number(discussion)
            if number is None:
                continue
            if not self.carries_marker(number, adapter):
                continue
            title = getattr(discussion, 'title', '') or ''
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


def _record_key(path_in_repo: str) -> str:
    """Return the aggregate a staged datastore path belongs to."""
    if path_in_repo.endswith(SAMPLES_SUFFIX):
        return path_in_repo[: -len(SAMPLES_SUFFIX)] + '.json'
    return path_in_repo


def _batches(
    operations: Sequence[CommitOperationAdd], size: int
) -> list[list[CommitOperationAdd]]:
    """Split operations into commits, never splitting one record across two.

    A record is its aggregate plus, sometimes, an instance-level sidecar named
    after it, and a Hub commit is atomic. Keeping both files in one commit is
    therefore what makes "this record landed" a question with an answer.

    Chunking the flat file list could put an aggregate in one commit and its
    sidecar in the next. When the second failed, the first was already in the
    datastore, the record could not be recorded as published because half
    of it had not arrived, and the retry sent the whole record again under a
    fresh UUID. The abandoned half stayed behind declaring a companion file
    that does not exist, which a human then has to clear out.

    A record larger than ``size`` gets a commit to itself rather than being
    split, since splitting is the thing this exists to prevent.
    """
    grouped: dict[str, list[CommitOperationAdd]] = {}
    for operation in operations:
        grouped.setdefault(_record_key(operation.path_in_repo), []).append(
            operation
        )

    batches: list[list[CommitOperationAdd]] = []
    current: list[CommitOperationAdd] = []
    for group in grouped.values():
        if current and len(current) + len(group) > size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


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


def commit_description(
    adapter: str,
    *,
    coverage_line: str,
    run_date: str,
    status: str,
    run_url: str | None = None,
    raw_reference: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    """Compose the description every datastore commit of one run carries."""
    lines = [
        f'Automated ingestion for the `{adapter}` adapter.',
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
        'under `every_eval_ever/adapters/`; this commit carries data only.',
    ]
    if notes:
        lines += ['', '### Notes', *[f'- {note}' for note in notes]]
    return '\n'.join(lines)


__all__ = [
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DATASTORE_REPO',
    'MARKER_PREFIX',
    'AmbiguousPullRequestError',
    'DatastoreSubmitter',
    'PartialSubmissionError',
    'PullRequest',
    'Submission',
    'SubmissionError',
    'SAMPLES_SUFFIX',
    'commit_description',
    'marker',
    'upload_operations',
]
