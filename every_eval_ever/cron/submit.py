"""Send one adapter's records to that adapter's own datastore pull request.

The rule the ticket asks for is one pull request per adapter, reused across
runs. Getting that wrong is expensive in both directions: opening a fresh
pull request every day buries reviewers, and guessing at which existing one
to reuse can push a scrape into somebody else's submission. So the pull
request is remembered by number, re-checked before use, and identified by two
things the cron itself controls, never by "the newest open one".

Those two are the account that opened it and the ``eee-cron-adapter`` line in
its body. The account matters most: the datastore is public, so anyone can
open a pull request whose first comment carries our marker, and a run that
adopted it would commit records onto a branch a stranger controls. The marker
then says which adapter it belongs to.

An ambiguous match is an error. There is no safe guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from huggingface_hub import CommitOperationAdd

DEFAULT_DATASTORE_REPO = 'evaleval/EEE_datastore'
#: Machine-readable line in the pull request body identifying its adapter.
MARKER_PREFIX = 'eee-cron-adapter:'
#: A single commit of thousands of files can 504 with the commit still
#: landing server-side, so batches are kept small.
DEFAULT_BATCH_SIZE = 300
#: How an instance-level sidecar is named after its aggregate. The two are
#: one record and are committed together.
SAMPLES_SUFFIX = '_samples.jsonl'
#: The comment that asks the datastore's validation bot to check what a pull
#: request now carries. Validation on the Hub side runs on request, not on
#: push: a pull request nobody comments on is a pull request nobody
#: validated (see evaleval/EEE_datastore discussion 168).
VALIDATION_COMMAND = '/eee validate changed'


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
    errored and the pull request ref could not be read to arbitrate, so the
    records may or may not have landed. A caller must keep them in flight
    rather than treat them as absent, because re-uploading them blind is the
    same duplicate by another route.
    """

    def __init__(
        self,
        message: str,
        *,
        pull_request: PullRequest | None,
        committed_paths: Sequence[str],
        unresolved_paths: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.pull_request = pull_request
        self.committed_paths = tuple(committed_paths)
        self.unresolved_paths = tuple(unresolved_paths)


def marker(adapter: str) -> str:
    return f'{MARKER_PREFIX} {adapter}'


def pull_request_title(adapter: str) -> str:
    return f'[Submission] cron: {adapter} (automated ingestion)'


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
    """The datastore pull request one adapter publishes into."""

    number: int
    url: str
    revision: str
    title: str


@dataclass(frozen=True)
class Submission:
    """What one publish attempt actually put in the datastore."""

    pull_request: PullRequest
    committed_paths: tuple[str, ...]
    #: Why the pull request body still describes an earlier run, when it does.
    #: Not a failure: the records are published either way.
    description_note: str | None = None
    #: Why the datastore's validator was not asked to check this pull
    #: request, when it was not. Not a failure either, but worth a human
    #: reading: an unvalidated pull request sits unreviewed until somebody
    #: posts the command by hand.
    validation_note: str | None = None


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
        self._identity: dict[str, Any] | None = None

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
                'reuse a pull request without knowing who opened it'
            )
        self._identity = identity
        return identity

    @property
    def author(self) -> str:
        """Return the account this run publishes as.

        Nothing this cron did not open is a candidate for reuse. A token
        without a resolvable identity is an error rather than an empty filter,
        because an empty filter is how every open pull request on a public
        datastore becomes adoptable.
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
                f'read-only. Opening a pull request on {self.repo_id} and '
                'writing the raw store both need write access.'
            )
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type='dataset')
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not reach the datastore {self.repo_id}: '
                f'{type(exc).__name__}: {exc}. Check EEE_DATASTORE_REPO_ID '
                'and that this token may read it.'
            ) from exc

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
        line the cron wrote into the body, not the title. A title is display
        metadata: anyone can edit it to something that looks like ours, and a
        reviewer renaming ours does not hand it to somebody else. A body that
        cannot be read is an error rather than a "no", because silently
        treating it as unowned opens a second pull request for the same
        adapter.

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

    def update_description(
        self, pull_request: PullRequest, description: str
    ) -> None:
        """Rewrite a reused pull request's body to describe this run.

        A pull request is opened once and published into for as long as it
        stays open, so a body written at open time describes whichever run
        happened to be first. A reviewer opening it a month later reads that
        run's date, its coverage line and its raw-snapshot path, none of which
        are true of what is in front of them.

        The rewritten body carries the ``eee-cron-adapter`` marker like any
        other, since that line is what says which adapter the pull request
        belongs to and dropping it would orphan it from the next run.
        """
        try:
            details = self.api.get_discussion_details(
                repo_id=self.repo_id,
                repo_type='dataset',
                discussion_num=pull_request.number,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not read pull request {pull_request.number} on '
                f'{self.repo_id} to refresh its description: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        comment = _opening_comment(details)
        comment_id = getattr(comment, 'id', None) if comment else None
        if not comment_id:
            raise SubmissionError(
                f'pull request {pull_request.number} on {self.repo_id} has no '
                'readable opening comment to refresh'
            )
        try:
            self.api.edit_discussion_comment(
                repo_id=self.repo_id,
                repo_type='dataset',
                discussion_num=pull_request.number,
                comment_id=comment_id,
                new_content=description,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not refresh the description of pull request '
                f'{pull_request.number} on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

    def request_validation(self, pull_request: PullRequest) -> None:
        """Post :data:`VALIDATION_COMMAND` on a pull request, as a new
        comment, which is what makes the datastore validate it."""
        try:
            self.api.comment_discussion(
                repo_id=self.repo_id,
                repo_type='dataset',
                discussion_num=pull_request.number,
                comment=VALIDATION_COMMAND,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise SubmissionError(
                f'could not request validation on pull request '
                f'{pull_request.number} on {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

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
        """Return the remembered pull request if it is still usable.

        Usable means: opened by this account, still open, still a pull
        request, and still carrying this adapter's marker in its body. A
        merged, closed, or repurposed discussion is treated as gone, so the
        next run opens a fresh one rather than pushing into something a
        reviewer has finished with. A remembered number that now points at
        somebody else's pull request is treated as gone for the same reason.
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
        """Find this adapter's pull request when no number is remembered.

        Every open pull request this account opened is checked for the marker,
        including ones whose title no longer looks like ours, because a title
        edit must not strand the pull request the cron has been publishing
        into. Two matches is an error rather than a choice: picking one would
        mean appending a scrape to a pull request nobody expected it in.
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

    def open_pull_request(
        self,
        adapter: str,
        *,
        operations: Sequence[CommitOperationAdd],
        description: str,
    ) -> tuple[PullRequest, bool]:
        """Open this adapter's pull request with its first batch of records.

        Returns the pull request and whether this call is what created it. A
        ``create_commit`` can time out after the Hub accepted it, and the call
        that opens a pull request is the one where that matters most: the
        number is reported in the reply nobody received, so the run fails
        knowing nothing, and the next run opens a second pull request holding
        the same records.

        So a failure here asks the Hub whether the pull request exists after
        all, by the same account-plus-marker rule used everywhere else. Finding
        it means the commit landed and this call simply lost the answer.
        Finding nothing means it did not, which is the error it always was.
        """
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
            landed = self._opened_despite(adapter)
            if landed is not None:
                return landed, False
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
            landed = self._opened_despite(adapter)
            if landed is not None:
                return landed, True
            raise SubmissionError(
                'the Hub accepted the commit but did not report a pull '
                'request number; check the repository before re-running so '
                'a second pull request is not opened'
            )
        revision = getattr(commit, 'pr_revision', None) or f'refs/pr/{number}'
        return (
            PullRequest(
                number=number,
                url=url
                or f'https://huggingface.co/datasets/{self.repo_id}/discussions/{number}',
                revision=revision,
                title=pull_request_title(adapter),
            ),
            True,
        )

    def _opened_despite(self, adapter: str) -> PullRequest | None:
        """Return this adapter's pull request if one exists after a failure.

        Answering "did the commit land" by looking for what it would have
        created. Only pull requests this account opened carrying this
        adapter's marker count, the same rule that governs reuse anywhere
        else, so nothing is adopted that the cron did not open.

        A lookup that itself fails answers nothing, and the caller reports the
        original failure. The pull request, if there is one, is found by the
        next run instead.
        """
        try:
            return self.find_by_marker(adapter)
        except AmbiguousPullRequestError:
            raise
        except SubmissionError:
            return None

    def upload(
        self,
        pull_request: PullRequest,
        *,
        operations: Sequence[CommitOperationAdd],
        message: str,
    ) -> list[Any]:
        """Add records to an existing pull request, in bounded batches."""
        return self._upload_batches(
            pull_request,
            batches=_batches(operations, self.batch_size),
            message=message,
            committed=[],
        )

    def _upload_batches(
        self,
        pull_request: PullRequest,
        *,
        batches: Sequence[Sequence[CommitOperationAdd]],
        message: str,
        committed: list[str],
        offset: int = 0,
        total: int | None = None,
    ) -> list[Any]:
        commits = []
        total = len(batches) + offset if total is None else total
        for index, batch in enumerate(batches, start=offset + 1):
            suffix = f' ({index}/{total})' if total > 1 else ''
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
                landed = self._paths_on_ref(pull_request, batch)
                if landed:
                    # The Hub accepted the commit and only the reply was
                    # lost, so this batch is on the pull request and the
                    # upload carries on. Stopping here instead would end a
                    # run whose every batch landed as a failure nothing
                    # retries, because its records are all accounted for.
                    committed.extend(landed)
                    continue
                unresolved: list[str] = []
                if landed is None:
                    unresolved = [operation.path_in_repo for operation in batch]
                    hint = (
                        ' Whether the failing batch landed could not be '
                        'checked either; if the error was a timeout whose '
                        'commit went through, a retry would duplicate its '
                        f'records, so inspect {pull_request.url} before '
                        're-running.'
                    )
                else:
                    hint = ''
                raise PartialSubmissionError(
                    f'could not add records to {pull_request.url}: '
                    f'{type(exc).__name__}: {exc}.{hint}',
                    pull_request=pull_request,
                    committed_paths=committed,
                    unresolved_paths=unresolved,
                ) from exc
            committed.extend(operation.path_in_repo for operation in batch)
        return commits

    def _paths_on_ref(
        self,
        pull_request: PullRequest,
        batch: Sequence[CommitOperationAdd],
    ) -> list[str] | None:
        """Return the failed batch's paths if its commit landed regardless.

        A ``create_commit`` can time out after the Hub has accepted the
        commit (see :data:`DEFAULT_BATCH_SIZE`), so the error alone does not
        say the batch is absent. Reporting only the earlier batches in that
        case makes the caller's ledger forget the ambiguous one, and the
        retry republishes it under fresh UUID paths. The pull request ref is
        the arbiter: a Hub commit is atomic, so either every path in the
        batch is there or none is. ``None`` means the ref could not be read,
        which the caller reports rather than resolves.
        """
        paths = [operation.path_in_repo for operation in batch]
        present = self.paths_present(paths, revision=pull_request.revision)
        if present is None:
            return None
        return paths if len(present) == len(paths) else []

    def paths_present(
        self, paths: Sequence[str], *, revision: str | None = None
    ) -> set[str] | None:
        """Return which of ``paths`` exist at ``revision``, or ``None``.

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

    def publish(
        self,
        adapter: str,
        *,
        pull_request: PullRequest | None,
        operations: Sequence[CommitOperationAdd],
        description: str,
        message: str,
    ) -> Submission:
        """Send one adapter's records, opening its pull request if needed.

        Every commit is bounded, including the one that opens the pull
        request: a cold start is the largest upload an adapter ever does, so
        exempting it from the batch size aimed the ambiguous-timeout problem
        at exactly the run most likely to hit it.

        Batches are whole records (see :func:`_batches`), so every commit
        either publishes a record completely or not at all.

        A reused pull request has its description rewritten afterwards, so the
        body a reviewer reads describes the run that last added to it rather
        than whichever run opened it.

        A submission that lands completely ends by requesting validation
        (see :meth:`request_validation`); a partial one leaves that to the
        retry that completes it.

        An opening commit that errored after landing is adopted rather than
        repeated, and what it left on the ref decides whether its batch counts
        as published.

        A failure after something landed raises
        :class:`PartialSubmissionError` carrying what did, so the caller can
        record it and retry the remainder instead of publishing it twice.
        """
        batches = _batches(operations, self.batch_size)
        committed: list[str] = []
        reused = pull_request is not None
        offset = 0
        if pull_request is None:
            first = batches[0] if batches else []
            pull_request, opened = self.open_pull_request(
                adapter, operations=first, description=description
            )
            batches = batches[1:]
            offset = 1
            if opened:
                committed.extend(operation.path_in_repo for operation in first)
            else:
                # The pull request exists, so the commit that created it
                # landed and this run only lost the reply. Its files are on
                # the ref or they are not; either answer is better than
                # assuming, because assuming they landed loses records and
                # assuming they did not duplicates them.
                landed = self._paths_on_ref(pull_request, first)
                if landed is None:
                    raise PartialSubmissionError(
                        f'a pull request for {adapter} was opened at '
                        f'{pull_request.url} but the error hid its reply, and '
                        'whether its first batch landed could not be checked; '
                        'inspect it before re-running',
                        pull_request=pull_request,
                        committed_paths=(),
                        unresolved_paths=[
                            operation.path_in_repo for operation in first
                        ],
                    )
                if landed:
                    committed.extend(landed)
                else:
                    # An empty pull request from an earlier interrupted run.
                    # Send this batch into it rather than opening another.
                    batches = [first, *batches]
                    offset = 0
        self._upload_batches(
            pull_request,
            batches=batches,
            message=message,
            committed=committed,
            offset=offset,
            total=len(batches) + offset,
        )
        note = None
        if reused:
            # After the records, not before: a body describing an upload that
            # then failed is worse than one describing last week's.
            try:
                self.update_description(pull_request, description)
            except SubmissionError as exc:
                # The records are in. A stale body is worth reporting and not
                # worth failing a run that published everything it meant to.
                note = f'{exc}; the body still describes an earlier run'
        validation_note = None
        # Last, once the pull request holds everything this run meant to
        # publish, so the validator reads the finished submission. A partial
        # submission never reaches here, which is deliberate: asking for
        # validation of half an upload wastes the reviewer the command
        # summons.
        try:
            self.request_validation(pull_request)
        except SubmissionError as exc:
            # The records are in, same bargain as the description: report
            # that nobody asked for validation rather than fail the run.
            validation_note = (
                f'{exc}; post `{VALIDATION_COMMAND}` on it manually'
            )
        return Submission(
            pull_request=pull_request,
            committed_paths=tuple(committed),
            description_note=note,
            validation_note=validation_note,
        )


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
    sidecar in the next. When the second failed, the first was already on the
    pull request, the record could not be recorded as published because half
    of it had not arrived, and the retry sent the whole record again under a
    fresh UUID. The abandoned half stayed on the pull request declaring a
    companion file that does not exist, which a human then has to clear out
    before it can be merged.

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
        'Records accumulate here across runs. The figures below describe the '
        'run that last added to this pull request.',
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
    'PartialSubmissionError',
    'PullRequest',
    'Submission',
    'SubmissionError',
    'SAMPLES_SUFFIX',
    'VALIDATION_COMMAND',
    'marker',
    'pull_request_description',
    'pull_request_title',
    'upload_operations',
]
