"""The cron's durable memory: raw snapshots and one ledger per adapter.

Records go to the datastore through a pull request, which is exactly why the
cron cannot keep its own state there. A previous attempt at this read its
state from a branch it only ever wrote pull requests to, so every run started
from scratch. State therefore lives in a repository the cron can write
directly, alongside the raw snapshots it is keeping anyway.

Layout of ``evaleval/EEE_raw`` (``main``)::

    raw/<adapter>/<date>/<run>/<sha256><ext>   payload bytes
    raw/<adapter>/<date>/<run>/manifest.jsonl  one line per capture
    raw/<adapter>/<date>/<run>/run.json        outcome, coverage, PR link
    state/<adapter>.json                       PR number, last run
    state/<adapter>.fingerprints               one sha256 per merged record
    state/<adapter>.pending                    one sha256 per record still
                                               waiting on its pull request
    state/<adapter>.inflight                   records this run is about to
                                               publish, written before it does

``<date>`` is the UTC run date and ``<run>`` identifies the run within it, so
a second attempt on one day adds a directory instead of overwriting the first
attempt's manifest and report. Which run wrote a snapshot is exactly what a
reader wants on the days there was more than one.

Fingerprints live in their own newline-delimited files: the merged set is
cumulative and reaches thousands of lines for the larger leaderboards, and
keeping it out of the JSON keeps the part a human reads small and the diffs
meaningful. Pending fingerprints are kept apart from merged ones because they
mean less: they say a record reached the adapter's open pull request, not the
datastore. When that pull request is merged they are promoted; when it is
closed without merging they are dropped, so the records they named are
resubmitted instead of being silently filtered forever.

Publication is the one step whose result cannot be recovered by repeating it,
so the intention to publish is written down before it happens. A run that
uploads records and then fails to write its ledger would otherwise leave them
on the pull request with nothing naming them, and the next run would send the
same evaluations again under fresh paths. The in-flight file says which
records went where; the next run checks them against the pull request and
records the ones that arrived.

The store is private, and that is enforced rather than documented. It holds
whole source payloads kept so a published record can be checked against what
it was converted from; republishing them is a different thing that nobody
agreed to. :meth:`RawStore.ensure_private` runs before a refresh starts and
:meth:`RawStore._require_private` runs before every commit, and neither ever
changes a repository's visibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

#: Default home for raw snapshots and cron state.
DEFAULT_RAW_REPO = 'evaleval/EEE_raw'
STATE_DIR = 'state'
RAW_DIR = 'raw'
MANIFEST_NAME = 'manifest.jsonl'
RUN_REPORT_NAME = 'run.json'
#: Payloads identical to the previous run are referenced, not re-uploaded.
UNCHANGED_MARKER = 'same_as'
#: How many times a state commit re-reads the head and tries again when
#: another adapter's job moved the branch first.
COMMIT_ATTEMPTS = 5


class StoreError(RuntimeError):
    """Raised when cron state cannot be read or written safely."""


class HubClient(Protocol):
    """The slice of ``HfApi`` this module uses."""

    def hf_hub_download(self, **kwargs: Any) -> str: ...

    def create_commit(self, **kwargs: Any) -> Any: ...

    def repo_info(self, **kwargs: Any) -> Any: ...

    def create_repo(self, **kwargs: Any) -> Any: ...


@dataclass
class AdapterState:
    """What the cron remembers about one adapter between runs."""

    adapter: str
    pull_request_number: int | None = None
    pull_request_url: str | None = None
    last_run_date: str | None = None
    last_raw_date: str | None = None
    #: Full path of the snapshot directory the previous run wrote, which is
    #: what the next run's de-duplication reads. Kept whole rather than rebuilt
    #: from :attr:`last_raw_date`, since a date no longer names one directory.
    last_raw_prefix: str | None = None
    last_status: str | None = None
    #: Records that were merged into the datastore. Kept forever.
    fingerprints: set[str] = field(default_factory=set)
    #: Records committed to :attr:`pull_request_number` but not merged yet.
    #: Promoted into :attr:`fingerprints` when that pull request merges,
    #: dropped when it is closed without merging.
    pending_fingerprints: set[str] = field(default_factory=set)
    #: Commit the state was read at, so a concurrent write is rejected
    #: instead of silently overwriting a newer one.
    parent_commit: str | None = None
    #: ``False`` when no state file existed yet, which is a cold start rather
    #: than "this adapter has published nothing".
    exists: bool = False

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    'adapter': self.adapter,
                    'pull_request_number': self.pull_request_number,
                    'pull_request_url': self.pull_request_url,
                    'last_run_date': self.last_run_date,
                    'last_raw_date': self.last_raw_date,
                    'last_raw_prefix': self.last_raw_prefix,
                    'last_status': self.last_status,
                    'fingerprint_count': len(self.fingerprints),
                    'pending_fingerprint_count': len(self.pending_fingerprints),
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )

    @property
    def known_fingerprints(self) -> set[str]:
        """Every fingerprint that should keep a record from re-uploading.

        Merged records are in the datastore and pending ones are already on
        the adapter's open pull request; re-publishing either would put the
        same evaluation up twice. What this deliberately does not decide is
        whether the pending set still deserves to be here: the caller settles
        that against the pull request's fate before running the adapter.
        """
        return self.fingerprints | self.pending_fingerprints

    def fingerprints_text(self) -> str:
        return ''.join(f'{value}\n' for value in sorted(self.fingerprints))

    def pending_fingerprints_text(self) -> str:
        return ''.join(
            f'{value}\n' for value in sorted(self.pending_fingerprints)
        )


@dataclass
class InflightBatch:
    """What a run is about to publish, recorded before it publishes it.

    Written in the same commit as the raw snapshot, ahead of the datastore
    upload, and emptied by the commit that records the run, except for
    records whose batch errored while the pull request was unreadable: those
    stay in flight, since whether they landed is exactly the question this
    file exists to answer. Finding a non-empty one at the start of a run
    means a previous run uploaded records, or may have, without recording
    them, so they are on the pull request with no fingerprint naming them.

    Each record is its fingerprint and every datastore path it consists of, so
    the next run can ask the pull request which of them arrived rather than
    assuming all or none did.
    """

    adapter: str
    run_date: str | None = None
    run_token: str | None = None
    #: The pull request the records were headed for, when one was known
    #: already. ``None`` on a cold start, where the upload itself opens it.
    pull_request_number: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    'adapter': self.adapter,
                    'run_date': self.run_date,
                    'run_token': self.run_token,
                    'pull_request_number': self.pull_request_number,
                    'records': [
                        {
                            'fingerprint': record['fingerprint'],
                            'paths': list(record['paths']),
                        }
                        for record in self.records
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )

    @property
    def paths(self) -> list[str]:
        return sorted(
            {path for record in self.records for path in record['paths']}
        )


def state_path(adapter: str) -> str:
    return f'{STATE_DIR}/{adapter}.json'


def fingerprints_path(adapter: str) -> str:
    return f'{STATE_DIR}/{adapter}.fingerprints'


def pending_fingerprints_path(adapter: str) -> str:
    return f'{STATE_DIR}/{adapter}.pending'


def inflight_path(adapter: str) -> str:
    return f'{STATE_DIR}/{adapter}.inflight'


def raw_prefix(adapter: str, run_date: date, run_token: str) -> str:
    """Return the directory one run's snapshot and report belong in.

    The date alone was not enough. Payloads are content-addressed and survive
    a second run that day, but ``manifest.jsonl`` and ``run.json`` are written
    at fixed names, so a re-run overwrote the record of what the first run
    fetched and what happened to it. Re-runs are the normal case for the runs
    worth reading about later: a job that was cancelled, a source that was
    down at 03:17, a manual run after a fix.
    """
    return f'{RAW_DIR}/{adapter}/{run_date.isoformat()}/{run_token}'


class RawStore:
    """Read and write the cron's snapshots and ledgers on the Hub."""

    def __init__(
        self,
        api: Any,
        *,
        repo_id: str = DEFAULT_RAW_REPO,
        revision: str = 'main',
    ) -> None:
        if not repo_id:
            raise StoreError('a raw-store repository id is required')
        self.api = api
        self.repo_id = repo_id
        self.revision = revision

    # -- visibility ------------------------------------------------------

    def _repo_info(self) -> Any:
        return self.api.repo_info(repo_id=self.repo_id, repo_type='dataset')

    def _public_error(self) -> StoreError:
        return StoreError(
            f'{self.repo_id} is public. It holds whole source payloads, kept '
            'so a record can be checked against what it was converted from '
            'and not so they can be republished, so nothing is written to it '
            'while the world can read it. Make it private, or point '
            'EEE_RAW_REPO_ID at a dataset that already is. Changing a '
            'repository from public to private is a decision with '
            'consequences for anyone already reading it, so this run reports '
            'it rather than doing it.'
        )

    def ensure_private(self, *, create: bool = True) -> None:
        """Check the store exists and is private, before a run does any work.

        Privacy here is the difference between an archive and a republication.
        Written down in a README it holds until the first time somebody
        creates the dataset by hand, accepts the Hub's public default, and
        gets a green run.

        Only a genuine "not found" counts as missing. A 500 or a network
        failure is not evidence that a public dataset is absent, and creating
        or blessing anything on that basis is how a public repository gets
        treated as a private one for the rest of its life.
        """
        try:
            info = self._repo_info()
        except RepositoryNotFoundError:
            info = None
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise StoreError(
                f'could not check whether {self.repo_id} is private: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

        if info is not None:
            if not getattr(info, 'private', False):
                raise self._public_error()
            return

        if not create:
            raise StoreError(
                f'{self.repo_id} does not exist. Create it as a private '
                'dataset, or point EEE_RAW_REPO_ID at one that does.'
            )

        try:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type='dataset',
                private=True,
                exist_ok=True,
            )
            created = self._repo_info()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise StoreError(
                f'{self.repo_id} does not exist and could not be created: '
                f'{type(exc).__name__}: {exc}. Create it by hand as a '
                'private dataset, or point EEE_RAW_REPO_ID at one that '
                'exists.'
            ) from exc
        if not getattr(created, 'private', False):
            raise StoreError(
                f'{self.repo_id} was created but reads back as public; '
                'refusing to treat it as a home for raw source payloads.'
            )

    def _require_private(self) -> None:
        """Refuse to write to a store the world can read.

        Checked immediately before every commit rather than only at startup:
        these runs are unattended and daily, and a repository that was private
        in August can be public tonight because of a cleanup or an org-wide
        settings sweep. An unanswerable check is a refusal, not a pass.
        """
        try:
            info = self._repo_info()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise StoreError(
                f'could not confirm that {self.repo_id} is still private: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        if not getattr(info, 'private', False):
            raise self._public_error()

    # -- reading ---------------------------------------------------------

    def _download_text(self, path: str) -> str | None:
        """Return a repository file's text, or ``None`` if it is absent.

        Only a genuine "not found" may become ``None``. An auth or transport
        error that quietly became an empty ledger would make the cron forget
        every fingerprint it knows and republish the entire history.
        """
        try:
            local = self.api.hf_hub_download(
                repo_id=self.repo_id,
                repo_type='dataset',
                revision=self.revision,
                filename=path,
            )
        except EntryNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise StoreError(
                f'could not read {path} from {self.repo_id}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        return Path(local).read_text(encoding='utf-8')

    def head_commit(self) -> str | None:
        """Return the revision the store is currently at, if resolvable."""
        try:
            info = self.api.dataset_info(self.repo_id, revision=self.revision)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise StoreError(
                f'could not resolve {self.repo_id}@{self.revision}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc
        return getattr(info, 'sha', None)

    def read_state(self, adapter: str) -> AdapterState:
        """Load one adapter's ledger, distinguishing absent from unreadable."""
        state = AdapterState(adapter=adapter)
        state.parent_commit = self.head_commit()

        raw = self._download_text(state_path(adapter))
        if raw is not None:
            state.exists = True
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StoreError(
                    f'{state_path(adapter)} in {self.repo_id} is not valid '
                    f'JSON: {exc}'
                ) from exc
            if not isinstance(payload, dict):
                raise StoreError(
                    f'{state_path(adapter)} in {self.repo_id} must contain a '
                    'JSON object'
                )
            state.pull_request_number = payload.get('pull_request_number')
            state.pull_request_url = payload.get('pull_request_url')
            state.last_run_date = payload.get('last_run_date')
            state.last_raw_date = payload.get('last_raw_date')
            state.last_raw_prefix = payload.get('last_raw_prefix')
            if state.last_raw_prefix is None and state.last_raw_date:
                # Written before snapshots were scoped to a run. The date was
                # the whole directory then, so it still names one.
                state.last_raw_prefix = (
                    f'{RAW_DIR}/{adapter}/{state.last_raw_date}'
                )
            state.last_status = payload.get('last_status')

        ledger = self._download_text(fingerprints_path(adapter))
        if ledger is not None:
            state.exists = True
            state.fingerprints = {
                line.strip() for line in ledger.splitlines() if line.strip()
            }

        pending = self._download_text(pending_fingerprints_path(adapter))
        if pending is not None:
            state.exists = True
            state.pending_fingerprints = {
                line.strip() for line in pending.splitlines() if line.strip()
            }
        return state

    def read_inflight(self, adapter: str) -> InflightBatch:
        """Return what the previous run was about to publish, if anything.

        An unreadable or malformed file is an error rather than an empty
        batch. Treating it as empty is what would bury the very records it
        exists to account for.
        """
        batch = InflightBatch(adapter=adapter)
        raw = self._download_text(inflight_path(adapter))
        if raw is None:
            return batch
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(
                f'{inflight_path(adapter)} in {self.repo_id} is not valid '
                f'JSON: {exc}'
            ) from exc
        if not isinstance(payload, dict):
            raise StoreError(
                f'{inflight_path(adapter)} in {self.repo_id} must contain a '
                'JSON object'
            )
        batch.run_date = payload.get('run_date')
        batch.run_token = payload.get('run_token')
        batch.pull_request_number = payload.get('pull_request_number')
        for record in payload.get('records') or ():
            fingerprint = (record or {}).get('fingerprint')
            paths = (record or {}).get('paths')
            if not fingerprint or not isinstance(paths, list) or not paths:
                raise StoreError(
                    f'{inflight_path(adapter)} in {self.repo_id} names a '
                    'record without a fingerprint or without paths'
                )
            batch.records.append(
                {'fingerprint': fingerprint, 'paths': list(paths)}
            )
        return batch

    def read_manifest(self, prefix: str) -> list[dict]:
        """Return the capture manifest under ``prefix``, or ``[]``."""
        raw = self._download_text(f'{prefix}/{MANIFEST_NAME}')
        if raw is None:
            return []
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    # -- writing ---------------------------------------------------------

    def commit(
        self,
        operations: list[CommitOperationAdd],
        *,
        message: str,
        parent_commit: str | None,
    ) -> Any:
        """Commit to the store, retrying when another adapter got there first.

        Every adapter job reads one shared head and writes back to the same
        branch, so a daily matrix of twenty adapters races on every run. The
        loser of that race has already published its records to the datastore
        by this point, and dropping its state commit would leave those
        records with no fingerprints, so the next run would publish them
        again under fresh paths.

        Retrying is safe rather than a lost update because a job only ever
        writes files it owns: ``state/<adapter>.*`` and this adapter's raw
        snapshot directory. The workflow's per-adapter concurrency group is
        what guarantees no second job is writing the same ones. A failure
        that is not a moved head is re-raised untouched, so a permission or
        transport error still fails the job.

        Visibility is confirmed here rather than trusted from startup, because
        the adapter has been running in between and a repository's visibility
        can change under it.
        """
        if not operations:
            return None
        self._require_private()
        parent = parent_commit
        for remaining in reversed(range(COMMIT_ATTEMPTS)):
            try:
                return self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    revision=self.revision,
                    operations=operations,
                    commit_message=message,
                    parent_commit=parent,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                moved = self._moved_head(parent) if remaining else None
                if moved is None:
                    raise StoreError(
                        f'could not write to {self.repo_id}: '
                        f'{type(exc).__name__}: {exc}'
                    ) from exc
                parent = moved
        return None

    def _moved_head(self, parent: str | None) -> str | None:
        """Return the branch head if it moved under us, else ``None``.

        A commit that was rejected while the head is exactly where we left it
        was not a race, so there is nothing to retry against.
        """
        if parent is None:
            return None
        try:
            current = self.head_commit()
        except StoreError:
            return None
        return current if current and current != parent else None


def plan_raw_upload(
    raw_dir: Path,
    *,
    prefix: str,
    previous_manifest: list[dict] | None = None,
    previous_prefix: str | None = None,
) -> tuple[list[CommitOperationAdd], list[dict]]:
    """Return the payloads to upload and the manifest describing the run.

    A payload whose sha256 is unchanged since the previous snapshot is not
    uploaded a second time; its manifest line points at the copy that is
    already stored. This is what keeps a daily snapshot of a leaderboard that
    rarely changes from costing a full copy per day.
    """
    raw_dir = Path(raw_dir)
    manifest_file = raw_dir / MANIFEST_NAME
    if not manifest_file.is_file():
        return [], []

    previous: dict[str, str] = {}
    if previous_manifest and previous_prefix:
        for entry in previous_manifest:
            if entry.get('kind') != 'payload':
                continue
            # An entry that was itself unchanged names a file the previous
            # run did not write, because that run referenced an earlier copy
            # instead of storing one. Following its target keeps every
            # reference pointing at bytes that exist, however many unchanged
            # runs have passed.
            target = entry.get(UNCHANGED_MARKER)
            if not target:
                stored = entry.get('path')
                if not stored:
                    continue
                target = f'{previous_prefix}/{stored}'
            previous[entry['sha256']] = target

    operations: list[CommitOperationAdd] = []
    manifest: list[dict] = []
    uploaded: set[str] = set()
    for line in manifest_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get('kind') != 'payload' or 'path' not in entry:
            manifest.append(entry)
            continue
        digest = entry['sha256']
        unchanged = previous.get(digest)
        if unchanged:
            manifest.append({**entry, UNCHANGED_MARKER: unchanged})
            continue
        manifest.append(entry)
        local = raw_dir / entry['path']
        if digest in uploaded or not local.is_file():
            continue
        uploaded.add(digest)
        operations.append(
            CommitOperationAdd(
                path_in_repo=f'{prefix}/{entry["path"]}',
                path_or_fileobj=str(local),
            )
        )

    operations.append(
        CommitOperationAdd(
            path_in_repo=f'{prefix}/{MANIFEST_NAME}',
            path_or_fileobj=(
                ''.join(
                    json.dumps(entry, sort_keys=True, ensure_ascii=False) + '\n'
                    for entry in manifest
                ).encode('utf-8')
            ),
        )
    )
    return operations, manifest


def run_report_operation(
    report: dict[str, Any], *, prefix: str
) -> CommitOperationAdd:
    """Return the commit operation for a run's outcome report."""
    return CommitOperationAdd(
        path_in_repo=f'{prefix}/{RUN_REPORT_NAME}',
        path_or_fileobj=(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
            + '\n'
        ).encode('utf-8'),
    )


def inflight_operation(batch: InflightBatch) -> CommitOperationAdd:
    """Return the commit operation that records or clears an in-flight batch.

    A batch with no records is written rather than deleted, so every run makes
    the same commit and "nothing in flight" is a file that says so instead of
    a file whose absence has to be interpreted.
    """
    return CommitOperationAdd(
        path_in_repo=inflight_path(batch.adapter),
        path_or_fileobj=batch.to_json().encode('utf-8'),
    )


def state_operations(state: AdapterState) -> list[CommitOperationAdd]:
    """Return the commit operations that persist one adapter's ledger."""
    return [
        CommitOperationAdd(
            path_in_repo=state_path(state.adapter),
            path_or_fileobj=state.to_json().encode('utf-8'),
        ),
        CommitOperationAdd(
            path_in_repo=fingerprints_path(state.adapter),
            path_or_fileobj=state.fingerprints_text().encode('utf-8'),
        ),
        CommitOperationAdd(
            path_in_repo=pending_fingerprints_path(state.adapter),
            path_or_fileobj=state.pending_fingerprints_text().encode('utf-8'),
        ),
    ]


__all__ = [
    'COMMIT_ATTEMPTS',
    'DEFAULT_RAW_REPO',
    'MANIFEST_NAME',
    'RAW_DIR',
    'RUN_REPORT_NAME',
    'STATE_DIR',
    'UNCHANGED_MARKER',
    'AdapterState',
    'InflightBatch',
    'RawStore',
    'StoreError',
    'fingerprints_path',
    'inflight_operation',
    'inflight_path',
    'pending_fingerprints_path',
    'plan_raw_upload',
    'raw_prefix',
    'run_report_operation',
    'state_operations',
    'state_path',
]
