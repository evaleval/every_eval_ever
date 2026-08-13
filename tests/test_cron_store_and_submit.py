"""The cron's memory, its datastore commits, and the leftover pull requests.

All are places where a plausible-looking shortcut loses data: a state read
that swallows an auth error forgets every fingerprint and republishes the
whole history, a failed batch dropped from the accounting is republished
under fresh paths, and settling the retired flow's pull requests against
"the newest open one" hands records to somebody else's submission.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
)

from every_eval_ever.cron import store, submit

RUN_DATE = date(2026, 8, 10)
YESTERDAY = date(2026, 8, 9)
#: Where today's snapshot goes, and where yesterday's went. Both carry a run
#: token, because a date on its own no longer names one directory.
PREFIX = store.raw_prefix('hle', RUN_DATE, 'run-2-1')
PREVIOUS_PREFIX = store.raw_prefix('hle', YESTERDAY, 'run-1-1')
#: The account the cron publishes as, in these tests.
CRON_USER = 'eee-cron'


class FakeHub:
    """A Hub stand-in holding repository files in memory."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        *,
        sha: str = 'headsha',
        discussions: list[Any] | None = None,
        user: str = CRON_USER,
        private: bool = True,
        exists: bool = True,
        token_role: str | None = None,
    ) -> None:
        self.files = dict(files or {})
        self.sha = sha
        self.discussions = list(discussions or [])
        self.user = user
        self.private = private
        self.exists = exists
        self.token_role = token_role
        #: Repo ids this Hub refuses to resolve, whatever `exists` says.
        self.unreachable: set[str] = set()
        self.commits: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.download_error: Exception | None = None
        self.commit_error: Exception | None = None
        self.details_error: Exception | None = None
        self.list_files_error: Exception | None = None
        self.whoami_error: Exception | None = None
        self.repo_info_error: Exception | None = None
        self.discussion_queries: list[dict[str, Any]] = []
        self.next_pr = 41

    # -- reads ----------------------------------------------------------

    def dataset_info(self, repo_id, revision=None, **kwargs):
        return type('Info', (), {'sha': self.sha})()

    def repo_info(self, repo_id=None, **kwargs):
        if repo_id in self.unreachable:
            raise RepositoryNotFoundError(f'{repo_id} not found')
        if self.repo_info_error is not None:
            raise self.repo_info_error
        if not self.exists:
            raise RepositoryNotFoundError(f'{repo_id} not found')
        return type('Info', (), {'sha': self.sha, 'private': self.private})()

    def whoami(self):
        if self.whoami_error is not None:
            raise self.whoami_error
        identity = {'name': self.user}
        if self.token_role is not None:
            identity['auth'] = {'accessToken': {'role': self.token_role}}
        return identity

    def hf_hub_download(self, *, filename, **kwargs):
        if self.download_error is not None:
            raise self.download_error
        if filename not in self.files:
            raise EntryNotFoundError(f'{filename} not found')
        import tempfile

        handle = tempfile.NamedTemporaryFile(
            'w', suffix='.txt', delete=False, encoding='utf-8'
        )
        handle.write(self.files[filename])
        handle.close()
        return handle.name

    def get_repo_discussions(self, **kwargs):
        self.discussion_queries.append(kwargs)
        author = kwargs.get('author')
        return iter(
            [
                discussion
                for discussion in self.discussions
                if author is None or discussion.author == author
            ]
        )

    def get_discussion_details(self, *, discussion_num, **kwargs):
        if self.details_error is not None:
            raise self.details_error
        for discussion in self.discussions:
            if discussion.num == discussion_num:
                comment = type(
                    'Comment',
                    (),
                    {
                        'content': discussion.body,
                        'id': f'comment-{discussion.num}',
                    },
                )()
                return type(
                    'Details',
                    (),
                    {
                        'events': [comment],
                        'status': discussion.status,
                        'is_pull_request': discussion.is_pull_request,
                    },
                )()
        raise EntryNotFoundError(f'discussion {discussion_num} not found')

    def list_repo_files(self, repo_id=None, **kwargs):
        if self.list_files_error is not None:
            raise self.list_files_error
        return sorted(self.files)

    # -- writes ---------------------------------------------------------

    def create_repo(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True
        self.private = kwargs.get('private', False)
        return type('RepoUrl', (), {'url': f'https://x/{kwargs["repo_id"]}'})()

    def create_commit(self, **kwargs):
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append(kwargs)
        for operation in kwargs['operations']:
            payload = operation.path_or_fileobj
            if isinstance(payload, bytes):
                self.files[operation.path_in_repo] = payload.decode('utf-8')
            else:
                self.files[operation.path_in_repo] = Path(payload).read_text(
                    'utf-8'
                )
        if kwargs.get('create_pr'):
            self.next_pr += 1
            number = self.next_pr
            return type(
                'CommitInfo',
                (),
                {
                    'pr_num': number,
                    'pr_url': (
                        'https://huggingface.co/datasets/'
                        f'{kwargs["repo_id"]}/discussions/{number}'
                    ),
                    'pr_revision': f'refs/pr/{number}',
                    'oid': 'newsha',
                },
            )()
        return type('CommitInfo', (), {'oid': 'newsha'})()


class FakeDiscussion:
    def __init__(
        self,
        num: int,
        title: str,
        *,
        status: str = 'open',
        is_pull_request: bool = True,
        body: str = '',
        author: str = CRON_USER,
    ) -> None:
        self.num = num
        self.title = title
        self.status = status
        self.is_pull_request = is_pull_request
        self.body = body
        self.author = author
        self.url = (
            'https://huggingface.co/datasets/evaleval/EEE_datastore/'
            f'discussions/{num}'
        )
        self.git_reference = f'refs/pr/{num}' if is_pull_request else None


# --- the store is private, and the code checks ---------------------------


def test_a_private_store_that_already_exists_is_left_alone() -> None:
    hub = FakeHub(private=True)

    store.RawStore(hub).ensure_private()

    assert hub.created == []


def test_a_public_store_stops_the_run_before_anything_is_written() -> None:
    """Raw payloads are kept as evidence, not republished."""
    hub = FakeHub(private=False)

    with pytest.raises(store.StoreError, match='is public'):
        store.RawStore(hub).ensure_private()

    assert hub.commits == []
    # Visibility is somebody else's decision to make, and undoing it does not
    # un-publish what was readable in the meantime.
    assert hub.created == []


def test_a_missing_store_is_created_private_and_read_back() -> None:
    hub = FakeHub(exists=False)

    store.RawStore(hub, repo_id='evaleval/EEE_raw').ensure_private()

    assert hub.created[0]['private'] is True
    assert hub.created[0]['repo_type'] == 'dataset'
    assert hub.private is True


def test_a_store_that_reads_back_public_after_creation_is_refused() -> None:
    """The create call reporting success is not the same as it being private."""
    hub = FakeHub(exists=False)
    real_create_repo = hub.create_repo

    def create_public(**kwargs):
        result = real_create_repo(**kwargs)
        hub.private = False
        return result

    hub.create_repo = create_public

    with pytest.raises(store.StoreError, match='reads back as public'):
        store.RawStore(hub).ensure_private()


@pytest.mark.parametrize(
    'error',
    [
        RuntimeError('500 Internal Server Error'),
        ConnectionError('network is unreachable'),
    ],
)
def test_an_unreadable_store_is_not_treated_as_a_missing_one(error) -> None:
    """Creating on a 500 is how a public dataset gets blessed as private."""
    hub = FakeHub(private=False)
    hub.repo_info_error = error

    with pytest.raises(store.StoreError, match='could not check'):
        store.RawStore(hub).ensure_private()

    assert hub.created == []


def test_a_commit_re_checks_visibility_rather_than_trusting_startup() -> None:
    """Runs are unattended and daily; the adapter ran in between."""
    hub = FakeHub(private=True)
    raw_store = store.RawStore(hub)
    raw_store.ensure_private()

    hub.private = False

    with pytest.raises(store.StoreError, match='is public'):
        raw_store.commit(
            store.state_operations(store.AdapterState(adapter='hle')),
            message='state',
            parent_commit='headsha',
        )

    assert hub.commits == []


def test_a_commit_whose_visibility_cannot_be_confirmed_is_refused() -> None:
    hub = FakeHub()
    hub.repo_info_error = RuntimeError('503')

    with pytest.raises(store.StoreError, match='could not confirm'):
        store.RawStore(hub).commit(
            store.state_operations(store.AdapterState(adapter='hle')),
            message='state',
            parent_commit='headsha',
        )

    assert hub.commits == []


# --- state ---------------------------------------------------------------


def test_a_cold_start_is_an_empty_ledger_that_says_so() -> None:
    state = store.RawStore(FakeHub()).read_state('hle')

    assert not state.exists
    assert state.fingerprints == set()
    assert state.pull_request_number is None
    assert state.parent_commit == 'headsha'


def test_state_round_trips_through_the_store() -> None:
    hub = FakeHub()
    raw_store = store.RawStore(hub)
    state = store.AdapterState(
        adapter='hle',
        pull_request_number=7,
        pull_request_url='https://example/7',
        last_run_date='2026-08-09',
        last_raw_date='2026-08-09',
        last_raw_prefix='raw/hle/2026-08-09/run-1-1',
        last_status='completed',
        fingerprints={'bbb', 'aaa'},
        pending_fingerprints={'ccc'},
    )

    raw_store.commit(
        store.state_operations(state),
        message='state',
        parent_commit='headsha',
    )
    reloaded = raw_store.read_state('hle')

    assert reloaded.exists
    assert reloaded.pull_request_number == 7
    assert reloaded.last_status == 'completed'
    assert reloaded.last_raw_prefix == 'raw/hle/2026-08-09/run-1-1'
    assert reloaded.fingerprints == {'aaa', 'bbb'}
    # Pending fingerprints survive apart from the durable ones, because the
    # two are settled differently when their pull request closes.
    assert reloaded.pending_fingerprints == {'ccc'}
    assert reloaded.known_fingerprints == {'aaa', 'bbb', 'ccc'}
    # Sorted, one per line, so a diff shows what actually changed.
    assert hub.files['state/hle.fingerprints'] == 'aaa\nbbb\n'
    assert hub.files['state/hle.pending'] == 'ccc\n'


def test_nothing_in_flight_reads_back_as_an_empty_batch() -> None:
    assert store.RawStore(FakeHub()).read_inflight('hle').records == []


def test_an_in_flight_batch_round_trips_through_the_store() -> None:
    hub = FakeHub()
    raw_store = store.RawStore(hub)
    batch = store.InflightBatch(
        adapter='hle',
        run_date='2026-08-10',
        run_token='run-2-1',
        pull_request_number=12,
        records=[{'fingerprint': 'aaa', 'paths': ['data/hle/a.json']}],
    )

    raw_store.commit(
        [store.inflight_operation(batch)],
        message='in flight',
        parent_commit='headsha',
    )
    reloaded = raw_store.read_inflight('hle')

    assert reloaded.pull_request_number == 12
    assert reloaded.run_token == 'run-2-1'
    assert reloaded.records == batch.records
    assert reloaded.paths == ['data/hle/a.json']


def test_an_emptied_in_flight_file_is_written_rather_than_deleted() -> None:
    """So every run makes the same commit, and "nothing in flight" is a fact
    the file states instead of one inferred from its absence."""
    hub = FakeHub()

    store.RawStore(hub).commit(
        [store.inflight_operation(store.InflightBatch(adapter='hle'))],
        message='settled',
        parent_commit='headsha',
    )

    assert 'state/hle.inflight' in hub.files
    assert json.loads(hub.files['state/hle.inflight'])['records'] == []


@pytest.mark.parametrize(
    'body',
    [
        'not json at all',
        json.dumps([1, 2]),
        json.dumps({'records': [{'paths': ['a.json']}]}),
        json.dumps({'records': [{'fingerprint': 'aaa'}]}),
    ],
)
def test_an_unreadable_in_flight_file_is_fatal(body: str) -> None:
    """Reading it as empty would bury the records it exists to account for."""
    hub = FakeHub({'state/hle.inflight': body})

    with pytest.raises(store.StoreError):
        store.RawStore(hub).read_inflight('hle')


@pytest.mark.parametrize(
    'error',
    [
        PermissionError('401 unauthorized'),
        ConnectionError('network is unreachable'),
    ],
)
def test_a_state_read_that_is_not_a_missing_file_is_fatal(error) -> None:
    """A silently empty ledger would republish the entire history."""
    hub = FakeHub()
    hub.download_error = error

    with pytest.raises(store.StoreError, match='could not read'):
        store.RawStore(hub).read_state('hle')


def test_malformed_state_is_fatal_rather_than_ignored() -> None:
    hub = FakeHub({'state/hle.json': 'not json'})

    with pytest.raises(store.StoreError, match='not valid JSON'):
        store.RawStore(hub).read_state('hle')


def test_a_fingerprint_file_alone_still_counts_as_existing_state() -> None:
    hub = FakeHub({'state/hle.fingerprints': 'aaa\n\nbbb\n'})

    state = store.RawStore(hub).read_state('hle')

    assert state.exists
    assert state.fingerprints == {'aaa', 'bbb'}


def test_state_writes_carry_the_commit_they_were_read_at() -> None:
    """So a concurrent run 409s instead of overwriting a newer ledger."""
    hub = FakeHub()
    raw_store = store.RawStore(hub)
    state = raw_store.read_state('hle')
    state.fingerprints.add('aaa')

    raw_store.commit(
        store.state_operations(state),
        message='state',
        parent_commit=state.parent_commit,
    )

    assert hub.commits[0]['parent_commit'] == 'headsha'


def test_a_rejected_write_is_reported_not_swallowed() -> None:
    hub = FakeHub()
    hub.commit_error = RuntimeError('412 precondition failed')

    with pytest.raises(store.StoreError, match='could not write'):
        store.RawStore(hub).commit(
            store.state_operations(store.AdapterState(adapter='hle')),
            message='state',
            parent_commit='headsha',
        )


def test_an_unresolvable_store_revision_is_fatal() -> None:
    hub = FakeHub()
    hub.dataset_info = _raise(RuntimeError('no such repo'))

    with pytest.raises(store.StoreError, match='could not resolve'):
        store.RawStore(hub).read_state('hle')


def test_the_store_refuses_an_empty_repository_id() -> None:
    with pytest.raises(store.StoreError, match='repository id is required'):
        store.RawStore(FakeHub(), repo_id='')


def _raise(error: Exception):
    def go(*args, **kwargs):
        raise error

    return go


# --- raw snapshots -------------------------------------------------------


def write_capture(raw_dir: Path, entries: list[dict], payloads: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / store.MANIFEST_NAME).write_text(
        ''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries),
        encoding='utf-8',
    )
    for name, body in payloads.items():
        (raw_dir / name).write_bytes(body)


def test_a_run_with_no_captures_uploads_nothing(tmp_path) -> None:
    operations, manifest = store.plan_raw_upload(
        tmp_path / 'raw', prefix=PREFIX
    )

    assert operations == []
    assert manifest == []


def test_payloads_and_a_manifest_are_uploaded_under_the_run_date(
    tmp_path,
) -> None:
    raw_dir = tmp_path / 'raw'
    write_capture(
        raw_dir,
        [
            {
                'kind': 'payload',
                'sha256': 'aaa',
                'path': 'aaa.json',
                'url': 'https://x',
            },
            {
                'kind': 'pointer',
                'pointer_kind': 'hf_dataset',
                'reference': 'a/b',
            },
        ],
        {'aaa.json': b'{}'},
    )

    operations, manifest = store.plan_raw_upload(raw_dir, prefix=PREFIX)

    paths = [operation.path_in_repo for operation in operations]
    assert paths == [
        f'{PREFIX}/aaa.json',
        f'{PREFIX}/manifest.jsonl',
    ]
    assert len(manifest) == 2


def test_an_unchanged_payload_is_referenced_not_re_uploaded(
    tmp_path,
) -> None:
    """Snapshotting a leaderboard that rarely changes must stay cheap."""
    raw_dir = tmp_path / 'raw'
    write_capture(
        raw_dir,
        [
            {
                'kind': 'payload',
                'sha256': 'aaa',
                'path': 'aaa.json',
                'url': 'https://x',
            }
        ],
        {'aaa.json': b'{}'},
    )
    previous = [
        {'kind': 'payload', 'sha256': 'aaa', 'path': 'aaa.json'},
    ]

    operations, manifest = store.plan_raw_upload(
        raw_dir,
        prefix=PREFIX,
        previous_manifest=previous,
        previous_prefix=PREVIOUS_PREFIX,
    )

    assert [operation.path_in_repo for operation in operations] == [
        f'{PREFIX}/manifest.jsonl'
    ]
    assert manifest[0][store.UNCHANGED_MARKER] == (
        f'{PREVIOUS_PREFIX}/aaa.json'
    )


def test_two_adapters_from_one_head_both_record_their_state() -> None:
    """The loser of the race has already published; it must still be recorded.

    Every job in the daily matrix reads the same raw-store head. Dropping the
    second one's state commit would leave the records it just put in the
    datastore with no fingerprints, so the next run would publish them again.
    """
    hub = FakeHub(sha='headsha')
    first = store.RawStore(hub)
    second = store.RawStore(hub)
    first_state = first.read_state('hle')
    second_state = second.read_state('mt_bench')
    assert first_state.parent_commit == second_state.parent_commit

    real_create_commit = hub.create_commit

    def reject_a_stale_parent(**kwargs):
        if kwargs.get('parent_commit') != hub.sha:
            raise RuntimeError('412 Precondition Failed')
        result = real_create_commit(**kwargs)
        hub.sha = f'{hub.sha}-moved'
        return result

    hub.create_commit = reject_a_stale_parent

    first_state.fingerprints.add('a')
    first.commit(
        store.state_operations(first_state),
        message='hle',
        parent_commit=first_state.parent_commit,
    )
    second_state.fingerprints.add('b')
    second.commit(
        store.state_operations(second_state),
        message='mt_bench',
        parent_commit=second_state.parent_commit,
    )

    assert hub.files['state/hle.fingerprints'].split() == ['a']
    assert hub.files['state/mt_bench.fingerprints'].split() == ['b']


def test_a_rejected_commit_that_is_not_a_race_still_fails() -> None:
    """A permission or transport error must not be retried into silence."""
    hub = FakeHub(sha='headsha')
    raw_store = store.RawStore(hub)
    state = raw_store.read_state('hle')
    hub.commit_error = RuntimeError('403 Forbidden')

    with pytest.raises(store.StoreError, match='403'):
        raw_store.commit(
            store.state_operations(state),
            message='hle',
            parent_commit=state.parent_commit,
        )


def test_a_reference_survives_a_run_of_unchanged_days(tmp_path) -> None:
    """Day three must point at day one, the only day that stored the bytes.

    Pointing at day two instead names a file day two never wrote, because
    day two referenced day one rather than storing a second copy. The
    manifest would then describe a snapshot that cannot be fetched.
    """
    raw_dir = tmp_path / 'raw'
    write_capture(
        raw_dir,
        [
            {
                'kind': 'payload',
                'sha256': 'aaa',
                'path': 'aaa.json',
                'url': 'https://x',
            }
        ],
        {'aaa.json': b'{}'},
    )
    day_one = 'raw/hle/2026-08-08/aaa.json'
    day_two = [
        {
            'kind': 'payload',
            'sha256': 'aaa',
            'path': 'aaa.json',
            store.UNCHANGED_MARKER: day_one,
        }
    ]

    operations, manifest = store.plan_raw_upload(
        raw_dir,
        prefix=PREFIX,
        previous_manifest=day_two,
        previous_prefix=PREVIOUS_PREFIX,
    )

    assert [operation.path_in_repo for operation in operations] == [
        f'{PREFIX}/manifest.jsonl'
    ]
    assert manifest[0][store.UNCHANGED_MARKER] == day_one


def test_a_changed_payload_is_uploaded_even_when_one_is_unchanged(
    tmp_path,
) -> None:
    raw_dir = tmp_path / 'raw'
    write_capture(
        raw_dir,
        [
            {'kind': 'payload', 'sha256': 'aaa', 'path': 'aaa.json'},
            {'kind': 'payload', 'sha256': 'bbb', 'path': 'bbb.json'},
        ],
        {'aaa.json': b'{}', 'bbb.json': b'[]'},
    )

    operations, _ = store.plan_raw_upload(
        raw_dir,
        prefix=PREFIX,
        previous_manifest=[
            {'kind': 'payload', 'sha256': 'aaa', 'path': 'aaa.json'}
        ],
        previous_prefix=PREVIOUS_PREFIX,
    )

    assert [operation.path_in_repo for operation in operations] == [
        f'{PREFIX}/bbb.json',
        f'{PREFIX}/manifest.jsonl',
    ]


def test_the_run_report_lands_beside_the_snapshot() -> None:
    operation = store.run_report_operation(
        {'status': 'completed'}, prefix=PREFIX
    )

    assert operation.path_in_repo == f'{PREFIX}/run.json'
    assert b'completed' in operation.path_or_fileobj


# --- pull requests the retired flow left behind ---------------------------


def submitter(
    discussions: list[Any],
) -> tuple[submit.DatastoreSubmitter, FakeHub]:
    hub = FakeHub(discussions=discussions)
    return submit.DatastoreSubmitter(hub), hub


def cron_pr(num: int, adapter: str = 'hle', **kwargs: Any) -> FakeDiscussion:
    """A pull request the retired flow opened, carrying its marker."""
    return FakeDiscussion(
        num,
        f'[Submission] cron: {adapter} (automated ingestion)',
        body=(
            f'Automated daily ingestion for the `{adapter}` adapter.\n\n'
            f'{submit.marker(adapter)}'
        ),
        **kwargs,
    )


def test_a_remembered_open_pull_request_is_reused() -> None:
    sub, _ = submitter([cron_pr(12)])

    found = sub.resolve_known('hle', 12)

    assert found.number == 12
    assert found.revision == 'refs/pr/12'


def test_a_merged_pull_request_is_not_reused() -> None:
    sub, _ = submitter([cron_pr(12, status='merged')])

    assert sub.resolve_known('hle', 12) is None


def test_a_discussion_that_is_not_a_pull_request_is_not_reused() -> None:
    sub, _ = submitter([cron_pr(12, is_pull_request=False)])

    assert sub.resolve_known('hle', 12) is None


def test_a_repurposed_pull_request_is_not_reused() -> None:
    """The number alone is not identity; a human may have repurposed it."""
    sub, _ = submitter([FakeDiscussion(12, 'Add my own eval by hand')])

    assert sub.resolve_known('hle', 12) is None


def test_a_title_that_looks_like_ours_is_not_enough() -> None:
    """Ownership is the body marker, and a title is display metadata.

    Anyone can open a pull request called ``cron: hle`` by hand. Publishing
    a scrape into it because the title matched would put automated records in
    a stranger's submission.
    """
    sub, _ = submitter(
        [
            FakeDiscussion(
                12,
                '[Submission] cron: hle (automated ingestion)',
                body='no marker',
            )
        ]
    )

    assert sub.resolve_known('hle', 12) is None
    assert sub.find_by_marker('hle') is None


def test_a_marked_pull_request_survives_a_title_edit() -> None:
    """A reviewer renaming ours must not strand it.

    Losing it would open a second pull request for the same adapter and
    republish everything the first one already holds.
    """
    sub, _ = submitter([cron_pr(12)])
    sub._open_pull_requests()[0].title = 'WIP: please hold'

    assert sub.resolve_known('hle', 12).number == 12
    assert sub.find_by_marker('hle').number == 12


def test_a_pull_request_is_found_by_marker_on_a_cold_start() -> None:
    sub, _ = submitter(
        [FakeDiscussion(9, 'Someone else adding data'), cron_pr(12)]
    )

    found = sub.find_by_marker('hle')

    assert found.number == 12


def test_another_adapters_pull_request_is_never_claimed() -> None:
    sub, _ = submitter([cron_pr(12, adapter='mmlu_pro')])

    assert sub.find_by_marker('hle') is None


def test_an_unrelated_open_pull_request_is_never_claimed() -> None:
    sub, _ = submitter(
        [FakeDiscussion(3, '[Submission] Add SWE-bench results')]
    )

    assert sub.find_by_marker('hle') is None


def test_a_read_only_token_is_refused_before_any_work(tmp_path) -> None:
    """It would pass every read and die on the first commit, an hour later."""
    hub = FakeHub(token_role='read')

    with pytest.raises(submit.SubmissionError, match='read-only'):
        submit.DatastoreSubmitter(hub).ensure_writable()


def test_a_token_whose_role_the_hub_does_not_report_is_allowed() -> None:
    """Only a commit proves a fine-grained token's scopes."""
    submit.DatastoreSubmitter(FakeHub(token_role=None)).ensure_writable()
    submit.DatastoreSubmitter(FakeHub(token_role='write')).ensure_writable()


def test_an_unreachable_datastore_is_reported_before_any_work() -> None:
    hub = FakeHub()
    hub.unreachable.add('evaleval/EEE_datastore')

    with pytest.raises(submit.SubmissionError, match='could not reach'):
        submit.DatastoreSubmitter(hub).ensure_writable()


def test_the_identity_is_asked_for_once(tmp_path) -> None:
    calls = []
    hub = FakeHub()
    real_whoami = hub.whoami
    hub.whoami = lambda: (calls.append(1), real_whoami())[1]
    sub = submit.DatastoreSubmitter(hub)

    sub.ensure_writable()
    sub.find_by_marker('hle')

    assert len(calls) == 1


def test_a_marker_from_another_account_is_never_claimed() -> None:
    """The datastore is public, so the marker is not proof of ownership.

    Anyone can open a pull request whose first comment carries
    ``eee-cron-adapter: hle``. Adopting it would commit records onto a branch
    and a description a stranger controls.
    """
    sub, _ = submitter([cron_pr(12, author='someone-else')])

    assert sub.resolve_known('hle', 12) is None
    assert sub.find_by_marker('hle') is None


def test_the_lookup_asks_the_hub_for_this_accounts_pull_requests() -> None:
    sub, hub = submitter([cron_pr(12)])

    sub.find_by_marker('hle')

    assert hub.discussion_queries[0]['author'] == CRON_USER


def test_a_token_with_no_resolvable_account_stops_the_run() -> None:
    """An unknown author would filter nothing, which is the wrong default."""
    sub, hub = submitter([cron_pr(12)])
    hub.whoami_error = RuntimeError('401 unauthorized')

    with pytest.raises(submit.SubmissionError, match='publishes as'):
        sub.find_by_marker('hle')


def test_an_unreadable_pull_request_body_stops_the_run() -> None:
    """Treating it as unowned would quietly open a duplicate."""
    sub, hub = submitter([cron_pr(12)])
    hub.details_error = RuntimeError('403')

    with pytest.raises(submit.SubmissionError, match='could not read'):
        sub.find_by_marker('hle')


def test_two_matching_pull_requests_stop_the_run() -> None:
    sub, _ = submitter([cron_pr(12), cron_pr(14)])

    with pytest.raises(submit.AmbiguousPullRequestError, match='12, 14'):
        sub.find_by_marker('hle')


def test_a_lookup_failure_is_reported_not_widened() -> None:
    hub = FakeHub()
    hub.get_repo_discussions = _raise(RuntimeError('403'))

    with pytest.raises(submit.SubmissionError, match='could not list'):
        submit.DatastoreSubmitter(hub).find_by_marker('hle')


# --- publishing ------------------------------------------------------------


def _upload_tree(tmp_path, count: int) -> Path:
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    for index in range(count):
        (upload / f'{index}.json').write_text('{}', encoding='utf-8')
    return tmp_path / 'upload'


def test_records_are_committed_to_the_default_branch(tmp_path) -> None:
    """No pull request, no ref: a passing run's records go straight in."""
    tree = _upload_tree(tmp_path, 2)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub)
    description = submit.commit_description(
        'hle',
        coverage_line='2 record(s) produced -> 2 uploaded',
        run_date='2026-08-10',
        status='completed',
    )

    submission = sub.publish(
        operations=submit.upload_operations(tree),
        description=description,
        message='cron: hle 2026-08-10 (2 record(s))',
    )

    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert 'create_pr' not in commit
    assert 'revision' not in commit
    assert commit['commit_message'] == 'cron: hle 2026-08-10 (2 record(s))'
    # The provenance a reviewer used to read in a pull request body is on
    # the commit itself.
    assert commit['commit_description'] == description
    assert len(submission.committed_paths) == 2


def test_a_large_batch_is_split_and_numbered(tmp_path) -> None:
    """A single huge commit can 504 while still landing server-side."""
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    submission = sub.publish(
        operations=submit.upload_operations(tree),
        description='body',
        message='hle 2026-08-10',
    )

    messages = [commit['commit_message'] for commit in hub.commits]
    assert messages == [
        'hle 2026-08-10 (1/3)',
        'hle 2026-08-10 (2/3)',
        'hle 2026-08-10 (3/3)',
    ]
    assert [len(commit['operations']) for commit in hub.commits] == [3, 3, 1]
    # Every commit of the run describes it, not just the first.
    assert all(
        commit['commit_description'] == 'body' for commit in hub.commits
    )
    assert len(submission.committed_paths) == 7


def test_a_record_and_its_sidecar_are_never_split_across_commits(
    tmp_path,
) -> None:
    """A commit is atomic, so a whole record in one commit lands or does not.

    Split across two, a failure on the second leaves the aggregate public,
    unrecordable (its companion never arrived), and republished under a fresh
    UUID next run, with the abandoned half still in the datastore naming a
    sidecar that does not exist.
    """
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    for index in range(4):
        (upload / f'{index}.json').write_text('{}', encoding='utf-8')
        (upload / f'{index}_samples.jsonl').write_text('{}\n', encoding='utf-8')
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    sub.publish(
        operations=submit.upload_operations(tmp_path / 'upload'),
        description='body',
        message='hle 2026-08-10',
    )

    for commit in hub.commits:
        stems = {
            operation.path_in_repo.removesuffix('_samples.jsonl').removesuffix(
                '.json'
            )
            for operation in commit['operations']
        }
        # Every stem in this commit brought both of its files with it.
        assert len(commit['operations']) == 2 * len(stems)
    # Two files per record and a cap of three means one record per commit.
    assert len(hub.commits) == 4


def test_a_failure_before_anything_landed_reports_nothing_committed(
    tmp_path,
) -> None:
    tree = _upload_tree(tmp_path, 2)
    hub = FakeHub()
    hub.commit_error = RuntimeError('502 Bad Gateway')
    sub = submit.DatastoreSubmitter(hub)

    with pytest.raises(submit.PartialSubmissionError) as caught:
        sub.publish(
            operations=submit.upload_operations(tree),
            description='body',
            message='hle 2026-08-10',
        )

    assert caught.value.committed_paths == ()
    assert caught.value.unresolved_paths == ()


def test_a_failure_after_the_first_batch_reports_what_landed(
    tmp_path,
) -> None:
    """A retry must not republish records that already reached the Hub.

    They would land under fresh UUID paths, so the datastore would hold
    the same evaluation twice with no way to tell which copy to keep.
    """
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    real_create_commit = hub.create_commit

    def fail_on_second(**kwargs):
        if len(hub.commits) >= 1:
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = fail_on_second

    with pytest.raises(submit.PartialSubmissionError) as caught:
        sub.publish(
            operations=submit.upload_operations(tree),
            description='body',
            message='hle 2026-08-10',
        )

    assert len(caught.value.committed_paths) == 3
    assert all(
        path.startswith('data/hle/') for path in caught.value.committed_paths
    )
    assert caught.value.unresolved_paths == ()


def test_a_batch_that_landed_despite_the_error_does_not_stop_the_upload(
    tmp_path,
) -> None:
    """The ambiguous timeout: the Hub accepted the commit, the client saw an
    error. The datastore is the arbiter of what actually landed, and a batch
    that is in it is a success, so the upload carries on. Stopping instead
    would end a run whose final batch landed this way as a failure with
    every record accounted for, which no retry ever completes."""
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    real_create_commit = hub.create_commit

    def land_then_time_out(**kwargs):
        if len(hub.commits) >= 1:
            real_create_commit(**kwargs)
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = land_then_time_out

    submission = sub.publish(
        operations=submit.upload_operations(tree),
        description='body',
        message='hle 2026-08-10',
    )

    # The clean first batch and the two ambiguous ones all count; nothing is
    # left for a retry.
    assert len(submission.committed_paths) == 7


def test_an_unanswerable_batch_is_reported_as_unresolved(tmp_path) -> None:
    """When the datastore cannot be read either, the batch is not guessed
    onto the ledger; the error says the datastore needs a look before a
    retry."""
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    real_create_commit = hub.create_commit

    def fail_on_second(**kwargs):
        if len(hub.commits) >= 1:
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = fail_on_second
    hub.list_files_error = ConnectionError('network is unreachable')

    with pytest.raises(submit.PartialSubmissionError) as caught:
        sub.publish(
            operations=submit.upload_operations(tree),
            description='body',
            message='hle 2026-08-10',
        )

    assert len(caught.value.committed_paths) == 3
    # The failing batch is carried as unresolved rather than dropped, so the
    # caller keeps it in flight instead of re-uploading it blind.
    assert len(caught.value.unresolved_paths) == 3
    assert not (
        set(caught.value.unresolved_paths) & set(caught.value.committed_paths)
    )
    assert (
        'inspect https://huggingface.co/datasets/evaleval/EEE_datastore '
        'before re-running'
    ) in str(caught.value)


# --- what happened to the last pull request -------------------------------


def test_the_status_of_an_open_pull_request_is_open() -> None:
    sub, _ = submitter([cron_pr(12)])

    assert sub.pull_request_status(12) == 'open'


@pytest.mark.parametrize('status', ['merged', 'closed'])
def test_a_finished_pull_requests_status_is_reported(status) -> None:
    sub, _ = submitter([cron_pr(12, status=status)])

    assert sub.pull_request_status(12) == status


def test_a_draft_pull_request_counts_as_open() -> None:
    sub, _ = submitter([cron_pr(12, status='draft')])

    assert sub.pull_request_status(12) == 'open'


def test_an_unreadable_pull_request_status_stops_the_run() -> None:
    """Both wrong guesses lose data: "merged" buries rejected records for
    good, "closed" republishes accepted ones."""
    sub, hub = submitter([cron_pr(12)])
    hub.details_error = ConnectionError('boom')

    with pytest.raises(submit.SubmissionError, match='settle its records'):
        sub.pull_request_status(12)


def test_a_number_that_is_not_a_pull_request_stops_the_run() -> None:
    sub, _ = submitter([cron_pr(12, is_pull_request=False)])

    with pytest.raises(submit.SubmissionError, match='not a pull request'):
        sub.pull_request_status(12)


def test_upload_operations_use_repository_relative_posix_paths(
    tmp_path,
) -> None:
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    (upload / 'a.json').write_text('{}', encoding='utf-8')

    operations = submit.upload_operations(tmp_path / 'upload')

    assert [operation.path_in_repo for operation in operations] == [
        'data/hle/org/model/a.json'
    ]


def test_the_description_carries_coverage_and_provenance() -> None:
    description = submit.commit_description(
        'hle',
        coverage_line='12 source row(s) -> 10 record(s) produced, 2 dropped',
        run_date='2026-08-10',
        status='partial',
        run_url='https://ci.example/run/7',
        raw_reference='raw/hle/2026-08-10',
        notes=['raw capture degraded'],
    )

    assert '2 dropped' in description
    assert 'https://ci.example/run/7' in description
    assert 'raw/hle/2026-08-10' in description
    assert 'raw capture degraded' in description
    assert 'type_of_addition' in description


def test_the_submitter_refuses_an_empty_repository_id() -> None:
    with pytest.raises(submit.SubmissionError, match='repository id'):
        submit.DatastoreSubmitter(FakeHub(), repo_id='')
