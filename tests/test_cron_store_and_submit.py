"""The cron's memory and its one-pull-request-per-adapter rule.

Both are places where a plausible-looking shortcut loses data: a state read
that swallows an auth error forgets every fingerprint and republishes the
whole history, and a pull-request lookup that falls back to "the newest open
one" pushes a scrape into somebody else's submission.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import EntryNotFoundError

from every_eval_ever.cron import store, submit

RUN_DATE = date(2026, 8, 10)
YESTERDAY = date(2026, 8, 9)


class FakeHub:
    """A Hub stand-in holding repository files in memory."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        *,
        sha: str = 'headsha',
        discussions: list[Any] | None = None,
    ) -> None:
        self.files = dict(files or {})
        self.sha = sha
        self.discussions = list(discussions or [])
        self.commits: list[dict[str, Any]] = []
        self.download_error: Exception | None = None
        self.commit_error: Exception | None = None
        self.details_error: Exception | None = None
        self.next_pr = 41

    # -- reads ----------------------------------------------------------

    def dataset_info(self, repo_id, revision=None, **kwargs):
        return type('Info', (), {'sha': self.sha})()

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
        return iter(self.discussions)

    def get_discussion_details(self, *, discussion_num, **kwargs):
        if self.details_error is not None:
            raise self.details_error
        for discussion in self.discussions:
            if discussion.num == discussion_num:
                comment = type('Comment', (), {'content': discussion.body})()
                return type('Details', (), {'events': [comment]})()
        raise EntryNotFoundError(f'discussion {discussion_num} not found')

    # -- writes ---------------------------------------------------------

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
    ) -> None:
        self.num = num
        self.title = title
        self.status = status
        self.is_pull_request = is_pull_request
        self.body = body
        self.url = (
            'https://huggingface.co/datasets/evaleval/EEE_datastore/'
            f'discussions/{num}'
        )
        self.git_reference = f'refs/pr/{num}' if is_pull_request else None


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
        last_status='completed',
        fingerprints={'bbb', 'aaa'},
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
    assert reloaded.fingerprints == {'aaa', 'bbb'}
    # Sorted, one per line, so a diff shows what actually changed.
    assert hub.files['state/hle.fingerprints'] == 'aaa\nbbb\n'


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
        tmp_path / 'raw', adapter='hle', run_date=RUN_DATE
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

    operations, manifest = store.plan_raw_upload(
        raw_dir, adapter='hle', run_date=RUN_DATE
    )

    paths = [operation.path_in_repo for operation in operations]
    assert paths == [
        'raw/hle/2026-08-10/aaa.json',
        'raw/hle/2026-08-10/manifest.jsonl',
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
        adapter='hle',
        run_date=RUN_DATE,
        previous_manifest=previous,
        previous_date=YESTERDAY.isoformat(),
    )

    assert [operation.path_in_repo for operation in operations] == [
        'raw/hle/2026-08-10/manifest.jsonl'
    ]
    assert manifest[0][store.UNCHANGED_MARKER] == (
        'raw/hle/2026-08-09/aaa.json'
    )


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
        adapter='hle',
        run_date=RUN_DATE,
        previous_manifest=[
            {'kind': 'payload', 'sha256': 'aaa', 'path': 'aaa.json'}
        ],
        previous_date=YESTERDAY.isoformat(),
    )

    assert [operation.path_in_repo for operation in operations] == [
        'raw/hle/2026-08-10/bbb.json',
        'raw/hle/2026-08-10/manifest.jsonl',
    ]


def test_the_run_report_lands_beside_the_snapshot() -> None:
    operation = store.run_report_operation(
        {'status': 'completed'}, adapter='hle', run_date=RUN_DATE
    )

    assert operation.path_in_repo == 'raw/hle/2026-08-10/run.json'
    assert b'completed' in operation.path_or_fileobj


# --- pull requests -------------------------------------------------------


def submitter(
    discussions: list[Any],
) -> tuple[submit.DatastoreSubmitter, FakeHub]:
    hub = FakeHub(discussions=discussions)
    return submit.DatastoreSubmitter(hub), hub


def cron_pr(num: int, adapter: str = 'hle', **kwargs: Any) -> FakeDiscussion:
    """An open pull request carrying the cron's ownership marker."""
    return FakeDiscussion(
        num,
        submit.pull_request_title(adapter),
        body=submit.pull_request_description(
            adapter,
            coverage_line='1 record',
            run_date='2026-08-10',
            status='completed',
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
        [FakeDiscussion(12, submit.pull_request_title('hle'), body='no marker')]
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


def test_opening_a_pull_request_returns_its_number_and_ref(tmp_path) -> None:
    sub, hub = submitter([])
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    (upload / 'a.json').write_text('{}', encoding='utf-8')

    opened = sub.open_pull_request(
        'hle',
        operations=submit.upload_operations(tmp_path / 'upload'),
        description=submit.pull_request_description(
            'hle',
            coverage_line='1 record(s) produced -> 1 uploaded',
            run_date='2026-08-10',
            status='completed',
        ),
    )

    assert opened.number == 42
    assert opened.revision == 'refs/pr/42'
    assert hub.commits[0]['create_pr'] is True
    assert submit.marker('hle') in hub.commits[0]['commit_description']


def test_uploads_target_the_pull_request_ref(tmp_path) -> None:
    sub, hub = submitter([])
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    (upload / 'a.json').write_text('{}', encoding='utf-8')

    sub.upload(
        submit.PullRequest(12, 'https://x/12', 'refs/pr/12', 'cron: hle'),
        operations=submit.upload_operations(tmp_path / 'upload'),
        message='hle 2026-08-10',
    )

    assert hub.commits[0]['revision'] == 'refs/pr/12'
    assert 'create_pr' not in hub.commits[0]


def test_a_large_batch_is_split_and_numbered(tmp_path) -> None:
    """A single huge commit can 504 while still landing server-side."""
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    for index in range(7):
        (upload / f'{index}.json').write_text('{}', encoding='utf-8')
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    sub.upload(
        submit.PullRequest(12, 'https://x/12', 'refs/pr/12', 'cron: hle'),
        operations=submit.upload_operations(tmp_path / 'upload'),
        message='hle 2026-08-10',
    )

    messages = [commit['commit_message'] for commit in hub.commits]
    assert messages == [
        'hle 2026-08-10 (1/3)',
        'hle 2026-08-10 (2/3)',
        'hle 2026-08-10 (3/3)',
    ]


def _upload_tree(tmp_path, count: int) -> Path:
    upload = tmp_path / 'upload' / 'data' / 'hle' / 'org' / 'model'
    upload.mkdir(parents=True)
    for index in range(count):
        (upload / f'{index}.json').write_text('{}', encoding='utf-8')
    return tmp_path / 'upload'


def test_a_cold_start_is_batched_like_any_other_upload(tmp_path) -> None:
    """The first run is the biggest one an adapter ever does.

    Opening the pull request with every operation in one commit aimed the
    ambiguous timeout that batching exists to avoid at exactly the run most
    likely to trip it.
    """
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)

    submission = sub.publish(
        'hle',
        pull_request=None,
        operations=submit.upload_operations(tree),
        description=submit.pull_request_description(
            'hle',
            coverage_line='7 record(s)',
            run_date='2026-08-10',
            status='completed',
        ),
        message='hle 2026-08-10',
    )

    assert [len(commit['operations']) for commit in hub.commits] == [3, 3, 1]
    assert hub.commits[0]['create_pr'] is True
    assert all('create_pr' not in commit for commit in hub.commits[1:])
    assert all(
        commit['revision'] == 'refs/pr/42' for commit in hub.commits[1:]
    )
    assert len(submission.committed_paths) == 7


def test_a_failure_after_the_first_batch_reports_what_landed(
    tmp_path,
) -> None:
    """A retry must not republish records that already reached the Hub.

    They would land under fresh UUID paths, so the pull request would hold
    the same evaluation twice with no way to tell which copy to keep.
    """
    tree = _upload_tree(tmp_path, 7)
    hub = FakeHub()
    sub = submit.DatastoreSubmitter(hub, batch_size=3)
    pull_request = submit.PullRequest(12, 'https://x/12', 'refs/pr/12', 'x')

    real_create_commit = hub.create_commit

    def fail_on_second(**kwargs):
        if len(hub.commits) >= 1:
            raise RuntimeError('504 Gateway Timeout')
        return real_create_commit(**kwargs)

    hub.create_commit = fail_on_second

    with pytest.raises(submit.PartialSubmissionError) as caught:
        sub.publish(
            'hle',
            pull_request=pull_request,
            operations=submit.upload_operations(tree),
            description='',
            message='hle 2026-08-10',
        )

    assert caught.value.pull_request is pull_request
    assert len(caught.value.committed_paths) == 3
    assert all(
        path.startswith('data/hle/') for path in caught.value.committed_paths
    )


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


def test_the_description_carries_coverage_and_the_marker() -> None:
    description = submit.pull_request_description(
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
    assert description.rstrip().endswith(submit.marker('hle'))
    assert 'type_of_addition' in description


def test_the_submitter_refuses_an_empty_repository_id() -> None:
    with pytest.raises(submit.SubmissionError, match='repository id'):
        submit.DatastoreSubmitter(FakeHub(), repo_id='')
