"""Each adapter keeps one datastore pull request, and large sets are batched."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from every_eval_ever.cron import publish

REPO = 'evaleval/EEE_datastore'


@dataclass
class _Discussion:
    title: str
    num: int
    author: str = 'eee-cron-bot'


@dataclass
class _CommitInfo:
    pr_url: str | None


CRON_ACCOUNT = 'eee-cron-bot'


class _FakeApi:
    """Records the commits a refresh would make."""

    def __init__(self, discussions: list[_Discussion] | None = None):
        self.discussions = discussions or []
        self.commits: list[dict] = []
        self.next_pr = 42

    def whoami(self):
        return {'name': CRON_ACCOUNT}

    def get_repo_discussions(self, **kwargs):
        self.query = kwargs
        author = kwargs.get('author')
        return iter(
            [
                discussion
                for discussion in self.discussions
                if author is None or discussion.author == author
            ]
        )

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        if kwargs.get('create_pr'):
            return _CommitInfo(
                pr_url=f'https://huggingface.co/datasets/{REPO}/discussions/'
                f'{self.next_pr}'
            )
        return _CommitInfo(pr_url=None)


def _records(root: Path, count: int, collection: str = 'vals-ai') -> Path:
    data_root = root / 'data'
    for index in range(count):
        directory = data_root / collection / 'dev' / f'model{index}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'record{index}.json').write_text('{}', encoding='utf-8')
    return data_root


def test_collect_finds_aggregates_and_sample_companions(tmp_path: Path):
    data_root = _records(tmp_path, 1)
    directory = data_root / 'vals-ai' / 'dev' / 'model0'
    (directory / 'record0_samples.jsonl').write_text('{}\n', encoding='utf-8')
    (directory / 'notes.txt').write_text('ignored', encoding='utf-8')

    found = publish.collect_files(data_root)

    assert [path.name for path in found] == [
        'record0.json',
        'record0_samples.jsonl',
    ]


def test_a_first_refresh_opens_the_adapters_pull_request(tmp_path: Path):
    data_root = _records(tmp_path, 2)
    api = _FakeApi()

    result = publish.publish(
        data_root, adapter='vals_ai', repo_id=REPO, api=api
    )

    assert result.reused_existing_pr is False
    assert result.pr_number == 42
    assert result.files == 2
    assert result.commits == 1
    assert api.commits[0]['create_pr'] is True
    assert api.commits[0]['commit_message'] == publish.pr_title('vals_ai')


def test_a_later_refresh_commits_onto_the_same_pull_request(tmp_path: Path):
    data_root = _records(tmp_path, 1)
    api = _FakeApi([_Discussion(title=publish.pr_title('vals_ai'), num=7)])

    result = publish.publish(
        data_root, adapter='vals_ai', repo_id=REPO, api=api
    )

    assert result.reused_existing_pr is True
    assert result.pr_number == 7
    assert api.commits[0]['revision'] == 'refs/pr/7'
    assert 'create_pr' not in api.commits[0]


def test_another_adapters_pull_request_is_not_reused(tmp_path: Path):
    data_root = _records(tmp_path, 1)
    api = _FakeApi([_Discussion(title=publish.pr_title('hle'), num=7)])

    result = publish.publish(
        data_root, adapter='vals_ai', repo_id=REPO, api=api
    )

    assert result.reused_existing_pr is False
    assert api.commits[0]['create_pr'] is True


def test_only_open_pull_requests_are_searched(tmp_path: Path):
    data_root = _records(tmp_path, 1)
    api = _FakeApi()

    publish.publish(data_root, adapter='vals_ai', repo_id=REPO, api=api)

    assert api.query['discussion_status'] == 'open'
    assert api.query['discussion_type'] == 'pull_request'
    assert api.query['repo_type'] == 'dataset'


def test_large_sets_are_split_across_commits(tmp_path: Path):
    data_root = _records(tmp_path, 5)
    api = _FakeApi()

    result = publish.publish(
        data_root,
        adapter='vals_ai',
        repo_id=REPO,
        api=api,
        files_per_commit=2,
    )

    assert result.commits == 3
    assert [len(commit['operations']) for commit in api.commits] == [2, 2, 1]
    # Only the first commit opens the PR; the rest land on its ref.
    assert api.commits[0]['create_pr'] is True
    assert [commit.get('revision') for commit in api.commits[1:]] == [
        'refs/pr/42',
        'refs/pr/42',
    ]


def test_the_pr_creating_commit_carries_the_bare_title(tmp_path: Path):
    # The Hub titles the PR from the first commit's message, and find_open_pr
    # matches that title exactly tomorrow. A batch suffix here would title the
    # PR '... (1/2)' and every later day would open a fresh PR.
    data_root = _records(tmp_path, 3)
    api = _FakeApi()

    publish.publish(
        data_root,
        adapter='vals_ai',
        repo_id=REPO,
        api=api,
        files_per_commit=2,
    )

    assert api.commits[0]['commit_message'] == publish.pr_title('vals_ai')
    assert api.commits[0]['commit_description'].startswith('Batch 1/2')
    assert api.commits[1]['commit_message'].endswith('(2/2)')


def test_a_batched_first_publish_is_found_again_the_next_day(tmp_path: Path):
    data_root = _records(tmp_path, 3)
    api = _FakeApi()
    publish.publish(
        data_root, adapter='vals_ai', repo_id=REPO, api=api, files_per_commit=2
    )
    # The Hub would have titled the PR from the first commit's message.
    api.discussions.append(
        _Discussion(title=api.commits[0]['commit_message'], num=42)
    )

    assert publish.find_open_pr(api, REPO, 'vals_ai') == 42


def test_a_strangers_pr_with_the_cron_title_is_not_adopted(tmp_path: Path):
    # The datastore is public: anyone can open a PR with any title. Committing
    # onto it would hand them control of where the cron's records land.
    data_root = _records(tmp_path, 1)
    api = _FakeApi(
        [
            _Discussion(
                title=publish.pr_title('vals_ai'),
                num=7,
                author='someone-else',
            )
        ]
    )

    result = publish.publish(
        data_root, adapter='vals_ai', repo_id=REPO, api=api
    )

    assert result.reused_existing_pr is False
    assert api.commits[0]['create_pr'] is True
    assert api.query['author'] == CRON_ACCOUNT


def test_records_keep_their_datastore_path(tmp_path: Path):
    data_root = _records(tmp_path, 1, collection='helm_lite')
    api = _FakeApi()

    publish.publish(data_root, adapter='helm', repo_id=REPO, api=api)

    operation = api.commits[0]['operations'][0]
    assert operation.path_in_repo == 'data/helm_lite/dev/model0/record0.json'


def test_publishing_nothing_is_an_error(tmp_path: Path):
    with pytest.raises(publish.PublishError, match='no record files'):
        publish.publish(
            tmp_path / 'data', adapter='vals_ai', repo_id=REPO, api=_FakeApi()
        )


def test_a_missing_pull_request_url_stops_before_the_next_batch(
    tmp_path: Path,
):
    data_root = _records(tmp_path, 4)

    class _NoUrlApi(_FakeApi):
        def create_commit(self, **kwargs):
            self.commits.append(kwargs)
            return _CommitInfo(pr_url=None)

    api = _NoUrlApi()

    with pytest.raises(publish.PublishError, match='did not return'):
        publish.publish(
            data_root,
            adapter='vals_ai',
            repo_id=REPO,
            api=api,
            files_per_commit=2,
        )

    # Stopped rather than sending the rest to the main branch.
    assert len(api.commits) == 1
