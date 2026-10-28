"""Offline Git and LFS fixtures for published aggregate identity reads."""

from __future__ import annotations

import hashlib
import os
import subprocess
from types import SimpleNamespace

import pytest

from every_eval_ever.cron import flat_snapshot as fs

PLAIN = 'flat/objects/aa/plain.json'
LFS = 'flat/objects/bb/large.json'
UNTOUCHED_LFS = 'flat/objects/cc/untouched.json'
RETIRED = 'flat/retired.jsonl'


def git(directory, *args):
    env = os.environ.copy()
    env.update(
        GIT_CONFIG_NOSYSTEM='1',
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_LFS_SKIP_SMUDGE='1',
        GIT_ALLOW_PROTOCOL='file',
        GIT_CONFIG_COUNT='3',
        GIT_CONFIG_KEY_0='filter.lfs.process',
        GIT_CONFIG_VALUE_0='',
        GIT_CONFIG_KEY_1='filter.lfs.smudge',
        GIT_CONFIG_VALUE_1='',
        GIT_CONFIG_KEY_2='filter.lfs.required',
        GIT_CONFIG_VALUE_2='false',
    )
    return subprocess.run(
        ['git', *args],
        cwd=directory,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def pointer(payload):
    return (
        'version https://git-lfs.github.com/spec/v1\n'
        f'oid sha256:{hashlib.sha256(payload).hexdigest()}\n'
        f'size {len(payload)}\n'
    ).encode()


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '--quiet')
    git(source, 'config', 'user.email', 'fixture@example.invalid')
    git(source, 'config', 'user.name', 'Offline fixture')
    plain = b'{"evaluation_results": [{"name": "fixture metric"}]}\n'
    payload = b'{"evaluation_results": [{"name": "larger fixture metric"}]}\n'
    files = {
        PLAIN: plain,
        'flat/objects/direct.json': b'{}\n',
        LFS: pointer(payload),
        UNTOUCHED_LFS: pointer(b'{"untouched": true}\n'),
        RETIRED: b'{"fixture": "retired"}\n',
        'data/unrelated/record.json': b'{"not_checked_out": true}\n',
    }
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(source, 'add', '.')
    git(source, 'commit', '--quiet', '-m', 'Pinned fixture')
    pin = git(source, 'rev-parse', 'HEAD')
    (source / PLAIN).write_bytes(b'{"later": true}\n')
    git(source, 'add', '.')
    git(source, 'commit', '--quiet', '-m', 'Later fixture')
    remote = tmp_path / 'remote.git'
    git(tmp_path, 'clone', '--bare', str(source), str(remote))
    cache = tmp_path / 'payload.json'
    cache.write_bytes(payload)
    return SimpleNamespace(
        source=source,
        remote=remote,
        pin=pin,
        tip=git(source, 'rev-parse', 'HEAD'),
        plain=plain,
        payload=payload,
        cache=cache,
    )


class FakeApi:
    endpoint = 'https://example.invalid'
    token = False

    def __init__(self, repository):
        self.repository = repository
        self.info_calls = []
        self.downloads = []

    def repo_info(self, **kwargs):
        self.info_calls.append(kwargs)
        return SimpleNamespace(sha=self.repository.pin)

    def hf_hub_download(self, **kwargs):
        self.downloads.append(kwargs)
        assert kwargs == {
            'repo_id': 'fixture/datastore',
            'repo_type': 'dataset',
            'filename': LFS,
            'revision': self.repository.pin,
        }
        return str(self.repository.cache)


def reader(repository, api=None, **kwargs):
    return fs.FlatSnapshotReader(
        api or FakeApi(repository),
        'fixture/datastore',
        remote_url=str(repository.remote),
        **kwargs,
    )


def test_one_shallow_pack_at_resolved_pin_and_sparse_checkout(
    repository, monkeypatch
):
    api = FakeApi(repository)
    calls = []
    run = subprocess.run

    def recording_run(command, **kwargs):
        calls.append((command, kwargs))
        return run(command, **kwargs)

    monkeypatch.setattr(fs.subprocess, 'run', recording_run)
    snapshot = reader(repository, api)
    with snapshot:
        assert snapshot.revision == repository.pin != repository.tip
        assert snapshot.pack_count == 1
        assert snapshot.read_many([PLAIN, RETIRED]) == {
            PLAIN: repository.plain,
            RETIRED: b'{"fixture": "retired"}\n',
        }
        assert snapshot.read('flat/objects/direct.json') == b'{}\n'
        assert snapshot.exists(LFS)
        assert not snapshot.exists('flat/objects/missing.json')
        checkout = snapshot._directory
        assert checkout is not None
        assert not (checkout / 'data').exists()
        assert (checkout / '.git/shallow').read_text().strip() == repository.pin
        assert api.downloads == []
        assert snapshot.lfs_download_count == 0
    assert not checkout.exists()
    assert api.info_calls == [
        {'repo_id': 'fixture/datastore', 'repo_type': 'dataset'}
    ]
    fetches = [command for command, _ in calls if command[1] == 'fetch']
    assert fetches == [
        ['git', 'fetch', '--depth', '1', str(repository.remote), repository.pin]
    ]
    assert all('--filter' not in ' '.join(command) for command, _ in calls)
    assert all(
        kwargs['env']['GIT_LFS_SKIP_SMUDGE'] == '1' for _, kwargs in calls
    )
    assert all(
        kwargs['env']['GIT_TERMINAL_PROMPT'] == '0' for _, kwargs in calls
    )


def test_only_touched_lfs_payload_downloaded_once_at_pin(repository):
    api = FakeApi(repository)
    with reader(repository, api) as snapshot:
        assert snapshot.read(PLAIN) == repository.plain
        assert api.downloads == []
        assert snapshot.read(LFS) == repository.payload
        assert snapshot.read(LFS) == repository.payload
        assert len(api.downloads) == snapshot.lfs_download_count == 1
        assert snapshot.lfs_download_paths == [LFS]
        assert UNTOUCHED_LFS not in snapshot.lfs_download_paths


@pytest.mark.parametrize('corruption', ['wrong size', 'wrong hash'])
def test_lfs_payload_hash_and_size_verified(repository, corruption):
    repository.cache.write_bytes(
        b'bad' if corruption == 'wrong size' else b'x' * len(repository.payload)
    )
    with reader(repository) as snapshot:
        with pytest.raises(fs.FlatSnapshotError, match='SHA256/size'):
            snapshot.read(LFS)


def test_supplied_pin_skips_revision_resolution(repository):
    api = FakeApi(repository)
    with reader(repository, api, revision=repository.pin) as snapshot:
        assert snapshot.read(PLAIN) == repository.plain
    assert api.info_calls == []


@pytest.mark.parametrize(
    'path',
    [
        'flat/objects/aa/private_samples.jsonl',
        'data/source/private_samples.jsonl',
        'flat/objects/../retired.json',
        '/flat/objects/x.json',
        'flat/objects/./x.json',
        'flat/objects\\x.json',
        'flat/manifests/example/entries.jsonl',
    ],
)
def test_identity_reader_rejects_samples_and_other_paths(repository, path):
    api = FakeApi(repository)
    with reader(repository, api) as snapshot:
        with pytest.raises(ValueError, match='Not a published aggregate'):
            snapshot.read(path)
    assert api.downloads == []


def test_cleanup_after_consumer_error(repository):
    snapshot = reader(repository)
    with pytest.raises(RuntimeError, match='consumer error'):
        with snapshot:
            checkout = snapshot._directory
            raise RuntimeError('consumer error')
    assert not checkout.exists()


def test_missing_git_fails_loudly_and_cleans_up(repository, monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(fs.subprocess, 'run', missing)
    snapshot = reader(repository)
    with pytest.raises(fs.FlatSnapshotError, match='Git is required'):
        with snapshot:
            pass
    assert snapshot._directory is None
    assert snapshot._temporary is None


def test_fetch_failure_cleans_up(repository, monkeypatch):
    run = subprocess.run

    def failing(command, **kwargs):
        if command[1] == 'fetch':
            raise subprocess.CalledProcessError(
                1, command, stderr='fixture failure'
            )
        return run(command, **kwargs)

    monkeypatch.setattr(fs.subprocess, 'run', failing)
    snapshot = reader(repository)
    with pytest.raises(
        fs.FlatSnapshotError, match='Git fetch failed.*fixture failure'
    ):
        with snapshot:
            pass
    assert snapshot._directory is None


@pytest.mark.parametrize('checked_ref', ['FETCH_HEAD^{commit}', 'HEAD'])
def test_fetched_and_checked_out_commit_must_match_pin(
    repository, monkeypatch, checked_ref
):
    original = fs.FlatSnapshotReader._git

    def wrong_pin(self, *args):
        if args == ('rev-parse', checked_ref):
            return repository.tip
        return original(self, *args)

    monkeypatch.setattr(fs.FlatSnapshotReader, '_git', wrong_pin)
    snapshot = reader(repository)
    with pytest.raises(fs.FlatSnapshotError, match='differs from pin'):
        with snapshot:
            pass
    assert snapshot._directory is None


def test_non_commit_revision_rejected_before_git(repository, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail('Git must not run for a mutable revision')

    monkeypatch.setattr(fs.subprocess, 'run', unexpected)
    with pytest.raises(fs.FlatSnapshotError, match='must be a commit SHA'):
        with reader(repository, revision='main'):
            pass


def test_lfs_filters_disabled_without_needing_git_lfs(repository, monkeypatch):
    # An inherited filter that fails must never run during the sparse checkout.
    monkeypatch.setenv('GIT_CONFIG_COUNT', '2')
    monkeypatch.setenv('GIT_CONFIG_KEY_0', 'filter.lfs.process')
    monkeypatch.setenv('GIT_CONFIG_VALUE_0', 'a-nonexistent-lfs-process')
    monkeypatch.setenv('GIT_CONFIG_KEY_1', 'filter.lfs.required')
    monkeypatch.setenv('GIT_CONFIG_VALUE_1', 'true')
    (repository.source / '.gitattributes').write_text('*.json filter=lfs\n')
    # Stage attributes only: existing pointer blobs remain ordinary Git blobs.
    git(repository.source, 'add', '.gitattributes')
    git(repository.source, 'commit', '--quiet', '-m', 'LFS attributes')
    pin = git(repository.source, 'rev-parse', 'HEAD')
    git(repository.remote, 'fetch', str(repository.source), pin)
    repository.pin = pin
    with reader(repository, revision=pin) as snapshot:
        assert snapshot.read(LFS) == repository.payload


def test_auth_header_uses_environment_and_does_not_enter_git_arguments(
    repository,
):
    api = FakeApi(repository)
    api.token = 'hf_fixture_secret'
    snapshot = fs.FlatSnapshotReader(api, 'fixture/datastore')
    assert (
        snapshot.remote_url
        == 'https://example.invalid/datasets/fixture/datastore'
    )
    config = {
        snapshot._env[f'GIT_CONFIG_KEY_{i}']: snapshot._env[
            f'GIT_CONFIG_VALUE_{i}'
        ]
        for i in range(int(snapshot._env['GIT_CONFIG_COUNT']))
    }
    assert config['http.https://example.invalid/.extraHeader'] == (
        'Authorization: Bearer hf_fixture_secret'
    )


@pytest.mark.parametrize(
    'data',
    [
        b'version https://git-lfs.github.com/spec/v2\n',
        b'version https://git-lfs.github.com/spec/v1\noid sha256:bad\nsize 2\n',
        b'version https://git-lfs.github.com/spec/v1\n\xff',
    ],
)
def test_malformed_lfs_pointer_fails_explicitly(data):
    with pytest.raises(fs.FlatSnapshotError, match='Malformed Git LFS pointer'):
        fs._pointer(data, LFS)
