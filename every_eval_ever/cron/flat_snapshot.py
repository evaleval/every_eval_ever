"""Read published aggregate objects from one pinned, shallow Git fetch."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import urlsplit

from huggingface_hub import HfApi
from huggingface_hub.utils import build_hf_headers

_COMMIT = re.compile(r'[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z')
_LFS_VERSION = b'version https://git-lfs.github.com/spec/'
_RETIRED_PATH = 'flat/retired.jsonl'


class FlatSnapshotError(RuntimeError):
    """The pinned published objects could not be read safely."""


def _pointer(data: bytes, path: str) -> tuple[str, int] | None:
    if not data.startswith(_LFS_VERSION):
        return None
    try:
        lines = data.decode('ascii').splitlines()
    except UnicodeDecodeError as exc:
        raise FlatSnapshotError(f'Malformed Git LFS pointer: {path}') from exc
    oid = [
        line.removeprefix('oid sha256:')
        for line in lines
        if line.startswith('oid sha256:')
    ]
    size = [
        line.removeprefix('size ') for line in lines if line.startswith('size ')
    ]
    if (
        lines[0] != 'version https://git-lfs.github.com/spec/v1'
        or len(oid) != 1
        or re.fullmatch(r'[0-9a-f]{64}', oid[0]) is None
        or len(size) != 1
        or re.fullmatch(r'[0-9]+', size[0]) is None
        or any(
            not line.startswith(('version ', 'oid sha256:', 'size ', 'ext-'))
            for line in lines
        )
    ):
        raise FlatSnapshotError(f'Malformed Git LFS pointer: {path}')
    return oid[0], int(size[0])


class FlatSnapshotReader:
    """A temporary checkout of aggregate objects and the retirement list.

    Resolve the dataset's current commit once, or supply an immutable ``revision``.
    Git transfers plain blobs in one pack with LFS smudging disabled. Only a read
    of an LFS-backed path downloads its payload, at the same pinned commit.
    Sample sidecars are outside this reader's scope.
    """

    def __init__(
        self,
        api: HfApi,
        repo_id: str,
        *,
        revision: str | None = None,
        remote_url: str | None = None,
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.revision = revision
        endpoint = getattr(api, 'endpoint', 'https://huggingface.co').rstrip(
            '/'
        )
        self.remote_url = remote_url or f'{endpoint}/datasets/{repo_id}'
        parsed = urlsplit(self.remote_url)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError('Git remote URLs must not contain credentials')
        self.pack_count = 0
        self.lfs_download_paths: list[str] = []
        self._payloads: dict[str, Path] = {}
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._directory: Path | None = None
        self._env = os.environ.copy()
        self._env.update(GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
        config = {
            'filter.lfs.smudge': '',
            'filter.lfs.process': '',
            'filter.lfs.required': 'false',
        }
        self._authorization = None
        if self.remote_url.startswith(endpoint + '/'):
            self._authorization = build_hf_headers(
                token=getattr(api, 'token', None)
            ).get('authorization')
            if self._authorization:
                config[f'http.{endpoint}/.extraHeader'] = (
                    f'Authorization: {self._authorization}'
                )
        count = int(self._env.get('GIT_CONFIG_COUNT', '0'))
        for key, value in config.items():
            self._env[f'GIT_CONFIG_KEY_{count}'] = key
            self._env[f'GIT_CONFIG_VALUE_{count}'] = value
            count += 1
        self._env['GIT_CONFIG_COUNT'] = str(count)

    @property
    def lfs_download_count(self) -> int:
        """Number of named LFS payload downloads requested by this reader."""
        return len(self.lfs_download_paths)

    def _git(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ['git', *args],
                cwd=self._directory,
                env=self._env,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FlatSnapshotError(
                'Git is required to read published flat objects in bulk'
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or '').strip()
            if self._authorization:
                detail = detail.replace(self._authorization, '[redacted]')
            raise FlatSnapshotError(
                f'Git {args[0]} failed while reading flat objects: {detail}'
            ) from exc
        return result.stdout.strip()

    def __enter__(self) -> FlatSnapshotReader:
        if self._temporary is not None or self.pack_count:
            raise FlatSnapshotError('A flat snapshot reader cannot be reused')
        if self.revision is None:
            self.revision = self.api.repo_info(
                repo_id=self.repo_id, repo_type='dataset'
            ).sha
        if not isinstance(self.revision, str) or not _COMMIT.fullmatch(
            self.revision
        ):
            raise FlatSnapshotError(
                'The datastore revision must be a commit SHA'
            )
        self.revision = self.revision.lower()
        self._temporary = tempfile.TemporaryDirectory(
            prefix='eee-flat-objects-'
        )
        self._directory = Path(self._temporary.name).resolve()
        try:
            self._git('init', '--quiet')
            self._git('fetch', '--depth', '1', self.remote_url, self.revision)
            self.pack_count += 1
            fetched = self._git('rev-parse', 'FETCH_HEAD^{commit}')
            if fetched != self.revision:
                raise FlatSnapshotError(
                    f'Fetched datastore commit {fetched} differs from pin '
                    f'{self.revision}'
                )
            self._git('sparse-checkout', 'init', '--no-cone')
            self._git(
                'sparse-checkout',
                'set',
                '--no-cone',
                '/flat/objects/**/*.json',
                f'/{_RETIRED_PATH}',
            )
            self._git('checkout', '--detach', self.revision)
            if self._git('rev-parse', 'HEAD') != self.revision:
                raise FlatSnapshotError(
                    'Checked-out datastore commit differs from pin'
                )
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        """Remove the temporary checkout; downloaded Hub cache entries remain."""
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._directory = None
        self._payloads.clear()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _path(self, path: str) -> Path:
        parts = PurePosixPath(path).parts
        allowed = path == _RETIRED_PATH or (
            len(parts) >= 3
            and parts[:2] == ('flat', 'objects')
            and parts[-1].endswith('.json')
        )
        if (
            not allowed
            or '..' in parts
            or '\\' in path
            or PurePosixPath(path).as_posix() != path
        ):
            raise ValueError(
                f'Not a published aggregate or retirement list: {path}'
            )
        if self._directory is None:
            raise FlatSnapshotError(
                'Open the flat snapshot reader before reading'
            )
        local = self._directory / path
        if local.is_symlink() or self._directory not in local.resolve().parents:
            raise FlatSnapshotError(
                f'Published object must be a regular file: {path}'
            )
        return local

    def exists(self, path: str) -> bool:
        """Return whether a permitted path exists in the pinned checkout."""
        return self._path(path).is_file()

    def read(self, path: str) -> bytes:
        """Read a published aggregate or retirement list, resolving LFS explicitly."""
        data = self._path(path).read_bytes()
        pointer = _pointer(data, path)
        if pointer is None:
            return data
        if path not in self._payloads:
            self._payloads[path] = Path(
                self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=path,
                    repo_type='dataset',
                    revision=self.revision,
                )
            )
            self.lfs_download_paths.append(path)
        payload = self._payloads[path].read_bytes()
        digest, size = pointer
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise FlatSnapshotError(
                f'Git LFS payload failed SHA256/size check: {path}'
            )
        return payload

    def read_many(self, paths: Iterable[str]) -> dict[str, bytes]:
        """Read each requested aggregate without retaining an extra byte cache."""
        return {path: self.read(path) for path in paths}
