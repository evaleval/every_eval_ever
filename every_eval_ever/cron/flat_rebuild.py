"""Rebuild the datastore's flat view against the Hub, without a clone.

``tools/build_flat_datastore.py`` in the EEE_datastore repository defines the
flat layout and rebuilds it from a local checkout. That checkout no longer
fits a GitHub Actions runner — ``data/`` alone is ~15 GB — so this module
writes the same flat layout against the Hub API instead:

1. read the current snapshot: ``flat/latest_manifest.json`` and the
   ``entries.jsonl`` it points at;
2. recursively list the repository and diff ``data/`` against the snapshot
   rows, using file size as the change signal;
3. download new and size-changed source files (a full initial build is gated
   behind ``--allow-bootstrap``);
4. rebuild rows, collection indexes and the manifest, preserving published
   bytes, accepting reserialization and reporting conflicting UUID reuse;
5. retire collection indexes whose collection no longer exists under
   ``data/`` (moved to ``flat/indexes/retired/``), trim snapshots past the
   retention window, and commit everything in bounded batches with the
   adapter ingestion's retry discipline (``cron.submit``).

The retention rule keeps any manifest that is the last remaining index into
an object, so no object in ``flat/objects/`` ever loses its row. Objects are
never deleted, and ``data/`` is never touched: it is the source of truth,
maintained by the ingestion cron and human pull requests. Records are not
validated here either — the ingestion cron validated each one before
committing it.

A run whose rebuilt manifest core hash matches the published one and that
has no missing index, retire or retention work commits nothing. Conflicts
still produce exit code 2; comparison failures and verification drift stop
publication with an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence
from uuid import UUID

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
)
from huggingface_hub.errors import EntryNotFoundError

# Imported as a module, not by name: the retry helpers are replaced in tests
# and the commit path has to see the replacement.
from every_eval_ever.cron import store
from every_eval_ever.cron.submit import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASTORE_REPO,
)

LATEST_MANIFEST_PATH = 'flat/latest_manifest.json'
INDEXES_PREFIX = 'flat/indexes/by_collection'
RETIRE_PREFIX = 'flat/indexes/retired'
BY_LEGACY_PATH = 'flat/indexes/by_legacy_path.jsonl'
MANIFESTS_PREFIX = 'flat/manifests'
DATA_PREFIX = 'data/'
#: Fields of a manifest excluded from its content hash, mirroring
#: ``manifest_core`` in ``tools/build_flat_datastore.py``.
CORE_IGNORED_FIELDS = frozenset(
    {'created_at', 'entries_path', 'manifest_core_sha256', 'manifest_path'}
)
DEFAULT_RETAIN_DAYS = 90
DEFAULT_KEEP_NEWEST = 2
#: Snapshots older than the retention window are scanned newest-first to
#: decide between trim and pin; unscanned ones are conservatively kept.
DEFAULT_SCAN_BUDGET = 20
DOWNLOAD_WORKERS = 8


class FlatRebuildError(RuntimeError):
    """Raised when the flat view cannot be rebuilt or published safely."""


@dataclass(frozen=True)
class Row:
    """One ``entries.jsonl`` row, mirroring the builder's row shape."""

    object_uuid: str
    object_path: str
    sha256: str
    size_bytes: int
    legacy_path: str
    benchmark: str
    eval_schema_version: str
    instance_level_available: bool = False
    instance_level_path: str | None = None
    instance_sha: str | None = None
    instance_level_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.instance_level_available:
            if (
                self.instance_level_path is None
                or self.instance_sha is None
                or self.instance_level_size_bytes is None
            ):
                raise ValueError(
                    'instance_level_available rows require companion metadata'
                )
            return
        object.__setattr__(self, 'instance_level_path', None)
        object.__setattr__(self, 'instance_sha', None)
        object.__setattr__(self, 'instance_level_size_bytes', None)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            'object_uuid': self.object_uuid,
            'object_path': self.object_path,
            'sha256': self.sha256,
            'size_bytes': self.size_bytes,
            'legacy_path': self.legacy_path,
            'benchmark': self.benchmark,
            'eval_schema_version': self.eval_schema_version,
            'record_type': 'aggregate',
            'instance_level_available': self.instance_level_available,
        }
        if self.instance_level_available:
            row['instance_level_path'] = self.instance_level_path
            row['instance_sha'] = self.instance_sha
            row['instance_level_size_bytes'] = self.instance_level_size_bytes
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> 'Row':
        return cls(
            object_uuid=row['object_uuid'],
            object_path=row['object_path'],
            sha256=row['sha256'],
            size_bytes=row['size_bytes'],
            legacy_path=row['legacy_path'],
            benchmark=row['benchmark'],
            eval_schema_version=row['eval_schema_version'],
            instance_level_available=bool(row['instance_level_available']),
            instance_level_path=row.get('instance_level_path'),
            instance_sha=row.get('instance_sha'),
            instance_level_size_bytes=row.get('instance_level_size_bytes'),
        )


@dataclass(frozen=True)
class ManifestInfo:
    """One snapshot directory under ``flat/manifests/``."""

    manifest_path: str
    entries_path: str
    created_at: str
    dir_path: str

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.created_at)


@dataclass(frozen=True)
class Diff:
    """What changed between a snapshot and the current ``data/`` tree."""

    new_paths: tuple[str, ...]
    new_sample_paths: tuple[str, ...]
    removed_rows: tuple[Row, ...]
    #: Aggregates still at their snapshot path whose size changed. Same-UUID
    #: content changes are resolved by build_rows against the existing flat
    #: object: a re-serialization is accepted, a semantic change is not.
    changed_aggregates: tuple[str, ...]
    #: Samples whose size changed under a still-present aggregate.
    changed_samples: tuple[str, ...]
    orphan_samples: tuple[str, ...]
    removed_sample_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    rows: tuple[Row, ...]
    errors: tuple[str, ...]
    #: Records re-emitted under a known UUID with different content. They
    #: cannot enter the flat view without breaking object immutability, so
    #: they are excluded (the old object stays, pinned by retention) and
    #: reported for upstream to resolve - usually a schema migration that
    #: should have carried fresh UUIDs.
    conflicts: tuple[str, ...]
    excluded_paths: frozenset[str]
    added: int
    moved: int
    #: UUIDs whose aggregate object must be uploaded (never seen before).
    upload_aggregates: frozenset[str]
    #: UUIDs whose samples object must be uploaded (new or newly attached).
    upload_samples: frozenset[str]


@dataclass
class RebuildReport:
    """What one run saw and did; printed and written to the step summary."""

    repo_id: str
    dry_run: bool = False
    noop: bool = False
    old_records: int = 0
    data_files: int = 0
    added_records: int = 0
    moved_records: int = 0
    removed_records: int = 0
    orphan_samples: int = 0
    records: int = 0
    collections: int = 0
    indexes_added: tuple[str, ...] = ()
    indexes_retired: tuple[str, ...] = ()
    manifests_total: int = 0
    manifests_trimmed: tuple[str, ...] = ()
    manifests_pinned: tuple[str, ...] = ()
    manifests_unscanned: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    manifest_core_sha256: str | None = None
    verified_files: int = 0
    verified_reserialized: int = 0
    commits: int = 0
    uploads: int = 0
    duration_seconds: float = 0.0


# -- shared hashing helpers (mirroring tools/build_flat_datastore.py) ------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(
        'utf-8'
    )


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def object_path_for(object_uuid: str, *, suffix: str) -> str:
    return str(
        PurePosixPath('flat', 'objects', object_uuid[:2], object_uuid[2:4])
        / f'{object_uuid}{suffix}'
    )


def samples_path_for(legacy_path: str) -> str:
    parent = PurePosixPath(legacy_path).parent
    stem = PurePosixPath(legacy_path).stem
    return str(parent / f'{stem}_samples.jsonl')


def manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in CORE_IGNORED_FIELDS
    }


def jsonl_text(rows: 'Sequence[Row] | Sequence[dict[str, Any]]') -> str:
    return ''.join(
        json.dumps(
            row.to_dict() if isinstance(row, Row) else row, sort_keys=True
        )
        + '\n'
        for row in rows
    )


def manifest_for(rows: Sequence[Row], *, created_at: str) -> dict[str, Any]:
    """Build the snapshot manifest exactly as the local builder would."""
    entries_text = jsonl_text(rows)
    encoded = entries_text.encode('utf-8')
    benchmarks = {row.benchmark for row in rows}
    instance_count = sum(1 for row in rows if row.instance_level_available)
    core: dict[str, Any] = {
        'source': {'type': 'legacy_data_tree', 'path': 'data'},
        'eval_schema_versions': sorted(
            {row.eval_schema_version for row in rows}
        ),
        'benchmark_count': len(benchmarks),
        'aggregate_file_count': len(rows),
        'instance_level_file_count': instance_count,
        'total_file_count': len(rows) + instance_count,
        'entries_sha256': sha256_bytes(encoded),
        'entries_size_bytes': len(encoded),
    }
    core_sha256 = sha256_bytes(stable_json_bytes(core))
    dir_path = f'{MANIFESTS_PREFIX}/sha256_{core_sha256}'
    return {
        **core,
        'created_at': created_at,
        'entries_path': f'{dir_path}/entries.jsonl',
        'manifest_core_sha256': core_sha256,
        'manifest_path': f'{dir_path}/manifest.json',
    }


# -- reading remote state --------------------------------------------------


def list_repo_sizes(api: HfApi, repo_id: str) -> dict[str, int]:
    """Map every file path in the repository to its size in bytes."""
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(
        repo_id=repo_id, repo_type='dataset', recursive=True
    ):
        size = getattr(item, 'size', None)
        if size is not None:
            sizes[item.path] = size
    return sizes


def read_snapshot(
    api: HfApi, repo_id: str
) -> tuple[dict[str, Any], list[Row]] | None:
    """Return the current snapshot manifest and its rows, or ``None``."""
    try:
        latest_local = api.hf_hub_download(
            repo_id=repo_id, repo_type='dataset', filename=LATEST_MANIFEST_PATH
        )
    except EntryNotFoundError:
        return None
    manifest = json.loads(Path(latest_local).read_text(encoding='utf-8'))
    entries_local = api.hf_hub_download(
        repo_id=repo_id, repo_type='dataset', filename=manifest['entries_path']
    )
    rows = [
        Row.from_dict(json.loads(line))
        for line in Path(entries_local).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    return manifest, rows


def download_bytes(
    api: HfApi, repo_id: str, paths: Sequence[str]
) -> dict[str, bytes]:
    """Download paths in parallel; a missing file aborts the run."""

    def one(path: str) -> tuple[str, bytes]:
        local = api.hf_hub_download(
            repo_id=repo_id, repo_type='dataset', filename=path
        )
        return path, Path(local).read_bytes()

    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        return dict(pool.map(one, paths))


# -- planning (pure; unit-tested without network) --------------------------


def diff_against_snapshot(
    old_rows: Sequence[Row], listing: dict[str, int]
) -> Diff:
    """Diff ``data/`` against the snapshot rows by path and size.

    New paths and size changes are downloaded for comparison against
    published objects. Companion presence and size are tracked separately.
    Size-equal files carry their row and hash forward; the optional deep
    verification pass checks for same-length drift.
    """
    aggregate_paths = {
        path
        for path in listing
        if path.startswith(DATA_PREFIX) and path.endswith('.json')
    }
    old_by_path = {row.legacy_path: row for row in old_rows}
    new_paths = tuple(
        path for path in sorted(aggregate_paths) if path not in old_by_path
    )
    new_sample_paths = tuple(
        samples_path_for(path)
        for path in sorted(aggregate_paths)
        if samples_path_for(path) in listing
        and (
            path not in old_by_path
            or not old_by_path[path].instance_level_available
        )
    )
    changed_aggregates = tuple(
        path
        for path in sorted(aggregate_paths)
        if path in old_by_path and listing[path] != old_by_path[path].size_bytes
    )
    changed_samples = tuple(
        path
        for path in sorted(aggregate_paths & old_by_path.keys())
        if old_by_path[path].instance_level_available
        and samples_path_for(path) in listing
        and listing[samples_path_for(path)]
        != old_by_path[path].instance_level_size_bytes
    )
    removed_sample_paths = tuple(
        path
        for path in sorted(aggregate_paths & old_by_path.keys())
        if old_by_path[path].instance_level_available
        and samples_path_for(path) not in listing
    )
    removed_rows = tuple(
        row for row in old_rows if row.legacy_path not in aggregate_paths
    )
    orphan_samples = tuple(
        sorted(
            path
            for path in listing
            if path.startswith(DATA_PREFIX)
            and path.endswith('_samples.jsonl')
            and path[: -len('_samples.jsonl')] + '.json' not in aggregate_paths
        )
    )
    return Diff(
        new_paths=new_paths,
        new_sample_paths=new_sample_paths,
        removed_rows=removed_rows,
        changed_aggregates=changed_aggregates,
        changed_samples=changed_samples,
        orphan_samples=orphan_samples,
        removed_sample_paths=removed_sample_paths,
    )


def semantically_equal(a: bytes, b: bytes, *, jsonl: bool = False) -> bool:
    """Compare JSON values or ordered JSONL records without coercing types."""
    if a == b:
        return True

    def equal(left: Any, right: Any) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                equal(left[key], right[key]) for key in left
            )
        if isinstance(left, list):
            return len(left) == len(right) and all(
                equal(x, y) for x, y in zip(left, right)
            )
        return left == right

    try:
        if jsonl:
            left = [json.loads(line) for line in a.splitlines() if line.strip()]
            right = [
                json.loads(line) for line in b.splitlines() if line.strip()
            ]
            return equal(left, right)
        return equal(json.loads(a), json.loads(b))
    except ValueError:
        return False


def verify_rows(
    api: HfApi,
    repo_id: str,
    rows: Sequence[Row],
    listing: dict[str, int],
    *,
    workers: int = DOWNLOAD_WORKERS,
    pending_objects: frozenset[str] = frozenset(),
) -> tuple[int, int, list[str]]:
    """Check inherited aggregates and companions against their snapshot hashes.

    Only explicitly planned new objects are skipped. Missing inherited files
    are drift. Reserialization is accepted only against intact published
    objects, using ordered, type-preserving JSONL comparison for companions.
    Returns ``(checked, reserialized, drift)``; source hashing streams bytes.
    """
    targets: list[tuple[str, str, str, bool]] = []
    for row in rows:
        targets.append((row.legacy_path, row.object_path, row.sha256, False))
        if row.instance_level_available:
            if not row.instance_level_path or not row.instance_sha:
                raise FlatRebuildError(
                    f'{row.legacy_path}: incomplete companion metadata'
                )
            targets.append(
                (
                    samples_path_for(row.legacy_path),
                    row.instance_level_path,
                    row.instance_sha,
                    True,
                )
            )
    targets = [target for target in targets if target[1] not in pending_objects]
    drift: list[str] = []

    def check(target: tuple[str, str, str, bool]) -> tuple[str, str | None]:
        source, destination, expected_sha, jsonl = target
        for path in (source, destination):
            if path not in listing:
                return 'drift', f'{path}: inherited file is missing'
        local = api.hf_hub_download(
            repo_id=repo_id, repo_type='dataset', filename=source
        )
        with Path(local).open('rb') as handle:
            actual_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual_sha == expected_sha:
            return 'clean', None
        object_local = api.hf_hub_download(
            repo_id=repo_id, repo_type='dataset', filename=destination
        )
        object_bytes = Path(object_local).read_bytes()
        if sha256_bytes(object_bytes) != expected_sha:
            return (
                'drift',
                f'{destination}: published object no longer matches its snapshot sha256',
            )
        if semantically_equal(
            object_bytes, Path(local).read_bytes(), jsonl=jsonl
        ):
            return 'reserialized', None
        return (
            'drift',
            f'{source}: data file no longer matches its snapshot sha256 ({expected_sha}) nor its flat object',
        )

    checked = 0
    reserialized = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, message in pool.map(check, targets):
            checked += 1
            if status == 'reserialized':
                reserialized += 1
            elif message:
                drift.append(message)
    return checked, reserialized, drift


def build_rows(
    old_rows: Sequence[Row],
    diff: Diff,
    contents: dict[str, bytes],
    existing_object: Callable[[str], bytes | None] | None = None,
    listing: dict[str, int] | None = None,
) -> BuildResult:
    """Turn the diff into the new sorted row list, or errors.

    A new ``data/`` file whose UUID already exists with different content is
    resolved against the existing flat object (fetched through
    ``existing_object``): a pure re-serialization of the same evaluation is
    accepted, a semantic change under the same UUID violates the flat
    layout's immutability and becomes a conflict that keeps the published
    object. The same UUID reappearing with identical content at a new path
    is a move: the row is re-pointed and nothing is re-uploaded. When
    ``listing`` is given, an upload destination that already exists is
    likewise adopted (the row takes the published bytes' sha256) or, if it
    holds different data, recorded as a conflict - a published object is
    never overwritten.
    """
    errors: list[str] = []
    conflicts: list[str] = []
    excluded: set[str] = set()
    old_by_path = {row.legacy_path: row for row in old_rows}
    old_by_uuid = {row.object_uuid: row for row in old_rows}
    removed_paths = {row.legacy_path for row in diff.removed_rows}
    rows: dict[str, Row] = {
        row.object_uuid: row
        for row in old_rows
        if row.legacy_path not in removed_paths
    }
    upload_aggregates: set[str] = set()
    upload_samples: set[str] = set()
    for uuid, row in list(rows.items()):
        if row.legacy_path in diff.removed_sample_paths:
            rows[uuid] = _row_without_samples(row)
        sample_path = samples_path_for(row.legacy_path)
        if sample_path in diff.new_sample_paths:
            if sample_path not in contents:
                errors.append(
                    f'{sample_path}: expected content was not downloaded'
                )
                continue
            attached, upload, conflict = _attach_samples(
                row, contents[sample_path], listing, existing_object, errors
            )
            if attached is not None:
                rows[uuid] = attached
            if upload:
                upload_samples.add(uuid)
            if conflict:
                conflicts.append(conflict)
                excluded.add(sample_path)
    for path in diff.changed_aggregates:
        old_row = old_by_path[path]
        new_bytes = contents.get(path)
        if new_bytes is None:
            errors.append(f'{path}: expected content was not downloaded')
            continue
        old_object = _fetch_existing(
            old_row.object_path, existing_object, errors
        )
        if old_object is None:
            continue
        if not semantically_equal(old_object, new_bytes):
            conflicts.append(
                f'{path}: content changed under immutable UUID '
                f'{old_row.object_uuid}; the row keeps its published object'
            )
            excluded.add(path)
    for path in diff.changed_samples:
        old_row = old_by_path[path]
        new_bytes = contents.get(samples_path_for(path))
        if new_bytes is None:
            errors.append(
                f'{samples_path_for(path)}: expected content was not downloaded'
            )
            continue
        old_object = _fetch_existing(
            old_row.instance_level_path, existing_object, errors
        )
        if old_object is None:
            continue
        if not semantically_equal(old_object, new_bytes, jsonl=True):
            conflicts.append(
                f'{samples_path_for(path)}: samples changed under immutable '
                f'UUID {old_row.object_uuid}; the row keeps its published object'
            )
            excluded.add(samples_path_for(path))
    seen = {uuid: row.legacy_path for uuid, row in rows.items()}
    added = 0
    moved = 0
    for path in diff.new_paths:
        data = contents.get(path)
        if data is None:
            errors.append(f'{path}: expected content was not downloaded')
            continue
        relative = PurePosixPath(path).relative_to('data')
        if len(relative.parts) < 2:
            errors.append(f'{path}: expected data/<collection>/.../<uuid>.json')
            continue
        benchmark = relative.parts[0]
        try:
            object_uuid = str(UUID(relative.stem))
        except ValueError:
            errors.append(f'{path}: filename is not a valid UUID')
            continue
        if object_uuid in seen:
            errors.append(
                f'{path}: duplicate UUID {object_uuid}; already present at {seen[object_uuid]}'
            )
            continue
        seen[object_uuid] = path
        if (
            samples_path_for(path) in diff.new_sample_paths
            and samples_path_for(path) not in contents
        ):
            errors.append(f'{path}: expected samples were not downloaded')
            continue
        schema_version = schema_version_of(data, path, errors)
        if schema_version is None:
            continue
        sha256 = sha256_bytes(data)
        old = old_by_uuid.get(object_uuid)
        if old is not None:
            if old.sha256 != sha256:
                old_object = _fetch_existing(
                    old.object_path, existing_object, errors
                )
                if old_object is None:
                    continue
                if not semantically_equal(old_object, data):
                    conflicts.append(
                        f'{path}: UUID {object_uuid} re-emitted with '
                        f'different content ({old.legacy_path}); excluded'
                    )
                    excluded.add(path)
                    continue
            moved += 1
            samples = contents.get(samples_path_for(path))
            row, conflict = _row_for_move(
                old, path, benchmark, samples, existing_object, errors, listing
            )
            if conflict:
                conflicts.append(conflict)
                excluded.add(samples_path_for(path))
            if row is None:
                continue
            if (
                row.instance_level_available
                and not old.instance_level_available
                and (listing is None or row.instance_level_path not in listing)
            ):
                upload_samples.add(object_uuid)
            rows[object_uuid] = row
            continue
        row = Row(
            object_uuid=object_uuid,
            object_path=object_path_for(object_uuid, suffix='.json'),
            sha256=sha256,
            size_bytes=len(data),
            legacy_path=path,
            benchmark=benchmark,
            eval_schema_version=schema_version,
        )
        if listing is not None and row.object_path in listing:
            old_object = _fetch_existing(
                row.object_path, existing_object, errors
            )
            if old_object is None:
                continue
            if not semantically_equal(old_object, data):
                conflicts.append(
                    f'{path}: destination object already holds different '
                    f'bytes under immutable UUID {object_uuid}; excluded'
                )
                excluded.add(path)
                continue
            # the object already holds this evaluation: adopt its bytes and
            # skip the upload
            row = replace(
                row,
                sha256=sha256_bytes(old_object),
                size_bytes=len(old_object),
            )
        else:
            upload_aggregates.add(object_uuid)
        added += 1
        samples = contents.get(samples_path_for(path))
        if samples is not None:
            attached, upload, conflict = _attach_samples(
                row, samples, listing, existing_object, errors
            )
            if attached is None:
                continue
            row = attached
            if upload:
                upload_samples.add(object_uuid)
            if conflict:
                conflicts.append(conflict)
                excluded.add(samples_path_for(path))
        rows[object_uuid] = row
    return BuildResult(
        rows=tuple(sorted(rows.values(), key=lambda row: row.legacy_path)),
        errors=tuple(errors),
        conflicts=tuple(conflicts),
        excluded_paths=frozenset(excluded),
        added=added,
        moved=moved,
        upload_aggregates=frozenset(upload_aggregates),
        upload_samples=frozenset(upload_samples),
    )


def _fetch_existing(
    object_path: str,
    existing_object: Callable[[str], bytes | None] | None,
    errors: list[str],
) -> bytes | None:
    """Fetch the published flat object, or None with a hard error.

    A content comparison that cannot fetch the published object is not
    guessable, so the caller treats it as a hard failure rather than a
    conflict it could exclude.
    """
    old_object = existing_object(object_path) if existing_object else None
    if old_object is None:
        errors.append(
            f'{object_path}: content changed under a UUID whose existing '
            'flat object could not be fetched to compare'
        )
    return old_object


def _attach_samples(
    row: Row,
    samples: bytes,
    listing: dict[str, int] | None,
    existing_object: Callable[[str], bytes | None] | None,
    errors: list[str],
) -> tuple[Row | None, bool, str | None]:
    """Attach new samples or adopt matching historical bytes; never overwrite."""
    destination = object_path_for(row.object_uuid, suffix='_samples.jsonl')
    if listing is not None and destination in listing:
        published = _fetch_existing(destination, existing_object, errors)
        if published is None:
            return None, False, None
        if not semantically_equal(published, samples, jsonl=True):
            return (
                row,
                False,
                (
                    f'{samples_path_for(row.legacy_path)}: destination object already '
                    f'holds different content under immutable UUID {row.object_uuid}; '
                    'the aggregate is published without these samples'
                ),
            )
        return _row_with_samples(row, published), False, None
    return _row_with_samples(row, samples), True, None


def _row_for_move(
    old: Row,
    path: str,
    benchmark: str,
    samples: bytes | None,
    existing_object: Callable[[str], bytes | None] | None,
    errors: list[str],
    listing: dict[str, int] | None = None,
) -> tuple[Row | None, str | None]:
    """Re-point an identical record to its new ``data/`` path.

    Returns ``(row, conflict)``. A samples change that is not a
    re-serialization cannot enter the flat view; the row keeps its published
    samples object and the caller records the conflict.
    """
    conflict = None
    if samples is not None and old.instance_level_available:
        old_object = _fetch_existing(
            old.instance_level_path, existing_object, errors
        )
        if old_object is None:
            return None, None
        if not semantically_equal(old_object, samples, jsonl=True):
            conflict = (
                f'{samples_path_for(path)}: samples changed under immutable '
                f'UUID {old.object_uuid}; the row keeps its published object'
            )
    row = replace(
        old,
        legacy_path=path,
        benchmark=benchmark,
    )
    if samples is None:
        row = _row_without_samples(row)
    if samples is not None and not old.instance_level_available:
        row, _, conflict = _attach_samples(
            row, samples, listing, existing_object, errors
        )
    return row, conflict


def _row_without_samples(row: Row) -> Row:
    return replace(
        row,
        instance_level_available=False,
        instance_level_path=None,
        instance_sha=None,
        instance_level_size_bytes=None,
    )


def _row_with_samples(row: Row, samples: bytes) -> Row:
    return replace(
        row,
        instance_level_available=True,
        instance_level_path=object_path_for(
            row.object_uuid, suffix='_samples.jsonl'
        ),
        instance_sha=sha256_bytes(samples),
        instance_level_size_bytes=len(samples),
    )


def schema_version_of(data: bytes, path: str, errors: list[str]) -> str | None:
    try:
        loaded = json.loads(data)
    except json.JSONDecodeError as exc:
        errors.append(f'{path}: invalid JSON ({exc})')
        return None
    if not isinstance(loaded, dict):
        errors.append(f'{path}: eval JSON must contain an object')
        return None
    version = loaded.get('schema_version')
    if not isinstance(version, str) or not version:
        errors.append(f'{path}: missing schema_version')
        return None
    return version


def plan_retire(
    listing: dict[str, int], live_benchmarks: set[str], *, today: str
) -> list[tuple[str, str]]:
    """Move stale collection index paths into ``flat/indexes/retired/``.

    An index whose collection no longer has rows under ``data/`` is a stale
    view of a removed or renamed collection (upstream renames are routine).
    Retiring keeps it browsable; the builder and validator only scan
    ``by_collection/``, so the retired copy stays invisible to them.
    """
    moves: list[tuple[str, str]] = []
    prefix = f'{INDEXES_PREFIX}/'
    for path in sorted(listing):
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        if relative.endswith('.jsonl') and '/' not in relative:
            benchmark = relative[: -len('.jsonl')]
            if benchmark in live_benchmarks:
                continue
        destination = f'{RETIRE_PREFIX}/{relative}'
        if destination in listing:
            stem, dot, suffix = destination.rpartition('.')
            destination = f'{stem}.{today}{dot}{suffix}'
        moves.append((path, destination))
    return moves


@dataclass(frozen=True)
class RetentionPlan:
    delete_dirs: tuple[str, ...]
    pinned: tuple[str, ...]
    unscanned: tuple[str, ...]


def plan_retention(
    manifests: Sequence[ManifestInfo],
    *,
    new_uuids: set[str],
    current_manifest_path: str | None,
    now: datetime,
    retain_days: int,
    keep_newest: int,
    scan_budget: int,
    uuids_of: Callable[[ManifestInfo], set[str]],
    new_manifest_path: str | None = None,
) -> RetentionPlan:
    """Classify old snapshot directories as trim, pin, or leave.

    Reference keys include aggregate UUIDs and companion object paths.
    A manifest older than the window is trimmed only if every object it
    references is still referenced by the new snapshot. A manifest holding
    any object the new snapshot dropped is the last index into that object
    and is pinned, so no object ever loses its row. Snapshots that could
    not be scanned within the budget are conservatively kept.
    """
    protected: set[str] = set()
    if new_manifest_path:
        protected.add(PurePosixPath(new_manifest_path).parent.as_posix())
    if current_manifest_path:
        protected.add(PurePosixPath(current_manifest_path).parent.as_posix())
    for info in sorted(manifests, key=lambda info: info.created, reverse=True)[
        :keep_newest
    ]:
        protected.add(info.dir_path)
    cutoff = now - timedelta(days=retain_days)
    old = sorted(
        (
            info
            for info in manifests
            if info.dir_path not in protected and info.created < cutoff
        ),
        key=lambda info: info.created,
        reverse=True,
    )
    delete: list[str] = []
    pinned: list[str] = []
    unscanned: list[str] = []
    for scanned, info in enumerate(old):
        if scanned >= scan_budget:
            unscanned.append(info.dir_path)
            continue
        try:
            uuids = uuids_of(info)
        except Exception as exc:  # noqa: BLE001 - unreadable snapshots are kept
            warnings.warn(
                f'Keeping unreadable snapshot {info.dir_path}: {exc}',
                stacklevel=2,
            )
            unscanned.append(info.dir_path)
            continue
        if any(uuid not in new_uuids for uuid in uuids):
            pinned.append(info.dir_path)
        else:
            delete.append(info.dir_path)
    return RetentionPlan(
        delete_dirs=tuple(delete),
        pinned=tuple(pinned),
        unscanned=tuple(unscanned),
    )


def batch_units(units: Sequence[Sequence[Any]], size: int) -> list[list[Any]]:
    """Greedy-fill commits with whole units; a unit is never split."""
    batches: list[list[Any]] = []
    current: list[Any] = []
    count = 0
    for unit in units:
        if current and count + len(unit) > size:
            batches.append(current)
            current = []
            count = 0
        current.extend(unit)
        count += len(unit)
    if current:
        batches.append(current)
    return batches


# -- publishing ------------------------------------------------------------


class FlatPublisher:
    """Commit whole units to the datastore with the ingestion's retries."""

    def __init__(
        self, api: HfApi, repo_id: str, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.batch_size = batch_size

    def publish(
        self, units: Sequence[Sequence[Any]], *, message: str, description: str
    ) -> int:
        batches = batch_units(units, self.batch_size)
        for index, batch in enumerate(batches, start=1):
            suffix = f' ({index}/{len(batches)})' if len(batches) > 1 else ''
            self._commit_batch(
                batch,
                message=f'{message}{suffix}',
                description=description,
            )
        return len(batches)

    def _commit_batch(
        self, operations: list[Any], *, message: str, description: str
    ) -> None:
        for attempt in range(1, store.COMMIT_ATTEMPTS + 1):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type='dataset',
                    operations=operations,
                    commit_message=message,
                    commit_description=description,
                )
                return
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                landed = self._landed(operations)
                if landed:
                    return
                if attempt < store.COMMIT_ATTEMPTS and store.is_commit_conflict(
                    exc
                ):
                    store.wait_before_retry(attempt)
                    continue
                raise FlatRebuildError(
                    f'could not commit to {self.repo_id} '
                    f'({type(exc).__name__}: {exc})'
                ) from exc

    def _landed(self, operations: Sequence[Any]) -> bool:
        """Verify every expected byte and deletion after a commit error.

        An unreadable result raises rather than guessing whether to retry.
        """
        try:
            files = set(
                self.api.list_repo_files(
                    repo_id=self.repo_id, repo_type='dataset'
                )
            )
            for operation in operations:
                if isinstance(operation, CommitOperationAdd):
                    if operation.path_in_repo not in files:
                        return False
                    local = self.api.hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type='dataset',
                        filename=operation.path_in_repo,
                    )
                    with operation.as_file() as expected:
                        expected_sha = hashlib.file_digest(
                            expected, 'sha256'
                        ).hexdigest()
                    with Path(local).open('rb') as actual:
                        actual_sha = hashlib.file_digest(
                            actual, 'sha256'
                        ).hexdigest()
                    if actual_sha != expected_sha:
                        return False
                elif operation.path_in_repo in files:
                    return False
            return True
        except Exception as exc:
            raise FlatRebuildError(
                f'Commit outcome is unverifiable for {self.repo_id}; stopping: {exc}'
            ) from exc


# -- orchestration ---------------------------------------------------------


def collect_manifests(
    api: HfApi, repo_id: str, listing: dict[str, int]
) -> list[ManifestInfo]:
    """Read every snapshot manifest under ``flat/manifests/``."""
    manifest_paths = sorted(
        path
        for path in listing
        if path.startswith(f'{MANIFESTS_PREFIX}/')
        and path.endswith('/manifest.json')
    )
    contents = download_bytes(api, repo_id, manifest_paths)
    infos: list[ManifestInfo] = []
    for path in manifest_paths:
        loaded = json.loads(contents[path].decode('utf-8'))
        infos.append(
            ManifestInfo(
                manifest_path=path,
                entries_path=loaded['entries_path'],
                created_at=loaded['created_at'],
                dir_path=str(PurePosixPath(path).parent),
            )
        )
    return infos


def operation_units(
    *,
    rows: Sequence[Row],
    build: BuildResult,
    contents: dict[str, bytes],
    index_adds: Sequence[str],
    rows_by_benchmark: dict[str, list[dict[str, Any]]],
    retire_moves: Sequence[tuple[str, str]],
    retire_contents: dict[str, bytes],
    new_manifest: dict[str, Any],
    delete_dirs: Sequence[str],
    manifests: Sequence[ManifestInfo],
    publish_snapshot: bool,
    by_legacy_bytes: bytes | None = None,
) -> list[list[Any]]:
    """Operation units in publication order.

    Objects, indexes, retire moves, the new snapshot, the pointer flip, and
    finally retention trims: any crash before the flip leaves the previous
    snapshot fully intact, and a flipped pointer is never invalidated by a
    later trim because the current snapshot is always protected. When the
    manifest core hash is unchanged (a run that only retires or trims), the
    snapshot files and pointer are already correct and are not re-committed.
    """
    units: list[list[Any]] = []
    row_by_uuid = {row.object_uuid: row for row in rows}
    for object_uuid in sorted(build.upload_aggregates):
        row = row_by_uuid[object_uuid]
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=row.object_path,
                    path_or_fileobj=contents[row.legacy_path],
                )
            ]
        )
    for object_uuid in sorted(build.upload_samples):
        row = row_by_uuid[object_uuid]
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=row.instance_level_path,
                    path_or_fileobj=contents[samples_path_for(row.legacy_path)],
                )
            ]
        )
    for benchmark in index_adds:
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=f'{INDEXES_PREFIX}/{benchmark}.jsonl',
                    path_or_fileobj=jsonl_text(
                        rows_by_benchmark[benchmark]
                    ).encode('utf-8'),
                )
            ]
        )
    if by_legacy_bytes is not None:
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=BY_LEGACY_PATH,
                    path_or_fileobj=by_legacy_bytes,
                )
            ]
        )
    for source, destination in retire_moves:
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=destination,
                    path_or_fileobj=retire_contents[source],
                ),
                CommitOperationDelete(path_in_repo=source),
            ]
        )
    if publish_snapshot:
        manifest_bytes = (
            json.dumps(new_manifest, indent=2, sort_keys=True) + '\n'
        ).encode('utf-8')
        entries_bytes = jsonl_text(rows).encode('utf-8')
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=new_manifest['entries_path'],
                    path_or_fileobj=entries_bytes,
                )
            ]
        )
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=new_manifest['manifest_path'],
                    path_or_fileobj=manifest_bytes,
                )
            ]
        )
        units.append(
            [
                CommitOperationAdd(
                    path_in_repo=LATEST_MANIFEST_PATH,
                    path_or_fileobj=manifest_bytes,
                )
            ]
        )
    by_dir = {info.dir_path: info for info in manifests}
    for dir_path in delete_dirs:
        info = by_dir[dir_path]
        units.append(
            [
                CommitOperationDelete(path_in_repo=info.manifest_path),
                CommitOperationDelete(path_in_repo=info.entries_path),
            ]
        )
    return units


def orchestrate(
    api: HfApi,
    repo_id: str,
    *,
    retain_days: int = DEFAULT_RETAIN_DAYS,
    keep_newest: int = DEFAULT_KEEP_NEWEST,
    scan_budget: int = DEFAULT_SCAN_BUDGET,
    allow_bootstrap: bool = False,
    dry_run: bool = False,
    verify: bool = False,
    now: datetime | None = None,
) -> RebuildReport:
    """Plan and, unless dry, publish one flat rebuild. Returns the report."""
    if not repo_id.strip():
        raise FlatRebuildError('repo_id must not be empty')
    started = time.monotonic()
    now = now or datetime.now(UTC)
    report = RebuildReport(repo_id=repo_id, dry_run=dry_run)

    snapshot = read_snapshot(api, repo_id)
    if snapshot is None and not allow_bootstrap:
        raise FlatRebuildError(
            f'{LATEST_MANIFEST_PATH} does not exist, so this would be a full '
            'initial build of every record in data/. Pass --allow-bootstrap '
            'to allow that.'
        )
    manifest, old_rows = snapshot or ({}, [])
    report.old_records = len(old_rows)

    listing = list_repo_sizes(api, repo_id)
    report.data_files = sum(
        1
        for path in listing
        if path.startswith(DATA_PREFIX) and path.endswith('.json')
    )

    diff = diff_against_snapshot(old_rows, listing)
    report.orphan_samples = len(diff.orphan_samples)
    needed = sorted(
        {
            *diff.new_paths,
            *diff.new_sample_paths,
            *diff.changed_aggregates,
            *(samples_path_for(path) for path in diff.changed_samples),
        }
    )
    contents = download_bytes(api, repo_id, needed)
    object_cache: dict[str, bytes] = {}

    def existing_object(object_path: str) -> bytes | None:
        """Fetch an existing flat object for semantic comparison."""
        if object_path not in listing:
            return None
        if object_path not in object_cache:
            object_cache.update(download_bytes(api, repo_id, [object_path]))
        return object_cache[object_path]

    build = build_rows(old_rows, diff, contents, existing_object, listing)
    if build.errors:
        raise FlatRebuildError(
            f'{len(build.errors)} record(s) cannot be flattened:\n'
            + '\n'.join(build.errors)
        )
    report.conflicts = build.conflicts
    rows = build.rows
    if verify:
        pending_objects = frozenset(
            [
                object_path_for(uuid, suffix='.json')
                for uuid in build.upload_aggregates
            ]
            + [
                object_path_for(uuid, suffix='_samples.jsonl')
                for uuid in build.upload_samples
            ]
        )
        checked, reserialized, drift = verify_rows(
            api, repo_id, rows, listing, pending_objects=pending_objects
        )
        report.verified_files = checked
        report.verified_reserialized = reserialized
        if drift:
            raise FlatRebuildError(
                f'{len(drift)} data file(s) no longer match the published '
                'snapshot:\n' + '\n'.join(drift)
            )
    report.added_records = build.added
    report.moved_records = build.moved
    report.removed_records = len(diff.removed_rows)
    report.records = len(rows)
    report.collections = len({row.benchmark for row in rows})

    new_manifest = manifest_for(rows, created_at=now.isoformat())
    report.manifest_core_sha256 = new_manifest['manifest_core_sha256']
    noop_core = bool(manifest) and manifest_core(manifest) == manifest_core(
        new_manifest
    )

    live_benchmarks = {row.benchmark for row in rows}
    retire_moves = plan_retire(listing, live_benchmarks, today=f'{now:%Y%m%d}')
    report.indexes_retired = tuple(source for source, _ in retire_moves)

    old_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_by_benchmark[row.benchmark].append(row.to_dict())
    new_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        new_by_benchmark[row.benchmark].append(row.to_dict())
    single = f'{INDEXES_PREFIX}/'
    present_indexes = {
        path[len(single) : -len('.jsonl')]
        for path in listing
        if path.startswith(single)
        and path.endswith('.jsonl')
        and '/' not in path[len(single) :]
    }
    index_adds: list[str] = []
    for benchmark in sorted(live_benchmarks):
        unchanged = old_by_benchmark.get(benchmark) == new_by_benchmark.get(
            benchmark
        )
        if f'{benchmark}' in present_indexes and unchanged:
            continue
        index_adds.append(benchmark)
    report.indexes_added = tuple(index_adds)

    manifests = collect_manifests(api, repo_id, listing)
    report.manifests_total = len(manifests)
    new_uuids = {row.object_uuid for row in rows}
    new_uuids.update(
        row.instance_level_path for row in rows if row.instance_level_available
    )

    def uuids_of(info: ManifestInfo) -> set[str]:
        local = api.hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=info.entries_path,
        )
        references: set[str] = set()
        for line in Path(local).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = Row.from_dict(json.loads(line))
            references.add(row.object_uuid)
            if row.instance_level_available:
                if not row.instance_level_path:
                    raise FlatRebuildError(
                        f'{info.entries_path}: missing samples object path'
                    )
                references.add(row.instance_level_path)
        return references

    retention = plan_retention(
        manifests,
        new_uuids=new_uuids,
        current_manifest_path=manifest.get('manifest_path'),
        new_manifest_path=new_manifest['manifest_path'],
        now=now,
        retain_days=retain_days,
        keep_newest=keep_newest,
        scan_budget=scan_budget,
        uuids_of=uuids_of,
    )
    report.manifests_trimmed = retention.delete_dirs
    report.manifests_pinned = retention.pinned
    report.manifests_unscanned = retention.unscanned

    if (
        noop_core
        and not index_adds
        and BY_LEGACY_PATH in listing
        and not retire_moves
        and not retention.delete_dirs
    ):
        report.noop = True
        report.duration_seconds = time.monotonic() - started
        return report

    retire_contents = download_bytes(
        api, repo_id, [source for source, _ in retire_moves]
    )
    by_legacy_bytes = jsonl_text(rows).encode('utf-8')
    if (
        BY_LEGACY_PATH in listing
        and existing_object(BY_LEGACY_PATH) == by_legacy_bytes
    ):
        by_legacy_bytes = None  # already published, byte for byte
    units = operation_units(
        rows=rows,
        build=build,
        contents=contents,
        index_adds=index_adds,
        rows_by_benchmark=dict(new_by_benchmark),
        retire_moves=retire_moves,
        retire_contents=retire_contents,
        new_manifest=new_manifest,
        delete_dirs=retention.delete_dirs,
        manifests=manifests,
        publish_snapshot=not noop_core,
        by_legacy_bytes=by_legacy_bytes,
    )
    report.uploads = sum(
        1
        for unit in units
        for operation in unit
        if isinstance(operation, CommitOperationAdd)
    )

    if dry_run:
        report.duration_seconds = time.monotonic() - started
        return report

    publisher = FlatPublisher(api, repo_id, batch_size=DEFAULT_BATCH_SIZE)
    report.commits = publisher.publish(
        units,
        message=(
            f'cron: flat rebuild {now:%Y-%m-%d} '
            f'({report.records} aggregate record(s))'
        ),
        description=(
            f'Rebuilt flat/ from data/ against snapshot '
            f'{manifest.get("manifest_core_sha256", "bootstrap")}: '
            f'+{report.added_records} new, -{report.removed_records} '
            f'removed, {len(retire_moves)} stale index(es) retired, '
            f'{len(retention.delete_dirs)} old snapshot(s) trimmed.'
        ),
    )
    report.duration_seconds = time.monotonic() - started
    return report


# -- reporting -------------------------------------------------------------


def summary_lines(report: RebuildReport) -> list[str]:
    lines = [
        f'## Flat rebuild — {report.repo_id}',
        '',
        f'- records: {report.old_records} -> {report.records} '
        f'(+{report.added_records} / -{report.removed_records}) across '
        f'{report.collections} collection(s)',
        f'- stale indexes retired: {len(report.indexes_retired)}',
        f'- snapshots: {report.manifests_total} total, '
        f'{len(report.manifests_trimmed)} trimmed, '
        f'{len(report.manifests_pinned)} pinned, '
        f'{len(report.manifests_unscanned)} unscanned',
        f'- manifest core sha256: `{report.manifest_core_sha256}`',
    ]
    if report.verified_files:
        lines.append(
            f'- deep verification: {report.verified_files} data file(s) '
            f'hashed, {report.verified_reserialized} re-serialization(s) '
            'accepted'
        )
    if report.noop:
        lines.append('- **no-op**: the published snapshot already matches')
    elif report.dry_run:
        lines.append(
            f'- **dry run**: {report.uploads} file(s) would be committed'
        )
    else:
        lines.append(
            f'- published in {report.commits} commit(s), '
            f'{report.uploads} file(s) uploaded'
        )
    if report.conflicts:
        lines.append(
            f'- **conflicts: {len(report.conflicts)}** record(s) excluded - '
            're-emitted under a known UUID with different content; the '
            'published objects stay, upstream should re-emit with fresh '
            'UUIDs'
        )
    for label, paths in (
        ('Excluded records', report.conflicts),
        ('Retired indexes', report.indexes_retired),
        ('Trimmed snapshots', report.manifests_trimmed),
        ('Pinned snapshots', report.manifests_pinned),
        ('Unscanned snapshots (kept)', report.manifests_unscanned),
    ):
        if paths:
            lines.append(f'\n<details><summary>{label}</summary>\n')
            lines.extend(f'- `{path}`' for path in paths)
            lines.append('\n</details>')
    return lines


def exit_code_for(report: RebuildReport) -> int:
    """0 clean or no-op, 2 completed with exclusions, per the house codes."""
    return 2 if report.conflicts else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--repo-id',
        default=os.environ.get('EEE_DATASTORE_REPO_ID')
        or DEFAULT_DATASTORE_REPO,
        help='EEE datastore dataset repository (unset or empty environment uses %(default)s).',
    )
    parser.add_argument(
        '--retain-days',
        type=int,
        default=DEFAULT_RETAIN_DAYS,
        help='Keep snapshots younger than this many days (default: %(default)s).',
    )
    parser.add_argument(
        '--keep-newest',
        type=int,
        default=DEFAULT_KEEP_NEWEST,
        help='Always keep this many newest snapshots (default: %(default)s).',
    )
    parser.add_argument(
        '--scan-budget',
        type=int,
        default=DEFAULT_SCAN_BUDGET,
        help='Max old snapshots to classify per run (default: %(default)s).',
    )
    parser.add_argument(
        '--allow-bootstrap',
        action='store_true',
        help='Allow a full initial build when no snapshot exists yet.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Plan and report, commit nothing.',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help=(
            'Deep pass: hash every inherited data file against the '
            'snapshot, so same-length edits the size diff cannot see fail '
            'loudly. Slow; intended for a periodic sweep, not every run.'
        ),
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        report = orchestrate(
            HfApi(),
            args.repo_id,
            retain_days=args.retain_days,
            keep_newest=args.keep_newest,
            scan_budget=args.scan_budget,
            allow_bootstrap=args.allow_bootstrap,
            dry_run=args.dry_run,
            verify=args.verify,
        )
    except FlatRebuildError as exc:
        if os.environ.get('GITHUB_ACTIONS'):
            print(f'::error::{exc}', file=sys.stderr)
        else:
            print(f'error: {exc}', file=sys.stderr)
        return 1
    lines = summary_lines(report)
    print('\n'.join(lines))
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with Path(summary_path).open('a', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    if report.conflicts and os.environ.get('GITHUB_ACTIONS'):
        print(
            f'::warning::{len(report.conflicts)} record(s) excluded from '
            'the flat view: re-emitted under known UUIDs with different '
            'content. See the run summary for the full list.'
        )
    print(f'done in {time.monotonic() - started:.1f}s')
    return exit_code_for(report)


if __name__ == '__main__':
    raise SystemExit(main())
