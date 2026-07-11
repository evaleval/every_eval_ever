"""Strict pre-download source indexing for incremental adapter runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


class SourceIndexError(ValueError):
    """Raised when source discovery or source-index data is ambiguous."""


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceIndexError(f'{context} must be a non-empty string')
    if value != value.strip():
        raise SourceIndexError(f'{context} must not contain outer whitespace')
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SourceIndexError(f'{context} must be an object')
    return value


def _require_field(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SourceIndexError(f'{context} is missing required field {key!r}')
    return mapping[key]


@dataclass(frozen=True)
class SourceCandidate:
    """Upstream item discovered without downloading its full payload."""

    adapter: str
    source_id: str
    revision: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.adapter, 'source candidate adapter')
        _require_nonempty_string(self.source_id, 'source candidate source_id')
        _require_nonempty_string(self.revision, 'source candidate revision')

    @property
    def key(self) -> str:
        payload = json.dumps(
            [self.adapter, self.source_id],
            ensure_ascii=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IndexedSource:
    candidate: SourceCandidate
    files: tuple[str, ...]


class DownloadAction(Enum):
    DOWNLOAD_NEW = 'download_new'
    DOWNLOAD_CHANGED = 'download_changed'
    SKIP_UNCHANGED = 'skip_unchanged'


@dataclass(frozen=True)
class SourceDecision:
    candidate: SourceCandidate
    action: DownloadAction
    existing_files: tuple[str, ...]
    previous_revision: str | None


@dataclass(frozen=True)
class SourceExecution:
    decision: SourceDecision
    files: tuple[str, ...]


@dataclass(frozen=True)
class SourceIndex:
    """Validated manifest index used before adapters download full payloads."""

    entries: Mapping[str, IndexedSource]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> SourceIndex:
        manifest = _require_mapping(manifest, 'manifest')
        manifest_files = _require_mapping(
            _require_field(manifest, 'files', 'manifest'), 'manifest.files'
        )
        raw_sources = _require_mapping(
            _require_field(manifest, 'sources', 'manifest'),
            'manifest.sources',
        )
        entries: dict[str, IndexedSource] = {}
        for raw_key, raw_entry in raw_sources.items():
            key = _require_nonempty_string(raw_key, 'manifest source key')
            if not _SHA256_RE.fullmatch(key):
                raise SourceIndexError(
                    f'manifest source key {key!r} must be a SHA-256 hex digest'
                )
            entry = _require_mapping(raw_entry, f'manifest.sources[{key!r}]')
            candidate = SourceCandidate(
                adapter=_require_nonempty_string(
                    _require_field(entry, 'adapter', f'manifest source {key}'),
                    f'manifest source {key}.adapter',
                ),
                source_id=_require_nonempty_string(
                    _require_field(
                        entry, 'source_id', f'manifest source {key}'
                    ),
                    f'manifest source {key}.source_id',
                ),
                revision=_require_nonempty_string(
                    _require_field(entry, 'revision', f'manifest source {key}'),
                    f'manifest source {key}.revision',
                ),
            )
            if candidate.key != key:
                raise SourceIndexError(
                    f'manifest source key {key!r} does not match adapter/source_id'
                )
            raw_files = _require_field(entry, 'files', f'manifest source {key}')
            if not isinstance(raw_files, list) or not raw_files:
                raise SourceIndexError(
                    f'manifest source {key}.files must be a non-empty array'
                )
            files = tuple(
                _require_nonempty_string(
                    file_path, f'manifest source {key}.files[]'
                )
                for file_path in raw_files
            )
            if len(set(files)) != len(files):
                raise SourceIndexError(
                    f'manifest source {key}.files contains duplicates'
                )
            missing_files = [
                path for path in files if path not in manifest_files
            ]
            if missing_files:
                raise SourceIndexError(
                    f'manifest source {key}.files references unknown accepted '
                    f'path {missing_files[0]!r}'
                )
            entries[key] = IndexedSource(candidate=candidate, files=files)
        return cls(entries=MappingProxyType(entries))

    def decide(self, candidate: SourceCandidate) -> SourceDecision:
        existing = (
            self.entries[candidate.key]
            if candidate.key in self.entries
            else None
        )
        if existing is None:
            return SourceDecision(
                candidate=candidate,
                action=DownloadAction.DOWNLOAD_NEW,
                existing_files=(),
                previous_revision=None,
            )
        if existing.candidate.revision == candidate.revision:
            return SourceDecision(
                candidate=candidate,
                action=DownloadAction.SKIP_UNCHANGED,
                existing_files=existing.files,
                previous_revision=existing.candidate.revision,
            )
        return SourceDecision(
            candidate=candidate,
            action=DownloadAction.DOWNLOAD_CHANGED,
            existing_files=existing.files,
            previous_revision=existing.candidate.revision,
        )

    def plan(
        self, candidates: Sequence[SourceCandidate]
    ) -> list[SourceDecision]:
        seen: set[str] = set()
        decisions: list[SourceDecision] = []
        for candidate in candidates:
            if candidate.key in seen:
                raise SourceIndexError(
                    'adapter discovery returned duplicate source identity '
                    f'{candidate.adapter}/{candidate.source_id}'
                )
            seen.add(candidate.key)
            decisions.append(self.decide(candidate))
        return decisions


def source_manifest_entry(
    candidate: SourceCandidate, files: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    """Build the manifest entry written after successful ingestion."""
    normalized_files = tuple(
        _require_nonempty_string(path, 'source manifest file path')
        for path in files
    )
    if not normalized_files:
        raise SourceIndexError(
            'source manifest entry requires at least one file'
        )
    if len(set(normalized_files)) != len(normalized_files):
        raise SourceIndexError(
            'source manifest entry file paths must be unique'
        )
    return (
        candidate.key,
        {
            'adapter': candidate.adapter,
            'source_id': candidate.source_id,
            'revision': candidate.revision,
            'files': list(normalized_files),
        },
    )


def execute_download_plan(
    decisions: Sequence[SourceDecision],
    download: Callable[[SourceCandidate], Sequence[str]],
) -> list[SourceExecution]:
    """Execute an incremental plan, calling ``download`` only when required."""
    executions: list[SourceExecution] = []
    for decision in decisions:
        if decision.action is DownloadAction.SKIP_UNCHANGED:
            if not decision.existing_files:
                raise SourceIndexError(
                    'unchanged source decision has no accepted files: '
                    f'{decision.candidate.adapter}/{decision.candidate.source_id}'
                )
            executions.append(
                SourceExecution(
                    decision=decision, files=decision.existing_files
                )
            )
            continue

        downloaded_files = tuple(download(decision.candidate))
        if not downloaded_files:
            raise SourceIndexError(
                'adapter download produced no files for '
                f'{decision.candidate.adapter}/{decision.candidate.source_id}'
            )
        for file_path in downloaded_files:
            _require_nonempty_string(file_path, 'downloaded source file path')
        if len(set(downloaded_files)) != len(downloaded_files):
            raise SourceIndexError(
                'adapter download returned duplicate file paths'
            )
        executions.append(
            SourceExecution(decision=decision, files=downloaded_files)
        )
    return executions
