"""Keep the source bytes an adapter converted, for later re-derivation.

A leaderboard scrape is not reproducible: the page changes, the API is
retired, and the record in the datastore becomes the only surviving evidence
of what the source said. This module snapshots what an adapter fetched so a
later correction can be checked against the input rather than guessed at.

Two kinds of entry, because two kinds of source. A ``payload`` entry holds
bytes we would otherwise lose (a leaderboard JSON, a scraped HTML page, a CSV
export), stored content-addressed as ``<sha256><ext>``. A ``pointer`` entry
names something already durably hosted and addressable at a revision (a
Hugging Face dataset, a git commit); a second copy of a pinned HF dataset buys
nothing, so only the reference and its resolved revision are recorded. The
revision is what makes that trade sound, so a pointer whose commit will not
resolve is dropped rather than written without one.

Capture is off unless a sink is active, so adapters behave identically when
run by hand. Automation activates one by setting :data:`CAPTURE_DIR_ENV`.
Capture never fails a conversion: an unwritable sink degrades and says so,
because losing a snapshot is bad and losing the refresh it came from is worse.
The cron reads what was dropped and refuses to publish records whose source
was not kept, so a degraded capture still stops a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

#: Directory automation points a run's snapshots at.
CAPTURE_DIR_ENV = 'EEE_RAW_CAPTURE_DIR'
#: Optional overrides, in whole megabytes.
MAX_PAYLOAD_MB_ENV = 'EEE_RAW_CAPTURE_MAX_PAYLOAD_MB'
MAX_TOTAL_MB_ENV = 'EEE_RAW_CAPTURE_MAX_TOTAL_MB'

MANIFEST_NAME = 'manifest.jsonl'
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 512 * 1024 * 1024

_EXTENSIONS = {
    'application/json': '.json',
    'application/ld+json': '.json',
    'text/json': '.json',
    'text/csv': '.csv',
    'application/csv': '.csv',
    'text/html': '.html',
    'application/xhtml+xml': '.html',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'application/xml': '.xml',
    'text/xml': '.xml',
    'application/x-yaml': '.yaml',
    'text/yaml': '.yaml',
}


def extension_for(content_type: str | None) -> str:
    """Return the file extension to store a payload of this media type under."""
    if not content_type:
        return '.bin'
    media_type = content_type.split(';', 1)[0].strip().lower()
    return _EXTENSIONS.get(media_type, '.bin')


class RawSink:
    """Collects one run's raw source payloads under a single directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.root = Path(root)
        self.max_payload_bytes = max_payload_bytes
        self.max_total_bytes = max_total_bytes
        self.total_bytes = 0
        self.errors: list[str] = []
        #: digest -> the filename actually written for it, so a second
        #: sighting under another content type points at the file that
        #: exists rather than the one its own extension would name.
        self._seen: dict[str, str] = {}

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def degraded(self) -> bool:
        """Whether any capture was dropped or could not be written."""
        return bool(self.errors)

    def record(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None = None,
        label: str | None = None,
    ) -> str | None:
        """Snapshot one fetched payload; return its sha256, or ``None``.

        Returns ``None`` when the payload was not stored, either because it is
        over a size cap or because the sink could not be written. Both cases
        are recorded so a run never looks complete when it is not.
        """
        digest = hashlib.sha256(content).hexdigest()
        entry: dict[str, Any] = {
            'kind': 'payload',
            'sha256': digest,
            'url': url,
            'bytes': len(content),
        }
        if content_type:
            entry['content_type'] = content_type
        if label:
            entry['label'] = label

        stored = self._seen.get(digest)
        if stored is not None:
            # Same bytes, already stored. Keep the line so the manifest still
            # records that this URL served them, but point it at the name the
            # bytes were written under. The same payload served as JSON and
            # again as HTML would otherwise name a .html file nobody wrote.
            #
            # Asked before the caps, because storing this costs nothing: the
            # file is already on disk and its bytes are already counted. Below
            # the caps the ordering was invisible; at them it decided whether
            # a run went red for keeping a byte it had kept an hour earlier.
            entry['path'] = stored
            entry['duplicate'] = True
            self._append(entry)
            return digest

        if len(content) > self.max_payload_bytes:
            return self._drop(
                entry, f'payload exceeds {self.max_payload_bytes} bytes'
            )
        if self.total_bytes + len(content) > self.max_total_bytes:
            return self._drop(
                entry, f'run exceeds {self.max_total_bytes} bytes of raw data'
            )

        name = f'{digest}{extension_for(content_type)}'
        entry['path'] = name

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / name
            if not target.exists():
                target.write_bytes(content)
        except OSError as exc:
            return self._drop(entry, f'could not write snapshot: {exc}')

        self._seen[digest] = name
        self.total_bytes += len(content)
        self._append(entry)
        return digest

    def record_pointer(
        self,
        *,
        kind: str,
        reference: str,
        revision: str | None = None,
        url: str | None = None,
        label: str | None = None,
        note: str | None = None,
        revision_required: bool = False,
    ) -> None:
        """Record a source that is already durably hosted at a revision.

        Use this instead of :meth:`record` for Hugging Face datasets and git
        repositories: re-downloading and re-storing content that is already
        addressable at a commit costs bandwidth and storage and adds nothing.

        That bargain only holds while the revision is known. A pointer with no
        commit behind it names a moving target, so ``revision_required``
        callers drop the entry instead of writing one, and the run fails the
        same way it does for payload bytes nobody kept. The alternative is a
        record whose source says "this dataset, at whatever it said that
        night", which cannot be checked against anything later.
        """
        entry: dict[str, Any] = {
            'kind': 'pointer',
            'pointer_kind': kind,
            'reference': reference,
        }
        if revision:
            entry['revision'] = revision
        if url:
            entry['url'] = url
        if label:
            entry['label'] = label
        if note:
            entry['note'] = note
        if revision_required and not revision:
            reason = 'no source revision resolved'
            if note:
                reason = f'{reason} ({note})'
            self._drop(entry, reason)
            return
        self._append(entry)

    def entries(self) -> list[dict[str, Any]]:
        """Return the manifest as parsed objects, or ``[]`` if not written."""
        if not self.manifest_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.manifest_path.read_text(
                encoding='utf-8'
            ).splitlines()
            if line.strip()
        ]

    def _drop(self, entry: dict[str, Any], reason: str) -> None:
        entry = {**entry, 'kind': 'dropped', 'reason': reason}
        entry.pop('path', None)
        self.errors.append(f'{entry.get("url") or "?"}: {reason}')
        print(f'raw capture: {reason} ({entry.get("url")})', file=sys.stderr)
        self._append(entry)
        return None

    def _append(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')
        except OSError as exc:
            message = f'could not append to raw manifest: {exc}'
            if message not in self.errors:
                self.errors.append(message)
                print(f'raw capture: {message}', file=sys.stderr)


_sink: RawSink | None = None
_sink_root: str | None = None
_sink_is_explicit = False


def _positive_mb(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        megabytes = int(value)
    except ValueError:
        megabytes = 0
    if megabytes <= 0:
        # Read from active_sink(), which the shared fetch helpers call on
        # every request. Raising here would let a typo in a workflow variable
        # crash the conversion this module only exists to snapshot.
        print(
            f'raw capture: ignoring {name}={value!r}, '
            'expected a positive whole number of MB',
            file=sys.stderr,
        )
        return None
    return megabytes * 1024 * 1024


def active_sink() -> RawSink | None:
    """Return the active sink, creating one from the environment if set.

    Reading the environment lazily is what lets an adapter capture without
    knowing capture exists: automation sets :data:`CAPTURE_DIR_ENV` on the
    subprocess and the shared HTTP helpers pick it up on their first call.
    """
    global _sink, _sink_root
    if _sink_is_explicit:
        return _sink
    root = os.environ.get(CAPTURE_DIR_ENV) or None
    if root is None:
        _sink = None
        _sink_root = None
        return None
    if _sink is None or _sink_root != root:
        _sink = RawSink(
            root,
            max_payload_bytes=(
                _positive_mb(MAX_PAYLOAD_MB_ENV) or DEFAULT_MAX_PAYLOAD_BYTES
            ),
            max_total_bytes=(
                _positive_mb(MAX_TOTAL_MB_ENV) or DEFAULT_MAX_TOTAL_BYTES
            ),
        )
        _sink_root = root
    return _sink


def activate(root: Path | str, **kwargs: Any) -> RawSink:
    """Activate a sink explicitly, for in-process callers and tests.

    An explicit sink takes precedence over :data:`CAPTURE_DIR_ENV` until
    :func:`deactivate` is called.
    """
    global _sink, _sink_root, _sink_is_explicit
    _sink = RawSink(root, **kwargs)
    _sink_root = None
    _sink_is_explicit = True
    return _sink


def deactivate() -> None:
    """Turn capture off and forget any explicit sink."""
    global _sink, _sink_root, _sink_is_explicit
    _sink = None
    _sink_root = None
    _sink_is_explicit = False


def record(
    *,
    url: str,
    content: bytes,
    content_type: str | None = None,
    label: str | None = None,
) -> str | None:
    """Snapshot a payload if capture is on; otherwise do nothing."""
    sink = active_sink()
    if sink is None:
        return None
    return sink.record(
        url=url, content=content, content_type=content_type, label=label
    )


def record_pointer(
    *,
    kind: str,
    reference: str,
    revision: str | None = None,
    url: str | None = None,
    label: str | None = None,
    note: str | None = None,
    revision_required: bool = False,
) -> None:
    """Record a durable source reference if capture is on."""
    sink = active_sink()
    if sink is None:
        return
    sink.record_pointer(
        kind=kind,
        reference=reference,
        revision=revision,
        url=url,
        label=label,
        note=note,
        revision_required=revision_required,
    )


def record_hf_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    label: str | None = None,
) -> None:
    """Record a Hugging Face dataset source at its resolved commit.

    Resolving the commit is what turns "we read this dataset" into something
    re-derivable, so a lookup that fails drops the entry rather than writing
    the requested revision instead. ``main`` in a manifest reads as an answer
    while naming whatever that branch points at today, which is the one thing
    a pointer must never do. The adapter still finishes; the run does not
    publish. Nothing is downloaded and no request is made when capture is off.
    """
    sink = active_sink()
    if sink is None:
        return

    resolved = None
    note = None
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, revision=revision)
        resolved = getattr(info, 'sha', None)
        if not resolved:
            note = 'the Hub returned no commit for this dataset'
    except Exception as exc:  # noqa: BLE001 - provenance must not break a run
        note = f'commit not resolved: {type(exc).__name__}: {exc}'
    if note and revision:
        note = f'{note}; requested {revision}'

    sink.record_pointer(
        kind='hf_dataset',
        reference=repo_id,
        revision=resolved,
        url=f'https://huggingface.co/datasets/{repo_id}',
        label=label,
        note=note,
        revision_required=True,
    )


def record_git_checkout(
    repo_url: str,
    checkout: Path | str,
    *,
    ref: str | None = None,
    label: str | None = None,
) -> None:
    """Record the exact commit a cloned working copy is sitting on.

    A checkout whose commit cannot be read is not evidence of anything, so it
    is dropped for the same reason a dataset with no resolved commit is: the
    records converted from it would name a source nobody can return to.
    """
    sink = active_sink()
    if sink is None:
        return

    import subprocess

    revision = None
    note = None
    try:
        completed = subprocess.run(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        revision = completed.stdout.strip() or None
    except Exception as exc:  # noqa: BLE001 - provenance must not break a run
        note = f'commit not resolved: {type(exc).__name__}: {exc}'

    if ref:
        note = f'ref={ref}' if note is None else f'ref={ref}; {note}'
    sink.record_pointer(
        kind='git',
        reference=repo_url,
        revision=revision,
        url=repo_url,
        label=label,
        note=note,
        revision_required=True,
    )


__all__ = [
    'CAPTURE_DIR_ENV',
    'DEFAULT_MAX_PAYLOAD_BYTES',
    'DEFAULT_MAX_TOTAL_BYTES',
    'MANIFEST_NAME',
    'MAX_PAYLOAD_MB_ENV',
    'MAX_TOTAL_MB_ENV',
    'RawSink',
    'activate',
    'active_sink',
    'deactivate',
    'extension_for',
    'record',
    'record_git_checkout',
    'record_hf_dataset',
    'record_pointer',
]
