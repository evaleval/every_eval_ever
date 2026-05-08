"""I/O helpers for EEE result files with transparent compression.

EEE result files come in two kinds:

- aggregate: ``<uuid>.json``
- per-instance samples: ``<uuid>_samples.jsonl``

This module recognizes both kinds *and* their compressed forms, using the
codec set the HuggingFace Hub already auto-decompresses
(https://huggingface.co/docs/hub/en/datasets-adding#file-formats):

  ``.gz``, ``.zst``, ``.bz2``, ``.xz``, ``.lz4``

``.zip`` is intentionally not supported as a stream codec — it is an
archive container, and conflicts with the duplicate-variant rule below.

All EEE-side tooling (validation, discovery, manifest generation,
viewer-parquet generation, conversion writers) should route file open and
file-suffix recognition through this module so the compressed forms behave
as transparent equivalents of the plain forms.

Duplicate-variant rule
----------------------

For any ``(folder, uuid, kind)`` triple, at most one physical variant may
be present. ``find_duplicate_variants`` enumerates violations; the
validator surfaces them as ``duplicate_variant`` errors.

Backwards compatibility
-----------------------

The default ``compression='none'`` everywhere preserves existing
behaviour — uncompressed plain ``.json`` and ``_samples.jsonl`` are still
the default written and read forms. Compression is opt-in.
"""

from __future__ import annotations

import io as _io
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TextIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: No compression — write/read plain UTF-8 text.
COMPRESSION_NONE = 'none'
COMPRESSION_GZ = 'gz'
COMPRESSION_ZST = 'zst'
COMPRESSION_BZ2 = 'bz2'
COMPRESSION_XZ = 'xz'
COMPRESSION_LZ4 = 'lz4'

#: All accepted compression names. ``COMPRESSION_NONE`` first so it is the
#: documented default in CLI help.
COMPRESSION_CHOICES: tuple[str, ...] = (
    COMPRESSION_NONE,
    COMPRESSION_GZ,
    COMPRESSION_ZST,
    COMPRESSION_BZ2,
    COMPRESSION_XZ,
    COMPRESSION_LZ4,
)

# Map filesystem suffix (lowercase, leading dot) to compression name.
_COMPRESSION_BY_SUFFIX: dict[str, str] = {
    '.gz': COMPRESSION_GZ,
    '.zst': COMPRESSION_ZST,
    '.bz2': COMPRESSION_BZ2,
    '.xz': COMPRESSION_XZ,
    '.lz4': COMPRESSION_LZ4,
}

# Map compression name to its filesystem suffix (with leading dot).
_SUFFIX_BY_COMPRESSION: dict[str, str] = {
    name: suf for suf, name in _COMPRESSION_BY_SUFFIX.items()
}

# Canonical filename suffixes for each kind. The kind detector matches on
# the *first* listed suffix that ends the filename (after stripping any
# compression suffix). The ordering matters: ``.jsonl`` must be checked
# before ``.json`` because ``.jsonl`` does not end in ``.json`` — but for
# kinds that share a final extension (none currently do) this would matter.
_KIND_SUFFIXES: dict[str, tuple[str, ...]] = {
    'samples': ('_samples.jsonl', '.jsonl'),
    'aggregate': ('.json',),
}

# Order in which kinds are tried by the detector.
_KIND_ORDER: tuple[str, ...] = ('samples', 'aggregate')


# ---------------------------------------------------------------------------
# Suffix recognition
# ---------------------------------------------------------------------------


def detect_compression(path: Path | str) -> str:
    """Return the compression name implied by ``path``'s trailing suffix.

    Returns one of :data:`COMPRESSION_CHOICES`. ``COMPRESSION_NONE`` for
    paths whose final suffix is not a recognized compression suffix.
    """
    p = Path(path)
    return _COMPRESSION_BY_SUFFIX.get(p.suffix.lower(), COMPRESSION_NONE)


def _strip_compression_suffix(name: str) -> str:
    """Return ``name`` with one trailing compression suffix removed (if any)."""
    for suf in _COMPRESSION_BY_SUFFIX:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return name


def is_eee_result(path: Path | str) -> str | None:
    """Classify a path as ``'aggregate'``, ``'samples'``, or ``None``.

    Recognizes both plain and compressed forms. Lenient about the bare
    ``.jsonl`` extension to keep parity with existing converter outputs
    that don't always include the ``_samples`` prefix (lm-eval emits
    ``samples_<task>_<datetime>.jsonl`` when no UUID is supplied).
    Returns ``None`` for any path that does not match an EEE result
    filename convention.
    """
    name = _strip_compression_suffix(Path(path).name)
    for kind in _KIND_ORDER:
        for suffix in _KIND_SUFFIXES[kind]:
            if name.endswith(suffix):
                return kind
    return None


def eee_uuid_stem(path: Path | str) -> str | None:
    """Return the bare UUID portion of an EEE result filename, or ``None``.

    Strips the compression suffix (if any) and the kind-specific suffix.
    For ``abc_samples.jsonl`` returns ``'abc'``; for ``samples_task.jsonl``
    (which lacks the canonical ``_samples`` prefix) returns
    ``'samples_task'``. The duplicate-variant rule keys on this stem, so
    files that share a stem and a kind across compressed/uncompressed
    variants will be flagged as collisions regardless of which suffix
    convention is in use.
    """
    name = _strip_compression_suffix(Path(path).name)
    for kind in _KIND_ORDER:
        for suffix in _KIND_SUFFIXES[kind]:
            if name.endswith(suffix):
                return name[: -len(suffix)]
    return None


def add_compression_suffix(path: Path | str, compression: str) -> Path:
    """Return ``path`` with the requested compression suffix appended.

    ``add_compression_suffix(Path('a.json'), 'gz')`` -> ``Path('a.json.gz')``.
    ``compression='none'`` returns the path unchanged.
    """
    p = Path(path)
    if compression == COMPRESSION_NONE:
        return p
    if compression not in _SUFFIX_BY_COMPRESSION:
        raise ValueError(
            f'unsupported compression {compression!r}; '
            f'choose from {COMPRESSION_CHOICES}'
        )
    return p.with_name(p.name + _SUFFIX_BY_COMPRESSION[compression])


# ---------------------------------------------------------------------------
# Open helper
# ---------------------------------------------------------------------------


def _missing_codec_msg(codec: str, package: str, extra: str) -> str:
    return (
        f'Reading/writing .{codec} EEE files requires the {package!r} '
        f"package. Install with: pip install 'every-eval-ever[{extra}]'"
    )


def open_eee_text(path: Path | str, mode: str = 'r') -> TextIO:
    """Open an EEE result file (json or jsonl) for text I/O.

    The compression codec is inferred from ``path``'s trailing suffix.
    ``mode`` may be ``'r'`` / ``'rt'`` for reading or ``'w'`` / ``'wt'``
    for writing. UTF-8 is assumed throughout.

    Raises ``ImportError`` (with an actionable message) if the path
    indicates a codec whose backing dependency is not installed.
    """
    if mode in ('r', 'rt'):
        text_mode = 'rt'
    elif mode in ('w', 'wt'):
        text_mode = 'wt'
    else:
        raise ValueError(
            f"open_eee_text mode must be 'r', 'rt', 'w', or 'wt'; got {mode!r}"
        )
    p = Path(path)
    cs = detect_compression(p)

    if cs == COMPRESSION_NONE:
        return open(p, text_mode, encoding='utf-8')
    if cs == COMPRESSION_GZ:
        import gzip
        return gzip.open(p, text_mode, encoding='utf-8')
    if cs == COMPRESSION_BZ2:
        import bz2
        return bz2.open(p, text_mode, encoding='utf-8')
    if cs == COMPRESSION_XZ:
        import lzma
        return lzma.open(p, text_mode, encoding='utf-8')
    if cs == COMPRESSION_ZST:
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in env w/o dep
            raise ImportError(_missing_codec_msg('zst', 'zstandard', 'zst')) from exc
        if text_mode == 'rt':
            reader = zstandard.ZstdDecompressor().stream_reader(open(p, 'rb'))
            return _io.TextIOWrapper(reader, encoding='utf-8')
        writer = zstandard.ZstdCompressor().stream_writer(open(p, 'wb'))
        return _ZstdTextWriter(writer, encoding='utf-8')
    if cs == COMPRESSION_LZ4:
        try:
            import lz4.frame  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_missing_codec_msg('lz4', 'lz4', 'lz4')) from exc
        binmode = 'rb' if text_mode == 'rt' else 'wb'
        return _io.TextIOWrapper(lz4.frame.open(p, mode=binmode), encoding='utf-8')

    # _COMPRESSION_BY_SUFFIX is exhaustive over COMPRESSION_CHOICES, so this
    # branch is unreachable.
    raise AssertionError(f'unhandled compression {cs!r}')  # pragma: no cover


class _ZstdTextWriter(_io.TextIOWrapper):
    """TextIOWrapper that flushes the underlying zstd stream_writer on close.

    ``ZstdCompressor.stream_writer`` requires an explicit ``close()`` /
    ``flush(zstandard.FLUSH_FRAME)`` to finalize the frame; without that the
    written file is silently truncated. Wrapping in a TextIOWrapper alone
    isn't enough — close cascades to the wrapped writer, but ``flush()``
    on the TextIOWrapper alone does not.
    """

    def close(self) -> None:  # pragma: no cover - exercised only with zstandard
        try:
            self.flush()
        finally:
            super().close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def iter_eee_results(roots: Iterable[Path | str]) -> Iterator[Path]:
    """Yield every EEE result file under each root (recursive).

    Files are yielded in deterministic (sorted) order. Roots that are
    themselves EEE result files are yielded as-is.
    """
    for root in roots:
        rp = Path(root)
        if rp.is_file():
            if is_eee_result(rp) is not None:
                yield rp
            continue
        if not rp.is_dir():
            continue
        for path in sorted(rp.rglob('*')):
            if path.is_file() and is_eee_result(path) is not None:
                yield path


def find_duplicate_variants(
    paths: Iterable[Path | str],
) -> list[tuple[Path, str, str, list[Path]]]:
    """Detect ``(folder, uuid, kind)`` groups with more than one variant.

    Each result is ``(folder, uuid_stem, kind, [physical_paths])``.
    """
    by_group: dict[tuple[Path, str, str], list[Path]] = defaultdict(list)
    seen_files: set[Path] = set()
    for path in paths:
        p = Path(path)
        if p in seen_files:
            continue
        seen_files.add(p)
        kind = is_eee_result(p)
        if kind is None:
            continue
        stem = eee_uuid_stem(p)
        if stem is None:
            continue
        by_group[(p.parent, stem, kind)].append(p)
    return [
        (folder, stem, kind, sorted(variants))
        for (folder, stem, kind), variants in by_group.items()
        if len(variants) > 1
    ]
