"""Transparent I/O for EEE aggregate and instance-level result files."""

from __future__ import annotations

import bz2
import gzip
import lzma
import sys
from collections import defaultdict
from collections.abc import Container, Iterable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Literal, TextIO, cast

Compression = Literal['none', 'gz', 'zst', 'bz2', 'xz', 'lz4']
ResultKind = Literal['aggregate', 'samples']
ResultKey = tuple[Path, str, ResultKind]

COMPRESSION_NONE: Compression = 'none'
COMPRESSION_CHOICES: tuple[Compression, ...] = (
    'none',
    'gz',
    'zst',
    'bz2',
    'xz',
    'lz4',
)
_COMPRESSION_BY_SUFFIX: dict[str, Compression] = {
    '.gz': 'gz',
    '.zst': 'zst',
    '.bz2': 'bz2',
    '.xz': 'xz',
    '.lz4': 'lz4',
}
_SUFFIX_BY_COMPRESSION = {
    compression: suffix
    for suffix, compression in _COMPRESSION_BY_SUFFIX.items()
}


class CodecUnavailableError(ImportError):
    """Raised when an optional compression codec is not installed."""


class CompressedReadError(OSError):
    """Raised when a compressed result cannot be decoded completely."""


def detect_compression(path: str | Path) -> Compression:
    """Return the compression codec implied by a path suffix."""
    return _COMPRESSION_BY_SUFFIX.get(
        Path(path).suffix.lower(), COMPRESSION_NONE
    )


def compression_suffix(compression: str) -> str:
    """Return the filesystem suffix for a supported compression codec."""
    if compression == COMPRESSION_NONE:
        return ''
    try:
        return _SUFFIX_BY_COMPRESSION[compression]
    except KeyError as exc:
        raise ValueError(
            f'unsupported compression {compression!r}; '
            f'choose from {COMPRESSION_CHOICES}'
        ) from exc


def normalize_compression(compression: str) -> Compression:
    """Validate and return a compression name."""
    if compression not in COMPRESSION_CHOICES:
        raise ValueError(
            f'unsupported compression {compression!r}; '
            f'choose from {COMPRESSION_CHOICES}'
        )
    return cast(Compression, compression)


def strip_compression_suffix(path: str | Path) -> Path:
    """Return a path with one recognized compression suffix removed."""
    path = Path(path)
    suffix = compression_suffix(detect_compression(path))
    if not suffix:
        return path
    return path.with_name(path.name[: -len(suffix)])


def strip_compression_suffix_text(path: str) -> str:
    """Return a POSIX-style path string without one compression suffix."""
    suffix = compression_suffix(detect_compression(path))
    return path[: -len(suffix)] if suffix else path


def add_compression_suffix(path: str | Path, compression: str) -> Path:
    """Append a supported compression suffix to an uncompressed path."""
    path = Path(path)
    compression = normalize_compression(compression)
    current = detect_compression(path)
    if current != COMPRESSION_NONE:
        if compression == current:
            return path
        raise ValueError(f'path is already compressed: {path}')
    suffix = compression_suffix(compression)
    return path.with_name(path.name + suffix) if suffix else path


def add_compression_suffix_text(path: str, compression: str) -> str:
    """Append a supported compression suffix to a repository path string."""
    compression = normalize_compression(compression)
    current = detect_compression(path)
    if current != COMPRESSION_NONE:
        if compression == current:
            return path
        raise ValueError(f'path is already compressed: {path}')
    return path + compression_suffix(compression)


def is_eee_result(path: str | Path) -> ResultKind | None:
    """Classify a plain or compressed EEE result by logical extension."""
    name = strip_compression_suffix(path).name.lower()
    if name.endswith('.jsonl'):
        return 'samples'
    if name.endswith('.json'):
        return 'aggregate'
    return None


def eee_uuid_stem(path: str | Path) -> str | None:
    """Return the logical filename stem shared by compression variants."""
    name = strip_compression_suffix(path).name
    lower = name.lower()
    if lower.endswith('_samples.jsonl'):
        return name[: -len('_samples.jsonl')]
    if lower.endswith('.jsonl'):
        return name[: -len('.jsonl')]
    if lower.endswith('.json'):
        return name[: -len('.json')]
    return None


def logical_result_key(path: str | Path) -> ResultKey | None:
    """Return directory, logical stem, and kind for a result path."""
    path = Path(path)
    kind = is_eee_result(path)
    stem = eee_uuid_stem(path)
    if kind is None or stem is None:
        return None
    return path.parent, stem, kind


def variant_paths(path: str | Path) -> tuple[Path, ...]:
    """Return every physical compression variant of one logical result."""
    logical = strip_compression_suffix(path)
    return tuple(
        add_compression_suffix(logical, compression)
        for compression in COMPRESSION_CHOICES
    )


def repo_variant_paths(path: str) -> tuple[str, ...]:
    """Return every repository-path variant of one logical result."""
    logical = strip_compression_suffix_text(path)
    return tuple(
        add_compression_suffix_text(logical, compression)
        for compression in COMPRESSION_CHOICES
    )


def present_repo_variants(
    path: str, available_files: Container[str]
) -> list[str]:
    """Return physical variants of a repository result that exist."""
    return [
        candidate
        for candidate in repo_variant_paths(path)
        if candidate in available_files
    ]


def existing_local_variants(path: str | Path) -> list[Path]:
    """Return physical variants beside a local result that exist."""
    return [candidate for candidate in variant_paths(path) if candidate.is_file()]


def iter_eee_results(roots: Iterable[str | Path]) -> Iterator[Path]:
    """Yield plain and compressed EEE result files below roots."""
    for root in roots:
        root = Path(root)
        if root.is_file():
            if is_eee_result(root) is not None:
                yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and is_eee_result(path) is not None:
                yield path


def find_duplicate_variants(
    paths: Iterable[str | Path],
) -> list[tuple[Path, str, ResultKind, list[Path]]]:
    """Return logical results that have more than one physical variant."""
    grouped: dict[ResultKey, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path in seen:
            continue
        seen.add(path)
        key = logical_result_key(path)
        if key is not None:
            grouped[key].append(path)
    return [
        (folder, stem, kind, sorted(variants))
        for (folder, stem, kind), variants in grouped.items()
        if len(variants) > 1
    ]


def _codec_install_message(codec: str) -> str:
    if codec == 'zst' and sys.version_info >= (3, 14):
        return (
            '.zst EEE files require the optional stdlib compression.zstd '
            'module; use a Python build with Zstandard support'
        )
    package = {'zst': 'backports.zstd', 'lz4': 'lz4'}[codec]
    return (
        f'.{codec} EEE files require the optional {package!r} package. '
        f'Install with: uv sync --extra {codec}'
    )


def _import_zstd():
    try:
        if sys.version_info >= (3, 14):
            from compression import zstd
        else:
            from backports import zstd
    except ImportError as exc:
        raise CodecUnavailableError(_codec_install_message('zst')) from exc
    return zstd


def _import_lz4_frame():
    try:
        import lz4.frame
    except ImportError as exc:
        raise CodecUnavailableError(_codec_install_message('lz4')) from exc
    return lz4.frame


@contextmanager
def open_eee_binary(
    path: str | Path, mode: str = 'rb'
) -> Iterator[BinaryIO]:
    """Open a plain or compressed EEE result as a binary stream."""
    if mode not in {'rb', 'wb'}:
        raise ValueError(f"mode must be 'rb' or 'wb'; got {mode!r}")

    path = Path(path)
    compression = detect_compression(path)
    if compression == 'none':
        with path.open(mode) as handle:
            yield handle
        return
    if compression == 'gz':
        if mode == 'rb':
            with gzip.open(path, mode) as handle:
                yield handle
        else:
            # Keep gzip publication/transcoding deterministic on every
            # supported Python version and do not embed a temporary filename.
            with path.open('wb') as raw_handle:
                with gzip.GzipFile(
                    filename='',
                    mode='wb',
                    fileobj=raw_handle,
                    mtime=0,
                ) as handle:
                    yield handle
        return
    if compression == 'bz2':
        with bz2.open(path, mode) as handle:
            yield handle
        return
    if compression == 'xz':
        with lzma.open(path, mode) as handle:
            yield handle
        return
    if compression == 'zst':
        with _import_zstd().open(path, mode=mode) as handle:
            yield handle
        return
    if compression == 'lz4':
        with _import_lz4_frame().open(path, mode=mode) as handle:
            yield handle
        return
    raise AssertionError(f'unhandled compression {compression!r}')


def open_eee_text(path: str | Path, mode: str = 'r') -> TextIO:
    """Open a plain or compressed EEE result as UTF-8 text."""
    if mode in {'r', 'rt'}:
        text_mode = 'rt'
    elif mode in {'w', 'wt'}:
        text_mode = 'wt'
    else:
        raise ValueError(
            f"mode must be 'r', 'rt', 'w', or 'wt'; got {mode!r}"
        )

    path = Path(path)
    compression = detect_compression(path)
    if compression == 'none':
        return path.open(text_mode, encoding='utf-8')
    if compression == 'gz':
        return gzip.open(path, text_mode, encoding='utf-8')
    if compression == 'bz2':
        return bz2.open(path, text_mode, encoding='utf-8')
    if compression == 'xz':
        return lzma.open(path, text_mode, encoding='utf-8')
    if compression == 'zst':
        return _import_zstd().open(path, mode=text_mode, encoding='utf-8')
    if compression == 'lz4':
        lz4_frame = _import_lz4_frame()
        return lz4_frame.open(path, mode=text_mode, encoding='utf-8')
    raise AssertionError(f'unhandled compression {compression!r}')


def _is_codec_exception(exc: Exception, compression: Compression) -> bool:
    if isinstance(exc, (EOFError, UnicodeError)):
        return True
    if isinstance(exc, OSError):
        # Codec libraries also use OSError for malformed streams, but genuine
        # filesystem errors carry errno and should stay ordinary I/O errors.
        if exc.errno is not None:
            return False
        return True
    if compression == 'xz' and isinstance(exc, lzma.LZMAError):
        return True
    module = type(exc).__module__
    if compression == 'zst':
        try:
            return isinstance(exc, _import_zstd().ZstdError)
        except CodecUnavailableError:
            return False
    if compression == 'lz4':
        # python-lz4 reports malformed frames as built-in RuntimeError rather
        # than a package-specific exception type.
        return isinstance(exc, RuntimeError) or module.startswith('lz4')
    return False


@contextmanager
def _normalize_read_errors(path: str | Path) -> Iterator[None]:
    """Translate codec-specific failures into one public read error."""
    try:
        yield
    except CodecUnavailableError:
        raise
    except Exception as exc:
        compression = detect_compression(path)
        if compression == COMPRESSION_NONE or not _is_codec_exception(
            exc, compression
        ):
            raise
        raise CompressedReadError(
            f'could not decode {Path(path).name} as {compression}: {exc}'
        ) from exc


def iter_eee_binary_chunks(
    path: str | Path, *, chunk_size: int = 1024 * 1024
) -> Iterator[bytes]:
    """Yield the uncompressed bytes of an EEE result without buffering it."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    with _normalize_read_errors(path):
        with open_eee_binary(path, 'rb') as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk


def read_eee_bytes(path: str | Path) -> bytes:
    """Read the complete uncompressed byte payload of an EEE result."""
    return b''.join(iter_eee_binary_chunks(path))


def read_eee_text(path: str | Path) -> str:
    """Read a complete EEE text file and normalize codec failures."""
    with _normalize_read_errors(path):
        with open_eee_text(path, 'r') as handle:
            return handle.read()


def iter_eee_text_lines(path: str | Path) -> Iterator[str]:
    """Iterate every line in an EEE result and normalize codec failures."""
    with _normalize_read_errors(path):
        with open_eee_text(path, 'r') as handle:
            yield from handle


def compress_bytes(content: bytes, compression: str) -> bytes:
    """Compress bytes deterministically for publication."""
    compression = normalize_compression(compression)
    if compression == 'none':
        return content
    if compression == 'gz':
        # GzipFile avoids the Python 3.12 gzip.compress(mtime=0) fast path,
        # whose header OS byte can vary with the underlying zlib/platform.
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as handle:
            handle.write(content)
        return buffer.getvalue()
    if compression == 'bz2':
        return bz2.compress(content)
    if compression == 'xz':
        return lzma.compress(content, format=lzma.FORMAT_XZ)
    if compression == 'zst':
        return _import_zstd().compress(content)
    if compression == 'lz4':
        return _import_lz4_frame().compress(content)
    raise AssertionError(f'unhandled compression {compression!r}')
