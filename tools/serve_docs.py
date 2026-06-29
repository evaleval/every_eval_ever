#!/usr/bin/env python3
"""Serve the Pandoc docs locally and rebuild them when source files change."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_docs

DEFAULT_BASE_URL = '/projects/every-eval-ever/docs'
DEFAULT_SITE_URL = 'https://evalevalai.com'
DEFAULT_SITE_ROOT = ROOT / '_site'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 4173
DEFAULT_POLL_INTERVAL = 1.0

FileFingerprint = tuple[int, int] | None


@dataclass(frozen=True)
class DevServerConfig:
    base_url: str
    site_url: str
    site_root: Path
    host: str
    port: int
    poll_interval: float


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def log(message: str, *, stream=sys.stdout) -> None:
    stamp = datetime.now().strftime('%H:%M:%S')
    print(f'[{stamp}] {message}', file=stream, flush=True)


def output_dir_for_base_url(site_root: Path, base_url: str) -> Path:
    normalized_base = build_docs.normalize_base_url(base_url)
    if not normalized_base:
        return site_root
    return site_root / normalized_base.lstrip('/')


def watched_paths(extra_paths: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    manifest = build_docs.source_manifest()
    source_files = manifest.get('source_files')
    if not isinstance(source_files, list) or not all(
        isinstance(item, str) for item in source_files
    ):
        raise build_docs.DocsBuildError(
            'source manifest must contain a source_files list.'
        )

    paths = [ROOT / item for item in source_files]
    for relative_path in ('pyproject.toml', 'uv.lock'):
        dependency_path = ROOT / relative_path
        if dependency_path.is_file():
            paths.append(dependency_path)
    paths.extend(extra_paths)
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def file_fingerprint(path: Path) -> FileFingerprint:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    return (stat.st_mtime_ns, stat.st_size)


def source_snapshot(paths: tuple[Path, ...]) -> dict[Path, FileFingerprint]:
    return {path: file_fingerprint(path) for path in paths}


def start_server(config: DevServerConfig) -> ReusableThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(config.site_root))
    server = ReusableThreadingHTTPServer((config.host, config.port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def build_once(config: DevServerConfig) -> Path:
    output_dir = output_dir_for_base_url(config.site_root, config.base_url)
    build_docs.build_docs(config.base_url, config.site_url, output_dir)
    return output_dir


def refresh_watched_paths(
    current_paths: tuple[Path, ...]
) -> tuple[Path, ...]:
    try:
        return watched_paths()
    except build_docs.DocsBuildError as exc:
        log(f'watch manifest validation failed: {exc}', stream=sys.stderr)
        return current_paths


def serve(config: DevServerConfig) -> int:
    log('building docs')
    try:
        output_dir = build_once(config)
        paths = watched_paths()
    except build_docs.DocsBuildError as exc:
        log(f'initial build failed: {exc}', stream=sys.stderr)
        return 1

    snapshot = source_snapshot(paths)
    url = (
        f'http://{config.host}:{config.port}'
        f'{build_docs.site_path(build_docs.normalize_base_url(config.base_url), "/")}'
    )
    log(f'serving {output_dir} at {url}')
    log(f'watching {len(paths)} source files')
    server = start_server(config)

    try:
        while True:
            time.sleep(config.poll_interval)
            paths = refresh_watched_paths(paths)
            next_snapshot = source_snapshot(paths)
            if next_snapshot == snapshot:
                continue

            log('source change detected; rebuilding docs')
            try:
                output_dir = build_once(config)
            except build_docs.DocsBuildError as exc:
                snapshot = next_snapshot
                log(
                    'rebuild failed; serving the last successful artifact. '
                    f'Fix the source error to rebuild again: {exc}',
                    stream=sys.stderr,
                )
                continue

            paths = watched_paths()
            snapshot = source_snapshot(paths)
            log(f'rebuild complete: {output_dir}')
    except KeyboardInterrupt:
        log('stopping docs server')
    finally:
        try:
            server.shutdown()
        except KeyboardInterrupt:
            pass
        server.server_close()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Serve the Every Eval Ever Pandoc docs with live rebuilds.'
    )
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--site-url', default=DEFAULT_SITE_URL)
    parser.add_argument('--site-root', type=Path, default=DEFAULT_SITE_ROOT)
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help='Seconds between source file checks.',
    )
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error('--poll-interval must be greater than zero.')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535.')
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = DevServerConfig(
        base_url=args.base_url,
        site_url=args.site_url,
        site_root=args.site_root,
        host=args.host,
        port=args.port,
        poll_interval=args.poll_interval,
    )
    return serve(config)


if __name__ == '__main__':
    raise SystemExit(main())
