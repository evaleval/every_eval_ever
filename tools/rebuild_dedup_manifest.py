#!/usr/bin/env python3
"""Rebuild the root datastore semantic-dedup manifest from aggregate JSON."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from every_eval_ever.dedup import (
    FINGERPRINT_VERSION,
    compute_file_fingerprint,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datastore', type=Path)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datastore = args.datastore.resolve()
    data_root = datastore / 'data'
    output = (args.output or datastore / 'manifest.json').resolve()
    if not data_root.is_dir():
        raise ValueError(
            f'datastore data directory does not exist: {data_root}'
        )

    files: dict[str, dict[str, str]] = {}
    for local_path in sorted(data_root.rglob('*.json')):
        repo_path = local_path.relative_to(datastore).as_posix()
        files[repo_path] = {
            'fingerprint': compute_file_fingerprint(local_path),
        }
    if not files:
        raise ValueError(f'no aggregate JSON files found under {data_root}')

    manifest = {
        'fingerprint_version': FINGERPRINT_VERSION,
        'files': files,
    }
    validate_manifest(manifest, manifest_path=str(output))

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f'.{output.name}.', dir=output.parent
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(f'wrote {len(files)} fingerprints to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
