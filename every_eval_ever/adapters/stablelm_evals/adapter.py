"""Publish the lm-eval harness outputs Stability AI committed alongside StableLM.

This adapter is a **wrapper**, not a converter. `Stability-AI/StableLM` commits raw
lm-evaluation-harness result files under `evals/`, and the in-tree `lm_eval`
converter already reads that format. All this file adds is the three things a
public repository of harness JSON needs before the converter can be pointed at it:

1. **Pin the source.** Resolve the revision to a commit sha before fetching, so
   every record cites bytes that cannot move.
2. **Repair the model identity.** The converter takes `model_info.id` from
   `config.model_args`'s `pretrained=` value and refuses an id with no publishing
   namespace, which is correct -- a placeholder developer would route unrelated
   models into one directory. One file here was run from a local checkout and
   needs its org supplied.
3. **Pin one collection.** Left alone the converter files each task into its own
   bare collection (`data/sciq/`, `data/piqa/`, ...), which mixes these numbers
   with every other source's records for the same benchmark and loses the
   provenance. `collection_override` keeps the source together.

**Copying this for another repository of harness JSON** means changing `SOURCE_*`,
`COLLECTION`, `list_result_files`, and `ORGLESS_MODEL_ORG`. Nothing else here is
specific to StableLM, and nothing here re-implements conversion.

    uv run python -m every_eval_ever.adapters.stablelm_evals.adapter \
        --output-dir /tmp/stablelm-evals-smoke/data/stablelm-evals
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
from every_eval_ever.helpers import (
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    save_failure_report,
)

SRC = 'stablelm_evals'
COLLECTION = 'stablelm-evals'

OWNER_REPO = 'Stability-AI/StableLM'
PROJECT_URL = f'https://github.com/{OWNER_REPO}'
COMMITS_API = f'https://api.github.com/repos/{OWNER_REPO}/commits/'
TREE_API = f'https://api.github.com/repos/{OWNER_REPO}/git/trees/'
RAW_BASE = f'https://raw.githubusercontent.com/{OWNER_REPO}/'

#: The harness that produced these files. `versions` in every result file is the
#: v0.3-era integer form (`{"arc_challenge": 0}`), not the v0.4 float form, and the
#: metric keys carry no `,none` suffix, so the runs predate v0.4.
EVAL_LIBRARY_NAME = 'lm-evaluation-harness'
EVAL_LIBRARY_VERSION = '0.3'

#: `evals/open_llm_leaderboard/` holds per-task files for two models rather than one
#: file per model, a different shape this wrapper does not read. Excluded by prefix
#: rather than silently skipped, and reported.
SKIP_PREFIX = 'evals/open_llm_leaderboard/'

#: The one file run from a local checkout, so `pretrained=` carries no namespace.
#: Supplying it here rather than defaulting a developer keeps the guess visible.
ORGLESS_MODEL_ORG = {
    'stablelm-3b-4e1t': 'stabilityai',
}

_PRETRAINED_RE = re.compile(r'(?<![\w-])pretrained=([^,]+)')


def resolve_commit(revision: str, *, timeout: float = 30.0) -> str | None:
    """Return the commit sha `revision` names, or None if it cannot be read."""
    if len(revision) == 40 and all(c in '0123456789abcdef' for c in revision.lower()):
        return revision.lower()
    try:
        resp = requests.get(f'{COMMITS_API}{revision}', timeout=timeout)
        resp.raise_for_status()
        return str(resp.json()['sha'])
    except Exception:  # noqa: BLE001 -- the caller decides whether this is fatal
        return None


def list_result_files(sha: str, *, timeout: float = 30.0) -> tuple[list[str], list[str]]:
    """Return `(result files, skipped files)` under `evals/` at `sha`."""
    resp = requests.get(f'{TREE_API}{sha}?recursive=1', timeout=timeout)
    resp.raise_for_status()
    tree = resp.json()
    if tree.get('truncated'):
        raise SystemExit(
            'the git tree listing came back truncated, so the file set would be '
            'incomplete and the run would silently publish a subset'
        )
    under_evals = [
        item['path'] for item in tree['tree']
        if item['type'] == 'blob' and item['path'].startswith('evals/')
    ]
    keep = [p for p in under_evals if not p.startswith(SKIP_PREFIX)]
    skipped = [p for p in under_evals if p.startswith(SKIP_PREFIX)]
    return sorted(keep), sorted(skipped)


def normalize_model_args(payload: dict[str, Any], source_ref: str) -> str:
    """Give `config.model_args` a `pretrained=` value that names a real repo.

    Returns the resolved `developer/model` id. Raises when the id has no namespace
    and none is registered, rather than inventing one.
    """
    config = payload.get('config')
    if not isinstance(config, dict):
        raise ValueError(f'{source_ref}: no config block, so no model identity')
    model_args = config.get('model_args')
    if not isinstance(model_args, str):
        raise ValueError(f'{source_ref}: config.model_args is not a string')
    found = _PRETRAINED_RE.search(model_args)
    if not found:
        raise ValueError(f'{source_ref}: config.model_args has no pretrained= value')
    raw = found.group(1).strip()
    if '/' in raw:
        return raw
    org = ORGLESS_MODEL_ORG.get(raw)
    if not org:
        raise ValueError(
            f'{source_ref}: pretrained={raw!r} names no publishing namespace and '
            'is not in ORGLESS_MODEL_ORG; add it there rather than letting the '
            'record claim an untrue identity'
        )
    resolved = f'{org}/{raw}'
    config['model_args'] = model_args.replace(found.group(0), f'pretrained={resolved}')
    return resolved


def stage_sources(
    sha: str, paths: list[str], staging: Path, *, timeout: float = 60.0
) -> tuple[dict[str, str], list[SourceRecordFailure]]:
    """Download each result file, repair its identity, write it where the converter
    will find it. Returns `(staged filename -> model id, failures)`.

    The converter discovers files by a `results_*.json` name, so each source file is
    staged under that pattern with its own basename preserved for traceability.
    """
    staged: dict[str, str] = {}
    failures: list[SourceRecordFailure] = []
    for path in paths:
        url = f'{RAW_BASE}{sha}/{path}'
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 -- one unreadable file is data
            failures.append(SourceRecordFailure(
                source_ref=path, reason=f'could not read source file: {exc}',
                source_record={'path': path, 'url': url},
            ))
            continue
        raw_capture.record(
            url=url, content=resp.content, content_type='application/json',
            label=f'StableLM evals {path}',
        )
        try:
            model_id = normalize_model_args(payload, path)
        except ValueError as exc:
            failures.append(SourceRecordFailure(
                source_ref=path, reason=str(exc),
                source_record={'path': path, 'config': payload.get('config')},
            ))
            continue
        slug = Path(path).name.removesuffix('.json')
        name = f'results_{slug}.json'
        (staging / name).write_text(json.dumps(payload))
        staged[name] = model_id
    return staged, failures


def convert(
    sha: str, staging: Path, staged: dict[str, str], retrieved_ts: str
) -> list[Any]:
    """Hand the staged files to the in-tree lm_eval converter."""
    metadata = {
        'source_organization_name': 'Stability AI',
        'source_organization_url': PROJECT_URL,
        'evaluator_relationship': 'third_party',
        'eval_library_name': EVAL_LIBRARY_NAME,
        'eval_library_version': EVAL_LIBRARY_VERSION,
        'parent_eval_output_dir': str(staging),
        'retrieved_timestamp': retrieved_ts,
        'source_commit': sha,
    }
    result = LMEvalAdapter().transform_from_directory_result(staging, metadata)
    return list(result.records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--output-dir', type=Path,
        default=Path(f'/tmp/{COLLECTION}-smoke/data/{COLLECTION}'),
        help='Collection directory to write into, i.e. <root>/data/stablelm-evals. '
             'Defaults outside any checkout.',
    )
    ap.add_argument(
        '--revision', default='main',
        help='Branch, tag or commit of Stability-AI/StableLM to convert. Resolved '
             'to a commit sha before anything is fetched.',
    )
    ap.add_argument(
        '--allow-unpinned-source', action='store_true',
        help='Proceed when the revision cannot be resolved to a commit sha.',
    )
    ap.add_argument('--limit', type=int, default=None, help='Convert only the first N files.')
    ap.add_argument(
        '--replace-existing', action='store_true',
        help='Replace records already in the output directory. Filenames are fresh '
             'uuid4s, so without this a populated directory is an error.',
    )
    ap.add_argument('--failure-report', type=Path, default=None)
    ap.add_argument(
        '--emit-source-version', action='store_true',
        help='Print the resolved source commit and exit without converting.',
    )
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    sha = resolve_commit(args.revision)
    if sha is None:
        if not args.allow_unpinned_source:
            raise SystemExit(
                f'could not resolve {args.revision!r} to a commit sha in '
                f'{OWNER_REPO}. Every record cites its source commit, so a moving '
                'reference is refused. Pass --allow-unpinned-source to override.'
            )
        sha = args.revision
    if args.emit_source_version:
        print(sha)
        return []

    output_dir = Path(args.output_dir)
    if output_dir.name != COLLECTION:
        raise SystemExit(
            f'--output-dir must end in {COLLECTION!r}; got {output_dir}. Otherwise '
            'the run reports one destination and writes to another.'
        )

    paths, skipped = list_result_files(sha)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f'no result files found under evals/ at {sha}')

    staging_root = Path(tempfile.mkdtemp(prefix='eee-stablelm-'))
    try:
        staging = staging_root / 'in'
        staging.mkdir()
        staged, failures = stage_sources(sha, paths, staging)
        if not staged:
            raise SystemExit(
                f'none of the {len(paths)} source file(s) yielded a usable model '
                'identity, so nothing would be published'
            )
        logs = convert(sha, staging, staged, str(time.time()))
        result = SourceConversionResult(
            source_name='StableLM evals',
            total_records=len(paths),
            records=[],
            failures=failures,
        )
        report = save_failure_report(
            result, args.failure_report or default_failure_report_path(output_dir)
        )
        print(
            f'Conversion accounting: {report} ({len(failures)} unconverted, '
            f'{len(skipped)} file(s) skipped as a different shape)'
        )

        existing = sorted(output_dir.glob('*/*/*.json'))
        if existing and not args.replace_existing:
            raise SystemExit(
                f'{len(existing)} record(s) already exist under {output_dir}, e.g. '
                f'{existing[0]}. Filenames are fresh uuid4s, so writing now would '
                'add a second copy of every evaluation. Pass --replace-existing.'
            )

        written = publish_evaluation_logs(
            logs,
            base_output_dir=output_dir.parent,
            file_uuids=[str(uuid.uuid4()) for _ in logs],
            collection_override=COLLECTION,
        )
        if existing and args.replace_existing:
            for path in existing:
                path.unlink()
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    print(
        f'Coverage: {len(paths)} source file(s) -> {len(written)} record(s); '
        f'{len(failures)} dropped, {len(skipped)} skipped -> {output_dir}'
    )
    if skipped:
        print(
            f'  skipped (per-task files, a shape this wrapper does not read): '
            f'{", ".join(skipped[:4])}'
            + (' ...' if len(skipped) > 4 else ''),
            file=sys.stderr,
        )
    result.raise_if_incomplete()
    return written


def main() -> None:
    run(parse_args())


if __name__ == '__main__':
    main()
