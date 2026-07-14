import argparse
import os
import sys
from pathlib import Path
from typing import List

from every_eval_ever.dedup import (
    DEFAULT_DATASET_REPO_ID,
    check_duplicates,
    load_manifest,
    validate_manifest,
)
from every_eval_ever.json_utils import strict_json_loads
from every_eval_ever.validation_core import repo_path_from_path


def expand_paths(paths: List[str]) -> List[str]:
    """Expand folders to aggregate JSON and instance JSONL paths."""
    file_paths: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(('.json', '.jsonl')):
            file_paths.append(path)
        elif os.path.isdir(path):
            for root, _, file_names in os.walk(path):
                for file_name in file_names:
                    if file_name.endswith(('.json', '.jsonl')):
                        file_paths.append(os.path.join(root, file_name))
        else:
            raise Exception(f'Could not find file or directory at path: {path}')
    return file_paths


def annotate_error(file_path: str, message: str, **kwargs) -> None:
    """If run in GitHub Actions, annotate errors."""
    if os.environ.get('GITHUB_ACTION'):
        joined_kwargs = ''.join(
            f',{key}={value}' for key, value in kwargs.items()
        )
        print(f'::error file={file_path}{joined_kwargs}::{message}')


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='check_duplicate_entries',
        description=(
            'Compare datastore candidates with the accepted manifest. Aggregate '
            'JSON is deduplicated; instance JSONL is reported as unsupported.'
        ),
    )
    parser.add_argument(
        'paths', nargs='+', type=str, help='File or folder paths under data/'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        help='Use a local datastore manifest instead of downloading one.',
    )
    parser.add_argument(
        '--dataset-repo-id',
        default=DEFAULT_DATASET_REPO_ID,
        help='Hugging Face dataset repository containing the manifest.',
    )
    parser.add_argument(
        '--revision',
        default='main',
        help='Dataset revision from which to load the manifest.',
    )
    args = parser.parse_args(argv)

    file_paths = expand_paths(args.paths)
    print()
    print(f'Checking {len(file_paths)} datastore files for duplicates...')
    print()

    try:
        if args.manifest is not None:
            manifest = strict_json_loads(
                args.manifest.read_text(encoding='utf-8')
            )
            validate_manifest(manifest, manifest_path=str(args.manifest))
        else:
            manifest = load_manifest(
                dataset_repo_id=args.dataset_repo_id,
                revision=args.revision,
            )

        repo_and_local_paths = [
            (repo_path_from_path(Path(file_path)), file_path)
            for file_path in file_paths
        ]
        repo_paths = [repo_path for repo_path, _ in repo_and_local_paths]
        local_paths = dict(repo_and_local_paths)
        dedup_report = check_duplicates(repo_paths, local_paths, manifest)
    except Exception as exc:
        print(
            f'Duplicate check failed: {type(exc).__name__}: {exc}',
            file=sys.stderr,
        )
        return 2
    duplicate_results = [
        result for result in dedup_report.results if result.duplicate_of
    ]
    skipped_results = [
        result for result in dedup_report.results if result.skipped_reason
    ]

    for result in skipped_results:
        print(f'Skipped {result.file_path}: {result.skipped_reason}.')
    if skipped_results:
        print()

    if not duplicate_results:
        print('No duplicates found.')
        print()
        return 0

    print('Found duplicate entries (semantic fingerprint match).')
    print()

    for index, result in enumerate(duplicate_results, start=1):
        print(f'Duplicate group {index}:')
        print(f'  - {result.file_path}')
        print(f'    duplicate_of: {result.duplicate_of}')
        annotate_error(
            result.file_path,
            f'Duplicate entry detected; semantic fingerprint matches {result.duplicate_of}.',
            title='DuplicateEntry',
        )
        print()

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
