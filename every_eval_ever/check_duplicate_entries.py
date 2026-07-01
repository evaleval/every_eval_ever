import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

from every_eval_ever.dedup import check_duplicates

IGNORE_KEYS = {'retrieved_timestamp', 'evaluation_id'}


def expand_paths(paths: List[str]) -> List[str]:
    """Expand folders to file paths."""
    file_paths: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith('.json'):
            file_paths.append(path)
        elif os.path.isdir(path):
            for root, _, file_names in os.walk(path):
                for file_name in file_names:
                    if file_name.endswith('.json'):
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


def normalize_list(items: List[Any]) -> List[Any]:
    normalized_items = [strip_ignored_keys(item) for item in items]
    # Sort to avoid false negatives when scrapers emit the same items in different orders.
    return sorted(
        normalized_items,
        key=lambda item: json.dumps(
            item, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ),
    )


def strip_ignored_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_ignored_keys(val)
            for key, val in value.items()
            if key not in IGNORE_KEYS
        }
    if isinstance(value, list):
        return normalize_list(value)
    return value


def normalized_hash(payload: Dict[str, Any]) -> str:
    normalized = strip_ignored_keys(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='check_duplicate_entries',
        description='Detects duplicate evaluation entries ignoring scrape timestamp fields.',
    )
    parser.add_argument(
        'paths', nargs='+', type=str, help='File or folder paths to JSON data'
    )
    args = parser.parse_args(argv)

    file_paths = expand_paths(args.paths)
    print()
    print(f'Checking {len(file_paths)} JSON files for duplicates...')
    print()

    local_paths = {file_path: file_path for file_path in file_paths}
    dedup_report = check_duplicates(file_paths, local_paths, {'files': {}})
    duplicate_results = [
        result for result in dedup_report.results if result.duplicate_of
    ]

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
