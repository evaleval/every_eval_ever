import argparse
import hashlib
import json
import os
from typing import Any, Dict, List


IGNORE_KEYS = {"retrieved_timestamp", "evaluation_id"}


def expand_paths(paths: List[str]) -> List[str]:
    """Expand folders to file paths."""
    file_paths: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".json"):
            file_paths.append(path)
        elif os.path.isdir(path):
            for root, _, file_names in os.walk(path):
                for file_name in file_names:
                    if file_name.endswith(".json"):
                        file_paths.append(os.path.join(root, file_name))
        else:
            raise Exception(f"Could not find file or directory at path: {path}")
    return file_paths


def annotate_error(file_path: str, message: str, **kwargs) -> None:
    """If run in GitHub Actions, annotate errors."""
    if os.environ.get("GITHUB_ACTION"):
        joined_kwargs = "".join(f",{key}={value}" for key, value in kwargs.items())
        print(f"::error file={file_path}{joined_kwargs}::{message}")


def normalize_list(items: List[Any]) -> List[Any]:
    normalized_items = [strip_ignored_keys(item) for item in items]
    # Sort to avoid false negatives when scrapers emit the same items in different orders.
    return sorted(
        normalized_items,
        key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
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
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_duplicate_entries",
        description="Detects duplicate evaluation entries ignoring scrape timestamp fields.",
    )
    parser.add_argument(
        "paths", nargs="+", type=str, help="File or folder paths to JSON data"
    )
    args = parser.parse_args()

    file_paths = expand_paths(args.paths)
    print()
    print(f"Checking {len(file_paths)} JSON files for duplicates...")
    print()

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            message = f"JSONDecodeError: {str(e)}"
            annotate_error(
                file_path,
                message,
                title="JSONDecodeError",
                col=e.colno,
                line=e.lineno,
            )
            print(f"{file_path}")
            print("  " + message)
            print()
            raise

        entry_hash = normalized_hash(payload)
        groups.setdefault(entry_hash, []).append(
            {
                "path": file_path,
                "evaluation_id": payload.get("evaluation_id"),
                "retrieved_timestamp": payload.get("retrieved_timestamp"),
            }
        )

    duplicate_groups = [entries for entries in groups.values() if len(entries) > 1]
    if not duplicate_groups:
        print("No duplicates found.")
        print()
        return

    ignore_label = ", ".join(f"`{key}`" for key in sorted(IGNORE_KEYS))
    print(f"Found duplicate entries (ignoring keys: {ignore_label}).")
    print()

    for index, entries in enumerate(duplicate_groups, start=1):
        print(f"Duplicate group {index} ({len(entries)} files):")
        for entry in entries:
            print(f"  - {entry['path']}")
            print(f"    evaluation_id: {entry.get('evaluation_id')}")
            print(f"    retrieved_timestamp: {entry.get('retrieved_timestamp')}")
            annotate_error(
                entry["path"],
                "Duplicate entry detected (ignoring `evaluation_id` and `retrieved_timestamp`).",
                title="DuplicateEntry",
            )
        print()

    raise SystemExit(1)


if __name__ == "__main__":
    main()
