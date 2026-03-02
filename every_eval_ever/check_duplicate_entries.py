"""Detect duplicate evaluation JSON entries while ignoring scrape-specific keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

IGNORE_KEYS = {"retrieved_timestamp", "evaluation_id"}


def expand_paths(paths: Iterable[str]) -> list[str]:
    """Expand file or directory inputs into JSON file paths."""
    file_paths: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file() and candidate.suffix == ".json":
            file_paths.append(str(candidate))
        elif candidate.is_dir():
            for subpath in candidate.rglob("*.json"):
                file_paths.append(str(subpath))
        else:
            raise Exception(f"Could not find file or directory at path: {path}")
    return file_paths


def annotate_error(file_path: str, message: str, **kwargs: object) -> None:
    """Emit GitHub Actions error annotations when available."""
    if os.environ.get("GITHUB_ACTION"):
        joined_kwargs = "".join(f",{key}={value}" for key, value in kwargs.items())
        print(f"::error file={file_path}{joined_kwargs}::{message}")


def normalize_list(items: list[Any]) -> list[Any]:
    normalized_items = [strip_ignored_keys(item) for item in items]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="every_eval_ever check-duplicates",
        description="Detect duplicate evaluation entries ignoring scrape timestamp fields.",
    )
    parser.add_argument("paths", nargs="+", type=str, help="File or folder paths to JSON data")
    args = parser.parse_args(argv)

    file_paths = expand_paths(args.paths)
    print(f"\nChecking {len(file_paths)} JSON files for duplicates...\n")

    groups: Dict[str, list[Dict[str, Any]]] = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as ex:
            message = f"JSONDecodeError: {str(ex)}"
            annotate_error(
                file_path,
                message,
                title="JSONDecodeError",
                col=ex.colno,
                line=ex.lineno,
            )
            print(f"{file_path}\n  {message}\n")
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
        print("No duplicates found.\n")
        return 0

    ignore_label = ", ".join(f"`{key}`" for key in sorted(IGNORE_KEYS))
    print(f"Found duplicate entries (ignoring keys: {ignore_label}).\n")

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
