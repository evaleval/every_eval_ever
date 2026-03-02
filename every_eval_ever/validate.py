"""Validation helpers and CLI for JSON data against bundled schemas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from every_eval_ever.schema import schema_json


def get_validator(*, schema_path: str | None = None, schema_name: str = "eval.schema.json") -> Validator:
    if schema_path is not None:
        with open(schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)
    else:
        schema = schema_json(schema_name)
    validator_cls = validator_for(schema)
    return validator_cls(schema)


def expand_paths(paths: Iterable[str]) -> list[str]:
    file_paths: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_file() and candidate.suffix == ".json":
            file_paths.append(str(candidate))
        elif candidate.is_dir():
            for subpath in candidate.rglob("*.json"):
                file_paths.append(str(subpath))
        else:
            raise FileNotFoundError(f"Could not find file or directory at path: {path}")
    return file_paths


def annotate_error(file_path: str, message: str, **kwargs: object) -> None:
    if os.environ.get("GITHUB_ACTION"):
        joined_kwargs = "".join(f",{key}={value}" for key, value in kwargs.items())
        print(f"::error file={file_path}{joined_kwargs}::{message}")


def validate_files(paths: Iterable[str], *, schema_path: str | None = None, schema_name: str = "eval.schema.json") -> tuple[int, int]:
    file_paths = expand_paths(paths)
    validator = get_validator(schema_path=schema_path, schema_name=schema_name)

    print(f"\nValidating {len(file_paths)} JSON files...\n")
    num_passed = 0
    num_failed = 0

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                instance = json.load(file)
            validator.validate(instance)
            num_passed += 1
        except ValidationError as ex:
            message = f"{type(ex).__name__}: {ex.message}"
            annotate_error(file_path, message, title=type(ex).__name__)
            print(f"{file_path}\n  {message}\n")
            num_failed += 1
        except json.JSONDecodeError as ex:
            message = f"{type(ex).__name__}: {str(ex)}"
            annotate_error(file_path, message, title=type(ex).__name__, col=ex.colno, line=ex.lineno)
            print(f"{file_path}\n  {message}\n")
            num_failed += 1

    print(f"{num_passed} file(s) passed; {num_failed} file(s) failed\n")
    return num_passed, num_failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="every_eval_ever validate",
        description="Validate JSON data against the every_eval_ever schema",
    )
    parser.add_argument("paths", nargs="+", help="File or directory paths to JSON data")
    parser.add_argument("--schema-path", type=str, default=None, help="Explicit path to schema JSON")
    parser.add_argument(
        "--schema",
        choices=["aggregate", "instance"],
        default="aggregate",
        help="Use bundled aggregate or instance-level schema",
    )
    args = parser.parse_args(argv)

    schema_name = "eval.schema.json" if args.schema == "aggregate" else "instance_level_eval.schema.json"
    _, failed = validate_files(args.paths, schema_path=args.schema_path, schema_name=schema_name)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
