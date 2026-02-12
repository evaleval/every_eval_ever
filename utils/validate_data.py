import argparse
import json
import os
from typing import List

from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for


def get_schema_validator(file_path: str) -> Validator:
    with open(file_path, "r") as f:
        schema = json.load(f)
        validator_cls = validator_for(schema)
        return validator_cls(schema)


def validate_file(file_path: str, validator: Validator) -> None:
    with open(file_path, "r") as f:
        instance = json.load(f)
    validator.validate(instance)


def expand_paths(paths: List[str]) -> List[str]:
    """Expand folders to file paths"""
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
    """If run in GitHub Actions, annotate errors"""
    if os.environ.get("GITHUB_ACTION"):
        joined_kwargs = "".join(f",{key}={value}" for key, value in kwargs.items())
        print(f"::error file={file_path}{joined_kwargs}::{message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="validate_data",
        description="Validates that the JSON data conforms to the JSON schema",
    )
    parser.add_argument(
        "paths", nargs="+", type=str, help="File or folder paths to the JSON data"
    )
    parser.add_argument(
        "-s",
        "--schema-path",
        type=str,
        help="File path to the JSON schema",
        required=True,
    )
    args = parser.parse_args()
    file_paths = expand_paths(args.paths)
    num_passed = 0
    num_failed = 0
    validator = get_schema_validator(args.schema_path)
    print()
    print(f"Validating {len(file_paths)} JSON files...")
    print()
    for file_path in file_paths:
        try:
            validate_file(file_path, validator)
            num_passed += 1
        except ValidationError as e:
            message = f"{type(e).__name__}: {e.message}"
            annotate_error(
                file_path, f"{type(e).__name__}: {e.message}", title=type(e).__name__
            )
            print(f"{file_path}")
            print("  " + message)
            print()
            num_failed += 1
        except json.JSONDecodeError as e:
            # e.colno
            message = f"{type(e).__name__}: {str(e)}"
            annotate_error(
                file_path,
                f"{type(e).__name__}: {str(e)}",
                title=type(e).__name__,
                col=e.colno,
                line=e.lineno,
            )
            print(f"{file_path}")
            print("  " + message)
            print()
            num_failed += 1
        except Exception as e:
            message = f"{type(e).__name__}: {str(e)}"
            annotate_error(
                file_path, f"{type(e).__name__}: {str(e)}", title=type(e).__name__
            )
            print(f"{file_path}")
            print("  " + message)
            print()
            raise
    print(f"{num_passed} file(s) passed; {num_failed} file(s) failed")
    print()
    if num_failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
