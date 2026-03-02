"""Top-level CLI for conversion and validation."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

EVALUATOR_RELATIONSHIP_CHOICES = [
    "first_party",
    "third_party",
    "collaborative",
    "other",
]


def _common_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_organization_name": args.source_organization_name,
        "evaluator_relationship": args.evaluator_relationship,
        "source_organization_url": args.source_organization_url,
        "source_organization_logo_url": args.source_organization_logo_url,
        "eval_library_name": args.eval_library_name,
        "eval_library_version": args.eval_library_version,
        "parent_eval_output_dir": args.output_dir,
    }


def _output_dir_for_log(base_output: Path, log: Any) -> Path:
    dataset = "unknown"
    if log.evaluation_results and log.evaluation_results[0].source_data:
        dataset = log.evaluation_results[0].source_data.dataset_name or "unknown"
    model_id = log.model_info.id or "unknown"
    parts = model_id.split("/", 1)
    developer = parts[0] if len(parts) == 2 else "unknown"
    model_name = parts[1] if len(parts) == 2 else model_id
    out_dir = base_output / dataset / developer / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_log(log: Any, base_output: Path, eval_uuid: str | None = None) -> Path:
    out_dir = _output_dir_for_log(base_output, log)
    eval_uuid = eval_uuid or str(uuid.uuid4())
    out_file = out_dir / f"{eval_uuid}.json"
    with out_file.open("w", encoding="utf-8") as file:
        json.dump(log.model_dump(mode="json", exclude_none=True), file, indent=2)
    return out_file


def _cmd_convert_lm_eval(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
    from every_eval_ever.converters.lm_eval.instance_level_adapter import (
        LMEvalInstanceLevelAdapter,
    )
    from every_eval_ever.converters.lm_eval.utils import find_samples_file

    adapter = LMEvalAdapter()
    metadata = _common_metadata(args)
    if args.inference_engine:
        metadata["inference_engine"] = args.inference_engine
    if args.inference_engine_version:
        metadata["inference_engine_version"] = args.inference_engine_version

    log_path = Path(args.log_path)
    metadata["parent_eval_output_dir"] = str(log_path.parent if log_path.is_file() else log_path)
    if log_path.is_file():
        logs = adapter.transform_from_file(log_path, metadata)
    elif log_path.is_dir():
        logs = adapter.transform_from_directory(log_path, metadata)
    else:
        raise FileNotFoundError(f"Path is not a file or directory: {log_path}")

    output_dir = Path(args.output_dir)
    for log in logs:
        eval_uuid = str(uuid.uuid4())
        if args.include_samples:
            meta = adapter.get_eval_metadata(log.evaluation_id)
            parent_dir = meta.get("parent_dir")
            task_name = meta.get("task_name")
            if parent_dir and task_name:
                samples_file = find_samples_file(Path(parent_dir), task_name)
                if samples_file:
                    detailed = LMEvalInstanceLevelAdapter().transform_and_save(
                        samples_path=samples_file,
                        evaluation_id=log.evaluation_id,
                        model_id=log.model_info.id,
                        task_name=task_name,
                        output_dir=str(_output_dir_for_log(output_dir, log)),
                        file_uuid=eval_uuid,
                    )
                    log.detailed_evaluation_results = detailed
        print(_write_log(log, output_dir, eval_uuid=eval_uuid))

    print(f"Converted {len(logs)} evaluation log(s).")
    return 0


def _cmd_convert_inspect(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.inspect.adapter import InspectAIAdapter

    adapter = InspectAIAdapter()
    metadata = _common_metadata(args)
    metadata["file_uuid"] = str(uuid.uuid4())

    log_path = Path(args.log_path)
    if log_path.is_file():
        logs = [adapter.transform_from_file(log_path, metadata)]
    elif log_path.is_dir():
        logs = adapter.transform_from_directory(log_path, metadata)
    else:
        raise FileNotFoundError(f"Path is not a file or directory: {log_path}")

    output_dir = Path(args.output_dir)
    for log in logs:
        print(_write_log(log, output_dir))

    print(f"Converted {len(logs)} evaluation log(s).")
    return 0


def _cmd_convert_helm(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.helm.adapter import HELMAdapter

    adapter = HELMAdapter()
    metadata = _common_metadata(args)
    metadata["file_uuid"] = str(uuid.uuid4())

    logs = adapter.transform_from_directory(
        Path(args.log_path),
        output_path=str(Path(args.output_dir) / "helm_output"),
        metadata_args=metadata,
    )
    output_dir = Path(args.output_dir)
    for log in logs:
        print(_write_log(log, output_dir))

    print(f"Converted {len(logs)} evaluation log(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="every_eval_ever",
        description=(
            "CLI for validating and converting evaluation results into the "
            "Every Eval Ever schema."
        ),
        epilog=(
            "Examples:\n"
            "  every_eval_ever validate data --schema aggregate\n"
            "  every_eval_ever check-duplicates data\n"
            "  every_eval_ever convert lm_eval eee --log-path results.json --output-dir data\n"
            "  every_eval_ever convert inspect eee --log-path inspect_log.json --output-dir data\n"
            "  every_eval_ever convert helm eee --log-path helm_run_dir --output-dir data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate JSON files against schema",
        description="Validate one or more JSON files/directories against the bundled schema.",
    )
    validate_parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files or directories containing JSON files.",
    )
    validate_parser.add_argument(
        "--schema-path",
        default=None,
        help="Optional explicit path to a JSON Schema file. Overrides --schema.",
    )
    validate_parser.add_argument(
        "--schema",
        choices=["aggregate", "instance"],
        default="aggregate",
        help="Which bundled schema to use when --schema-path is not provided.",
    )

    check_duplicates_parser = subparsers.add_parser(
        "check-duplicates",
        help="Detect duplicate evaluation JSON entries",
        description=(
            "Detect duplicate evaluation entries while ignoring scrape-specific "
            "keys (evaluation_id and retrieved_timestamp)."
        ),
    )
    check_duplicates_parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files or directories containing JSON files.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert source eval logs to every_eval_ever",
        description="Convert outputs from supported eval frameworks into Every Eval Ever JSON.",
    )
    convert_subparsers = convert_parser.add_subparsers(dest="source", required=True)

    for source in ["lm_eval", "inspect", "helm"]:
        source_parser = convert_subparsers.add_parser(
            source,
            help=f"Convert {source} logs",
            description=f"Convert {source} evaluation outputs to Every Eval Ever format.",
        )
        source_parser.add_argument(
            "dst",
            nargs="?",
            default="eee",
            help="Destination schema family. Only 'eee' is currently supported.",
        )
        source_parser.add_argument(
            "--log-path",
            required=True,
            help="Path to source log file or directory to convert.",
        )
        source_parser.add_argument(
            "--output-dir",
            default="data",
            help="Base output directory where converted files are written.",
        )
        source_parser.add_argument(
            "--source-organization-name",
            default="unknown",
            help="Organization name for source_metadata.source_organization_name.",
        )
        source_parser.add_argument(
            "--evaluator-relationship",
            default="third_party",
            choices=EVALUATOR_RELATIONSHIP_CHOICES,
            help="Relationship between evaluator and model developer.",
        )
        source_parser.add_argument(
            "--source-organization-url",
            default=None,
            help="Optional organization URL for source metadata.",
        )
        source_parser.add_argument(
            "--source-organization-logo-url",
            default=None,
            help="Optional organization logo URL for source metadata.",
        )
        source_parser.add_argument(
            "--eval-library-name",
            default=source,
            help="Evaluation library name recorded in eval_library.name.",
        )
        source_parser.add_argument(
            "--eval-library-version",
            default="unknown",
            help="Evaluation library version recorded in eval_library.version.",
        )

        if source == "lm_eval":
            source_parser.add_argument(
                "--include-samples",
                action="store_true",
                help="Also convert lm-eval sample JSONL into instance-level output.",
            )
            source_parser.add_argument(
                "--inference-engine",
                default=None,
                help="Override inferred inference engine (e.g. vllm, transformers).",
            )
            source_parser.add_argument(
                "--inference-engine-version",
                default=None,
                help="Inference engine version to record in model_info.inference_engine.version.",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        from every_eval_ever.validate import main as validate_main

        return validate_main([
            *args.paths,
            "--schema",
            args.schema,
            *( ["--schema-path", args.schema_path] if args.schema_path else [] ),
        ])

    if args.command == "check-duplicates":
        from every_eval_ever.check_duplicate_entries import (
            main as check_duplicates_main,
        )

        return check_duplicates_main(args.paths)

    if args.command == "convert":
        if getattr(args, "dst", "eee") != "eee":
            raise ValueError("Only destination 'eee' is currently supported")

        if args.source == "lm_eval":
            return _cmd_convert_lm_eval(args)
        if args.source == "inspect":
            return _cmd_convert_inspect(args)
        if args.source == "helm":
            return _cmd_convert_helm(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
