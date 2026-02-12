"""CLI for converting lm-evaluation-harness output to every_eval_ever format."""

import argparse
import json
import sys
import uuid
from pathlib import Path

from .adapter import LMEvalAdapter


def main():
    parser = argparse.ArgumentParser(
        description="Convert lm-evaluation-harness output to every_eval_ever format"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to results JSON file or directory containing results files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--source_organization_name",
        type=str,
        default="",
        help="Name of the organization that ran the evaluation",
    )
    parser.add_argument(
        "--evaluator_relationship",
        type=str,
        default="first_party",
        choices=["first_party", "third_party", "collaborative", "other"],
        help="Relationship of the evaluator to the model",
    )
    parser.add_argument(
        "--source_organization_url",
        type=str,
        default=None,
        help="URL of the source organization",
    )
    parser.add_argument(
        "--include_samples",
        action="store_true",
        help="Include instance-level sample data (requires --log_samples in original eval)",
    )

    args = parser.parse_args()

    adapter = LMEvalAdapter()
    output_dir = Path(args.output_dir)

    metadata_args = {
        "source_organization_name": args.source_organization_name,
        "evaluator_relationship": args.evaluator_relationship,
        "source_organization_url": args.source_organization_url,
    }
    if args.include_samples:
        metadata_args["output_dir"] = str(output_dir)

    log_path = Path(args.log_path)

    if log_path.is_file():
        logs = adapter.transform_from_file(log_path, metadata_args)
    elif log_path.is_dir():
        logs = adapter.transform_from_directory(log_path, metadata_args)
    else:
        print(f"Error: {log_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    for log in logs:
        # Organize as: output_dir/{evaluation_name}/{developer}/{model_name}/{uuid}.json
        # Use the first evaluation result's name (before any /filter suffix) as the task name
        if log.evaluation_results:
            eval_name = log.evaluation_results[0].evaluation_name.split("/")[0]
        else:
            eval_name = "unknown"

        model_parts = log.model_info.id.split("/")
        if len(model_parts) >= 2:
            developer = model_parts[0]
            model_name = "/".join(model_parts[1:])
        else:
            developer = "unknown"
            model_name = log.model_info.id

        out_path = output_dir / eval_name / developer / model_name
        out_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{uuid.uuid4()}.json"
        out_file = out_path / file_name

        with open(out_file, "w") as f:
            json.dump(log.model_dump(mode="json", exclude_none=True), f, indent=2)

        print(f"  {out_file}")

    print(f"\nConverted {len(logs)} evaluation log(s).")


if __name__ == "__main__":
    main()
