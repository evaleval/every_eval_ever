"""
Migration script to update reward-bench JSON files from schema v0.1.0 to v0.2.0.

Key changes:
- schema_version: "0.1.0" -> "0.2.0"
- Remove top-level "source_data" field
- Add "source_data" to each evaluation result item
- Remove "inference_platform": "unknown" from model_info (now optional)

For RewardBench v1 results (evaluation_id starts with "reward-bench/"):
    source_data = {"dataset_name": "RewardBench", "source_type": "hf_dataset", "hf_repo": "allenai/reward-bench"}

For RewardBench v2 results (evaluation_id starts with "reward-bench-2/"):
    source_data = {"dataset_name": "RewardBench 2", "source_type": "hf_dataset", "hf_repo": "allenai/reward-bench-2-results"}

Usage:
    python -m scripts.rewardbench.migrate_to_v020
"""

import json
from pathlib import Path


DATA_DIR = Path("data/reward-bench")

V1_SOURCE_DATA = {
    "dataset_name": "RewardBench",
    "source_type": "hf_dataset",
    "hf_repo": "allenai/reward-bench",
}

V2_SOURCE_DATA = {
    "dataset_name": "RewardBench 2",
    "source_type": "hf_dataset",
    "hf_repo": "allenai/reward-bench-2-results",
}


def migrate_file(filepath: Path) -> bool:
    """
    Migrate a single JSON file from v0.1.0 to v0.2.0.

    Returns True if the file was modified, False if it was already up to date.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Skip files that are already v0.2.0
    if data.get("schema_version") == "0.2.0":
        return False

    # Determine source_data based on evaluation_id
    evaluation_id = data.get("evaluation_id", "")
    if evaluation_id.startswith("reward-bench-2/"):
        source_data = V2_SOURCE_DATA
    else:
        source_data = V1_SOURCE_DATA

    # 1. Update schema_version
    data["schema_version"] = "0.2.0"

    # 2. Remove top-level source_data
    data.pop("source_data", None)

    # 3. Add source_data to each evaluation result
    for result in data.get("evaluation_results", []):
        if "source_data" not in result:
            result["source_data"] = source_data

    # 4. Clean up model_info: remove inference_platform if "unknown"
    model_info = data.get("model_info", {})
    if model_info.get("inference_platform") == "unknown":
        del model_info["inference_platform"]

    # Write back
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    return True


def main():
    """Migrate all reward-bench JSON files to v0.2.0."""
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist")
        return

    json_files = sorted(DATA_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {DATA_DIR}")

    migrated = 0
    skipped = 0
    errors = 0

    for filepath in json_files:
        try:
            if migrate_file(filepath):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  Error migrating {filepath}: {e}")
            errors += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already v0.2.0): {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
