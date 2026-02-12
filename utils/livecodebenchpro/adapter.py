"""
Script to migrate existing Live Code Bench Pro data from schema 0.1.0 to 0.2.0.

Moves top-level source_data URLs into per-evaluation_result source_data fields
using SourceDataUrl, matching each URL to its evaluation by difficulty.

Usage:
    uv run python utils/livecodebenchpro/adapter.py
"""

import json
from pathlib import Path

BASE_URL = "https://webhook.cp-bench.orzzh.com/leaderboard/llm/difficulty"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "livecodebenchpro"

# Map evaluation_name -> difficulty for URL matching
DIFFICULTY_FOR_EVAL = {
    "Hard Problems": "hard",
    "Medium Problems": "medium",
    "Easy Problems": "easy",
}


def make_source_data(difficulty: str) -> dict:
    """Build a SourceDataUrl dict for a given difficulty."""
    return {
        "dataset_name": f"{difficulty.capitalize()} Problems",
        "source_type": "url",
        "url": [f"{BASE_URL}?difficulty={difficulty}&benchmark_mode=live"],
    }


def migrate_file(filepath: Path) -> None:
    """Migrate a single JSON file from 0.1.0 to 0.2.0."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if data.get("schema_version") == "0.2.0":
        print(f"Skipping (already 0.2.0): {filepath}")
        return

    if data.get("schema_version") != "0.1.0":
        raise ValueError(f"{filepath}: expected schema_version 0.1.0, got {data.get('schema_version')}")

    # Remove top-level source_data
    if "source_data" not in data:
        raise ValueError(f"{filepath}: missing top-level source_data")
    del data["source_data"]

    # Add source_data to each evaluation_result
    for result in data["evaluation_results"]:
        eval_name = result.get("evaluation_name")
        if not eval_name:
            raise ValueError(f"{filepath}: evaluation_result missing evaluation_name")

        difficulty = DIFFICULTY_FOR_EVAL.get(eval_name)
        if not difficulty:
            raise ValueError(f"{filepath}: unknown evaluation_name '{eval_name}'")

        result["source_data"] = make_source_data(difficulty)

    data["schema_version"] = "0.2.0"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = list(DATA_DIR.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {DATA_DIR}")

    print(f"Migrating {len(files)} files in {DATA_DIR}...")

    for filepath in files:
        migrate_file(filepath)
        print(f"Migrated: {filepath}")

    print(f"\nDone! Migrated {len(files)} files to schema 0.2.0.")


if __name__ == "__main__":
    main()
