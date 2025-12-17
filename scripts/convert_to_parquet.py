from pathlib import Path
import sys
import subprocess
import json
from huggingface_hub import HfApi
import shutil
import os

from json_to_parquet import add_to_parquet
from config import HF_DATASET_REPO


def download_leaderboards(
    output_dir: Path,
    leaderboard_names: set[str],
    *,
    api: HfApi | None = None,
    repo_files: list[str] | None = None,
) -> set[str]:
    """Download leaderboard parquets from the HuggingFace dataset.

    Args:
        output_dir (Path): Directory to save the downloaded parquets.
        leaderboard_names (set[str]): Set of leaderboard splits to download.

    Returns:
        set[str]: Set of leaderboard splits that were downloaded.

    """
    try:
        if api is None:
            hf_token = os.environ.get("HF_TOKEN")
            api = HfApi(token=hf_token) if hf_token else HfApi()

        files = repo_files
        if files is None:
            files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")

        available_leaderboards = {
            file.removeprefix("data/").removesuffix(".parquet")
            for file in files
            if file.startswith("data/") and file.endswith(".parquet")
        }

        downloaded: set[str] = set()

        for lb in leaderboard_names:
            if lb in available_leaderboards:
                file_path = f"data/{lb}.parquet"
                local_path = output_dir / f"{lb}.parquet"

                downloaded_path = api.hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                    filename=file_path,
                )

                shutil.copy(downloaded_path, local_path)
                downloaded.add(lb)

        if downloaded:
            print(f"Downloaded {len(downloaded)} existing parquet(s) from HuggingFace")
        return downloaded

    except Exception as e:
        print(f"HF download failed: {e}")
        return set()


def detect_modified_leaderboards() -> dict[str, list[str]]:
    """
    Detect "modified" leaderboards under the data/ folder using git diff.
    Here, "modified" includes both newly added leaderboards (folders) and
    leaderboards with new or changed JSON entries.

    Returns:
        dict[str, list[str]]: Dictionary mapping leaderboard names to the specific files that modified.

    Raises:
        subprocess.CalledProcessError: If the git command fails.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD", "data/"],
            capture_output=True,
            text=True,
            check=True,
        )

        modified_files = result.stdout.strip().split("\n")
        if not modified_files or modified_files == [""]:
            print("No changes detected in data/")
            return {}

        leaderboards_to_files: dict[str, list[str]] = {}
        for f in modified_files:
            if f.startswith("data/") and f.endswith(".json"):
                path_parts = Path(f).relative_to("data").parts
                if path_parts:
                    leaderboard = path_parts[0]
                    if leaderboard not in leaderboards_to_files:
                        leaderboards_to_files[leaderboard] = []
                    leaderboards_to_files[leaderboard].append(f)

        return leaderboards_to_files

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git command failed: {e}")
        sys.exit(1)


def convert_modified_leaderboards():
    """
    The main function in this script that orchestrates the procedure.
    It detects the changes using detect_modified_leaderboards() and then downloads the relevant leaderboard parquets.
    The modified leaderboards are then converted to parquet, and updates the manifest for upload in modified_leaderboards.json file.

    """

    data_dir = Path("data")
    output_dir = Path("parquet_output")
    output_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    leaderboards_to_files: dict[str, list[str]] = detect_modified_leaderboards()

    hf_is_empty = False
    repo_files: list[str] | None = None
    try:
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi(token=hf_token) if hf_token else HfApi()
        repo_files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
        parquet_files = [
            f for f in repo_files if f.startswith("data/") and f.endswith(".parquet")
        ]
        hf_is_empty = len(parquet_files) == 0
        if hf_is_empty:
            print("HuggingFace dataset is empty - will sync ALL leaderboards")
    except Exception as e:
        print(f"Could not check HF dataset status: {e}")
        hf_is_empty = True

    if hf_is_empty:
        print("Syncing ALL leaderboards to HuggingFace")
        all_leaderboards = [d.name for d in data_dir.iterdir() if d.is_dir()]
        leaderboards_to_files = {
            lb: [str(f) for f in (data_dir / lb).rglob("*.json")]
            for lb in all_leaderboards
        }
        print(f"Found {len(all_leaderboards)} leaderboard(s) to sync")

    if len(leaderboards_to_files) == 0:
        print("No changes detected, nothing to upload")
        manifest = {
            "modified": [],
            "actually_modified": [],
            "downloaded": [],
            "errors": 0,
        }
        (output_dir / "modified_leaderboards.json").write_text(
            json.dumps(manifest, indent=2)
        )
        sys.exit(0)

    modified_leaderboards = set(leaderboards_to_files.keys())
    total_modified_files = sum(len(files) for files in leaderboards_to_files.values())

    downloaded = download_leaderboards(
        output_dir,
        modified_leaderboards,
        api=api if "api" in locals() else None,
        repo_files=repo_files,
    )
    new_leaderboards = modified_leaderboards - downloaded

    print(
        f"\nDetected {len(modified_leaderboards)} modified leaderboard(s) with {total_modified_files} modified file(s):"
    )
    for lb, files in leaderboards_to_files.items():
        status = " (new)" if lb in new_leaderboards else ""
        print(f"  {lb}: {len(files)} file(s){status}")

    converted_count = 0
    error_count = 0
    actually_modified = []

    for leaderboard_name, modified_files in leaderboards_to_files.items():
        parquet_path = output_dir / f"{leaderboard_name}.parquet"
        leaderboard_dir = data_dir / leaderboard_name

        if not parquet_path.exists():
            print(
                f"\nConverting: {leaderboard_name} (NEW - no existing parquet in HuggingFace)"
            )
        else:
            print(
                f"\nConverting: {leaderboard_name} ({len(modified_files)} modified file(s)) [full rebuild]"
            )

        # For any modified leaderboard, rebuild its parquet from all JSON files
        # under the leaderboard directory, instead of incrementally appending.
        input_data = str(leaderboard_dir)
        try:
            data_modified = add_to_parquet(input_data, str(parquet_path), rebuild=True)
            print(f"   ✓ {parquet_path.name}")
            converted_count += 1
            if data_modified:
                actually_modified.append(leaderboard_name)
        except Exception as e:
            print(f"Error converting {leaderboard_name}: {e}")
            error_count += 1

    manifest = {
        # All leaderboards detected with file changes under data/
        "modified": list(modified_leaderboards),
        "actually_modified": actually_modified,
        "downloaded": list(downloaded),
        "errors": error_count,
    }
    (output_dir / "modified_leaderboards.json").write_text(
        json.dumps(manifest, indent=2)
    )

    if error_count > 0 or (len(modified_leaderboards) > 0 and converted_count == 0):
        print(f"\n✗ Failed: {error_count} errors, {converted_count} succeeded")
        sys.exit(1 if error_count > 0 else 0)

    print(f"\nSuccessfully converted {converted_count} leaderboard(s)")


if __name__ == "__main__":
    convert_modified_leaderboards()
