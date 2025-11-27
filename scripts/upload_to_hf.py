from huggingface_hub import login, HfApi
import pandas as pd
from pathlib import Path
import sys
import os
import json

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "evaleval/every_eval_ever")
PARQUET_DIR = Path("parquet_output")
MANIFEST_PATH = PARQUET_DIR / "modified_leaderboards.json"

def upload_modified_parquets():
    """
    Upload modified leaderboard parquets to the HuggingFace dataset.
    Also updates the README.md file with the new leaderboards.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HF_TOKEN from environment")
        api = HfApi(token=hf_token)
    elif "--login" in sys.argv:
        print("Logging in to HuggingFace...")
        login()
        api = HfApi()
    else:
        api = HfApi()
        try:
            api.whoami()
            print("Using existing HuggingFace token")
        except Exception:
            print("ERROR: Not logged in. Run with --login flag or set HF_TOKEN environment variable")
            sys.exit(1)
    
    if not MANIFEST_PATH.exists():
        print(f"ERROR: No manifest found at {MANIFEST_PATH}")
        print("Run convert_to_parquet.py first to generate the manifest")
        sys.exit(1)
    
    manifest = json.loads(MANIFEST_PATH.read_text())
    modified_leaderboards = manifest.get("modified", [])
    
    if not modified_leaderboards:
        print("\nNo modified leaderboards to upload (per manifest)")
        sys.exit(0)
    
    print(f"\nManifest found: {len(modified_leaderboards)} leaderboard(s) to upload")
    
    files_to_upload = [
        PARQUET_DIR / f"{lb}.parquet"
        for lb in modified_leaderboards
    ]
    
    files_to_upload = [f for f in files_to_upload if f.exists()]
    
    if not files_to_upload:
        print(f"ERROR: No parquet files to upload in {PARQUET_DIR}")
        sys.exit(1)
    
    print(f"\nUploading {len(files_to_upload)} parquet file(s):")
    for pf in files_to_upload:
        print(f"  - {pf.stem}")
    
    try:
        existing_files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
        all_leaderboards = {
            f.removeprefix("data/").removesuffix(".parquet")
            for f in existing_files
            if f.startswith("data/") and f.endswith(".parquet") and f.count('/') == 1
        }
        
        all_leaderboards.update(modified_leaderboards)
        
        # Generate YAML list of all leaderboards (each leaderboard is a separate split)
        splits_yaml = '\n'.join(
            f"  - split: {lb}\n    path: data/{lb}.parquet"
            for lb in sorted(all_leaderboards)
        )
        """
        This readme update is auto-generated, and is needed because the viewer doesn't do it otherwise.
        """
        readme_content = f"""---
configs:
- config_name: default
  data_files:
{splits_yaml}
---

# Every Eval Ever Dataset

Evaluation results from various AI model leaderboards.

## Usage

```python
from datasets import load_dataset

# Load specific leaderboard
dataset = load_dataset("{HF_DATASET_REPO}", split="hfopenllm_v2")

# Load all
dataset = load_dataset("{HF_DATASET_REPO}")
```

## Available Leaderboards (Splits)

{chr(10).join(f'- `{lb}`' for lb in sorted(all_leaderboards))}

## Schema

- `model_name`, `model_id`, `model_developer`: Model information  
- `evaluation_source_name`: Leaderboard name  
- `evaluation_results`: JSON string with all metrics  
- Additional metadata for reproducibility

Auto-updated via GitHub Actions.
"""
        
        readme_path = PARQUET_DIR / "README.md"
        readme_path.write_text(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update dataset card"
        )
        print(f"\n✓ Updated README.md with {len(all_leaderboards)} leaderboards")
        
    except Exception as e:
        print(f"\nWarning: Failed to update README: {e}")
    
    uploaded_count = 0
    error_count = 0
    
    for parquet_file in files_to_upload:
        leaderboard_name = parquet_file.stem
        
        path_in_repo = f"data/{leaderboard_name}.parquet"
        
        try:
            print(f"\nUploading: {leaderboard_name}")
            
            df = pd.read_parquet(parquet_file)
            print(f"   {len(df)} rows, {len(df.columns)} columns")
            
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update {leaderboard_name} leaderboard data"
            )
            
            print(f"   SUCCESS: Uploaded → {path_in_repo}")
            uploaded_count += 1
            
        except Exception as e:
            print(f"   ERROR: Error uploading {leaderboard_name}: {e}")
            error_count += 1
    
    if error_count > 0:
        print(f"\n {error_count} file(s) failed to upload")
        sys.exit(1)
    
    print(f"\n Uploaded {uploaded_count} file(s) to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    upload_modified_parquets()