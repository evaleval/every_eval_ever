#!/usr/bin/env python3
"""
Every Eval Ever - Adapter Orchestrator

This script manages the automated execution of all data adapters for the 
Every Eval Ever (EEE) project. It is designed to run periodically (e.g., via 
GitHub Actions cron jobs) to ensure evaluation data remains up-to-date.

Key Capabilities:
- Smart Data Detection: Employs HTTP HEAD requests to monitor upstream data 
  (CSV/JSON files) for changes using ETag, Last-Modified, or Content-Length headers.
- Prioritized Execution: Identifies adapters with new remote data and executes them first.
- Resource Optimization: Defers "heavy" adapters (high time or size cost) unless 
  their data is stale or a monthly forced run is triggered.
- Automated Validation: Automatically validates all generated EEE JSON records 
  against the official schema.
- Hugging Face Integration: Downloads the previous execution state and seamlessly 
  pushes new or updated records to a Pull Request on the target HF dataset repo.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

REPO_ID = "deeplumiere/EEE_datastore"
REPO_TYPE = "dataset"
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "adapter_stats.json"
REPORT_FILE = DATA_DIR / "run_report.json"
UTILS_DIR = Path("utils")

# Heuristics
HEAVY_TIME_S = 60
HEAVY_SIZE_MB = 50

# Central configuration for adapters that require file downloads
ADAPTER_CONFIGS = {
    "arc_agi": {"url": "https://arcprize.org/media/data/leaderboard/evaluations.json", "file_arg": "--input-json"},
    "artificial_analysis": {"url": "https://artificialanalysis.ai/api/v2/data/llms/models", "file_arg": "--input-json"},
    "bfcl": {"url": "https://gorilla.cs.berkeley.edu/data_overall.csv", "file_arg": "--csv"},
    "hfopenllm_v2": {"url": "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted", "file_arg": "--input-json"},
    "sciarena": {"url": "https://sciarena.allen.ai/api/leaderboard", "file_arg": "--input-json"}
}

def get_dir_size_mb(path: Path) -> float:
    """
    Calculate the total size of a directory in megabytes.

    Args:
        path (Path): The directory path to measure.

    Returns:
        float: The total size in megabytes. Returns 0.0 if the path does not exist.
    """
    total = 0
    if not path.exists():
        return 0.0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)

def download_hf_json(filename: str, default: dict, revision: str = "main") -> dict:
    """
    Download and parse a JSON file from the target Hugging Face dataset repository.

    Args:
        filename (str): The path to the file within the repository.
        default (dict): The default dictionary to return if the file is missing or fails to parse.
        revision (str): The git revision (branch, tag, or PR ref) to download from.

    Returns:
        dict: The parsed JSON content, or the default value upon failure.
    """
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type=REPO_TYPE, revision=revision)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except EntryNotFoundError:
        return default
    except Exception as e:
        print(f"Warning: Could not download {filename} from HF, using default. ({e})")
        return default

def download_file(url: str, output: Path) -> None:
    """
    Download a file from a URL to a local destination.

    Args:
        url (str): The source URL to download from.
        output (Path): The local destination path to write the file to.
    """
    req = urllib.request.Request(url, headers={'User-Agent': 'every-eval-ever adapter runner'})
    with urllib.request.urlopen(req, timeout=60) as response, open(output, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def check_headers(url: str) -> dict:
    """
    Perform an HTTP HEAD request to retrieve cache-validation headers.

    This mechanism allows the orchestrator to detect if remote data has changed 
    without downloading the entire file payload, saving bandwidth and execution time.

    Args:
        url (str): The remote URL to inspect.

    Returns:
        dict: A dictionary containing 'url_etag', 'url_last_modified', 
              and 'url_content_length' if provided by the server. 
              Returns an empty dictionary if the request fails or no URL is provided.
    """
    if not url:
        return {}
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'every-eval-ever adapter runner'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return {
                "url_etag": response.headers.get('ETag'),
                "url_last_modified": response.headers.get('Last-Modified'),
                "url_content_length": response.headers.get('Content-Length')
            }
    except Exception as e:
        print(f"Warning: Failed to check HEAD for {url}: {e}")
        return {}

def is_stale(adapter: str, stats: dict, current_headers: dict) -> tuple[bool, str]:
    """
    Determine whether an adapter's remote data is stale and requires re-execution.

    An adapter is flagged as stale if any of the following conditions are met:
    1. The last recorded run resulted in a failure.
    2. Remote data headers (ETag, Last-Modified, Content-Length) have changed 
       since the last successful run.
    3. It has not been successfully executed in over 7 days (or 30 days if 
       headers are successfully verified as unchanged).

    Args:
        adapter (str): The name of the adapter to evaluate.
        stats (dict): The historical statistics dictionary containing previous run metadata.
        current_headers (dict): The freshly retrieved HTTP headers for the adapter's data URL.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating if the adapter 
                          should run, and a string explaining the reason.
    """
    adapter_stat = stats.get(adapter, {})
    last_success = adapter_stat.get("last_success_ts", 0)
    days_since_success = (time.time() - last_success) / 86400
    
    if adapter_stat.get("last_failed", False):
        return True, "last run failed"
        
    if current_headers:
        etag = current_headers.get("url_etag")
        last_modified = current_headers.get("url_last_modified")
        content_length = current_headers.get("url_content_length")
        
        stored_etag = adapter_stat.get("url_etag")
        stored_lm = adapter_stat.get("url_last_modified")
        stored_cl = adapter_stat.get("url_content_length")
        
        if etag and stored_etag and etag != stored_etag:
            return True, "ETag changed"
        if last_modified and stored_lm and last_modified != stored_lm:
            return True, "Last-Modified changed"
        if content_length and stored_cl and content_length != stored_cl:
            return True, "Content-Length changed"
            
        if (etag and not stored_etag) or (last_modified and not stored_lm) or (content_length and not stored_cl):
            return True, "new header available"
            
        if etag or last_modified or content_length:
            if days_since_success >= 30:
                return True, "fallback 30 days"
            return False, "data unchanged"
            
    if days_since_success >= 7:
        return True, "fallback 7 days"
        
    return False, "not stale"

def get_dynamic_args(adapter: str, env: dict) -> dict:
    """
    Dynamically discover what arguments an adapter accepts by running its --help.

    This avoids brittle static source code parsing and respects the adapter's
    actual runtime interface.

    Args:
        adapter (str): The name of the adapter.
        env (dict): The environment variables to run the subprocess with (needs PYTHONPATH).

    Returns:
        dict: A mapping of capabilities, e.g., 'accepts_output_dir'.
    """
    args_config = {
        "accepts_output_dir": False,
        "accepts_from_hf": False
    }
    try:
        cmd = ["uv", "run", "python", "-m", f"utils.{adapter}.adapter", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        help_text = res.stdout
        if "--output-dir" in help_text:
            args_config["accepts_output_dir"] = True
        if "--from-hf" in help_text:
            args_config["accepts_from_hf"] = True
    except Exception as e:
        print(f"Warning: Failed to parse help for {adapter}: {e}")
        
    return args_config

def main() -> None:
    """
    Main orchestrator entry point.
    
    Coordinates the fetching of historical state, evaluation of adapter staleness,
    prioritized execution of stale adapters, output validation, and the automated 
    creation or updating of a Pull Request on Hugging Face.
    """
    parser = argparse.ArgumentParser(description="Run all Every Eval Ever adapters robustly.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload to Hugging Face")
    args = parser.parse_args()

    # Create data dir
    DATA_DIR.mkdir(exist_ok=True)
    
    # Initialize API and check for existing PR
    api = HfApi()
    print("Checking for existing PRs...")
    try:
        current_user = api.whoami().get("name")
    except Exception as e:
        print(f"Failed to get current user: {e}")
        current_user = None

    prs = api.get_repo_discussions(repo_id=REPO_ID, repo_type=REPO_TYPE)
    open_prs = [
        pr for pr in prs 
        if getattr(pr, "is_pull_request", False) 
        and pr.status == "open"
        and (pr.author == current_user if current_user else True)
    ]
    
    existing_pr = max(open_prs, key=lambda x: x.num) if open_prs else None
    
    revision = f"refs/pr/{existing_pr.num}" if existing_pr else "main"
    if existing_pr:
        print(f"Found existing PR #{existing_pr.num}. Using revision {revision} for state.")

    # Download existing state
    # Try data/ prefix first, fallback to root
    stats = download_hf_json("data/adapter_stats.json", {}, revision=revision)
    if not stats:
        stats = download_hf_json("adapter_stats.json", {}, revision=revision)
        
    last_report = download_hf_json("data/run_report.json", {}, revision=revision)
    if not last_report:
        last_report = download_hf_json("run_report.json", {}, revision=revision)

    today = datetime.datetime.now()
    is_sunday = today.weekday() == 6
    is_monthly_run = today.day <= 7
    
    # We will build a new report for today
    today_str = today.strftime("%Y-%m-%d")
    current_report = {
        "date": today_str,
        "adapters": {}
    }

    adapters = [d.name for d in UTILS_DIR.iterdir() if d.is_dir() and (d / "adapter.py").exists()]
    
    adapter_info = []
    print("Analyzing adapters...")
    for adapter in adapters:
        adapter_config = ADAPTER_CONFIGS.get(adapter, {})
        url = adapter_config.get("url")
        file_arg = adapter_config.get("file_arg")
        
        current_headers = check_headers(url)
        stale, reason = is_stale(adapter, stats, current_headers)
        adapter_info.append({
            "name": adapter,
            "stale": stale,
            "reason": reason,
            "url": url,
            "file_arg": file_arg,
            "headers": current_headers
        })
        
    # Sort adapters: stale first, then alphabetically
    adapter_info.sort(key=lambda x: (not x["stale"], x["name"]))
    
    for info in adapter_info:
        adapter = info["name"]
        adapter_stat = stats.get(adapter, {})
        time_s = adapter_stat.get("time_s", 0)
        size_mb = adapter_stat.get("size_mb", 0)
        
        is_heavy = time_s > HEAVY_TIME_S or size_mb > HEAVY_SIZE_MB
        stale = info["stale"]
        
        if is_heavy and not is_monthly_run and not stale:
            print(f"Skipping heavy adapter {adapter} (runs monthly or when stale)")
            continue
            
        print(f"\n--- Running adapter: {adapter} (Stale: {stale}, Reason: {info['reason']}) ---")
        
        url = info["url"]
        file_arg = info["file_arg"]
        
        adapter_data_dir = DATA_DIR / adapter
        # Clean previous run data if any
        if adapter_data_dir.exists():
            shutil.rmtree(adapter_data_dir)
            
        adapter_data_dir.mkdir(parents=True, exist_ok=True)
            
        cmd = ["uv", "run", "python", "-m", f"utils.{adapter}.adapter"]
        
        # Prepare environment for adapter with every_eval_ever in PYTHONPATH
        env = os.environ.copy()
        eee_path = str(Path("every_eval_ever").absolute())
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{eee_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = eee_path

        # Dynamically inspect adapter capabilities
        accepted_args = get_dynamic_args(adapter, env)

        if accepted_args["accepts_output_dir"]:
            cmd.extend(["--output-dir", str(adapter_data_dir)])
            
        if accepted_args["accepts_from_hf"]:
            cmd.append("--from-hf")
        
        tmp_file = None
        try:
            if file_arg:
                if not url:
                    print(f"Warning: Adapter {adapter} requires {file_arg} but no URL is configured!")
                    continue
                    
                tmp_file = DATA_DIR / f"{adapter}_input{Path(url).suffix or '.json'}"
                print(f"Downloading required input from {url}")
                download_file(url, tmp_file)
                cmd.extend([file_arg, str(tmp_file)])
                
            start_t = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            elapsed = time.time() - start_t
            
            if res.returncode != 0:
                print(f"Execution FAILED for {adapter}:\n{res.stderr}")
                stats.setdefault(adapter, {})["last_failed"] = True
                current_report["adapters"][adapter] = {
                    "execution_failed": True,
                    "error": res.stderr[-500:]
                }
                continue
            
            print(f"Execution succeeded in {elapsed:.2f}s")
            
            # Validation
            print("Validating outputs...")
            val_cmd = ["uv", "run", "python", "-m", "every_eval_ever", "validate", "--format", "json", str(adapter_data_dir)]
            val_res = subprocess.run(val_cmd, capture_output=True, text=True, env=env)
            
            # Even if val_res fails (which it will if any file is invalid), it outputs JSON
            try:
                # the validate command might print other stuff? we'll try to extract JSON
                out_str = val_res.stdout.strip()
                # find the first [
                idx = out_str.find('[')
                if idx != -1:
                    val_data = json.loads(out_str[idx:])
                else:
                    val_data = []
            except json.JSONDecodeError as e:
                print(f"Failed to parse validation JSON for {adapter}: {e}")
                print(f"Stdout was: {val_res.stdout}")
                val_data = []
            
            valid_files = 0
            failed_files = 0
            errors_list = []
            
            for f_report in val_data:
                if f_report.get("valid"):
                    valid_files += 1
                else:
                    failed_files += 1
                    errs = f_report.get("errors", [])
                    errors_list.extend(errs)
                    # Delete invalid file
                    invalid_f = Path(f_report.get("file"))
                    if invalid_f.exists():
                        invalid_f.unlink()
            
            print(f"Validation: {valid_files} valid, {failed_files} failed")
            
            # Size measurement
            dir_size_mb = get_dir_size_mb(adapter_data_dir)
            
            # Update stats
            update_data = {
                "time_s": elapsed,
                "size_mb": dir_size_mb,
                "last_success_ts": time.time(),
                "last_failed": False
            }
            if info["headers"]:
                for k, v in info["headers"].items():
                    if v:
                        update_data[k] = v
            stats.setdefault(adapter, {}).update(update_data)
            
            current_report["adapters"][adapter] = {
                "execution_failed": False,
                "valid_files": valid_files,
                "failed_files": failed_files,
                "errors": errors_list[:50] # save up to 50 errors in report
            }
            
        except Exception as e:
            print(f"Exception during {adapter} processing: {e}")
            stats.setdefault(adapter, {})["last_failed"] = True
            current_report["adapters"][adapter] = {
                "execution_failed": True,
                "error": str(e)
            }
        finally:
            if tmp_file and tmp_file.exists():
                tmp_file.unlink()

    # Save stats and report
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(current_report, f, indent=2)

    # Upload to HF
    if not args.dry_run:
        print("Uploading to Hugging Face...")
        try:
            if existing_pr:
                print(f"Updating existing PR #{existing_pr.num}")
                pr_num = existing_pr.num
            else:
                print("Creating new PR...")
                new_pr = api.create_pull_request(
                    repo_id=REPO_ID,
                    title="Automated Adapter Data Update",
                    description="Data update from GitHub Actions",
                    repo_type=REPO_TYPE
                )
                pr_num = new_pr.num
                print(f"Created new PR #{pr_num}")
                
            upload_revision = f"refs/pr/{pr_num}"
            
            from huggingface_hub import CommitOperationAdd
            operations = []
            
            # Add stats and report files
            if STATS_FILE.exists():
                operations.append(CommitOperationAdd(path_in_repo=STATS_FILE.as_posix(), path_or_fileobj=STATS_FILE))
            if REPORT_FILE.exists():
                operations.append(CommitOperationAdd(path_in_repo=REPORT_FILE.as_posix(), path_or_fileobj=REPORT_FILE))
                
            # Add all files from adapters that were actually run this session
            for adapter in current_report.get("adapters", {}).keys():
                adapter_data_dir = DATA_DIR / adapter
                if adapter_data_dir.exists():
                    for filepath in adapter_data_dir.rglob("*"):
                        if filepath.is_file():
                            operations.append(CommitOperationAdd(
                                path_in_repo=filepath.as_posix(), 
                                path_or_fileobj=filepath
                            ))

            if operations:
                print(f"Pushing {len(operations)} updated file(s) using create_commit to revision {upload_revision}...")
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    revision=upload_revision,
                    commit_message="Automated Adapter Data Update",
                    operations=operations
                )
                print("Upload complete!")
            else:
                print("No files to upload!")
                
        except Exception as e:
            print(f"Upload failed: {e}")
if __name__ == "__main__":
    main()
