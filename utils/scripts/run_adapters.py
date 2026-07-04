#!/usr/bin/env python3

import argparse
import ast
import datetime
import json
import os
import re
import shutil
import subprocess
import time
import traceback
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

def get_dir_size_mb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)

def download_hf_json(filename: str, default: dict, revision: str = "main") -> dict:
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type=REPO_TYPE, revision=revision)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except EntryNotFoundError:
        return default
    except Exception as e:
        print(f"Warning: Could not download {filename} from HF, using default. ({e})")
        return default

def download_file(url: str, output: Path):
    req = urllib.request.Request(url, headers={'User-Agent': 'every-eval-ever adapter runner'})
    with urllib.request.urlopen(req, timeout=60) as response, open(output, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def check_headers(url: str) -> dict:
    """Check remote URL headers to detect if data has changed."""
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
            
    if days_since_success >= 7:
        return True, "fallback 7 days"
        
    return False, "not stale"

def get_input_requirement(adapter_path: Path):
    """Read explicit adapter configuration from config.json."""
    config_path = adapter_path.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return (
                config.get("requires_json", False),
                config.get("requires_csv", False),
                config.get("url"),
                config.get("arg_name")
            )
        except Exception as e:
            print(f"Warning: Failed to parse {config_path}: {e}")
            
    return False, False, None, None

def main():
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

    try:
        prs = api.get_repo_discussions(repo_id=REPO_ID, repo_type=REPO_TYPE)
        open_prs = [
            pr for pr in prs 
            if getattr(pr, "is_pull_request", False) 
            and pr.status == "open"
            and (pr.author == current_user if current_user else True)
        ]
        existing_pr = max(open_prs, key=lambda x: x.num) if open_prs else None
    except Exception as e:
        print(f"Error fetching PRs from Hugging Face: {e}")
        traceback.print_exc()
        existing_pr = None
    
    revision = f"refs/pr/{existing_pr.num}" if existing_pr else "main"
    if existing_pr:
        print(f"Found existing PR #{existing_pr.num} ({existing_pr.url}). Using revision {revision} for state.")
    else:
        print("No open PR found. Will run against main and create a new PR later if needed.")
        
    any_failures = False

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
    
    # Gather adapter requirements and check staleness
    adapter_info = []
    print("Analyzing adapters...")
    for adapter in adapters:
        adapter_path = UTILS_DIR / adapter / "adapter.py"
        try:
            requires_json, requires_csv, url, arg_name = get_input_requirement(adapter_path)
        except Exception as e:
            print(f"Warning: Could not parse adapter {adapter} requirements: {e}")
            requires_json, requires_csv, url, arg_name = False, False, None, None

        current_headers = check_headers(url)
        stale, reason = is_stale(adapter, stats, current_headers)
        
        adapter_info.append({
            "name": adapter,
            "stale": stale,
            "reason": reason,
            "url": url,
            "headers": current_headers,
            "requires_json": requires_json,
            "requires_csv": requires_csv,
            "arg_name": arg_name,
            "adapter_path": adapter_path
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
        
        adapter_path = info["adapter_path"]
        requires_json = info["requires_json"]
        requires_csv = info["requires_csv"]
        url = info["url"]
        arg_name = info["arg_name"]
        
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

        # Only pass --output-dir if the adapter script mentions it
        content = adapter_path.read_text(encoding="utf-8")
        if "--output-dir" in content:
            cmd.extend(["--output-dir", str(adapter_data_dir)])
            
        if "--from-hf" in content:
            cmd.append("--from-hf")
        
        tmp_file = None
        try:
            if (requires_json or requires_csv) and not url:
                print(f"Skipping adapter {adapter} because it requires {arg_name} but no URL was found in the script.")
                continue

            if requires_json and url:
                tmp_file = DATA_DIR / f"{adapter}_input.json"
                print(f"Downloading required JSON input from {url}")
                download_file(url, tmp_file)
                cmd.extend([arg_name, str(tmp_file)])
            elif requires_csv and url:
                tmp_file = DATA_DIR / f"{adapter}_input.csv"
                print(f"Downloading required CSV input from {url}")
                download_file(url, tmp_file)
                cmd.extend([arg_name, str(tmp_file)])
                
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
                any_failures = True
                continue
            
            print(f"Execution succeeded in {elapsed:.2f}s")
            
            # Validation
            print("Validating outputs...")
            val_cmd = ["uv", "run", "python", "-m", "every_eval_ever", "validate", "--format", "json", str(adapter_data_dir)]
            val_res = subprocess.run(val_cmd, capture_output=True, text=True, env=env)
            
            # The validation command outputs JSON, but may include log prefixes.
            # Find the first '[' to parse the validation array.
            try:
                out_str = val_res.stdout.strip()
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
            
            is_successful = valid_files > 0 and failed_files == 0
            
            if is_successful:
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
                    "errors": errors_list[:50]
                }
            else:
                stats.setdefault(adapter, {})["last_failed"] = True
                current_report["adapters"][adapter] = {
                    "execution_failed": False,
                    "valid_files": valid_files,
                    "failed_files": failed_files,
                    "error": f"Validation failed: {valid_files} valid, {failed_files} failed.",
                    "errors": errors_list[:50]
                }
                any_failures = True
            
        except Exception as e:
            print(f"Exception during {adapter} processing: {e}")
            traceback.print_exc()
            stats.setdefault(adapter, {})["last_failed"] = True
            current_report["adapters"][adapter] = {
                "execution_failed": True,
                "error": str(e)
            }
            any_failures = True
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
                try:
                    new_pr = api.create_pull_request(
                        repo_id=REPO_ID,
                        title="Automated Adapter Data Update",
                        description="Data update from GitHub Actions",
                        repo_type=REPO_TYPE
                    )
                    pr_num = new_pr.num
                    print(f"Successfully created new PR #{pr_num} ({new_pr.url})")
                except Exception as pr_err:
                    print(f"Error: Failed to create new PR. {pr_err}")
                    traceback.print_exc()
                    any_failures = True
                    raise pr_err
                
            upload_revision = f"refs/pr/{pr_num}"
            
            print(f"Uploading using upload_large_folder to revision {upload_revision}...")
            commit_info = api.upload_large_folder(
                repo_id=REPO_ID,
                folder_path=".",
                repo_type=REPO_TYPE,
                revision=upload_revision,
                allow_patterns=["data/**"]
            )
            commit_url = getattr(commit_info, 'commit_url', getattr(commit_info, 'pr_url', str(commit_info)))
            print(f"Upload complete! URL: {commit_url}")
        except Exception as e:
            print(f"Upload failed: {e}")
            traceback.print_exc()
            any_failures = True
            
    import sys
    if any_failures:
        print("One or more adapters failed, or upload failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()