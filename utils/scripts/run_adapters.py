#!/usr/bin/env python3
"""Orchestrator for running Every Eval Ever adapters.

Handles intelligent scheduling via adapter_stats, duplicate detection using
content-aware fingerprinting, schema validation, and upload to HuggingFace.
Designed to run daily via GitHub Actions.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from every_eval_ever.check_duplicate_entries import normalized_hash

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ID = "deeplumiere/EEE_datastore"
REPO_TYPE = "dataset"
DATA_DIR = Path("data")
STATS_FILE = DATA_DIR / "adapter_stats.json"
REPORT_FILE = DATA_DIR / "run_report.json"
UTILS_DIR = Path("utils")

# Adapters exceeding these thresholds are classified as "heavy" and only run
# during the monthly window (first 7 days) or when their source is stale.
HEAVY_TIME_S = 900
HEAVY_SIZE_MB = 250


# ── File & Network Utilities ─────────────────────────────────────────────────


def get_dir_size_mb(path: Path) -> float:
    """Calculate total size of all files in a directory, in megabytes."""
    if not path.exists():
        return 0.0
    return sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    ) / (1024 * 1024)


def download_hf_json(
    filename: str,
    default: dict | list,
    revision: str = "main",
) -> dict | list:
    """Download and parse a JSON file from HuggingFace Hub.

    Returns *default* when the file doesn't exist on the given revision.
    """
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            revision=revision,
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except EntryNotFoundError:
        return default
    except Exception as e:
        print(f"Warning: Could not download {filename} from HF ({e})")
        return default


def download_file(url: str, output: Path) -> None:
    """Download a remote file to a local path."""
    headers = {"User-Agent": "every-eval-ever adapter runner"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(output, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} while downloading {url}: {e.reason}")
        raise ValueError(f"Download failed with HTTP {e.code}: {e.reason}") from e


def check_url_headers(url: str | None) -> dict[str, str | None]:
    """Fetch HTTP HEAD headers from a URL to detect source-data changes."""
    if not url:
        return {}
    try:
        headers = {"User-Agent": "every-eval-ever adapter runner"}
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {
                "url_etag": resp.headers.get("ETag"),
                "url_last_modified": resp.headers.get("Last-Modified"),
                "url_content_length": resp.headers.get("Content-Length"),
            }
    except Exception as e:
        print(f"Warning: HEAD request failed for {url}: {e}")
        return {}


def save_json(path: Path, data: Any) -> None:
    """Write *data* to a JSON file with readable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ── Staleness Detection ─────────────────────────────────────────────────────

_HEADER_CHECKS: list[tuple[str, str]] = [
    ("url_etag", "ETag changed"),
    ("url_last_modified", "Last-Modified changed"),
    ("url_content_length", "Content-Length changed"),
]


def get_assigned_day_of_week(adapter_name: str) -> int:
    """Consistently assign an adapter to a day of the week (0-6) based on its name."""
    # Use MD5 to get a stable integer hash
    hash_val = int(hashlib.md5(adapter_name.encode('utf-8')).hexdigest(), 16)
    return hash_val % 7


def is_stale(
    adapter: str,
    stats: dict,
    current_headers: dict[str, str | None],
) -> tuple[bool, str]:
    """Decide whether an adapter's data is stale and needs re-running.

    Priority order:
    1. Assigned day of week has not arrived → not stale (deferred).
    2. Previous run failed → stale (retry immediately on its assigned day).
    3. Source HTTP headers changed → stale.
    4. New headers available that weren't previously tracked → stale.
    5. 7+ days since last *data change* → stale (fallback).
    """
    assigned_dow = get_assigned_day_of_week(adapter)
    current_dow = datetime.datetime.now().weekday()  # Monday is 0, Sunday is 6

    stat = stats.get(adapter, {})
    last_check_ts = stat.get("last_check_ts", 0)
    
    # If it was already checked today, don't check again (prevents multiple runs on the assigned day)
    # UNLESS it's a force run, but that bypasses this function.
    # We will just rely on the assigned day of the week to distribute the runs.
    if assigned_dow != current_dow:
        return False, f"assigned to day {assigned_dow}, today is {current_dow}"

    if stat.get("last_failed"):
        return True, "last run failed"

    # Compare current source headers against stored values.
    if current_headers:
        has_new_header = False
        for key, label in _HEADER_CHECKS:
            current = current_headers.get(key)
            stored = stat.get(key)
            if current and stored and current != stored:
                return True, label
            if current and not stored:
                has_new_header = True
        if has_new_header:
            return True, "new header available"

    # Fallback: time since last data change (or last success if never tracked).
    last_change = stat.get(
        "last_data_change_ts", stat.get("last_success_ts", 0)
    )
    if (time.time() - last_change) / 86400 >= 7:
        return True, "7-day fallback"

    return False, "not stale"


# ── Adapter Discovery ───────────────────────────────────────────────────────


def load_adapter_config(adapter_dir: Path) -> dict:
    """Read adapter configuration from ``config.json``."""
    config_path = adapter_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse {config_path}: {e}")
        return {}


def discover_adapters(stats: dict) -> list[dict]:
    """Scan *UTILS_DIR* for adapters and gather scheduling metadata.

    Returns a list of info dicts, sorted stale-first then alphabetically.
    """
    infos: list[dict] = []
    for adapter_dir in UTILS_DIR.iterdir():
        adapter_path = adapter_dir / "adapter.py"
        if not adapter_dir.is_dir() or not adapter_path.exists():
            continue

        config = load_adapter_config(adapter_dir)
        url = config.get("url")
        headers = check_url_headers(url)
        stale, reason = is_stale(adapter_dir.name, stats, headers)

        infos.append({
            "name": adapter_dir.name,
            "adapter_path": adapter_path,
            "stale": stale,
            "reason": reason,
            "url": url,
            "headers": headers,
            "requires_json": config.get("requires_json", False),
            "requires_csv": config.get("requires_csv", False),
            "arg_name": config.get("arg_name"),
        })

    infos.sort(key=lambda x: (not x["stale"], x["name"]))
    return infos


# ── Duplicate Detection ─────────────────────────────────────────────────────


def compute_data_fingerprint(data_dir: Path) -> str:
    """Compute a stable fingerprint for all JSON outputs in a directory.

    Uses ``normalized_hash`` from ``check_duplicate_entries`` which strips
    scrape-specific fields (``retrieved_timestamp``, ``evaluation_id``)
    before hashing.  This means identical evaluation data always produces
    the same fingerprint regardless of when it was scraped.

    Returns an empty string when the directory contains no JSON files.
    """
    file_hashes: list[str] = []
    for json_file in sorted(data_dir.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            file_hashes.append(normalized_hash(payload))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Could not hash {json_file.name}: {e}")
            continue

    if not file_hashes:
        return ""

    combined = "\n".join(sorted(file_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ── Adapter Execution ────────────────────────────────────────────────────────


def build_adapter_env() -> dict[str, str]:
    """Build the environment dict for adapter subprocesses.

    Adds ``every_eval_ever`` to ``PYTHONPATH`` so adapters can import it.
    """
    env = os.environ.copy()
    eee_path = str(Path("every_eval_ever").absolute())
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{eee_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = eee_path
    return env


def prepare_adapter_command(
    info: dict,
    adapter_data_dir: Path,
) -> tuple[list[str], Path | None]:
    """Build the CLI command for an adapter and download any required input.

    Returns ``(command, tmp_input_file)``.  *tmp_input_file* is ``None``
    when no download was needed.

    Raises ``ValueError`` when the adapter requires input data but no URL
    is configured.
    """
    adapter = info["name"]
    adapter_path: Path = info["adapter_path"]
    content = adapter_path.read_text(encoding="utf-8")

    cmd = ["uv", "run", "python", "-m", f"utils.{adapter}.adapter"]
    if "--output-dir" in content:
        cmd.extend(["--output-dir", str(adapter_data_dir)])
    if "--from-hf" in content:
        cmd.append("--from-hf")

    requires_json: bool = info["requires_json"]
    requires_csv: bool = info["requires_csv"]
    url: str | None = info["url"]
    arg_name: str | None = info["arg_name"]

    if (requires_json or requires_csv) and not url:
        raise ValueError(
            f"requires {'JSON' if requires_json else 'CSV'} input "
            f"({arg_name}) but no URL configured"
        )

    tmp_file: Path | None = None
    if requires_json and url:
        tmp_file = DATA_DIR / f"{adapter}_input.json"
        print(f"[{adapter}] Downloading JSON input from {url}")
        download_file(url, tmp_file)
        cmd.extend([arg_name, str(tmp_file)])
    elif requires_csv and url:
        tmp_file = DATA_DIR / f"{adapter}_input.csv"
        print(f"[{adapter}] Downloading CSV input from {url}")
        download_file(url, tmp_file)
        cmd.extend([arg_name, str(tmp_file)])

    return cmd, tmp_file


def validate_adapter_outputs(
    adapter_data_dir: Path,
    env: dict[str, str],
) -> tuple[int, int, list]:
    """Run schema validation on adapter outputs, deleting invalid files.

    Returns ``(valid_count, failed_count, error_list)``.
    """
    cmd = [
        "uv", "run", "python", "-m", "every_eval_ever",
        "validate", "--format", "json", str(adapter_data_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    try:
        stdout = result.stdout.strip()
        idx = stdout.find("[")
        val_data = json.loads(stdout[idx:]) if idx != -1 else []
    except json.JSONDecodeError as e:
        print(f"[{adapter_data_dir.name}] Failed to parse validation output: {e}")
        val_data = []

    valid_count = 0
    failed_count = 0
    errors: list = []

    for report in val_data:
        if report.get("valid"):
            valid_count += 1
        else:
            failed_count += 1
            errors.extend(report.get("errors", []))
            invalid_path = Path(report.get("file", ""))
            if invalid_path.exists():
                if invalid_path.is_file():
                    invalid_path.unlink()
                else:
                    print(f"[{adapter_data_dir.name}] Warning: Validation reported a directory as a failed file: {invalid_path}")

    return valid_count, failed_count, errors


# ── PR Management ────────────────────────────────────────────────────────────


def find_existing_pr(api: HfApi) -> Any | None:
    """Find the most recent open PR by the current user on the HF repo."""
    try:
        current_user = api.whoami().get("name")
    except Exception as e:
        print(f"Warning: Could not identify current HF user: {e}")
        current_user = None

    try:
        discussions = api.get_repo_discussions(
            repo_id=REPO_ID, repo_type=REPO_TYPE
        )
        open_prs = [
            d
            for d in discussions
            if getattr(d, "is_pull_request", False)
            and d.status in ("open", "draft")
            and (d.author == current_user if current_user else True)
        ]
        return max(open_prs, key=lambda x: x.num) if open_prs else None
    except Exception as e:
        print(f"Warning: Could not fetch PRs: {e}")
        traceback.print_exc()
        return None


def load_remote_state(revision: str) -> tuple[dict, dict]:
    """Download adapter_stats and run_report from HuggingFace.

    Tries the ``data/`` prefix first, then falls back to root-level paths
    to handle both repository layouts.
    """
    stats = download_hf_json("data/adapter_stats.json", {}, revision=revision)
    if not stats:
        stats = download_hf_json("adapter_stats.json", {}, revision=revision)

    report = download_hf_json("data/run_report.json", {}, revision=revision)
    if not report:
        report = download_hf_json("run_report.json", {}, revision=revision)

    return stats, report


def create_new_pr(api: HfApi) -> int:
    """Create a new PR on the HF dataset repo and return its number."""
    pr = api.create_pull_request(
        repo_id=REPO_ID,
        title="Automated Adapter Data Update",
        description="Data update from GitHub Actions",
        repo_type=REPO_TYPE,
    )
    print(f"  Created new PR #{pr.num} ({pr.url})")
    return pr.num


# ── Upload ───────────────────────────────────────────────────────────────────


def upload_to_hf(
    api: HfApi,
    existing_pr: Any | None,
) -> bool:
    """Upload data directory to HuggingFace via a PR.

    Reuses an existing open PR when available, otherwise creates a new one.
    Returns ``True`` on success.
    """
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set, skipping upload.")
        return False

    try:
        if existing_pr:
            pr_num = existing_pr.num
            print(f"  Uploading to existing PR #{pr_num}")
        else:
            pr_num = create_new_pr(api)

        revision = f"refs/pr/{pr_num}"
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(DATA_DIR),
            path_in_repo="data",
            repo_type=REPO_TYPE,
            revision=revision,
            commit_message=f"Automated data update ({today})",
        )
        print(f"  Upload complete to PR #{pr_num}")
        return True
    except Exception as e:
        print(f"  Upload failed: {e}")
        traceback.print_exc()
        return False


# ── Single-Adapter Processing ────────────────────────────────────────────────


def process_adapter(
    info: dict,
    stats: dict,
    today: datetime.datetime,
    env: dict[str, str],
) -> dict:
    """Run a single adapter through the full pipeline.

    Steps: schedule check → execute → validate → fingerprint → update stats.

    Returns a result dict with keys: ``status``, ``report_entry``, ``failed``.
    """
    adapter = info["name"]
    stat = stats.get(adapter, {})
    is_heavy = (
        stat.get("time_s", 0) > HEAVY_TIME_S
        or stat.get("size_mb", 0) > HEAVY_SIZE_MB
    )

    # ── Scheduling gate ──────────────────────────────────────────────────
    # Non-stale adapters are skipped.
    if not info["stale"]:
        label = "heavy, deferred" if is_heavy else "not stale"
        print(f"[{adapter}] Skipping ({label})")
        return _skip_result(label)

    # ── Prepare workspace ────────────────────────────────────────────────
    adapter_data_dir = DATA_DIR / adapter
    if adapter_data_dir.exists():
        shutil.rmtree(adapter_data_dir)
    adapter_data_dir.mkdir(parents=True, exist_ok=True)

    tmp_file: Path | None = None
    try:
        cmd, tmp_file = prepare_adapter_command(info, adapter_data_dir)

        # ── Execute ──────────────────────────────────────────────────────
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"[{adapter}] FAILED ({elapsed:.1f}s):\n{result.stderr[-500:]}")
            stats.setdefault(adapter, {})["last_failed"] = True
            return {
                "status": "exec_failed",
                "report_entry": {
                    "execution_failed": True,
                    "error": result.stderr[-500:],
                },
                "failed": True,
            }

        # ── Validate ─────────────────────────────────────────────────────
        print(f"[{adapter}] Ran in {elapsed:.1f}s, validating...")
        valid, failed, errors = validate_adapter_outputs(adapter_data_dir, env)
        print(f"[{adapter}] Validation: {valid} valid, {failed} failed")

        if valid == 0 or failed > 0:
            stats.setdefault(adapter, {})["last_failed"] = True
            return {
                "status": "validation_failed",
                "report_entry": {
                    "execution_failed": False,
                    "valid_files": valid,
                    "failed_files": failed,
                    "error": f"Validation: {valid} valid, {failed} failed",
                    "errors": errors[:50],
                },
                "failed": True,
            }

        # ── Duplicate detection ──────────────────────────────────────────
        fingerprint = compute_data_fingerprint(adapter_data_dir)
        stored_fingerprint = stat.get("data_fingerprint", "")
        data_changed = fingerprint != stored_fingerprint

        if data_changed:
            print(f"[{adapter}] New data detected (fingerprint changed)")
        else:
            print(f"[{adapter}] Data unchanged (same fingerprint as existing). Discarding outputs.")
            shutil.rmtree(adapter_data_dir)

        # ── Update stats ─────────────────────────────────────────────────
        update: dict[str, Any] = {
            "time_s": elapsed,
            "size_mb": get_dir_size_mb(adapter_data_dir) if adapter_data_dir.exists() else 0,
            "last_success_ts": time.time(),
            "last_check_ts": time.time(),
            "last_failed": False,
            "data_fingerprint": fingerprint,
        }
        if data_changed:
            update["last_data_change_ts"] = time.time()

        # Persist source headers for future staleness comparisons.
        for key, value in info["headers"].items():
            if value:
                update[key] = value

        stats.setdefault(adapter, {}).update(update)

        return {
            "status": "success",
            "report_entry": {
                "execution_failed": False,
                "valid_files": valid,
                "failed_files": failed,
                "data_changed": data_changed,
                "errors": errors[:50],
            },
            "failed": False,
        }

    except ValueError as e:
        # Raised by prepare_adapter_command when input URL is missing.
        print(f"[{adapter}] Skipping: {e}")
        return _skip_result(str(e))

    except Exception as e:
        print(f"[{adapter}] Exception: {e}")
        traceback.print_exc()
        stats.setdefault(adapter, {})["last_failed"] = True
        return {
            "status": "exception",
            "report_entry": {
                "execution_failed": True,
                "error": str(e),
            },
            "failed": True,
        }

    finally:
        if tmp_file and tmp_file.exists():
            tmp_file.unlink()


def _skip_result(reason: str) -> dict:
    """Build a result dict for a skipped adapter."""
    return {
        "status": "skipped",
        "report_entry": None,
        "failed": False,
        "skip_reason": reason,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all Every Eval Ever adapters with intelligent scheduling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run adapters locally but do not upload to HuggingFace",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force run all adapters regardless of staleness",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: discover, schedule, run, and upload adapters."""
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    # ── PR discovery & remote state ──────────────────────────────────────
    api = HfApi()
    print("Checking for existing PRs...")
    existing_pr = find_existing_pr(api)

    revision = f"refs/pr/{existing_pr.num}" if existing_pr else "main"
    if existing_pr:
        print(
            f"Found PR #{existing_pr.num} ({existing_pr.url}), "
            f"loading state from {revision}"
        )
    else:
        print("No open PR found, loading state from main")

    stats, _ = load_remote_state(revision)

    # ── Discover & analyse adapters ──────────────────────────────────────
    print("\nAnalysing adapters...")
    adapter_infos = discover_adapters(stats)
    
    if getattr(args, "force_all", False):
        print("Force run requested! Marking all adapters as stale.")
        for info in adapter_infos:
            info["stale"] = True
            info["reason"] = "forced by --force-all flag"

    print(
        f"Found {len(adapter_infos)} adapter(s): "
        f"{sum(1 for a in adapter_infos if a['stale'])} stale, "
        f"{sum(1 for a in adapter_infos if not a['stale'])} current"
    )

    today = datetime.datetime.now()
    env = build_adapter_env()
    report: dict[str, Any] = {
        "date": today.strftime("%Y-%m-%d"),
        "adapters": {},
    }
    any_failures = False
    summary: dict[str, list[str]] = {
        "ran": [],
        "skipped": [],
        "failed": [],
    }

    # ── Process each adapter ─────────────────────────────────────────────
    import concurrent.futures

    print(f"\n{'─' * 60}")
    for info in adapter_infos:
        print(f"Adapter: {info['name']} | Stale: {info['stale']} ({info['reason']})")

    print("\nRunning adapters in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_adapter, info, stats, today, env): info["name"]
            for info in adapter_infos
        }
        
        for future in concurrent.futures.as_completed(futures):
            adapter = futures[future]
            try:
                result = future.result()
                if result["report_entry"]:
                    report["adapters"][adapter] = result["report_entry"]

                if result["failed"]:
                    any_failures = True
                    summary["failed"].append(adapter)
                elif result["status"] == "success":
                    summary["ran"].append(adapter)
                else:
                    summary["skipped"].append(adapter)
            except Exception as e:
                print(f"[{adapter}] Unhandled exception in thread: {e}")
                traceback.print_exc()
                any_failures = True
                summary["failed"].append(adapter)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Run summary ({today.strftime('%Y-%m-%d')})")
    print(f"  Ran:     {len(summary['ran'])}  {summary['ran']}")
    print(f"  Skipped: {len(summary['skipped'])}  {summary['skipped']}")
    print(f"  Failed:  {len(summary['failed'])}  {summary['failed']}")
    print(f"{'═' * 60}")

    # ── Save state ───────────────────────────────────────────────────────
    save_json(STATS_FILE, stats)
    save_json(REPORT_FILE, report)

    # ── Upload ───────────────────────────────────────────────────────────
    has_results = bool(report["adapters"])
    if not args.dry_run and has_results:
        print("\nUploading data directory to HuggingFace (only changed files will be uploaded)...")
        if not upload_to_hf(api, existing_pr):
            any_failures = True
    elif not has_results:
        print("\nNo adapters ran — nothing to upload.")

    if any_failures:
        print("\nFinished with failures.")
        return 1

    print("\nAll done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
