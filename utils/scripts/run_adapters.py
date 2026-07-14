#!/usr/bin/env python3
"""Run cron-ready adapters with strict validation and collection-scoped PRs.

The orchestrator intentionally uses an explicit adapter contract table. It does
not infer CLI arguments or output ownership by searching adapter source text.
Adapters that cannot yet refresh atomically are reported as blocked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from every_eval_ever.cron_archive import (
    RawArtifact,
    append_ledger_event,
    archive_raw_artifacts,
    github_run_id,
)
from every_eval_ever.dedup import (
    compute_file_fingerprint,
    load_manifest,
    select_unique_files,
    validate_manifest,
)
from every_eval_ever.json_utils import strict_json_loads
from every_eval_ever.validation_core import resolve_companion_repo_path

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "utils"
DATASET_REPO_ID = "evaleval/EEE_datastore"
DATASET_REPO_TYPE = "dataset"
PR_TITLE_PREFIX = "[adapter-cron]"
DEFAULT_TIMEOUT_S = 60 * 60
REPORT_PATH = Path(
    os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
) / "adapter-cron-report.json"


@dataclass(frozen=True)
class CommandSpec:
    args: tuple[str, ...] = ()
    output_mode: Literal["collection", "root", "fixed"] = "collection"


@dataclass(frozen=True)
class RawCaptureSpec:
    option: str
    filename: str
    media_type: str


@dataclass(frozen=True)
class AdapterContract:
    name: str
    collections: tuple[str, ...]
    commands: tuple[CommandSpec, ...] = (CommandSpec(),)
    required_env: tuple[str, ...] = ()
    enabled: bool = True
    blocked_reason: str | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    raw_capture: RawCaptureSpec | None = None


CONTRACTS: tuple[AdapterContract, ...] = (
    AdapterContract(
        "arc_agi",
        ("arc-agi",),
        enabled=False,
        blocked_reason="requires --input-json; no source artifact URL is configured",
    ),
    AdapterContract(
        "artificial_analysis",
        ("artificial-analysis-llms",),
        required_env=("ARTIFICIAL_ANALYSIS_API_KEY",),
        raw_capture=RawCaptureSpec(
            "--save-raw-json",
            "payload.json",
            "application/json",
        ),
    ),
    AdapterContract(
        "bfcl",
        ("bfcl",),
        enabled=False,
        blocked_reason="requires --input-csv; no source artifact URL is configured",
    ),
    AdapterContract(
        "cocoabench",
        ("cocoabench",),
        enabled=False,
        blocked_reason="requires the benchmark-author CSV artifact",
    ),
    AdapterContract(
        "exgentic",
        (
            "appworld_test_normal",
            "browsecompplus",
            "swe-bench",
            "tau-bench-2_airline",
            "tau-bench-2_retail",
            "tau-bench-2_telecom",
        ),
        commands=(CommandSpec(("--from-hf",), "root"),),
    ),
    AdapterContract(
        "global-mmlu-lite",
        ("global-mmlu-lite",),
        enabled=False,
        blocked_reason=(
            "live fetch adapter catches source failures and exits successfully; "
            "historical migration only"
        ),
    ),
    AdapterContract(
        "hal",
        (
            "hal-assistantbench",
            "hal-corebench-hard",
            "hal-gaia",
            "hal-online-mind2web",
            "hal-scicode",
            "hal-scienceagentbench",
            "hal-swebench-verified-mini",
            "hal-taubench-airline",
            "hal-usaco",
        ),
        commands=(CommandSpec(("--benchmark", "all"), "root"),),
    ),
    AdapterContract(
        "helm",
        (
            "helm_air_bench",
            "helm_capabilities",
            "helm_classic",
            "helm_instruct",
            "helm_lite",
            "helm_mmlu",
            "helm_safety",
        ),
        commands=tuple(
            CommandSpec(("--leaderboard_name", leaderboard), "fixed")
            for leaderboard in (
                "HELM_AIR_Bench",
                "HELM_Capabilities",
                "HELM_Classic",
                "HELM_Instruct",
                "HELM_Lite",
                "HELM_MMLU",
                "HELM_Safety",
            )
        ),
        timeout_s=2 * DEFAULT_TIMEOUT_S,
    ),
    AdapterContract(
        "hfopenllm_v2",
        ("hfopenllm_v2",),
        enabled=False,
        blocked_reason="per-model exception handling can emit a partial refresh",
    ),
    AdapterContract(
        "hle",
        ("hle",),
        raw_capture=RawCaptureSpec(
            "--save-raw-json",
            "payload.json",
            "application/json",
        ),
    ),
    AdapterContract(
        "livecodebenchpro",
        ("livecodebenchpro",),
        enabled=False,
        blocked_reason="checked-in script is an obsolete schema migrator, not a fetch adapter",
    ),
    AdapterContract(
        "llm_stats",
        ("llm-stats",),
        required_env=("LLM_STATS_API_KEY",),
        raw_capture=RawCaptureSpec(
            "--save-raw-json",
            "payload.json",
            "application/json",
        ),
    ),
    AdapterContract(
        "mercor_eval",
        ("ace", "apex-agents", "apex-v1"),
        commands=(CommandSpec((), "root"),),
        required_env=("MERCOR_EVAL_API_EVALEVAL_KEY",),
        raw_capture=RawCaptureSpec(
            "--save-raw-json",
            "payload.json",
            "application/json",
        ),
    ),
    AdapterContract(
        "mmlu_pro",
        ("MMLU-Pro",),
        raw_capture=RawCaptureSpec(
            "--save-raw-csv",
            "payload.csv",
            "text/csv",
        ),
    ),
    AdapterContract(
        "mt_bench",
        ("mt-bench",),
        raw_capture=RawCaptureSpec(
            "--save-raw-jsonl",
            "payload.jsonl",
            "application/x-ndjson",
        ),
    ),
    AdapterContract(
        "multi_swe_bench",
        ("multi-swe-bench-leaderboard",),
        commands=(CommandSpec((), "fixed"),),
        timeout_s=2 * DEFAULT_TIMEOUT_S,
    ),
    AdapterContract("openeval", ("openeval",), timeout_s=2 * DEFAULT_TIMEOUT_S),
    AdapterContract(
        "rewardbench",
        ("reward-bench",),
        commands=(CommandSpec((), "fixed"),),
    ),
    AdapterContract(
        "sciarena",
        ("sciarena",),
        enabled=False,
        blocked_reason="requires --input-json; no source artifact URL is configured",
    ),
    AdapterContract(
        "swe_bench_verified",
        ("swe-bench-verified-leaderboard",),
        commands=(CommandSpec((), "fixed"),),
        timeout_s=2 * DEFAULT_TIMEOUT_S,
    ),
    AdapterContract(
        "swe_polybench",
        ("swe-polybench-leaderboard",),
        commands=(CommandSpec((), "fixed"),),
        timeout_s=2 * DEFAULT_TIMEOUT_S,
    ),
    AdapterContract(
        "terminal_bench_2",
        ("terminal-bench-2.0",),
        commands=(CommandSpec((), "fixed"),),
        raw_capture=RawCaptureSpec(
            "--save-raw-html",
            "payload.html",
            "text/html",
        ),
    ),
    AdapterContract(
        "vals_ai",
        ("vals-ai",),
        raw_capture=RawCaptureSpec(
            "--save-raw-json",
            "payload.json",
            "application/json",
        ),
    ),
)

# Scheduled GitHub runs are opt-in. Repository- and multi-benchmark adapters
# stay available through an explicit --adapter invocation without consuming
# the shared daily cron budget.
CRON_ALLOWLIST = frozenset(
    {
        "artificial_analysis",
        "hle",
        "llm_stats",
        "mercor_eval",
        "mmlu_pro",
        "mt_bench",
        "terminal_bench_2",
        "vals_ai",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Audit contracts only")
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help=(
            "Fetch, validate, and archive raw inputs without reading the "
            "datastore manifest or opening datastore PRs"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not open or update PRs")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Run every allowlisted adapter",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        dest="adapters",
        help="Run only this adapter; may be repeated",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Use a local datastore manifest instead of downloading main",
    )
    parser.add_argument(
        "--ingestion-repo",
        default=os.environ.get("EEE_INGESTION_REPO_ID"),
        help=(
            "Private HF dataset for raw inputs and the ingestion ledger. "
            "Required unless EEE_INGESTION_REPO_ID is set."
        ),
    )
    return parser.parse_args()


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assigned_days(name: str) -> set[int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    start = digest[0] % 7
    return {start, (start + 2) % 7, (start + 4) % 7}


def audit_contracts() -> list[str]:
    errors: list[str] = []
    discovered = {
        path.parent.name for path in UTILS_DIR.glob("*/adapter.py")
    }
    configured = {contract.name for contract in CONTRACTS}
    for name in sorted(discovered - configured):
        errors.append(f"adapter has no cron contract: {name}")
    for name in sorted(configured - discovered):
        errors.append(f"cron contract has no adapter.py: {name}")
    for name in sorted(CRON_ALLOWLIST - configured):
        errors.append(f"cron allowlist has no adapter contract: {name}")

    owners: dict[str, str] = {}
    for contract in CONTRACTS:
        if not contract.collections:
            errors.append(f"{contract.name}: collections must not be empty")
        if not contract.enabled and not contract.blocked_reason:
            errors.append(f"{contract.name}: disabled contract needs a reason")
        if contract.name in CRON_ALLOWLIST and not contract.enabled:
            errors.append(
                f"{contract.name}: blocked adapter is cron-allowlisted"
            )
        if contract.name in CRON_ALLOWLIST and contract.raw_capture is None:
            errors.append(
                f"{contract.name}: cron adapter needs a raw capture contract"
            )
        if contract.raw_capture is not None and len(contract.commands) != 1:
            errors.append(
                f"{contract.name}: raw capture requires exactly one command"
            )
        for collection in contract.collections:
            previous = owners.get(collection)
            if previous is not None:
                errors.append(
                    f"collection {collection!r} owned by both {previous} and "
                    f"{contract.name}"
                )
            owners[collection] = contract.name
        for command in contract.commands:
            if command.output_mode == "collection" and len(contract.collections) != 1:
                errors.append(
                    f"{contract.name}: collection output mode requires one collection"
                )
    return errors


def print_audit(errors: list[str]) -> None:
    print("Adapter cron contracts")
    for contract in CONTRACTS:
        if not contract.enabled:
            state = f"blocked: {contract.blocked_reason}"
        elif contract.name in CRON_ALLOWLIST:
            state = "scheduled"
        else:
            state = "manual only"
        print(
            f"- {contract.name}: {state}; "
            f"collections={','.join(contract.collections)}"
        )
    if errors:
        print("\nContract errors:")
        for error in errors:
            print(f"- {error}")


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def selected_contracts(args: argparse.Namespace) -> list[AdapterContract]:
    requested = set(args.adapters or [])
    by_name = {contract.name: contract for contract in CONTRACTS}
    known = set(by_name)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown adapter selection: {unknown}")
    blocked = sorted(name for name in requested if not by_name[name].enabled)
    if blocked:
        raise ValueError(f"adapter is not cron-ready: {blocked}")

    today = dt.datetime.now(dt.UTC).weekday()
    selected: list[AdapterContract] = []
    for contract in CONTRACTS:
        if requested and contract.name not in requested:
            continue
        if not contract.enabled:
            continue
        if requested:
            selected.append(contract)
            continue
        if contract.name not in CRON_ALLOWLIST:
            continue
        if args.force_all or today in assigned_days(contract.name):
            selected.append(contract)
    return selected


def command_for(
    contract: AdapterContract,
    spec: CommandSpec,
    workspace: Path,
) -> list[str]:
    adapter_path = UTILS_DIR / contract.name / "adapter.py"
    command = [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT),
        str(adapter_path),
        *spec.args,
    ]
    if spec.output_mode == "collection":
        output = workspace / "data" / contract.collections[0]
        command.extend(("--output-dir", str(output)))
    elif spec.output_mode == "root":
        command.extend(("--output-dir", str(workspace / "data")))
    if contract.raw_capture is not None:
        raw_path = workspace / "raw" / contract.raw_capture.filename
        command.extend((contract.raw_capture.option, str(raw_path)))
    return command


def run_contract(
    contract: AdapterContract,
    env: dict[str, str],
) -> tuple[Path | None, dict[str, Any]]:
    missing_env = [name for name in contract.required_env if not env.get(name)]
    if missing_env:
        return None, {
            "status": "failed",
            "error": f"missing required environment variables: {missing_env}",
        }

    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"eee-adapter-{contract.name}-",
            dir=os.environ.get("RUNNER_TEMP"),
        )
    )
    if contract.raw_capture is not None:
        (workspace / "raw").mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.UTC)
    logs: list[dict[str, Any]] = []
    for spec in contract.commands:
        command = command_for(contract, spec, workspace)
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=contract.timeout_s,
            check=False,
        )
        logs.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
        if result.returncode != 0:
            return None, {
                "status": "failed",
                "error": f"adapter command exited {result.returncode}",
                "commands": logs,
            }

    raw_input_bytes = 0
    if contract.raw_capture is not None:
        raw_path = workspace / "raw" / contract.raw_capture.filename
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            return None, {
                "status": "failed",
                "error": (
                    "adapter did not emit its required raw input snapshot: "
                    f"{raw_path}"
                ),
                "commands": logs,
            }
        raw_input_bytes = raw_path.stat().st_size

    data_root = workspace / "data"
    actual = (
        {path.name for path in data_root.iterdir() if path.is_dir()}
        if data_root.exists()
        else set()
    )
    expected = set(contract.collections)
    loose_files = (
        [str(path) for path in data_root.iterdir() if path.is_file()]
        if data_root.exists()
        else []
    )
    if actual != expected or loose_files:
        return workspace, {
            "status": "failed",
            "error": (
                f"output ownership mismatch: expected={sorted(expected)}, "
                f"actual={sorted(actual)}, loose_files={loose_files}"
            ),
            "raw_input_bytes": raw_input_bytes,
            "commands": logs,
        }

    validate_command = [
        "uv",
        "run",
        "--project",
        str(REPO_ROOT),
        "every_eval_ever",
        "validate",
        str(data_root),
        "--format",
        "json",
    ]
    validation = subprocess.run(
        validate_command,
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=contract.timeout_s,
        check=False,
    )
    try:
        reports = json.loads(validation.stdout)
    except json.JSONDecodeError as exc:
        return workspace, {
            "status": "failed",
            "error": f"validator output was not JSON: {exc}",
            "validator_stdout_tail": validation.stdout[-2000:],
            "validator_stderr_tail": validation.stderr[-2000:],
            "raw_input_bytes": raw_input_bytes,
            "commands": logs,
        }
    invalid = [report for report in reports if not report.get("valid")]
    warned = [report for report in reports if report.get("warnings")]
    if validation.returncode != 0 or invalid or warned or not reports:
        return workspace, {
            "status": "failed",
            "error": (
                f"validation failed: returncode={validation.returncode}, "
                f"invalid={len(invalid)}, warned={len(warned)}, "
                f"reports={len(reports)}"
            ),
            "validation_errors": invalid[:20],
            "validation_warnings": warned[:20],
            "raw_input_bytes": raw_input_bytes,
            "commands": logs,
        }

    output_paths = [path for path in data_root.rglob("*") if path.is_file()]
    elapsed = (dt.datetime.now(dt.UTC) - started).total_seconds()
    return workspace, {
        "status": "ready",
        "elapsed_s": elapsed,
        "validated_files": len(reports),
        "output_files": len(output_paths),
        "output_bytes": sum(path.stat().st_size for path in output_paths),
        "raw_input_bytes": raw_input_bytes,
        "commands": logs,
    }


def collect_raw_artifacts(
    contracts: list[AdapterContract],
    workspaces: dict[str, Path],
) -> list[RawArtifact]:
    artifacts: list[RawArtifact] = []
    for contract in contracts:
        workspace = workspaces.get(contract.name)
        capture = contract.raw_capture
        if workspace is None or capture is None:
            continue
        artifacts.append(
            RawArtifact(
                adapter=contract.name,
                logical_name=capture.filename,
                local_path=workspace / "raw" / capture.filename,
                media_type=capture.media_type,
            )
        )
    return artifacts


def open_collection_prs(api: HfApi) -> dict[str, Any]:
    current_user = api.whoami().get("name")
    result: dict[str, Any] = {}
    for discussion in api.get_repo_discussions(
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
    ):
        if not getattr(discussion, "is_pull_request", False):
            continue
        if discussion.status not in {"open", "draft"}:
            continue
        if current_user and discussion.author != current_user:
            continue
        title = str(discussion.title)
        prefix = f"{PR_TITLE_PREFIX} "
        if not title.startswith(prefix):
            continue
        collection = title.removeprefix(prefix)
        if collection in result:
            raise ValueError(f"multiple open adapter cron PRs for {collection}")
        result[collection] = discussion
    return result


def augment_manifest_with_pending_prs(
    api: HfApi,
    manifest: dict[str, Any],
    prs: dict[str, Any],
) -> dict[str, Any]:
    augmented = json.loads(json.dumps(manifest))
    manifest_files = augmented["files"]
    for collection, discussion in sorted(prs.items()):
        revision = f"refs/pr/{discussion.num}"
        prefix = f"data/{collection}/"
        repo_files = api.list_repo_files(
            repo_id=DATASET_REPO_ID,
            repo_type=DATASET_REPO_TYPE,
            revision=revision,
        )
        pending = [
            path
            for path in repo_files
            if path.startswith(prefix)
            and path.endswith(".json")
            and path not in manifest_files
        ]
        for repo_path in pending:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                repo_type=DATASET_REPO_TYPE,
                filename=repo_path,
                revision=revision,
                token=api.token,
            )
            manifest_files[repo_path] = {
                "fingerprint": compute_file_fingerprint(local_path)
            }
    return augmented


def staged_aggregates(
    workspaces: dict[str, Path],
) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
    repo_paths: list[str] = []
    local_paths: dict[str, Path] = {}
    roots: dict[str, Path] = {}
    for adapter, workspace in sorted(workspaces.items()):
        roots[adapter] = workspace
        for path in sorted((workspace / "data").rglob("*.json")):
            repo_path = path.relative_to(workspace).as_posix()
            if repo_path in local_paths:
                raise ValueError(f"two adapters emitted the same path: {repo_path}")
            repo_paths.append(repo_path)
            local_paths[repo_path] = path
    return repo_paths, local_paths, roots


def select_new_files(
    manifest: dict[str, Any],
    workspaces: dict[str, Path],
) -> tuple[dict[str, dict[str, Path]], list[dict[str, str]]]:
    repo_paths, local_paths, _ = staged_aggregates(workspaces)
    dedup = select_unique_files(repo_paths, local_paths, manifest)
    selected: dict[str, dict[str, Path]] = {}
    duplicates = [
        {
            "file": result.file_path,
            "duplicate_of": (
                result.duplicate_of
                or result.matched_manifest_path
                or "unknown"
            ),
        }
        for result in dedup.duplicate_results
    ]
    for repo_path in dedup.accepted_paths:
        local = local_paths[repo_path]
        collection = repo_path.split("/", 2)[1]
        selected.setdefault(collection, {})[repo_path] = local

        data = strict_json_loads(local.read_bytes())
        if not isinstance(data, dict):
            raise ValueError(f"{local}: aggregate JSON must be an object")
        companion = resolve_companion_repo_path(repo_path, data)
        if companion is None:
            continue
        workspace = local
        for _ in Path(repo_path).parts:
            workspace = workspace.parent
        companion_local = workspace / companion
        if not companion_local.is_file():
            raise ValueError(
                f"{repo_path}: companion does not exist: {companion}"
            )
        selected[collection][companion] = companion_local
    return selected, duplicates


def ensure_collection_pr(
    api: HfApi,
    collection: str,
    existing: Any | None,
) -> Any:
    if existing is not None:
        return existing
    return api.create_pull_request(
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
        title=f"{PR_TITLE_PREFIX} {collection}",
        description=(
            "Automated adapter refresh. This PR is collection-scoped and was "
            "created only after strict validation and manifest-backed semantic "
            "deduplication."
        ),
    )


def upload_collections(
    api: HfApi,
    selected: dict[str, dict[str, Path]],
    prs: dict[str, Any],
) -> dict[str, str]:
    urls: dict[str, str] = {}
    today = dt.datetime.now(dt.UTC).date().isoformat()
    for collection, files in sorted(selected.items()):
        discussion = ensure_collection_pr(api, collection, prs.get(collection))
        revision = f"refs/pr/{discussion.num}"
        operations = [
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(local_path),
            )
            for repo_path, local_path in sorted(files.items())
        ]
        api.create_commit(
            repo_id=DATASET_REPO_ID,
            repo_type=DATASET_REPO_TYPE,
            revision=revision,
            operations=operations,
            commit_message=f"Refresh {collection} ({today})",
        )
        urls[collection] = str(discussion.url)
    return urls


def ledger_adapter_results(
    entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    fields = (
        "status",
        "error",
        "elapsed_s",
        "validated_files",
        "output_files",
        "output_bytes",
        "raw_input_bytes",
    )
    return {
        name: {field: entry[field] for field in fields if field in entry}
        for name, entry in entries.items()
    }


def main() -> int:
    args = parse_args()
    contract_errors = audit_contracts()
    if args.audit:
        print_audit(contract_errors)
        return 1 if contract_errors else 0
    if contract_errors:
        print_audit(contract_errors)
        return 1

    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "allowlisted": sorted(CRON_ALLOWLIST),
        "blocked": {
            contract.name: contract.blocked_reason
            for contract in CONTRACTS
            if not contract.enabled
        },
        "manual_only": sorted(
            contract.name
            for contract in CONTRACTS
            if contract.enabled and contract.name not in CRON_ALLOWLIST
        ),
        "adapters": {},
        "duplicates": [],
        "selected_files": {},
        "prs": {},
    }
    run_id = github_run_id(
        os.environ.get("GITHUB_RUN_ID"),
        os.environ.get("GITHUB_RUN_ATTEMPT"),
    )
    report["run_id"] = run_id
    report["ingestion_repo"] = args.ingestion_repo
    ingestion_api: HfApi | None = None
    raw_archived = False
    final_event_written = False
    try:
        contracts = selected_contracts(args)
        report["scheduled"] = [contract.name for contract in contracts]
        ingestion_token = os.environ.get("EEE_INGESTION_HF_TOKEN")
        if not ingestion_token:
            raise ValueError(
                "EEE_INGESTION_HF_TOKEN is required for the private raw archive"
            )
        if not args.ingestion_repo:
            raise ValueError(
                "EEE_INGESTION_REPO_ID or --ingestion-repo is required"
            )
        ingestion_api = HfApi(token=ingestion_token)
        env = build_env()
        workspaces: dict[str, Path] = {}
        failed = False
        for contract in contracts:
            print(f"[{contract.name}] running")
            workspace, entry = run_contract(contract, env)
            report["adapters"][contract.name] = entry
            if workspace is not None:
                workspaces[contract.name] = workspace
            if entry.get("status") != "ready":
                failed = True
                print(f"[{contract.name}] failed: {entry.get('error')}")
            else:
                print(
                    f"[{contract.name}] ready: "
                    f"{entry.get('validated_files')} validated files, "
                    f"{entry.get('output_bytes')} output bytes"
                )

        raw_artifacts = collect_raw_artifacts(contracts, workspaces)
        archived = archive_raw_artifacts(
            ingestion_api,
            repo_id=args.ingestion_repo,
            run_id=run_id,
            artifacts=raw_artifacts,
            run_metadata={
                "selected_adapters": [contract.name for contract in contracts],
                "adapter_results": ledger_adapter_results(report["adapters"]),
            },
        )
        raw_archived = True
        report["raw_archive"] = [
            {
                "adapter": artifact.adapter,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "path": artifact.archive_path,
            }
            for artifact in archived
        ]

        if failed:
            report["status"] = "adapter_failed_no_upload"
            append_ledger_event(
                ingestion_api,
                repo_id=args.ingestion_repo,
                run_id=run_id,
                phase="failed",
                payload={
                    "status": report["status"],
                    "adapter_results": ledger_adapter_results(
                        report["adapters"]
                    ),
                },
            )
            final_event_written = True
            save_report(args.report, report)
            return 1

        if args.archive_only:
            report["status"] = "archive_only"
            append_ledger_event(
                ingestion_api,
                repo_id=args.ingestion_repo,
                run_id=run_id,
                phase="completed",
                payload={
                    "status": report["status"],
                    "adapter_results": ledger_adapter_results(
                        report["adapters"]
                    ),
                    "duplicates": [],
                    "selected_files": {},
                    "prs": {},
                },
            )
            final_event_written = True
            save_report(args.report, report)
            print(
                f"status={report['status']} adapters={len(contracts)} "
                f"report={args.report}"
            )
            return 0

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN is required for datastore deduplication and PRs"
            )
        api = HfApi(token=token)
        prs = open_collection_prs(api)
        if args.manifest is None:
            manifest = load_manifest(
                api=api,
                dataset_repo_id=DATASET_REPO_ID,
                revision="main",
            )
        else:
            manifest = strict_json_loads(args.manifest.read_bytes())
            if not isinstance(manifest, dict):
                raise ValueError("local manifest must contain a JSON object")
            validate_manifest(manifest, manifest_path=str(args.manifest))
        manifest = augment_manifest_with_pending_prs(api, manifest, prs)
        selected, duplicates = select_new_files(manifest, workspaces)
        report["duplicates"] = duplicates
        report["selected_files"] = {
            collection: sorted(files)
            for collection, files in selected.items()
        }

        if args.dry_run:
            report["status"] = "dry_run"
        elif selected:
            report["prs"] = upload_collections(api, selected, prs)
            report["status"] = "uploaded"
        else:
            report["status"] = "no_changes"
        append_ledger_event(
            ingestion_api,
            repo_id=args.ingestion_repo,
            run_id=run_id,
            phase="completed",
            payload={
                "status": report["status"],
                "adapter_results": ledger_adapter_results(report["adapters"]),
                "duplicates": report["duplicates"],
                "selected_files": report["selected_files"],
                "prs": report["prs"],
            },
        )
        final_event_written = True
        save_report(args.report, report)
        print(
            f"status={report['status']} collections={len(selected)} "
            f"duplicates={len(duplicates)} report={args.report}"
        )
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if (
            ingestion_api is not None
            and raw_archived
            and not final_event_written
        ):
            try:
                append_ledger_event(
                    ingestion_api,
                    repo_id=args.ingestion_repo,
                    run_id=run_id,
                    phase="failed",
                    payload={
                        "status": report["status"],
                        "error": report["error"],
                        "duplicates": report.get("duplicates", []),
                        "selected_files": report.get("selected_files", {}),
                        "prs": report.get("prs", {}),
                    },
                )
            except Exception as ledger_exc:
                report["ledger_error"] = (
                    f"{type(ledger_exc).__name__}: {ledger_exc}"
                )
        save_report(args.report, report)
        print(report["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
