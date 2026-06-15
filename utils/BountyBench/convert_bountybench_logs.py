#!/usr/bin/env python3
"""Convert BountyBench run logs to Every Eval Ever schema.

Reads all JSON logs from a BountyBench run directory, deduplicates by
(task_dir, bounty_number) keeping the best result, then converts to
EEE aggregate + instance-level format.

Requires every_eval_ever to be installed:
    pip install every_eval_ever
    # or: uv add every_eval_ever

Usage:
    python convert_bountybench_logs.py \
        --logs-dir /path/to/bountybench/logs/2026-03-26 \
        --output-dir /path/to/eee_output \
        --source-org "Your Organization"

    # Dry-run (shows per-bounty results without writing files)
    python convert_bountybench_logs.py \
        --logs-dir /path/to/bountybench/logs/2026-03-26 \
        --output-dir /path/to/eee_output \
        --source-org "Your Organization" \
        --dry-run

Input log structure
-------------------
BountyBench produces one JSON file per bounty per run, in a structure like:
`logs/{date}/{workflow}/{task}_{bounty_idx}/{model}/{model}_{workflow}_{task}_{bounty_idx}_{timestamp}.json`,
e.g. `bountybench/logs/2026-03-18/DetectWorkflow/astropy_0/anthropic-claude-opus-4-6/anthropic-claude-opus-4-6_DetectWorkflow_astropy_0_4477350480_2026-03-18_11-39-06.json`.
Where {workflow} is one of: DetectWorkflow, ExploitWorkflow, PatchWorkflow.
It does not explicitly link a single run's per-bounty logs together with a common ID
or produce an aggregated "summary" log. The logs within a given {date} folder can be
from multiple runs (potentially with different configurations) and a single run can
span more than one {date} folder.

Each JSON file contains:
  - workflow_metadata: workflow name, task directory, bounty number, success/complete flags
  - workflow_usage: token counts and query time
  - resources_used.model.config: model name and generation parameters
  - phase_messages: the full agent transcript as a list of phases, each with
    agent_messages and action_messages (tool calls + results)
  - additional_metadata.bounty_metadata: CVE, CWE, severity, bounty link

Output structure
----------------
This script produces one EEE aggregate+instance pair per (model, workflow)
combination found in the input directory. It does NOT aggregate across different
workflows — each workflow (Detect, Exploit, Patch) is a distinct evaluation task
with its own success criteria, so they are kept separate. Run the script once per
workflow directory:

    python convert_bountybench_logs.py --logs-dir .../DetectWorkflow ...
    python convert_bountybench_logs.py --logs-dir .../ExploitWorkflow ...
    python convert_bountybench_logs.py --logs-dir .../PatchWorkflow ...

Output files per (model, workflow):
    {output_dir}/BountyBench/{developer}/{model}/{uuid}.json          # aggregate
    {output_dir}/BountyBench/{developer}/{model}/{uuid}_samples.jsonl # instance-level

Deduplication
-------------
BountyBench runs can produce multiple logs for the same bounty (retries).
This script keeps the best result per (task_dir, bounty_number), preferring
success > complete > non-startup-failure > latest timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    DetailedEvaluationResults,
    EvalLibrary,
    EvalLimits,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    GenerationArgs,
    GenerationConfig,
    HashAlgorithm,
    MetricConfig,
    ModelInfo,
    Sandbox,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
    Uncertainty,
    StandardError,
)
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Message,
    Performance,
    TokenUsage,
    ToolCall,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BOUNTYBENCH_GITHUB = "https://github.com/bountybench/bountybench"

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_bounty_log(path: Path) -> dict[str, Any]:
    """Parse a single BountyBench JSON log into a normalized dict."""
    data = json.loads(path.read_text())

    wm = data.get("workflow_metadata") or {}
    summary = wm.get("workflow_summary") or {}
    task = wm.get("task") or {}
    usage = data.get("workflow_usage") or {}
    resources = data.get("resources_used") or {}
    model_cfg = (resources.get("model") or {}).get("config") or {}
    bounty_meta = (data.get("additional_metadata") or {}).get("bounty_metadata") or {}

    # Determine model
    model = model_cfg.get("model", "")
    if not model:
        fname = path.stem
        if fname.startswith("claude-code_"):
            model = "claude-code"
        else:
            parts = fname.split("_")
            model = parts[0].replace("-", "/", 1) if parts else "unknown"

    task_dir = task.get("task_dir", "unknown")
    bounty_number = str(task.get("bounty_number", "0"))

    # Check for startup failure
    phase_messages = data.get("phase_messages") or []
    max_iterations = 0
    if phase_messages:
        max_iterations = phase_messages[0].get("max_iterations", 0) or 0
    startup_failure = max_iterations == 0

    # Parse timestamps
    start_time = data.get("start_time", "")
    end_time = data.get("end_time", "")
    duration_ms = None
    try:
        if start_time and end_time:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t0 = datetime.strptime(start_time, fmt)
            t1 = datetime.strptime(end_time, fmt)
            duration_ms = (t1 - t0).total_seconds() * 1000
    except Exception:
        pass

    return {
        "path": path,
        "raw": data,
        "task_dir": task_dir,
        "bounty_number": bounty_number,
        "bounty_id": f"{task_dir.replace('bountytasks/', '')}_{bounty_number}",
        "model": model,
        "workflow": wm.get("workflow_name", "unknown"),
        "success": summary.get("success", False),
        "complete": summary.get("complete", False),
        "startup_failure": startup_failure,
        "input_tokens": usage.get("total_input_tokens", 0) or 0,
        "output_tokens": usage.get("total_output_tokens", 0) or 0,
        "query_time_ms": usage.get("total_query_time_taken_in_ms", 0) or 0,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "max_iterations": max_iterations,
        "phase_messages": phase_messages,
        "model_config": model_cfg,
        "bounty_metadata": bounty_meta,
        "cve": bounty_meta.get("CVE", ""),
        "severity": str(bounty_meta.get("severity", "")),
        "cwe": bounty_meta.get("CWE", ""),
    }


def collect_logs(logs_dir: Path) -> list[dict[str, Any]]:
    """Collect all JSON logs from a directory tree."""
    results = []
    for f in sorted(logs_dir.rglob("*.json")):
        try:
            results.append(parse_bounty_log(f))
        except Exception as e:
            logger.warning("Error parsing %s: %s", f, e)
    return results


def deduplicate_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep best log per (task_dir, bounty_number): success > complete > rest.

    Among ties in status, prefer latest by start_time.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}

    def rank(log: dict) -> tuple:
        return (
            log["success"],  # True > False
            log["complete"],  # True > False
            not log["startup_failure"],  # non-startup > startup
            log["start_time"],  # latest wins
        )

    for log in logs:
        key = (log["task_dir"], log["bounty_number"])
        existing = best.get(key)
        if existing is None or rank(log) > rank(existing):
            best[key] = log

    return sorted(best.values(), key=lambda x: x["bounty_id"])


def filter_usable(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove startup failures (no usable data)."""
    usable = [lg for lg in logs if not lg["startup_failure"]]
    removed = len(logs) - len(usable)
    if removed:
        logger.info("Filtered out %d startup failures", removed)
    return usable


# ---------------------------------------------------------------------------
# EEE conversion — aggregate
# ---------------------------------------------------------------------------


def model_id_from_bb(model: str) -> str:
    """Convert BountyBench model string to HuggingFace-style model ID."""
    if "/" in model:
        return model
    if model:
        logger.warning(
            "Model %r is not in provider/model format; using unknown/%s", model, model
        )
        return f"unknown/{model}"
    logger.warning("No model found in log; using unknown/unknown")
    return "unknown/unknown"


def model_developer(model_id: str) -> str:
    """Extract developer name from model ID."""
    if "/" in model_id:
        dev = model_id.split("/")[0]
        return dev.replace("-", " ").title()
    return "Unknown"


def convert_timestamp_to_unix(ts: str) -> str:
    """Convert ISO timestamp to Unix epoch string."""
    if not ts:
        return str(datetime.now().timestamp())
    try:
        dt = datetime.fromisoformat(ts)
        return str(dt.timestamp())
    except Exception:
        return str(datetime.now().timestamp())


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def sha256_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_aggregate(
    logs: list[dict[str, Any]],
    file_uuid: str,
    output_dir: Path,
    source_org: str,
) -> EvaluationLog:
    """Build the aggregate EvaluationLog from deduplicated bounty logs."""
    # All logs should be same model/workflow
    model = logs[0]["model"]
    model_id = model_id_from_bb(model)
    workflow = logs[0]["workflow"]

    # Timestamps: use earliest start, latest end
    start_times = [lg["start_time"] for lg in logs if lg["start_time"]]
    eval_timestamp = min(start_times) if start_times else ""
    eval_unix = convert_timestamp_to_unix(eval_timestamp)
    retrieved_unix = str(datetime.now().timestamp())

    # Score: success rate
    n_success = sum(1 for lg in logs if lg["success"])
    n_total = len(logs)
    success_rate = n_success / n_total if n_total > 0 else 0.0

    # Model info
    if "/" in model_id:
        dev_slug, model_name = model_id.split("/", 1)
    else:
        dev_slug, model_name = "unknown", model_id

    model_info = ModelInfo(
        name=model,
        id=model_id,
        developer=model_developer(model_id),
    )

    # Source data
    source_data = SourceDataUrl(
        dataset_name="BountyBench",
        source_type="url",
        url=[BOUNTYBENCH_GITHUB],
        additional_details={
            "num_bounties": str(n_total),
            "workflow": workflow,
        },
    )

    # Generation config
    model_cfg = logs[0]["model_config"]
    max_iter = max(lg["max_iterations"] for lg in logs)
    generation_config = GenerationConfig(
        generation_args=GenerationArgs(
            temperature=model_cfg.get("temperature"),
            max_tokens=model_cfg.get("max_output_tokens"),
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(name="bash", description="Kali Linux terminal"),
                ],
            ),
            eval_limits=EvalLimits(message_limit=max_iter),
            sandbox=Sandbox(type="docker"),
        ),
        additional_details={
            "max_input_tokens": str(model_cfg.get("max_input_tokens", "")),
            "helm": str(model_cfg.get("helm", False)),
        }
        if model_cfg
        else None,
    )

    # Metric config
    metric_config = MetricConfig(
        evaluation_description=f"BountyBench {workflow} — success rate across bounties",
        metric_id="accuracy",
        metric_name="Success Rate",
        metric_kind="accuracy",
        metric_unit="proportion",
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=1.0,
    )

    # Uncertainty
    import math

    stderr = (
        math.sqrt(success_rate * (1 - success_rate) / n_total) if n_total > 0 else 0
    )
    uncertainty = Uncertainty(
        standard_error=StandardError(value=stderr, method="analytic"),
        num_samples=n_total,
    )

    eval_result = EvaluationResult(
        evaluation_result_id=f"bountybench_detect_{dev_slug}_{model_name}",
        evaluation_name=f"BountyBench - {workflow}",
        source_data=source_data,
        evaluation_timestamp=eval_unix,
        metric_config=metric_config,
        score_details=ScoreDetails(
            score=success_rate,
            details={
                "successes": str(n_success),
                "total": str(n_total),
            },
            uncertainty=uncertainty,
        ),
        generation_config=generation_config,
    )

    # Instance-level results path
    jsonl_filename = f"{file_uuid}_samples.jsonl"

    # We'll fill in checksum/total_rows after writing the JSONL
    detailed_results = DetailedEvaluationResults(
        format=Format.jsonl,
        file_path=f"./{jsonl_filename}",
        hash_algorithm=HashAlgorithm.sha256,
    )

    evaluation_id = f"BountyBench/{model_id.replace('/', '_')}/{eval_unix}"

    source_metadata = SourceMetadata(
        source_name="BountyBench",
        source_type=SourceType.evaluation_run,
        source_organization_name=source_org,
        evaluator_relationship=EvaluatorRelationship.third_party,
    )

    eval_library = EvalLibrary(
        name="bountybench",
        version="unknown",
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        evaluation_timestamp=eval_unix,
        retrieved_timestamp=retrieved_unix,
        source_metadata=source_metadata,
        eval_library=eval_library,
        model_info=model_info,
        evaluation_results=[eval_result],
        detailed_evaluation_results=detailed_results,
    )


# ---------------------------------------------------------------------------
# EEE conversion — instance-level
# ---------------------------------------------------------------------------


def build_messages_from_phases(
    phase_messages: list[dict],
) -> list[Message]:
    """Convert BountyBench phase_messages into EEE Message list."""
    messages: list[Message] = []
    turn_idx = 0

    for phase in phase_messages:
        for am in phase.get("agent_messages") or []:
            agent_id = am.get("agent_id", "unknown")
            msg_text = am.get("message") or ""

            # System messages
            if agent_id == "system":
                messages.append(
                    Message(
                        turn_idx=turn_idx,
                        role="system",
                        content=msg_text,
                    )
                )
                turn_idx += 1
                continue

            # Agent messages map to assistant turns
            if msg_text:
                messages.append(
                    Message(
                        turn_idx=turn_idx,
                        role="assistant",
                        content=msg_text,
                        tool_calls=None,
                    )
                )
                turn_idx += 1

            # Action messages map to tool calls + tool results
            for action in am.get("action_messages") or []:
                resource_id = action.get("resource_id") or "unknown"
                command = action.get("command") or ""
                meta = action.get("additional_metadata") or {}
                action_command = command or meta.get("command", "")
                result_text = action.get("message") or ""

                if resource_id == "model":
                    # Model action: this is an LLM call, emit as assistant
                    if action_command:
                        tool_call_id = f"tc_{turn_idx}"
                        messages.append(
                            Message(
                                turn_idx=turn_idx,
                                role="assistant",
                                content=None,
                                tool_calls=[
                                    ToolCall(
                                        id=tool_call_id,
                                        name="bash",
                                        arguments={"command": action_command},
                                    )
                                ],
                            )
                        )
                        turn_idx += 1
                elif action_command or result_text:
                    # Tool execution: command + result
                    tool_call_id = f"tc_{turn_idx}"
                    if action_command:
                        messages.append(
                            Message(
                                turn_idx=turn_idx,
                                role="assistant",
                                content=None,
                                tool_calls=[
                                    ToolCall(
                                        id=tool_call_id,
                                        name=resource_id,
                                        arguments={"command": action_command},
                                    )
                                ],
                            )
                        )
                        turn_idx += 1
                    if result_text:
                        messages.append(
                            Message(
                                turn_idx=turn_idx,
                                role="tool",
                                content=result_text[:10000],  # cap very long outputs
                                tool_call_id=[tool_call_id],
                            )
                        )
                        turn_idx += 1

    return messages


def build_instance_level(
    log: dict[str, Any],
    evaluation_id: str,
    model_id: str,
    evaluation_name: str,
) -> InstanceLevelEvaluationLog:
    """Build one instance-level record for a single bounty."""
    bounty_id = log["bounty_id"]
    task_name = log["task_dir"].replace("bountytasks/", "")
    cve = log["cve"]
    severity = log["severity"]
    cwe = log["cwe"]

    # Input: the system prompt / task description
    phase_messages = log["phase_messages"]
    system_msg = ""
    if phase_messages:
        for am in phase_messages[0].get("agent_messages", []):
            if am.get("agent_id") == "system":
                system_msg = am.get("message", "")
                break

    raw_input = system_msg or f"BountyBench {task_name} bounty {log['bounty_number']}"
    reference = ["success"] if log["success"] else ["failure"]

    sample_input = Input(
        raw=raw_input,
        reference=reference,
    )

    # Messages
    messages = build_messages_from_phases(phase_messages)

    # Evaluation
    score = 1.0 if log["success"] else 0.0
    n_turns = len(messages) if messages else 1
    tool_calls_count = sum(len(msg.tool_calls) for msg in messages if msg.tool_calls)

    evaluation = Evaluation(
        score=score,
        is_correct=log["success"],
        num_turns=n_turns,
        tool_calls_count=tool_calls_count,
    )

    # Token usage
    token_usage = TokenUsage(
        input_tokens=log["input_tokens"],
        output_tokens=log["output_tokens"],
        total_tokens=log["input_tokens"] + log["output_tokens"],
    )

    # Performance
    performance = None
    if log["duration_ms"] is not None:
        performance = Performance(
            latency_ms=log["duration_ms"],
            generation_time_ms=log["query_time_ms"] or None,
        )

    # Answer attribution
    answer_attribution = [
        AnswerAttributionItem(
            turn_idx=max(0, n_turns - 1),
            source="workflow_metadata.workflow_summary.success",
            extracted_value=str(log["success"]),
            extraction_method="exact_match",
            is_terminal=True,
        )
    ]

    # Metadata
    metadata: dict[str, Any] = {
        "task_dir": log["task_dir"],
        "bounty_number": log["bounty_number"],
        "workflow": log["workflow"],
        "complete": str(log["complete"]),
    }
    if cve:
        metadata["CVE"] = cve
    if severity:
        metadata["severity"] = severity
    if cwe:
        metadata["CWE"] = cwe
    bounty_link = log["bounty_metadata"].get("bounty_link", "")
    if bounty_link:
        metadata["bounty_link"] = bounty_link

    sample_hash = sha256_string(raw_input + "".join(reference))

    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        model_id=model_id,
        evaluation_name=evaluation_name,
        sample_id=bounty_id,
        sample_hash=sample_hash,
        interaction_type=InteractionType.agentic,
        input=sample_input,
        output=None,
        messages=messages
        if messages
        else [
            Message(turn_idx=0, role="system", content=raw_input),
        ],
        answer_attribution=answer_attribution,
        evaluation=evaluation,
        token_usage=token_usage,
        performance=performance,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert_run(
    logs: list[dict[str, Any]],
    output_dir: Path,
    source_org: str,
) -> bool:
    """Convert a set of deduplicated logs into EEE format."""
    if not logs:
        logger.error("No logs to convert")
        return False

    file_uuid = str(uuid.uuid4())
    model = logs[0]["model"]
    model_id = model_id_from_bb(model)

    if "/" in model_id:
        dev_slug, model_name = model_id.split("/", 1)
    else:
        dev_slug, model_name = "unknown", model_id

    dest_dir = output_dir / "BountyBench" / dev_slug / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build aggregate
    agg = build_aggregate(logs, file_uuid, output_dir, source_org)
    evaluation_id = agg.evaluation_id
    evaluation_name = agg.evaluation_results[0].evaluation_name

    # Build instance-level records
    instance_logs: list[InstanceLevelEvaluationLog] = []
    for log in logs:
        instance = build_instance_level(log, evaluation_id, model_id, evaluation_name)
        instance_logs.append(instance)

    # Write instance-level JSONL
    jsonl_path = dest_dir / f"{file_uuid}_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for inst in instance_logs:
            line = json.dumps(
                inst.model_dump(mode="json", exclude_none=True), ensure_ascii=False
            )
            f.write(line + "\n")
    logger.info("Wrote %d instance-level records to %s", len(instance_logs), jsonl_path)

    # Update aggregate with checksum and row count
    agg.detailed_evaluation_results.checksum = sha256_file(jsonl_path)
    agg.detailed_evaluation_results.total_rows = len(instance_logs)

    # Write aggregate JSON
    agg_path = dest_dir / f"{file_uuid}.json"
    agg_path.write_text(agg.model_dump_json(indent=4, exclude_none=True))
    logger.info("Wrote aggregate to %s", agg_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory containing BountyBench JSON logs (e.g. bountybench/bountybench/logs/2026-03-26)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for EEE files",
    )
    parser.add_argument(
        "--source-org",
        type=str,
        required=True,
        help="Name of the organization that ran the evaluation (used in source_metadata)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("Scanning %s for BountyBench logs...", args.logs_dir)
    all_logs = collect_logs(args.logs_dir)
    logger.info("Found %d raw logs", len(all_logs))

    if not all_logs:
        logger.error("No logs found")
        sys.exit(1)

    # Group by (model, workflow)
    groups: dict[tuple[str, str], list[dict]] = {}
    for log in all_logs:
        key = (log["model"], log["workflow"])
        groups.setdefault(key, []).append(log)

    for (model, workflow), group_logs in sorted(groups.items()):
        logger.info(
            "Processing %s / %s (%d raw logs)", model, workflow, len(group_logs)
        )

        deduped = deduplicate_logs(group_logs)
        logger.info("  After dedup: %d bounties", len(deduped))

        usable = filter_usable(deduped)
        logger.info("  After filtering: %d usable bounties", len(usable))

        n_success = sum(1 for lg in usable if lg["success"])
        n_complete = sum(1 for lg in usable if lg["complete"])
        logger.info("  Success: %d, Complete: %d", n_success, n_complete)

        if args.dry_run:
            for log in usable:
                status = "✅" if log["success"] else ("✓" if log["complete"] else "✗")
                print(
                    f"  {status} {log['bounty_id']}: {log['cve']} (tokens: {log['input_tokens'] + log['output_tokens']:,})"
                )
            continue

        if convert_run(usable, args.output_dir, args.source_org):
            logger.info("  ✅ Conversion complete")
        else:
            logger.error("  ❌ Conversion failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
