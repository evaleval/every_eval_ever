#!/usr/bin/env python3
"""Convert BountyBench run logs into Every Eval Ever records.

BountyBench (https://github.com/bountybench/bountybench) writes one JSON log
per bounty per attempt. This adapter emits one aggregate EvaluationLog per
(model, workflow, configuration) together with a per-bounty instance sidecar.

Run:
    uv run python -m every_eval_ever.adapters.bountybench.adapter \
        --logs-dir bountybench/logs/2026-03-26 \
        --output-dir /tmp/eee-bountybench/data/bountybench \
        --source-org 'Your Organization'

The input layout, the identity rules and every flag are documented in
``every_eval_ever/adapters/README.md`` under "BountyBench".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
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
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    datastore_output_dir,
    datastore_path_components,
    datastore_repo_file_path,
    default_failure_report_path,
    save_failure_report,
)
from every_eval_ever.instance_level_types import (
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Message,
    Performance,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = 'BountyBench'
COLLECTION = 'bountybench'
DEFAULT_OUTPUT_DIR = Path(f'data/{COLLECTION}')
BOUNTYBENCH_GITHUB = 'https://github.com/bountybench/bountybench'
TOOL_OUTPUT_CHAR_LIMIT = 10_000
WORKFLOW_SLUGS = {
    'DetectWorkflow': 'detect',
    'ExploitWorkflow': 'exploit',
    'PatchWorkflow': 'patch',
}


def workflow_slug(workflow: str) -> str:
    """Return the short evaluation name for a BountyBench workflow.

    Only the workflows in ``WORKFLOW_SLUGS`` have a slug. ``parse_bounty_log``
    rejects any other name before it reaches here, so an evaluation identity is
    never derived from an unmapped label (where two distinct names could
    normalize to one slug and collide).
    """
    return WORKFLOW_SLUGS[workflow]


def parse_source_time(value: str, source_tz: tzinfo) -> datetime:
    """Parse one BountyBench timestamp, which carries no UTC offset.

    A naive value is read in ``source_tz`` rather than in the converting host's
    zone, so one log yields one instant — and one evaluation_id — everywhere.
    """
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            parsed = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f'unparseable timestamp {value!r}') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_tz)
    return parsed


def resolve_model_identity(
    model: str, developer_override: str | None
) -> tuple[str, str]:
    """Return (model_id, developer) for a BountyBench model string.

    BountyBench records some agents without a provider prefix. Those cannot be
    routed without being told the developer, and the datastore path rules
    reject publishing them under ``unknown``.
    """
    if '/' in model:
        model_id = model
    elif developer_override:
        model_id = f'{developer_override}/{model}'
    else:
        raise ValueError(
            f'model {model!r} has no developer prefix; pass --model-developer '
            'to say which organization published it'
        )
    # Resolve the datastore route now, so an identity that cannot be routed is
    # one recorded log failure rather than an abort of the whole batch at
    # publication. The returned developer is the routed directory name.
    _collection, developer, _model_name = datastore_path_components(
        COLLECTION, model_id
    )
    return model_id, developer


def _model_from_filename(path: Path) -> str:
    """Recover the model from the log filename when the config omits it.

    The filename carries the model name alone, so an identity recovered this
    way has no developer and is refused until ``--model-developer`` names one.
    """
    return path.stem.split('_')[0]


def _require(value: Any, field_name: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f'missing {field_name}')
    return value


def parse_bounty_log(
    path: Path, source_tz: tzinfo, developer_override: str | None
) -> dict[str, Any]:
    """Parse one BountyBench JSON log into a normalized dict.

    Raises for a log whose identity cannot be established, so the caller
    records it instead of publishing an invented one.
    """
    data = json.loads(path.read_text())

    workflow_metadata = data.get('workflow_metadata') or {}
    summary = workflow_metadata.get('workflow_summary') or {}
    task = workflow_metadata.get('task') or {}
    usage = data.get('workflow_usage') or {}
    resources = data.get('resources_used') or {}
    model_config = (resources.get('model') or {}).get('config') or {}
    bounty_metadata = (data.get('additional_metadata') or {}).get(
        'bounty_metadata'
    ) or {}

    model = model_config.get('model') or _model_from_filename(path)
    _require(model, 'model identity (resources_used.model.config.model)')
    model_id, developer = resolve_model_identity(model, developer_override)
    task_dir = _require(task.get('task_dir'), 'workflow_metadata.task.task_dir')
    bounty_number = _require(
        task.get('bounty_number'), 'workflow_metadata.task.bounty_number'
    )
    workflow = _require(
        workflow_metadata.get('workflow_name'),
        'workflow_metadata.workflow_name',
    )
    if workflow not in WORKFLOW_SLUGS:
        raise ValueError(
            f'unknown workflow {workflow!r}; BountyBench defines '
            f'{", ".join(sorted(WORKFLOW_SLUGS))}. A new workflow needs an '
            'entry in WORKFLOW_SLUGS before its logs convert, so its '
            'evaluation identity is assigned deliberately rather than derived '
            'from the raw name and possibly collided with another.'
        )

    # max_iterations is the configured agent budget; a log that records no
    # phase never ran an iteration, i.e. the harness failed at startup.
    phase_messages = data.get('phase_messages') or []
    max_iterations = 0
    if phase_messages:
        max_iterations = phase_messages[0].get('max_iterations', 0) or 0

    # start_time keys the evaluation's identity, so a log without one is
    # rejected here — one recorded failure — rather than aborting the whole
    # batch later when its group has no timestamp to build an aggregate from.
    start_time = _require(data.get('start_time'), 'start_time')
    end_time = data.get('end_time') or ''
    start_dt = parse_source_time(start_time, source_tz)
    end_dt = parse_source_time(end_time, source_tz) if end_time else None
    duration_ms = None
    if start_dt and end_dt:
        duration_ms = max((end_dt - start_dt).total_seconds() * 1000, 0.0)

    return {
        'path': path,
        'task_dir': task_dir,
        'bounty_number': str(bounty_number),
        'bounty_id': f'{task_dir.replace("bountytasks/", "")}_{bounty_number}',
        'model': model,
        'model_id': model_id,
        'developer': developer,
        'workflow': workflow,
        'success': bool(summary.get('success', False)),
        'complete': bool(summary.get('complete', False)),
        'startup_failure': max_iterations == 0,
        'input_tokens': max(usage.get('total_input_tokens', 0) or 0, 0),
        'output_tokens': max(usage.get('total_output_tokens', 0) or 0, 0),
        'query_time_ms': max(
            usage.get('total_query_time_taken_in_ms', 0) or 0, 0
        ),
        'start_time': start_time,
        'start_epoch': str(start_dt.timestamp()) if start_dt else '',
        'duration_ms': duration_ms,
        'max_iterations': max_iterations,
        'phase_messages': phase_messages,
        'model_config': model_config,
        'bounty_metadata': bounty_metadata,
    }


def collect_logs(
    logs_dir: Path, source_tz: tzinfo, developer_override: str | None
) -> tuple[list[dict[str, Any]], list[SourceRecordFailure], int]:
    """Parse every JSON log under ``logs_dir``, collecting unusable ones."""
    parsed: list[dict[str, Any]] = []
    failures: list[SourceRecordFailure] = []
    candidates = sorted(logs_dir.rglob('*.json'))
    for path in candidates:
        try:
            parsed.append(parse_bounty_log(path, source_tz, developer_override))
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=str(path), reason=f'{type(exc).__name__}: {exc}'
                )
            )
    return parsed, failures, len(candidates)


def config_fingerprint(log: dict[str, Any]) -> str:
    """Identify the configuration one log was produced under.

    BountyBench records no run id, so the configuration is the finest run
    boundary the source supports, and one aggregate reporting one set of
    generation settings must not average over two of them. The agent's
    iteration budget is one of those settings — a run allowed 40 steps and a
    run allowed 10 did not evaluate the model under the same conditions — so it
    joins the model config in the fingerprint rather than being averaged over.
    """
    payload = json.dumps(
        {
            'model_config': log['model_config'],
            'max_iterations': log['max_iterations'],
        },
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def group_logs(
    logs: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Partition parsed logs by (model, workflow, configuration)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for log in logs:
        key = (log['model_id'], log['workflow'], config_fingerprint(log))
        groups[key].append(log)
    return dict(groups)


def select_attempts(
    logs: list[dict[str, Any]], policy: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reduce a partition to one log per bounty, returning (kept, attempts)."""
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log in logs:
        attempts[log['bounty_id']].append(log)

    repeated = sorted(key for key, group in attempts.items() if len(group) > 1)
    if repeated and policy == 'reject':
        raise SystemExit(
            f'{len(repeated)} bount(ies) have more than one attempt under a '
            f'single configuration, e.g. {repeated[0]} '
            f'({len(attempts[repeated[0]])} attempts). BountyBench records no '
            'run id, so these cannot be attributed to separate runs, and the '
            'attempts disagree on the score. Pass --attempt-policy best or '
            'latest to collapse them; the aggregate then reports the choice '
            'in attempt_selection.'
        )

    def rank(log: dict[str, Any]) -> tuple:
        epoch = float(log['start_epoch']) if log['start_epoch'] else 0.0
        if policy == 'latest':
            # The latest attempt that produced a scorable run: prefer a
            # non-startup log, then the one that started last.
            return (not log['startup_failure'], epoch)
        # best: success, then completion, then a scored attempt, then the
        # earliest start, as documented — so a tie resolves to the first run,
        # not the last.
        return (
            log['success'],
            log['complete'],
            not log['startup_failure'],
            -epoch,
        )

    kept = [max(group, key=rank) for group in attempts.values()]
    kept.sort(key=lambda log: log['bounty_id'])
    return kept, {key: len(group) for key, group in attempts.items()}


def startup_exclusions(
    logs: list[dict[str, Any]],
) -> list[SourceRecordExclusion]:
    """Record every startup-failure log, which carries no attempt to score."""
    return [
        SourceRecordExclusion(
            source_ref=str(log['path']),
            reason='startup failure: no agent iteration ran, so the log holds '
            'no transcript and no attempted solution',
        )
        for log in logs
        if log['startup_failure']
    ]


def build_aggregate(
    logs: list[dict[str, Any]],
    attempts: dict[str, int],
    fingerprint: str,
    args: argparse.Namespace,
    retrieved_unix: str,
) -> EvaluationLog:
    """Build the aggregate EvaluationLog for one configuration partition."""
    first = logs[0]
    model_id = first['model_id']
    workflow = first['workflow']
    slug = workflow_slug(workflow)

    start_epochs = sorted(
        log['start_epoch'] for log in logs if log['start_epoch']
    )
    if not start_epochs:
        raise ValueError(
            f'{model_id} / {workflow}: no parseable start_time in any log, so '
            'the evaluation has no timestamp to key its identity on'
        )
    eval_unix = start_epochs[0]

    n_success = sum(1 for log in logs if log['success'])
    n_total = len(logs)
    success_rate = n_success / n_total
    standard_error = math.sqrt(success_rate * (1 - success_rate) / n_total)

    # max_iterations is part of the fingerprint, so every log in this partition
    # shares one iteration budget.
    iteration_limit = first['max_iterations']
    metric_details = {
        'config_fingerprint': fingerprint,
        'attempt_selection': args.attempt_policy,
        'n_attempts_total': str(sum(attempts.values())),
        'n_bounties_with_multiple_attempts': str(
            sum(1 for count in attempts.values() if count > 1)
        ),
    }
    score_breakdown = {
        'successes': str(n_success),
        'total': str(n_total),
        'n_bounties_completed': str(sum(1 for log in logs if log['complete'])),
    }

    model_config = first['model_config']
    max_tokens = model_config.get('max_output_tokens')
    generation_details = {}
    if model_config.get('max_input_tokens'):
        generation_details['max_input_tokens'] = str(
            model_config['max_input_tokens']
        )
    if 'helm' in model_config:
        generation_details['helm'] = str(model_config['helm'])
    generation_config = GenerationConfig(
        generation_args=GenerationArgs(
            temperature=model_config.get('temperature'),
            max_tokens=max_tokens if max_tokens else None,
            max_attempts=max(attempts.values()),
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(
                        name='bash', description='Kali Linux terminal'
                    ),
                ]
            ),
            eval_limits=EvalLimits(message_limit=iteration_limit),
            sandbox=Sandbox(type='docker'),
        ),
        additional_details=generation_details or None,
    )

    eval_result = EvaluationResult(
        evaluation_result_id=slug,
        evaluation_name=f'{COLLECTION}.{slug}',
        source_data=SourceDataUrl(
            dataset_name=SOURCE_NAME,
            source_type='url',
            url=[BOUNTYBENCH_GITHUB],
        ),
        evaluation_timestamp=eval_unix,
        metric_config=MetricConfig(
            evaluation_description=(
                f'BountyBench {workflow}: share of bounties the agent '
                'resolved, as judged by the benchmark verifier'
            ),
            metric_id='accuracy',
            metric_name='Success Rate',
            metric_kind='accuracy',
            metric_unit='proportion',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
            additional_details=metric_details,
        ),
        score_details=ScoreDetails(
            score=success_rate,
            details=score_breakdown,
            uncertainty=Uncertainty(
                standard_error=StandardError(
                    value=standard_error, method='analytic'
                ),
                num_samples=n_total,
            ),
        ),
        generation_config=generation_config,
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'{SOURCE_NAME}/{model_id.replace("/", "_")}/{slug}/'
            f'{fingerprint}/{eval_unix}'
        ),
        evaluation_timestamp=eval_unix,
        retrieved_timestamp=retrieved_unix,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type=SourceType.evaluation_run,
            source_organization_name=args.source_org,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'bountybench_repo': BOUNTYBENCH_GITHUB,
                'source_timezone': str(args.source_timezone),
                'workflow': workflow,
            },
        ),
        eval_library=EvalLibrary(
            name=COLLECTION, version=args.bountybench_version
        ),
        model_info=ModelInfo(
            name=first['model'],
            id=model_id,
            developer=first['developer'],
            additional_details={
                'deployment_type': args.deployment_type,
                'model_availability': args.model_availability,
            },
        ),
        evaluation_results=[eval_result],
    )


def _truncate(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_CHAR_LIMIT:
        return text
    dropped = len(text) - TOOL_OUTPUT_CHAR_LIMIT
    return f'{text[:TOOL_OUTPUT_CHAR_LIMIT]}\n[truncated {dropped} characters]'


def _claim_open_call(
    open_calls: list[tuple[str, str]], command: str
) -> str | None:
    """Pop the call one result belongs to, matching command then arrival order.

    The executor result carries the command it ran, so a result is paired to
    the earliest still-open call for that exact command; a result with no
    command (or no match) falls back to the oldest open call. Either way one
    call is claimed at most once, so two open calls never share a result.
    """
    for index, (open_command, call_id) in enumerate(open_calls):
        if command and open_command == command:
            open_calls.pop(index)
            return call_id
    if open_calls:
        return open_calls.pop(0)[1]
    return None


def build_messages_from_phases(phase_messages: list[dict]) -> list[Message]:
    """Flatten BountyBench phases into one ordered EEE transcript.

    BountyBench records a command twice: once as the model's chosen action, and
    again as the executing resource's action carrying its output. Each
    execution must therefore become exactly one tool call plus, where the log
    has output for it, one tool result naming that call's id. Several calls can
    be open at once, so open calls are held in a FIFO and each result claims the
    matching one — consecutive model commands no longer overwrite one another.
    """
    messages: list[Message] = []
    turn_idx = 0
    call_seq = 0
    open_calls: list[tuple[str, str]] = []

    def add(**fields: Any) -> None:
        nonlocal turn_idx
        messages.append(Message(turn_idx=turn_idx, **fields))
        turn_idx += 1

    for phase in phase_messages:
        for agent_message in phase.get('agent_messages') or []:
            text = agent_message.get('message') or ''
            if agent_message.get('agent_id') == 'system':
                add(role='system', content=text)
            elif text:
                add(role='assistant', content=text)

            for action in agent_message.get('action_messages') or []:
                resource = action.get('resource_id') or 'unknown'
                meta = action.get('additional_metadata') or {}
                command = action.get('command') or meta.get('command') or ''
                result = action.get('message') or ''
                # An executor echoing a still-open command is the same
                # execution, not a second one.
                echoes = (
                    resource != 'model'
                    and command
                    and any(cmd == command for cmd, _ in open_calls)
                )
                if command and not echoes:
                    call_id = f'call_{call_seq}'
                    call_seq += 1
                    open_calls.append((command, call_id))
                    add(
                        role='assistant',
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                name='bash'
                                if resource == 'model'
                                else resource,
                                arguments={'command': command},
                            )
                        ],
                    )
                if result and resource != 'model':
                    claimed = _claim_open_call(open_calls, command)
                    add(
                        role='tool',
                        content=_truncate(result),
                        tool_call_id=[claimed] if claimed else None,
                    )
    return messages


def _sample_hash(raw: str, reference: list[str]) -> str:
    """Hash a sample's input, canonically across adapters."""
    payload = json.dumps(
        {'raw': raw, 'reference': reference},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_instance_level(
    log: dict[str, Any], aggregate: EvaluationLog, n_attempts: int
) -> InstanceLevelEvaluationLog:
    """Build the instance-level record for one bounty."""
    eval_result = aggregate.evaluation_results[0]
    task_name = log['task_dir'].replace('bountytasks/', '')

    system_message = ''
    first_phase = (log['phase_messages'][:1] or [{}])[0]
    for agent_message in first_phase.get('agent_messages') or []:
        if agent_message.get('agent_id') == 'system':
            system_message = agent_message.get('message') or ''
            break
    raw_input = system_message or (
        f'BountyBench {task_name} bounty {log["bounty_number"]}'
    )

    messages = build_messages_from_phases(log['phase_messages']) or [
        Message(turn_idx=0, role='system', content=raw_input)
    ]

    metadata = {
        'task_dir': log['task_dir'],
        'bounty_number': log['bounty_number'],
        'workflow': log['workflow'],
        'complete': str(log['complete']),
        'n_attempts': str(n_attempts),
        'attempt_start_time': log['start_time'],
    }
    for source_key, target_key in (
        ('CVE', 'cve'),
        ('CWE', 'cwe'),
        ('severity', 'severity'),
        ('bounty_link', 'bounty_link'),
    ):
        value = log['bounty_metadata'].get(source_key)
        if value not in (None, ''):
            metadata[target_key] = str(value)

    performance = None
    if log['duration_ms'] is not None:
        performance = Performance(
            latency_ms=log['duration_ms'],
            generation_time_ms=log['query_time_ms'] or None,
        )

    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=aggregate.evaluation_id,
        model_id=aggregate.model_info.id,
        evaluation_name=eval_result.evaluation_name,
        evaluation_result_id=eval_result.evaluation_result_id,
        sample_id=log['bounty_id'],
        # Nothing about the outcome may enter this hash: BountyBench scores by
        # running the exploit or patch, so the only available "reference" would
        # be the observed result, which differs per model for the same bounty.
        sample_hash=_sample_hash(raw_input, []),
        interaction_type=InteractionType.agentic,
        input=Input(raw=raw_input, reference=[]),
        messages=messages,
        # The verdict comes from BountyBench's verifier rather than from
        # parsing the transcript, so no turn can be cited for it.
        answer_attribution=[],
        evaluation=Evaluation(
            score=1.0 if log['success'] else 0.0,
            is_correct=log['success'],
            num_turns=len(messages),
            tool_calls_count=sum(
                len(message.tool_calls)
                for message in messages
                if message.tool_calls
            ),
        ),
        token_usage=TokenUsage(
            input_tokens=log['input_tokens'],
            output_tokens=log['output_tokens'],
            total_tokens=log['input_tokens'] + log['output_tokens'],
        ),
        performance=performance,
        metadata=metadata,
    )


def stage_instances(
    logs: list[dict[str, Any]],
    aggregate: EvaluationLog,
    attempts: dict[str, int],
    staged_path: Path,
) -> tuple[str, int]:
    """Write one sidecar into the staging tree, returning (sha256, rows)."""
    digest = hashlib.sha256()
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    with staged_path.open('wb') as handle:
        for log in logs:
            instance = build_instance_level(
                log, aggregate, attempts.get(log['bounty_id'], 1)
            )
            payload = (
                json.dumps(
                    instance.model_dump(mode='json', exclude_none=True),
                    ensure_ascii=False,
                )
                + '\n'
            ).encode('utf-8')
            handle.write(payload)
            digest.update(payload)
    return digest.hexdigest(), len(logs)


def convert(
    args: argparse.Namespace,
) -> tuple[
    list[EvaluationLog],
    list[list[dict[str, Any]]],
    list[dict[str, int]],
    SourceConversionResult,
]:
    """Parse and group a BountyBench log tree without writing anything."""
    parsed, failures, n_candidates = collect_logs(
        args.logs_dir, args.source_timezone, args.model_developer
    )
    logger.info('parsed %d of %d candidate logs', len(parsed), n_candidates)
    if not n_candidates:
        raise SystemExit(f'no JSON logs found under {args.logs_dir}')
    if failures and not args.allow_partial:
        raise SystemExit(
            f'{len(failures)} of {n_candidates} log(s) could not be parsed, '
            f'e.g. {failures[0].source_ref}: {failures[0].reason}. Converting '
            'anyway would report a success rate over a silently reduced '
            'denominator, so nothing has been written. Pass --allow-partial '
            'to convert the rest, which records the omissions and exits '
            'non-zero.'
        )

    logs: list[EvaluationLog] = []
    kept_logs: list[list[dict[str, Any]]] = []
    attempt_counts: list[dict[str, int]] = []
    exclusions: list[SourceRecordExclusion] = []
    retrieved_unix = args.retrieved_timestamp or str(
        datetime.now(timezone.utc).timestamp()
    )

    # Every group is examined before anything is published, so a group with
    # nothing to convert cannot decide whether the later ones are written.
    for key, group in sorted(group_logs(parsed).items()):
        model_id, workflow, fingerprint = key
        # Record startup failures from the whole group, before per-bounty
        # selection can drop an attempt whose bounty also has a usable one, and
        # exactly once — no synthetic group-level exclusion on top.
        exclusions.extend(startup_exclusions(group))
        kept, attempts = select_attempts(group, args.attempt_policy)
        usable = [log for log in kept if not log['startup_failure']]
        logger.info(
            '%s / %s / %s: %d log(s) -> %d bount(ies), %d usable',
            model_id,
            workflow,
            fingerprint,
            len(group),
            len(kept),
            len(usable),
        )
        if not usable:
            continue
        logs.append(
            build_aggregate(usable, attempts, fingerprint, args, retrieved_unix)
        )
        kept_logs.append(usable)
        attempt_counts.append(attempts)

    result = SourceConversionResult(
        source_name=SOURCE_NAME,
        total_records=n_candidates,
        records=logs,
        failures=failures,
        exclusions=exclusions,
    )
    return logs, kept_logs, attempt_counts, result


def run(args: argparse.Namespace) -> int:
    """Convert one BountyBench log tree and publish it."""
    # Publication forces the collection to COLLECTION, so a requested directory
    # whose final component is anything else would be silently redirected. Fail
    # instead, before any work, so the written location matches the flag. A
    # dry run writes nothing, so it need not care where output would land.
    if not args.dry_run and args.output_dir.name != COLLECTION:
        raise SystemExit(
            f'--output-dir must end in {COLLECTION!r}, the datastore '
            f'collection this adapter writes; got {args.output_dir}'
        )
    logger.info('scanning %s for BountyBench logs', args.logs_dir)
    logs, kept_logs, attempt_counts, result = convert(args)

    if args.dry_run:
        for log, usable in zip(logs, kept_logs):
            eval_result = log.evaluation_results[0]
            print(
                f'{log.model_info.id} {eval_result.evaluation_name}: '
                f'{eval_result.score_details.score:.3f} over '
                f'{len(usable)} bount(ies)'
            )
        print(
            f'{len(result.failures)} unparsed log(s), '
            f'{len(result.exclusions)} excluded record(s)'
        )
        return len(logs)

    if not logs:
        raise SystemExit(
            f'no convertible logs under {args.logs_dir}: '
            f'{len(result.exclusions)} record(s) were excluded and '
            f'{len(result.failures)} could not be parsed'
        )

    file_uuids = [str(uuid.uuid4()) for _ in logs]
    with tempfile.TemporaryDirectory(prefix='eee-bountybench-') as staging:
        staging_root = Path(staging)
        for log, file_uuid, usable, attempts in zip(
            logs, file_uuids, kept_logs, attempt_counts
        ):
            filename = f'{file_uuid}_samples.jsonl'
            checksum, rows = stage_instances(
                usable,
                log,
                attempts,
                datastore_output_dir(
                    staging_root,
                    COLLECTION,
                    log.model_info.id,
                    log.model_info.developer,
                )
                / filename,
            )
            log.detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl,
                # The repository-relative path the datastore resolves, not a
                # basename; the publisher and the merge gate both check it.
                file_path=datastore_repo_file_path(
                    COLLECTION,
                    log.model_info.id,
                    log.model_info.developer,
                    filename,
                ),
                hash_algorithm=HashAlgorithm.sha256,
                checksum=checksum,
                total_rows=rows,
            )

        # Written before publication so a publication failure cannot take the
        # record of what was omitted with it.
        if result.failures or result.exclusions:
            logger.info(
                'partial conversion report: %s',
                save_failure_report(
                    result, default_failure_report_path(args.output_dir)
                ),
            )

        published = publish_evaluation_logs(
            logs,
            args.output_dir.parent,
            file_uuids,
            staged_output_dir=staging_root,
            collection_override=COLLECTION,
        )

    for path in published:
        print(path)
    result.raise_if_incomplete()
    return len(published)


def _timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'unknown timezone {name!r}: {exc}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--logs-dir',
        type=Path,
        required=True,
        help='directory tree of BountyBench JSON logs',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'datastore collection directory (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--source-org',
        required=True,
        help='organization that ran the evaluation',
    )
    parser.add_argument(
        '--attempt-policy',
        choices=('reject', 'best', 'latest'),
        default='reject',
        help='what to do when one bounty has several attempts '
        'under one configuration (default: reject)',
    )
    parser.add_argument(
        '--source-timezone',
        type=_timezone,
        default=ZoneInfo('UTC'),
        help='timezone of the naive BountyBench timestamps (default: UTC)',
    )
    parser.add_argument(
        '--model-developer',
        help='developer for models BountyBench records without '
        'a provider prefix, e.g. claude-code',
    )
    parser.add_argument(
        '--retrieved-timestamp', help='override the record-creation timestamp'
    )
    parser.add_argument(
        '--bountybench-version',
        default='unknown',
        help="BountyBench version the logs came from (default: 'unknown')",
    )
    parser.add_argument(
        '--deployment-type',
        choices=('self_deployed', 'externally_managed', 'unknown'),
        default='externally_managed',
        help='who served the model (default: '
        'externally_managed, the hosted APIs BountyBench '
        'drives)',
    )
    parser.add_argument(
        '--model-availability',
        choices=('open_weights', 'closed_weights', 'unknown'),
        default='unknown',
        help='whether the evaluated weights are available',
    )
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='convert the parseable logs when some fail, '
        'recording the omissions and exiting non-zero',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='report each partition without writing files',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    written = main()
    print(f'Wrote {written} BountyBench evaluation log(s).')
