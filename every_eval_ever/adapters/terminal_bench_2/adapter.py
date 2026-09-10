"""
Script to convert Terminal-Bench 2.0 leaderboard data to the EvalEval schema format.

Data source:
- Terminal-Bench 2.0 leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.0
  The page renders its table in the browser, so the rows are read from the
  `leaderboard-read` endpoint it calls rather than from the served HTML.

Terminal-Bench is an agentic coding benchmark that evaluates agent+model pairs on
87 terminal-based tasks with 5 trials each. Each leaderboard entry represents a
unique agent+model combination. Agent metadata is stored in model_info.additional_details.

Usage:
    uv run python -m every_eval_ever.adapters.terminal_bench_2.adapter
"""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    ConfidenceInterval,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

LEADERBOARD_URL = 'https://www.tbench.ai/leaderboard/terminal-bench/2.0'
LEADERBOARD_API_URL = (
    'https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read'
)
LEADERBOARD_PACKAGE = 'terminal-bench/terminal-bench-2'
LEADERBOARD_NAME = '2-0'
OUTPUT_DIR = 'data/terminal-bench-2.0'
TASK_COUNT = 87
TRIALS_PER_TASK = 5
CONFIDENCE_LEVEL = 0.95

ORG_SLUG_MAP = {
    'Google': 'google',
    'OpenAI': 'openai',
    'Anthropic': 'anthropic',
    'xAI': 'xai',
    'Moonshot AI': 'moonshot-ai',
    'Z-AI': 'zhipu-ai',
    'Z.ai': 'zhipu-ai',
    'DeepSeek': 'deepseek',
    'Alibaba': 'alibaba',
    'MiniMax': 'minimax',
    'Minimax': 'minimax',
    'Kimi': 'moonshot-ai',
    'Multiple': 'multiple',
    'Block': 'block',
    'Factory': 'factory',
    'Forge Code': 'forge-code',
    'KRAFTON AI': 'krafton-ai',
    'Coder': 'coder',
    'OpenBlock Labs': 'openblock-labs',
    'Bigai': 'bigai',
    'JetBrains': 'jetbrains',
    'Feeling AI': 'feeling-ai',
    'Antigma Labs': 'antigma-labs',
    'Roam': 'roam',
    'LangChain': 'langchain',
    'OpenSage': 'opensage',
    'Terminal Bench': 'terminal-bench',
    'Intelligent Internet': 'intelligent-internet',
    'Warp': 'warp',
    'Letta': 'letta',
    'Abacus.AI': 'abacus-ai',
    'OpenHands': 'openhands',
    'Anomaly Innovations': 'anomaly-innovations',
    'CAMEL-AI': 'camel-ai',
    'ADYA': 'adya',
    'Princeton': 'princeton',
    'TUM': 'tum',
    'iflow': 'iflow',
}


def fetch_leaderboard_payload(
    api_url: str = LEADERBOARD_API_URL,
    package: str = LEADERBOARD_PACKAGE,
    name: str = LEADERBOARD_NAME,
) -> dict:
    """Read the leaderboard rows the Terminal-Bench page renders."""
    body = json.dumps({'package': package, 'name': name}).encode('utf-8')
    request = Request(
        api_url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'EEE-adapter/1.0',
        },
        method='POST',
    )
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
            content_type = response.headers.get('Content-Type')
    except HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:400]
        raise ValueError(
            f'leaderboard-read returned {exc.code} for '
            f'{package}/{name}: {detail}'
        ) from exc
    raw_capture.record(url=api_url, content=raw, content_type=content_type)
    payload = json.loads(raw.decode('utf-8', errors='strict'))
    if not isinstance(payload, dict):
        raise ValueError(
            f'leaderboard-read returned {type(payload).__name__}, not an object'
        )
    return payload


def save_raw_json(payload: dict, path: Path | None) -> None:
    """Persist the exact fetched source outside the validated data tree."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8',
    )


def is_subpath(path: Path, parent: Path) -> bool:
    """Return whether a raw artifact would be placed inside output data."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_entries(path: Path) -> list[dict]:
    """Load a saved normalized leaderboard snapshot for offline replay."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    entries = payload.get('entries') if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError('--input-json must contain a list of entry objects')
    return entries


def _text(value: object) -> str | None:
    """Return a non-blank source string or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _agent_label(metadata: dict) -> str | None:
    """Read the agent's display label, falling back to its bare name."""
    display = metadata.get('agent_display')
    if isinstance(display, dict):
        label = _text(display.get('label'))
        if label is not None:
            return label
    return _text(metadata.get('agent_name'))


def _model_label(metadata: dict) -> str | None:
    """Read the model's display name, falling back to the first model id.

    A handful of rows name several models for one agent run. The display name
    covers those ("Claude Sonnet 4.5 + GPT-5"), so the list is only a fallback.
    """
    display = _text(metadata.get('model_display'))
    if display is not None:
        return display
    names = metadata.get('model_names')
    if isinstance(names, list) and names:
        return _text(names[0])
    return None


def parse_leaderboard_payload(
    payload: dict,
) -> SourceConversionResult[dict]:
    """Normalize leaderboard rows and retain malformed-row provenance."""
    rows = payload.get('rows')
    if not isinstance(rows, list):
        raise ValueError(
            'leaderboard-read payload has no "rows" list; '
            f'got keys {sorted(payload)!r}'
        )

    entries = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    for row_index, row in enumerate(rows):
        row_ref = f'leaderboard row {row_index + 1}'
        if not isinstance(row, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=f'row is {type(row).__name__}, not an object',
                    source_record={'row': row},
                )
            )
            continue
        metadata = row.get('metadata')
        metrics = row.get('metrics')
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason='row is missing its metadata or metrics object',
                    source_record=row,
                )
            )
            continue
        # A row the source is not displaying is withheld upstream rather than
        # malformed, so it is excluded with its reason instead of failing.
        status = _text(row.get('status'))
        if status is not None and status != 'display':
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=row_ref,
                    reason=f'source marks this row {status!r}, not displayed',
                    source_record=row,
                )
            )
            continue
        try:
            rank = int(row['rank'])
        except (KeyError, TypeError, ValueError):
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=f'could not read rank {row.get("rank")!r}',
                    source_record=row,
                )
            )
            continue
        entries.append(
            {
                'rank': rank,
                'agent': _agent_label(metadata),
                'model': _model_label(metadata),
                'date': _text(metadata.get('date')),
                'agent_org': _text(metadata.get('agent_org')),
                'model_org': _text(metadata.get('model_org')),
                'accuracy': metrics.get('accuracy'),
                'ci95_half_width': metrics.get('accuracy_ci95_half_width'),
            }
        )
    if not entries and not failures:
        # Publishing nothing has to be loud. A leaderboard that returned rows
        # but displays none of them reads as success otherwise.
        detail = (
            f'withheld all {len(exclusions)} of its rows'
            if exclusions
            else 'returned no rows'
        )
        failures.append(
            SourceRecordFailure(
                source_ref=LEADERBOARD_API_URL,
                reason=(
                    f'leaderboard {LEADERBOARD_PACKAGE}/{LEADERBOARD_NAME} '
                    f'{detail}'
                ),
            )
        )
    return SourceConversionResult(
        source_name='Terminal-Bench 2.0 leaderboard rows',
        total_records=len(rows),
        records=entries,
        failures=failures,
        exclusions=exclusions,
    )


def get_org_slug(org_name: str) -> str:
    return sanitize_filename(
        ORG_SLUG_MAP.get(
            org_name,
            org_name.lower().replace(' ', '-').replace('.', '-'),
        )
    )


def get_model_slug(model_name: str) -> str:
    return sanitize_filename(model_name.lower().replace(' ', '-'))


def make_model_id(model_org: str, model_name: str) -> str:
    return f'{get_org_slug(model_org)}/{get_model_slug(model_name)}'


def convert_entry(
    entry: dict,
    retrieved_timestamp: str,
    leaderboard_url: str = LEADERBOARD_URL,
) -> EvaluationLog:
    """Convert a single leaderboard entry to an EvaluationLog."""
    agent = require_identity(
        entry.get('agent'),
        f'Terminal-Bench agent for rank {entry.get("rank")!r}',
    )
    model_org = require_identity(
        entry.get('model_org'),
        f'Terminal-Bench developer for rank {entry.get("rank")!r}',
    )
    model_name = require_identity(
        entry.get('model'),
        f'Terminal-Bench model for rank {entry.get("rank")!r}',
    )
    date = require_identity(
        entry.get('date'),
        f'Terminal-Bench date for rank {entry.get("rank")!r}',
    )
    accuracy = float(entry.get('accuracy'))
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 100.0:
        raise ValueError(
            'Terminal-Bench accuracy must be a finite percentage between '
            f'0 and 100, got {entry.get("accuracy")!r}'
        )
    half_width_value = entry.get('ci95_half_width')
    half_width = (
        None if half_width_value is None else float(half_width_value)
    )
    if half_width is not None and (
        not math.isfinite(half_width) or half_width < 0.0
    ):
        raise ValueError(
            'Terminal-Bench confidence half-width must be a finite '
            f'non-negative number, got {half_width_value!r}'
        )
    model_id = make_model_id(model_org, model_name)
    agent_slug = sanitize_filename(agent.lower().replace(' ', '-'))
    model_slug = get_model_slug(model_name)

    eval_id = (
        f'terminal-bench-2.0/{agent_slug}__{model_slug}/{retrieved_timestamp}'
    )

    uncertainty = None
    if half_width is not None:
        # The source publishes `accuracy_ci95_half_width`, a half-width and not
        # a standard error, so it is recorded as the interval it describes.
        uncertainty = Uncertainty(
            confidence_interval=ConfidenceInterval(
                lower=accuracy - half_width,
                upper=accuracy + half_width,
                confidence_level=CONFIDENCE_LEVEL,
                method='reported by the Terminal-Bench leaderboard',
            ),
            num_samples=TASK_COUNT * TRIALS_PER_TASK,
        )

    eval_result = EvaluationResult(
        evaluation_result_id=f'{eval_id}#accuracy',
        evaluation_name='terminal-bench-2.0',
        source_data=SourceDataUrl(
            dataset_name='terminal-bench-2.0',
            source_type='url',
            url=[leaderboard_url],
        ),
        evaluation_timestamp=date,
        metric_config=MetricConfig(
            evaluation_description='Task resolution accuracy across 87 terminal tasks with 5 trials each',
            # Namespaced, not the registry's `accuracy`: this is the share of 87
            # tasks resolved, averaged over 5 trials each, on the leaderboard's
            # own percent scale. The registry carries no Terminal-Bench metric,
            # and joining a trial-averaged resolution rate to plain `accuracy`
            # would merge two different quantities.
            metric_id='terminal-bench-2.0.accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=100,
        ),
        score_details=ScoreDetails(
            score=accuracy,
            uncertainty=uncertainty,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    available_tools=[
                        AvailableTool(
                            name='terminal',
                            description='Full terminal/shell access',
                        ),
                    ],
                ),
                execution_command=(
                    'harbor run -d terminal-bench/terminal-bench-2 '
                    f'-a "{agent}" -m "{model_name}" '
                    f'-k {TRIALS_PER_TASK}'
                ),
            ),
        ),
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=date,
        source_metadata=SourceMetadata(
            source_name='Terminal-Bench 2.0',
            source_type='documentation',
            source_organization_name='Terminal-Bench',
            source_organization_url='https://www.tbench.ai',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name='harbor', version='unknown'),
        model_info=ModelInfo(
            name=model_name,
            id=model_id,
            developer=model_org,
            additional_details={
                'agent_name': agent,
                'agent_organization': require_identity(
                    entry.get('agent_org'),
                    f'Terminal-Bench agent organization for rank '
                    f'{entry.get("rank")!r}',
                ),
            },
        ),
        evaluation_results=[eval_result],
    )


def convert_logs(
    entries: list[dict],
    retrieved_timestamp: str | None = None,
    leaderboard_url: str = LEADERBOARD_URL,
) -> SourceConversionResult[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    bundles = []
    failures: list[SourceRecordFailure] = []
    for index, entry in enumerate(entries):
        try:
            eval_log = convert_entry(entry, timestamp, leaderboard_url)
            org_slug = get_org_slug(entry['model_org'])
            model_slug = get_model_slug(entry['model'])
        except Exception as e:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'leaderboard row {index}',
                    reason=str(e),
                    source_record=entry,
                )
            )
            continue
        bundles.append((eval_log, org_slug, model_slug))
    if not bundles and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref='Terminal-Bench 2.0 input',
                reason='converted 0 source records',
            )
        )
    return SourceConversionResult(
        source_name='Terminal-Bench 2.0',
        total_records=len(entries),
        records=bundles,
        failures=failures,
    )


def make_logs(
    entries: list[dict],
    retrieved_timestamp: str | None = None,
    leaderboard_url: str = LEADERBOARD_URL,
) -> list[tuple[EvaluationLog, str, str]]:
    result = convert_logs(entries, retrieved_timestamp, leaderboard_url)
    result.raise_if_incomplete()
    return result.records


def export(
    bundles: list[tuple[EvaluationLog, str, str]],
    output_dir: str | Path,
) -> list[Path]:
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=log,
            base_dir=output_dir,
            developer=developer,
            model_name=model_name,
        )
        for log, developer, model_name in bundles
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and convert the Terminal-Bench 2.0 leaderboard.',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Replay a saved normalized list of leaderboard entries.',
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help='Save the fetched leaderboard payload outside --output-dir.',
    )
    parser.add_argument(
        '--leaderboard-url',
        default=LEADERBOARD_URL,
        help='Human-facing leaderboard URL recorded as each record\'s source.',
    )
    parser.add_argument(
        '--leaderboard-api-url',
        default=LEADERBOARD_API_URL,
        help='Endpoint the rows are read from (for testing or source moves).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected source rows and reasons here. Defaults beside '
            '--output-dir when any row fails.'
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.save_raw_json is not None and is_subpath(
        args.save_raw_json,
        args.output_dir,
    ):
        raise SystemExit(
            '--save-raw-json must point outside --output-dir so the '
            'validator cannot mistake the source payload for evaluation data'
        )
    if args.input_json is not None:
        entries = load_entries(args.input_json)
        parsed = SourceConversionResult(
            source_name='Terminal-Bench 2.0 input JSON',
            total_records=len(entries),
            records=entries,
            failures=[],
        )
    else:
        payload = fetch_leaderboard_payload(args.leaderboard_api_url)
        save_raw_json(payload, args.save_raw_json)
        parsed = parse_leaderboard_payload(payload)

    converted = convert_logs(
        parsed.records,
        leaderboard_url=args.leaderboard_url,
    )
    result = SourceConversionResult(
        source_name='Terminal-Bench 2.0',
        total_records=parsed.total_records,
        records=converted.records,
        failures=[*parsed.failures, *converted.failures],
        exclusions=parsed.exclusions,
    )
    paths = export(result.records, args.output_dir)
    for path in paths:
        print(path)
    print(f'Generated {len(paths)} files in {args.output_dir}/')
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()


if __name__ == '__main__':
    main()
