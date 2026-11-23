"""
Convert the Terminal-Bench leaderboards to the EvalEval schema format.

Data source:
- Terminal-Bench leaderboards: https://www.tbench.ai (one per benchmark version)
- JSON behind them: the Harbor ``leaderboard-read`` function the page POSTs
  ``{"package": ..., "name": ...}`` to. The package and leaderboard name of
  each version are the ones the site's own version picker uses.

Terminal-Bench evaluates agent+model pairs on terminal tasks, each attempted
several times; the published accuracy is the share of trials the task's
verifier accepted. Each version is its own collection,
``data/terminal-bench-<version>/``. Agent metadata is stored in
``model_info.additional_details``.

Usage:
    uv run python -m every_eval_ever.adapters.terminal_bench.adapter
"""

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
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
    StandardError,
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

#: The endpoint tbench.ai's own leaderboard page reads.
LEADERBOARD_API_URL = (
    'https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read'
)
OUTPUT_DIR = 'data'


@dataclass(frozen=True)
class BenchmarkVersion:
    """One Terminal-Bench leaderboard, as the site's version picker names it."""

    version: str
    package: str
    leaderboard: str
    #: The dataset reference ``harbor run -d`` takes for this version.
    run_dataset: str
    #: Set only where the benchmark documents them; the leaderboard publishes
    #: the trial count per row, not the task/trial split.
    task_count: int | None = None
    trials_per_task: int | None = None

    @property
    def collection(self) -> str:
        return f'terminal-bench-{self.version}'

    @property
    def leaderboard_url(self) -> str:
        return (
            f'https://www.tbench.ai/leaderboard/terminal-bench/{self.version}'
        )


VERSIONS = (
    BenchmarkVersion(
        '2.0',
        package='terminal-bench/terminal-bench-2',
        leaderboard='2-0',
        run_dataset='terminal-bench/terminal-bench-2',
        task_count=87,
        trials_per_task=5,
    ),
    BenchmarkVersion(
        '2.1',
        package='terminal-bench/terminal-bench-2-1',
        leaderboard='main',
        run_dataset='terminal-bench/terminal-bench-2-1',
    ),
    BenchmarkVersion(
        '3.0',
        package='terminal-bench/terminal-bench',
        leaderboard='3-0-0',
        run_dataset='terminal-bench/terminal-bench@3.0.0',
    ),
    BenchmarkVersion(
        '4.0',
        package='terminal-bench/terminal-bench',
        leaderboard='4-0-0',
        run_dataset='terminal-bench/terminal-bench@4.0.0',
    ),
)
VERSIONS_BY_KEY = {spec.version: spec for spec in VERSIONS}
TB2 = VERSIONS_BY_KEY['2.0']

#: ``pass_at_<k>`` metrics, published by some versions on a 0-1 scale.
PASS_AT_KEY = re.compile(r'^pass_at_(\d+)$')

#: Row metrics kept verbatim in ``score_details.details`` when published.
DETAIL_METRICS = (
    'n_trials',
    'successes',
    'total_cost_usd',
    'total_tokens',
    'uncached_input_tokens',
    'cached_input_tokens',
    'output_tokens',
    'avg_trial_duration_sec',
    'reward_hacks',
)

ORG_SLUG_MAP = {
    'Google': 'google',
    'OpenAI': 'openai',
    'Anthropic': 'anthropic',
    'xAI': 'xai',
    'Moonshot AI': 'moonshot-ai',
    'Z-AI': 'zhipu-ai',
    'Z.ai': 'zhipu-ai',
    'Z.AI': 'zhipu-ai',
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
    spec: BenchmarkVersion = TB2,
    api_url: str = LEADERBOARD_API_URL,
) -> dict:
    """POST the leaderboard query the tbench.ai page makes; return its JSON."""
    request = Request(
        api_url,
        data=json.dumps(
            {'package': spec.package, 'name': spec.leaderboard}
        ).encode('utf-8'),
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'EEE-adapter/1.0',
        },
    )
    with urlopen(request, timeout=60) as response:
        body = response.read()
        content_type = response.headers.get('Content-Type')
    raw_capture.record(
        url=api_url,
        content=body,
        content_type=content_type,
        label=f'{spec.package}:{spec.leaderboard}',
    )
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError(
            f'leaderboard endpoint returned {type(payload).__name__}, '
            'expected an object'
        )
    return payload


def save_raw_payload(payload: dict, path: Path | None) -> None:
    """Persist the fetched source outside the validated data tree."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
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


def _label(value) -> str | None:
    """A display field is either a plain string or ``{"label", "url"}``."""
    if isinstance(value, dict):
        value = value.get('label')
    return value if isinstance(value, str) else None


def parse_leaderboard_payload(payload: dict) -> SourceConversionResult[dict]:
    """Turn published leaderboard rows into normalized entries.

    Rows the leaderboard is not displaying are exclusions; a row without a
    metadata or metrics object is a failure carrying the source row.
    """
    rows = payload.get('rows')
    if not isinstance(rows, list):
        raise ValueError('leaderboard payload has no rows list')
    entries = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    for index, row in enumerate(rows):
        row_ref = f'leaderboard row {row.get("id", index)}'
        if row.get('status') != 'display':
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=row_ref,
                    reason=f'not published (status {row.get("status")!r})',
                    source_record=row,
                )
            )
            continue
        metadata = row.get('metadata')
        metrics = row.get('metrics')
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason='row has no metadata or metrics object',
                    source_record=row,
                )
            )
            continue
        # 2.0 publishes 0 trials for every row: not a count, just unset.
        n_trials = row.get('n_trials') or metrics.get('n_trials') or None
        details = {
            key: str(metrics[key])
            for key in DETAIL_METRICS
            if metrics.get(key) is not None
        }
        if n_trials is not None:
            details['n_trials'] = str(n_trials)
        entries.append(
            {
                'id': row.get('id'),
                'rank': row.get('rank'),
                'agent': _label(metadata.get('agent_display')),
                'model': _label(metadata.get('model_display')),
                'date': metadata.get('date'),
                'agent_org': _label(metadata.get('agent_org')),
                'model_org': _label(metadata.get('model_org')),
                'reasoning_effort': metadata.get('reasoning_effort'),
                'accuracy': metrics.get('accuracy'),
                'stderr': metrics.get('accuracy_stderr'),
                'ci95_half_width': metrics.get('accuracy_ci95_half_width'),
                'n_trials': n_trials,
                'pass_at': {
                    int(match.group(1)): value
                    for key, value in metrics.items()
                    if (match := PASS_AT_KEY.match(key)) and value is not None
                },
                'details': details,
            }
        )
    return SourceConversionResult(
        source_name='Terminal-Bench leaderboard',
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


def _non_negative(entry: dict, key: str, label: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f'Terminal-Bench {label} must be a finite non-negative number, '
            f'got {value!r}'
        )
    return number


def convert_entry(
    entry: dict,
    retrieved_timestamp: str,
    leaderboard_url: str | None = None,
    spec: BenchmarkVersion = TB2,
) -> EvaluationLog:
    """Convert a single leaderboard entry to an EvaluationLog."""
    leaderboard_url = leaderboard_url or spec.leaderboard_url
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
    stderr = _non_negative(entry, 'stderr', 'standard error')
    half_width = _non_negative(entry, 'ci95_half_width', '95% CI half-width')
    model_id = make_model_id(model_org, model_name)
    agent_slug = sanitize_filename(agent.lower().replace(' ', '-'))
    model_slug = get_model_slug(model_name)

    # The leaderboard row id is unique and stable; agent + model is neither
    # (one model runs under several reasoning efforts).
    row_id = entry.get('id')
    eval_id = (
        f'{spec.collection}/{row_id}'
        if row_id
        else f'{spec.collection}/{agent_slug}__{model_slug}/{retrieved_timestamp}'
    )

    num_samples = entry.get('n_trials')
    if num_samples is None and spec.task_count and spec.trials_per_task:
        num_samples = spec.task_count * spec.trials_per_task
    uncertainty = None
    if stderr is not None or half_width is not None:
        uncertainty = Uncertainty(
            standard_error=(
                None if stderr is None else StandardError(value=stderr)
            ),
            # The source publishes the half-width; the bounds are exactly
            # accuracy -/+ it (rounded off float noise), not clipped to the
            # score range.
            confidence_interval=(
                None
                if half_width is None
                else ConfidenceInterval(
                    lower=round(accuracy - half_width, 10),
                    upper=round(accuracy + half_width, 10),
                    confidence_level=0.95,
                )
            ),
            num_samples=None if num_samples is None else int(num_samples),
        )

    if spec.task_count and spec.trials_per_task:
        description = (
            f'Task resolution accuracy across {spec.task_count} terminal '
            f'tasks with {spec.trials_per_task} trials each'
        )
    else:
        description = (
            'Share of trials whose task verifier accepted the result on '
            f'Terminal-Bench {spec.version}'
        )
    execution_command = (
        f'harbor run -d {spec.run_dataset} -a "{agent}" -m "{model_name}"'
    )
    if spec.trials_per_task:
        execution_command += f' -k {spec.trials_per_task}'

    eval_result = EvaluationResult(
        evaluation_result_id=f'{eval_id}#accuracy',
        evaluation_name=spec.collection,
        source_data=SourceDataUrl(
            dataset_name=spec.collection,
            source_type='url',
            url=[leaderboard_url],
        ),
        evaluation_timestamp=date,
        metric_config=MetricConfig(
            evaluation_description=description,
            # Namespaced, not the registry's `accuracy`: this is a share of
            # verifier-accepted trials, on the leaderboard's own percent
            # scale, and each version is a different task set.
            metric_id=f'{spec.collection}.accuracy',
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
            details=entry.get('details') or None,
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
                execution_command=execution_command,
            ),
        ),
    )

    pass_results = [
        _pass_at_result(spec, eval_id, date, leaderboard_url, k, value)
        for k, value in sorted((entry.get('pass_at') or {}).items())
    ]

    additional_details = {
        'agent_name': agent,
        'agent_organization': require_identity(
            entry.get('agent_org'),
            f'Terminal-Bench agent organization for rank {entry.get("rank")!r}',
        ),
    }
    if entry.get('reasoning_effort'):
        additional_details['reasoning_effort'] = str(entry['reasoning_effort'])

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=date,
        source_metadata=SourceMetadata(
            source_name=f'Terminal-Bench {spec.version}',
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
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result, *pass_results],
    )


def _pass_at_result(
    spec: BenchmarkVersion,
    eval_id: str,
    date: str,
    leaderboard_url: str,
    k: int,
    value,
) -> EvaluationResult:
    """One published pass@k, on the leaderboard's own 0-1 scale."""
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f'Terminal-Bench pass@{k} must be a finite proportion between '
            f'0 and 1, got {value!r}'
        )
    return EvaluationResult(
        evaluation_result_id=f'{eval_id}#pass_at_{k}',
        evaluation_name=spec.collection,
        source_data=SourceDataUrl(
            dataset_name=spec.collection,
            source_type='url',
            url=[leaderboard_url],
        ),
        evaluation_timestamp=date,
        metric_config=MetricConfig(
            evaluation_description=(
                f'Share of tasks solved in at least one of {k} trials on '
                f'Terminal-Bench {spec.version}'
            ),
            metric_id='pass_at_k',
            metric_name=f'Pass@{k}',
            metric_kind='pass_rate',
            metric_unit='proportion',
            metric_parameters={'k': k},
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=1,
        ),
        score_details=ScoreDetails(score=score),
    )


def convert_logs(
    entries: list[dict],
    retrieved_timestamp: str | None = None,
    leaderboard_url: str | None = None,
    spec: BenchmarkVersion = TB2,
) -> SourceConversionResult[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    bundles = []
    failures: list[SourceRecordFailure] = []
    for index, entry in enumerate(entries):
        try:
            eval_log = convert_entry(entry, timestamp, leaderboard_url, spec)
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
                source_ref=f'Terminal-Bench {spec.version} input',
                reason='converted 0 source records',
            )
        )
    return SourceConversionResult(
        source_name=f'Terminal-Bench {spec.version}',
        total_records=len(entries),
        records=bundles,
        failures=failures,
    )


def make_logs(
    entries: list[dict],
    retrieved_timestamp: str | None = None,
    leaderboard_url: str | None = None,
    spec: BenchmarkVersion = TB2,
) -> list[tuple[EvaluationLog, str, str]]:
    result = convert_logs(entries, retrieved_timestamp, leaderboard_url, spec)
    result.raise_if_incomplete()
    return result.records


def export(
    bundles: list[tuple[EvaluationLog, str, str]],
    output_dir: str | Path,
) -> list[Path]:
    """Write bundles under ``output_dir``, the collection directory."""
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
        description='Fetch and convert the Terminal-Bench leaderboards.',
    )
    parser.add_argument(
        '--version',
        choices=[*VERSIONS_BY_KEY, 'all'],
        default='all',
        help='Which leaderboard version to convert (default: all).',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help=(
            'Replay a saved normalized list of leaderboard entries for one '
            '--version instead of fetching.'
        ),
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help=(
            'Save each fetched leaderboard as <version>.json in this '
            'directory, outside --output-dir.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=(
            'Data root; each version is written to '
            f'<output-dir>/terminal-bench-<version>/ (default: {OUTPUT_DIR}).'
        ),
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected source rows and reasons here (one --version '
            'only). Defaults beside --output-dir when any row fails.'
        ),
    )
    return parser.parse_args(argv)


def convert_version(
    spec: BenchmarkVersion, args: argparse.Namespace
) -> SourceConversionResult:
    """Fetch (or replay), convert and write one version's leaderboard."""
    if args.input_json is not None:
        entries = load_entries(args.input_json)
        parsed = SourceConversionResult(
            source_name=f'Terminal-Bench {spec.version} input JSON',
            total_records=len(entries),
            records=entries,
            failures=[],
        )
    else:
        try:
            payload = fetch_leaderboard_payload(spec)
            parsed = parse_leaderboard_payload(payload)
        except (OSError, ValueError) as exc:
            parsed = SourceConversionResult(
                source_name=f'Terminal-Bench {spec.version}',
                total_records=1,
                records=[],
                failures=[
                    SourceRecordFailure(
                        source_ref=f'{spec.package}:{spec.leaderboard}',
                        reason=f'could not read the leaderboard: {exc}',
                    )
                ],
            )
        else:
            if args.save_raw_json is not None:
                save_raw_payload(
                    payload, args.save_raw_json / f'{spec.version}.json'
                )

    converted = convert_logs(parsed.records, spec=spec)
    result = SourceConversionResult(
        source_name=f'Terminal-Bench {spec.version}',
        total_records=parsed.total_records,
        records=converted.records,
        failures=[*parsed.failures, *converted.failures],
        exclusions=parsed.exclusions,
    )
    collection_dir = args.output_dir / spec.collection
    paths = export(result.records, collection_dir)
    print(
        f'Terminal-Bench {spec.version}: {len(paths)} files in {collection_dir}/'
    )
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(collection_dir),
        )
        print(f'Failure report: {report_path}')
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.save_raw_json is not None and is_subpath(
        args.save_raw_json,
        args.output_dir,
    ):
        raise SystemExit(
            '--save-raw-json must point outside --output-dir so the '
            'validator cannot mistake source JSON for evaluation data'
        )
    specs = (
        VERSIONS if args.version == 'all' else (VERSIONS_BY_KEY[args.version],)
    )
    if len(specs) > 1 and (args.input_json or args.failure_report):
        raise SystemExit(
            '--input-json and --failure-report need a single --version'
        )
    results = [convert_version(spec, args) for spec in specs]
    for result in results:
        result.raise_if_incomplete()


if __name__ == '__main__':
    main()
