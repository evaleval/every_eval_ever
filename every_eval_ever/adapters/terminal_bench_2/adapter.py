"""
Script to convert Terminal-Bench 2.0 leaderboard data to the EvalEval schema format.

Data source:
- Terminal-Bench 2.0 leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.0

Terminal-Bench is an agentic coding benchmark that evaluates agent+model pairs on
87 terminal-based tasks with 5 trials each. Each leaderboard entry represents a
unique agent+model combination. Agent metadata is stored in model_info.additional_details.

Usage:
    uv run python -m every_eval_ever.adapters.terminal_bench_2.adapter
"""

import argparse
import json
import math
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
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
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

LEADERBOARD_URL = 'https://www.tbench.ai/leaderboard/terminal-bench/2.0'
OUTPUT_DIR = 'data/terminal-bench-2.0'
TASK_COUNT = 87
TRIALS_PER_TASK = 5

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


class _LeaderboardTableParser(HTMLParser):
    """Extract text cells from HTML table rows without a scraper dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag == 'tr':
            self._row = []
        elif tag == 'td' and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == 'td' and self._row is not None:
            self._row.append(''.join(self._cell_parts or []).strip())
            self._cell_parts = None
        elif tag == 'tr' and self._row is not None:
            self.rows.append(self._row)
            self._row = None


_ACCURACY_RE = re.compile(
    r'^\s*(?P<score>\d+(?:\.\d+)?)%\s*'
    r'(?:±\s*(?P<stderr>\d+(?:\.\d+)?|N/A))?\s*$'
)


def fetch_leaderboard_html(url: str = LEADERBOARD_URL) -> str:
    """Fetch the rendered Terminal-Bench leaderboard page."""
    request = Request(url, headers={'User-Agent': 'EEE-adapter/1.0'})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        content_type = response.headers.get('Content-Type')
    raw_capture.record(url=url, content=body, content_type=content_type)
    return body.decode('utf-8', errors='strict')


def save_raw_html(html: str, path: Path | None) -> None:
    """Persist the exact fetched source outside the validated data tree."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding='utf-8')


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


def parse_leaderboard_html(
    html: str,
) -> SourceConversionResult[dict]:
    """Parse table rows and retain malformed leaderboard-row provenance."""
    parser = _LeaderboardTableParser()
    parser.feed(html)
    parser.close()

    entries = []
    failures: list[SourceRecordFailure] = []
    candidate_rows = 0
    for row_index, cells in enumerate(parser.rows):
        rank_position = next(
            (index for index, value in enumerate(cells) if value.isdigit()),
            None,
        )
        if rank_position is None:
            continue
        candidate_rows += 1
        row_ref = f'HTML leaderboard row {row_index + 1}'
        values = cells[rank_position : rank_position + 7]
        if len(values) != 7:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=(
                        'leaderboard row has fewer than seven fields after '
                        'its rank'
                    ),
                    source_record={'cells': cells},
                )
            )
            continue
        rank, agent, model, date, agent_org, model_org, accuracy = values
        match = _ACCURACY_RE.fullmatch(accuracy)
        if match is None:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=f'could not parse accuracy cell {accuracy!r}',
                    source_record={'cells': cells},
                )
            )
            continue
        score = float(match.group('score'))
        stderr_text = match.group('stderr')
        stderr = None if stderr_text in (None, 'N/A') else float(stderr_text)
        if not 0.0 <= score <= 100.0:
            failures.append(
                SourceRecordFailure(
                    source_ref=row_ref,
                    reason=f'accuracy must be between 0 and 100, got {score}',
                    source_record={'cells': cells},
                )
            )
            continue
        entries.append(
            {
                'rank': int(rank),
                'agent': agent,
                'model': model,
                'date': date,
                'agent_org': agent_org,
                'model_org': model_org,
                'accuracy': score,
                'stderr': stderr,
            }
        )
    if candidate_rows == 0:
        raise ValueError('no ranked leaderboard rows found in source HTML')
    return SourceConversionResult(
        source_name='Terminal-Bench 2.0 leaderboard HTML',
        total_records=candidate_rows,
        records=entries,
        failures=failures,
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
    stderr_value = entry.get('stderr')
    stderr = None if stderr_value is None else float(stderr_value)
    if stderr is not None and (not math.isfinite(stderr) or stderr < 0.0):
        raise ValueError(
            'Terminal-Bench standard error must be a finite non-negative '
            f'number, got {stderr_value!r}'
        )
    model_id = make_model_id(model_org, model_name)
    agent_slug = sanitize_filename(agent.lower().replace(' ', '-'))
    model_slug = get_model_slug(model_name)

    eval_id = (
        f'terminal-bench-2.0/{agent_slug}__{model_slug}/{retrieved_timestamp}'
    )

    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(
            standard_error=StandardError(value=stderr),
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
        '--save-raw-html',
        type=Path,
        help='Save the fetched leaderboard HTML outside --output-dir.',
    )
    parser.add_argument(
        '--leaderboard-url',
        default=LEADERBOARD_URL,
        help='Terminal-Bench leaderboard URL (for testing or source moves).',
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
    if args.save_raw_html is not None and is_subpath(
        args.save_raw_html,
        args.output_dir,
    ):
        raise SystemExit(
            '--save-raw-html must point outside --output-dir so the '
            'validator cannot mistake source HTML for evaluation data'
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
        html = fetch_leaderboard_html(args.leaderboard_url)
        save_raw_html(html, args.save_raw_html)
        parsed = parse_leaderboard_html(html)

    converted = convert_logs(
        parsed.records,
        leaderboard_url=args.leaderboard_url,
    )
    result = SourceConversionResult(
        source_name='Terminal-Bench 2.0',
        total_records=parsed.total_records,
        records=converted.records,
        failures=[*parsed.failures, *converted.failures],
    )
    paths = export(result.records, args.output_dir)
    for path in paths:
        print(path)
    print(f'Generated {len(paths)} files in {args.output_dir}/')
    if result.failures:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()


if __name__ == '__main__':
    main()
