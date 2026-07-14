"""Fetch and convert the official Terminal-Bench 2.0 leaderboard.

The leaderboard is server-rendered and includes a structured JSON payload in
its Next.js flight data. The adapter archives or replays the original HTML
rather than maintaining a copied leaderboard in source code.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from every_eval_ever.adapters.terminal_bench_2.provenance import (
    terminal_bench_provenance,
)
from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import SCHEMA_VERSION, save_evaluation_log

LEADERBOARD_URL = 'https://www.tbench.ai/leaderboard/terminal-bench/2.0'
DEFAULT_OUTPUT_DIR = Path('data/terminal-bench-2.0')
REQUEST_TIMEOUT_S = 120
USER_AGENT = 'every-eval-ever terminal-bench-2 adapter'
NUM_TASKS = 89
NUM_TRIALS = 5

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
}


class _FlightScriptParser(HTMLParser):
    """Collect inline Next.js flight scripts from a leaderboard response."""

    def __init__(self) -> None:
        super().__init__()
        self.in_script = False
        self.scripts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.in_script = tag == 'script'

    def handle_endtag(self, tag: str) -> None:
        if tag == 'script':
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and 'self.__next_f.push' in data:
            self.scripts.append(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-url',
        default=LEADERBOARD_URL,
        help=f'Leaderboard URL (default: {LEADERBOARD_URL}).',
    )
    parser.add_argument(
        '--input-html',
        type=Path,
        help='Replay a saved leaderboard HTML response instead of fetching.',
    )
    parser.add_argument(
        '--save-raw-html',
        type=Path,
        help='Save the exact HTML response used by this run.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    return parser.parse_args()


def fetch_leaderboard_html(url: str) -> bytes:
    response = requests.get(
        url,
        headers={'User-Agent': USER_AGENT},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.content


def _flight_messages(html: str) -> list[str]:
    parser = _FlightScriptParser()
    parser.feed(html)

    messages: list[str] = []
    for script in parser.scripts:
        match = re.fullmatch(
            r'self\.__next_f\.push\((.*)\)\s*',
            script,
            re.DOTALL,
        )
        if match is None:
            continue
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[1], str)
        ):
            messages.append(value[1])
    return messages


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'leaderboard row has invalid {field!r}: {value!r}')
    return value.strip()


def _required_text_list(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f'leaderboard row has invalid {field!r}: {value!r}')
    return [item.strip() for item in value]


def _optional_text(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'leaderboard row has invalid {field!r}: {value!r}')
    return value.strip()


def _number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'leaderboard row has invalid {field!r}: {value!r}')
    return float(value)


def _normalize_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    models = _required_text_list(row, 'model')
    model_orgs = _required_text_list(row, 'modelOrganization')
    accuracy = _number(row, 'accuracy')
    if not 0 <= accuracy <= 1:
        raise ValueError(
            f'leaderboard row accuracy must be in [0, 1], got {accuracy}'
        )

    raw_stderr = row.get('stderr')
    if raw_stderr is None:
        stderr = None
    else:
        stderr = _number(row, 'stderr')
        if stderr < 0:
            raise ValueError(
                f'leaderboard row stderr must be non-negative, got {stderr}'
            )

    verified = row.get('verified')
    if not isinstance(verified, bool):
        raise ValueError(
            f"leaderboard row has invalid 'verified': {verified!r}"
        )

    return {
        'rank': rank,
        'agent': _required_text(row, 'agent'),
        'model': models[0] if len(models) == 1 else 'Multiple',
        'date': _required_text(row, 'date'),
        'agent_org': _required_text(row, 'agentOrganization'),
        'model_org': (
            model_orgs[0] if len(set(model_orgs)) == 1 else 'Multiple'
        ),
        # Match the precision published in the visible official leaderboard.
        'accuracy': round(accuracy * 100, 1),
        # The embedded source field is an actual standard error, not the
        # 95%-interval half-width displayed after the ± symbol in the table.
        'stderr': None if stderr is None else round(stderr * 100, 6),
        'verified': verified,
        'agent_name': _optional_text(row, 'agentName'),
        'agent_version': _optional_text(row, 'agentVersion'),
        'agent_url': _optional_text(row, 'agentUrl'),
        'integration_method': _optional_text(row, 'integrationMethod'),
        'model_names': _required_text_list(row, 'modelNames'),
        'model_providers': _required_text_list(row, 'modelProviders'),
    }


def extract_leaderboard_rows(
    html: str | bytes,
) -> list[dict[str, Any]]:
    """Extract and validate the official structured leaderboard rows."""
    if isinstance(html, bytes):
        html = html.decode('utf-8')

    decoder = json.JSONDecoder()
    candidates: list[list[Any]] = []
    marker = '"rows":'

    for message in _flight_messages(html):
        position = message.find(marker)
        if position < 0:
            continue
        try:
            value, _ = decoder.raw_decode(message, position + len(marker))
        except json.JSONDecodeError as exc:
            raise ValueError(
                'Terminal-Bench flight data contains malformed rows JSON'
            ) from exc
        if isinstance(value, list):
            candidates.append(value)

    if len(candidates) != 1:
        raise ValueError(
            'Expected exactly one Terminal-Bench leaderboard rows payload, '
            f'found {len(candidates)}'
        )
    if not candidates[0]:
        raise ValueError('Terminal-Bench leaderboard returned no rows')

    normalized: list[dict[str, Any]] = []
    for rank, row in enumerate(candidates[0], start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f'Terminal-Bench leaderboard row {rank} is not an object'
            )
        normalized.append(_normalize_row(row, rank))
    return normalized


def get_org_slug(org_name: str) -> str:
    return ORG_SLUG_MAP.get(
        org_name,
        re.sub(r'[^a-z0-9]+', '-', org_name.casefold()).strip('-'),
    )


def get_model_slug(model_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', model_name.casefold()).strip('-')


def make_model_id(model_org: str, model_name: str) -> str:
    return f'{get_org_slug(model_org)}/{get_model_slug(model_name)}'


def convert_entry(
    entry: dict[str, Any],
    retrieved_timestamp: str,
    *,
    source_url: str = LEADERBOARD_URL,
) -> EvaluationLog:
    """Convert one normalized leaderboard entry to an EvaluationLog."""
    model_id = make_model_id(entry['model_org'], entry['model'])
    provenance = terminal_bench_provenance(model_id)
    agent_slug = re.sub(r'[^a-z0-9]+', '-', entry['agent'].casefold()).strip(
        '-'
    )
    model_slug = get_model_slug(entry['model'])
    eval_id = (
        f'terminal-bench-2.0/{agent_slug}__{model_slug}/{retrieved_timestamp}'
    )

    uncertainty = None
    if entry['stderr'] is not None:
        uncertainty = Uncertainty(
            standard_error=StandardError(
                value=entry['stderr'],
                method='reported by Terminal-Bench',
            ),
            num_samples=NUM_TASKS * NUM_TRIALS,
        )

    additional_details = {
        'agent_name': entry['agent'],
        'agent_organization': entry['agent_org'],
        'deployment_type': provenance.deployment_type,
        'model_availability': provenance.model_availability,
        'leaderboard_verified': str(entry['verified']).lower(),
        'source_agent_name': entry['agent_name'] or 'unknown',
        'source_agent_version': entry['agent_version'] or 'unknown',
        'source_agent_url': entry['agent_url'] or 'unknown',
        'integration_method': entry['integration_method'] or 'unknown',
        'source_model_names': json.dumps(entry['model_names']),
        'source_model_providers': json.dumps(entry['model_providers']),
    }

    eval_result = EvaluationResult(
        evaluation_name='terminal-bench-2.0',
        source_data=SourceDataUrl(
            dataset_name='terminal-bench-2.0',
            source_type='url',
            url=[source_url],
        ),
        evaluation_timestamp=entry['date'],
        metric_config=MetricConfig(
            evaluation_description=(
                f'Task resolution accuracy across {NUM_TASKS} terminal tasks '
                f'with {NUM_TRIALS} trials each'
            ),
            metric_id='terminal_bench_2.task_resolution_accuracy',
            metric_name='Task Resolution Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=100,
        ),
        score_details=ScoreDetails(
            score=entry['accuracy'],
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
                    f'-a "{entry["agent"]}" -m "{entry["model"]}" -k 5'
                ),
            ),
        ),
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=entry['date'],
        source_metadata=SourceMetadata(
            source_name='Terminal-Bench 2.0',
            source_type='documentation',
            source_organization_name='Terminal-Bench',
            source_organization_url='https://www.tbench.ai',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name='harbor', version='unknown'),
        model_info=ModelInfo(
            name=entry['model'],
            id=model_id,
            developer=entry['model_org'],
            inference_platform=provenance.inference_platform,
            inference_engine=InferenceEngine(
                name=provenance.inference_engine_name,
                version=provenance.inference_engine_version,
            ),
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def main() -> None:
    args = parse_args()
    if args.input_html is None:
        html = fetch_leaderboard_html(args.source_url)
    else:
        html = args.input_html.read_bytes()

    if args.save_raw_html is not None:
        args.save_raw_html.parent.mkdir(parents=True, exist_ok=True)
        args.save_raw_html.write_bytes(html)

    rows = extract_leaderboard_rows(html)
    retrieved_timestamp = str(time.time())
    bundles = [
        (
            entry,
            convert_entry(
                entry,
                retrieved_timestamp,
                source_url=args.source_url,
            ),
        )
        for entry in rows
    ]

    for entry, eval_log in bundles:
        filepath = save_evaluation_log(
            eval_log,
            args.output_dir,
            get_org_slug(entry['model_org']),
            get_model_slug(entry['model']),
        )
        print(f'[{entry["rank"]:3d}] {filepath}')

    print(f'\nGenerated {len(bundles)} files in {args.output_dir}/')


if __name__ == '__main__':
    main()
