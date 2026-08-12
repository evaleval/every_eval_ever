"""
Script to convert Exgentic Open Agent Leaderboard results to the EvalEval schema format.

Exgentic is an open-source framework for evaluating AI agents across multiple
benchmarks (AppWorld, SWE-bench, BrowseComp+, Tau2, etc.) with different agent
frameworks (Claude Code, LiteLLM Tool Calling, SmolAgents, etc.) and models.

Each evaluation run produces a results.json file containing aggregate scores,
session counts, cost data, and per-session details. This adapter reads those
results and converts them to EEE-conformant JSON files.

Data source:
- Exgentic experiments output: results.json files produced by `exgentic batch aggregate`
- HuggingFace dataset: https://huggingface.co/datasets/Exgentic/results

Usage:
    # From local experiment results
    uv run python -m every_eval_ever.adapters.exgentic.adapter --results-dir /path/to/experiments

    # From HuggingFace dataset
    uv run python -m every_eval_ever.adapters.exgentic.adapter --from-hf
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
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
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

OUTPUT_DIR = 'data/exgentic'
HF_DATASET = 'Exgentic/results'

# Map model name prefixes to developer organizations
MODEL_DEVELOPER_MAP = {
    'claude': ('Anthropic', 'anthropic'),
    'gpt': ('OpenAI', 'openai'),
    'gemini': ('Google', 'google'),
    'deepseek': ('DeepSeek', 'deepseek'),
    'kimi': ('Moonshot AI', 'moonshot'),
}


def parse_model_info(model_name: str) -> tuple[str, str, str]:
    """Extract developer display name, developer slug, and model slug from exgentic model_name.

    Exgentic model names follow the pattern: provider/platform/model-name
    e.g. 'openai/aws/claude-opus-4-5', 'openai/Azure/gpt-5.2-2025-12-11'

    Returns:
        (developer_display, developer_slug, model_slug)
    """
    parts = model_name.split('/')
    raw_model = parts[-1] if parts else model_name

    developer_display = None
    developer_slug = None
    lower = raw_model.lower()
    for prefix, (display, slug) in MODEL_DEVELOPER_MAP.items():
        if lower.startswith(prefix):
            developer_display = display
            developer_slug = slug
            break

    if developer_display is None or developer_slug is None:
        raise ValueError(
            f'Cannot determine Exgentic model developer from {model_name!r}'
        )
    return developer_display, developer_slug, raw_model


def make_agent_slug(agent_name: str) -> str:
    """Convert agent display name to a URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', agent_name.lower()).strip('-')


def convert_result(result: dict, retrieved_timestamp: str) -> EvaluationLog:
    """Convert a single exgentic result dict to an EvaluationLog."""
    model_name_raw = require_identity(
        result.get('model_name'), 'Exgentic model name'
    )
    developer_display, developer_slug, model_slug = parse_model_info(
        model_name_raw
    )
    model_id = f'{developer_slug}/{model_slug}'

    benchmark = require_identity(
        result.get('benchmark_name') or result.get('benchmark'),
        'Exgentic benchmark',
    )
    agent_name = require_identity(
        result.get('agent_name') or result.get('agent'),
        'Exgentic agent',
    )
    agent_framework = result.get('agent') or make_agent_slug(agent_name)
    agent_slug = make_agent_slug(agent_name)
    subset = result.get('subset_name')

    eval_name = benchmark.lower().replace(' ', '-')
    if subset:
        eval_name = f'{eval_name}/{subset}'

    score = result.get('benchmark_score')
    if score is None:
        score = result.get('average_score')
    if score is None:
        raise ValueError('Exgentic benchmark score is required')

    # Build uncertainty from session counts
    total = result.get('total_sessions')
    uncertainty = None
    if total and int(total) > 0:
        uncertainty = Uncertainty(num_samples=int(total))

    # Build score details
    details: dict[str, str] = {}
    if result.get('average_agent_cost') is not None:
        details['average_agent_cost'] = str(
            round(float(result['average_agent_cost']), 2)
        )
    if result.get('total_run_cost') is not None:
        details['total_run_cost'] = str(
            round(float(result['total_run_cost']), 2)
        )
    if result.get('average_steps') is not None:
        details['average_steps'] = str(round(float(result['average_steps']), 2))
    if result.get('percent_finished') is not None:
        details['percent_finished'] = str(
            round(float(result['percent_finished']), 4)
        )

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataUrl(
            dataset_name=eval_name,
            source_type='url',
            url=['https://github.com/Exgentic/exgentic'],
        ),
        evaluation_timestamp=retrieved_timestamp,
        metric_config=MetricConfig(
            evaluation_description=f'{benchmark} benchmark evaluation'
            + (f' ({subset} subset)' if subset else ''),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(
            score=round(float(score), 4) if score is not None else 0.0,
            uncertainty=uncertainty,
            details=details if details else None,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    additional_details={
                        'agent_name': agent_name,
                        'agent_framework': agent_framework,
                    },
                ),
            ),
        ),
    )

    sanitized_model_id = model_id.replace('/', '_')
    evaluation_id = (
        f'{eval_name}/{agent_slug}__{sanitized_model_id}/{retrieved_timestamp}'
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name='Exgentic Open Agent Leaderboard',
            source_type='evaluation_run',
            source_organization_name='Exgentic',
            source_organization_url='https://github.com/Exgentic',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(
            name='exgentic',
            version='0.1.0',
        ),
        model_info=ModelInfo(
            name=model_slug,
            id=model_id,
            developer=developer_display,
            additional_details={
                'agent_name': agent_name,
                'agent_framework': agent_framework,
            },
        ),
        evaluation_results=[eval_result],
    )


def collect_results_from_dir(
    results_dir: str,
) -> SourceConversionResult[dict]:
    """Load usable result files and preserve every rejected source reference."""
    results = []
    base = Path(results_dir)

    config_paths = sorted(base.rglob('config.json'))
    failures: list[SourceRecordFailure] = []
    for config_path in config_paths:
        config = None
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            run_id = config.get('run_id')
            if not run_id:
                raise ValueError('missing run_id')
            results_path = config_path.parent / run_id / 'results.json'
            if not results_path.is_file():
                raise FileNotFoundError(f'{results_path}: file not found')
            payload = json.loads(results_path.read_text(encoding='utf-8'))
            if 'benchmark_score' not in payload:
                raise ValueError(f'{results_path}: missing benchmark_score')
            results.append(payload)
        except (json.JSONDecodeError, OSError) as e:
            failures.append(
                SourceRecordFailure(
                    source_ref=str(config_path),
                    reason=str(e),
                    source_record=config,
                )
            )
        except (TypeError, ValueError) as e:
            failures.append(
                SourceRecordFailure(
                    source_ref=str(config_path),
                    reason=str(e),
                    source_record=config,
                )
            )
    return SourceConversionResult(
        source_name='Exgentic local results',
        total_records=len(config_paths),
        records=results,
        failures=failures,
    )


def load_results_from_dir(results_dir: str) -> list[dict]:
    """Strict API for callers that require every local result to load."""
    result = collect_results_from_dir(results_dir)
    result.raise_if_incomplete()
    return result.records


def load_results_from_hf() -> list[dict]:
    """Load results from the HuggingFace dataset (default subset with raw exgentic data)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' package required. Install with: pip install datasets"
        )
        sys.exit(1)

    raw_capture.record_hf_dataset(HF_DATASET)
    ds = load_dataset(HF_DATASET, split='train')
    return list(ds)


def convert_results(
    results: list[dict],
    retrieved_timestamp: str,
    output_dir: str = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert usable results while retaining rejected source records."""
    outputs = []
    failures: list[SourceRecordFailure] = []
    for index, result in enumerate(results):
        try:
            eval_log = convert_result(result, retrieved_timestamp)
            model_id = require_identity(
                eval_log.model_info.id,
                'Exgentic model id',
            )
            if '/' not in model_id:
                raise ValueError(
                    f'Exgentic model id must be developer/model: {model_id!r}'
                )
            developer_slug, model_name = model_id.split('/', 1)
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=output_dir,
                    developer=developer_slug,
                    model_name=model_name,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'result row {index}',
                    reason=str(exc),
                    source_record=result,
                )
            )
    return SourceConversionResult(
        source_name='Exgentic',
        total_records=len(results),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Exgentic results to Every Eval Ever format'
    )
    parser.add_argument(
        '--results-dir',
        help='Path to exgentic experiments directory containing config.json files',
    )
    parser.add_argument(
        '--from-hf',
        action='store_true',
        help=f'Load results from HuggingFace dataset ({HF_DATASET})',
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory for EEE JSON files (default: {OUTPUT_DIR})',
    )
    args = parser.parse_args(argv)

    if not args.results_dir and not args.from_hf:
        parser.error('Specify either --results-dir or --from-hf')
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    if args.results_dir:
        loaded = collect_results_from_dir(args.results_dir)
        results = loaded.records
    else:
        results = load_results_from_hf()
        loaded = SourceConversionResult(
            source_name='Exgentic Hugging Face results',
            total_records=len(results),
            records=results,
            failures=[],
        )

    print(f'Loaded {len(results)} of {loaded.total_records} source result(s)')

    retrieved_timestamp = str(time.time())
    converted = convert_results(results, retrieved_timestamp, args.output_dir)
    combined = SourceConversionResult(
        source_name='Exgentic',
        total_records=loaded.total_records,
        records=converted.records,
        failures=[*loaded.failures, *converted.failures],
    )
    if not combined.records and not combined.failures:
        combined.failures.append(
            SourceRecordFailure(
                source_ref=(args.results_dir or 'Hugging Face dataset'),
                reason='no source results found',
            )
        )

    paths = save_evaluation_logs(combined.records)
    for path in paths:
        print(f'  {path}')

    if combined.failures:
        report_path = save_failure_report(
            combined,
            default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        combined.raise_if_incomplete()

    print(f'\nGenerated {len(paths)} file(s) in {args.output_dir}/')


if __name__ == '__main__':
    main()
