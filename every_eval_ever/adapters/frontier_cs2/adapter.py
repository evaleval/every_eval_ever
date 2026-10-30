#!/usr/bin/env python3
"""Convert FrontierCS 2 benchmark runs into Every Eval Ever records.

Data source:
- Hugging Face dataset: https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-runs
- Manifest: metadata/trace-manifest.tsv
- Formal results: metadata/formal-results.json
- Reference: FrontierCS 2: A Benchmark for Computer Science R&D (https://huggingface.co/papers/2610.29932)

FrontierCS 2 evaluates autonomous code and science agents across 48 computer science
research tasks and 19 model candidates, producing 912 formal evaluation runs.

This adapter emits one aggregate EvaluationLog per model candidate (19 total logs),
each containing 48 EvaluationResult items corresponding to the formal task suite.

Usage:
    # Direct execution against Hugging Face
    python -m every_eval_ever.adapters.frontier_cs2.adapter --output-dir data/frontier-cs-2

    # Offline replay from local fixture files
    python -m every_eval_ever.adapters.frontier_cs2.adapter \\
        --input-json /path/to/formal-results.json \\
        --input-tsv /path/to/trace-manifest.tsv \\
        --output-dir data/frontier-cs-2
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
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

logger = logging.getLogger(__name__)

HF_REPO_ID = 'FrontierCS2/FrontierCS-2-runs'
COLLECTION = 'frontier-cs-2'
DEFAULT_OUTPUT_DIR = Path(f'data/{COLLECTION}')
PAPER_URL = 'https://huggingface.co/papers/2610.29932'
GITHUB_URL = 'https://github.com/FrontierCS/FrontierCS-2'
WEBSITE_URL = 'https://frontier-cs.org'

# Upstream model identifier -> (Developer Display, Developer Slug, Model Slug, Model Availability)
# Deployment type is uniformly 'externally_managed' as all formal runs executed against external APIs / services.
MODEL_CATALOG: dict[str, tuple[str, str, str, str]] = {
    'astra': ('OpenAI', 'openai', 'gpt-6-astra', 'closed_weights'),
    'sol': ('OpenAI', 'openai', 'gpt-6.1-sol', 'closed_weights'),
    'terra': ('OpenAI', 'openai', 'gpt-6.1-terra', 'closed_weights'),
    'gemini': ('Google', 'google', 'gemini-4', 'closed_weights'),
    'gemini-agy': ('Google', 'google', 'gemini-4-argon', 'closed_weights'),
    'carbon-g': ('Google', 'google', 'gemini-4-carbon', 'closed_weights'),
    'opus': ('Anthropic', 'anthropic', 'claude-opus-5.5', 'closed_weights'),
    'fable51': ('Anthropic', 'anthropic', 'claude-fable-5.1', 'closed_weights'),
    'ds': ('DeepSeek', 'deepseek', 'deepseek-v4.1-flash', 'open_weights'),
    'ds1017': ('DeepSeek', 'deepseek', 'deepseek-v4.1-flash-1017', 'open_weights'),
    'k27': ('Moonshot AI', 'moonshot', 'kimi-k2.7-code', 'closed_weights'),
    'k3': ('Moonshot AI', 'moonshot', 'kimi-k3', 'open_weights'),
    'k31': ('Moonshot AI', 'moonshot', 'kimi-k3.1-pro', 'closed_weights'),
    'glm': ('Zhipu AI', 'zhipu-ai', 'glm-5.3-zcode', 'closed_weights'),
    'glmf': ('Zhipu AI', 'zhipu-ai', 'glm-5.3-flash-zcode', 'closed_weights'),
    'qwen': ('Alibaba', 'alibaba', 'qwen-3.8-max', 'closed_weights'),
    'step': ('StepFun', 'stepfun', 'step-5-preview', 'closed_weights'),
    'muse': ('Muse', 'muse', 'muse-spark-1.3', 'closed_weights'),
    'wonder': ('Muse', 'muse', 'muse-wonder-1.0', 'closed_weights'),
}


def fetch_hf_file(repo_id: str, filename: str, revision: str = 'main') -> bytes:
    """Fetch a file from Hugging Face dataset via HTTP with raw-capture support."""
    url = f'https://huggingface.co/datasets/{repo_id}/raw/{revision}/{filename}'
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'every_eval_ever/frontier_cs2'},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
        content_type = resp.headers.get('Content-Type')
    raw_capture.record(url=url, content=content, content_type=content_type)
    return content


def get_source_version(repo_id: str = HF_REPO_ID) -> str:
    """Obtain a deterministic source freshness token for --emit-source-version."""
    url = f'https://huggingface.co/api/datasets/{repo_id}'
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'every_eval_ever/frontier_cs2'},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.loads(resp.read().decode('utf-8'))
    commit_sha = info.get('sha') or 'unknown'
    return f'hf:{repo_id}:{commit_sha[:16]}'


def parse_trace_manifest(content: str) -> dict[str, dict[str, str]]:
    """Parse trace-manifest.tsv mapping run_id -> manifest row."""
    reader = csv.DictReader(io.StringIO(content), delimiter='\t')
    by_run_id: dict[str, dict[str, str]] = {}
    for row in reader:
        run_id = row.get('run_id')
        if run_id:
            by_run_id[run_id] = row
    return by_run_id


def sanitize_str_dict(raw: dict[str, Any]) -> dict[str, str]:
    """Ensure all dictionary values are strings for schema compatibility."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, str):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def make_evaluation_result(
    run: dict[str, Any],
    task: dict[str, Any],
    manifest_entry: dict[str, str] | None,
) -> EvaluationResult:
    """Construct an EvaluationResult for a single task run."""
    task_name = task.get('task') or run.get('task', 'unknown_task')
    family = task.get('family', 'frontier-cs-2.0/systems')
    family_slug = family.replace('/', '.').replace('-', '_')

    # Bounded platform run score: 0.0 to 100.0
    raw_score = run.get('score')
    score_status = 'accepted'
    if raw_score is None:
        score = 0.0
        score_status = 'unscored_failure'
    else:
        score = max(0.0, min(100.0, float(raw_score)))

    details_map: dict[str, Any] = {
        'run_id': run.get('run_id'),
        'verdict': run.get('verdict') or 'UNKNOWN',
        'outcome': run.get('outcome') or 'UNKNOWN',
        'score_status': score_status,
        'platform_state': run.get('platform_state'),
        'result_source': run.get('result_source'),
        'failure_code': run.get('failure_code'),
        'task_display_name': task.get('display_name'),
        'task_category': task.get('category'),
        'paper_title': task.get('paper_title'),
        'beats_reference': run.get('beats_reference'),
        'margin': run.get('margin'),
        'reference_score': run.get('reference_score'),
        'checks_passed': run.get('checks_passed'),
        'checks_total': run.get('checks_total'),
        'elapsed_seconds': run.get('elapsed_seconds'),
        'total_tokens': run.get('total_tokens'),
        'model_requests': run.get('model_requests'),
        'patch_path': run.get('patch_path'),
        'patch_sha256': run.get('patch_sha256'),
        'trace_path': run.get('trace_path'),
        'trace_sha256': run.get('trace_sha256'),
    }
    if manifest_entry:
        details_map['manifest_sha256'] = manifest_entry.get('sha256')
        details_map['manifest_size'] = manifest_entry.get('size')

    source_data = SourceDataUrl(
        dataset_name=COLLECTION,
        source_type='url',
        url=[GITHUB_URL, f'https://huggingface.co/datasets/{HF_REPO_ID}'],
    )

    metric_config = MetricConfig(
        evaluation_description=(
            f'FrontierCS 2 evaluation score for {task.get("display_name", task_name)} '
            f'({task.get("category", "General")}).'
        ),
        metric_id='frontier-cs-2.score',
        metric_name='Platform Run Score',
        metric_kind='score',
        metric_unit='points',
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=100.0,
    )

    return EvaluationResult(
        evaluation_result_id=run.get('record_id') or f'{task_name}/{run.get("model_id")}',
        evaluation_name=f'{family_slug}.{task_name}',
        source_data=source_data,
        evaluation_timestamp=run.get('finished_at') or run.get('created_at'),
        metric_config=metric_config,
        score_details=ScoreDetails(
            score=score,
            details=sanitize_str_dict(details_map),
        ),
    )


def convert_model_log(
    model_spec: dict[str, Any],
    runs: list[dict[str, Any]],
    tasks_by_name: dict[str, dict[str, Any]],
    manifest_by_run_id: dict[str, dict[str, str]],
    generated_at: str,
    retrieved_timestamp: str,
    output_dir: Path,
) -> EvaluationLogOutput:
    """Build a complete EvaluationLog for one model candidate across all its task runs."""
    raw_model_id = model_spec['model_id']
    candidate_name = model_spec.get('candidate_name', raw_model_id)

    if raw_model_id not in MODEL_CATALOG:
        raise ValueError(f'unrecognized model candidate ID {raw_model_id!r}')

    dev_display, dev_slug, model_slug, availability = MODEL_CATALOG[raw_model_id]
    canonical_model_id = f'{dev_slug}/{model_slug}'

    # Build evaluation results for all runs belonging to this model
    results: list[EvaluationResult] = []
    for run in runs:
        task_name = run.get('task')
        task_info = tasks_by_name.get(task_name, {'task': task_name, 'display_name': task_name})
        manifest_entry = manifest_by_run_id.get(run.get('run_id', ''))
        results.append(make_evaluation_result(run, task_info, manifest_entry))

    results.sort(key=lambda r: r.evaluation_name)

    model_info = ModelInfo(
        name=candidate_name,
        id=canonical_model_id,
        developer=dev_display,
        additional_details={
            'deployment_type': 'externally_managed',
            'model_availability': availability,
            'source_model_id': raw_model_id,
        },
    )

    source_metadata = SourceMetadata(
        source_name='FrontierCS 2 Formal Evaluation Runs',
        source_type=SourceType.evaluation_run,
        source_organization_name='FrontierCS Consortium',
        source_organization_url=WEBSITE_URL,
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details={
            'huggingface_dataset': HF_REPO_ID,
            'paper_url': PAPER_URL,
            'github_url': GITHUB_URL,
            'upstream_generated_at': generated_at,
        },
    )

    eval_library = EvalLibrary(
        name='frontier-cs-2',
        version='2.0',
        additional_details={'benchmark_family': 'frontier-cs-2.0'},
    )

    eval_log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'frontier-cs-2/{raw_model_id}/{generated_at}',
        evaluation_timestamp=generated_at,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=source_metadata,
        eval_library=eval_library,
        model_info=model_info,
        evaluation_results=results,
    )

    return EvaluationLogOutput(
        eval_log=eval_log,
        base_dir=output_dir,
        developer=dev_slug,
        model_name=model_slug,
    )


def convert_payload(
    formal_results: dict[str, Any],
    manifest_by_run_id: dict[str, dict[str, str]],
    output_dir: Path,
    retrieved_timestamp: str,
) -> SourceConversionResult:
    """Transform formal results payload and trace manifest into EvaluationLog outputs."""
    models = formal_results.get('models', [])
    tasks = formal_results.get('tasks', [])
    runs = formal_results.get('runs', [])
    generated_at = formal_results.get('generated_at', '2026-10-30T00:00:00Z')

    tasks_by_name = {t.get('task'): t for t in tasks if isinstance(t, dict) and t.get('task')}

    runs_by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        mid = run.get('model_id')
        if mid:
            runs_by_model.setdefault(mid, []).append(run)

    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []

    for model_spec in models:
        mid = model_spec.get('model_id')
        try:
            model_runs = runs_by_model.get(mid, [])
            output = convert_model_log(
                model_spec,
                model_runs,
                tasks_by_name,
                manifest_by_run_id,
                generated_at,
                retrieved_timestamp,
                output_dir,
            )
            outputs.append(output)
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'model {mid}',
                    reason=str(exc),
                    source_record=model_spec,
                )
            )

    return SourceConversionResult(
        source_name='FrontierCS 2 Formal Runs',
        total_records=len(models),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert FrontierCS 2 benchmark runs into Every Eval Ever records.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output collection directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--from-hf',
        action='store_true',
        default=True,
        help='Fetch formal results and trace manifest directly from Hugging Face Hub.',
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Local path to formal-results.json for offline replay.',
    )
    parser.add_argument(
        '--input-tsv',
        type=Path,
        help='Local path to trace-manifest.tsv for offline replay.',
    )
    parser.add_argument(
        '--emit-source-version',
        action='store_true',
        help='Print the current upstream source version token and exit 0.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.emit_source_version:
        token = get_source_version(HF_REPO_ID)
        print(token)
        return 0

    raw_capture.record_hf_dataset(HF_REPO_ID)

    if args.input_json and args.input_tsv:
        formal_results_text = args.input_json.read_text(encoding='utf-8')
        manifest_text = args.input_tsv.read_text(encoding='utf-8')
    else:
        formal_results_text = fetch_hf_file(
            HF_REPO_ID, 'metadata/formal-results.json'
        ).decode('utf-8')
        manifest_text = fetch_hf_file(
            HF_REPO_ID, 'metadata/trace-manifest.tsv'
        ).decode('utf-8')

    formal_results = json.loads(formal_results_text)
    manifest_by_run_id = parse_trace_manifest(manifest_text)

    retrieved_timestamp = str(time.time())
    result = convert_payload(
        formal_results=formal_results,
        manifest_by_run_id=manifest_by_run_id,
        output_dir=args.output_dir,
        retrieved_timestamp=retrieved_timestamp,
    )

    written_paths = save_evaluation_logs(result.records)
    for p in written_paths:
        print(f'  {p}')

    if result.failures:
        report_path = save_failure_report(
            result,
            default_failure_report_path(args.output_dir),
        )
        print(f'Failure report written to {report_path}', file=sys.stderr)
        result.raise_if_incomplete()

    print(f'Converted {len(written_paths)} models ({result.total_records} source records) into {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
