"""
Script to convert SWE-bench Verified leaderboard data to the EvalEval schema format.

Data source:
- SWE-bench experiments repo: https://github.com/swe-bench/experiments
  Cloned to a temporary directory at runtime; cleaned up on exit.

Each subdirectory under evaluation/verified/ is a submission with:
  - metadata.yaml: model/org info, tags
  - results/results.json: resolved/no_generation/no_logs instance lists

Score = len(resolved) / 500  (500 total SWE-bench Verified instances)

Usage:
    uv run python -m every_eval_ever.adapters.swe_bench_verified.adapter
    uv run python -m every_eval_ever.adapters.swe_bench_verified.adapter \
        --output-dir /tmp/smoke/data/swe-bench-verified-leaderboard
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from every_eval_ever.adapters.swe_helpers import parse_date_from_dir
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
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    get_developer,
    get_model_id,
    raw_capture,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

SWE_BENCH_REPO = 'https://github.com/swe-bench/experiments'
SWE_BENCH_SUBDIR = 'evaluation/verified'
OUTPUT_DIR = 'data/swe-bench-verified-leaderboard'


def normalize_org(org) -> str:
    """Normalize org field which can be str, list, or None."""
    if isinstance(org, list):
        return ', '.join(str(o) for o in org if o)
    return str(org) if org else ''


def normalize_model_name(model) -> str:
    """Normalize a raw model value to a clean model name string.

    Handles:
    - HuggingFace URLs: https://huggingface.co/org/model → org/model
    - Plain strings returned as-is
    """
    if not model:
        return ''
    s = str(model)
    if s.startswith('https://huggingface.co/'):
        s = s[len('https://huggingface.co/') :]
    return s


def get_primary_model(tags: dict, info: dict, dir_name: str) -> str:
    """Extract the primary model name from tags, falling back to submission info."""
    raw = tags.get('model')
    # tags.model can be a list or a plain string
    if isinstance(raw, list):
        models = raw
    elif raw is not None:
        models = [raw]
    else:
        models = []

    if models:
        return normalize_model_name(models[0])
    # Submission metadata name is source-provided; the directory name is not
    # silently treated as model identity.
    return normalize_model_name(info.get('name'))


def convert_submission(
    submission_dir: Path, retrieved_timestamp: str, total_instances: int
) -> EvaluationLog:
    """Convert a single SWE-bench submission directory to an EvaluationLog."""
    dir_name = submission_dir.name
    if total_instances <= 0:
        raise ValueError('SWE-bench total_instances must be positive')

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            'pyyaml is required to run this adapter. Install it with: pip install pyyaml'
        ) from e

    # Read metadata
    with open(
        submission_dir / 'metadata.yaml',
        encoding='utf-8',
    ) as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f'metadata.yaml is not an object for {dir_name}')

    # Read results
    with open(
        submission_dir / 'results' / 'results.json',
        encoding='utf-8',
    ) as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise ValueError(f'results.json is not an object for {dir_name}')

    tags = metadata.get('tags', {}) or {}
    info = metadata.get('info', {}) or {}

    # Primary model: first element of tags.model (list or string), fallback to submission name
    primary_model = require_identity(
        get_primary_model(tags, info, dir_name),
        'SWE-bench Verified model',
    )

    developer = require_identity(
        get_developer(primary_model),
        'SWE-bench Verified model developer',
    )
    model_id = get_model_id(primary_model, developer)

    # Score: resolved / total_instances
    resolved = results.get('resolved', [])
    if not isinstance(resolved, list):
        raise ValueError(f'resolved must be a list for {dir_name}')
    if len(resolved) > total_instances:
        raise ValueError(
            f'resolved count exceeds total_instances for {dir_name}'
        )
    score = len(resolved) / total_instances

    # Build additional_details (all values must be strings)
    additional_details: dict[str, str] = {
        'submission_name': str(info.get('name', '')),
        'agent_organization': normalize_org(tags.get('org', '')),
        'open_source_model': str(tags.get('os_model', '')),
        'open_source_system': str(tags.get('os_system', '')),
        'verified': str(tags.get('checked', '')),
        'attempts': str((tags.get('system') or {}).get('attempts', '')),
        'submission_dir': dir_name,
    }
    site = info.get('site')
    if site:
        additional_details['site'] = str(site)
    report = info.get('report')
    if report:
        additional_details['report'] = str(report)

    # Score details
    score_details: dict[str, str] = {
        'resolved_count': str(len(resolved)),
    }
    no_generation = results.get('no_generation', [])
    if no_generation:
        score_details['no_generation_count'] = str(len(no_generation))
    no_logs = results.get('no_logs', [])
    if no_logs:
        score_details['no_logs_count'] = str(len(no_logs))

    # Sanitize identifier components for use in evaluation_id
    sanitized_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id.replace('/', '_'))
    submission_slug = re.sub(r'[^a-zA-Z0-9_.-]', '_', dir_name)
    eval_id = f'swe-bench-verified/{sanitized_id}/{submission_slug}/{retrieved_timestamp}'
    evaluation_timestamp = parse_date_from_dir(dir_name)

    eval_result = EvaluationResult(
        evaluation_name='SWE-bench Verified',
        source_data=SourceDataUrl(
            dataset_name='SWE-bench Verified',
            source_type='url',
            url=['https://www.swebench.com'],
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description=(
                'Fraction of 500 verified GitHub issues resolved (0.0–1.0)'
            ),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=ScoreDetails(
            score=score,
            details=score_details,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    available_tools=[AvailableTool(name='bash')],
                ),
            ),
        ),
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=evaluation_timestamp,
        source_metadata=SourceMetadata(
            source_name='SWE-bench Verified Leaderboard',
            source_type='documentation',
            source_organization_name='SWE-bench',
            source_organization_url='https://www.swebench.com',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name='swe-bench', version='unknown'),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def convert_submissions(
    submissions: list[Path],
    retrieved_timestamp: str,
    total_instances: int,
    output_dir: str | Path = OUTPUT_DIR,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert usable submissions and retain each rejected source path."""
    outputs = []
    failures = []
    if not submissions:
        failures.append(
            SourceRecordFailure(
                source_ref='SWE-bench Verified submission discovery',
                reason='no submission directories found',
            )
        )
    for submission_dir in submissions:
        try:
            eval_log = convert_submission(
                submission_dir,
                retrieved_timestamp,
                total_instances,
            )
            model_id = require_identity(
                eval_log.model_info.id,
                'SWE-bench Verified model id',
            )
            if '/' not in model_id:
                raise ValueError(
                    f'model id must be developer/model: {model_id!r}'
                )
            developer, model_name = model_id.split('/', 1)
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=output_dir,
                    developer=developer,
                    model_name=model_name,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=str(submission_dir),
                    reason=str(exc),
                    source_record={
                        'submission_dir': str(submission_dir),
                    },
                )
            )
    return SourceConversionResult(
        source_name='SWE-bench Verified',
        total_records=len(submissions),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert the SWE-bench Verified leaderboard to EEE records.'
        )
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Datastore collection directory (default: {OUTPUT_DIR}).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = args.output_dir
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            'datasets is required to run this adapter. Install it with: uv add datasets'
        ) from e

    retrieved_timestamp = str(time.time())
    raw_capture.record_hf_dataset('SWE-bench/SWE-bench_Verified')
    ds = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
    total_instances = len(ds)
    if total_instances == 0:
        raise ValueError(
            'SWE-bench/SWE-bench_Verified returned zero test instances'
        )
    print(
        f'Loaded {total_instances} instances from SWE-bench/SWE-bench_Verified\n'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Cloning {SWE_BENCH_REPO} into {tmpdir} ...')
        subprocess.run(
            ['git', 'clone', '--depth=1', SWE_BENCH_REPO, tmpdir],
            check=True,
        )
        raw_capture.record_git_checkout(SWE_BENCH_REPO, tmpdir)

        swe_bench_path = Path(tmpdir) / SWE_BENCH_SUBDIR
        submissions = sorted(d for d in swe_bench_path.iterdir() if d.is_dir())
        print(f'Found {len(submissions)} submission directories\n')

        result = convert_submissions(
            submissions,
            retrieved_timestamp,
            total_instances,
            output_dir,
        )
        paths = save_evaluation_logs(result.records)
        for path in paths:
            print(f'  Saved: {path}')
        if result.failures:
            report_path = save_failure_report(
                result,
                default_failure_report_path(output_dir),
            )
            print(f'Failure report: {report_path}')

    print(
        f'\nGenerated {len(paths)} files, {len(result.failures)} errors '
        f'→ {output_dir}/'
    )
    result.raise_if_incomplete()


if __name__ == '__main__':
    main()
