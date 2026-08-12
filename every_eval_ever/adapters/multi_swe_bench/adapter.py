"""
Script to convert Multi-SWE-Bench leaderboard data to the EvalEval schema format.

Data source:
- Multi-SWE-bench experiments repo: https://github.com/multi-swe-bench/experiments
  Cloned to a temporary directory at runtime; cleaned up on exit.

Each subdirectory under evaluation/<lang>/verified/ is a submission with:
  - metadata.yaml: name, orgIcon, oss, site, verified
  - results/results.json: resolved/unresolved/etc instance lists

Score = len(resolved) / total_instances  (from results.json)

Usage:
    uv run python -m every_eval_ever.adapters.multi_swe_bench.adapter
    uv run python -m every_eval_ever.adapters.multi_swe_bench.adapter         --output-dir /tmp/smoke/data/multi-swe-bench-leaderboard
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from every_eval_ever.adapters.swe_helpers import (
    parse_date_from_dir,
    parse_model_from_dir,
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

MULTI_SWE_REPO = 'https://github.com/multi-swe-bench/experiments'
LANGUAGES = ['c', 'c++', 'go', 'java', 'javascript', 'rust', 'typescript']
OUTPUT_BASE = 'data/multi-swe-bench-leaderboard'


def convert_submission(
    submission_dir: Path,
    lang: str,
    retrieved_timestamp: str,
) -> EvaluationLog:
    """Convert a single Multi-SWE-Bench submission directory to an EvaluationLog."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            'pyyaml is required to run this adapter. Install it with: pip install pyyaml'
        ) from e

    dir_name = submission_dir.name

    with open(submission_dir / 'metadata.yaml', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f'metadata.yaml is not an object for {dir_name}')

    with open(
        submission_dir / 'results' / 'results.json',
        encoding='utf-8',
    ) as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise ValueError(f'results.json is not an object for {dir_name}')

    total_instances = int(results.get('total_instances', 0))
    if total_instances <= 0:
        raise ValueError(f'total_instances must be positive for {dir_name}')

    resolved = results.get('resolved', [])
    if not isinstance(resolved, list):
        raise ValueError(f'resolved must be a list for {dir_name}')
    if len(resolved) > total_instances:
        raise ValueError(
            f'resolved count exceeds total_instances for {dir_name}'
        )
    score = len(resolved) / total_instances

    agent, primary_model = parse_model_from_dir(dir_name)
    primary_model = require_identity(primary_model, 'Multi-SWE-bench model')
    developer = require_identity(
        get_developer(primary_model),
        'Multi-SWE-bench model developer',
    )
    model_id = get_model_id(primary_model, developer)

    sanitized_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id.replace('/', '_'))
    submission_slug = re.sub(r'[^a-zA-Z0-9_.-]', '_', dir_name)
    eval_id = f'multi-swe-bench/{lang}/{sanitized_id}/{submission_slug}/{retrieved_timestamp}'

    evaluation_timestamp = parse_date_from_dir(dir_name)

    additional_details: dict[str, str] = {
        'submission_name': str(metadata.get('name', '')),
        'language': lang,
        'oss': str(metadata.get('oss', '')),
        'site': str(metadata.get('site', '')),
        'verified': str(metadata.get('verified', '')),
        'submission_dir': dir_name,
        'agent': agent,
    }

    score_details: dict[str, str] = {
        'resolved_count': str(len(resolved)),
        'total_instances': str(total_instances),
        'submitted_instances': str(results.get('submitted_instances', '')),
        'completed_instances': str(results.get('completed_instances', '')),
        'unresolved_instances': str(results.get('unresolved_instances', '')),
        'empty_error_patch_instances': str(
            results.get('empty_error_patch_instances', '')
        ),
    }

    dataset_label = f'Multi-SWE-bench ({lang})'
    eval_name = f'Multi-SWE-Bench ({lang})'

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataUrl(
            dataset_name=dataset_label,
            source_type='url',
            url=[
                'https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench'
            ],
        ),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=MetricConfig(
            evaluation_description=f'Fraction of {lang} GitHub issues resolved (0.0–1.0)',
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
            source_name='Multi-SWE-Bench Leaderboard',
            source_type='documentation',
            source_organization_name='ByteDance-Seed',
            source_organization_url='https://github.com/multi-swe-bench/experiments',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name='multi-swe-bench', version='unknown'),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def convert_submissions(
    submissions: list[tuple[Path, str]],
    retrieved_timestamp: str,
    output_dir: str | Path = OUTPUT_BASE,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert all usable submissions and retain rejected source paths."""
    outputs = []
    failures = []
    if not submissions:
        failures.append(
            SourceRecordFailure(
                source_ref='Multi-SWE-bench submission discovery',
                reason='no submission directories found',
            )
        )
    for submission_dir, lang in submissions:
        try:
            eval_log = convert_submission(
                submission_dir,
                lang,
                retrieved_timestamp,
            )
            model_id = require_identity(
                eval_log.model_info.id,
                'Multi-SWE-bench model id',
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
                        'language': lang,
                    },
                )
            )
    return SourceConversionResult(
        source_name='Multi-SWE-bench',
        total_records=len(submissions),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert the Multi-SWE-bench leaderboard to EEE records.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_BASE),
        help=f'Datastore collection directory (default: {OUTPUT_BASE}).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = args.output_dir
    retrieved_timestamp = str(time.time())

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Cloning {MULTI_SWE_REPO} into {tmpdir} ...')
        subprocess.run(
            ['git', 'clone', '--depth=1', MULTI_SWE_REPO, tmpdir],
            env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'},
            check=True,
        )
        raw_capture.record_git_checkout(MULTI_SWE_REPO, tmpdir)

        source_submissions = []
        source_failures = []
        for lang in LANGUAGES:
            verified_path = Path(tmpdir) / 'evaluation' / lang / 'verified'
            if not verified_path.exists():
                source_failures.append(
                    SourceRecordFailure(
                        source_ref=str(verified_path),
                        reason='expected verified submission directory is missing',
                        source_record={'language': lang},
                    )
                )
                continue

            submissions = sorted(
                d for d in verified_path.iterdir() if d.is_dir()
            )
            print(f'\n[{lang}] Found {len(submissions)} submissions')
            source_submissions.extend(
                (submission_dir, lang) for submission_dir in submissions
            )

        converted = convert_submissions(
            source_submissions,
            retrieved_timestamp,
            output_dir,
        )
        result = SourceConversionResult(
            source_name='Multi-SWE-bench',
            total_records=(converted.total_records + len(source_failures)),
            records=converted.records,
            failures=[*source_failures, *converted.failures],
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
