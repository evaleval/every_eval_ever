"""
Script to convert SWE-PolyBench leaderboard data to the EvalEval schema format.

Data source:
- SWE-PolyBench experiments repo: https://github.com/amazon-science/SWE-PolyBench
  Branch: submission. Cloned to a temporary directory at runtime; cleaned up on exit.
- HF datasets: AmazonScience/SWE-PolyBench (PB) and AmazonScience/SWE-PolyBench_Verified (PBVerified).

Each subdirectory under evaluation/{PB,PBVerified}/ is a submission with:
  - metadata.yaml: name, oss, site, pass_rate, logo
  - logs/<instance_id>_result.json: per-instance resolved status

Score = resolved_count_in_lang / total_instances_for_lang_from_hf_dataset

One EvaluationLog is written per (dataset x submission x language).

Usage:
    cd every_eval_ever
    .venv/bin/python -m every_eval_ever.adapters.swe_polybench.adapter
"""

import json
import re
import subprocess
import tempfile
import time
from collections import Counter
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
    SourceDataHf,
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
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import require_identity

POLY_REPO = 'https://github.com/amazon-science/SWE-PolyBench'
POLY_BRANCH = 'submission'

DATASETS = {
    'PB': 'AmazonScience/SWE-PolyBench',
    'PBVerified': 'AmazonScience/SWE-PolyBench_Verified',
}
DATASET_LABELS = {'PB': 'pb', 'PBVerified': 'pb-verified'}
DATASET_DISPLAY = {
    'PB': 'SWE-PolyBench',
    'PBVerified': 'SWE-PolyBench Verified',
}

OUTPUT_BASE = 'data/swe-polybench-leaderboard'


def convert_submission(
    submission_dir: Path,
    ds: str,
    lang: str,
    resolved_count: int,
    patch_applied_count: int,
    no_p2p_failed_count: int,
    total_instances_for_lang: int,
    retrieved_timestamp: str,
    metadata: dict,
) -> EvaluationLog:
    dir_name = submission_dir.name
    ds_label = DATASET_LABELS[ds]
    ds_display = DATASET_DISPLAY[ds]
    hf_repo = DATASETS[ds]

    agent, primary_model = parse_model_from_dir(dir_name)
    primary_model = require_identity(primary_model, 'SWE-PolyBench model')
    developer = require_identity(
        get_developer(primary_model),
        'SWE-PolyBench model developer',
    )
    model_id = get_model_id(primary_model, developer)

    sanitized_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', model_id.replace('/', '_'))
    submission_slug = re.sub(r'[^a-zA-Z0-9_.-]', '_', dir_name)
    eval_id = f'swe-polybench/{ds_label}/{lang}/{sanitized_id}/{submission_slug}/{retrieved_timestamp}'

    evaluation_timestamp = parse_date_from_dir(dir_name)
    if total_instances_for_lang <= 0:
        raise ValueError(
            f'total instances must be positive for language {lang!r}'
        )
    if resolved_count > total_instances_for_lang:
        raise ValueError(
            f'resolved count exceeds total instances for language {lang!r}'
        )
    score = resolved_count / total_instances_for_lang

    additional_details: dict[str, str] = {
        'submission_name': str(metadata.get('name', '')),
        'language': lang,
        'dataset': ds_label,
        'oss': str(metadata.get('oss', '')),
        'site': str(metadata.get('site', '')),
        'pass_rate': str(metadata.get('pass_rate', '')),
        'submission_dir': dir_name,
        'agent': agent,
    }

    score_details: dict[str, str] = {
        'resolved_count': str(resolved_count),
        'total_instances_for_language': str(total_instances_for_lang),
        'patch_applied_count': str(patch_applied_count),
        'no_p2p_failed_count': str(no_p2p_failed_count),
    }

    eval_name = f'{ds_display} ({lang})'
    dataset_label = f'{ds_display} ({lang})'

    eval_result = EvaluationResult(
        evaluation_name=eval_name,
        source_data=SourceDataHf(
            dataset_name=dataset_label,
            source_type='hf_dataset',
            hf_repo=hf_repo,
            hf_split='test',
            samples_number=total_instances_for_lang,
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
            source_name='SWE-PolyBench Leaderboard',
            source_type='documentation',
            source_organization_name='AmazonScience',
            source_organization_url='https://github.com/amazon-science/SWE-PolyBench',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name='swe-polybench', version='unknown'),
        model_info=ModelInfo(
            name=primary_model,
            id=model_id,
            developer=developer,
            additional_details=additional_details,
        ),
        evaluation_results=[eval_result],
    )


def load_hf_instance_maps(ds: str) -> tuple[dict[str, str], Counter]:
    """Return (instance_id -> language, Counter of language totals) from HF dataset."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            'datasets is required to run this adapter. Install it with: pip install datasets'
        ) from e

    hf_repo = DATASETS[ds]
    print(f'  Loading HF dataset {hf_repo} ...')
    dataset = load_dataset(hf_repo, split='test')
    id_to_lang: dict[str, str] = {}
    lang_counts: Counter = Counter()
    for row in dataset:
        iid = require_identity(
            row.get('instance_id'),
            f'{hf_repo} instance id',
        )
        lang = require_identity(
            row.get('language'),
            f'{hf_repo} instance language',
        )
        if iid in id_to_lang and id_to_lang[iid] != lang:
            raise ValueError(
                f'{hf_repo} instance {iid!r} has conflicting languages '
                f'{id_to_lang[iid]!r} and {lang!r}'
            )
        id_to_lang[iid] = lang
        lang_counts[lang] += 1
    if not id_to_lang:
        raise ValueError(f'{hf_repo} returned zero test instances')
    return id_to_lang, lang_counts


def process_submission_result(
    submission_dir: Path,
    ds: str,
    id_to_lang: dict[str, str],
    lang_counts: Counter,
    retrieved_timestamp: str,
    yaml,
) -> SourceConversionResult[tuple[EvaluationLog, str]]:
    """Convert known instances and retain per-file failures."""
    dir_name = submission_dir.name
    metadata_path = submission_dir / 'metadata.yaml'
    if not metadata_path.exists():
        return SourceConversionResult(
            source_name=f'SWE-PolyBench {ds} {dir_name}',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=str(metadata_path),
                    reason='metadata.yaml not found',
                )
            ],
        )

    try:
        with open(metadata_path, encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        if not isinstance(metadata, dict):
            raise ValueError('metadata.yaml must contain an object')
    except Exception as exc:
        return SourceConversionResult(
            source_name=f'SWE-PolyBench {ds} {dir_name}',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=str(metadata_path),
                    reason=str(exc),
                )
            ],
        )

    logs_dir = submission_dir / 'logs'
    if not logs_dir.exists():
        return SourceConversionResult(
            source_name=f'SWE-PolyBench {ds} {dir_name}',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=str(logs_dir),
                    reason='logs directory not found',
                    source_record=metadata,
                )
            ],
        )

    result_files = sorted(logs_dir.glob('*_result.json'))
    if not result_files:
        return SourceConversionResult(
            source_name=f'SWE-PolyBench {ds} {dir_name}',
            total_records=0,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=str(logs_dir),
                    reason='no *_result.json files found',
                    source_record=metadata,
                )
            ],
        )

    # Aggregate per language
    langs_in_submission: set[str] = set()
    resolved_by_lang: Counter = Counter()
    patch_applied_by_lang: Counter = Counter()
    no_p2p_failed_by_lang: Counter = Counter()

    failures = []
    for result_file in result_files:
        data = None
        try:
            with open(result_file, encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('result file must contain an object')
            iid = require_identity(
                data.get('instance_id'),
                'SWE-PolyBench result instance id',
            )
            lang = id_to_lang.get(iid)
            if lang is None:
                raise ValueError(
                    f'instance id {iid!r} is not present in {DATASETS[ds]}'
                )
            langs_in_submission.add(lang)
            if data.get('resolved', False):
                resolved_by_lang[lang] += 1
            if data.get('patch_applied', False):
                patch_applied_by_lang[lang] += 1
            if data.get('no_p2p_failed', False):
                no_p2p_failed_by_lang[lang] += 1
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=str(result_file),
                    reason=str(exc),
                    source_record=data,
                )
            )

    # Only emit records for languages actually present in this submission's result
    # files, to avoid spurious 0-score entries for uncovered languages.
    results = []
    for lang in langs_in_submission:
        total = lang_counts.get(lang)
        if total is None or total <= 0:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{submission_dir} language {lang!r}',
                    reason='language has no denominator in source dataset',
                    source_record={'language': lang},
                )
            )
            continue
        try:
            eval_log = convert_submission(
                submission_dir=submission_dir,
                ds=ds,
                lang=lang,
                resolved_count=resolved_by_lang.get(lang, 0),
                patch_applied_count=patch_applied_by_lang.get(lang, 0),
                no_p2p_failed_count=no_p2p_failed_by_lang.get(lang, 0),
                total_instances_for_lang=total,
                retrieved_timestamp=retrieved_timestamp,
                metadata=metadata,
            )
            results.append((eval_log, lang))
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{submission_dir} language {lang!r}',
                    reason=str(exc),
                    source_record={
                        'metadata': metadata,
                        'language': lang,
                    },
                )
            )
    return SourceConversionResult(
        source_name=f'SWE-PolyBench {ds} {dir_name}',
        total_records=len(result_files),
        records=results,
        failures=failures,
    )


def process_submission(
    submission_dir: Path,
    ds: str,
    id_to_lang: dict[str, str],
    lang_counts: Counter,
    retrieved_timestamp: str,
    yaml,
) -> list[tuple[EvaluationLog, str]]:
    """Strict API for callers that require every result file to convert."""
    result = process_submission_result(
        submission_dir,
        ds,
        id_to_lang,
        lang_counts,
        retrieved_timestamp,
        yaml,
    )
    result.raise_if_incomplete()
    return result.records


def main():
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            'pyyaml is required to run this adapter. Install it with: pip install pyyaml'
        ) from e

    retrieved_timestamp = str(time.time())
    # Load HF datasets first
    hf_maps: dict[str, tuple[dict[str, str], Counter]] = {}
    for ds in ('PB', 'PBVerified'):
        id_to_lang, lang_counts = load_hf_instance_maps(ds)
        hf_maps[ds] = (id_to_lang, lang_counts)
        print(
            f'  [{ds}] {sum(lang_counts.values())} instances: {dict(lang_counts)}'
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'\nCloning {POLY_REPO} (branch={POLY_BRANCH}) into {tmpdir} ...')
        subprocess.run(
            [
                'git',
                'clone',
                '--branch',
                POLY_BRANCH,
                '--depth=1',
                POLY_REPO,
                tmpdir,
            ],
            check=True,
        )

        outputs = []
        failures = []
        total_records = 0
        for ds in ('PB', 'PBVerified'):
            eval_path = Path(tmpdir) / 'evaluation' / ds
            if not eval_path.exists():
                total_records += 1
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(eval_path),
                        reason='expected evaluation dataset directory is missing',
                        source_record={'dataset': ds},
                    )
                )
                continue

            id_to_lang, lang_counts = hf_maps[ds]
            submissions = sorted(d for d in eval_path.iterdir() if d.is_dir())
            print(f'\n[{ds}] Found {len(submissions)} submissions')

            for submission_dir in submissions:
                converted = process_submission_result(
                    submission_dir,
                    ds,
                    id_to_lang,
                    lang_counts,
                    retrieved_timestamp,
                    yaml,
                )
                total_records += converted.total_records
                failures.extend(converted.failures)
                for eval_log, lang in converted.records:
                    try:
                        model_id = require_identity(
                            eval_log.model_info.id,
                            'SWE-PolyBench model id',
                        )
                        if '/' not in model_id:
                            raise ValueError(
                                'model id must be developer/model: '
                                f'{model_id!r}'
                            )
                        developer, model_name = model_id.split('/', 1)
                        outputs.append(
                            EvaluationLogOutput(
                                eval_log=eval_log,
                                base_dir=OUTPUT_BASE,
                                developer=developer,
                                model_name=model_name,
                            )
                        )
                    except Exception as exc:
                        failures.append(
                            SourceRecordFailure(
                                source_ref=(
                                    f'{submission_dir} language {lang!r}'
                                ),
                                reason=str(exc),
                                source_record={
                                    'submission_dir': str(submission_dir),
                                    'dataset': ds,
                                    'language': lang,
                                },
                            )
                        )

        if not outputs and not failures:
            failures.append(
                SourceRecordFailure(
                    source_ref='SWE-PolyBench submission discovery',
                    reason='no submission result files found',
                )
            )
        result = SourceConversionResult(
            source_name='SWE-PolyBench',
            total_records=total_records,
            records=outputs,
            failures=failures,
        )
        paths = save_evaluation_logs(result.records)
        for path in paths:
            print(f'  Saved: {path}')
        if result.failures:
            report_path = save_failure_report(
                result,
                default_failure_report_path(OUTPUT_BASE),
            )
            print(f'Failure report: {report_path}')

    print(
        f'\nGenerated {len(paths)} files, {len(result.failures)} errors '
        f'→ {OUTPUT_BASE}/'
    )
    result.raise_if_incomplete()


if __name__ == '__main__':
    main()
