#!/usr/bin/env python3
"""Convert public tau-bench leaderboard submissions into EEE records.

Data source:
- tau-bench leaderboard: https://taubench.com
- Static submissions JSON:
  https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions

The adapter emits one ``EvaluationLog`` per tau-bench submission. Each log
contains one ``EvaluationResult`` per populated domain metric, for example
``tau_bench.text.retail.pass_1`` or
``tau_bench.text.banking_knowledge.cost``.

Usage:
    uv run python -m utils.tau_bench.adapter --output-dir data/tau-bench
    uv run python -m utils.tau_bench.adapter \\
        --input-dir /tmp/tau2-submissions --output-dir /tmp/eee-tau-bench
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    SourceType,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    fetch_json,
    sanitize_filename,
    save_evaluation_log,
)

SOURCE_NAME = 'tau-bench Leaderboard'
SOURCE_ORGANIZATION = 'Sierra'
SOURCE_ORGANIZATION_URL = 'https://taubench.com'
LEADERBOARD_URL = 'https://taubench.com'
SUBMISSIONS_TREE_URL = (
    'https://github.com/sierra-research/tau2-bench/tree/main/'
    'web/leaderboard/public/submissions'
)
RAW_SUBMISSIONS_BASE_URL = (
    'https://raw.githubusercontent.com/sierra-research/tau2-bench/main/'
    'web/leaderboard/public/submissions'
)
DEFAULT_OUTPUT_DIR = 'data/tau-bench'

MANIFEST_FILE_NAME = 'manifest.json'
SUBMISSION_FILE_NAME = 'submission.json'
MANIFEST_SECTIONS = (
    'submissions',
    'voice_submissions',
    'legacy_submissions',
)
DOMAINS = ('retail', 'airline', 'telecom', 'banking_knowledge')
PASS_METRICS = ('pass_1', 'pass_2', 'pass_3', 'pass_4')

ORGANIZATION_SLUGS = {
    'Alibaba Cloud': 'alibaba',
    'Anthropic': 'anthropic',
    'DeepSeek': 'deepseek',
    'Distyl AI': 'distyl',
    'Google': 'google',
    'Moonshot AI': 'moonshot-ai',
    'Multiple providers': 'multiple',
    'NVIDIA': 'nvidia',
    'OpenAI': 'openai',
    'Qwen': 'qwen',
    'Sierra': 'sierra',
    'xAI': 'xai',
    'Zhipu AI': 'zhipu-ai',
}


@dataclass(frozen=True)
class TauBenchSubmission:
    submission_id: str
    manifest_section: str
    submission: dict[str, Any]
    source_url: str


@dataclass(frozen=True)
class EvaluationBundle:
    log: EvaluationLog
    developer: str
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert tau-bench leaderboard JSON into EEE records.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--base-url',
        default=RAW_SUBMISSIONS_BASE_URL,
        help='Base URL containing manifest.json and submission folders.',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help=(
            'Read a local tau-bench submissions directory instead of '
            'fetching from --base-url.'
        ),
    )
    parser.add_argument(
        '--sections',
        nargs='+',
        choices=MANIFEST_SECTIONS,
        default=list(MANIFEST_SECTIONS),
        help='Manifest sections to export.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Optional maximum number of submissions to export.',
    )
    return parser.parse_args()


def load_submissions(
    *,
    input_dir: Path | None = None,
    base_url: str = RAW_SUBMISSIONS_BASE_URL,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
) -> list[TauBenchSubmission]:
    if input_dir is not None:
        return load_submissions_from_dir(input_dir, sections)
    return load_submissions_from_url(base_url, sections)


def load_submissions_from_url(
    base_url: str,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
) -> list[TauBenchSubmission]:
    base_url = base_url.rstrip('/')
    manifest = fetch_json(f'{base_url}/{MANIFEST_FILE_NAME}')
    records = []
    for section, submission_id in iter_manifest_ids(manifest, sections):
        source_url = f'{base_url}/{submission_id}/{SUBMISSION_FILE_NAME}'
        submission = fetch_json(source_url)
        records.append(
            TauBenchSubmission(
                submission_id=submission_id,
                manifest_section=section,
                submission=submission,
                source_url=source_url,
            )
        )
    return records


def load_submissions_from_dir(
    input_dir: Path,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
) -> list[TauBenchSubmission]:
    manifest_path = input_dir / MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = []
    for section, submission_id in iter_manifest_ids(manifest, sections):
        path = input_dir / submission_id / SUBMISSION_FILE_NAME
        submission = json.loads(path.read_text(encoding='utf-8'))
        records.append(
            TauBenchSubmission(
                submission_id=submission_id,
                manifest_section=section,
                submission=submission,
                source_url=submission_source_url(submission_id),
            )
        )
    return records


def iter_manifest_ids(
    manifest: dict[str, Any],
    sections: list[str] | tuple[str, ...],
) -> list[tuple[str, str]]:
    pairs = []
    seen: set[str] = set()
    for section in sections:
        values = manifest.get(section) or []
        if not isinstance(values, list):
            raise ValueError(f'Manifest section {section!r} must be a list')
        for value in values:
            submission_id = str(value)
            if submission_id in seen:
                continue
            seen.add(submission_id)
            pairs.append((section, submission_id))
    return pairs


def make_logs(
    records: list[TauBenchSubmission],
    *,
    retrieved_timestamp: str | None = None,
) -> list[EvaluationBundle]:
    retrieved_timestamp = retrieved_timestamp or str(time.time())
    bundles = []
    for record in records:
        bundle = make_log(record, retrieved_timestamp)
        if bundle is not None:
            bundles.append(bundle)
    return bundles


def make_log(
    record: TauBenchSubmission,
    retrieved_timestamp: str,
) -> EvaluationBundle | None:
    submission = record.submission
    model_name = required_str(submission, 'model_name', record.submission_id)
    model_org = required_str(
        submission, 'model_organization', record.submission_id
    )
    developer = organization_slug(model_org)
    model_slug = slugify(model_name)
    model_id = f'{developer}/{model_slug}'
    results = make_results(record, model_id)
    if not results:
        return None

    evaluation_timestamp = evaluation_date(submission)
    version = (
        (submission.get('methodology') or {}).get('tau2_bench_version')
    ) or 'unknown'
    sanitized_model_id = sanitize_filename(model_id)
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'tau-bench/{sanitized_model_id}/'
            f'{record.submission_id}/{retrieved_timestamp}'
        ),
        evaluation_timestamp=evaluation_timestamp,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=make_source_metadata(record),
        eval_library=EvalLibrary(
            name='tau2-bench',
            version=str(version),
            additional_details=_clean_details(
                {
                    'leaderboard_url': LEADERBOARD_URL,
                    'submissions_tree_url': SUBMISSIONS_TREE_URL,
                }
            ),
        ),
        model_info=ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
            additional_details=make_model_details(record),
        ),
        evaluation_results=results,
    )
    return EvaluationBundle(log=log, developer=developer, model_name=model_slug)


def make_results(
    record: TauBenchSubmission,
    model_id: str,
) -> list[EvaluationResult]:
    submission = record.submission
    all_results = submission.get('results') or {}
    if not isinstance(all_results, dict):
        raise ValueError(
            f'{record.submission_id} has invalid results: expected object'
        )

    results = []
    for domain in DOMAINS:
        domain_results = all_results.get(domain)
        if domain_results is None:
            continue
        if not isinstance(domain_results, dict):
            raise ValueError(
                f'{record.submission_id}/{domain} results must be an object'
            )

        for metric in PASS_METRICS:
            score = parse_score(
                domain_results.get(metric),
                context=f'{record.submission_id}/{domain}/{metric}',
            )
            if score is None:
                continue
            results.append(
                make_result(
                    record,
                    model_id=model_id,
                    domain=domain,
                    metric=metric,
                    score=score,
                    domain_results=domain_results,
                )
            )

        cost = parse_score(
            domain_results.get('cost'),
            context=f'{record.submission_id}/{domain}/cost',
        )
        if cost is not None:
            results.append(
                make_result(
                    record,
                    model_id=model_id,
                    domain=domain,
                    metric='cost',
                    score=cost,
                    domain_results=domain_results,
                )
            )
    return results


def make_result(
    record: TauBenchSubmission,
    *,
    model_id: str,
    domain: str,
    metric: str,
    score: float,
    domain_results: dict[str, Any],
) -> EvaluationResult:
    submission = record.submission
    modality = str(submission.get('modality') or 'text')
    metric_config = make_metric_config(domain=domain, metric=metric)
    evaluation_name = f'tau_bench.{modality}.{domain}.{metric}'

    return EvaluationResult(
        evaluation_result_id=(
            f'tau_bench:{record.submission_id}:{domain}:{metric}'
        ),
        evaluation_name=evaluation_name,
        source_data=make_source_data(record, domain, domain_results),
        evaluation_timestamp=evaluation_date(submission),
        metric_config=metric_config,
        score_details=ScoreDetails(
            score=score,
            details=_clean_details(
                {
                    'submission_id': record.submission_id,
                    'model_id': model_id,
                    'domain': domain,
                    'metric': metric,
                    'raw_score': domain_results.get(metric),
                    'retrieval_config': domain_results.get('retrieval_config'),
                }
            ),
        ),
        generation_config=make_generation_config(submission, domain),
    )


def make_metric_config(*, domain: str, metric: str) -> MetricConfig:
    if metric.startswith('pass_'):
        k = int(metric.split('_', 1)[1])
        return MetricConfig(
            evaluation_description=(
                f'tau-bench {domain} Pass^{k} success rate reported on '
                'the public leaderboard.'
            ),
            metric_id='tau_bench.pass_at_k',
            metric_name=f'Pass^{k}',
            metric_kind='pass_rate',
            metric_unit='percent',
            metric_parameters={'k': k},
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
            additional_details=_clean_details(
                {
                    'domain': domain,
                    'score_scale': 'percent_0_to_100',
                }
            ),
        )

    if metric == 'cost':
        return MetricConfig(
            evaluation_description=(
                f'Average tau-bench cost per trajectory for {domain}, in USD, '
                'when reported by the submission.'
            ),
            metric_id='tau_bench.cost_per_trajectory',
            metric_name='Cost per trajectory',
            metric_kind='cost',
            metric_unit='usd_per_trajectory',
            lower_is_better=True,
            additional_details=_clean_details({'domain': domain}),
        )

    raise ValueError(f'Unsupported tau-bench metric: {metric}')


def make_source_data(
    record: TauBenchSubmission,
    domain: str,
    domain_results: dict[str, Any],
) -> SourceDataUrl:
    return SourceDataUrl(
        dataset_name=f'tau-bench {domain}',
        source_type='url',
        url=[LEADERBOARD_URL, record.source_url],
        additional_details=_clean_details(
            {
                'domain': domain,
                'submission_id': record.submission_id,
                'manifest_section': record.manifest_section,
                'retrieval_config': domain_results.get('retrieval_config'),
                'trajectory_file': (
                    (record.submission.get('trajectory_files') or {}).get(
                        domain
                    )
                    if isinstance(
                        record.submission.get('trajectory_files'), dict
                    )
                    else None
                ),
            }
        ),
    )


def make_source_metadata(record: TauBenchSubmission) -> SourceMetadata:
    submission = record.submission
    return SourceMetadata(
        source_name=SOURCE_NAME,
        source_type=SourceType.documentation,
        source_organization_name=SOURCE_ORGANIZATION,
        source_organization_url=SOURCE_ORGANIZATION_URL,
        evaluator_relationship=evaluator_relationship(submission),
        additional_details=_clean_details(
            {
                'leaderboard_url': LEADERBOARD_URL,
                'submissions_tree_url': SUBMISSIONS_TREE_URL,
                'submission_source_url': record.source_url,
                'submission_id': record.submission_id,
                'manifest_section': record.manifest_section,
                'submission_date': submission.get('submission_date'),
                'submission_type': submission.get('submission_type'),
                'modality': submission.get('modality') or 'text',
                'submitting_organization': submission.get(
                    'submitting_organization'
                ),
            }
        ),
    )


def make_model_details(
    record: TauBenchSubmission,
) -> dict[str, str] | None:
    submission = record.submission
    model_release = submission.get('model_release') or {}
    references = submission.get('references') or []
    return _clean_details(
        {
            'raw_model_organization': submission.get('model_organization'),
            'submitting_organization': submission.get(
                'submitting_organization'
            ),
            'submission_id': record.submission_id,
            'submission_date': submission.get('submission_date'),
            'submission_type': submission.get('submission_type'),
            'modality': submission.get('modality') or 'text',
            'is_new': submission.get('is_new'),
            'trajectories_available': submission.get('trajectories_available'),
            'reasoning_effort': submission.get('reasoning_effort'),
            'model_release_date': model_release.get('release_date'),
            'model_release_announcement_url': model_release.get(
                'announcement_url'
            ),
            'references': references or None,
        }
    )


def make_generation_config(
    submission: dict[str, Any],
    domain: str,
) -> GenerationConfig:
    methodology = submission.get('methodology') or {}
    voice_config = submission.get('voice_config') or {}
    pipeline = (
        voice_config.get('pipeline') if isinstance(voice_config, dict) else None
    )
    return GenerationConfig(
        generation_args=GenerationArgs(
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(
                        name=f'tau-bench {domain} tools',
                        description=(
                            'Domain-specific customer service tools exposed '
                            'by the tau-bench environment.'
                        ),
                    )
                ],
                additional_details=_clean_details({'domain': domain}),
            )
        ),
        additional_details=_clean_details(
            {
                'evaluation_date': methodology.get('evaluation_date'),
                'tau2_bench_version': methodology.get('tau2_bench_version'),
                'user_simulator': methodology.get('user_simulator'),
                'methodology_notes': methodology.get('notes'),
                'verification': methodology.get('verification'),
                'submission_type': submission.get('submission_type'),
                'modality': submission.get('modality') or 'text',
                'reasoning_effort': submission.get('reasoning_effort'),
                'voice_provider': voice_config.get('provider')
                if isinstance(voice_config, dict)
                else None,
                'voice_model': voice_config.get('model')
                if isinstance(voice_config, dict)
                else None,
                'voice_tick_duration_seconds': voice_config.get(
                    'tick_duration_seconds'
                )
                if isinstance(voice_config, dict)
                else None,
                'voice_max_steps_seconds': voice_config.get('max_steps_seconds')
                if isinstance(voice_config, dict)
                else None,
                'voice_user_tts_provider': voice_config.get('user_tts_provider')
                if isinstance(voice_config, dict)
                else None,
                'voice_pipeline': pipeline,
            }
        ),
    )


def evaluator_relationship(
    submission: dict[str, Any],
) -> EvaluatorRelationship:
    model_org = slugify(str(submission.get('model_organization') or ''))
    submitter = slugify(str(submission.get('submitting_organization') or ''))
    if model_org and submitter and model_org == submitter:
        return EvaluatorRelationship.first_party
    return EvaluatorRelationship.third_party


def evaluation_date(submission: dict[str, Any]) -> str | None:
    methodology = submission.get('methodology') or {}
    value = methodology.get('evaluation_date') or submission.get(
        'submission_date'
    )
    return str(value) if value else None


def required_str(
    payload: dict[str, Any],
    key: str,
    submission_id: str,
) -> str:
    value = payload.get(key)
    if value is None or str(value).strip() == '':
        raise ValueError(f'{submission_id} is missing required field {key}')
    return str(value)


def parse_score(raw: Any, *, context: str) -> float | None:
    if raw is None or raw == '':
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Non-numeric tau-bench score for {context}: {raw!r}'
        ) from exc


def organization_slug(name: str) -> str:
    return ORGANIZATION_SLUGS.get(name, slugify(name))


def slugify(value: str) -> str:
    base = re.sub(r'[^\w.\-]+', '-', value.strip().lower())
    base = re.sub(r'-{2,}', '-', base).strip('-')
    return sanitize_filename(base) or 'unknown'


def submission_source_url(submission_id: str) -> str:
    return f'{RAW_SUBMISSIONS_BASE_URL}/{submission_id}/{SUBMISSION_FILE_NAME}'


def export_logs(
    bundles: list[EvaluationBundle],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    paths = []
    for bundle in bundles:
        paths.append(
            save_evaluation_log(
                bundle.log,
                output_dir,
                bundle.developer,
                bundle.model_name,
            )
        )
    return paths


def _clean_details(values: dict[str, Any]) -> dict[str, str] | None:
    details = {
        key: _detail_value(value)
        for key, value in values.items()
        if value is not None
    }
    return details or None


def _detail_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def run(args: argparse.Namespace) -> int:
    records = load_submissions(
        input_dir=args.input_dir,
        base_url=args.base_url,
        sections=args.sections,
    )
    if args.limit is not None:
        records = records[: args.limit]
    bundles = make_logs(records)
    paths = export_logs(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} tau-bench submission log(s).')
