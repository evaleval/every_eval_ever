#!/usr/bin/env python3
"""Convert the TIGER-Lab MMLU-Pro leaderboard into Every Eval Ever records.

Data source:
- TIGER-Lab leaderboard CSV hosted as a Hugging Face dataset:
  https://huggingface.co/datasets/TIGER-Lab/mmlu_pro_leaderboard_submission
  Direct CSV: .../resolve/main/results.csv
- Leaderboard Space: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
- Underlying benchmark: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Paper: https://arxiv.org/abs/2406.01574

Each CSV row carries a model's overall accuracy plus 14 per-subject
accuracies (Biology, Business, Chemistry, Computer Science, Economics,
Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology,
Other), all reported as proportions in [0, 1] from a 5-shot CoT setup
described in Wang et al. 2024.

For every model the adapter emits one ``EvaluationLog`` with 15
``EvaluationResult`` entries: ``mmlu_pro/overall`` and one
``mmlu_pro/<subject_slug>`` per category.

Usage:
    uv run python -m utils.mmlu_pro.adapter --output-dir data/mmlu-pro
    uv run python -m utils.mmlu_pro.adapter \\
        --input-csv /tmp/mmlu_pro.csv --output-dir /tmp/mmlu-pro-smoke
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from pathlib import Path
from typing import Iterable

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    SourceType,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    get_developer,
    get_model_id,
    sanitize_filename,
    save_evaluation_log,
)

SOURCE_NAME = 'MMLU-Pro Leaderboard'
SOURCE_ORGANIZATION = 'TIGER-Lab'
SOURCE_ORGANIZATION_URL = 'https://tiger-ai-lab.github.io'
LEADERBOARD_SPACE_URL = 'https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro'
RESULTS_HF_REPO = 'TIGER-Lab/mmlu_pro_leaderboard_submission'
RESULTS_CSV_URL = (
    f'https://huggingface.co/datasets/{RESULTS_HF_REPO}'
    '/resolve/main/results.csv'
)
BENCHMARK_HF_REPO = 'TIGER-Lab/MMLU-Pro'
PAPER_URL = 'https://arxiv.org/abs/2406.01574'
GITHUB_URL = 'https://github.com/TIGER-AI-Lab/MMLU-Pro'
DEFAULT_OUTPUT_DIR = 'data/mmlu-pro'
DATASET_TOTAL_QUESTIONS = 12000

SUBJECTS: tuple[str, ...] = (
    'Biology',
    'Business',
    'Chemistry',
    'Computer Science',
    'Economics',
    'Engineering',
    'Health',
    'History',
    'Law',
    'Math',
    'Philosophy',
    'Physics',
    'Psychology',
    'Other',
)

# The CSV's "Data Source" column has a couple of obvious typos in the
# upstream file. Normalize them so downstream consumers see a consistent
# string.
DATA_SOURCE_NORMALIZATIONS: dict[str, str] = {
    'TIGER-LAb': 'TIGER-Lab',
    'Sefl-Reported': 'Self-Reported',
}

# A few MMLU-Pro models aren't covered by the helpers/developer.py
# pattern map. Provide explicit overrides for those.
DEVELOPER_OVERRIDES: dict[str, str] = {
    'exaone': 'lg-ai',
    'mammoth': 'tiger-lab',
    'smaug': 'abacus-ai',
    'athene': 'nexusflow',
    'tulu': 'allenai',
    'sailor2': 'sail',
    'xverse': 'xverse',
    'internlm': 'shanghai-ai-lab',
    'orca': 'microsoft',
    'wizardlm': 'wizardlm',
    'minicpm': 'openbmb',
    'rho': 'microsoft',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert the MMLU-Pro leaderboard CSV into EEE records.'
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        help='Read a saved CSV instead of fetching from the HF dataset.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--source-url',
        default=RESULTS_CSV_URL,
        help='Override the upstream results CSV URL.',
    )
    return parser.parse_args()


def fetch_csv(url: str) -> str:
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.text


def load_csv_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def parse_rows(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    return [
        {key: (value or '').strip() for key, value in row.items() if key}
        for row in reader
    ]


def slugify(value: str) -> str:
    base = re.sub(r'[^\w.\-]+', '-', value.strip().lower())
    base = re.sub(r'-{2,}', '-', base).strip('-')
    return sanitize_filename(base) or 'unknown'


def subject_slug(subject: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', subject.lower()).strip('_')


def normalize_data_source(value: str) -> str:
    return DATA_SOURCE_NORMALIZATIONS.get(value, value)


def parse_size(value: str) -> float | None:
    if not value or value.lower() == 'unk':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_developer(model_name: str) -> str:
    lower = model_name.lower()
    for key, developer in DEVELOPER_OVERRIDES.items():
        if lower.startswith(key) or f'-{key}' in lower:
            return developer
    return get_developer(model_name)


def make_source_data() -> SourceDataHf:
    return SourceDataHf(
        dataset_name='MMLU-Pro leaderboard submissions (TIGER-Lab)',
        source_type='hf_dataset',
        hf_repo=RESULTS_HF_REPO,
        hf_split='train',
        additional_details={
            'results_csv_url': RESULTS_CSV_URL,
            'leaderboard_space_url': LEADERBOARD_SPACE_URL,
            'benchmark_hf_repo': BENCHMARK_HF_REPO,
            'paper_url': PAPER_URL,
            'github_url': GITHUB_URL,
            'dataset_total_questions': str(DATASET_TOTAL_QUESTIONS),
            'prompt_style': '5-shot CoT',
        },
    )


def make_metric_config(
    *,
    metric_id: str,
    metric_name: str,
    description: str,
) -> MetricConfig:
    return MetricConfig(
        evaluation_description=description,
        metric_id=metric_id,
        metric_name=metric_name,
        metric_kind='accuracy',
        metric_unit='proportion',
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=1.0,
        additional_details={
            'aggregation': 'accuracy_over_subset',
            'prompt_style': '5-shot CoT',
        },
    )


def make_evaluation_result(
    *, result_id: str, name: str, description: str, score: float
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_result_id=result_id,
        evaluation_name=name,
        source_data=make_source_data(),
        metric_config=make_metric_config(
            metric_id=result_id,
            metric_name=name,
            description=description,
        ),
        score_details=ScoreDetails(score=score),
    )


def parse_score(raw: str) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _source_metadata_extras(
    data_source: str, raw_data_source: str
) -> dict[str, str]:
    extras: dict[str, str] = {
        'leaderboard_space_url': LEADERBOARD_SPACE_URL,
        'results_csv_url': RESULTS_CSV_URL,
        'paper_url': PAPER_URL,
        'github_url': GITHUB_URL,
        'leaderboard_data_source': data_source or 'unknown',
    }
    if raw_data_source and raw_data_source != data_source:
        extras['raw_leaderboard_data_source'] = raw_data_source
    return extras


def make_log(
    row: dict[str, str], retrieved_timestamp: str
) -> tuple[EvaluationLog, str, str] | None:
    model_name = row.get('Models', '').strip()
    if not model_name:
        return None
    overall = parse_score(row.get('Overall', ''))
    if overall is None:
        return None

    developer = normalize_developer(model_name)
    model_slug = slugify(model_name)
    model_id = get_model_id(model_slug, developer)
    raw_data_source = row.get('Data Source', '').strip()
    data_source = normalize_data_source(raw_data_source)
    size_b = parse_size(row.get('Model Size(B)', ''))

    results: list[EvaluationResult] = [
        make_evaluation_result(
            result_id='mmlu_pro/overall',
            name='MMLU-Pro (overall)',
            description=(
                'Overall accuracy across the ~12,000-question MMLU-Pro '
                'benchmark, evaluated 5-shot with chain-of-thought.'
            ),
            score=overall,
        )
    ]
    for subject in SUBJECTS:
        subject_score = parse_score(row.get(subject, ''))
        if subject_score is None:
            continue
        slug = subject_slug(subject)
        results.append(
            make_evaluation_result(
                result_id=f'mmlu_pro/{slug}',
                name=f'MMLU-Pro ({subject})',
                description=(
                    f'Accuracy on the MMLU-Pro {subject} subset, '
                    'evaluated 5-shot with chain-of-thought.'
                ),
                score=subject_score,
            )
        )

    model_additional: dict[str, str] = {'raw_model_name': model_name}
    if size_b is not None:
        model_additional['size_billions_parameters'] = str(size_b)
    if data_source:
        model_additional['leaderboard_data_source'] = data_source
    if raw_data_source and raw_data_source != data_source:
        model_additional['raw_leaderboard_data_source'] = raw_data_source

    sanitized_model_id = model_id.replace('/', '_')
    data_source_slug = slugify(data_source) if data_source else 'unknown'
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'mmlu-pro/{sanitized_model_id}/{data_source_slug}/'
            f'{retrieved_timestamp}'
        ),
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type=SourceType.documentation,
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=SOURCE_ORGANIZATION_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=_source_metadata_extras(
                data_source, raw_data_source
            ),
        ),
        eval_library=EvalLibrary(
            name='MMLU-Pro leaderboard (TIGER-Lab)', version='unknown'
        ),
        model_info=ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
            additional_details=model_additional,
        ),
        evaluation_results=results,
    )
    return log, developer, model_slug


def make_logs(
    rows: Iterable[dict[str, str]],
    retrieved_timestamp: str | None = None,
) -> list[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    bundles: list[tuple[EvaluationLog, str, str]] = []
    # The CSV occasionally has identical duplicate rows (e.g. 'LLaDA' is
    # listed twice with the same score and source); skip those. But the
    # same model is also legitimately reported by both 'TIGER-Lab' and
    # 'Self-Reported' with different numbers — we want both of those.
    # Dedup on (model_id, data_source, overall) so legitimate variants
    # survive and exact dupes are dropped.
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        result = make_log(row, timestamp)
        if result is None:
            continue
        log, developer, slug = result
        data_source = (log.source_metadata.additional_details or {}).get(
            'leaderboard_data_source', ''
        )
        overall = next(
            (
                str(r.score_details.score)
                for r in log.evaluation_results
                if r.evaluation_result_id == 'mmlu_pro/overall'
            ),
            '',
        )
        key = (log.model_info.id, data_source, overall)
        if key in seen:
            continue
        seen.add(key)
        bundles.append((log, developer, slug))
    return bundles


def export(
    bundles: list[tuple[EvaluationLog, str, str]], output_dir: Path
) -> list[Path]:
    paths = []
    for log, developer, model_slug in bundles:
        path = save_evaluation_log(log, output_dir, developer, model_slug)
        paths.append(path)
    return paths


def run(args: argparse.Namespace) -> int:
    if args.input_csv is not None:
        text = load_csv_text(args.input_csv)
    else:
        text = fetch_csv(args.source_url)
    rows = parse_rows(text)
    bundles = make_logs(rows)
    paths = export(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} MMLU-Pro model log(s).')
