#!/usr/bin/env python3
"""Convert Scale SEAL Humanity's Last Exam leaderboard into Every Eval Ever records.

Data source:
- Public leaderboard page: https://labs.scale.com/leaderboard/humanitys_last_exam
  The page is rendered with Next.js App Router; the leaderboard rows are
  embedded as JSON inside ``self.__next_f.push([1, "..."])`` RSC chunks.
- Underlying benchmark: https://agi.safe.ai/ (CAIS, 2,500 frozen questions)

Each leaderboard row has shape:
    {
      "model": "gemini-3.1-pro-preview (thinking high)",
      "version": "",
      "rank": 1,
      "score": 46.44,                     # accuracy %
      "confidenceInterval_upper": 1.96,   # 95% CI half-width
      "contaminationMessage": "...",
      "company": "google",
      "createdAt": "2026-04-10T15:51:06.000Z",
      "deprecated": false,
      "calibrationError": 51,             # %, lower is better
      "maxScore": 49.852
    }

For every model the adapter emits one ``EvaluationLog`` with two
``EvaluationResult`` entries: ``hle/accuracy`` and
``hle/calibration_error``.

Usage:
    uv run python -m utils.hle.adapter --output-dir data/hle
    uv run python -m utils.hle.adapter \\
        --input-json /tmp/hle_seal_payload.json --output-dir /tmp/hle-smoke
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
    ConfidenceInterval,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    JudgeConfig,
    LlmScoring,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    get_model_id,
    sanitize_filename,
    save_evaluation_log,
)

SOURCE_NAME = "Scale SEAL Humanity's Last Exam Leaderboard"
SOURCE_ORGANIZATION = 'Scale'
SOURCE_ORGANIZATION_URL = 'https://labs.scale.com'
LEADERBOARD_URL = 'https://labs.scale.com/leaderboard/humanitys_last_exam'
HLE_HOME_URL = 'https://agi.safe.ai/'
HLE_DATASET_HF_URL = 'https://huggingface.co/datasets/cais/hle'
DEFAULT_OUTPUT_DIR = 'data/hle'
JUDGE_MODEL_ID = 'openai/o3-mini-2025-01-31'
JUDGE_PROMPT_DESCRIPTION = (
    "Scale SEAL evaluates Humanity's Last Exam at temperature 0.0. The "
    'judge model o3-mini-2025-01-31 acts as an automatic answer extractor '
    'and grader against ground-truth solutions for each of the 2,500 '
    'frozen questions.'
)
DATASET_TOTAL_QUESTIONS = 2500

# Map the leaderboard's lowercase company slug to the canonical developer
# slug used elsewhere in the EEE data tree (matches helpers/developer.py
# patterns: e.g. 'kimi' -> 'moonshotai', 'grok' -> 'xai').
COMPANY_TO_DEVELOPER: dict[str, str] = {
    'amazon': 'amazon',
    'anthropic': 'anthropic',
    'google': 'google',
    'meta': 'meta',
    'mistral': 'mistralai',
    'moonshot': 'moonshotai',
    'openai': 'openai',
    'zai': 'zhipu-ai',
}


@dataclass(frozen=True)
class LeaderboardRow:
    raw: dict[str, Any]

    @property
    def model_display(self) -> str:
        return str(self.raw.get('model') or '').strip()

    @property
    def company(self) -> str:
        return str(self.raw.get('company') or '').strip().lower()

    @property
    def score(self) -> float:
        return float(self.raw['score'])

    @property
    def confidence_half_width(self) -> float | None:
        value = self.raw.get('confidenceInterval_upper')
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def calibration_error(self) -> float | None:
        value = self.raw.get('calibrationError')
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def created_at(self) -> str | None:
        value = self.raw.get('createdAt')
        return str(value) if value else None

    @property
    def contamination_message(self) -> str | None:
        value = self.raw.get('contaminationMessage')
        return str(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Scale SEAL's Humanity's Last Exam leaderboard to EEE "
            'records.'
        ),
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help=(
            'Read a saved JSON payload (e.g. produced by an earlier run) '
            'instead of fetching the leaderboard page live.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--source-url',
        default=LEADERBOARD_URL,
        help='Override the upstream leaderboard URL.',
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help=(
            'After fetching, write the parsed rows to this path so future '
            'runs can replay offline with --input-json.'
        ),
    )
    return parser.parse_args()


def fetch_html(url: str) -> str:
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.text


def parse_rsc_payload(html: str) -> list[dict[str, Any]]:
    """Extract leaderboard rows from the embedded Next.js App Router payload.

    The Scale Labs page renders client data via ``self.__next_f.push([1,
    "<escaped-rsc-chunk>"])`` script tags. The chunk containing the
    leaderboard rows is identifiable by its high concentration of model
    name strings.
    """
    chunks = re.findall(
        r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL
    )
    if not chunks:
        raise ValueError(
            'No __next_f payload found in leaderboard HTML. The page '
            'structure may have changed.'
        )

    def model_name_score(chunk: str) -> int:
        lower = chunk.lower()
        return (
            lower.count('gpt-5') + lower.count('claude') + lower.count('gemini')
        )

    best = max(chunks, key=model_name_score)
    unescaped = best.encode().decode('unicode_escape')

    rank_one_idx = unescaped.find('"rank":1,')
    if rank_one_idx < 0:
        raise ValueError(
            'Could not locate a `"rank":1` row in the parsed RSC chunk.'
        )

    start = _find_enclosing_array_start(unescaped, rank_one_idx)
    end = _find_matching_close(unescaped, start)
    blob = unescaped[start : end + 1]
    rows = json.loads(blob)
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            'Parsed leaderboard payload is not a non-empty list of rows.'
        )
    return rows


def _find_enclosing_array_start(text: str, idx: int) -> int:
    depth = 0
    for i in range(idx, -1, -1):
        if text[i] == ']':
            depth += 1
        elif text[i] == '[':
            if depth == 0:
                return i
            depth -= 1
    raise ValueError('Unmatched array bracket while parsing RSC chunk.')


def _find_matching_close(text: str, start: int) -> int:
    depth = 1
    for j in range(start + 1, len(text)):
        if text[j] == '[':
            depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('Unmatched array bracket while parsing RSC chunk.')


def load_payload_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, dict) and isinstance(payload.get('rows'), list):
        return payload['rows']
    if isinstance(payload, list):
        return payload
    raise ValueError(
        '--input-json must contain either a list of rows or an object '
        'with a top-level "rows" array.'
    )


def slugify_model(raw: str) -> str:
    base = re.sub(r'[^\w.\-]+', '-', raw.strip().lower())
    base = re.sub(r'-{2,}', '-', base).strip('-')
    return sanitize_filename(base) or 'unknown'


def normalize_developer(company: str) -> str:
    if not company:
        return 'unknown'
    return COMPANY_TO_DEVELOPER.get(company, company)


def make_judge_model_info() -> ModelInfo:
    return ModelInfo(
        name='o3-mini-2025-01-31',
        id=JUDGE_MODEL_ID,
        developer='openai',
    )


def make_generation_config() -> GenerationConfig:
    """Per Scale SEAL's published methodology, the model under evaluation
    runs at temperature 0.0. Per the schema, that belongs on
    ``EvaluationResult.generation_config.generation_args.temperature``
    (eval.schema.json L400) — not in source/metric ``additional_details``.
    """
    return GenerationConfig(generation_args=GenerationArgs(temperature=0.0))


def make_source_data() -> SourceDataUrl:
    return SourceDataUrl(
        dataset_name="Humanity's Last Exam (Scale SEAL leaderboard)",
        source_type='url',
        url=[LEADERBOARD_URL, HLE_HOME_URL, HLE_DATASET_HF_URL],
        additional_details={
            'dataset_total_questions': str(DATASET_TOTAL_QUESTIONS),
        },
    )


def stringify_optional(value: Any) -> str | None:
    if value is None or value == '':
        return None
    return str(value)


def make_accuracy_result(
    row: LeaderboardRow, judge_model_info: ModelInfo
) -> EvaluationResult:
    score = row.score
    half_width = row.confidence_half_width
    uncertainty: Uncertainty | None = None
    if half_width is not None:
        uncertainty = Uncertainty(
            confidence_interval=ConfidenceInterval(
                lower=score - half_width,
                upper=score + half_width,
                confidence_level=0.95,
                method='reported_by_source',
            ),
        )

    score_details_kwargs: dict[str, Any] = {'score': score}
    if uncertainty is not None:
        score_details_kwargs['uncertainty'] = uncertainty
    score_details_kwargs['details'] = {
        'rank': str(row.raw.get('rank')),
        'max_score_observed': str(row.raw.get('maxScore')),
    }

    return EvaluationResult(
        evaluation_result_id='hle/accuracy',
        evaluation_name="Humanity's Last Exam (accuracy)",
        source_data=make_source_data(),
        metric_config=MetricConfig(
            evaluation_description=(
                "Accuracy on the 2,500-question Humanity's Last Exam "
                'multimodal benchmark, as reported on the Scale SEAL '
                'leaderboard. Reported as a percentage with 95% confidence '
                'interval (Wilson interval, computed by Scale).'
            ),
            metric_id='hle.accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
            llm_scoring=LlmScoring(
                judges=[JudgeConfig(model_info=judge_model_info)],
                input_prompt=JUDGE_PROMPT_DESCRIPTION,
            ),
            additional_details={
                'aggregation': 'accuracy_over_full_dataset',
            },
        ),
        score_details=ScoreDetails(**score_details_kwargs),
        generation_config=make_generation_config(),
    )


def make_calibration_result(
    row: LeaderboardRow, judge_model_info: ModelInfo
) -> EvaluationResult | None:
    if row.calibration_error is None:
        return None
    return EvaluationResult(
        evaluation_result_id='hle/calibration_error',
        evaluation_name="Humanity's Last Exam (calibration error)",
        source_data=make_source_data(),
        metric_config=MetricConfig(
            evaluation_description=(
                'Calibration error: the extent to which the model is over- '
                'or under-confident in its answers. Models supply 0–100 '
                'confidence scores alongside answers; calibration error '
                'measures the deviation from perfect calibration.'
            ),
            metric_id='hle.calibration_error',
            metric_name='Calibration Error',
            metric_kind='calibration_error',
            metric_unit='percent',
            lower_is_better=True,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
            llm_scoring=LlmScoring(
                judges=[JudgeConfig(model_info=judge_model_info)],
                input_prompt=JUDGE_PROMPT_DESCRIPTION,
            ),
        ),
        score_details=ScoreDetails(score=float(row.calibration_error)),
        generation_config=make_generation_config(),
    )


def make_log(
    row: LeaderboardRow, retrieved_timestamp: str
) -> tuple[EvaluationLog, str, str]:
    developer = normalize_developer(row.company)
    model_slug = slugify_model(row.model_display)
    model_id = get_model_id(model_slug, developer)

    judge_model_info = make_judge_model_info()
    results: list[EvaluationResult] = [
        make_accuracy_result(row, judge_model_info)
    ]
    calibration_result = make_calibration_result(row, judge_model_info)
    if calibration_result is not None:
        results.append(calibration_result)

    additional_details = {
        'leaderboard_company': row.company,
        'raw_model_display_name': row.model_display,
        'rank': str(row.raw.get('rank')),
    }
    contamination = row.contamination_message
    if contamination:
        additional_details['contamination_message'] = contamination
    deprecated = row.raw.get('deprecated')
    if deprecated is not None:
        additional_details['deprecated'] = 'true' if deprecated else 'false'

    sanitized_model_id = model_id.replace('/', '_')
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'hle/{sanitized_model_id}/{retrieved_timestamp}',
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=row.created_at,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type=SourceType.documentation,
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=SOURCE_ORGANIZATION_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'leaderboard_url': LEADERBOARD_URL,
                'hle_home_url': HLE_HOME_URL,
                'hle_dataset_hf_url': HLE_DATASET_HF_URL,
            },
        ),
        eval_library=EvalLibrary(
            name='Scale SEAL HLE leaderboard', version='unknown'
        ),
        model_info=ModelInfo(
            name=row.model_display,
            id=model_id,
            developer=developer,
            additional_details=additional_details,
        ),
        evaluation_results=results,
    )
    return log, developer, model_slug


def make_logs(
    rows: list[dict[str, Any]],
    retrieved_timestamp: str | None = None,
) -> list[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    bundles = []
    seen_ids: set[str] = set()
    for raw_row in rows:
        row = LeaderboardRow(raw=raw_row)
        if not row.model_display or row.raw.get('score') is None:
            continue
        log, developer, slug = make_log(row, timestamp)
        if log.model_info.id in seen_ids:
            raise ValueError(
                f'Duplicate model id {log.model_info.id!r} in leaderboard '
                'payload. Two rows produced the same canonical id; check '
                'COMPANY_TO_DEVELOPER and slugify_model.'
            )
        seen_ids.add(log.model_info.id)
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
    if args.input_json is not None:
        rows = load_payload_file(args.input_json)
    else:
        html = fetch_html(args.source_url)
        rows = parse_rsc_payload(html)
        if args.save_raw_json is not None:
            args.save_raw_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_raw_json.write_text(
                json.dumps(
                    {
                        'source_url': args.source_url,
                        'fetched_at': str(time.time()),
                        'rows': rows,
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )

    bundles = make_logs(rows)
    paths = export(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} HLE model log(s).')
