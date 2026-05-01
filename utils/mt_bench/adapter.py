#!/usr/bin/env python3
"""Convert MT-Bench (LMSYS / FastChat) judgments into Every Eval Ever records.

Data source:
- Pre-generated GPT-4 single-answer judgments hosted by LMSYS:
  https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl
- Repository: https://github.com/lm-sys/FastChat (Apache 2.0)
- Paper: https://arxiv.org/abs/2306.05685

Each input line has shape:
    {"question_id": int, "model": str, "judge": [judge_model, prompt_template],
     "user_prompt": str, "judgment": str, "score": int (1-10, or -1),
     "turn": int (1 or 2), "tstamp": float}

For every model the adapter emits one aggregate JSON file with three results:
``mt_bench/overall``, ``mt_bench/turn_1``, ``mt_bench/turn_2`` — matching the
aggregation that ``fastchat/llm_judge/show_result.py`` performs (mean across
non-``-1`` scores, optionally filtered by turn).

Usage:
    uv run python -m utils.mt_bench.adapter --output-dir data/mt-bench
    uv run python -m utils.mt_bench.adapter \\
        --input-jsonl /tmp/gpt-4_single.jsonl --output-dir /tmp/mt-bench-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    JudgeConfig,
    LlmScoring,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    get_developer,
    get_model_id,
    sanitize_filename,
    save_evaluation_log,
)

SOURCE_NAME = 'MT-Bench'
SOURCE_ORGANIZATION = 'LMSYS'
SOURCE_ORGANIZATION_URL = 'https://lmsys.org'
PAPER_URL = 'https://arxiv.org/abs/2306.05685'
GITHUB_URL = 'https://github.com/lm-sys/FastChat'
JUDGMENT_URL = (
    'https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/'
    'data/mt_bench/model_judgment/gpt-4_single.jsonl'
)
DEFAULT_OUTPUT_DIR = 'data/mt-bench'
JUDGE_MODEL_NAME = 'gpt-4'
JUDGE_PROMPT_DESCRIPTION = (
    'MT-Bench single-answer grading prompt. The judge is asked to rate the '
    "assistant's answer on a scale of 1 to 10 using the format '[[rating]]'. "
    'Turn-2 uses a multi-turn variant that shows the full two-turn dialogue. '
    'Templates: see https://github.com/lm-sys/FastChat/blob/main/fastchat/'
    'llm_judge/data/judge_prompts.jsonl'
)

# FastChat's MT-Bench was scored in 2023 against models that pre-date most of
# the developer mappings in helpers/developer.py. Provide explicit overrides
# for every model in the canonical pre-generated file so each model lands
# under a meaningful developer slug rather than under "unknown".
DEVELOPER_OVERRIDES: dict[str, str] = {
    'alpaca-13b': 'stanford',
    'baize-v2-13b': 'project-baize',
    'chatglm-6b': 'thudm',
    'claude-instant-v1': 'anthropic',
    'claude-v1': 'anthropic',
    'dolly-v2-12b': 'databricks',
    'falcon-40b-instruct': 'tiiuae',
    'fastchat-t5-3b': 'lmsys',
    'gpt-3.5-turbo': 'openai',
    'gpt-4': 'openai',
    'gpt4all-13b-snoozy': 'nomic-ai',
    'guanaco-33b': 'timdettmers',
    'guanaco-65b': 'timdettmers',
    'h2ogpt-oasst-open-llama-13b': 'h2oai',
    'koala-13b': 'young-geng',
    'llama-13b': 'meta',
    'Llama-2-7b-chat': 'meta',
    'Llama-2-13b-chat': 'meta',
    'Llama-2-70b-chat': 'meta',
    'mpt-7b-chat': 'mosaicml',
    'mpt-30b-chat': 'mosaicml',
    'mpt-30b-instruct': 'mosaicml',
    'nous-hermes-13b': 'nousresearch',
    'oasst-sft-4-pythia-12b': 'openassistant',
    'oasst-sft-7-llama-30b': 'openassistant',
    'palm-2-chat-bison-001': 'google',
    'rwkv-4-raven-14b': 'rwkv',
    'stablelm-tuned-alpha-7b': 'stabilityai',
    'tulu-30b': 'allenai',
    'vicuna-7b-v1.3': 'lmsys',
    'vicuna-13b-v1.3': 'lmsys',
    'vicuna-33b-v1.3': 'lmsys',
    'wizardlm-13b': 'wizardlm',
    'wizardlm-30b': 'wizardlm',
}


@dataclass
class ModelScores:
    model: str
    overall: list[float] = field(default_factory=list)
    turn1: list[float] = field(default_factory=list)
    turn2: list[float] = field(default_factory=list)
    judge_prompt_templates: set[str] = field(default_factory=set)
    judge_models: set[str] = field(default_factory=set)
    question_ids: set[int] = field(default_factory=set)
    earliest_judgment_ts: float | None = None
    latest_judgment_ts: float | None = None

    def add(self, row: dict) -> None:
        score = row.get('score')
        try:
            score = float(score)
        except (TypeError, ValueError):
            return
        if score == -1 or math.isnan(score) or math.isinf(score):
            return

        turn = row.get('turn')
        self.overall.append(score)
        if turn == 1:
            self.turn1.append(score)
        elif turn == 2:
            self.turn2.append(score)

        question_id = row.get('question_id')
        if isinstance(question_id, int):
            self.question_ids.add(question_id)

        judge = row.get('judge')
        if isinstance(judge, list) and len(judge) >= 2:
            self.judge_models.add(str(judge[0]))
            self.judge_prompt_templates.add(str(judge[1]))

        ts = row.get('tstamp')
        if isinstance(ts, (int, float)):
            ts = float(ts)
            if (
                self.earliest_judgment_ts is None
                or ts < self.earliest_judgment_ts
            ):
                self.earliest_judgment_ts = ts
            if self.latest_judgment_ts is None or ts > self.latest_judgment_ts:
                self.latest_judgment_ts = ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert MT-Bench LMSYS judgments to EEE records.'
    )
    parser.add_argument(
        '--input-jsonl',
        type=Path,
        help=(
            'Read pre-downloaded gpt-4_single.jsonl instead of fetching from '
            'the LMSYS Hugging Face Space.'
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
        default=JUDGMENT_URL,
        help='Override the upstream JSONL URL.',
    )
    return parser.parse_args()


def fetch_judgments(url: str) -> Iterable[dict]:
    import requests

    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        yield json.loads(line)


def load_judgments_file(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate(rows: Iterable[dict]) -> dict[str, ModelScores]:
    by_model: dict[str, ModelScores] = {}
    for row in rows:
        model = row.get('model')
        if not isinstance(model, str) or not model:
            continue
        scores = by_model.setdefault(model, ModelScores(model=model))
        scores.add(row)
    return by_model


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float], mean_value: float) -> float | None:
    if len(values) < 2:
        return None
    variance = sum((v - mean_value) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def normalize_developer_and_slug(model: str) -> tuple[str, str]:
    override = DEVELOPER_OVERRIDES.get(model)
    if override is not None:
        developer = override
    else:
        developer = get_developer(model)
    slug = sanitize_filename(model)
    return developer, slug


def make_judge_model_info() -> ModelInfo:
    return ModelInfo(
        name=JUDGE_MODEL_NAME,
        id=get_model_id(JUDGE_MODEL_NAME),
        developer='openai',
    )


def make_source_data() -> SourceDataUrl:
    return SourceDataUrl(
        dataset_name='MT-Bench (single-answer GPT-4 judgments)',
        source_type='url',
        url=[JUDGMENT_URL, GITHUB_URL, PAPER_URL],
        additional_details={
            'judge_model': JUDGE_MODEL_NAME,
            'paper_url': PAPER_URL,
        },
    )


def make_metric_config(
    *,
    metric_id: str,
    metric_name: str,
    description: str,
    judge_prompt_templates: list[str],
    judge_models: list[str],
    sample_count: int,
    judge_model_info: ModelInfo,
) -> MetricConfig:
    return MetricConfig(
        evaluation_description=description,
        metric_id=metric_id,
        metric_name=metric_name,
        metric_kind='judge_score',
        metric_unit='points',
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=1.0,
        max_score=10.0,
        llm_scoring=LlmScoring(
            judges=[JudgeConfig(model_info=judge_model_info)],
            input_prompt=JUDGE_PROMPT_DESCRIPTION,
        ),
        additional_details={
            'aggregation': 'mean',
            'judge_models_json': json.dumps(judge_models, sort_keys=True),
            'judge_prompt_templates_json': json.dumps(
                judge_prompt_templates, sort_keys=True
            ),
            'judgment_count': str(sample_count),
        },
    )


def make_score_details(values: list[float]) -> ScoreDetails:
    score = mean(values)
    sd = stddev(values, score)
    se = sd / math.sqrt(len(values)) if sd is not None else None
    return ScoreDetails(
        score=round(score, 4),
        uncertainty=Uncertainty(
            standard_error=StandardError(value=se, method='analytic')
            if se is not None
            else None,
            standard_deviation=sd,
            num_samples=len(values),
        ),
        details={
            'min_judgment_score': str(min(values)),
            'max_judgment_score': str(max(values)),
            'judgment_count': str(len(values)),
        },
    )


def make_evaluation_result(
    *,
    result_id: str,
    name: str,
    description: str,
    values: list[float],
    judge_prompt_templates: list[str],
    judge_models: list[str],
    judge_model_info: ModelInfo,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_result_id=result_id,
        evaluation_name=name,
        source_data=make_source_data(),
        metric_config=make_metric_config(
            metric_id=result_id,
            metric_name=name,
            description=description,
            judge_prompt_templates=judge_prompt_templates,
            judge_models=judge_models,
            sample_count=len(values),
            judge_model_info=judge_model_info,
        ),
        score_details=make_score_details(values),
    )


def make_log(
    scores: ModelScores,
    retrieved_timestamp: str,
) -> tuple[EvaluationLog, str, str] | None:
    if not scores.overall:
        return None

    developer, model_slug = normalize_developer_and_slug(scores.model)
    model_id = get_model_id(scores.model, developer)
    judge_model_info = make_judge_model_info()
    judge_prompt_templates = sorted(scores.judge_prompt_templates)
    judge_models = sorted(scores.judge_models)

    results = [
        make_evaluation_result(
            result_id='mt_bench/overall',
            name='MT-Bench (overall)',
            description=(
                'Mean GPT-4 single-answer rating across both turns of the '
                '80 MT-Bench questions (1-10 scale).'
            ),
            values=scores.overall,
            judge_prompt_templates=judge_prompt_templates,
            judge_models=judge_models,
            judge_model_info=judge_model_info,
        )
    ]
    if scores.turn1:
        results.append(
            make_evaluation_result(
                result_id='mt_bench/turn_1',
                name='MT-Bench (turn 1)',
                description=(
                    'Mean GPT-4 single-answer rating for the first-turn '
                    'response on the 80 MT-Bench questions (1-10 scale).'
                ),
                values=scores.turn1,
                judge_prompt_templates=judge_prompt_templates,
                judge_models=judge_models,
                judge_model_info=judge_model_info,
            )
        )
    if scores.turn2:
        results.append(
            make_evaluation_result(
                result_id='mt_bench/turn_2',
                name='MT-Bench (turn 2)',
                description=(
                    'Mean GPT-4 single-answer rating for the second-turn '
                    'response on the 80 MT-Bench questions (1-10 scale).'
                ),
                values=scores.turn2,
                judge_prompt_templates=judge_prompt_templates,
                judge_models=judge_models,
                judge_model_info=judge_model_info,
            )
        )

    sanitized_model_id = model_id.replace('/', '_')
    additional_source_details = {
        'judgment_url': JUDGMENT_URL,
        'github_url': GITHUB_URL,
        'paper_url': PAPER_URL,
        'judge_model': JUDGE_MODEL_NAME,
        'judge_prompt_templates_json': json.dumps(judge_prompt_templates),
        'distinct_questions': str(len(scores.question_ids)),
    }
    if scores.earliest_judgment_ts is not None:
        additional_source_details['earliest_judgment_tstamp'] = str(
            scores.earliest_judgment_ts
        )
    if scores.latest_judgment_ts is not None:
        additional_source_details['latest_judgment_tstamp'] = str(
            scores.latest_judgment_ts
        )

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(f'mt-bench/{sanitized_model_id}/{retrieved_timestamp}'),
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=(
            str(scores.latest_judgment_ts)
            if scores.latest_judgment_ts is not None
            else None
        ),
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type=SourceType.documentation,
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=SOURCE_ORGANIZATION_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=additional_source_details,
        ),
        eval_library=EvalLibrary(
            name='FastChat (llm_judge)', version='unknown'
        ),
        model_info=ModelInfo(
            name=scores.model,
            id=model_id,
            developer=developer,
            additional_details={'raw_model_name': scores.model},
        ),
        evaluation_results=results,
    )
    return log, developer, model_slug


def make_logs(
    rows: Iterable[dict],
    retrieved_timestamp: str | None = None,
) -> list[tuple[EvaluationLog, str, str]]:
    timestamp = retrieved_timestamp or str(time.time())
    by_model = aggregate(rows)
    bundles = []
    for model in sorted(by_model):
        result = make_log(by_model[model], timestamp)
        if result is not None:
            bundles.append(result)
    return bundles


def export(
    bundles: list[tuple[EvaluationLog, str, str]],
    output_dir: Path,
) -> list[Path]:
    paths = []
    for log, developer, model_slug in bundles:
        path = save_evaluation_log(log, output_dir, developer, model_slug)
        paths.append(path)
    return paths


def run(args: argparse.Namespace) -> int:
    if args.input_jsonl is not None:
        rows: Iterable[dict] = list(load_judgments_file(args.input_jsonl))
    else:
        rows = list(fetch_judgments(args.source_url))

    bundles = make_logs(rows)
    paths = export(bundles, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} MT-Bench model log(s).')
