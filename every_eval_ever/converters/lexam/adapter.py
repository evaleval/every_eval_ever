"""Adapter for converting LEXam public leaderboard HTML to every_eval_ever format."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
from every_eval_ever.eval_types import (
    AggregationMethod,
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
    SourceDataHf,
    SourceMetadata,
    SourceType,
    Uncertainty,
)

logger = logging.getLogger(__name__)

LEADERBOARD_URL = (
    'https://raw.githubusercontent.com/LEXam-Benchmark/'
    'lexam-benchmark.github.io/main/index.html'
)
LEADERBOARD_PAGE_URL = 'https://lexam-benchmark.github.io/'
HF_REPO = 'LEXam-Benchmark/LEXam'
BENCHMARK_KEY = 'lexam'
OPEN_QUESTIONS_SAMPLES = 2541
MCQ_SAMPLES = 4696
MCQ_CONFIGS = 'mcq_4_choices,mcq_8_choices,mcq_16_choices,mcq_32_choices'

OPEN_SECTION_TITLE = 'Leaderboard on LEXam – Open Questions'
MCQ_SECTION_TITLE = 'Leaderboard on LEXam – Multiple-Choice Questions'

_MEDAL_RE = re.compile(r'[\U0001f947-\U0001f949]')


@dataclass(frozen=True)
class LeaderboardRow:
    """A single model row from a LEXam leaderboard table."""

    model_name: str
    score: float


@dataclass(frozen=True)
class ModelIdentity:
    """Canonical model identity used for Every Eval Ever output paths."""

    developer: str
    model_id: str


_MODEL_IDENTITIES = {
    'Apertus-70B': ModelIdentity('swiss-ai', 'swiss-ai/Apertus-70B'),
    'Apertus-8B': ModelIdentity('swiss-ai', 'swiss-ai/Apertus-8B'),
    'Claude-3.7-Sonnet': ModelIdentity(
        'anthropic', 'anthropic/Claude-3.7-Sonnet'
    ),
    'Claude-4.5-Sonnet': ModelIdentity(
        'anthropic', 'anthropic/Claude-4.5-Sonnet'
    ),
    'DeepSeek-R1': ModelIdentity('deepseek-ai', 'deepseek-ai/DeepSeek-R1'),
    'DeepSeek-V3': ModelIdentity('deepseek-ai', 'deepseek-ai/DeepSeek-V3'),
    'DeepSeek-V3.2-Exp': ModelIdentity(
        'deepseek-ai', 'deepseek-ai/DeepSeek-V3.2-Exp'
    ),
    'DeepSeek-V3.2-chat': ModelIdentity(
        'deepseek-ai', 'deepseek-ai/DeepSeek-V3.2-chat'
    ),
    'DeepSeek-V3.2-reasoner': ModelIdentity(
        'deepseek-ai', 'deepseek-ai/DeepSeek-V3.2-reasoner'
    ),
    'EuroLLM-9B-it': ModelIdentity(
        'utter-project', 'utter-project/EuroLLM-9B-it'
    ),
    'GPT-4.1': ModelIdentity('openai', 'openai/GPT-4.1'),
    'GPT-4.1-mini': ModelIdentity('openai', 'openai/GPT-4.1-mini'),
    'GPT-4.1-nano': ModelIdentity('openai', 'openai/GPT-4.1-nano'),
    'GPT-4o': ModelIdentity('openai', 'openai/GPT-4o'),
    'GPT-4o-mini': ModelIdentity('openai', 'openai/GPT-4o-mini'),
    'GPT-5': ModelIdentity('openai', 'openai/GPT-5'),
    'GPT-5-mini': ModelIdentity('openai', 'openai/GPT-5-mini'),
    'GPT-5-nano': ModelIdentity('openai', 'openai/GPT-5-nano'),
    'GPT-OSS-120B': ModelIdentity('openai', 'openai/GPT-OSS-120B'),
    'GPT-OSS-20B': ModelIdentity('openai', 'openai/GPT-OSS-20B'),
    'Gemini-2.5-Pro': ModelIdentity('google', 'google/Gemini-2.5-Pro'),
    'Gemini-3-Pro-preview': ModelIdentity(
        'google', 'google/Gemini-3-Pro-preview'
    ),
    'Gemma-2-9B-it': ModelIdentity('google', 'google/Gemma-2-9B-it'),
    'Gemma-3-12B-it': ModelIdentity('google', 'google/Gemma-3-12B-it'),
    'Llama-3.1-405B-it': ModelIdentity(
        'meta-llama', 'meta-llama/Llama-3.1-405B-it'
    ),
    'Llama-3.1-8B-it': ModelIdentity(
        'meta-llama', 'meta-llama/Llama-3.1-8B-it'
    ),
    'Llama-3.3-70B-it': ModelIdentity(
        'meta-llama', 'meta-llama/Llama-3.3-70B-it'
    ),
    'Llama-4-Maverick': ModelIdentity(
        'meta-llama', 'meta-llama/Llama-4-Maverick'
    ),
    'Ministral-8B-it': ModelIdentity('mistralai', 'mistralai/Ministral-8B-it'),
    'O3-mini': ModelIdentity('openai', 'openai/O3-mini'),
    'Phi-4': ModelIdentity('microsoft', 'microsoft/Phi-4'),
    'QwQ-32B': ModelIdentity('qwen', 'qwen/QwQ-32B'),
    'Qwen-2.5-7B-it': ModelIdentity('qwen', 'qwen/Qwen-2.5-7B-it'),
    'Qwen3-235B': ModelIdentity('qwen', 'qwen/Qwen3-235B'),
    'Qwen3-32B': ModelIdentity('qwen', 'qwen/Qwen3-32B'),
    'Qwen3-Next': ModelIdentity('qwen', 'qwen/Qwen3-Next'),
}


def _fetch_html(url: str = LEADERBOARD_URL) -> str:
    """Download leaderboard HTML from *url*."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _clean_model_name(raw_name: str) -> str:
    """Strip medal glyphs and whitespace from a leaderboard model name."""
    return _MEDAL_RE.sub('', raw_name).strip()


def _model_identity(model_name: str) -> ModelIdentity:
    """Return the explicit model identity for a LEXam leaderboard name."""
    if model_name not in _MODEL_IDENTITIES:
        raise ValueError(
            f'No model identity mapping for LEXam leaderboard model: {model_name}'
        )
    return _MODEL_IDENTITIES[model_name]


def _extract_section_rows(
    html: str, section_title: str
) -> list[LeaderboardRow]:
    """Parse model/score rows from the table under *section_title*."""
    title_idx = html.find(section_title)
    if title_idx == -1:
        raise ValueError(f'Leaderboard section not found: {section_title}')

    table_start = html.find('<table', title_idx)
    if table_start == -1:
        raise ValueError(f'No table found after section: {section_title}')

    table_end = html.find('</table>', table_start)
    if table_end == -1:
        raise ValueError(f'Unclosed table for section: {section_title}')

    table_html = html[table_start:table_end]
    row_re = re.compile(
        r'<tr[^>]*>\s*'
        r'<td[^>]*>(?:<strong>)?(\d+)(?:</strong>)?</td>\s*'
        r'<td[^>]*>(.*?)</td>\s*'
        r'<td[^>]*>(?:<strong>)?([\d.]+)(?:</strong>)?</td>\s*'
        r'</tr>',
        re.DOTALL | re.IGNORECASE,
    )

    rows: list[LeaderboardRow] = []
    for row_match in row_re.finditer(table_html):
        _, model_cell, score_text = row_match.groups()
        model_name = re.sub(r'<[^>]+>', '', model_cell)
        model_name = _clean_model_name(model_name)
        if not model_name:
            continue
        rows.append(
            LeaderboardRow(
                model_name=model_name,
                score=float(score_text),
            )
        )
    if not rows:
        raise ValueError(f'No leaderboard rows found for: {section_title}')
    return rows


def _open_question_source() -> SourceDataHf:
    return SourceDataHf(
        dataset_name=BENCHMARK_KEY,
        source_type='hf_dataset',
        hf_repo=HF_REPO,
        hf_split='test',
        samples_number=OPEN_QUESTIONS_SAMPLES,
        additional_details={
            'benchmark_section': 'open_questions',
            'config': 'open_question',
        },
    )


def _mcq_source() -> SourceDataHf:
    return SourceDataHf(
        dataset_name=BENCHMARK_KEY,
        source_type='hf_dataset',
        hf_repo=HF_REPO,
        hf_split='test',
        samples_number=MCQ_SAMPLES,
        additional_details={
            'benchmark_section': 'multiple_choice_questions',
            'configs': MCQ_CONFIGS,
        },
    )


def _open_question_judge_scoring() -> LlmScoring:
    return LlmScoring(
        judges=[
            JudgeConfig(
                model_info=ModelInfo(
                    name='gpt-4o',
                    id='openai/gpt-4o-2024-11-20',
                    developer='openai',
                ),
            ),
            JudgeConfig(
                model_info=ModelInfo(
                    name='DeepSeek-V3',
                    id='deepseek-ai/DeepSeek-V3',
                    developer='deepseek-ai',
                ),
            ),
            JudgeConfig(
                model_info=ModelInfo(
                    name='Qwen3-32B',
                    id='qwen/Qwen3-32B',
                    developer='qwen',
                ),
            ),
        ],
        input_prompt=(
            'Expert-validated LLM-as-a-Judge ensemble scoring open-ended '
            'law exam answers against reference answers.'
        ),
        aggregation_method=AggregationMethod.average,
        additional_details={'validation': 'human expert validated'},
    )


def _build_open_question_result(score: float) -> EvaluationResult:
    return EvaluationResult(
        evaluation_name='Open Question Judge Score',
        metric_config=MetricConfig(
            metric_id='lexam.open_question_judge_score',
            metric_name='Open Question Judge Score',
            metric_kind='accuracy',
            metric_unit='percent',
            evaluation_description=(
                'Mean LLM-judge score on open-ended law exam questions '
                '(0-100 scale).'
            ),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
            llm_scoring=_open_question_judge_scoring(),
        ),
        score_details=ScoreDetails(
            score=round(score, 2),
            uncertainty=Uncertainty(num_samples=OPEN_QUESTIONS_SAMPLES),
        ),
        source_data=_open_question_source(),
    )


def _build_mcq_result(score: float) -> EvaluationResult:
    return EvaluationResult(
        evaluation_name='Multiple-Choice Accuracy',
        metric_config=MetricConfig(
            metric_id='lexam.mcq_accuracy',
            metric_name='Multiple-Choice Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            evaluation_description=(
                'Accuracy on LEXam multiple-choice law exam questions '
                'across all MCQ configs (0-100 scale).'
            ),
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
        ),
        score_details=ScoreDetails(
            score=round(score, 2),
            uncertainty=Uncertainty(num_samples=MCQ_SAMPLES),
        ),
        source_data=_mcq_source(),
    )


class LEXamAdapter:
    """Converts LEXam public leaderboard rows into EvaluationLog objects."""

    def fetch_leaderboard(
        self,
        html: str | None = None,
        url: str = LEADERBOARD_URL,
    ) -> list[EvaluationLog]:
        """Fetch the LEXam leaderboard and return one log per model.

        Args:
            html: Optional pre-fetched HTML (used in tests).
            url: Leaderboard HTML URL when *html* is not provided.

        Returns:
            One EvaluationLog per model, combining open and MCQ metrics when
            both are available.
        """
        page_html = html if html is not None else _fetch_html(url)
        open_rows = _extract_section_rows(page_html, OPEN_SECTION_TITLE)
        mcq_rows = _extract_section_rows(page_html, MCQ_SECTION_TITLE)

        open_scores = {row.model_name: row.score for row in open_rows}
        mcq_scores = {row.model_name: row.score for row in mcq_rows}
        model_names = sorted(set(open_scores) | set(mcq_scores))

        retrieved_ts = get_current_unix_timestamp()
        logs: list[EvaluationLog] = []

        for model_name in model_names:
            evaluation_results: list[EvaluationResult] = []
            if model_name in open_scores:
                evaluation_results.append(
                    _build_open_question_result(open_scores[model_name])
                )
            if model_name in mcq_scores:
                evaluation_results.append(
                    _build_mcq_result(mcq_scores[model_name])
                )
            if not evaluation_results:
                continue

            identity = _model_identity(model_name)

            logs.append(
                EvaluationLog(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=(
                        f'{BENCHMARK_KEY}/{identity.model_id}/{retrieved_ts}'
                    ),
                    retrieved_timestamp=retrieved_ts,
                    eval_library=EvalLibrary(
                        name='lexam',
                        version='1.0',
                        additional_details={
                            'leaderboard_url': LEADERBOARD_PAGE_URL,
                            'github': 'https://github.com/LEXam-Benchmark/LEXam',
                        },
                    ),
                    source_metadata=SourceMetadata(
                        source_name='LEXam Leaderboard',
                        source_type=SourceType.documentation,
                        source_organization_name='LEXam-Benchmark',
                        source_organization_url=(
                            'https://github.com/LEXam-Benchmark/LEXam'
                        ),
                        evaluator_relationship=EvaluatorRelationship.collaborative,
                        additional_details={
                            'leaderboard_page': LEADERBOARD_PAGE_URL,
                            'source_html': url,
                        },
                    ),
                    model_info=ModelInfo(
                        name=model_name,
                        id=identity.model_id,
                        developer=identity.developer,
                    ),
                    evaluation_results=evaluation_results,
                )
            )

        logger.info('Converted %d LEXam leaderboard model(s).', len(logs))
        return logs
