"""Adapter for converting AlpacaEval leaderboard CSVs to every_eval_ever format."""

import csv
import io
import re
from typing import Any, Dict, List, Optional

import requests

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
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
    StandardError,
    Uncertainty,
)

# ---------------------------------------------------------------------------
# Leaderboard configurations
# ---------------------------------------------------------------------------

LEADERBOARDS: Dict[str, Dict[str, Any]] = {
    'v1': {
        'url': (
            'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/'
            'src/alpaca_eval/leaderboards/data_AlpacaEval/'
            'alpaca_eval_gpt4_leaderboard.csv'
        ),
        'source_name': 'AlpacaEval 1.0',
        'version': '1.0',
        'baseline': 'text_davinci_003',
        'annotator': 'alpaca_eval_gpt4',
        'has_lc': False,
    },
    'v2': {
        'url': (
            'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/'
            'src/alpaca_eval/leaderboards/data_AlpacaEval_2/'
            'weighted_alpaca_eval_gpt4_turbo_leaderboard.csv'
        ),
        'source_name': 'AlpacaEval 2.0',
        'version': '2.0',
        'baseline': 'gpt4_turbo',
        'annotator': 'weighted_alpaca_eval_gpt4_turbo',
        'has_lc': True,
    },
}

# Map substrings in lowercase model names → canonical developer IDs
_DEVELOPER_MAP = [
    ('gpt-4', 'openai'),
    ('gpt-3', 'openai'),
    ('gpt4', 'openai'),
    ('gpt3', 'openai'),
    ('o1-', 'openai'),
    ('o3-', 'openai'),
    ('o4-', 'openai'),
    ('chatgpt', 'openai'),
    ('davinci', 'openai'),
    ('claude', 'anthropic'),
    ('gemini', 'google'),
    ('palm', 'google'),
    ('bard', 'google'),
    ('llama', 'meta-llama'),
    ('mistral', 'mistralai'),
    ('mixtral', 'mistralai'),
    ('falcon', 'tiiuae'),
    ('vicuna', 'lmsys'),
    ('alpaca', 'stanford'),
    ('koala', 'berkeley'),
    ('orca', 'microsoft'),
    ('phi-', 'microsoft'),
    ('phi_', 'microsoft'),
    ('wizardlm', 'WizardLM'),
    ('qwen', 'Qwen'),
    ('deepseek', 'deepseek-ai'),
    ('yi-', '01-ai'),
    ('gemma', 'google'),
    ('command', 'CohereForAI'),
    ('cohere', 'CohereForAI'),
    ('solar', 'upstage'),
    ('zephyr', 'HuggingFaceH4'),
    ('tulu', 'allenai'),
    ('olmo', 'allenai'),
    ('xwinlm', 'Xwin-LM'),
    ('guanaco', 'timdettmers'),
    ('openchat', 'openchat'),
]


def _infer_developer(model_name: str) -> Optional[str]:
    lower = model_name.lower()
    for pattern, dev in _DEVELOPER_MAP:
        if pattern in lower:
            return dev
    return None


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fetch_csv(url: str) -> List[Dict[str, str]]:
    """Download a CSV from *url* and return a list of row dicts."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


def _model_name_from_row(row: Dict[str, str]) -> str:
    """Extract the model name from a CSV row (first/unnamed column)."""
    for key in ('', 'Unnamed: 0', 'model', 'Model'):
        if key in row and row[key].strip():
            return row[key].strip()
    # Fallback: first value
    return next(iter(row.values()), '').strip()


def _build_evaluation_results(
    row: Dict[str, str], cfg: Dict[str, Any]
) -> List[EvaluationResult]:
    """Build EvaluationResult list from a single CSV row."""
    results = []

    source_data = SourceDataUrl(
        dataset_name=cfg['source_name'],
        source_type='url',
        url=['https://github.com/tatsu-lab/alpaca_eval'],
    )

    win_rate = _to_float(row.get('win_rate'))
    lc_win_rate = _to_float(row.get('length_controlled_winrate'))
    std_err = _to_float(row.get('standard_error'))
    lc_std_err = _to_float(row.get('lc_standard_error'))
    discrete_wr = _to_float(row.get('discrete_win_rate'))
    avg_length = _to_float(row.get('avg_length'))

    def _wr_uncertainty(se_val: Optional[float]) -> Optional[Uncertainty]:
        if se_val is None:
            return None
        return Uncertainty(
            standard_error=StandardError(
                value=round(se_val / 100, 6), method='bootstrap'
            )
        )

    # Win Rate (raw)
    if win_rate is not None:
        results.append(
            EvaluationResult(
                evaluation_name='Win Rate',
                metric_config=MetricConfig(
                    evaluation_description=(
                        f'Fraction of outputs preferred over the '
                        f'{cfg["baseline"]} baseline by the '
                        f'{cfg["annotator"]} judge.'
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(
                    score=round(win_rate / 100, 6),
                    uncertainty=_wr_uncertainty(std_err),
                ),
                source_data=source_data,
            )
        )

    # Length-Controlled Win Rate (v2)
    if lc_win_rate is not None:
        results.append(
            EvaluationResult(
                evaluation_name='Length-Controlled Win Rate',
                metric_config=MetricConfig(
                    evaluation_description=(
                        'Win rate debiased for output length, raising '
                        'Chatbot Arena rank correlation from 0.93 to 0.98.'
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(
                    score=round(lc_win_rate / 100, 6),
                    uncertainty=_wr_uncertainty(lc_std_err),
                ),
                source_data=source_data,
            )
        )

    # Discrete Win Rate
    if discrete_wr is not None:
        results.append(
            EvaluationResult(
                evaluation_name='Discrete Win Rate',
                metric_config=MetricConfig(
                    evaluation_description=(
                        'Binary win rate — no partial credit for ties.'
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(score=round(discrete_wr / 100, 6)),
                source_data=source_data,
            )
        )

    # Average response length (informational)
    if avg_length is not None:
        results.append(
            EvaluationResult(
                evaluation_name='Average Response Length',
                metric_config=MetricConfig(
                    evaluation_description=(
                        'Mean number of tokens in model responses.'
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=100000.0,
                ),
                score_details=ScoreDetails(score=avg_length),
                source_data=source_data,
            )
        )

    return results


class AlpacaEvalAdapter:
    """Converts AlpacaEval leaderboard CSV rows into EvaluationLog objects."""

    def fetch_leaderboard(self, version: str = 'v2') -> List[EvaluationLog]:
        """Fetch a leaderboard by version key ('v1' or 'v2') and return logs.

        Args:
            version: Leaderboard version — 'v1' (AlpacaEval 1.0) or
                     'v2' (AlpacaEval 2.0, weighted LC win rate).

        Returns:
            List of EvaluationLog objects, one per model row.
        """
        if version not in LEADERBOARDS:
            raise ValueError(
                f'Unknown version {version!r}. Choose from: '
                + ', '.join(LEADERBOARDS)
            )
        cfg = LEADERBOARDS[version]
        rows = _fetch_csv(cfg['url'])
        retrieved_ts = get_current_unix_timestamp()

        benchmark_key = f'alpaca_eval_{version}'
        logs = []

        for row in rows:
            model_name = _model_name_from_row(row)
            if not model_name:
                continue

            # Skip NullModel placeholder
            if re.fullmatch(r'null.?model', model_name, re.IGNORECASE):
                continue

            win_rate = row.get('win_rate', '').strip()
            if not win_rate:
                continue

            developer = _infer_developer(model_name)
            model_id = (
                f'{developer}/{model_name}' if developer else model_name
            )

            evaluation_id = (
                f'{benchmark_key}/{model_id}/{retrieved_ts}'
            )

            eval_results = _build_evaluation_results(row, cfg)
            if not eval_results:
                continue

            log = EvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                retrieved_timestamp=retrieved_ts,
                eval_library=EvalLibrary(
                    name='alpaca_eval',
                    version=cfg['version'],
                    additional_details={
                        'annotator': cfg['annotator'],
                        'baseline_model': cfg['baseline'],
                        'github': (
                            'https://github.com/tatsu-lab/alpaca_eval'
                        ),
                    },
                ),
                source_metadata=SourceMetadata(
                    source_name=cfg['source_name'],
                    source_type=SourceType.documentation,
                    source_organization_name='Stanford CRFM / Tatsu Lab',
                    source_organization_url=(
                        'https://github.com/tatsu-lab/alpaca_eval'
                    ),
                    evaluator_relationship=EvaluatorRelationship.third_party,
                ),
                model_info=ModelInfo(
                    name=model_name,
                    id=model_id,
                    developer=developer,
                ),
                evaluation_results=eval_results,
            )
            logs.append(log)

        return logs
