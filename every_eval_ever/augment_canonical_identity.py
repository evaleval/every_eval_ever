"""Backfill canonical metric/eval identity into existing datastore files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from every_eval_ever.validate import expand_paths

WORDLE_METRICS: dict[str, dict[str, Any]] = {
    'win_rate': {
        'metric_id': 'win_rate',
        'metric_name': 'Win Rate',
        'metric_kind': 'win_rate',
        'metric_unit': 'proportion',
    },
    'avg_attempts': {
        'metric_id': 'mean_attempts',
        'metric_name': 'Average Attempts',
        'metric_kind': 'count',
        'metric_unit': 'attempts',
    },
    'avg_latency_ms': {
        'metric_id': 'latency_mean',
        'metric_name': 'Average Latency',
        'metric_kind': 'latency',
        'metric_unit': 'ms',
    },
}

HFOPENLLM_METRICS: dict[str, dict[str, Any]] = {
    'IFEval': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'BBH': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MATH Level 5': {
        'metric_id': 'exact_match',
        'metric_name': 'Exact Match',
        'metric_kind': 'exact_match',
        'metric_unit': 'proportion',
    },
    'GPQA': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MUSR': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MMLU-PRO': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
}

SINGLE_SCORE_BENCHMARK_FAMILIES = {
    'appworld_test_normal',
    'browsecompplus',
    'la_leaderboard',
    'multi-swe-bench-leaderboard',
    'swe-bench',
    'swe-polybench-leaderboard',
}

INSPECT_ACCURACY_BENCHMARK_FAMILIES = {
    'GAIA',
    'MathVista',
    'MMLU-Pro',
    'MMMU-Multiple-Choice',
    'MMMU-Open-Ended',
    'big_bench_hard',
    'commonsense_qa',
    'cybench',
    'cyse2_interpreter_abuse',
    'cyse2_prompt_injection',
    'cyse2_vulnerability_exploit',
    'gdm_intercode_ctf',
    'gpqa_diamond',
    'gsm8k',
    'hellaswag',
    'mbpp',
    'openai_humaneval',
    'piqa',
}


@dataclass
class CanonicalPatch:
    evaluation_name: str | None = None
    metric_id: str | None = None
    metric_name: str | None = None
    metric_kind: str | None = None
    metric_unit: str | None = None
    metric_parameters: dict[str, str | float | bool | None] | None = None


@dataclass
class AugmentationReport:
    original_file_path: Path
    file_path: Path
    benchmark_family: str
    aggregate_changed: bool
    sample_file_path: Path | None
    sample_changed: bool
    changed_results: int
    sample_rows: int
    path_changed: bool = False

    @property
    def changed(self) -> bool:
        return self.aggregate_changed or self.sample_changed or self.path_changed


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
    return slug or 'unknown'


def _store_raw_evaluation_name(
    metric_config: dict[str, Any], raw_evaluation_name: str
) -> None:
    additional = metric_config.get('additional_details')
    if not isinstance(additional, dict):
        additional = {}
        metric_config['additional_details'] = additional
    additional.setdefault('raw_evaluation_name', raw_evaluation_name)


def _set_metric_field(
    metric_config: dict[str, Any], key: str, value: Any
) -> bool:
    if value is None:
        return False
    if metric_config.get(key) in (None, ''):
        metric_config[key] = value
        return True
    return False


def _benchmark_metric_namespace(benchmark_family: str) -> str:
    namespace = re.sub(r'[^a-z0-9]+', '_', benchmark_family.lower()).strip('_')
    return namespace or 'benchmark'


def _score_metric_unit(metric_config: dict[str, Any]) -> str:
    min_score = metric_config.get('min_score')
    max_score = metric_config.get('max_score')

    if min_score in {0, 0.0} and max_score in {1, 1.0}:
        return 'proportion'
    if min_score in {0, 0.0} and max_score in {100, 100.0}:
        return 'points'
    return 'points'


def _metric_from_phrase(phrase: str) -> CanonicalPatch | None:
    normalized = phrase.strip().lower().rstrip(':.')

    pass_at_k_match = re.fullmatch(r'pass@(?P<k>\d+)', normalized)
    if pass_at_k_match is not None:
        k = int(pass_at_k_match.group('k'))
        return CanonicalPatch(
            metric_id='pass_at_k',
            metric_name=f'Pass@{k}',
            metric_kind='pass_rate',
            metric_unit='proportion',
            metric_parameters={'k': k},
        )

    ndcg_match = re.fullmatch(r'ndcg@(?P<k>\d+)', normalized)
    if ndcg_match is not None:
        k = int(ndcg_match.group('k'))
        return CanonicalPatch(
            metric_id='ndcg',
            metric_name=f'NDCG@{k}',
            metric_kind='ndcg',
            metric_unit='proportion',
            metric_parameters={'k': k},
        )

    rouge_match = re.fullmatch(r'rouge[-_ ]?(?P<n>\d+)', normalized)
    if rouge_match is not None:
        n = int(rouge_match.group('n'))
        return CanonicalPatch(
            metric_id=f'rouge_{n}',
            metric_name=f'ROUGE-{n}',
            metric_kind='rouge',
            metric_unit='proportion',
            metric_parameters={'n': n},
        )

    bleu_match = re.fullmatch(r'bleu[-_ ]?(?P<n>\d+)', normalized)
    if bleu_match is not None:
        n = int(bleu_match.group('n'))
        return CanonicalPatch(
            metric_id=f'bleu_{n}',
            metric_name=f'BLEU-{n}',
            metric_kind='bleu',
            metric_unit='proportion',
            metric_parameters={'n': n},
        )

    if normalized in {'acc', 'accuracy', 'cot correct', 'correct'}:
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )
    if normalized in {'equivalent', 'equivalent (cot)'}:
        return CanonicalPatch(
            metric_id='equivalent_cot',
            metric_name='Equivalent (CoT)',
            metric_kind='accuracy',
            metric_unit='proportion',
        )
    if 'strict acc' in normalized:
        return CanonicalPatch(
            metric_id='strict_accuracy',
            metric_name='Strict Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )
    if normalized == 'em':
        return CanonicalPatch(
            metric_id='exact_match',
            metric_name='Exact Match',
            metric_kind='exact_match',
            metric_unit='proportion',
        )
    if normalized == 'f1':
        return CanonicalPatch(
            metric_id='f1',
            metric_name='F1',
            metric_kind='f1',
            metric_unit='proportion',
        )
    if normalized in {'mean win rate', 'win rate'}:
        return CanonicalPatch(
            metric_id='win_rate',
            metric_name='Win Rate',
            metric_kind='win_rate',
            metric_unit='proportion',
        )
    if normalized == 'mean':
        return CanonicalPatch(
            metric_id='mean_score',
            metric_name='Mean Score',
            metric_kind='score',
            metric_unit='proportion',
        )
    if normalized in {'mean score', 'score', 'wb score'}:
        return CanonicalPatch(
            metric_id='score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )
    if normalized in {'stderr', 'standard error'}:
        return CanonicalPatch(
            metric_id='standard_error',
            metric_name='Standard Error',
            metric_kind='standard_error',
            metric_unit='proportion',
        )
    if normalized in {'std', 'stddev', 'standard deviation'}:
        return CanonicalPatch(
            metric_id='standard_deviation',
            metric_name='Standard Deviation',
            metric_kind='standard_deviation',
            metric_unit='proportion',
        )
    if normalized == 'harmlessness':
        return CanonicalPatch(
            metric_id='harmlessness',
            metric_name='Harmlessness',
            metric_kind='score',
            metric_unit='points',
        )
    return None


def _wordle_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    prefix = 'wordle_arena_'
    if not raw_evaluation_name.startswith(prefix):
        return None

    metric_suffix = raw_evaluation_name[len(prefix) :]
    metric = WORDLE_METRICS.get(metric_suffix)
    if metric is None:
        return None

    return CanonicalPatch(
        evaluation_name='wordle_arena',
        metric_id=metric['metric_id'],
        metric_name=metric['metric_name'],
        metric_kind=metric['metric_kind'],
        metric_unit=metric['metric_unit'],
    )


def _fibble_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    legacy_match = re.fullmatch(
        r'fibble_arena_(?P<variant>1lie|[2-9]lies)_(?P<metric>.+)',
        raw_evaluation_name,
    )
    if legacy_match is not None:
        metric = WORDLE_METRICS.get(legacy_match.group('metric'))
        if metric is None:
            return None

        return CanonicalPatch(
            evaluation_name=f'fibble_arena_{legacy_match.group("variant")}',
            metric_id=metric['metric_id'],
            metric_name=metric['metric_name'],
            metric_kind=metric['metric_kind'],
            metric_unit=metric['metric_unit'],
        )

    family_match = re.fullmatch(
        r'(?P<family>fibble(?:[1-5])?_arena)_(?P<metric>.+)',
        raw_evaluation_name,
    )
    if family_match is None:
        return None

    metric = WORDLE_METRICS.get(family_match.group('metric'))
    if metric is None:
        return None

    return CanonicalPatch(
        evaluation_name=family_match.group('family'),
        metric_id=metric['metric_id'],
        metric_name=metric['metric_name'],
        metric_kind=metric['metric_kind'],
        metric_unit=metric['metric_unit'],
    )


def _ifeval_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    match = re.fullmatch(
        r'(?P<slice>prompt_strict|prompt_loose|inst_strict|inst_loose|final)'
        r'_(?P<metric>acc|stderr)'
        r' on inspect_evals/ifeval(?: for scorer .+)?',
        raw_evaluation_name,
    )
    if match is None:
        return None

    metric_key = match.group('metric')
    metric = _metric_from_phrase(
        'accuracy' if metric_key == 'acc' else 'standard error'
    )
    if metric is None:
        return None
    metric.evaluation_name = match.group('slice')
    return metric


def _agentharm_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    overall_match = re.fullmatch(
        r'inspect_evals/(?P<metric>'
        r'avg_score|avg_full_score|avg_refusals|avg_score_non_refusals'
        r') on inspect_evals/agentharm(?: for scorer .+)?',
        raw_evaluation_name,
    )
    if overall_match is not None:
        metric_key = overall_match.group('metric')
        if metric_key == 'avg_score':
            return CanonicalPatch(
                evaluation_name='agentharm',
                metric_id='average_score',
                metric_name='Average Score',
                metric_kind='score',
                metric_unit='proportion',
            )
        if metric_key == 'avg_full_score':
            return CanonicalPatch(
                evaluation_name='agentharm',
                metric_id='average_full_score',
                metric_name='Average Full Score',
                metric_kind='score',
                metric_unit='proportion',
            )
        if metric_key == 'avg_refusals':
            return CanonicalPatch(
                evaluation_name='agentharm',
                metric_id='average_refusal_rate',
                metric_name='Average Refusal Rate',
                metric_kind='refusal_rate',
                metric_unit='proportion',
            )
        if metric_key == 'avg_score_non_refusals':
            return CanonicalPatch(
                evaluation_name='agentharm',
                metric_id='average_score_non_refusals',
                metric_name='Average Score (Non-Refusals)',
                metric_kind='score',
                metric_unit='proportion',
            )

    category_match = re.fullmatch(
        r'(?P<category>[A-Za-z]+)_avg_(?P<metric>scores|refusals)'
        r' on inspect_evals/agentharm(?: for scorer .+)?',
        raw_evaluation_name,
    )
    if category_match is None:
        return None

    category = category_match.group('category')
    metric_kind = category_match.group('metric')
    if metric_kind == 'scores':
        return CanonicalPatch(
            evaluation_name=category,
            metric_id='average_score',
            metric_name='Average Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    return CanonicalPatch(
        evaluation_name=category,
        metric_id='average_refusal_rate',
        metric_name='Average Refusal Rate',
        metric_kind='refusal_rate',
        metric_unit='proportion',
    )


def _generic_inspect_eval_patch(
    benchmark_family: str,
    raw_evaluation_name: str,
    metric_config: dict[str, Any],
) -> CanonicalPatch | None:
    match = re.fullmatch(
        r'(?P<metric>accuracy|mean|std) on '
        r'(?:(?:inspect_evals/)?(?P<slice>[^ ]+))'
        r'(?: for scorer .+)?',
        raw_evaluation_name,
        flags=re.IGNORECASE,
    )
    if match is None:
        metric_phrase = str(metric_config.get('evaluation_description') or '')
    else:
        metric_phrase = match.group('metric')

    metric = _metric_from_phrase(metric_phrase)
    if metric is None:
        return None

    metric.evaluation_name = benchmark_family
    return metric


def _alphaxiv_patch(
    payload: dict[str, Any], result: dict[str, Any]
) -> CanonicalPatch:
    raw_evaluation_name = str(result.get('evaluation_name') or '').strip()
    metric_config = result.setdefault('metric_config', {})

    if not raw_evaluation_name:
        additional = metric_config.get('additional_details')
        if isinstance(additional, dict):
            raw_evaluation_name = str(additional.get('alphaxiv_y_axis') or '').strip()
    if not raw_evaluation_name:
        raw_evaluation_name = 'Score'

    source_data = result.get('source_data')
    evaluation_name = ''
    if isinstance(source_data, dict):
        evaluation_name = str(source_data.get('dataset_name') or '').strip()
    if not evaluation_name:
        evaluation_id = str(payload.get('evaluation_id') or '')
        if '/' in evaluation_id:
            evaluation_name = evaluation_id.split('/', 1)[0].strip()
    if not evaluation_name:
        evaluation_name = 'alphaxiv'

    return CanonicalPatch(
        evaluation_name=evaluation_name,
        metric_id=_slug(raw_evaluation_name),
        metric_name=raw_evaluation_name,
        metric_kind='score',
        metric_unit=_score_metric_unit(metric_config),
    )


def _apex_agents_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    pass_match = re.fullmatch(
        r'(?P<slice>.+) Pass@(?P<k>\d+)', raw_evaluation_name
    )
    if pass_match is not None:
        k = int(pass_match.group('k'))
        return CanonicalPatch(
            evaluation_name=pass_match.group('slice'),
            metric_id='pass_at_k',
            metric_name=f'Pass@{k}',
            metric_kind='pass_rate',
            metric_unit='proportion',
            metric_parameters={'k': k},
        )

    score_match = re.fullmatch(r'(?P<slice>.+) Mean Score', raw_evaluation_name)
    if score_match is not None:
        return CanonicalPatch(
            evaluation_name=score_match.group('slice'),
            metric_id='mean_score',
            metric_name='Mean Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    return None


def _bfcl_patch(
    raw_evaluation_name: str, metric_config: dict[str, Any]
) -> CanonicalPatch | None:
    metric_id = metric_config.get('metric_id')
    if not isinstance(metric_id, str) or not metric_id.startswith('bfcl.'):
        return None

    parts = metric_id.split('.')
    if len(parts) < 3:
        return None

    eval_slice = '.'.join(parts[1:-1])
    if not eval_slice:
        return None

    if raw_evaluation_name != metric_id and raw_evaluation_name == eval_slice:
        return None

    return CanonicalPatch(evaluation_name=eval_slice)


def _rewardbench_patch(
    raw_evaluation_name: str, metric_config: dict[str, Any]
) -> CanonicalPatch | None:
    description = str(metric_config.get('evaluation_description') or '')
    lowered = description.lower()

    if raw_evaluation_name == 'Score':
        return CanonicalPatch(
            evaluation_name='RewardBench',
            metric_id='rewardbench.score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    if 'accuracy' in lowered:
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )

    if 'score' in lowered:
        return CanonicalPatch(
            metric_id='score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    return None


def _description_metric_patch(metric_config: dict[str, Any]) -> CanonicalPatch | None:
    description = str(metric_config.get('evaluation_description') or '')
    if ' on ' not in description:
        return None
    metric_phrase = description.split(' on ', 1)[0]
    return _metric_from_phrase(metric_phrase)


def _score_suffix_patch(
    benchmark_family: str,
    raw_evaluation_name: str,
    metric_config: dict[str, Any],
) -> CanonicalPatch | None:
    score_match = re.fullmatch(r'(?P<slice>.+?) Score', raw_evaluation_name)
    if score_match is None:
        return None

    eval_slice = score_match.group('slice').strip()
    evaluation_name = (
        benchmark_family if eval_slice.lower() == 'overall' else eval_slice
    )
    namespace = _benchmark_metric_namespace(benchmark_family)
    return CanonicalPatch(
        evaluation_name=evaluation_name,
        metric_id=f'{namespace}.score',
        metric_name='Score',
        metric_kind='score',
        metric_unit=_score_metric_unit(metric_config),
    )


def _single_score_patch(
    benchmark_family: str, metric_config: dict[str, Any]
) -> CanonicalPatch:
    namespace = _benchmark_metric_namespace(benchmark_family)
    return CanonicalPatch(
        metric_id=f'{namespace}.score',
        metric_name='Score',
        metric_kind='score',
        metric_unit=_score_metric_unit(metric_config),
    )


def _theory_of_mind_patch(
    raw_evaluation_name: str, metric_config: dict[str, Any]
) -> CanonicalPatch | None:
    match = re.fullmatch(
        r'(?P<metric>.+?) on (?P<slice>.+?)(?: for scorer .+)?',
        raw_evaluation_name,
    )
    if match is not None:
        patch = _metric_from_phrase(match.group('metric'))
        if patch is not None:
            patch.evaluation_name = match.group('slice').strip()
            return patch

    return _metric_from_phrase(str(metric_config.get('evaluation_description') or ''))


def _helm_patch(
    benchmark_family: str,
    raw_evaluation_name: str,
    metric_config: dict[str, Any],
) -> CanonicalPatch | None:
    description = str(metric_config.get('evaluation_description') or '')

    if raw_evaluation_name.lower().startswith('mean '):
        metric = _metric_from_phrase(raw_evaluation_name)
        if metric is None:
            return None
        metric.evaluation_name = benchmark_family
        return metric

    if ' on ' in description:
        metric_phrase = description.split(' on ', 1)[0]
        return _metric_from_phrase(metric_phrase)

    return None


def _swe_bench_verified_mini_patch(
    benchmark_family: str,
    raw_evaluation_name: str,
    metric_config: dict[str, Any],
) -> CanonicalPatch | None:
    match = re.fullmatch(
        r'(?P<metric>mean|std) on inspect_evals/swe_bench_verified_mini(?: for scorer .+)?',
        raw_evaluation_name,
        flags=re.IGNORECASE,
    )
    if match is None:
        metric_phrase = str(metric_config.get('evaluation_description') or '')
    else:
        metric_phrase = match.group('metric')

    metric_phrase = metric_phrase.strip().lower()
    if metric_phrase == 'mean':
        return CanonicalPatch(
            evaluation_name=benchmark_family,
            metric_id='mean_score',
            metric_name='Mean Score',
            metric_kind='score',
            metric_unit=_score_metric_unit(metric_config),
        )
    if metric_phrase == 'std':
        return CanonicalPatch(
            evaluation_name=benchmark_family,
            metric_id='standard_deviation',
            metric_name='Standard Deviation',
            metric_kind='standard_deviation',
            metric_unit=_score_metric_unit(metric_config),
        )
    return None


def _canonical_patch_for_result(
    benchmark_family: str,
    result: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> CanonicalPatch | None:
    raw_evaluation_name = str(result.get('evaluation_name') or '')
    metric_config = result.setdefault('metric_config', {})

    if benchmark_family == 'alphaxiv':
        return _alphaxiv_patch(payload or {}, result)

    if benchmark_family == 'global-mmlu-lite':
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )

    if benchmark_family == 'hfopenllm_v2':
        metric = HFOPENLLM_METRICS.get(raw_evaluation_name)
        if metric is None:
            return None
        return CanonicalPatch(
            metric_id=metric['metric_id'],
            metric_name=metric['metric_name'],
            metric_kind=metric['metric_kind'],
            metric_unit=metric['metric_unit'],
        )

    if benchmark_family == 'reward-bench':
        return _rewardbench_patch(raw_evaluation_name, metric_config)

    if benchmark_family == 'terminal-bench-2.0':
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Task Resolution Accuracy',
            metric_kind='accuracy',
            metric_unit='percentage',
        )

    if benchmark_family == 'wordle_arena':
        return _wordle_patch(raw_evaluation_name)

    if benchmark_family == 'fibble_arena':
        return _fibble_patch(raw_evaluation_name)

    if re.fullmatch(r'fibble[1-5]_arena', benchmark_family):
        return _fibble_patch(raw_evaluation_name)

    if benchmark_family == 'apex-agents':
        return _apex_agents_patch(raw_evaluation_name)

    if benchmark_family in {'ace', 'apex-v1'}:
        return _score_suffix_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    if benchmark_family == 'bfcl':
        return _bfcl_patch(raw_evaluation_name, metric_config)

    if benchmark_family.startswith('helm_'):
        return _helm_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    if benchmark_family == 'livecodebenchpro':
        return _description_metric_patch(metric_config)

    if benchmark_family == 'swe-bench-verified-mini':
        return _swe_bench_verified_mini_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    if benchmark_family in SINGLE_SCORE_BENCHMARK_FAMILIES or benchmark_family.startswith(
        'tau-bench-2_'
    ):
        return _single_score_patch(benchmark_family, metric_config)

    if benchmark_family == 'IFEval':
        return _ifeval_patch(raw_evaluation_name)

    if benchmark_family == 'agentharm':
        return _agentharm_patch(raw_evaluation_name)

    if benchmark_family == 'cvebench':
        return _generic_inspect_eval_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    if benchmark_family in INSPECT_ACCURACY_BENCHMARK_FAMILIES:
        return _generic_inspect_eval_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    if benchmark_family == 'theory_of_mind':
        return _theory_of_mind_patch(raw_evaluation_name, metric_config)

    return None


def _apply_patch(
    result: dict[str, Any], patch: CanonicalPatch | None
) -> tuple[bool, str]:
    raw_evaluation_name = str(result.get('evaluation_name') or '')
    if patch is None:
        return False, raw_evaluation_name

    metric_config = result.setdefault('metric_config', {})
    changed = False

    if (
        patch.evaluation_name is not None
        and result.get('evaluation_name') != patch.evaluation_name
    ):
        _store_raw_evaluation_name(metric_config, raw_evaluation_name)
        result['evaluation_name'] = patch.evaluation_name
        changed = True

    changed |= _set_metric_field(metric_config, 'metric_id', patch.metric_id)
    changed |= _set_metric_field(
        metric_config, 'metric_name', patch.metric_name
    )
    changed |= _set_metric_field(
        metric_config, 'metric_kind', patch.metric_kind
    )
    changed |= _set_metric_field(
        metric_config, 'metric_unit', patch.metric_unit
    )

    if patch.metric_parameters and metric_config.get('metric_parameters') in (
        None,
        {},
    ):
        metric_config['metric_parameters'] = patch.metric_parameters
        changed = True

    return changed, raw_evaluation_name


def _metric_fragment(metric_config: dict[str, Any]) -> str:
    metric_id = str(
        metric_config.get('metric_id')
        or metric_config.get('metric_name')
        or 'score'
    )
    fragment = _slug(metric_id)
    parameters = metric_config.get('metric_parameters')
    if isinstance(parameters, dict):
        for key, value in sorted(parameters.items()):
            fragment += f'__{_slug(str(key))}_{_slug(str(value))}'
    return fragment


def _raw_evaluation_name(result: dict[str, Any]) -> str:
    metric_config = result.get('metric_config', {})
    if not isinstance(metric_config, dict):
        return str(result.get('evaluation_name') or '')

    additional = metric_config.get('additional_details')
    if isinstance(additional, dict):
        raw_name = additional.get('raw_evaluation_name')
        if raw_name:
            return str(raw_name)

    return str(result.get('evaluation_name') or '')


def _assign_evaluation_result_ids(
    payload: dict[str, Any],
    sample_updates: dict[str, dict[str, str]],
) -> bool:
    evaluation_id = str(payload.get('evaluation_id') or '')
    counts: dict[str, int] = {}
    changed = False

    for result in payload.get('evaluation_results', []):
        raw_evaluation_name = _raw_evaluation_name(result)
        eval_name = str(result.get('evaluation_name') or '')
        base = (
            f'{evaluation_id}'
            f'#{_slug(eval_name)}'
            f'#{_metric_fragment(result.get("metric_config", {}))}'
        )
        counts[base] = counts.get(base, 0) + 1
        evaluation_result_id = base
        if counts[base] > 1:
            evaluation_result_id = f'{base}__{counts[base]}'

        if result.get('evaluation_result_id') != evaluation_result_id:
            result['evaluation_result_id'] = evaluation_result_id
            changed = True

        sample_updates[raw_evaluation_name] = {
            'evaluation_name': eval_name,
            'evaluation_result_id': evaluation_result_id,
            'evaluation_id': evaluation_id,
        }

    return changed


def _add_sample_alias_updates(
    benchmark_family: str,
    payload: dict[str, Any],
    sample_updates: dict[str, dict[str, str]],
) -> None:
    if benchmark_family != 'swe-bench-verified-mini':
        return

    evaluation_id = str(payload.get('evaluation_id') or '')
    mean_result = None
    for result in payload.get('evaluation_results', []):
        metric_config = result.get('metric_config', {})
        if metric_config.get('metric_id') == 'mean_score':
            mean_result = result
            break

    if mean_result is None:
        return

    alias_update = {
        'evaluation_name': str(
            mean_result.get('evaluation_name') or benchmark_family
        ),
        'evaluation_result_id': str(
            mean_result.get('evaluation_result_id') or ''
        ),
        'evaluation_id': evaluation_id,
    }
    for raw_name in (
        'inspect_evals/swe_bench_verified_mini',
        'swe_bench_verified_mini',
    ):
        sample_updates[raw_name] = alias_update.copy()


def infer_benchmark_family(
    file_path: Path, payload: dict[str, Any] | None = None
) -> str:
    parts = list(file_path.parts)
    if 'data' in parts:
        idx = parts.index('data')
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if payload is None:
        return 'unknown'

    evaluation_id = str(payload.get('evaluation_id') or '')
    if '/' in evaluation_id:
        return evaluation_id.split('/', 1)[0]
    return 'unknown'


def _ensure_eval_library(
    payload: dict[str, Any], benchmark_family: str
) -> bool:
    if benchmark_family != 'alphaxiv':
        return False

    changed = False
    eval_library = payload.get('eval_library')
    if not isinstance(eval_library, dict):
        eval_library = {}
        payload['eval_library'] = eval_library
        changed = True

    if eval_library.get('name') in (None, ''):
        eval_library['name'] = 'alphaxiv'
        changed = True
    if eval_library.get('version') in (None, ''):
        eval_library['version'] = 'unknown'
        changed = True

    return changed


def augment_aggregate_payload(
    payload: dict[str, Any], benchmark_family: str | None = None
) -> tuple[dict[str, Any], int, dict[str, dict[str, str]], bool]:
    benchmark_family = benchmark_family or infer_benchmark_family(
        Path('.'), payload
    )
    changed_results = 0
    sample_updates: dict[str, dict[str, str]] = {}
    payload_changed = _ensure_eval_library(payload, benchmark_family)

    for result in payload.get('evaluation_results', []):
        patch = _canonical_patch_for_result(
            benchmark_family, result, payload=payload
        )
        changed, _ = _apply_patch(result, patch)
        if changed:
            changed_results += 1

    ids_changed = _assign_evaluation_result_ids(payload, sample_updates)
    _add_sample_alias_updates(benchmark_family, payload, sample_updates)
    return payload, changed_results, sample_updates, (
        ids_changed or payload_changed
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def canonicalize_datastore_path(file_path: Path) -> Path:
    parts = list(file_path.parts)
    if 'data' not in parts:
        return file_path

    idx = parts.index('data')
    relative = parts[idx + 1 :]
    if len(relative) <= 4:
        return file_path

    benchmark_family = relative[0]
    developer = relative[-3]
    model_name = relative[-2]
    filename = relative[-1]
    return Path(*parts[: idx + 1], benchmark_family, developer, model_name, filename)


def _move_file(source: Path, target: Path) -> None:
    if source == target:
        return
    if target.exists():
        raise FileExistsError(
            f'Cannot normalize path for {source}: target already exists at {target}'
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def augment_sample_rows(
    rows: list[dict[str, Any]],
    sample_updates: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], bool]:
    changed = False

    for row in rows:
        raw_evaluation_name = str(row.get('evaluation_name') or '')
        update = sample_updates.get(raw_evaluation_name)
        if update is None:
            continue

        if row.get('evaluation_name') != update['evaluation_name']:
            row['evaluation_name'] = update['evaluation_name']
            changed = True

        if row.get('evaluation_result_id') != update['evaluation_result_id']:
            row['evaluation_result_id'] = update['evaluation_result_id']
            changed = True

        updated_evaluation_id = update.get('evaluation_id')
        if (
            updated_evaluation_id is not None
            and row.get('evaluation_id') != updated_evaluation_id
        ):
            row['evaluation_id'] = updated_evaluation_id
            changed = True

    return rows, changed


def augment_aggregate_file(
    file_path: Path,
    *,
    write_changes: bool = False,
    update_samples: bool = True,
    normalize_paths: bool = True,
) -> AugmentationReport:
    original_file_path = file_path
    payload = json.loads(file_path.read_text(encoding='utf-8'))
    benchmark_family = infer_benchmark_family(file_path, payload)
    payload, changed_results, sample_updates, ids_changed = (
        augment_aggregate_payload(payload, benchmark_family=benchmark_family)
    )
    aggregate_changed = changed_results > 0 or ids_changed

    sample_file_path = file_path.with_name(f'{file_path.stem}_samples.jsonl')
    sample_exists = sample_file_path.exists()
    sample_changed = False
    sample_rows = 0

    if update_samples and sample_exists:
        rows = _read_jsonl(sample_file_path)
        sample_rows = len(rows)
        rows, sample_changed = augment_sample_rows(rows, sample_updates)

        detailed = payload.get('detailed_evaluation_results')
        if (
            isinstance(detailed, dict)
            and detailed.get('total_rows') != sample_rows
        ):
            detailed['total_rows'] = sample_rows
            aggregate_changed = True

        if write_changes and sample_changed:
            _write_jsonl(sample_file_path, rows)

    if write_changes and aggregate_changed:
        _write_json(file_path, payload)

    target_file_path = (
        canonicalize_datastore_path(file_path) if normalize_paths else file_path
    )
    target_sample_file_path = (
        canonicalize_datastore_path(sample_file_path)
        if sample_exists and normalize_paths
        else sample_file_path
    )
    path_changed = target_file_path != file_path

    reported_file_path = target_file_path if path_changed else file_path
    reported_sample_file_path = (
        target_sample_file_path if sample_exists else None
    )

    if write_changes and path_changed:
        if sample_exists and target_sample_file_path is not None:
            _move_file(sample_file_path, target_sample_file_path)
        _move_file(file_path, target_file_path)

    return AugmentationReport(
        original_file_path=original_file_path,
        file_path=reported_file_path,
        benchmark_family=benchmark_family,
        aggregate_changed=aggregate_changed,
        sample_file_path=reported_sample_file_path,
        sample_changed=sample_changed,
        changed_results=changed_results,
        sample_rows=sample_rows,
        path_changed=path_changed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever augment-canonical-identity',
        description=(
            'Backfill metric identity and canonical eval names into existing '
            'EEE aggregate JSON files.'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Aggregate JSON files or directories containing datastore files.',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Write changes in place. Without this flag, run as a dry run.',
    )
    parser.add_argument(
        '--skip-samples',
        action='store_true',
        help='Do not update companion *_samples.jsonl files.',
    )
    parser.add_argument(
        '--skip-path-normalization',
        action='store_true',
        help='Do not flatten over-nested datastore paths to benchmark/developer/model.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    aggregate_files = [
        path
        for path in expand_paths(args.paths)
        if path.suffix == '.json' and not path.name.endswith('_samples.json')
    ]

    reports = [
        augment_aggregate_file(
            path,
            write_changes=args.write,
            update_samples=not args.skip_samples,
            normalize_paths=not args.skip_path_normalization,
        )
        for path in aggregate_files
    ]

    changed_reports = [report for report in reports if report.changed]

    action = 'Updated' if args.write else 'Would update'
    for report in changed_reports:
        details = [
            f'results={report.changed_results}',
            f'sample_rows={report.sample_rows}',
            f'samples_changed={report.sample_changed}',
        ]
        if report.path_changed:
            details.insert(
                0, f'path={report.original_file_path} -> {report.file_path}'
            )
        print(
            f'{action}: {report.file_path} '
            f'[{report.benchmark_family}] '
            f'({", ".join(details)})'
        )

    print(
        f'Scanned {len(reports)} aggregate file(s); '
        f'{len(changed_reports)} would change.'
        if not args.write
        else f'Scanned {len(reports)} aggregate file(s); '
        f'updated {len(changed_reports)}.'
    )
    return 0
