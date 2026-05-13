import json
from pathlib import Path
from typing import Any, List, Tuple

_HELM_IMPORT_ERROR: Exception | None = None
try:
    from helm.benchmark.adaptation.scenario_state import RequestState
except (
    Exception
) as ex:  # pragma: no cover - exercised only when optional deps missing
    _HELM_IMPORT_ERROR = ex
    RequestState = Any  # type: ignore[assignment]


def _require_helm_dependencies() -> None:
    if _HELM_IMPORT_ERROR is not None:
        raise ImportError(
            'HELM converter dependencies are missing. '
            "Install with: pip install 'every_eval_ever[helm]'"
        ) from _HELM_IMPORT_ERROR


from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.utils import sha256_string
from every_eval_ever.converters.helm.utils import extract_all_reasonings
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
    Performance,
    TokenUsage,
)


def _score_from_stat(stat) -> float | None:
    value = getattr(stat, 'mean', None)
    if value is None:
        count = getattr(stat, 'count', None)
        total = getattr(stat, 'sum', None)
        if count:
            try:
                value = total / count
            except (TypeError, ValueError, ZeroDivisionError):
                return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stat_name_part(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return value.get('name') or str(value)
    return getattr(value, 'name', None) or str(value)


def _evaluation_result_id(
    metric_name: str | None,
    split=None,
    perturbation=None,
) -> str | None:
    if metric_name is None:
        return None
    parts = [metric_name]
    split_part = _stat_name_part(split)
    perturbation_part = _stat_name_part(perturbation)
    if split_part:
        parts.append(split_part)
    if perturbation_part:
        parts.append(perturbation_part)
    return ':'.join(parts)


# Metric names whose per-instance score is a correctness signal in [0, 1]
# where ``score > 0`` reasonably maps to ``is_correct=True``. Anything not
# in this allowlist (token counts, runtime, finish-reason flags, logprobs,
# etc.) gets ``is_correct=False`` because we have no correctness claim
# from a bookkeeping/resource metric. Keep this list tight and named after
# the actual HELM stat names — broaden only for verified correctness
# semantics.
_BINARY_CORRECTNESS_METRIC_NAMES: frozenset[str] = frozenset({
    'exact_match',
    'quasi_exact_match',
    'prefix_exact_match',
    'quasi_prefix_exact_match',
    'exact_match@5',
    'quasi_exact_match@5',
    'prefix_exact_match@5',
    'quasi_prefix_exact_match@5',
    'ifeval_strict_accuracy',
    'chain_of_thought_correctness',
    'math_equiv',
    'math_equiv_chain_of_thought',
})


def _is_correct_for_metric(metric_name: str | None, score: float) -> bool:
    """Decide ``is_correct`` honestly per metric name.

    For correctness metrics in the allowlist, the HELM convention is that
    score==1.0 means correct and 0.0 means wrong, so any positive score
    rounds up to "correct". For anything else (bookkeeping / resource
    stats, or graded metrics like rouge_l/bleu where >0 is not a correctness
    signal) we deliberately do not claim correctness.
    """
    if metric_name is None:
        return False
    if metric_name in _BINARY_CORRECTNESS_METRIC_NAMES:
        return score > 0
    return False


class HELMInstanceLevelDataAdapter:
    def __init__(
        self,
        evaulation_id: str,
        format: str,
        hash_algorithm: str,
        evaluation_dir: str,
    ):
        _require_helm_dependencies()
        self.evaluation_id = evaulation_id
        self.format = format
        self.hash_algorithm = hash_algorithm
        self.evaluation_dir = evaluation_dir
        self.path = f'{evaluation_dir}/{evaulation_id}.{format}'

    def _save_json(self, items: List[InstanceLevelEvaluationLog]):
        eval_dir_path = Path(self.evaluation_dir)
        eval_dir_path.mkdir(parents=True, exist_ok=True)
        path = Path(self.path)

        with path.open('w', encoding='utf-8') as f:
            for item in items:
                json_line = json.dumps(
                    item.model_dump(mode='json'), ensure_ascii=False
                )
                f.write(json_line + '\n')

        print(
            f'Instance-level eval log was successfully saved to {self.path} path.'
        )

    def convert_instance_level_logs(
        self,
        evaluation_name: str,
        model_id: str,
        request_states: List[RequestState],
        per_instance_stats_list: List,
    ) -> Tuple[str, int]:
        instance_level_logs: List[InstanceLevelEvaluationLog] = []
        for state in request_states:
            inst_stats = next(
                (
                    s
                    for s in per_instance_stats_list
                    if s.instance_id == state.instance.id
                ),
                None,
            )

            correct_refs = [
                r.output.text
                for r in state.instance.references
                if 'correct' in r.tags
            ]
            completions = (
                [c.text for c in state.result.completions]
                if state.result and state.result.completions
                else []
            )
            reasoning_traces = extract_all_reasonings(state)
            if isinstance(reasoning_traces, str):
                reasoning_traces = [reasoning_traces]
            if reasoning_traces is None:
                reasoning_traces = []
            reasoning_traces = [
                trace for trace in reasoning_traces if isinstance(trace, str)
            ]

            metric_stats = list(inst_stats.stats) if inst_stats else []
            if not metric_stats:
                correct_completions = sum(
                    1 for c in completions if c.strip() in correct_refs
                )
                fallback_score = (
                    correct_completions / len(completions)
                    if completions
                    else 0.0
                )
                metric_stats = [None]

            token_usage = None
            if inst_stats:
                p_tokens = next(
                    (
                        s.sum
                        for s in inst_stats.stats
                        if s.name.name == 'num_prompt_tokens'
                    ),
                    0,
                )
                c_tokens = next(
                    (
                        s.sum
                        for s in inst_stats.stats
                        if s.name.name == 'num_completion_tokens'
                    ),
                    0,
                )
                o_tokens = next(
                    (
                        s.sum
                        for s in inst_stats.stats
                        if s.name.name == 'num_output_tokens'
                    ),
                    0,
                )

                cot_tokens = int(c_tokens) - int(o_tokens)

                token_usage = TokenUsage(
                    input_tokens=int(p_tokens),
                    output_tokens=int(o_tokens),
                    reasoning_tokens=cot_tokens if cot_tokens else None,
                    total_tokens=int(p_tokens + c_tokens),
                )

            for stat in metric_stats:
                if stat is None:
                    metric_name = None
                    evaluation_result_id = None
                    score = fallback_score
                    # Fallback path: ``score`` here is an exact-match
                    # proxy from completion-vs-reference matching, so
                    # the correctness claim is honest in the same sense
                    # as the legacy single-row behavior.
                    is_correct = score > 0
                else:
                    stat_name = getattr(stat, 'name', None)
                    metric_name = getattr(stat_name, 'name', None)
                    evaluation_result_id = _evaluation_result_id(
                        metric_name,
                        getattr(stat_name, 'split', None),
                        getattr(stat_name, 'perturbation', None),
                    )
                    score = _score_from_stat(stat)
                    if score is None:
                        continue
                    is_correct = _is_correct_for_metric(metric_name, score)
                instance_level_logs.append(
                    InstanceLevelEvaluationLog(
                        schema_version=SCHEMA_VERSION,
                        evaluation_id=self.evaluation_id,
                        model_id=model_id,
                        evaluation_name=evaluation_name,
                        evaluation_result_id=evaluation_result_id,
                        sample_id=str(state.instance.id),
                        sample_hash=sha256_string(
                            state.request.prompt + (correct_refs[0] if correct_refs else '')
                        ),  # TODO use all references
                        interaction_type=InteractionType.single_turn,
                        input=Input(
                            raw=state.request.prompt,
                            reference=correct_refs if correct_refs else [],
                            choices=(
                                list(state.output_mapping.values())
                                if state.output_mapping
                                else [
                                    ref.output.text
                                    for ref in state.instance.references
                                ]
                            ),
                        ),
                        output=Output(
                            raw=completions, reasoning_trace=reasoning_traces
                        ),
                        answer_attribution=[
                            AnswerAttributionItem(
                                turn_idx=0,
                                source='output.raw',
                                extracted_value=state.result.completions[
                                    0
                                ].text.strip()
                                if state.result and state.result.completions
                                else '',
                                extraction_method='exact_match',
                                is_terminal=True,
                            )
                        ],
                        evaluation=Evaluation(
                            score=float(score), is_correct=is_correct
                        ),
                        token_usage=token_usage,
                        performance=Performance(
                            generation_time_ms=state.result.request_time * 1000
                            if state.result.request_time
                            else None
                        ),
                    )
                )

        self._save_json(instance_level_logs)
        return self.path, len(instance_level_logs)
