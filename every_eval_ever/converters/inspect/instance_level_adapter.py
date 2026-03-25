import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_INSPECT_IMPORT_ERROR: Exception | None = None
try:
    from inspect_ai.log import EvalSample, EvalSampleReductions
    from inspect_ai.model import (
        ChatMessage,
        ChatMessageAssistant,
        ChatMessageTool,
        ChatMessageUser,
        ModelUsage,
    )
except (
    Exception
) as ex:  # pragma: no cover - exercised only when optional deps missing
    _INSPECT_IMPORT_ERROR = ex
    ChatMessage = ChatMessageAssistant = ChatMessageTool = ChatMessageUser = (
        ModelUsage
    ) = EvalSample = EvalSampleReductions = Any  # type: ignore[assignment]


def _require_inspect_dependencies() -> None:
    if _INSPECT_IMPORT_ERROR is not None:
        raise ImportError(
            'Inspect converter dependencies are missing. '
            "Install with: pip install 'every_eval_ever[inspect]'"
        ) from _INSPECT_IMPORT_ERROR


from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.utils import sha256_string
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Message,
    Output,
    Performance,
    TokenUsage,
    ToolCall,
)


class InspectInstanceLevelDataAdapter:
    def __init__(
        self,
        evaulation_id: str,
        format: str,
        hash_algorithm: str,
        evaluation_dir: str,
    ):
        _require_inspect_dependencies()
        self.evaluation_id = evaulation_id
        self.format = format
        self.hash_algorithm = hash_algorithm
        self.evaluation_dir = evaluation_dir
        self.path = f'{evaluation_dir}/{evaulation_id}.{format}'

    def _serialize_input(self, raw_input) -> str:
        if isinstance(raw_input, str):
            return raw_input
        parts = []
        for msg in raw_input:
            if not isinstance(msg, ChatMessageUser):
                continue
            content = getattr(msg, 'content', '')
            if isinstance(content, list):
                content = ' '.join(
                    block.text if hasattr(block, 'text') else str(block)
                    for block in content
                )
            parts.append(content)
        return '\n'.join(parts)

    def _parse_content_with_reasoning(
        self, content: List[Any]
    ) -> Tuple[str, str]:
        response = None
        reasoning_trace = None
        for part in content:
            if part.type and part.type == 'reasoning':
                reasoning_trace = part.reasoning  # or part.summary
            elif part.type and part.type == 'text':
                response = part.text

        return response, reasoning_trace

    def _get_token_usage(self, usage: ModelUsage | None):
        return (
            TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_cache_write=usage.input_tokens_cache_write,
                input_tokens_cache_read=usage.input_tokens_cache_read,
                reasoning_tokens=usage.reasoning_tokens,
            )
            if usage
            else None
        )

    def _handle_chat_messages(
        self, turn_idx: int, message: ChatMessage
    ) -> Message:
        role = message.role
        content = message.content
        reasoning = None
        if isinstance(content, List):
            content, reasoning = self._parse_content_with_reasoning(content)

        tool_calls: List[ToolCall] = []
        tool_call_id = None

        if isinstance(message, ChatMessageAssistant):
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function,
                    arguments={
                        str(k): json.dumps(v)
                        for k, v in tool_call.arguments.items()
                    }
                    if tool_call.arguments
                    else None,
                )
                for tool_call in message.tool_calls or []
            ]

        if isinstance(message, ChatMessageUser) or isinstance(
            message, ChatMessageTool
        ):
            tool_call_id = (
                [message.tool_call_id]
                if isinstance(message.tool_call_id, str)
                else message.tool_call_id
            )

        return Message(
            turn_idx=turn_idx,
            role=role,
            content=content,
            reasoning_trace=reasoning,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    def _save_json(self, items: list[InstanceLevelEvaluationLog]):
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

    def _normalize_sample_id(self, sample_id: Any) -> str:
        return '' if sample_id is None else str(sample_id)

    def _parse_score_value(
        self, value: Any
    ) -> Tuple[float | None, bool]:
        if isinstance(value, bool):
            return (1.0 if value else 0.0), True

        if isinstance(value, (int, float)):
            score = float(value)
            return score, score in {0.0, 1.0}

        if isinstance(value, str):
            normalized = value.strip()
            normalized_upper = normalized.upper()
            if normalized_upper == 'C':
                return 1.0, True
            if normalized_upper == 'I':
                return 0.0, True
            if normalized_upper == 'TRUE':
                return 1.0, True
            if normalized_upper == 'FALSE':
                return 0.0, True

            try:
                score = float(normalized)
            except (TypeError, ValueError):
                return None, False

            return score, score in {0.0, 1.0}

        return None, False

    def _build_reduction_lookups(
        self, reductions: List[EvalSampleReductions] | None
    ) -> Tuple[
        Dict[Tuple[str, str], Tuple[float, bool]],
        Dict[str, List[Tuple[float, bool]]],
    ]:
        reductions_by_sample_and_scorer: Dict[
            Tuple[str, str], Tuple[float, bool]
        ] = {}
        reductions_by_sample: Dict[str, List[Tuple[float, bool]]] = {}

        if not reductions:
            return reductions_by_sample_and_scorer, reductions_by_sample

        for reduction in reductions:
            scorer_name = self._normalize_sample_id(
                getattr(reduction, 'scorer', None)
            )
            reduced_samples = getattr(reduction, 'samples', None) or []

            for reduced_sample in reduced_samples:
                sample_id = self._normalize_sample_id(
                    getattr(reduced_sample, 'sample_id', None)
                )
                if not sample_id:
                    continue

                parsed_score, parsed_is_binary = self._parse_score_value(
                    getattr(reduced_sample, 'value', None)
                )
                if parsed_score is None:
                    continue

                reductions_by_sample[sample_id] = (
                    reductions_by_sample.get(sample_id, [])
                    + [(parsed_score, parsed_is_binary)]
                )

                if scorer_name:
                    reductions_by_sample_and_scorer[(sample_id, scorer_name)] = (
                        parsed_score,
                        parsed_is_binary,
                    )

        return reductions_by_sample_and_scorer, reductions_by_sample

    def _resolve_evaluation_score(
        self,
        sample: EvalSample,
        response_in_reference: bool,
        reductions_by_sample_and_scorer: Dict[
            Tuple[str, str], Tuple[float, bool]
        ],
        reductions_by_sample: Dict[str, List[Tuple[float, bool]]],
    ) -> Tuple[float, bool]:
        sample_id = self._normalize_sample_id(getattr(sample, 'id', None))

        if sample.scores:
            for scorer_name in sample.scores.keys():
                scorer_key = self._normalize_sample_id(scorer_name)
                matched = reductions_by_sample_and_scorer.get(
                    (sample_id, scorer_key)
                )
                if matched is not None:
                    score, _ = matched
                    return score, False

        sample_reduction_scores = reductions_by_sample.get(sample_id, [])
        unique_sample_reduction_scores = set(sample_reduction_scores)
        if len(unique_sample_reduction_scores) == 1:
            score, _ = unique_sample_reduction_scores.pop()
            return score, False

        if sample.scores:
            for score in sample.scores.values():
                parsed_score, _ = self._parse_score_value(
                    getattr(score, 'value', None)
                )
                if parsed_score is not None:
                    return parsed_score, False

        fallback_score = 1.0 if response_in_reference else 0.0
        return fallback_score, True

    def convert_instance_level_logs(
        self,
        evaluation_name: str,
        model_id: str,
        samples: List[EvalSample],
        reductions: List[EvalSampleReductions] | None = None,
    ) -> Tuple[str, int]:
        instance_level_logs: List[InstanceLevelEvaluationLog] = []
        reductions_by_sample_and_scorer, reductions_by_sample = (
            self._build_reduction_lookups(reductions)
        )

        for sample in samples:
            sample_input = Input(
                raw=self._serialize_input(sample.input),
                reference=[sample.target]
                if isinstance(sample.target, str)
                else list(sample.target),
                choices=sample.choices,
            )

            reasoning_trace = None
            message = sample.output.choices[0].message
            content = message.content

            if isinstance(content, list):
                response, reasoning_trace = self._parse_content_with_reasoning(
                    content
                )
            else:
                response = content

            if sample.scores:
                # TODO Consider multiple scores
                for scorer_name, score in sample.scores.items():
                    if score.answer:
                        response = score.answer
                    elif score.explanation:
                        response = score.explanation

            processed_messages = [
                self._handle_chat_messages(msg_idx, msg)
                for msg_idx, msg in enumerate(sample.messages)
            ]

            counted_assistant_roles = sum(
                [msg.role.lower() == 'assistant' for msg in processed_messages]
            )
            counted_tool_roles = sum(
                [msg.role.lower() == 'tool' for msg in processed_messages]
            )

            if counted_tool_roles:
                interaction_type = InteractionType.agentic
            elif counted_assistant_roles > 1:
                interaction_type = InteractionType.multi_turn
            else:
                interaction_type = InteractionType.single_turn

            if interaction_type == InteractionType.single_turn:
                sample_output = Output(
                    raw=[response]
                    if isinstance(response, str)
                    else list(response),
                    reasoning_trace=[reasoning_trace]
                    if isinstance(reasoning_trace, str)
                    else reasoning_trace,
                )
                messages = None
            else:
                sample_output = None
                messages = processed_messages

            response_in_reference = response in sample_input.reference
            (
                evaluation_score,
                is_fallback_score,
            ) = self._resolve_evaluation_score(
                sample,
                response_in_reference,
                reductions_by_sample_and_scorer,
                reductions_by_sample,
            )
            is_correct = (
                response_in_reference
                if is_fallback_score
                else evaluation_score > 0
            )

            evaluation = Evaluation(
                score=evaluation_score,
                is_correct=is_correct,
                num_turns=len(messages) if messages else 1,
                tool_calls_count=sum(
                    len(msg.tool_calls) if msg.tool_calls else 0
                    for msg in messages
                )
                if messages
                else 0,
            )

            answer_attribution: List[AnswerAttributionItem] = []

            token_usage = self._get_token_usage(sample.output.usage)

            if sample.total_time and sample.working_time:
                performance = Performance(
                    latency_ms=int(
                        (sample.total_time - sample.working_time) * 1000
                    ),
                    generation_time_ms=int(sample.working_time * 1000),
                )
            else:
                performance = None

            instance_level_log = InstanceLevelEvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=self.evaluation_id,
                model_id=model_id,
                evaluation_name=evaluation_name,
                sample_id=str(sample.id),
                sample_hash=sha256_string(
                    sample_input.raw + ''.join(sample_input.reference)
                ),
                interaction_type=interaction_type,
                input=sample_input,
                output=sample_output,
                messages=messages,
                answer_attribution=answer_attribution,
                evaluation=evaluation,
                token_usage=token_usage,
                performance=performance,
                error=f'{sample.error.message}\n{sample.error.traceback}'
                if sample.error
                else None,
                metadata={
                    'stop_reason': str(sample.output.stop_reason)
                    if sample.output.stop_reason
                    else '',
                    'epoch': str(sample.epoch),
                },
            )

            instance_level_logs.append(instance_level_log)

        self._save_json(instance_level_logs)

        return self.path, len(instance_level_logs)
