import json
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelUsage
)
from inspect_ai.log import (
    EvalSample
)
from pathlib import Path
from typing import Any, List, Tuple

from instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    Interaction,
    InteractionType,
    Performance,
    Output,
    TokenUsage,
    ToolCall
)

from eval_converters import SCHEMA_VERSION
from eval_converters.inspect.utils import sha256_string


class InspectInstanceLevelDataAdapter:
    def __init__(self, evaulation_id: str, format: str, hash_algorithm: str, evaluation_dir: str):
        self.evaluation_id = evaulation_id
        self.format = format
        self.hash_algorithm = hash_algorithm
        self.evaluation_dir = evaluation_dir
        self.path = f'{evaluation_dir}/{evaulation_id}.{format}'

    def _parse_content_with_reasoning(
        self,
        content: List[Any]
    ) -> Tuple[str, str]:
        response = None
        reasoning_trace = None
        for part in content:
            if part.type and part.type == "reasoning":
                reasoning_trace = part.reasoning # or part.summary
            elif part.type and part.type == "text":
                response = part.text
        
        return response, reasoning_trace


    def _get_token_usage(self, usage: ModelUsage | None):
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_cache_write=usage.input_tokens_cache_write,
            input_tokens_cache_read=usage.input_tokens_cache_read,
            reasoning_tokens=usage.reasoning_tokens,
        ) if usage else None

    def _handle_chat_messages(
        self,
        turn_idx: int,
        message: ChatMessage
    ) -> Interaction:
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
                    arguments=tool_call.arguments
                )
                for tool_call in message.tool_calls or []
            ]
            
        if isinstance(message, ChatMessageUser) or isinstance(message, ChatMessageTool):
            tool_call_id = message.tool_call_id

        return Interaction(
            turn_idx=turn_idx,
            role=role,
            content=content,
            reasoning_trace=reasoning,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        )

    def _save_json(
        self,
        items: list[InstanceLevelEvaluationLog]
    ):
        eval_dir_path = Path(self.evaluation_dir)
        eval_dir_path.mkdir(parents=True, exist_ok=True)
        path = Path(self.path)

        with path.open("w", encoding="utf-8") as f:
            for item in items:
                json_line = json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=False
                )
                f.write(json_line + "\n")
        
        print(f'Instance-level eval log was successfully saved to {self.path} path.')

    def convert_instance_level_logs(
        self, 
        evaluation_name: str,
        model_id: str,
        samples: List[EvalSample]
    ) -> Tuple[str, int]:
        instance_level_logs: List[InstanceLevelEvaluationLog] = []

        for sample in samples:
            sample_input = Input(
                raw=sample.input,
                reference=sample.target,
                choices=sample.choices
            )

            reasoning_trace = None
            message = sample.output.choices[0].message
            content = message.content

            if isinstance(content, list):
                response, reasoning_trace = self._parse_content_with_reasoning(content)
            else:
                response = content

            if sample.scores:
                # TODO Consider multiple scores
                for scorer_name, score in sample.scores.items():
                    if score.answer:
                        response = score.answer
                    elif score.explanation:
                        response = score.explanation

            sample_output = Output(
                raw=response,
                reasoning_trace=reasoning_trace
            )

            interactions = []
            turn_idx = 0
            for message in sample.messages:
                interactions.append(
                    self._handle_chat_messages(
                        turn_idx,
                        message
                    )
                )

                if message.role == "assistant":
                    turn_idx += 1

            if any(interaction.role.lower() == 'tool' for interaction in interactions):
                interaction_type = InteractionType.agentic
            elif len(interactions) <= 3:
                interaction_type = InteractionType.single_turn
            else:
                interaction_type = InteractionType.multi_turn
                
            evaluation = Evaluation(
                score=1.0 if sample_input.reference == response else 0.0,
                is_correct=sample_input.reference == response,
                num_turns=len(interactions),
                tool_calls_count=sum(
                    len(intr.tool_calls) if intr.tool_calls else 0
                    for intr in interactions
                )
            )

            answer_attribution: List[AnswerAttributionItem] = []

            token_usage = self._get_token_usage(sample.output.usage)

            if sample.total_time and sample.working_time:
                performance = Performance(
                    latency_ms=int((sample.total_time - sample.working_time) * 1000),
                    generation_time_ms=int(sample.working_time * 1000)
                )
            else:
                performance = None

            instance_level_log = InstanceLevelEvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=self.evaluation_id,
                model_id=model_id,
                evaluation_name=evaluation_name,
                sample_id=sample.id,
                sample_hash=sha256_string(sample_input.raw + sample_input.reference),
                interaction_type=interaction_type,
                input=sample_input,
                output=sample_output,
                interactions=interactions,
                answer_attribution=answer_attribution,
                evaluation=evaluation,
                token_usage=token_usage,
                performance=performance,
                error=f'{sample.error.message}\n{sample.error.traceback}' if sample.error else None,
                metadata={
                    'stop_reason': sample.output.stop_reason,
                    'epoch': sample.epoch
                }
            )

            instance_level_logs.append(instance_level_log)

        self._save_json(instance_level_logs)

        return self.path, len(instance_level_logs)