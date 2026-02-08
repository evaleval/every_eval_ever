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
from typing import List, Optional, Tuple

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

from scripts.eval_converters import SCHEMA_VERSION
from scripts.eval_converters.inspect.utils import sha256_string


class InspectInstanceLevelDataAdapter:
    def __init__(self, evaulation_id: str, format: str, hash_algorithm: str):
        self.evaluation_id = evaulation_id
        self.format = format
        self.hash_algorithm = hash_algorithm
        self.path = f'{evaulation_id}.{format}'

    def get_token_usage(self, usage: Optional[ModelUsage]):
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_cache_write=usage.input_tokens_cache_write,
            input_tokens_cache_read=usage.input_tokens_cache_read,
            reasoning_tokens=usage.reasoning_tokens,
        ) if usage else None

    def handle_chat_messages(
        self,
        turn_idx: int,
        message: ChatMessage
    ) -> Interaction:
        role = message.role
        content = message.content
        reasoning = None
        if isinstance(content, List):
            for c in content:
                type = c.type
                if type == 'reasoning':
                    reasoning = c.reasoning
                
            content = str(content)
        
        tool_calls: List[ToolCall] = []
        tool_call_id = None

        if isinstance(message, ChatMessageAssistant):
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function,
                    arguments=tool_call.arguments
                )
                for tool_call in message.tool_calls
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

    def _save_jsonl(
        self,
        items: list[InstanceLevelEvaluationLog]
    ):
        path = Path(self.path)

        with path.open("w", encoding="utf-8") as f:
            for item in items:
                json_line = json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=False
                )
                f.write(json_line + "\n")

    def convert_instance_level_logs(
        self, 
        evaluation_name: str,
        model_id: str,
        samples: List[EvalSample]
    ) -> str:
        instance_level_logs: List[InstanceLevelEvaluationLog] = []
        for sample in samples:
            sample_input = Input(
                raw=sample.input,
                reference=sample.target,
                choices=sample.choices
            )

            if sample.scores:
                # TODO What about multiple scores?
                for scorer_name, score in sample.scores.items():
                    response = score.answer
            else:
                response = sample.output.choices[0].message.content

            sample_output = Output(
                raw=response,
                reasoning_trace=None # TODO
            )

            interactions = []
            for message_idx, message in enumerate(sample.messages):
                interactions.append(
                    self.handle_chat_messages(
                        message_idx,
                        message
                    )
                )

            if len(interactions) == 1:
                interaction_type = InteractionType.single_turn
            else:
                if any(interaction.role.lower() == 'tool' for interaction in interactions):
                    interaction_type = InteractionType.agentic
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

            token_usage = self.get_token_usage(sample.output.usage)

            performance = Performance(
                latency_ms=int((sample.total_time - sample.working_time) * 1000),
                generation_time_ms=int(sample.working_time * 1000)
            )

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
                metadata={'stop_reason': sample.output.stop_reason}
            )

            instance_level_logs.append(instance_level_log)

        self._save_jsonl(instance_level_logs)

        return self.path