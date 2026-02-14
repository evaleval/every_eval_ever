import json

from helm.benchmark.adaptation.scenario_state import (
    RequestState
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
)

from eval_converters import SCHEMA_VERSION
from eval_converters.common.utils import sha256_string
from eval_converters.helm.utils import extract_all_reasonings


class HELMInstanceLevelDataAdapter:
    def __init__(
        self, 
        evaulation_id: str, 
        format: str, 
        hash_algorithm: str, 
        evaluation_dir: str
    ):
        self.evaluation_id = evaulation_id
        self.format = format
        self.hash_algorithm = hash_algorithm
        self.evaluation_dir = evaluation_dir
        self.path = f'{evaluation_dir}/{evaulation_id}.{format}'

    def _save_json(
        self,
        items: List[InstanceLevelEvaluationLog]
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
        request_states: List[RequestState],
        per_instance_stats_list: List
    ) -> Tuple[str, int]:
        instance_level_logs: List[InstanceLevelEvaluationLog] = []
        for state in request_states:
            inst_stats = next((s for s in per_instance_stats_list if s.instance_id == state.instance.id), None)
            
            correct_refs = [r.output.text for r in state.instance.references if "correct" in r.tags]
            completions = (
                [c.text for c in state.result.completions] 
                if state.result and state.result.completions
                else []
            )
            reasoning_traces = extract_all_reasonings(state)

            is_correct = False
            score = 0.0
            if inst_stats:
                em_stat = next((s for s in inst_stats.stats if s.name.name == "exact_match"), None)
                if em_stat:
                    score = em_stat.mean
                    is_correct = em_stat.mean > 0
                else: # TODO check for more specific tasks
                    correct_completions = sum(1 for c in completions if c.strip() in correct_refs)
                    score = correct_completions / len(completions)
                    is_correct = score > 0
                    
            
            token_usage = None
            if inst_stats:
                p_tokens = next((s.sum for s in inst_stats.stats if s.name.name == "num_prompt_tokens"), 0)
                c_tokens = next((s.sum for s in inst_stats.stats if s.name.name == "num_completion_tokens"), 0)
                o_tokens = next((s.sum for s in inst_stats.stats if s.name.name == "num_output_tokens"), 0)

                cot_tokens = int(c_tokens) - int(o_tokens)
                
                token_usage = TokenUsage(
                    input_tokens=int(p_tokens),
                    output_tokens=int(o_tokens),
                    reasoning_tokens=cot_tokens if cot_tokens else None,
                    total_tokens=int(p_tokens + c_tokens)
                )
            
                # messages: Optional[List[Dict[str, str]]] = None
                # """Used for chat models.
                # If messages is specified for a chat model, the prompt is ignored.
                # Otherwise, the client should convert the prompt into a message."""

            instance_level_logs.append(InstanceLevelEvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=self.evaluation_id,
                model_id=model_id,
                evaluation_name=evaluation_name,
                sample_id=state.instance.id,
                sample_hash=sha256_string(state.request.prompt + correct_refs[0]), # TODO use all references
                interaction_type=InteractionType.single_turn,
                input=Input(
                    raw=state.request.prompt,
                    reference=correct_refs if correct_refs else [],
                    choices=(
                        state.output_mapping.values()
                        if state.output_mapping
                        else [ref.output.text for ref in state.instance.references]
                    )
                ),
                output=Output(
                    raw=completions,
                    reasoning_trace=reasoning_traces
                ),
                answer_attribution=[AnswerAttributionItem(
                    turn_idx=0,
                    source="output.raw",
                    extracted_value=state.result.completions[0].text.strip() if state.result and state.result.completions else "",
                    extraction_method="exact_match",
                    is_terminal=True
                )],
                evaluation=Evaluation(score=float(score), is_correct=is_correct),
                token_usage=token_usage,
                performance=Performance(
                    generation_time_ms=state.result.request_time * 1000 if state.result.request_time else None
                )
            ))

        self._save_json(instance_level_logs)
        return self.path, len(instance_level_logs)