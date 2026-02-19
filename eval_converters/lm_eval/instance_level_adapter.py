"""Instance-level adapter for converting lm-eval per-sample logs."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eval_converters import SCHEMA_VERSION
from eval_types import DetailedEvaluationResults, Format, HashAlgorithm
from instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
)


class LMEvalInstanceLevelAdapter:
    """Converts lm-eval per-sample JSONL to instance-level every_eval_ever format."""

    def transform_samples(
        self,
        samples_path: Union[str, Path],
        evaluation_id: str,
        model_id: str,
        task_name: str,
    ) -> List[InstanceLevelEvaluationLog]:
        """Transform a samples JSONL file into instance-level logs."""
        samples_path = Path(samples_path)
        results = []

        with open(samples_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                log = self._transform_sample(
                    sample, evaluation_id, model_id, task_name
                )
                results.append(log)

        return results

    def transform_and_save(
        self,
        samples_path: Union[str, Path],
        evaluation_id: str,
        model_id: str,
        task_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        file_uuid: Optional[str] = None,
    ) -> Optional[DetailedEvaluationResults]:
        """Transform samples and save to JSONL, returning a DetailedEvaluationResults pointer.

        If output_dir is None, returns None (skips instance-level output).
        If file_uuid is provided, the output file is named {file_uuid}_samples.jsonl
        so it shares the UUID of the corresponding evaluation result file.
        """
        if output_dir is None:
            return None

        logs = self.transform_samples(samples_path, evaluation_id, model_id, task_name)
        if not logs:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if file_uuid:
            out_file = output_dir / f"{file_uuid}_samples.jsonl"
        else:
            out_file = output_dir / f"samples_{task_name}.jsonl"

        with open(out_file, "w") as f:
            for log in logs:
                f.write(
                    json.dumps(log.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )

        file_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()

        return DetailedEvaluationResults(
            format=Format.jsonl,
            file_path=str(out_file),
            hash_algorithm=HashAlgorithm.sha256,
            checksum=file_hash,
            total_rows=len(logs),
        )

    def _transform_sample(
        self,
        sample: Dict[str, Any],
        evaluation_id: str,
        model_id: str,
        task_name: str,
    ) -> InstanceLevelEvaluationLog:
        """Transform a single lm-eval sample into an instance-level log."""
        # Extract prompt from arguments
        arguments = sample.get("arguments", {})
        prompt = ""
        if arguments:
            first_arg = arguments.get("gen_args_0", {})
            prompt = first_arg.get("arg_0", "")

        target = str(sample.get("target", ""))

        # Extract model output
        raw_output = self._extract_output(sample)

        # Determine correctness from metric values
        metrics = sample.get("metrics", [])
        score = None
        is_correct = None
        for metric_name in metrics:
            if metric_name in sample:
                val = sample[metric_name]
                if isinstance(val, (int, float)):
                    score = float(val)
                    is_correct = score == 1.0
                    break

        if score is None:
            score = 0.0
            is_correct = False

        # Build sample hash from input + reference for cross-model comparison
        hash_input = json.dumps({"raw": prompt, "reference": target}, sort_keys=True)
        sample_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # Build evaluation_name: include filter if not "none"
        filter_name = sample.get("filter", "none")
        eval_name = task_name
        if filter_name != "none":
            eval_name = f"{task_name}/{filter_name}"

        # Build answer attribution
        # For lm-eval, the answer is always extracted from the model's single-turn output.
        # The extraction_method depends on the filter applied.
        extraction_method = "none"
        if filter_name != "none":
            extraction_method = filter_name

        answer_attribution = [
            AnswerAttributionItem(
                turn_idx=0,
                source="output.raw",
                extracted_value=raw_output,
                extraction_method=extraction_method,
                is_terminal=True,
            )
        ]

        return InstanceLevelEvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_name=eval_name,
            sample_id=str(sample.get("doc_id", 0)),
            sample_hash=sample_hash,
            interaction_type=InteractionType.single_turn,
            input=Input(
                raw=prompt,
                reference=[target],
                choices=self._extract_choices(sample),
            ),
            output=Output(raw=[raw_output]),
            answer_attribution=answer_attribution,
            evaluation=Evaluation(
                score=score,
                is_correct=is_correct,
            ),
            metadata={
                "doc_hash": str(sample.get("doc_hash", "")),
                "prompt_hash": str(sample.get("prompt_hash", "")),
                "target_hash": str(sample.get("target_hash", "")),
                "filter": str(filter_name),
                "lm_eval_metrics": json.dumps({
                    m: sample.get(m) for m in metrics if m in sample
                }),
            },
        )

    def _is_multiple_choice(self, sample: dict[str, Any]) -> bool:
        """Check if a sample is multiple-choice by inspecting the arguments structure."""
        arguments = sample.get("arguments", {})
        return len(arguments) > 1 and "gen_args_1" in arguments

    def _extract_output(self, sample: dict[str, Any]) -> str:
        """Extract the model's output from a sample."""
        filtered_resps = sample.get("filtered_resps", [])
        resps = sample.get("resps", [])

        if self._is_multiple_choice(sample):
            # For multiple-choice, find the selected choice index from filtered_resps.
            # Each entry is [log_prob, is_greedy]; the model picks the highest log_prob.
            source = filtered_resps if filtered_resps else resps
            if not source:
                return ""
            try:
                log_probs = []
                for resp in source:
                    if isinstance(resp, list) and resp:
                        val = resp[0] if isinstance(resp[0], list) else resp
                        log_probs.append(float(val[0]))
                    else:
                        log_probs.append(float("-inf"))
                selected_idx = log_probs.index(max(log_probs))
                # Return the choice text from arguments if available
                choices = self._extract_choices(sample)
                if choices and selected_idx < len(choices):
                    return choices[selected_idx]
                return str(selected_idx)
            except (ValueError, TypeError, IndexError):
                return str(filtered_resps)

        # For generation tasks, use the first response
        source = filtered_resps if filtered_resps else resps
        if not source:
            return ""

        first = source[0]
        if isinstance(first, list):
            return str(first[0]) if first else ""
        return str(first)

    def _extract_choices(self, sample: dict[str, Any]) -> list[str] | None:
        """Extract multiple choice options from arguments structure."""
        arguments = sample.get("arguments", {})
        if not self._is_multiple_choice(sample):
            return None
        # Collect arg_1 (continuation text) from each gen_args_N in order
        choices = []
        idx = 0
        while f"gen_args_{idx}" in arguments:
            arg = arguments[f"gen_args_{idx}"]
            if "arg_1" in arg:
                choices.append(str(arg["arg_1"]).strip())
            idx += 1
        return choices if choices else None
