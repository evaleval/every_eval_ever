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
            sample_id=sample.get("doc_id", 0),
            sample_hash=sample_hash,
            interaction_type=InteractionType.single_turn,
            input=Input(
                raw=prompt,
                references=[target],
                choices=self._extract_choices(sample),
            ),
            output=Output(raw=[raw_output]),
            answer_attribution=answer_attribution,
            evaluation=Evaluation(
                score=score,
                is_correct=is_correct,
            ),
            metadata={
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target_hash": sample.get("target_hash"),
                "filter": filter_name,
                "lm_eval_metrics": {
                    m: sample.get(m) for m in metrics if m in sample
                },
            },
        )

    def _extract_output(self, sample: Dict[str, Any]) -> str:
        """Extract the model's output from a sample."""
        # Prefer filtered_resps (post-filter), fall back to raw resps
        filtered_resps = sample.get("filtered_resps", [])
        resps = sample.get("resps", [])

        source = filtered_resps if filtered_resps else resps
        if not source:
            return ""

        first = source[0]
        if isinstance(first, list):
            return str(first[0]) if first else ""
        return str(first)

    def _extract_choices(self, sample: Dict[str, Any]) -> Optional[List[str]]:
        """Extract multiple choice options if available."""
        doc = sample.get("doc", {})
        for key in ("choices", "options", "answers"):
            if key in doc and isinstance(doc[key], list):
                return [str(c) for c in doc[key]]
        return None
