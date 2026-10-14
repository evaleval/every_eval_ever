"""Instance-level adapter for SimpleAudit per-scenario results.

One row per entry of a run's `results[]`: the scenario is a (usually)
multi-turn conversation the auditor ran against the target, so rows use
`interaction_type="multi_turn"` with the full transcript in `messages`.
"""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    Format,
    HashAlgorithm,
)
from every_eval_ever.helpers.io import datastore_repo_file_path
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Message,
    TokenUsage,
)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


class SimpleAuditInstanceLevelAdapter:
    """Converts SimpleAudit per-scenario results to instance-level logs."""

    def transform_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        evaluation_id: str,
        model_id: str,
    ) -> List[InstanceLevelEvaluationLog]:
        """Transform one run's `results[]` into instance-level logs."""
        return [
            self._transform_scenario(scenario, evaluation_id, model_id)
            for scenario in scenarios
        ]

    def transform_and_save(
        self,
        scenarios: List[Dict[str, Any]],
        evaluation_id: str,
        model_id: str,
        output_dir: Union[str, Path],
        file_uuid: str,
        collection: str,
        developer: Optional[str] = None,
    ) -> Optional[DetailedEvaluationResults]:
        """Write the sidecar JSONL and return its DetailedEvaluationResults pointer.

        The sidecar shares the aggregate log's UUID and lives beside it under
        `data/<collection>/<developer>/<model>/`, which is what the validator's
        canonical-path checks expect.
        """
        try:
            parsed_uuid = uuid.UUID(file_uuid)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f'invalid file_uuid: {file_uuid!r}') from exc
        if parsed_uuid.version != 4:
            raise ValueError(f'file_uuid must be UUIDv4: {file_uuid!r}')
        file_uuid = str(parsed_uuid)
        expected_name = f'{file_uuid}_samples.jsonl'
        repository_file_path = datastore_repo_file_path(
            collection,
            model_id,
            developer,
            expected_name,
        )

        logs = self.transform_scenarios(scenarios, evaluation_id, model_id)
        if not logs:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / expected_name
        serialized = '\n'.join(
            json.dumps(
                log.model_dump(mode='json'),
                ensure_ascii=False,
                allow_nan=False,
            )
            for log in logs
        )
        out_file.write_text(serialized + '\n', encoding='utf-8')

        file_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()

        return DetailedEvaluationResults(
            format=Format.jsonl,
            file_path=repository_file_path,
            hash_algorithm=HashAlgorithm.sha256,
            checksum=file_hash,
            total_rows=len(logs),
        )

    def _transform_scenario(
        self,
        scenario: Dict[str, Any],
        evaluation_id: str,
        model_id: str,
    ) -> InstanceLevelEvaluationLog:
        """Transform one `results[]` entry into an instance-level log."""
        sample_id = str(scenario.get('scenario_name') or '')
        severity = str(scenario.get('severity') or '')
        conversation = scenario.get('conversation') or []

        messages = [
            Message(
                turn_idx=turn_idx,
                role=str(turn.get('role') or ''),
                content=(
                    str(turn['content'])
                    if turn.get('content') is not None
                    else None
                ),
            )
            for turn_idx, turn in enumerate(conversation)
            if isinstance(turn, dict)
        ]

        # `input.raw` is the scenario's description; when absent, fall back to
        # the first user turn so the field never holds a placeholder.
        raw_input = scenario.get('scenario_description')
        if not isinstance(raw_input, str) or not raw_input.strip():
            fallback = next(
                (
                    turn.get('content')
                    for turn in conversation
                    if isinstance(turn, dict)
                    and turn.get('role') == 'user'
                    and turn.get('content') is not None
                ),
                '',
            )
            raw_input = str(fallback)
        raw_input = raw_input or ''

        # Reference answers: the scenario's expected behavior, if any.
        expected_behavior = scenario.get('expected_behavior') or []
        reference = [str(item) for item in expected_behavior] or ['']

        hash_input = json.dumps(
            {'raw': raw_input, 'reference': reference}, sort_keys=True
        )
        sample_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # Point at the aggregate result this row contributes to: the safety
        # score aggregates over these scenarios (ERROR/ungraded ones are
        # excluded there, and said so in metadata).
        last_assistant_idx = max(
            (
                message.turn_idx
                for message in messages
                if message.role == 'assistant'
            ),
            default=-1,
        )
        attribution_source = (
            f'messages[{last_assistant_idx}].content'
            if last_assistant_idx >= 0
            else 'messages[0].content'
        )
        extracted_value = (
            messages[last_assistant_idx].content
            if last_assistant_idx >= 0
            else (messages[0].content if messages else '')
        ) or ''

        token_usage = None
        input_tokens = _as_int(scenario.get('target_input_tokens'))
        output_tokens = _as_int(scenario.get('target_output_tokens'))
        if input_tokens is not None and output_tokens is not None:
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(
                    _as_int(scenario.get('target_total_tokens'))
                    or input_tokens + output_tokens
                ),
            )

        error = None
        if severity == 'ERROR':
            error = 'scenario failed to grade (SimpleAudit severity ERROR)'

        metadata: Dict[str, str] = {
            'severity': severity,
            'is_graded': 'false' if severity in ('ERROR', 'ungraded') else 'true',
            'issues_found': json.dumps(
                scenario.get('issues_found') or [], ensure_ascii=False
            ),
            'positive_behaviors': json.dumps(
                scenario.get('positive_behaviors') or [], ensure_ascii=False
            ),
            'scenario_summary': str(scenario.get('summary') or ''),
            'scenario_recommendations': json.dumps(
                scenario.get('recommendations') or [], ensure_ascii=False
            ),
            'judgment': json.dumps(
                scenario.get('judgment') or {}, ensure_ascii=False
            ),
            'auditor_input_tokens': str(_as_int(scenario.get('auditor_input_tokens')) or 0),
            'auditor_output_tokens': str(_as_int(scenario.get('auditor_output_tokens')) or 0),
            'judge_input_tokens': str(_as_int(scenario.get('judge_input_tokens')) or 0),
            'judge_output_tokens': str(_as_int(scenario.get('judge_output_tokens')) or 0),
        }

        num_turns = len(messages) or None

        return InstanceLevelEvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_name='safety_score',
            evaluation_result_id='safety_score',
            sample_id=sample_id,
            sample_hash=sample_hash,
            interaction_type=InteractionType.multi_turn,
            input=Input(raw=raw_input, reference=reference),
            output=None,
            messages=messages,
            answer_attribution=[
                AnswerAttributionItem(
                    turn_idx=max(last_assistant_idx, 0),
                    source=attribution_source,
                    extracted_value=extracted_value,
                    extraction_method='custom',
                    is_terminal=True,
                )
            ],
            evaluation=Evaluation(
                # 1.0 for a clean pass, 0.0 otherwise: the record keeps the
                # full severity in metadata, so collapsing here loses nothing
                # a reader cannot recover.
                score=1.0 if severity == 'pass' else 0.0,
                is_correct=severity == 'pass',
                num_turns=num_turns,
            ),
            token_usage=token_usage,
            error=error,
            metadata=metadata,
        )
