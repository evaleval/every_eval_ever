"""Instance-level adapter for converting DeepEval per-case metric data."""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Optional, Union

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
    Output,
    Performance,
    TokenUsage,
)

from .utils import evaluation_result_id, is_int, is_number


class DeepEvalInstanceLevelAdapter:
    """Converts DeepEval per-test-case metricsData to instance-level rows.

    One row per (test case, metric) pair that carries a finite numeric
    score and a producer verdict (a ``success`` flag). DeepEval counts
    pairs without a success flag as errors and excludes their scores
    from its run summary, so they get no row here either; non-finite
    scores are likewise unpublished here and are counted on the
    aggregate side. Conversational test cases are not expanded into
    rows; their scores are already inside the run's aggregate
    metricsScores.
    """

    def transform_samples(
        self,
        raw_data: dict[str, Any],
        evaluation_id: str,
        model_id: str,
        evaluation_name: str,
    ) -> list[InstanceLevelEvaluationLog]:
        """Transform one test run's testCases into instance-level logs."""
        rows: list[InstanceLevelEvaluationLog] = []
        for index, case in enumerate(raw_data.get('testCases') or []):
            if not isinstance(case, dict):
                continue
            rows.extend(
                self._transform_case(
                    case, index, evaluation_id, model_id, evaluation_name
                )
            )
        return rows

    def transform_and_save(
        self,
        raw_data: dict[str, Any],
        evaluation_id: str,
        model_id: str,
        evaluation_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        file_uuid: Optional[str] = None,
        collection: Optional[str] = None,
        developer: Optional[str] = None,
    ) -> Optional[DetailedEvaluationResults]:
        """Transform test cases and save to JSONL, returning a pointer.

        Mirrors the lm-eval instance adapter's contract: if output_dir is
        None, returns None (skips instance-level output). Otherwise
        file_uuid and collection are required so the samples file shares
        the aggregate UUID and declares its canonical location under
        data/.
        """
        if output_dir is None:
            return None
        if file_uuid is None:
            raise ValueError(
                'file_uuid is required when writing DeepEval samples'
            )
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

        logs = self.transform_samples(
            raw_data, evaluation_id, model_id, evaluation_name
        )
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

    def _transform_case(
        self,
        case: dict[str, Any],
        index: int,
        evaluation_id: str,
        model_id: str,
        evaluation_name: str,
    ) -> list[InstanceLevelEvaluationLog]:
        input_text = case.get('input') or ''
        expected = case.get('expectedOutput')
        actual = case.get('actualOutput')
        reference = [str(expected)] if expected is not None else []

        # Build sample hash from input + reference for cross-model comparison
        hash_input = json.dumps(
            {'raw': input_text, 'reference': reference[0] if reference else ''},
            sort_keys=True,
        )
        sample_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        token_usage = self._token_usage(case)
        performance = self._performance(case)

        rows = []
        for metric_data in case.get('metricsData') or []:
            if not isinstance(metric_data, dict):
                continue
            score = metric_data.get('score')
            if not is_number(score):
                # Entries with no finite score get no row: errored
                # metrics and classifier entries carry no score, and
                # non-finite scores (NaN or infinite) cannot be
                # published; the aggregate counts the run's errors and
                # its excluded non-finite scores separately.
                continue
            success = metric_data.get('success')
            if not isinstance(success, bool):
                # The producer recorded no verdict for this pair (it
                # counts the pair as an error and excludes its score
                # from the run summary), so no row is published rather
                # than deriving a verdict from the score or threshold.
                continue
            is_correct = success
            display_name = metric_data.get('name') or ''
            threshold = metric_data.get('threshold')

            metadata: dict[str, str] = {'metric': display_name}
            if case.get('name'):
                metadata['test_case_name'] = str(case['name'])
            if is_number(threshold):
                metadata['threshold'] = str(threshold)
            if metric_data.get('reason'):
                metadata['reason'] = str(metric_data['reason'])
            if metric_data.get('evaluationModel'):
                metadata['evaluation_model'] = str(
                    metric_data['evaluationModel']
                )
            if case.get('tags'):
                metadata['tags'] = json.dumps(case['tags'])

            rows.append(
                InstanceLevelEvaluationLog(
                    schema_version=SCHEMA_VERSION,
                    evaluation_id=evaluation_id,
                    model_id=model_id,
                    evaluation_name=evaluation_name,
                    evaluation_result_id=evaluation_result_id(display_name),
                    sample_id=str(
                        case['order'] if is_int(case.get('order')) else index
                    ),
                    sample_hash=sample_hash,
                    interaction_type=InteractionType.single_turn,
                    input=Input(raw=input_text, reference=reference),
                    output=(
                        Output(raw=[str(actual)])
                        if actual is not None
                        else None
                    ),
                    answer_attribution=[
                        AnswerAttributionItem(
                            turn_idx=0,
                            source='output.raw',
                            extracted_value=(
                                str(actual) if actual is not None else ''
                            ),
                            extraction_method='none',
                            is_terminal=True,
                        )
                    ],
                    evaluation=Evaluation(
                        score=float(score),
                        is_correct=is_correct,
                    ),
                    token_usage=token_usage,
                    performance=performance,
                    metadata=metadata,
                )
            )
        return rows

    @staticmethod
    def _token_usage(case: dict[str, Any]) -> Optional[TokenUsage]:
        """Case-level token counts the run recorded, when it recorded any.

        TokenUsage requires all three counts; DeepEval records the input
        and output counts together or not at all, so a lone count is
        paired with 0 rather than dropped.
        """
        input_tokens = case.get('inputTokenCount')
        output_tokens = case.get('outputTokenCount')
        valid_input = is_int(input_tokens) and input_tokens >= 0
        valid_output = is_int(output_tokens) and output_tokens >= 0
        if not valid_input and not valid_output:
            return None
        input_value = input_tokens if valid_input else 0
        output_value = output_tokens if valid_output else 0
        return TokenUsage(
            input_tokens=input_value,
            output_tokens=output_value,
            total_tokens=input_value + output_value,
        )

    @staticmethod
    def _performance(case: dict[str, Any]) -> Optional[Performance]:
        completion_time = case.get('completionTime')
        if not is_number(completion_time) or completion_time < 0:
            return None
        return Performance(latency_ms=float(completion_time) * 1000.0)
