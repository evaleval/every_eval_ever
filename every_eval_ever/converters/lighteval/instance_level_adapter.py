"""Instance-level adapter for converting lighteval per-sample details."""

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
)

# The three columns of a lighteval details parquet, one per field of
# DetailsLogger.Detail (lighteval/logging/info_loggers.py).
DOC_COLUMN = 'doc'
RESPONSE_COLUMN = 'model_response'
METRIC_COLUMN = 'metric'
REQUIRED_COLUMNS = (DOC_COLUMN, RESPONSE_COLUMN, METRIC_COLUMN)

# Doc.sampling_methods, which is how a row says whether the model ranked choices
# or generated text. Doc's own docstring: choices holds "all options" for a
# multiple-choice task but "reference answers" for a generative one, so the same
# column means two different things and only one of them is a presented choice.
LOGPROBS_SAMPLING = 'LOGPROBS'


class LightevalInstanceLevelAdapter:
    """Converts a lighteval details parquet to instance-level EEE format."""

    def transform_details(
        self,
        details_path: Union[str, Path],
        evaluation_id: str,
        model_id: str,
        task_key: str,
    ) -> List[InstanceLevelEvaluationLog]:
        """Transform one task's details parquet into instance-level logs."""
        rows = self._read_details(Path(details_path))
        return [
            self._transform_row(row, evaluation_id, model_id, task_key)
            for row in rows
        ]

    def transform_and_save(
        self,
        details_path: Union[str, Path],
        evaluation_id: str,
        model_id: str,
        task_key: str,
        output_dir: Optional[Union[str, Path]] = None,
        file_uuid: Optional[str] = None,
        collection: Optional[str] = None,
        developer: Optional[str] = None,
    ) -> Optional[DetailedEvaluationResults]:
        """Transform details and save to JSONL, returning a pointer to the file.

        If output_dir is None, returns None (skips instance-level output).
        Otherwise file_uuid and collection are required so the samples file
        shares the aggregate UUID and declares its canonical location under
        data/.
        """
        if output_dir is None:
            return None
        if file_uuid is None:
            raise ValueError(
                'file_uuid is required when writing lighteval details'
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

        logs = self.transform_details(
            details_path, evaluation_id, model_id, task_key
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

    def _read_details(self, details_path: Path) -> List[Dict[str, Any]]:
        """Read a details parquet into one dict per sample."""
        import pandas as pd

        try:
            frame = pd.read_parquet(details_path)
        except ImportError as exc:
            raise ImportError(
                'reading lighteval details requires a parquet engine; install '
                "the converter's extra with `uv sync --extra lighteval`"
            ) from exc

        missing = [
            column for column in REQUIRED_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f'{details_path} is not a lighteval details file: no '
                f'{", ".join(missing)} column(s). Found '
                f'{", ".join(map(str, frame.columns))}.'
            )
        return frame.to_dict(orient='records')

    def _transform_row(
        self,
        row: Dict[str, Any],
        evaluation_id: str,
        model_id: str,
        task_key: str,
    ) -> InstanceLevelEvaluationLog:
        """Transform one details row into an instance-level log."""
        doc = _as_mapping(row.get(DOC_COLUMN))
        response = _as_mapping(row.get(RESPONSE_COLUMN))
        metrics = _as_mapping(row.get(METRIC_COLUMN))

        prompt = _as_text(doc.get('query'))
        choices = [
            _as_text(choice) for choice in _as_sequence(doc.get('choices'))
        ]
        gold_indices = _gold_indices(doc.get('gold_index'))
        reference = [
            choices[index]
            for index in gold_indices
            if 0 <= index < len(choices)
        ]
        sampling_methods = [
            _as_text(method)
            for method in _as_sequence(doc.get('sampling_methods'))
        ]

        generations = [
            _as_text(text) for text in _as_sequence(response.get('text'))
        ]
        post_processed = [
            _as_text(text)
            for text in _as_sequence(response.get('text_post_processed'))
        ]
        logprobs = [
            value
            for value in _as_sequence(response.get('logprobs'))
            if _is_finite(value)
        ]

        raw_output, extracted_value, extraction_method = self._extract_answer(
            generations, post_processed, logprobs, choices
        )

        primary_metric, score = _primary_metric(metrics)
        presented_choices = (
            choices
            if _presents_choices(sampling_methods, logprobs, choices)
            else None
        )

        # Build the sample hash from input + reference so the same dataset row
        # hashes alike across models and harnesses.
        hash_input = json.dumps(
            {'raw': prompt, 'reference': reference}, sort_keys=True
        )
        sample_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        metadata = {
            'lighteval_metrics': json.dumps(
                {
                    str(name): _python_scalar(value)
                    for name, value in metrics.items()
                    if _is_json_scalar(value)
                },
                sort_keys=True,
            ),
            'task_key': task_key,
        }
        if primary_metric is not None:
            metadata['primary_metric'] = primary_metric
        if sampling_methods:
            metadata['lighteval_sampling_methods'] = ','.join(sampling_methods)
        for key in ('truncated_tokens_count', 'padded_tokens_count'):
            value = response.get(key)
            if _is_finite(value):
                metadata[key] = str(int(value))
        if logprobs:
            metadata['choice_logprobs'] = json.dumps(
                [float(value) for value in logprobs]
            )

        return InstanceLevelEvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            model_id=model_id,
            evaluation_name=task_key,
            sample_id=_as_text(doc.get('id')),
            sample_hash=sample_hash,
            interaction_type=InteractionType.single_turn,
            input=Input(
                raw=prompt,
                reference=reference,
                choices=presented_choices or None,
            ),
            output=Output(raw=raw_output),
            answer_attribution=[
                AnswerAttributionItem(
                    turn_idx=0,
                    source='output.raw',
                    extracted_value=extracted_value,
                    extraction_method=extraction_method,
                    is_terminal=True,
                )
            ],
            evaluation=Evaluation(
                score=score,
                # lighteval's per-sample metrics are the metric's own value on
                # that sample, so 1.0 is exactly correct for acc/em and this
                # says "not a perfect score" for anything continuous. The full
                # per-sample metric mapping is in metadata either way.
                is_correct=score == 1.0,
            ),
            metadata=metadata,
        )

    def _extract_answer(
        self,
        generations: List[str],
        post_processed: List[str],
        logprobs: List[float],
        choices: List[str],
    ) -> tuple[List[str], str, str]:
        """Decide what the model answered and how that was read off.

        Returns the raw outputs, the single extracted answer, and the name of
        the extraction. A generative task carries its answer in `text`; a
        loglikelihood task generates nothing and answers by scoring each
        choice, so the answer is the highest-scoring one.
        """
        if generations:
            if post_processed and post_processed != generations:
                # lighteval's ModelResponse.post_process strips reasoning tags
                # and leaves the original in `text`.
                return (
                    generations,
                    post_processed[0],
                    'reasoning_tags_removed',
                )
            return generations, generations[0], 'none'

        if logprobs and choices and len(logprobs) == len(choices):
            selected = choices[logprobs.index(max(logprobs))]
            return [selected], selected, 'argmax_choice_logprob'

        if logprobs:
            # Scored, but the choices cannot be lined up with the scores, so
            # the index is reported rather than a choice guessed at.
            selected = str(logprobs.index(max(logprobs)))
            return [selected], selected, 'argmax_choice_logprob_index'

        return [], '', 'none'


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Read a parquet struct column as a plain dict."""
    if isinstance(value, dict):
        return value
    if hasattr(value, 'items'):
        return dict(value.items())
    return {}


def _as_sequence(value: Any) -> Sequence[Any]:
    """Read a parquet list column as a sequence, treating a scalar as empty."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, dict):
        return ()
    try:
        return list(value)
    except TypeError:
        return ()


def _as_text(value: Any) -> str:
    """Stringify a details value, mapping a missing one to the empty string."""
    if value is None:
        return ''
    return str(value)


def _presents_choices(
    sampling_methods: List[str], logprobs: List[float], choices: List[str]
) -> bool:
    """Decide whether doc.choices were shown to the model or are gold answers.

    Publishing a generative task's golds as `input.choices` would claim the
    model was given the answer to pick from, so the two cases cannot share a
    field. Runs predating `sampling_methods` are read off the response instead:
    one score per choice means the model was asked to rank them.
    """
    if sampling_methods:
        return LOGPROBS_SAMPLING in sampling_methods
    return bool(logprobs) and bool(choices) and len(logprobs) == len(choices)


def _python_scalar(value: Any) -> Any:
    """Unwrap a numpy scalar into the Python value it stands for.

    Parquet columns arrive as numpy scalars, and numpy's integer types do not
    subclass int, so an isinstance check against Python's own numeric types
    silently discards every integer lighteval writes -- gold_index included,
    which would empty `reference` on any task whose gold is a list.
    """
    item = getattr(value, 'item', None)
    if callable(item) and getattr(value, 'shape', None) == ():
        return item()
    return value


def _gold_indices(value: Any) -> List[int]:
    """Read Doc.gold_index, which is an int for one gold and a list for many."""
    if _is_finite(value):
        return [int(_python_scalar(value))]
    return [
        int(_python_scalar(item))
        for item in _as_sequence(value)
        if _is_finite(item)
    ]


def _is_finite(value: Any) -> bool:
    """Report whether a value is a real number that JSON can round-trip."""
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_json_scalar(value: Any) -> bool:
    """Report whether a metric value survives strict JSON serialization."""
    value = _python_scalar(value)
    if isinstance(value, bool) or isinstance(value, str):
        return True
    return _is_finite(value)


def _primary_metric(metrics: Dict[str, Any]) -> tuple[Optional[str], float]:
    """Pick the metric that becomes `evaluation.score`.

    The schema takes one score per sample while lighteval can record several,
    so the first metric the run wrote is used and its name is carried in
    metadata alongside the full mapping. Returns 0.0 with no name when the row
    holds no finite metric value.
    """
    for name, value in metrics.items():
        if _is_finite(value):
            return str(name), float(value)
    return None, 0.0
