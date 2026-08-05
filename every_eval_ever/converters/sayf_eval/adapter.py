"""Adapter for converting sayf-eval results records to every_eval_ever format.

sayf-eval writes a canonical *results record* per run
(``<output_dir>/results/<model>/results_<ts>.json``) that embeds the full
pipeline configuration (decoding params, ``<think>`` handling, denominator
policy, judge model) next to the per-task metrics, plus declared per-task dataset
provenance (``task_sources``). This adapter maps that record onto the Every Eval
Ever schema: **one EvaluationLog per task**, with each task's metrics
(accuracy, plus CVSS MAD for VSP / micro-F1 for ATE) as ``evaluation_results``.

Aggregate-only: the record carries no prompt/gold/response text, and this adapter
never produces instance-level ``_samples.jsonl`` — matching sayf-eval's dual-use
security posture (item text stays private).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from every_eval_ever.converters.common.utils import (
    convert_timestamp_to_unix_format,
    get_current_unix_timestamp,
)
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    JudgeConfig,
    LlmScoring,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
    SourceType,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
)

# API providers whose weights are not publicly available. Best-effort; anything
# else is recorded as "unknown" rather than guessed.
_CLOSED_PROVIDERS = {
    'openai',
    'anthropic',
    'azure',
    'gemini',
    'google',
    'vertex_ai',
    'xai',
    'bedrock',
    'cohere',
    'mistral',
}

# litellm transport prefixes stripped from a local model string to recover the
# real (HuggingFace-style) model id, e.g. "hosted_vllm/Qwen/Qwen3-8B" -> "Qwen/Qwen3-8B".
_LOCAL_PREFIXES = ('hosted_vllm/', 'openai/')

# Secondary per-task metrics beyond accuracy, keyed by the record's metric name.
# (metric_id, metric_name, metric_kind, metric_unit, lower_is_better, min, max, description)
_SECONDARY_METRICS = [
    (
        'mad',
        'cvss_mad',
        'CVSS MAD',
        'mae',
        'points',
        True,
        0.0,
        10.0,
        'Mean absolute difference between predicted and gold CVSS v3.1 base scores (0 identical, 10 worst).',
    ),
    (
        'f1',
        'f1_micro',
        'Micro-F1',
        'f1',
        'proportion',
        False,
        0.0,
        1.0,
        'Micro-averaged F1 over parent MITRE ATT&CK technique IDs.',
    ),
]


class SayfEvalAdapter(BaseEvaluationAdapter):
    """Converts a sayf-eval results record to every_eval_ever format."""

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='sayf-eval-adapter',
            version='0.1.0',
            supported_library_versions=['0.1.*'],
            description='Converts sayf-eval results records (aggregate scores + pipeline/judge config) to every_eval_ever format',
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.CUSTOM

    # -- model / judge -------------------------------------------------------

    def _build_model_info(self, m: Dict[str, Any]) -> ModelInfo:
        """Map a sayf-eval ModelInfo dict {name, provider, base_url} to EEE ModelInfo."""
        name = m.get('name', '') or ''
        provider = (m.get('provider') or '').lower()
        base_url = m.get('base_url')
        is_local = bool(base_url) or provider.startswith('hosted_vllm')

        ident = name
        if is_local:
            for pref in _LOCAL_PREFIXES:
                if ident.startswith(pref):
                    ident = ident[len(pref) :]
                    break
        developer = ident.split('/')[0] if '/' in ident else (provider or ident)

        inference_platform = None
        inference_engine = None
        if is_local:
            deployment_type = 'self_deployed'
            model_availability = 'open_weights'
            inference_engine = InferenceEngine(name='vllm', version=None)
        else:
            deployment_type = 'externally_managed'
            model_availability = (
                'closed_weights' if provider in _CLOSED_PROVIDERS else 'unknown'
            )
            inference_platform = provider or None

        return ModelInfo(
            name=name,
            id=ident,
            developer=developer or None,
            inference_platform=inference_platform,
            inference_engine=inference_engine,
            additional_details={
                'deployment_type': deployment_type,
                'model_availability': model_availability,
            },
        )

    def _build_llm_scoring(
        self, record: Dict[str, Any], judge_info: ModelInfo
    ) -> LlmScoring:
        pipeline = record.get('pipeline', {})
        input_prompt = pipeline.get('scoring') or (
            'LLM-as-judge: a single judge call performs answer extraction and a '
            'CORRECT/INCORRECT verdict against the gold answer.'
        )
        return LlmScoring(
            judges=[JudgeConfig(model_info=judge_info)],
            input_prompt=input_prompt,
        )

    # -- source data ---------------------------------------------------------

    def _build_source_data(
        self, task: str, src: Optional[Dict[str, Any]], collection_prefix: str
    ):
        """Map a task's declared provenance to an EEE source_data variant.

        The collection folder in the datastore is derived from ``dataset_name``,
        so it is set to a clean slug; the human title and HF subset are preserved
        in ``additional_details``.
        """
        slug = f'{collection_prefix}{task.replace("_", "-")}'
        src = src or {}
        title = src.get('dataset_name') or task
        stype = src.get('type', 'other')

        if stype == 'hf_dataset':
            extra = {'title': title}
            if src.get('subset'):
                extra['subset'] = str(src['subset'])
            return SourceDataHf(
                dataset_name=slug,
                source_type='hf_dataset',
                hf_repo=src.get('hf_repo'),
                hf_split=src.get('split'),
                additional_details=extra,
            )
        if stype == 'url' and src.get('url'):
            return SourceDataUrl(
                dataset_name=slug,
                source_type='url',
                url=list(src['url']),
                additional_details={'title': title},
            )
        return SourceDataPrivate(
            dataset_name=slug,
            source_type='other',
            additional_details={'title': title},
        )

    # -- generation config ---------------------------------------------------

    def _build_generation_config(
        self, record: Dict[str, Any]
    ) -> GenerationConfig:
        pipeline = record.get('pipeline', {})
        override = pipeline.get('max_tokens_override')
        args = GenerationArgs(
            temperature=pipeline.get('temperature'),
            top_p=pipeline.get('top_p'),
            max_tokens=override if isinstance(override, int) else None,
        )
        # Everything GenerationArgs forbids goes into additional_details (strings).
        additional: Dict[str, str] = {}
        if pipeline.get('seed') is not None:
            additional['seed'] = str(pipeline['seed'])
        if pipeline.get('answer_stop'):
            additional['answer_stop'] = json.dumps(pipeline['answer_stop'])
        for key in ('max_tokens', 'think_handling', 'denominator_policy'):
            if pipeline.get(key):
                # 'max_tokens' here is sayf-eval's budget *policy* string.
                out_key = 'max_tokens_policy' if key == 'max_tokens' else key
                additional[out_key] = str(pipeline[key])
        return GenerationConfig(
            generation_args=args,
            additional_details=additional or None,
        )

    # -- evaluation results --------------------------------------------------

    def _build_evaluation_results(
        self,
        task: str,
        metrics: Dict[str, Any],
        source_data,
        llm_scoring: LlmScoring,
        gen_config: GenerationConfig,
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        def _result(
            metric_id,
            name,
            kind,
            unit,
            lower,
            lo,
            hi,
            desc,
            score,
            details=None,
        ):
            return EvaluationResult(
                evaluation_result_id=f'{task}/{metric_id}',
                evaluation_name=task,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=desc,
                    metric_id=metric_id,
                    metric_name=name,
                    metric_kind=kind,
                    metric_unit=unit,
                    lower_is_better=lower,
                    score_type=ScoreType.continuous,
                    min_score=lo,
                    max_score=hi,
                    llm_scoring=llm_scoring,
                ),
                score_details=ScoreDetails(score=score, details=details),
                generation_config=gen_config,
            )

        if isinstance(metrics.get('accuracy'), (int, float)):
            details = {
                k: str(metrics[k])
                for k in ('correct', 'total', 'skipped')
                if k in metrics
            }
            results.append(
                _result(
                    'accuracy',
                    'Accuracy',
                    'accuracy',
                    'proportion',
                    False,
                    0.0,
                    1.0,
                    'LLM-as-judge accuracy over attempted items (unparseable/empty answers count as '
                    'incorrect; judge-API failures excluded from numerator and denominator).',
                    float(metrics['accuracy']),
                    details or None,
                )
            )

        for (
            key,
            mid,
            name,
            kind,
            unit,
            lower,
            lo,
            hi,
            desc,
        ) in _SECONDARY_METRICS:
            if isinstance(metrics.get(key), (int, float)):
                results.append(
                    _result(
                        mid,
                        name,
                        kind,
                        unit,
                        lower,
                        lo,
                        hi,
                        desc,
                        float(metrics[key]),
                    )
                )

        return results

    # -- single task -> one EvaluationLog ------------------------------------

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Build one EvaluationLog for the task named in ``metadata_args['task_name']``."""
        record = raw_data
        task = metadata_args['task_name']
        metrics = record.get('results', {}).get(task, {})

        model_info = self._build_model_info(record.get('model', {}))
        judge_info = self._build_model_info(record.get('judge', {}))
        llm_scoring = self._build_llm_scoring(record, judge_info)
        gen_config = self._build_generation_config(record)

        collection_prefix = metadata_args.get('collection_prefix', '') or ''
        source_data = self._build_source_data(
            task,
            (record.get('task_sources') or {}).get(task),
            collection_prefix,
        )

        eval_results = self._build_evaluation_results(
            task, metrics, source_data, llm_scoring, gen_config
        )
        if not eval_results:
            raise ValueError(f'Task {task!r} has no numeric metrics to convert')

        retrieved_timestamp = get_current_unix_timestamp()
        evaluation_id = f'{task}/{model_info.id}/{retrieved_timestamp}'
        eval_timestamp = record.get('created_at')
        if eval_timestamp:
            try:
                eval_timestamp = convert_timestamp_to_unix_format(
                    eval_timestamp
                )
            except Exception:
                eval_timestamp = None

        source_metadata = SourceMetadata(
            source_name='sayf-eval',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get(
                'source_organization_name', ''
            ),
            source_organization_url=metadata_args.get(
                'source_organization_url'
            ),
            source_organization_logo_url=metadata_args.get(
                'source_organization_logo_url'
            ),
            evaluator_relationship=EvaluatorRelationship(
                metadata_args.get('evaluator_relationship', 'third_party')
            ),
        )

        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name') or 'sayf-eval',
            version=record.get('sayf_eval_version')
            or metadata_args.get('eval_library_version', 'unknown'),
        )

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=eval_timestamp,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=eval_results,
        )

    # -- file / directory entrypoints ----------------------------------------

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a sayf-eval results record JSON into one EvaluationLog per task."""
        file_path = Path(file_path)
        record = self._load_file(file_path)
        tasks = sorted((record.get('results') or {}).keys())
        logs: List[EvaluationLog] = []
        for task in tasks:
            log = self._transform_single(
                record, {**metadata_args, 'task_name': task}
            )
            logs.append(log)
        return logs

    @staticmethod
    def _find_records(dir_path: Path) -> List[Path]:
        """Locate sayf-eval results records under an output directory.

        Records live at ``<output_dir>/results/<model>/results_<ts>.json``; a
        recursive ``results_*.json`` glob finds them wherever the dir is rooted.
        """
        return sorted(dir_path.glob('**/results_*.json'))

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any] = None
    ) -> List[EvaluationLog]:
        result = self.transform_from_directory_result(
            dir_path, metadata_args or {}
        )
        result.raise_if_incomplete()
        return result.records

    def transform_from_directory_result(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> SourceConversionResult[EvaluationLog]:
        dir_path = Path(dir_path)
        record_files = self._find_records(dir_path)
        if not record_files:
            raise ValueError(
                f'No sayf-eval results_*.json records found under {dir_path}'
            )

        all_logs: List[EvaluationLog] = []
        failures: List[SourceRecordFailure] = []
        for record_file in record_files:
            try:
                all_logs.extend(
                    self.transform_from_file(record_file, metadata_args)
                )
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(record_file),
                        reason=str(exc),
                        source_record={'path': str(record_file)},
                    )
                )

        return SourceConversionResult(
            source_name=f'sayf-eval records under {dir_path}',
            total_records=len(all_logs) + len(failures),
            records=all_logs,
            failures=failures,
        )
