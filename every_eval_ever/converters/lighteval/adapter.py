"""Adapter for converting lighteval output to every_eval_ever format."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceMetadata,
    SourceType,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
)

from .utils import (
    KNOWN_METRIC_BOUNDS,
    STDERR_SUFFIX,
    find_metric_spec,
    flatten_model_config,
    higher_is_better_for,
    is_derived_aggregate_key,
    is_finite_number,
    parse_results_file_timestamp,
    split_task_key,
    stderr_method_for,
)

# lighteval writes "?" when it cannot read its own git SHA, which is the normal
# case for a pip install.
_UNKNOWN_SHA = '?'


class LightevalAdapter(BaseEvaluationAdapter):
    """Converts lighteval results files to every_eval_ever format."""

    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)
        # Stores per-log metadata so callers can find the source file after
        # transform. Keyed by evaluation_id -> {"parent_dir": str,
        # "task_key": str}
        self._eval_metadata = {}

    def get_eval_metadata(self, evaluation_id: str) -> Dict[str, Any]:
        """Return stored metadata for a given evaluation_id."""
        return self._eval_metadata.get(evaluation_id, {})

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='lighteval-adapter',
            version='0.1.0',
            supported_library_versions=['0.13.*'],
            description='Converts lighteval output to every_eval_ever format',
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.LIGHTEVAL

    def _extract_model_info(
        self,
        raw_data: Dict[str, Any],
        metadata_args: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Extract model information from a lighteval results file."""
        metadata_args = metadata_args or {}
        config_general = raw_data.get('config_general') or {}
        model_config = config_general.get('model_config')
        if not isinstance(model_config, dict):
            model_config = {}

        model_name = config_general.get('model_name') or model_config.get(
            'model_name'
        )
        if not model_name:
            raise ValueError(
                'lighteval results file has no config_general.model_name'
            )

        developer = None
        if '/' in model_name:
            developer = model_name.split('/')[0]

        # The dumped model config carries no discriminator for the backend that
        # produced it, so the engine cannot be read off the file.
        engine_name = metadata_args.get('inference_engine')
        engine_version = metadata_args.get('inference_engine_version')
        inference_engine = None
        if engine_name:
            inference_engine = InferenceEngine(
                name=engine_name, version=engine_version
            )

        # LiteLLM and inference-provider runs state their platform outright.
        inference_platform = model_config.get('provider') or metadata_args.get(
            'inference_platform'
        )

        additional, redacted = flatten_model_config(model_config)
        if redacted:
            additional['redacted_model_config_keys'] = ','.join(redacted)

        return ModelInfo(
            name=model_name,
            id=model_name,
            developer=developer,
            inference_platform=inference_platform,
            inference_engine=inference_engine,
            additional_details=additional or None,
        )

    def _get_tasks(self, raw_data: Dict[str, Any]) -> List[str]:
        """Get the keys of `results` that lighteval measured rather than averaged."""
        results = raw_data.get('results') or {}
        tasks = []
        for task_key, task_results in results.items():
            if is_derived_aggregate_key(task_key):
                continue
            if not isinstance(task_results, dict):
                continue
            if any(
                self._is_metric_entry(key, value)
                for key, value in task_results.items()
            ):
                tasks.append(task_key)
        return tasks

    @staticmethod
    def _is_metric_entry(key: str, value: Any) -> bool:
        """Report whether one `results` entry is a usable metric score."""
        if key.endswith(STDERR_SUFFIX):
            return False
        return is_finite_number(value)

    def _build_source_data(self, task_config: Dict[str, Any], task_name: str):
        """Build source_data from a lighteval task config."""
        dataset_name = task_config.get('name') or task_name
        hf_repo = task_config.get('hf_repo')
        if not hf_repo:
            return SourceDataPrivate(
                dataset_name=dataset_name,
                source_type='other',
            )

        evaluation_splits = task_config.get('evaluation_splits') or []
        additional = {}
        if task_config.get('hf_subset'):
            additional['hf_subset'] = str(task_config['hf_subset'])
        if task_config.get('hf_revision'):
            additional['hf_revision'] = str(task_config['hf_revision'])
        if len(evaluation_splits) > 1:
            additional['evaluation_splits'] = json.dumps(
                list(evaluation_splits)
            )

        original_num_docs = task_config.get('original_num_docs')
        return SourceDataHf(
            dataset_name=dataset_name,
            source_type='hf_dataset',
            hf_repo=hf_repo,
            hf_split=evaluation_splits[0] if evaluation_splits else None,
            samples_number=(
                original_num_docs
                if isinstance(original_num_docs, int) and original_num_docs >= 0
                else None
            ),
            additional_details=additional or None,
        )

    def _build_generation_config(
        self,
        task_config: Dict[str, Any],
        generation_parameters: Dict[str, Any],
        num_fewshots: Optional[int],
    ) -> Optional[GenerationConfig]:
        """Build generation config from the run's model and task configs."""
        generation_size = task_config.get('generation_size')
        max_tokens = generation_parameters.get('max_new_tokens')
        if not isinstance(max_tokens, int) or max_tokens < 1:
            max_tokens = (
                generation_size
                if isinstance(generation_size, int) and generation_size >= 1
                else None
            )

        args = GenerationArgs(
            temperature=generation_parameters.get('temperature'),
            top_p=generation_parameters.get('top_p'),
            top_k=generation_parameters.get('top_k'),
            max_tokens=max_tokens,
        )

        additional = {}
        for key, value in generation_parameters.items():
            if key in ('temperature', 'top_p', 'top_k', 'max_new_tokens'):
                continue
            if value is None:
                continue
            additional[key] = (
                value if isinstance(value, str) else json.dumps(value)
            )
        if num_fewshots is not None:
            additional['num_fewshots'] = str(num_fewshots)
        if task_config.get('stop_sequence'):
            additional['stop_sequence'] = json.dumps(
                list(task_config['stop_sequence'])
            )
        if task_config.get('few_shots_select'):
            additional['few_shots_select'] = str(
                task_config['few_shots_select']
            )

        stated_args = {
            args.temperature,
            args.top_p,
            args.top_k,
            args.max_tokens,
        } - {None}
        if not stated_args and not additional:
            return None
        return GenerationConfig(
            generation_args=args,
            additional_details=additional or None,
        )

    def _build_evaluation_results(
        self,
        raw_data: Dict[str, Any],
        task_key: str,
        evaluation_timestamp: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Build the EvaluationResult list for a single lighteval task."""
        task_results = raw_data['results'][task_key]
        task_name, num_fewshots = split_task_key(task_key)
        task_config = (raw_data.get('config_tasks') or {}).get(task_key) or {}
        config_general = raw_data.get('config_general') or {}
        model_config = config_general.get('model_config')
        generation_parameters = {}
        if isinstance(model_config, dict):
            declared = model_config.get('generation_parameters')
            if isinstance(declared, dict):
                generation_parameters = declared

        source_data = self._build_source_data(task_config, task_name)
        gen_config = self._build_generation_config(
            task_config, generation_parameters, num_fewshots
        )
        effective_num_docs = task_config.get('effective_num_docs')
        num_samples = (
            effective_num_docs
            if isinstance(effective_num_docs, int) and effective_num_docs >= 0
            else None
        )

        results = []
        for metric_name, value in task_results.items():
            if not self._is_metric_entry(metric_name, value):
                continue

            metric_spec = find_metric_spec(task_config, metric_name)
            higher_is_better = higher_is_better_for(metric_spec, metric_name)

            metric_details = {}
            if higher_is_better is None:
                # EEE requires a direction; record that the run did not give one.
                metric_details['direction_status'] = 'assumed_higher_is_better'
                higher_is_better = True

            bounds = KNOWN_METRIC_BOUNDS.get(metric_name)
            if bounds is None:
                # Preserve metrics whose mathematical range is not yet known
                # without falsely declaring them continuous and unbounded.
                metric_details['bounds_status'] = 'unknown'
                metric_config = MetricConfig(
                    evaluation_description=metric_name,
                    metric_name=metric_name,
                    lower_is_better=not higher_is_better,
                    additional_details=metric_details,
                )
            else:
                metric_config = MetricConfig(
                    evaluation_description=metric_name,
                    metric_name=metric_name,
                    lower_is_better=not higher_is_better,
                    score_type=ScoreType.continuous,
                    min_score=bounds[0],
                    max_score=bounds[1],
                    additional_details=metric_details or None,
                )

            # lighteval writes stderr into the same dict as the metric it
            # belongs to, and sets it to NaN when the estimate overflowed.
            stderr_value = task_results.get(f'{metric_name}{STDERR_SUFFIX}')
            if not is_finite_number(stderr_value):
                stderr_value = None

            uncertainty = None
            if stderr_value is not None or num_samples is not None:
                uncertainty = Uncertainty(
                    standard_error=(
                        StandardError(
                            value=stderr_value,
                            method=stderr_method_for(metric_spec, metric_name),
                        )
                        if stderr_value is not None
                        else None
                    ),
                    num_samples=num_samples,
                )

            results.append(
                EvaluationResult(
                    evaluation_name=task_key,
                    source_data=source_data,
                    evaluation_timestamp=evaluation_timestamp,
                    metric_config=metric_config,
                    score_details=ScoreDetails(
                        score=value,
                        uncertainty=uncertainty,
                    ),
                    generation_config=gen_config,
                )
            )

        return results

    def _count_dropped_scores(
        self, raw_data: Dict[str, Any], task_key: str
    ) -> int:
        """Count metrics skipped because their aggregated score was not finite."""
        task_results = raw_data['results'][task_key]
        return sum(
            1
            for key, value in task_results.items()
            if not key.endswith(STDERR_SUFFIX) and not is_finite_number(value)
        )

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Transform a single lighteval task's results into an EvaluationLog.

        Expects metadata_args to contain 'task_key' specifying which task.
        """
        task_key = metadata_args['task_key']
        model_info = self._extract_model_info(raw_data, metadata_args)
        config_general = raw_data.get('config_general') or {}

        retrieved_timestamp = get_current_unix_timestamp()
        eval_timestamp = metadata_args.get('evaluation_timestamp')

        evaluation_id = f'{task_key}/{model_info.id}/{retrieved_timestamp}'
        evaluation_results = self._build_evaluation_results(
            raw_data, task_key, eval_timestamp
        )
        if not evaluation_results:
            raise ValueError(
                f'lighteval task {task_key!r} has no finite metric scores'
            )

        evaluator_rel_str = metadata_args.get(
            'evaluator_relationship', 'first_party'
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        eval_library_details = {}
        lighteval_sha = config_general.get('lighteval_sha')
        if lighteval_sha and lighteval_sha != _UNKNOWN_SHA:
            eval_library_details['lighteval_sha'] = str(lighteval_sha)
        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'lighteval'),
            version=metadata_args.get('eval_library_version', 'unknown'),
            additional_details=eval_library_details or None,
        )

        source_details = {}
        unknown_bounds_count = sum(
            result.metric_config.additional_details is not None
            and result.metric_config.additional_details.get('bounds_status')
            == 'unknown'
            for result in evaluation_results
        )
        if unknown_bounds_count:
            source_details['metrics_with_unknown_bounds'] = str(
                unknown_bounds_count
            )
        dropped_scores = self._count_dropped_scores(raw_data, task_key)
        if dropped_scores:
            source_details['metrics_dropped_non_finite'] = str(dropped_scores)
        derived_rows = metadata_args.get('derived_aggregate_keys') or []
        if derived_rows:
            source_details['lighteval_derived_rows_not_converted'] = ','.join(
                derived_rows
            )
        elapsed = config_general.get('total_evaluation_time_secondes')
        if elapsed is not None:
            source_details['total_evaluation_time_seconds'] = str(elapsed)
        for key in ('job_id', 'max_samples', 'num_fewshot_seeds'):
            if config_general.get(key) is not None:
                source_details[key] = str(config_general[key])

        source_metadata = SourceMetadata(
            source_name='lighteval',
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
            evaluator_relationship=evaluator_relationship,
            additional_details=source_details or None,
        )

        # Store metadata so callers can trace a log back to its results file
        self._eval_metadata[evaluation_id] = {
            'parent_dir': metadata_args.get('parent_eval_output_dir'),
            'task_key': task_key,
        }

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=eval_timestamp,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a lighteval results JSON file into EvaluationLogs.

        Returns one EvaluationLog per measured task in the results file.
        """
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)
        tasks = self._get_tasks(raw_data)

        derived_keys = sorted(
            key
            for key in (raw_data.get('results') or {})
            if is_derived_aggregate_key(key)
        )
        metadata_args = {
            **metadata_args,
            'parent_eval_output_dir': str(file_path.parent),
            'derived_aggregate_keys': derived_keys,
            # The results filename holds the only wall-clock time in the run.
            'evaluation_timestamp': parse_results_file_timestamp(file_path),
        }

        results = []
        for task_key in tasks:
            task_metadata = {**metadata_args, 'task_key': task_key}
            results.append(self._transform_single(raw_data, task_metadata))

        return results

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        result = self.transform_from_directory_result(dir_path, metadata_args)
        result.raise_if_incomplete()
        return result.records

    def transform_from_directory_result(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> SourceConversionResult[EvaluationLog]:
        """Transform all lighteval files while retaining per-file failures.

        Searches for results_*.json files recursively, because lighteval nests
        them under results/<org>/<model>/.
        """
        dir_path = Path(dir_path)
        results_files = sorted(dir_path.glob('**/results_*.json'))
        if not results_files:
            raise ValueError(
                f'No lighteval results_*.json files found under {dir_path}'
            )

        all_logs: list[EvaluationLog] = []
        failures: list[SourceRecordFailure] = []
        for results_file in results_files:
            try:
                all_logs.extend(
                    self.transform_from_file(results_file, metadata_args)
                )
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(results_file),
                        reason=str(exc),
                        source_record={'path': str(results_file)},
                    )
                )

        return SourceConversionResult(
            source_name=f'lighteval evaluations under {dir_path}',
            total_records=len(all_logs) + len(failures),
            records=all_logs,
            failures=failures,
        )
