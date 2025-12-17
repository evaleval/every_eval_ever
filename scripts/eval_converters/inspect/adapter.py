import os

from inspect_ai.log import EvalLog, EvalSpec, EvalStats, read_eval_log

from pathlib import Path
from typing import Any, Dict, List, Union

from eval_types import (
    DetailedEvaluationResultsPerSample,
    EvaluationLog,
    EvaluationResult,
    MetricConfig,
    ModelInfo,
    ScoreType,
    ScoreDetails,
    SourceData,
    SourceMetadata,
    SourceType
)

from scripts.eval_converters.common.adapter import AdapterMetadata, BaseEvaluationAdapter, SupportedLibrary
from scripts.eval_converters.common.error import AdapterError
from scripts.eval_converters.common.utils import convert_timestamp_to_unix_format
from scripts.eval_converters.inspect.utils import extract_model_info_from_model_path
from scripts.eval_converters import SCHEMA_VERSION

class InspectAIAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the Inspect AI library into the unified schema format.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
			name="InspectAdapter",
			version="0.0.1",
			description="Adapter for transforming HELM evaluation outputs to unified schema format"
		)

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI
        
    def transform_from_directory(self, dir_path: Union[str, Path]):
        raise NotImplementedError("Inspect AI adapter do not support loading logs from directory!")

    def transform_from_file(self, file_path: Union[str, Path], metadata_args: Dict = None) -> Union[EvaluationLog, List[EvaluationLog]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            eval_log: EvalLog = self._load_file(file_path)
            return self.transform(eval_log, metadata_args)
        except AdapterError as e:
            raise e
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)} for InspectAIAdapter")

    def _transform_single(
        self, raw_data: EvalLog, metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        eval_spec: EvalSpec = raw_data.eval
        eval_stats: EvalStats = raw_data.stats

        retrieved_timestamp = eval_stats.started_at or eval_spec.created
        retrieved_unix_timestamp = convert_timestamp_to_unix_format(retrieved_timestamp)
        
        source_data = SourceData(
            dataset_name=eval_spec.dataset.name.split('/')[-1],
            hf_repo=eval_spec.dataset.location,
            samples_number=eval_spec.dataset.samples,
            sample_ids=eval_spec.dataset.sample_ids
        )

        source_metadata = SourceMetadata(
            source_name='inspect_ai',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get('source_organization_name'),
            source_organization_url=metadata_args.get('source_organization_url'),
            source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
            evaluator_relationship=metadata_args.get('evaluator_relationship')
        )

        model_path = eval_spec.model
        self._check_if_model_is_on_huggingface(model_path)

        if raw_data.samples:
            detailed_model_name = raw_data.samples[0].output.model
            model_path_parts = model_path.split('/')

            if model_path_parts[-1] in detailed_model_name:
                model_path_parts[-1] = detailed_model_name

            model_path = '/'.join(model_path_parts)

        model_info: ModelInfo = extract_model_info_from_model_path(model_path)

        results = raw_data.results
        evaluation_results = []

        generation_config = {
            gen_config: value 
            for gen_config, value in vars(eval_spec.model_generate_config).items() if value is not None
        }

        for scorer_results in results.scores:
            scorer_name = scorer_results.scorer
            for metric in scorer_results.metrics:
                metric_info = scorer_results.metrics[metric]
                if metric_info.name != 'stderr':
                    evaluation_results.append(EvaluationResult(
                        evaluation_name=scorer_name,
                        evaluation_timestamp=convert_timestamp_to_unix_format(eval_stats.completed_at),
                        metric_config=MetricConfig(
                            evaluation_description=metric_info.name,
                            lower_is_better=False, # probably there is no access to such info
                            score_type=ScoreType.continuous,
                            min_score=0,
                            max_score=1
                        ),
                        score_details=ScoreDetails(
                            score=metric_info.value
                        ),
                        generation_config=generation_config
                    ))

        detailed_evaluation_results_per_samples = []
        for sample in raw_data.samples:
            if sample.scores:
                response = sample.scores.get('choice').answer
            else:
                response = sample.output.choices[0].message.content

            detailed_evaluation_results_per_samples.append(
                DetailedEvaluationResultsPerSample(
                    sample_id=str(sample.id),
                    input=sample.input,
                    ground_truth=sample.target,
                    response=response,
                    choices=sample.choices
                )
            )

        evaluation_id = f'inspect_ai/{model_path}/{eval_spec.dataset.name}/{retrieved_unix_timestamp}'

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_unix_timestamp,
            source_data=source_data,
            source_metadata=source_metadata,
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results_per_samples=detailed_evaluation_results_per_samples
        )
        
    def _load_file(self, file_path) -> EvalLog:
        return read_eval_log(file_path)