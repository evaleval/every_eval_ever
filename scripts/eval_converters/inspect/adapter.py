import os

from inspect_ai.log import (
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalSampleSummary,
    EvalScore,
    EvalStats,
    EvalSpec,
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample,
    read_eval_log_sample_summaries,
)

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

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

from scripts.eval_converters.common.adapter import (
    AdapterMetadata, 
    BaseEvaluationAdapter, 
    SupportedLibrary
)

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

    def _build_evaluation_result(
        self,
        scorer_name: str,
        metric_info: EvalMetric,
        evaluation_timestamp: str,
        generation_config: Dict[str, Any],
    ) -> EvaluationResult:
        return EvaluationResult(
            evaluation_name=scorer_name,
            evaluation_timestamp=evaluation_timestamp,
            metric_config=MetricConfig(
                evaluation_description=metric_info.name,
                lower_is_better=False, # no metadata available
                score_type=ScoreType.continuous,
                min_score=0,
                max_score=1,
            ),
            score_details=ScoreDetails(
                score=metric_info.value
            ),
            generation_config=generation_config,
        )

    def extract_evaluation_results(
        self,
        scores: List[EvalScore],
        generation_config: Dict[str, Any],
        timestamp: str
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for scorer in scores:
            for _, metric_info in scorer.metrics.items():
                if metric_info.name == "stderr":
                    continue

                results.append(
                    self._build_evaluation_result(
                        scorer_name=scorer.scorer,
                        metric_info=metric_info,
                        evaluation_timestamp=timestamp,
                        generation_config=generation_config,
                    )
                )

        return results

    def transform_from_directory(
        self,
        dir_path: Union[str, Path],
        metadata_args: Dict[str, Any] = None
    ) -> List[EvaluationLog]:
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory path {dir_path} does not exist!")
        
        log_paths: List[Path] = list_eval_logs(dir_path.absolute().as_posix())
        try:
            return [self.transform_from_file(urlparse(log_path.name).path, metadata_args) for log_path in log_paths]
        except Exception as e:
            raise AdapterError(f"Failed to load file from directory {dir_path}: {str(e)} for InspectAIAdapter")

    def transform_from_file(
        self, 
        file_path: Union[str, Path], 
        metadata_args: Dict[str, Any] = None
    ) -> Union[EvaluationLog, List[EvaluationLog]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            eval_data: Tuple[EvalLog, List[EvalSampleSummary], EvalSample | None] = (
                self._load_file(file_path)
            )
            return self.transform(eval_data, metadata_args)
        except AdapterError as e:
            raise e
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)} for InspectAIAdapter")

    def _transform_single(
        self, 
        raw_data: Tuple[EvalLog, List[EvalSampleSummary], EvalSample | None], 
        metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        raw_eval_log, sample_summaries, single_sample = raw_data
        eval_spec: EvalSpec = raw_eval_log.eval
        eval_stats: EvalStats = raw_eval_log.stats

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

        if single_sample:
            detailed_model_name = single_sample.output.model
            model_path_parts = model_path.split('/')

            if model_path_parts[-1] in detailed_model_name:
                model_path_parts[-1] = detailed_model_name

            model_path = '/'.join(model_path_parts)

        model_info: ModelInfo = extract_model_info_from_model_path(model_path)
        
        results: EvalResults = raw_eval_log.results

        generation_config = {
            gen_config: value 
            for gen_config, value in vars(eval_spec.model_generate_config).items() if value is not None
        }

        evaluation_results = (
            self.extract_evaluation_results(
                results.scores, 
                generation_config,
                retrieved_unix_timestamp
            )
            if results and results.scores
            else []
        )

        detailed_evaluation_results_per_samples = []
        for sample_summary in sample_summaries:
            if sample_summary.scores:
                response = sample_summary.scores.get('choice').answer
            else:
                response = sample_summary.output.choices[0].message.content

            detailed_evaluation_results_per_samples.append(
                DetailedEvaluationResultsPerSample(
                    sample_id=str(sample_summary.id),
                    input=sample_summary.input,
                    ground_truth=sample_summary.target,
                    response=response,
                    choices=sample_summary.choices
                )
            )

        evaluation_id = f'{eval_spec.dataset.name}/{model_path.replace('/', '_')}/{retrieved_unix_timestamp}'

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
    
    def _load_file(self, file_path) -> Tuple[EvalLog, List[EvalSampleSummary], EvalSample | None]:
        log = read_eval_log(file_path, header_only=True)
        summaries = read_eval_log_sample_summaries(file_path)
        first_sample = (
            read_eval_log_sample(
                file_path,
                summaries[0].id,
                summaries[0].epoch
            )
            if summaries
            else None
        )

        return log, summaries, first_sample