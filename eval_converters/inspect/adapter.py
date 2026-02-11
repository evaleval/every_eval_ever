import os
import uuid

from inspect_ai.log import (
    EvalDataset,
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
from inspect_ai.log import EvalPlan as InspectEvalPlan
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

from eval_types import (
    AdditionalPropertiesObject,
    AgenticEvalConfig,
    AvailableTool,
    DetailedEvaluationResults,
    EvalLimits,
    EvalPlan,
    EvaluationLog,
    EvaluationResult,
    Format,
    GenerationArgs,
    GenerationConfig,
    HashAlgorithm,
    JudgeConfig,
    MetricConfig,
    ModelInfo,
    LlmScoring,
    Sandbox,
    ScoreType,
    ScoreDetails,
    SourceDataHf,
    SourceMetadata,
    SourceType,
    StandardError,
    Uncertainty
)

from eval_converters.common.adapter import (
    AdapterMetadata, 
    BaseEvaluationAdapter, 
    SupportedLibrary
)

from eval_converters.common.error import AdapterError
from eval_converters.common.utils import (
    convert_timestamp_to_unix_format, 
    get_current_unix_timestamp
)
from eval_converters.inspect.instance_level_adapter import (
    InspectInstanceLevelDataAdapter
)
from eval_converters.inspect.utils import (
    extract_model_info_from_model_path, sha256_file
)
from eval_converters import SCHEMA_VERSION

class InspectAIAdapter(BaseEvaluationAdapter):
    """
    Adapter for transforming evaluation outputs from the Inspect AI library into the unified schema format.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
			name="InspectAdapter",
			version="0.0.1",
			description="Adapter for transforming Inspect evaluation outputs to unified schema format"
		)

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI

    def _extract_uncertainty(
        self,
        stderr_value: float,
        stddev_value: float,
        num_samples: int
    ) -> Uncertainty:
        return Uncertainty(
            standard_error=StandardError(
                value=stderr_value
            ) if stderr_value else None,
            standard_deviation=stddev_value,
            num_samples=num_samples
        )

    def _build_evaluation_result(
        self,
        scorer_name: str,
        metric_info: EvalMetric,
        llm_grader: LlmScoring,
        source_data: SourceDataHf,
        evaluation_timestamp: str,
        generation_config: Dict[str, Any],
        stderr_value: float | None = None,
        stddev_value: float | None = None,
        num_samples: int = 0
    ) -> EvaluationResult:
        return EvaluationResult(
            evaluation_name=scorer_name,
            source_data=source_data,
            evaluation_timestamp=evaluation_timestamp,
            metric_config=MetricConfig(
                evaluation_description=metric_info.name,
                lower_is_better=False, # no metadata available
                score_type=ScoreType.continuous,
                min_score=0,
                max_score=1,
                llm_scoring=llm_grader
            ),
            score_details=ScoreDetails(
                score=metric_info.value,
                uncertainty=self._extract_uncertainty(
                    stderr_value,
                    stddev_value,
                    num_samples
                )
            ),
            generation_config=generation_config,
        )

    def _extract_evaluation_results(
        self,
        scores: List[EvalScore],
        source_data: SourceDataHf,
        generation_config: Dict[str, Any],
        num_samples: int,
        timestamp: str
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for scorer in scores:
            llm_grader = None
            if scorer.params and scorer.params.get("grader_model"):
                llm_grader = LlmScoring(
                    judges=[
                        JudgeConfig(
                            model_info=extract_model_info_from_model_path(
                                self._safe_get(scorer.params.get("grader_model"), "model")
                            )
                        )
                    ],
                    input_prompt=self._safe_get(scorer.params, "grader_template")
                )
            
            stderr_value = next(
                (m.value for m in scorer.metrics.values() if m.name == "stderr"),
                None,
            )

            stddev_value = next(
                (m.value for m in scorer.metrics.values() if m.name in {"std", "stddev"}),
                None,
            )

            for _, metric_info in scorer.metrics.items():
                if metric_info.name == "stderr":
                    continue

                scorer_name = scorer.name or scorer.scorer
                
                results.append(
                    self._build_evaluation_result(
                        scorer_name=scorer_name,
                        metric_info=metric_info,
                        llm_grader=llm_grader,
                        source_data=source_data,
                        evaluation_timestamp=timestamp,
                        generation_config=generation_config,
                        stderr_value=stderr_value,
                        stddev_value=stddev_value,
                        num_samples=num_samples
                    )
                )

        return results
    
    def _extract_source_data(
        self,
        dataset: EvalDataset,
        task_name: str
    ) -> SourceDataHf:
        dataset_name = (
            dataset.name.split('/')[-1]
            if dataset.name 
            else task_name.split('/')[-1]
        )
        return SourceDataHf( # TODO add hf_split
            source_type='hf_dataset',
            dataset_name=dataset_name,
            hf_repo=dataset.location,
            samples_number=dataset.samples,
            sample_ids=dataset.sample_ids,
            additional_details=AdditionalPropertiesObject(
                shuffled = dataset.shuffled
            )
        )

    def _safe_get(self, obj: Any, field: str):
        cur = obj
        
        if cur is None:
            return None
        
        if isinstance(cur, dict):
            cur = cur.get(field)
        else:
            cur = getattr(cur, field, None)

        return cur

    def _extract_available_tools(
        self, 
        eval_plan: InspectEvalPlan
    ) -> List[AvailableTool]:
        """Extracts and flattens tools from the evaluation plan steps."""
        
        tools_in_plan_steps = [
            step.params.get("tools", []) 
            for step in eval_plan.steps 
            if step.solver == "use_tools"
        ]
        
        return [
            AvailableTool(
                name=self._safe_get(tool, 'name'),
                description=self._safe_get(tool, 'description'),
                parameters=self._safe_get(tool, 'params'),
            )
            for tool_list in tools_in_plan_steps
            if isinstance(tool_list, list) and tool_list
            for tool in tool_list[0]
        ]
    
    def _extract_prompt_template(
        self,
        plan: InspectEvalPlan
    ) -> str | None:
        for step in plan.steps:
            if step.solver == "prompt_template":
                return self._safe_get(step.params, "template")
        
        return None

    def _extract_generation_config(
        self,
        spec: EvalSpec,
        inspect_plan: InspectEvalPlan
    ) -> GenerationConfig:
        eval_config = spec.model_generate_config
        eval_generation_config = {
            gen_config: str(value) 
            for gen_config, value in vars(eval_config).items() if value is not None
        }
        eval_sandbox = spec.task_args.get("sandbox", None)
        sandbox_type, sandbox_config = ((eval_sandbox or []) + [None, None])[:2]

        eval_plan = EvalPlan(
            name=inspect_plan.name,
            steps=inspect_plan.steps,
            config=inspect_plan.config.model_dump(),
        )

        eval_limits = EvalLimits(
            time_limit=spec.config.time_limit,
            message_limit=spec.config.message_limit,
            token_limit=spec.config.token_limit
        )

        max_attempts = spec.task_args.get("max_attempts") or eval_config.max_retries # TODO not sure if max_attempts == max_retries in this case

        reasoning = (
            True 
            if eval_config.reasoning_effort and eval_config.reasoning_effort.lower() != 'none'
            else False
        )

        available_tools: List[AvailableTool] = self._extract_available_tools(
            inspect_plan
        )

        generation_args = GenerationArgs(
            temperature=eval_config.temperature,
            top_p=eval_config.top_p,
            top_k=eval_config.top_k,
            max_tokens=eval_config.max_tokens,
            reasoning=reasoning,
            prompt_template=self._extract_prompt_template(inspect_plan),
            agentic_eval_config=AgenticEvalConfig(
                available_tools=available_tools
            ),
            eval_plan=eval_plan,
            eval_limits=eval_limits,
            sandbox=Sandbox(
                type=sandbox_type,
                config=sandbox_config
            ),
            max_attempts=max_attempts,
        )

        additional_details = AdditionalPropertiesObject.model_validate(
            eval_generation_config
        )
        
        return GenerationConfig(
            generation_args=generation_args,
            additional_details=additional_details or None
        )

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
        metadata_args: Dict[str, Any] = None,
        header_only: bool = False
    ) -> Union[
        EvaluationLog,
        List[EvaluationLog]
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File path {file_path} does not exists!')
        
        try:
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            eval_data: Tuple[EvalLog, List[EvalSampleSummary], EvalSample | None] = (
                self._load_file(file_path, header_only=header_only)
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

        evaluation_timestamp = eval_stats.started_at or eval_spec.created
        evaluation_unix_timestamp = convert_timestamp_to_unix_format(evaluation_timestamp)
        retrieved_unix_timestamp = get_current_unix_timestamp()

        if not evaluation_unix_timestamp:
            evaluation_unix_timestamp = retrieved_unix_timestamp

        source_metadata = SourceMetadata(
            source_name='inspect_ai',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get('source_organization_name'),
            source_organization_url=metadata_args.get('source_organization_url'),
            source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
            evaluator_relationship=metadata_args.get('evaluator_relationship')
        )

        source_data = self._extract_source_data(
            eval_spec.dataset, eval_spec.task
        )

        model_path = eval_spec.model

        num_samples = len(raw_eval_log.samples) if raw_eval_log.samples else 0
        single_sample = raw_eval_log.samples[0] if raw_eval_log.samples else single_sample

        if single_sample:
            detailed_model_name = single_sample.output.model

            if "/" in model_path:
                prefix, rest = model_path.split("/", 1)

                if rest != detailed_model_name:
                    model_path = f"{prefix}/{detailed_model_name}"
                else:
                    model_path = f"{prefix}/{rest}"
            else:
                model_path = detailed_model_name

        model_info: ModelInfo = extract_model_info_from_model_path(model_path)
        
        generation_config = self._extract_generation_config(eval_spec, raw_eval_log.plan)

        results: EvalResults | None = raw_eval_log.results

        evaluation_results = (
            self._extract_evaluation_results(
                results.scores if results else [],
                source_data,
                generation_config,
                num_samples,
                evaluation_unix_timestamp
            )
            if results and results.scores
            else []
        )

        evaluation_id = f'{source_data.dataset_name}/{model_path.replace('/', '_')}/{evaluation_unix_timestamp}'
        
        model_dev, model_name = model_info.id.split('/')
        parent_eval_output_dir = metadata_args.get('parent_eval_output_dir')
        evaluation_dir = f'{parent_eval_output_dir}/{source_data.dataset_name}/{model_dev}/{model_name}'
        detailed_results_id = f'{metadata_args.get('file_uuid')}_samples'
        
        evaluation_name = eval_spec.dataset.name or eval_spec.task

        if raw_eval_log.samples:
            instance_level_log_path, instance_level_rows_number = InspectInstanceLevelDataAdapter(
                detailed_results_id, Format.jsonl.value, HashAlgorithm.sha256.value, evaluation_dir
            ).convert_instance_level_logs(
                evaluation_name, model_info.id, raw_eval_log.samples
            )

            detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl,
                file_path=instance_level_log_path,
                hash_algorithm=HashAlgorithm.sha256.value,
                checksum=sha256_file(instance_level_log_path),
                total_rows=instance_level_rows_number
            )
        else:
            detailed_evaluation_results = None

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            evaluation_timestamp=evaluation_unix_timestamp,
            retrieved_timestamp=retrieved_unix_timestamp,
            source_metadata=source_metadata,
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results=detailed_evaluation_results
        )
    
    def _load_file(
        self, file_path, header_only=False
    ) -> Tuple[EvalLog, List[EvalSampleSummary], EvalSample | None]:
        log = read_eval_log(file_path, header_only=header_only)
        if header_only:
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
        else:
            summaries = []
            first_sample = None

        return log, summaries, first_sample