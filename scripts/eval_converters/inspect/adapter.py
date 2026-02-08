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
    AgenticEvalConfig,
    AvailableTool,
    ConfidenceInterval,
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
    SourceDataHF,
    SourceMetadata,
    SourceType
)

from scripts.eval_converters.common.adapter import (
    AdapterMetadata, 
    BaseEvaluationAdapter, 
    SupportedLibrary
)

from scripts.eval_converters.common.error import AdapterError
from scripts.eval_converters.common.utils import (
    convert_timestamp_to_unix_format, 
    get_current_unix_timestamp
)
from scripts.eval_converters.inspect.instance_level_adapter import (
    InspectInstanceLevelDataAdapter
)
from scripts.eval_converters.inspect.utils import (
    extract_model_info_from_model_path, sha256_file
)
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
			description="Adapter for transforming Inspect evaluation outputs to unified schema format"
		)

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.INSPECT_AI

    def _calculate_confidence_interval(
        self, 
        estimate, 
        stderr, 
        z=1.96
    ) -> ConfidenceInterval:
        if stderr is None or not isfinite(stderr):
            return None

        margin = z * stderr
        lower = estimate - margin
        upper = estimate + margin

        lower = max(0.0, lower)
        upper = min(1.0, upper)

        return ConfidenceInterval(
            lower=lower,
            upper=upper
        )

    def _build_evaluation_result(
        self,
        scorer_name: str,
        metric_info: EvalMetric,
        llm_grader: LlmScoring,
        stderr_value: float,
        source_data: SourceDataHF,
        evaluation_timestamp: str,
        generation_config: Dict[str, Any],
    ) -> EvaluationResult:
        conf_interval = self._calculate_confidence_interval(
            metric_info.value,
            stderr_value
        )

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
                confidence_interval=conf_interval
            ),
            generation_config=generation_config,
        )

    def _extract_evaluation_results(
        self,
        scores: List[EvalScore],
        source_data: SourceDataHF,
        generation_config: Dict[str, Any],
        timestamp: str
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for scorer in scores:
            llm_grader = None
            if scorer.params and scorer.params.grader_model:
                llm_grader = LlmScoring(
                    judges=[
                        JudgeConfig(
                            model_info=extract_model_info_from_model_path(
                                scorer.params.grader_model.name
                            )
                        )
                    ],
                    input_prompt=scorer.params.grader_template
                )
            
            stderr_metric = [
                metric_info
                for _, metric_info in scorer.metrics.items()
                if metric_info.name == "stderr"
            ]
            stderr_value = stderr_metric[0].value if stderr_metric else 0.0

            for _, metric_info in scorer.metrics.items():
                if metric_info.name == "stderr":
                    continue

                results.append(
                    self._build_evaluation_result(
                        scorer_name=scorer.scorer,
                        metric_info=metric_info,
                        llm_grader=llm_grader,
                        stderr_value=stderr_value,
                        source_data=source_data,
                        evaluation_timestamp=timestamp,
                        generation_config=generation_config,
                    )
                )

        return results
    
    def _extract_source_data(
        self,
        dataset: EvalDataset
    ) -> SourceDataHF:
        return SourceDataHF( # TODO add hf_split
            source_type='hf_dataset',
            dataset_name=dataset.name.split('/')[-1],
            hf_repo=dataset.location,
            samples_number=dataset.samples,
            sample_ids=dataset.sample_ids
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
        all_tools: List[AvailableTool] = []

        for step in eval_plan.steps:
            if step.solver == "use_tools":
                lists_of_tools = step.params.get("tools") or []
                for tools in lists_of_tools:
                    for tool in tools:
                        description = self._safe_get(tool, 'description')
                        all_tools.append(
                            AvailableTool(
                                name=self._safe_get(tool, 'name'),
                                description=description,
                                parameters=self._safe_get(tool, 'params'),
                            )
                        )

        return all_tools

    def _extract_generation_config(
        self,
        spec: EvalSpec,
        plan: InspectEvalPlan
    ) -> GenerationConfig:
        eval_config = spec.model_generate_config
        eval_generation_config = {
            gen_config: str(value) 
            for gen_config, value in vars(eval_config).items() if value is not None
        }
        eval_sandbox = spec.task_args.get("sandbox")
        sandbox_type, sandbox_config = ((eval_sandbox or []) + [None, None])[:2]

        eval_plan = EvalPlan(
            name=plan.name,
            steps=plan.steps,
            config=plan.config.model_dump(),
        )

        eval_limits = EvalLimits(
            time_limit=spec.config.time_limit,
            message_limit=spec.config.message_limit,
            token_limit=spec.config.token_limit
        )

        max_attempts = spec.task_args.get("max_attempts") or eval_config.max_retries

        reasoning = (
            True 
            if eval_config.reasoning_effort and eval_config.reasoning_effort.lower() != 'none'
            else False
        )

        available_tools: List[AvailableTool] = self._extract_available_tools(
            plan
        )

        generation_args = GenerationArgs(
            temperature=eval_config.temperature,
            top_p=eval_config.top_p,
            top_k=eval_config.top_k,
            max_tokens=eval_config.max_tokens,
            reasoning=reasoning,
            prompt_template=None,
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

        additional_details = ', '.join(
            f"{k}={v}" for k, v in eval_generation_config.items()
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

        source_metadata = SourceMetadata(
            source_name='inspect_ai',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get('source_organization_name'),
            source_organization_url=metadata_args.get('source_organization_url'),
            source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
            evaluator_relationship=metadata_args.get('evaluator_relationship')
        )

        source_data = self._extract_source_data(
            eval_spec.dataset
        )

        model_path = eval_spec.model

        single_sample = raw_eval_log.samples[0] if raw_eval_log.samples else single_sample
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

        generation_config = self._extract_generation_config(eval_spec, raw_eval_log.plan)

        evaluation_results = (
            self._extract_evaluation_results(
                results.scores if results else [],
                source_data,
                generation_config,
                evaluation_unix_timestamp
            )
            if results and results.scores
            else []
        )

        evaluation_id = f'{source_data.dataset_name}/{model_path.replace('/', '_')}/{evaluation_unix_timestamp}'

        detailed_results_id = f'{source_data.dataset_name}_{model_path.replace('/', '_')}_{uuid.uuid4()}'

        instance_level_adapter = InspectInstanceLevelDataAdapter(
            detailed_results_id, 
            Format.jsonl,
            HashAlgorithm.sha256
        )
        instance_level_log_path = instance_level_adapter.convert_instance_level_logs(
            eval_spec.dataset.name,
            model_info.id, 
            raw_eval_log.samples
        )

        detailed_evaluation_results = DetailedEvaluationResults(
            format=Format.jsonl,
            file_path=instance_level_log_path,
            hash_algorithm=HashAlgorithm.sha256,
            checksum=sha256_file(instance_level_log_path),
            total_rows=len(eval_spec.dataset.sample_ids)
        )

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