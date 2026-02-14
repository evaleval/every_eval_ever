import os
import datetime
from typing import Any, Dict, List, Tuple
from pathlib import Path
from dacite import from_dict

from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.adaptation.scenario_state import AdapterSpec, RequestState, ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json

from eval_types import (
    AdditionalPropertiesObject,
    DetailedEvaluationResults,
    EvaluationLog,
    EvaluationResult,
    MetricConfig,
    ModelInfo,
    ScoreType,
    ScoreDetails,
    SourceMetadata,
    SourceType,
    SourceDataHf,
    GenerationConfig,
    GenerationArgs,
    Format,
    HashAlgorithm,
    Uncertainty
)

from instance_level_types import (
    InstanceLevelEvaluationLog
)

from eval_converters.common.adapter import AdapterMetadata, BaseEvaluationAdapter, SupportedLibrary
from eval_converters.common.utils import sha256_file
from eval_converters.helm.utils import extract_reasoning
from eval_converters.helm.instance_level_adapter import (
    HELMInstanceLevelDataAdapter
)
from eval_converters import SCHEMA_VERSION

register_builtin_configs_from_helm_package()


class HELMAdapter(BaseEvaluationAdapter):
    """
    Adapter for HELM outputs that dynamically extracts all metrics and
    consolidates instance-level logs into a single JSONL file.
    """
    SCENARIO_STATE_FILE = 'scenario_state.json'
    RUN_SPEC_FILE = 'run_spec.json'
    SCENARIO_FILE = 'scenario.json'
    STATS_FILE = 'stats.json'
    PER_INSTANCE_STATS_FILE = 'per_instance_stats.json'
    REQUIRED_LOG_FILES = [SCENARIO_STATE_FILE, RUN_SPEC_FILE, SCENARIO_FILE, PER_INSTANCE_STATS_FILE]

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="HELMAdapter",
            version="0.0.1",
            description="HELM adapter with dynamic metrics and unified JSONL instance logging"
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.HELM

    def _directory_contains_required_files(self, dir_path):
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            return all(required_file in files for required_file in self.REQUIRED_LOG_FILES)
        
        return False
    
    def _extract_model_info(self, model_deployment_name: str) -> ModelInfo:
        """Extracts model metadata from the HELM deployment registry."""
        deployment = get_model_deployment(model_deployment_name)
        client_args = getattr(deployment.client_spec, "args", None)

        if "huggingface" in deployment.name or not client_args:
             model_id = deployment.model_name
        else:
            model_id = client_args.get("pretrained_model_name_or_path", deployment.model_name)

        return ModelInfo(
            name=deployment.model_name,
            id=model_id,
            developer=deployment.model_name.split("/", 1)[0],
            inference_platform=deployment.name.split("/", 1)[0]
        )
    
    def _load_file_if_exists(self, dir_path, file_name) -> Any:
        path = Path(f'{dir_path}/{file_name}')
        if path.exists():
            return self._load_file(path)
        
        return None

    def _load_evaluation_run_logfiles(self, dir_path) -> Dict:
        scenario_state_dict = self._load_file_if_exists(dir_path, self.SCENARIO_STATE_FILE)
        run_spec_dict = self._load_file_if_exists(dir_path, self.RUN_SPEC_FILE)
        scenario_dict = self._load_file_if_exists(dir_path, self.SCENARIO_FILE)
        stats = self._load_file_if_exists(dir_path, self.STATS_FILE)
		
        with open(f'{dir_path}/{self.PER_INSTANCE_STATS_FILE}', "r") as f:
            per_instance_stats = from_json(f.read(), List[PerInstanceStats])
            
        return {
			'per_instance_stats': per_instance_stats,
			'run_spec_dict': run_spec_dict,
			'scenario_dict': scenario_dict,
			'scenario_state_dict': scenario_state_dict,
			'stats': stats
		}

    def transform_from_directory(self, dir_path: str, output_path: str, metadata_args: Dict[str, Any]):
        """
        Transforms HELM results into one aggregate EvaluationLog and one 
        instance-level JSONL file containing all samples.
        """
        # all_instance_logs: List[InstanceLevelEvaluationLog] = []
        aggregate_logs: List[EvaluationLog] = []

        if self._directory_contains_required_files(dir_path):
            data = self._load_evaluation_run_logfiles(dir_path)
            agg = self._transform_single(data, metadata_args)
            aggregate_logs.append(agg)
        else:
            for entry in os.scandir(dir_path):
                if entry.is_dir() and self._directory_contains_required_files(entry.path):
                    data = self._load_evaluation_run_logfiles(entry.path)
                    agg = self._transform_single(data, metadata_args)
                    aggregate_logs.append(agg)

        # # Write all consolidated instance logs to JSONL
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     for log in all_instance_logs:
        #         f.write(json.dumps(log.model_dump(), ensure_ascii=False) + '\n')
        
        return aggregate_logs


    def _extract_generation_args(
        self, 
        adapter_spec: AdapterSpec, 
        request_state: RequestState
    ) -> GenerationArgs:
        """
        Extracts generation arguments from HELM objects.
        
        Args:
            adapter_spec: The global adapter specification from run_spec.json.
            request: The specific request object from scenario_state.json (optional).
        """
        temperature = request_state.request.temperature or getattr(adapter_spec, 'temperature', None)
        max_tokens = request_state.request.max_tokens or getattr(adapter_spec, 'max_tokens', None)
        top_p = request_state.request.top_p or getattr(adapter_spec, 'top_p', None)
        top_k = request_state.request.top_k_per_token or getattr(adapter_spec, 'top_k_per_token', None)

        is_reasoning = extract_reasoning(request_state) is not None

        return GenerationArgs(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning=is_reasoning
        )
    

    def _extract_evaluation_time(self, request_states: List[RequestState]) -> str | None:
        request_datetimes = [
            state.result.request_datetime
            for state in request_states
            if state.result and state.result.request_datetime
        ]
        return str(min(request_datetimes)) if request_datetimes else None
    

    def _extract_dataset_name(self, run_spec_name: str, scenario_name: str | None) -> str:
        if scenario_name:
            return scenario_name
        
        if 'dataset' in run_spec_name:
            eval_metadata = run_spec_name.split(':', 1)
            if len(eval_metadata) > 1:
                fields = eval_metadata[1].split(',')

                for f in fields:
                    if 'dataset' in f and '=' in f:
                        return f.split('=', 1)[1]

        return run_spec_name.split(':')[0]


    def _extract_metric_names(self, run_spec: RunSpec) -> List[str]:
        metric_names = []
        for metric_spec in run_spec.metric_specs:
            names = metric_spec.args.get('names')
            if names:
                metric_names.extend(names)
            else:
                metric_names.append(metric_spec.class_name.split('.')[-1])

        return metric_names

    def _transform_single(
        self, 
        raw_data: Dict, 
        metadata_args: Dict[str, Any]
    ) -> Tuple[EvaluationLog, List[InstanceLevelEvaluationLog]]:
        run_spec = from_dict(data_class=RunSpec, data=raw_data['run_spec_dict'])
        scenario_state = from_dict(data_class=ScenarioState, data=raw_data['scenario_state_dict'])
        scenario_dict = raw_data['scenario_dict']
        stats_raw = [from_dict(data_class=Stat, data=s) for s in (raw_data.get('stats') or [])]
        per_instance_stats_list = raw_data['per_instance_stats'] or []
        
        adapter_spec = run_spec.adapter_spec
        request_states = scenario_state.request_states
        
        retrieved_timestamp=str(int(datetime.datetime.now().timestamp()))
        evaluation_timestamp = self._extract_evaluation_time(request_states) or retrieved_timestamp
        
        model_info = self._extract_model_info(adapter_spec.model_deployment)

        dataset_name = self._extract_dataset_name(
            run_spec.name, 
            scenario_dict.get('name') if scenario_dict else None
        )
        
        source_data = SourceDataHf( # TODO check if always available HF dataset
            dataset_name=dataset_name,
            source_type="hf_dataset",
            samples_number=len(set(state.instance.id for state in request_states)),
            sample_ids=[state.instance.id for state in request_states],
            additional_details=AdditionalPropertiesObject(
                scenario_name=run_spec.scenario_spec.class_name,
                scenario_args=run_spec.scenario_spec.args
            )
        )

        evaluation_id = f"{source_data.dataset_name}/{model_info.id.replace('/', '_')}/{evaluation_timestamp}"

        metric_names = self._extract_metric_names(run_spec)

        evaluation_results: List[EvaluationResult] = []

        for metric_name in set(metric_names):
            metric_config = MetricConfig(
                evaluation_description=metric_name,
                lower_is_better=False, # TODO schema.json check
                score_type=ScoreType.continuous,
                min_score=0,
                max_score=1
            )
            
            matching_stats = [s for s in stats_raw if s.name.name == metric_name and not s.name.perturbation]

            for stat in matching_stats:
                evaluation_name = (
                    f'{metric_name} on {source_data.dataset_name}'
                    if not stat.name.split
                    else f'{metric_name} {stat.name.split} on {source_data.dataset_name}'
                )

                evaluation_results.append(
                    EvaluationResult(
                        evaluation_name=evaluation_name,
                        source_data=source_data,
                        evaluation_timestamp=evaluation_timestamp,
                        metric_config=metric_config,
                        score_details=ScoreDetails(
                            score=stat.mean or (stat.sum / stat.count if stat.count else 0.0),
                            uncertainty=Uncertainty(
                                standard_deviation=stat.stddev,
                                num_samples=adapter_spec.max_eval_instances or len(request_states)
                            ),
                            details=AdditionalPropertiesObject(
                                count=stat.count, 
                                split=stat.name.split,
                                perturbation=stat.name.perturbation
                            )
                        ),
                        generation_config=GenerationConfig(
                            generation_args=self._extract_generation_args(adapter_spec=adapter_spec, request_state=request_states[0]),
                            additional_details=AdditionalPropertiesObject(
                                stop_sequences=request_states[0].request.stop_sequences,
                                presence_penalty=request_states[0].request.presence_penalty,
                                frequency_penalty=request_states[0].request.frequency_penalty,
                                num_completions=request_states[0].request.num_completions
                            )
                        )
                    )
                )

        if request_states:
            parent_eval_output_dir = metadata_args.get('parent_eval_output_dir')
            detailed_results_id = f'{metadata_args.get('file_uuid')}_samples'
            model_dev, model_name = model_info.id.split('/', 1)
            evaluation_dir = f'{parent_eval_output_dir}/{source_data.dataset_name}/{model_dev}/{model_name}'

            instance_level_log_path, instance_level_rows_number = HELMInstanceLevelDataAdapter(
                detailed_results_id, 
                Format.jsonl.value, 
                HashAlgorithm.sha256.value, 
                evaluation_dir
            ).convert_instance_level_logs(
                dataset_name, 
                model_info.id, 
                request_states,
                per_instance_stats_list
            )

            detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl,
                file_path=instance_level_log_path,
                hash_algorithm=HashAlgorithm.sha256,
                checksum=sha256_file(instance_level_log_path),
                total_rows=instance_level_rows_number
            )
        else:
            detailed_evaluation_results = None

        eval_log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            evaluation_timestamp=evaluation_timestamp,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name='HELM',
                source_type=SourceType.evaluation_run,
                source_organization_name=metadata_args.get('source_organization_name') or 'Stanford CRFM',
                source_organization_url=metadata_args.get('source_organization_url'),
                source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
                evaluator_relationship=metadata_args.get('evaluator_relationship') or 'third_party',
            ),
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results=detailed_evaluation_results
        )

        return eval_log