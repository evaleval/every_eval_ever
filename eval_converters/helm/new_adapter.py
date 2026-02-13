import os
import datetime
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from dacite import from_dict

# HELM specific imports
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.adaptation.scenario_state import AdapterSpec, RequestState, ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json

# New Schema Types
from eval_types import (
    AdditionalPropertiesObject,
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
    HashAlgorithm
)

from instance_level_types import (
    InstanceLevelEvaluationLog,
    InteractionType,
    Input,
    Output,
    Evaluation as InstanceEvaluation,
    AnswerAttributionItem,
    TokenUsage,
    Performance
)

from eval_converters.common.adapter import AdapterMetadata, BaseEvaluationAdapter, SupportedLibrary
from eval_converters.helm.instance_level_adapter import (
    HELMInstanceLevelDataAdapter
)
from eval_converters import SCHEMA_VERSION

# initialize the registry
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
    
    def _load_evaluation_run_logfiles(self, dir_path) -> Dict:
        scenario_state_dict = self._load_file(Path(f'{dir_path}/{self.SCENARIO_STATE_FILE}'))
        run_spec_dict = self._load_file(Path(f'{dir_path}/{self.RUN_SPEC_FILE}'))
        scenario_dict = self._load_file(Path(f'{dir_path}/{self.SCENARIO_FILE}'))
        stats = self._load_file(Path(f'{dir_path}/{self.STATS_FILE}'))
		
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
        all_instance_logs: List[InstanceLevelEvaluationLog] = []
        aggregate_logs: List[EvaluationLog] = []

        if self._directory_contains_required_files(dir_path):
            data = self._load_evaluation_run_logfiles(dir_path)
            agg, instances = self._transform_single(data, metadata_args)
            aggregate_logs.append(agg)
            all_instance_logs.extend(instances)
        else:
            for entry in os.scandir(dir_path):
                if entry.is_dir() and self._directory_contains_required_files(entry.path):
                    data = self._load_evaluation_run_logfiles(entry.path)
                    agg, instances = self._transform_single(data, metadata_args)
                    aggregate_logs.append(agg)
                    all_instance_logs.extend(instances)

        # Write all consolidated instance logs to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for log in all_instance_logs:
                f.write(json.dumps(log.model_dump(), ensure_ascii=False) + '\n')
        
        return aggregate_logs


    def _extract_generation_args(self, adapter_spec: AdapterSpec, request_state: RequestState = None) -> GenerationArgs:
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

        # TODO check it
        reasoning = bool(adapter_spec.chain_of_thought_prefix or adapter_spec.chain_of_thought_suffix)

        # [instructions] + [instance_prefix] + [input_prefix] + {input} + [input_suffix] + [output_prefix]
        template_parts = []
        if adapter_spec.instructions:
            template_parts.append(adapter_spec.instructions)
        
        # instance_prefix usually handles the spacing between few-shot examples
        if adapter_spec.instance_prefix:
            template_parts.append(f"{adapter_spec.instance_prefix}")
        
        # Define the core input/output structure
        input_template = (
            f"{adapter_spec.input_prefix or ''}"
            "{input}"
            f"{adapter_spec.input_suffix or ''}"
            f"{adapter_spec.output_prefix or ''}"
        )
        template_parts.append(input_template)
        
        prompt_template = "\n".join(template_parts) if template_parts else None

        return GenerationArgs(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning=reasoning,
            prompt_template=prompt_template
        )
    
    def _transform_single(
        self, raw_data: Dict, metadata_args: Dict[str, Any]
    ) -> Tuple[EvaluationLog, List[InstanceLevelEvaluationLog]]:
        """Transforms raw HELM data into the unified schema formats."""
        scenario_state = from_dict(data_class=ScenarioState, data=raw_data['scenario_state_dict'])
        run_spec = from_dict(data_class=RunSpec, data=raw_data['run_spec_dict'])
        per_instance_stats_list = raw_data['per_instance_stats']
        stats_raw = [from_dict(data_class=Stat, data=stat_info) for stat_info in raw_data['stats']]
        
        request_states = scenario_state.request_states
        timestamp = str(min(state.result.request_datetime for state in request_states))
        model_info = self._extract_model_info(run_spec.adapter_spec.model_deployment) # scenario_state.adapter_spec
        
        source_data = SourceDataHf( # TODO check if always available HF dataset
            dataset_name=raw_data['scenario_dict'].get('name', 'unknown'),
            source_type="hf_dataset",
            samples_number=len(set(state.instance.id for state in request_states)),
            sample_ids=[state.instance.id for state in request_states],
            additional_details=AdditionalPropertiesObject({
                'scenario_name': run_spec.scenario_spec.class_name,
                'scenario_args': run_spec.scenario_spec.args
            })
        )

        evaluation_id = f"{source_data.dataset_name}/{model_info.id.replace('/', '_')}/{timestamp}"

        # TODO 1. Process Instance Level Logs
        instance_logs: List[InstanceLevelEvaluationLog] = []
        for state in request_states:
            # Match instance to its statistics
            inst_stats = next((s for s in per_instance_stats_list if s.instance_id == state.instance.id), None)
            
            is_correct = False
            score = 0.0
            if inst_stats:
                em_stat = next((s for s in inst_stats.stats if s.name.name == "exact_match"), None)
                if em_stat:
                    score = em_stat.mean
                    is_correct = em_stat.mean > 0

            correct_refs = [r.output.text for r in state.instance.references if "correct" in r.tags]
            
            token_usage = None
            if inst_stats:
                p_tokens = next((s.sum for s in inst_stats.stats if s.name.name == "num_prompt_tokens"), 0)
                c_tokens = next((s.sum for s in inst_stats.stats if s.name.name == "num_completion_tokens"), 0)
                token_usage = TokenUsage(
                    input_tokens=int(p_tokens),
                    output_tokens=int(c_tokens),
                    total_tokens=int(p_tokens + c_tokens)
                )

            instance_logs.append(InstanceLevelEvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                model_id=model_info.id,
                evaluation_name=run_spec.name,
                sample_id=state.instance.id,
                interaction_type=InteractionType.single_turn,
                input=Input(
                    raw=state.instance.input.text,
                    formatted=state.request.prompt,
                    reference=correct_refs[0] if correct_refs else ""
                ),
                output=Output(
                    raw=state.result.completions[0].text if state.result.completions else ""
                ),
                answer_attribution=[AnswerAttributionItem(
                    turn_idx=0,
                    source="output.raw",
                    extracted_value=state.result.completions[0].text.strip() if state.result.completions else "",
                    extraction_method="exact_match",
                    is_terminal=True
                )],
                evaluation=InstanceEvaluation(score=float(score), is_correct=is_correct),
                token_usage=token_usage,
                performance=Performance(
                    latency_ms=state.result.request_time * 1000 if state.result.request_time else None
                )
            ))

        evaluation_results: List[EvaluationResult] = []
        metric_names = []
        for metric_spec in run_spec.metric_specs:
            names = metric_spec.args.get('names')
            if names:
                metric_names.extend(names)
            else:
                metric_names.append(metric_spec.class_name.split('.')[-1])

        for metric_name in set(metric_names):
            metric_config = MetricConfig(
                evaluation_description=metric_name,
                lower_is_better=False, # TODO schema.json check
                score_type=ScoreType.continuous,
                min_score=0,
                max_score=1
            )
            matching_stats = [s for s in stats_raw if s.name.name == metric_name]
            # stats: List[Stat] = [from_dict(data_class=Stat, data=stat_info) for stat_info in stats]
            # details = {field: getattr(stat, field) for field in generic_details_fields}

            for stat in matching_stats:
                evaluation_results.append(
                    EvaluationResult(
                        evaluation_name=run_spec.name,
                        source_data=source_data,
                        evaluation_timestamp=timestamp,
                        metric_config=metric_config,
                        score_details=ScoreDetails(
                            score=stat.mean,
                            details=AdditionalPropertiesObject(
                                {
                                    "count": stat.count, 
                                    "split": stat.name.split,
                                    "perturbation": stat.name.perturbation
                                }
                            )
                        ),
                        generation_config=GenerationConfig(
                            generation_args=self._extract_generation_args(adapter_spec=adapter_spec, request_state=request_state),
                            additional_details=AdditionalPropertiesObject(
                                
                            )
                        )
                    )
                )

        if xd:
            parent_eval_output_dir = metadata_args.get('parent_eval_output_dir')
            detailed_results_id = f'{metadata_args.get('file_uuid')}_samples'
            evaluation_dir = f'{parent_eval_output_dir}/{source_data.dataset_name}/{model_dev}/{model_name}'

            evaluation_name = eval_spec.dataset.name or eval_spec.task

            instance_level_log_path, instance_level_rows_number = HELMInstanceLevelDataAdapter(
                detailed_results_id, 
                Format.jsonl.value, 
                HashAlgorithm.sha256.value, 
                evaluation_dir
            ).convert_instance_level_logs(
                evaluation_name, 
                model_info.id, 
                request_states,
                per_instance_stats_list
            )
        else:
            detailed_evaluation_results = None

        agg_log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            evaluation_timestamp=timestamp,
            retrieved_timestamp=str(int(datetime.datetime.now().timestamp())),
            source_metadata=SourceMetadata(
                source_name='HELM',
                source_type=SourceType.evaluation_run,
                source_organization_name=metadata_args.get('source_organization_name', 'Stanford CRFM'),
                source_organization_url=metadata_args.get('source_organization_url'),
                source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
                evaluator_relationship=metadata_args.get('evaluator_relationship', 'third_party'),
            ),
            model_info=model_info,
            evaluation_results=evaluation_results
        )

        return agg_log, instance_logs