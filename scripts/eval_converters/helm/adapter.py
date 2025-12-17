import os
from typing import Any, Dict, List, Optional, Union
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.presentation.schema import Schema, read_schema
from helm.benchmark.scenarios.scenario import Reference
from helm.benchmark.adaptation.scenario_state import AdapterSpec, RequestState, ScenarioState
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json
from dacite import from_dict
from pathlib import Path

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
from scripts.eval_converters import SCHEMA_VERSION

# run this just once in your process to initialize the registry
register_builtin_configs_from_helm_package()


class HELMAdapter(BaseEvaluationAdapter):
	"""
	Adapter for transforming evaluation outputs from the HELM library
	into the unified schema format.
	"""
	SCENARIO_STATE_FILE = 'scenario_state.json'
	RUN_SPEC_FILE = 'run_spec.json'
	SCENARIO_FILE = 'scenario.json'
	STATS_FILE = 'stats.json'
	PER_INSTANCE_STATS_FILE = 'per_instance_stats.json'
	REQUIRED_LOG_FILES = [SCENARIO_STATE_FILE, RUN_SPEC_FILE, SCENARIO_FILE, SCENARIO_STATE_FILE, PER_INSTANCE_STATS_FILE]

	@property
	def metadata(self) -> AdapterMetadata:
		return AdapterMetadata(
			name="HELMAdapter",
			version="0.0.1",
			description="Adapter for transforming HELM evaluation outputs to unified schema format"
		)

	@property
	def supported_library(self) -> SupportedLibrary:
		return SupportedLibrary.HELM

	@staticmethod
	def get_main_metric_name(run_path: str, schema_path: str) -> str:
		if schema_path.endswith(".json"):
			with open(schema_path, "r") as f:
				schema = from_json(f.read(), Schema)
		elif schema_path.endswith(".yaml"):
			schema = read_schema(schema_path)
		else:
			raise Exception(f"schema_path ended with unknown extension: {schema_path}")
		run_spec_path = os.path.join(run_path, "run_spec.json")
		with open(run_spec_path, "r") as f:
			run_spec = from_json(f.read(), RunSpec)
		for group in run_spec.groups:
			if group in schema.name_to_run_group and "main_name" in schema.name_to_run_group[group].environment:
				return schema.name_to_run_group[group].environment["main_name"]
		raise Exception(f"Could not find main metric name for {run_path}")
	
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
	
	def directory_contains_required_files(self, dir_path):
		if os.path.isdir(dir_path):
			files = os.listdir(dir_path)
			return all(required_file in files for required_file in self.REQUIRED_LOG_FILES)
		
		return False

	def transform_from_directory(
		self, dir_path: str, metadata_args: Dict[str, Any]
	) -> Optional[Union[EvaluationLog, List[EvaluationLog]]]:
		"""
		Transforms evaluation logs found in the specified directory into the unified schema.
		It handles both a single evaluation run (if the directory itself contains log files)
		or multiple runs (if the directory contains subdirectories, each with log files).
		"""
		try:
			if self.directory_contains_required_files(dir_path):
				# Single evaluation run in the current directory
				data = self._load_evaluation_run_logfiles(dir_path)
				return self._transform_single(data, metadata_args)
			else:
				# Multiple evaluation runs in subdirectories
				converted_evals: List[EvaluationLog] = []
				
				for entry in os.scandir(dir_path):
					eval_run_dirpath = entry.path
					if entry.is_dir() and self.directory_contains_required_files(eval_run_dirpath):
						data = self._load_evaluation_run_logfiles(eval_run_dirpath)
						converted_evals.append(self._transform_single(data, metadata_args))
				
				return converted_evals
				
		except Exception as e:
			print(f'Error during conversion to unified schema in directory "{dir_path}": {e}')
			return None
	
	def _get_correct_response(self, references: List[Reference]) -> Optional[List[str]]:
		"""Extracts the text of the first reference that has tags."""
		return [
			ref.output.text
			for ref in references if ref.tags
		]

	def _extract_detailed_evaluation_info_for_samples(
		self, request_states: List[RequestState]
	) -> List[DetailedEvaluationResultsPerSample]:
		"""
		Extracts detailed evaluation information for each sample from the request states.
		"""
		results: List[DetailedEvaluationResultsPerSample] = []
		
		for state in request_states:
			references = state.instance.references or []
			correct_responses = self._get_correct_response(references)

			ground_truth = None
			choices_list = None
			
			if state.output_mapping:
				choices_list = [
					[choice, response] for choice, response in state.output_mapping.items()
				]
				
				ground_truth = [
					choice for choice, response in state.output_mapping.items() 
					if choice in correct_responses or response in correct_responses
				]
				
			elif correct_responses:
				ground_truth = correct_responses
			
			results.append(
				DetailedEvaluationResultsPerSample(
					sample_id=state.instance.id,
					input=state.instance.input.text,
					prompt=state.request.prompt,
					ground_truth=ground_truth,
					response=state.result.completions[0].text.strip() if state.result.completions else '',
					choices=choices_list
				)
			)
				
		return results

	def _extract_model_info(self, adapter_spec: AdapterSpec) -> ModelInfo:
		deployment = get_model_deployment(adapter_spec.model_deployment)
		client_args = getattr(deployment.client_spec, "args", None)

		if "huggingface" in deployment.name or not client_args:
			model_id = deployment.model_name
		else:
			model_id = client_args.get("pretrained_model_name_or_path", deployment.model_name)

		return ModelInfo(
			name=deployment.model_name,
			id=model_id,
			developer=deployment.model_name.split("/", 1)[0],
			inference_platform=deployment.name.split("/", 1)[0],
		)
	
	def _extract_generation_config(self, adapter_spec: AdapterSpec) -> Dict[str, Any]:
		return {
			'temperature': adapter_spec.temperature,
			'max_tokens': adapter_spec.max_tokens,
			'stop_sequences': adapter_spec.stop_sequences,
			'instructions': adapter_spec.instructions,
			'input_prefix': adapter_spec.input_prefix,
			'input_suffix': adapter_spec.input_suffix,
			'output_prefix': adapter_spec.output_prefix,
			'output_suffix': adapter_spec.output_suffix,
			'instance_prefix': adapter_spec.instance_prefix
		}


	def _transform_single(
		self, raw_data: Dict, metadata_args: Dict[str, Any]
	) -> EvaluationLog:
		"""
		Args:
			raw_data: Single evaluation record in HELM format (dictionary with log files generated by HELM, each file is loaded as JSON format)

		Returns:
			EvaluationLog in unified schema format
		"""

		scenario_state_dict = raw_data['scenario_state_dict']
		run_spec_dict = raw_data['run_spec_dict']
		scenario_dict = raw_data['scenario_dict']
		stats = raw_data['stats']

		scenario_state = from_dict(data_class=ScenarioState, data=scenario_state_dict)
		adapter_spec = scenario_state.adapter_spec
		request_states = scenario_state.request_states

		run_spec = from_dict(data_class=RunSpec, data=run_spec_dict)

		stats: List[Stat] = [from_dict(data_class=Stat, data=stat_info) for stat_info in stats]

		timestamp = str(min(state.result.request_datetime for state in request_states))

		source_data = SourceData(
            dataset_name=scenario_dict.get('name'),
            samples_number=len(set(state.instance.id for state in request_states)),#len(request_states),
            sample_ids=[state.instance.id for state in request_states],
			additional_details={
				'scenario_name': run_spec.scenario_spec.class_name,
				'scenario_args': run_spec.scenario_spec.args
			}
        )

		source_metadata = SourceMetadata(
            source_name='helm',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get('source_organization_name'),
            source_organization_url=metadata_args.get('source_organization_url'),
            source_organization_logo_url=metadata_args.get('source_organization_logo_url'),
            evaluator_relationship=metadata_args.get('evaluator_relationship')
        )

		model_info = self._extract_model_info(adapter_spec)

		evaluation_results: List[EvaluationResult] = []

		metric_names = []
		for metric_spec in run_spec.metric_specs: 
			metric_names.extend(
				metric_spec.args.get('names') if metric_spec.args else []
			)

		for metric_name in metric_names:
			metric_config = MetricConfig(
				evaluation_description=metric_name,
				lower_is_better=False, # TODO is not always true, possible to fetch correct value from schema.json
				score_type=ScoreType.continuous,
				min_score=0,
				max_score=1
			)

			# TODO consider to filter out a subset of relevant stats
			for stat in stats:
				if not stat.name.name.startswith(metric_name):
					continue

				generic_details_fields = (
					"count", "sum", "sum_squared", "min", "max", "mean", "variance", "stddev"
				)
				details = {field: getattr(stat, field) for field in generic_details_fields}
				details['split'] = stat.name.split
				details['perturbation'] = stat.name.perturbation

				score_details = ScoreDetails(score=stat.mean, details=details)

				evaluation_results.append(
					EvaluationResult(
						evaluation_name=run_spec.adapter_spec.method,
						evaluation_timestamp=timestamp,
						metric_config=metric_config,
						score_details=score_details,
						detailed_evaluation_results_url=None,
						generation_config=self._extract_generation_config(adapter_spec)
					)
				)
		
		detailed_eval_results = self._extract_detailed_evaluation_info_for_samples(request_states)

		scenario_subject = run_spec.scenario_spec.args.get('subject')
		dataset_unique_name = source_data.dataset_name
		if scenario_subject:
			dataset_unique_name += f"/{scenario_subject}"

		evaluation_id = f'helm/{model_info.id}/{dataset_unique_name}/{timestamp}'

		return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=timestamp,
            source_data=source_data,
            source_metadata=source_metadata,
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results_per_samples=detailed_eval_results
        ) 	