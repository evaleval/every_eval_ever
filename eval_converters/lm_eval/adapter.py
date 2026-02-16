"""Adapter for converting lm-evaluation-harness output to every_eval_ever format."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eval_converters import SCHEMA_VERSION
from eval_converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from eval_converters.common.utils import get_current_unix_timestamp
from eval_types import (
    AdditionalPropertiesObject,
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

from .utils import (
    parse_model_args,
    MODEL_TYPE_TO_INFERENCE_ENGINE,
    MODEL_TYPE_TO_INFERENCE_PLATFORM,
    KNOWN_METRIC_BOUNDS,
)


class LMEvalAdapter(BaseEvaluationAdapter):
    """Converts lm-evaluation-harness results to every_eval_ever format."""

    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)
        # Stores per-log metadata so callers can find sample files after transform.
        # Keyed by evaluation_id -> {"parent_dir": str, "task_name": str}
        self._eval_metadata = {}

    def get_eval_metadata(self, evaluation_id: str) -> Dict[str, Any]:
        """Return stored metadata for a given evaluation_id."""
        return self._eval_metadata.get(evaluation_id, {})

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="lm-eval-adapter",
            version="0.1.0",
            supported_library_versions=["0.4.*"],
            description="Converts lm-evaluation-harness output to every_eval_ever format",
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.LM_EVAL

    def _extract_model_info(
        self, raw_data: Dict[str, Any], metadata_args: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """Extract model information from lm-eval results."""
        metadata_args = metadata_args or {}
        config = raw_data.get("config", {})
        model_type = config.get("model", "")
        model_args_str = config.get("model_args", "")

        if isinstance(model_args_str, dict):
            model_args = model_args_str
        else:
            model_args = parse_model_args(model_args_str)

        model_name = raw_data.get("model_name", "")
        pretrained = model_args.get("pretrained", model_name)

        developer = None
        if "/" in pretrained:
            developer = pretrained.split("/")[0]

        inference_platform = MODEL_TYPE_TO_INFERENCE_PLATFORM.get(model_type)

        # Determine inference engine name: CLI override > auto-detection from model type
        engine_name = metadata_args.get("inference_engine") or MODEL_TYPE_TO_INFERENCE_ENGINE.get(model_type)
        engine_version = metadata_args.get("inference_engine_version")

        inference_engine = None
        if engine_name:
            inference_engine = InferenceEngine(name=engine_name, version=engine_version)

        additional = {}
        if config.get("model_num_parameters"):
            additional["num_parameters"] = config["model_num_parameters"]
        if config.get("model_dtype"):
            additional["dtype"] = config["model_dtype"]
        if config.get("model_revision"):
            additional["revision"] = config["model_revision"]
        if config.get("model_sha"):
            additional["sha"] = config["model_sha"]
        if raw_data.get("lm_eval_version"):
            additional["lm_eval_version"] = raw_data["lm_eval_version"]
        if model_args_str:
            additional["model_args"] = model_args_str

        return ModelInfo(
            name=pretrained,
            id=pretrained,
            developer=developer,
            inference_platform=inference_platform,
            inference_engine=inference_engine,
            additional_details=(
                AdditionalPropertiesObject.model_validate(additional)
                if additional
                else None
            ),
        )

    def _get_tasks(self, raw_data: Dict[str, Any]) -> List[str]:
        """Get task names that have actual metric results (leaf tasks and groups)."""
        results = raw_data.get("results", {})
        tasks = []
        for task_name, task_results in results.items():
            # Skip group placeholder entries (only have alias and " " keys)
            non_alias_keys = [k for k in task_results if k != "alias"]
            if non_alias_keys == [" "]:
                continue
            # Skip if no numeric metric values
            has_metric = any(
                isinstance(v, (int, float))
                for k, v in task_results.items()
                if k not in ("alias", "samples", "name", "sample_len", "sample_count")
                and "_stderr," not in k
            )
            if not has_metric:
                continue
            tasks.append(task_name)
        return tasks

    def _build_source_data(
        self, task_config: Dict[str, Any], task_name: str
    ):
        """Build source_data from task config."""
        dataset_path = task_config.get("dataset_path", "")
        dataset_name = task_config.get("task", task_name)

        if (
            dataset_path
            and "/" in str(dataset_path)
            and not str(dataset_path).startswith("/")
        ):
            return SourceDataHf(
                dataset_name=dataset_name,
                source_type="hf_dataset",
                hf_repo=dataset_path,
                hf_split=(
                    task_config.get("test_split")
                    or task_config.get("validation_split")
                ),
            )
        return SourceDataPrivate(
            dataset_name=dataset_name,
            source_type="other",
        )

    def _build_generation_config(
        self, task_config: Dict[str, Any]
    ) -> Optional[GenerationConfig]:
        """Build generation config from task config."""
        gen_kwargs = task_config.get("generation_kwargs", {})
        if not gen_kwargs:
            return None

        args = GenerationArgs(
            temperature=gen_kwargs.get("temperature"),
            top_p=gen_kwargs.get("top_p"),
            top_k=gen_kwargs.get("top_k"),
            max_tokens=gen_kwargs.get("max_gen_toks"),
        )

        additional = {}
        for k, v in gen_kwargs.items():
            if k not in ("temperature", "top_p", "top_k", "max_gen_toks"):
                additional[k] = (
                    json.dumps(v) if not isinstance(v, (str, int, float, bool)) else v
                )
        if task_config.get("num_fewshot") is not None:
            additional["num_fewshot"] = task_config["num_fewshot"]

        return GenerationConfig(
            generation_args=args,
            additional_details=(
                AdditionalPropertiesObject.model_validate(additional)
                if additional
                else None
            ),
        )

    def _build_evaluation_results(
        self, raw_data: Dict[str, Any], task_name: str
    ) -> List[EvaluationResult]:
        """Build EvaluationResult list for a single task."""
        task_results = raw_data["results"][task_name]
        task_config = raw_data.get("configs", {}).get(task_name, {})
        higher_is_better = raw_data.get("higher_is_better", {}).get(task_name, {})
        n_samples = raw_data.get("n-samples", {}).get(task_name, {})

        source_data = self._build_source_data(task_config, task_name)
        gen_config = self._build_generation_config(task_config)
        eval_timestamp = raw_data.get("date")
        if eval_timestamp is not None:
            eval_timestamp = str(int(eval_timestamp))

        results = []
        for key, value in task_results.items():
            if key in ("alias", "samples", "name", "sample_len", "sample_count"):
                continue
            if "_stderr," in key:
                continue
            if not isinstance(value, (int, float)):
                continue

            if "," in key:
                metric_name, filter_name = key.split(",", 1)
            else:
                metric_name = key
                filter_name = "none"

            stderr_key = f"{metric_name}_stderr,{filter_name}"
            stderr_val = task_results.get(stderr_key)

            is_higher_better = higher_is_better.get(metric_name, True)

            bounds = KNOWN_METRIC_BOUNDS.get(metric_name)
            min_score = bounds[0] if bounds else None
            max_score = bounds[1] if bounds else None

            description = metric_name
            if filter_name != "none":
                description = f"{metric_name} (filter: {filter_name})"

            metric_config = MetricConfig(
                evaluation_description=description,
                lower_is_better=not is_higher_better,
                score_type=ScoreType.continuous,
                min_score=min_score,
                max_score=max_score,
            )

            uncertainty = None
            num_samples = (
                n_samples.get("effective")
                or task_results.get("samples")
                or task_results.get("sample_len")
            )
            if stderr_val is not None or num_samples:
                uncertainty = Uncertainty(
                    standard_error=(
                        StandardError(value=stderr_val, method="bootstrap")
                        if stderr_val is not None
                        else None
                    ),
                    num_samples=num_samples,
                )

            eval_name = task_name
            if filter_name != "none":
                eval_name = f"{task_name}/{filter_name}"

            results.append(
                EvaluationResult(
                    evaluation_name=eval_name,
                    source_data=source_data,
                    evaluation_timestamp=eval_timestamp,
                    metric_config=metric_config,
                    score_details=ScoreDetails(
                        score=value,
                        uncertainty=uncertainty,
                    ),
                    generation_config=gen_config,
                )
            )

        return results

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Transform a single task's results into an EvaluationLog.

        Expects metadata_args to contain 'task_name' specifying which task.
        """
        task_name = metadata_args["task_name"]
        model_info = self._extract_model_info(raw_data, metadata_args)

        retrieved_timestamp = get_current_unix_timestamp()
        eval_timestamp = raw_data.get("date")
        if eval_timestamp is not None:
            eval_timestamp = str(int(eval_timestamp))

        evaluation_id = f"{task_name}/{model_info.id}/{retrieved_timestamp}"
        evaluation_results = self._build_evaluation_results(raw_data, task_name)

        evaluator_rel_str = metadata_args.get(
            "evaluator_relationship", "first_party"
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        source_metadata = SourceMetadata(
            source_name="lm-evaluation-harness",
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get(
                "source_organization_name", ""
            ),
            source_organization_url=metadata_args.get("source_organization_url"),
            source_organization_logo_url=metadata_args.get(
                "source_organization_logo_url"
            ),
            evaluator_relationship=evaluator_relationship,
        )

        # Store metadata so callers can find sample files after transform
        self._eval_metadata[evaluation_id] = {
            "parent_dir": metadata_args.get("parent_eval_output_dir"),
            "task_name": task_name,
        }

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=eval_timestamp,
            source_metadata=source_metadata,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a lm-eval results JSON file into EvaluationLogs.

        Returns one EvaluationLog per leaf task in the results file.
        """
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)
        tasks = self._get_tasks(raw_data)

        # Pass the parent directory so instance-level adapter can find samples files
        if "parent_eval_output_dir" not in metadata_args:
            metadata_args = {
                **metadata_args,
                "parent_eval_output_dir": str(file_path.parent),
            }

        results = []
        for task_name in tasks:
            task_metadata = {**metadata_args, "task_name": task_name}
            log = self._transform_single(raw_data, task_metadata)
            results.append(log)

        return results

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform all lm-eval results files in a directory.

        Searches for results_*.json files recursively.
        """
        dir_path = Path(dir_path)
        results_files = sorted(dir_path.glob("**/results_*.json"))

        all_logs = []
        for results_file in results_files:
            logs = self.transform_from_file(results_file, metadata_args)
            all_logs.extend(logs)

        return all_logs
