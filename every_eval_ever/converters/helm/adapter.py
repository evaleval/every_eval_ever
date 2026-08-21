import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, cast

_HELM_IMPORT_ERROR: Exception | None = None
try:
    from dacite import Config as DaciteConfig
    from dacite import from_dict
    from helm.benchmark.adaptation.scenario_state import (
        AdapterSpec,
        RequestState,
        ScenarioState,
    )
    from helm.benchmark.config_registry import (
        register_builtin_configs_from_helm_package,
    )
    from helm.benchmark.metrics.metric import PerInstanceStats
    from helm.benchmark.metrics.statistic import Stat
    from helm.benchmark.model_deployment_registry import (
        ModelDeploymentNotFoundError,
        get_model_deployment,
    )
    from helm.benchmark.run_spec import RunSpec
    from helm.common.codec import from_json
except (
    Exception
) as ex:  # pragma: no cover - exercised only when optional deps missing
    _HELM_IMPORT_ERROR = ex
    DaciteConfig = cast(Any, None)
    from_dict = cast(Any, None)
    PerInstanceStats = cast(Any, None)
    AdapterSpec = cast(Any, None)
    RequestState = cast(Any, None)
    ScenarioState = cast(Any, None)
    Stat = cast(Any, None)
    RunSpec = cast(Any, None)
    get_model_deployment = cast(Any, None)
    register_builtin_configs_from_helm_package = cast(Any, None)
    from_json = cast(Any, None)
    ModelDeploymentNotFoundError = cast(Any, Exception)

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from every_eval_ever.converters.common.metrics import (
    count_unknown_bounds,
    metric_config_fields,
)
from every_eval_ever.converters.common.utils import sha256_file
from every_eval_ever.converters.helm.instance_level_adapter import (
    HELMInstanceLevelDataAdapter,
    _evaluation_result_id,
    _score_from_stat,
    _stat_name_part,
)
from every_eval_ever.converters.helm.metrics import (
    HELM_HARNESS_ID,
    HELM_METRIC_BOUNDS,
    is_core_metric,
    metric_bounds_name,
    metric_parameters,
)
from every_eval_ever.converters.helm.utils import extract_reasoning
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    Format,
    GenerationArgs,
    GenerationConfig,
    HashAlgorithm,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    SourceDataHf,
    SourceMetadata,
    SourceType,
    Uncertainty,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
    datastore_output_dir,
    datastore_repo_file_path,
    require_identity,
    require_uuid4,
)


def _instance_counts_by_result_id(per_instance_stats: List) -> Dict[str, int]:
    """How many distinct instances each aggregate result was computed over.

    HELM aggregates a run by averaging one value per train trial
    (``Stat.take_mean()`` in ``helm.benchmark.metrics.metric``), so the ``count``
    on a stat in ``stats.json`` is the number of trials, which is 1 for the
    single-trial runs that make up almost every published suite. The number of
    instances behind a score is therefore only recoverable from the per-instance
    stats, keyed the same way the aggregate results are.
    """
    instance_ids: Dict[str, set] = {}
    for entry in per_instance_stats:
        instance_id = getattr(entry, 'instance_id', None)
        if instance_id is None:
            continue
        for stat in getattr(entry, 'stats', None) or []:
            stat_name = getattr(stat, 'name', None)
            result_id = _evaluation_result_id(
                getattr(stat_name, 'name', None),
                getattr(stat_name, 'split', None),
                getattr(stat_name, 'perturbation', None),
            )
            if result_id is None:
                continue
            instance_ids.setdefault(result_id, set()).add(str(instance_id))
    return {
        result_id: len(ids) for result_id, ids in instance_ids.items() if ids
    }


def _instance_counts_by_split(stats_raw: List) -> Dict[str, int]:
    """The instance count HELM itself reports for each split.

    HELM emits a ``num_instances`` stat per split, whose mean is that split's
    instance count. It is the only sample count available for a run converted
    without its per-instance stats, where the alternative is the run-wide count
    covering every split at once.
    """
    counts: Dict[str, int] = {}
    for stat in stats_raw:
        stat_name = getattr(stat, 'name', None)
        if getattr(stat_name, 'name', None) != 'num_instances':
            continue
        if getattr(stat_name, 'perturbation', None) is not None:
            continue
        split = _stat_name_part(getattr(stat_name, 'split', None)) or ''
        mean = getattr(stat, 'mean', None)
        if isinstance(mean, (int, float)) and mean > 0:
            counts[split] = int(mean)
    return counts


def _require_helm_dependencies() -> None:
    if _HELM_IMPORT_ERROR is not None:
        raise ImportError(
            'HELM converter dependencies are missing. '
            'Install with: uv sync --extra helm '
            "(or pip install 'every_eval_ever[helm]')."
        ) from _HELM_IMPORT_ERROR


if register_builtin_configs_from_helm_package is not None:
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
    REQUIRED_LOG_FILES = [
        SCENARIO_STATE_FILE,
        RUN_SPEC_FILE,
        SCENARIO_FILE,
        PER_INSTANCE_STATS_FILE,
    ]

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='HELMAdapter',
            version='0.0.1',
            supported_library_versions=['helm'],
            description='HELM adapter with dynamic metrics and unified JSONL instance logging',
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.HELM

    def _directory_contains_required_files(self, dir_path):
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            return all(
                required_file in files
                for required_file in self.REQUIRED_LOG_FILES
            )

        return False

    def _split_model_id(self, model_id: str | None) -> tuple[str, str]:
        """Split a required developer/model identifier."""
        model_id = require_identity(model_id, 'HELM model id')
        if '/' not in model_id:
            raise ValueError(
                f"HELM model id must use 'developer/model' format: {model_id!r}"
            )
        developer, name = model_id.split('/', 1)
        return (
            require_identity(developer, 'HELM model developer'),
            require_identity(name, 'HELM model name'),
        )

    def _extract_model_info(self, adapter_spec: AdapterSpec) -> ModelInfo:
        """Extracts model metadata from HELM, tolerating missing deployments."""
        fallback_model_name = getattr(adapter_spec, 'model', None)
        model_deployment_name = (
            getattr(adapter_spec, 'model_deployment', None) or ''
        ).strip()

        if not model_deployment_name:
            model_name = require_identity(
                fallback_model_name,
                'HELM adapter_spec.model',
            )
            developer, _ = self._split_model_id(model_name)
            return ModelInfo(
                name=model_name,
                id=model_name,
                developer=developer,
                inference_platform='unknown',
            )

        try:
            deployment = get_model_deployment(model_deployment_name)
        except ModelDeploymentNotFoundError:
            model_name = require_identity(
                fallback_model_name or model_deployment_name,
                'HELM model id',
            )
            developer, _ = self._split_model_id(model_name)
            inference_platform = (
                model_deployment_name.split('/', 1)[0]
                if '/' in model_deployment_name
                else 'unknown'
            )
            return ModelInfo(
                name=model_name,
                id=model_name,
                developer=developer,
                inference_platform=inference_platform,
            )

        client_args = getattr(deployment.client_spec, 'args', None)

        if 'huggingface' in deployment.name or not client_args:
            model_id = deployment.model_name
        else:
            model_id = client_args.get(
                'pretrained_model_name_or_path', deployment.model_name
            )

        developer, _ = self._split_model_id(deployment.model_name)
        return ModelInfo(
            name=deployment.model_name,
            id=model_id,
            developer=developer,
            inference_platform=deployment.name.split('/', 1)[0],
        )

    def _load_file_if_exists(self, dir_path, file_name) -> Any:
        path = Path(f'{dir_path}/{file_name}')
        if path.exists():
            return self._load_file(path)

        return None

    def _load_evaluation_run_logfiles(self, dir_path) -> Dict:
        """Load the HELM files needed for aggregate and detail conversion."""
        scenario_state_dict = self._load_file_if_exists(
            dir_path, self.SCENARIO_STATE_FILE
        )
        run_spec_dict = self._load_file_if_exists(dir_path, self.RUN_SPEC_FILE)
        scenario_dict = self._load_file_if_exists(dir_path, self.SCENARIO_FILE)
        stats = self._load_file_if_exists(dir_path, self.STATS_FILE)

        with open(f'{dir_path}/{self.PER_INSTANCE_STATS_FILE}', 'r') as f:
            per_instance_stats = from_json(f.read(), List[PerInstanceStats])

        return {
            'per_instance_stats': per_instance_stats,
            'run_spec_dict': run_spec_dict,
            'scenario_dict': scenario_dict,
            'scenario_state_dict': scenario_state_dict,
            'stats': stats,
        }

    def transform_from_directory(
        self,
        dir_path: str | Path,
        metadata_args: Dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> List[EvaluationLog]:
        result = self.transform_from_directory_result(
            dir_path,
            metadata_args=metadata_args,
            output_path=output_path,
        )
        result.raise_if_incomplete()
        return [log for log, _ in result.records]

    def transform_from_directory_result(
        self,
        dir_path: str | Path,
        metadata_args: Dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> SourceConversionResult[tuple[EvaluationLog, str | None]]:
        """
        Transform HELM runs while retaining failures for individual run dirs.
        """
        aggregate_logs: list[tuple[EvaluationLog, str | None]] = []
        failures: list[SourceRecordFailure] = []
        metadata_args = metadata_args or {}
        if output_path and not metadata_args.get('parent_eval_output_dir'):
            metadata_args = {
                **metadata_args,
                'parent_eval_output_dir': output_path,
            }
        dir_path = str(dir_path)

        file_uuids = metadata_args.get('file_uuids')
        writes_samples = bool(metadata_args.get('parent_eval_output_dir'))

        if self._directory_contains_required_files(dir_path):
            run_paths = [dir_path]
            if file_uuids is not None:
                if not isinstance(file_uuids, list) or len(file_uuids) != 1:
                    raise ValueError(
                        'metadata_args["file_uuids"] must contain exactly one '
                        'UUID for a single HELM run'
                    )
                run_uuids = file_uuids
            else:
                run_uuids = [metadata_args.get('file_uuid')]
        else:
            run_entries = sorted(
                (
                    entry
                    for entry in os.scandir(dir_path)
                    if entry.is_dir()
                    and self._directory_contains_required_files(entry.path)
                ),
                key=lambda entry: entry.path,
            )
            if not run_entries:
                raise ValueError(
                    f'No valid HELM run directories found in {dir_path}'
                )
            if writes_samples and (
                not isinstance(file_uuids, list)
                or len(file_uuids) != len(run_entries)
            ):
                raise ValueError(
                    'metadata_args["file_uuids"] must contain exactly one UUID '
                    f'for each HELM run ({len(run_entries)} required)'
                )
            run_paths = [entry.path for entry in run_entries]
            run_uuids = (
                file_uuids if writes_samples else [None] * len(run_paths)
            )

        for converted_idx, run_path in enumerate(run_paths):
            per_log_metadata_args = dict(metadata_args)
            file_uuid = None
            try:
                if writes_samples:
                    file_uuid = require_uuid4(
                        run_uuids[converted_idx],
                        f'file_uuids[{converted_idx}]',
                    )
                    per_log_metadata_args['file_uuid'] = file_uuid
                data = self._load_evaluation_run_logfiles(run_path)
                agg = self._transform_single(data, per_log_metadata_args)
                aggregate_logs.append((agg, file_uuid))
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(run_path),
                        reason=str(exc),
                        source_record={'path': str(run_path)},
                    )
                )

        return SourceConversionResult(
            source_name=f'HELM runs under {dir_path}',
            total_records=len(run_paths),
            records=aggregate_logs,
            failures=failures,
        )

    def _extract_generation_args(
        self, adapter_spec: AdapterSpec, request_state: RequestState
    ) -> GenerationArgs:
        """
        Extracts generation arguments from HELM objects.

        Args:
            adapter_spec: The global adapter specification from run_spec.json.
            request: The specific request object from scenario_state.json (optional).
        """
        req = request_state.request
        temperature = (
            req.temperature
            if req.temperature is not None
            else getattr(adapter_spec, 'temperature', None)
        )
        max_tokens = (
            req.max_tokens
            if req.max_tokens is not None
            else getattr(adapter_spec, 'max_tokens', None)
        )
        # multiple_choice_separate_* methods score by log-prob and set max_tokens=0;
        # GenerationArgs requires max_tokens >= 1, so treat 0 as None (not applicable)
        if max_tokens == 0:
            max_tokens = None
        top_p = (
            req.top_p
            if req.top_p is not None
            else getattr(adapter_spec, 'top_p', None)
        )
        top_k = (
            req.top_k_per_token
            if req.top_k_per_token is not None
            else getattr(adapter_spec, 'top_k_per_token', None)
        )

        is_reasoning = extract_reasoning(request_state) is not None

        return GenerationArgs(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning=is_reasoning,
        )

    def _extract_evaluation_time(
        self, request_states: List[RequestState]
    ) -> str | None:
        request_datetimes = [
            state.result.request_datetime
            for state in request_states
            if state.result and state.result.request_datetime
        ]
        return str(min(request_datetimes)) if request_datetimes else None

    def _extract_dataset_name(
        self, run_spec_name: str, scenario_name: str | None
    ) -> str:
        """Prefer scenario metadata, falling back to HELM run-spec names."""
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

    def _transform_single(
        self, raw_data: Dict, metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Convert one HELM run into aggregate JSON plus detail JSONL.

        The aggregate ``evaluation_result_id`` values are generated from
        core metrics in ``stats.json`` with the same helper used by the
        instance converter so every metric-specific detail row can join
        back to an aggregate result.
        """
        run_spec = from_dict(data_class=RunSpec, data=raw_data['run_spec_dict'])
        # cast=[str] coerces int instance IDs to str; newer HELM versions
        # (e.g. long-context suite) store instance.id as int in the JSON.
        try:
            scenario_state = from_dict(
                data_class=ScenarioState,
                data=raw_data['scenario_state_dict'],
                config=DaciteConfig(cast=[str]),
            )
        except AssertionError as exc:
            # MediaObject.__post_init__ asserts that local media files exist.
            # Speech/audio/vision benchmarks store media as local paths; if the
            # asset files were not downloaded alongside the run JSON, this fires.
            raise FileNotFoundError(
                f'Run requires local media assets that are not present on this '
                f'machine. Download the benchmark media files alongside the run '
                f'directory and retry. Original assertion: {exc}'
            ) from exc
        scenario_dict = raw_data['scenario_dict']
        stats_raw = [
            from_dict(data_class=Stat, data=s)
            for s in (raw_data.get('stats') or [])
        ]
        per_instance_stats_list = raw_data['per_instance_stats'] or []

        adapter_spec = run_spec.adapter_spec
        request_states = scenario_state.request_states

        retrieved_timestamp = str(int(datetime.datetime.now().timestamp()))
        evaluation_timestamp = (
            self._extract_evaluation_time(request_states) or retrieved_timestamp
        )

        model_info = self._extract_model_info(adapter_spec)

        dataset_name = self._extract_dataset_name(
            run_spec.name, scenario_dict.get('name') if scenario_dict else None
        )

        source_data = SourceDataHf(  # TODO check if always available HF dataset
            dataset_name=dataset_name,
            source_type='hf_dataset',
            samples_number=len(
                set(state.instance.id for state in request_states)
            ),
            sample_ids=[str(state.instance.id) for state in request_states],
            additional_details={
                'scenario_name': str(run_spec.scenario_spec.class_name),
                'scenario_args': json.dumps(run_spec.scenario_spec.args)
                if run_spec.scenario_spec.args
                else '',
            },
        )

        evaluation_id = f'{source_data.dataset_name}/{model_info.id.replace("/", "_")}/{evaluation_timestamp}'

        # Build aggregate results from core HELM stats themselves, not
        # only from run_spec.metric_specs. The instance-level converter emits
        # one row per core per-instance stat, so aggregate IDs must cover
        # the same core namespace for detailed rows to be joinable.
        # TODO: Consider promoting bookkeeping telemetry into structured
        # fields such as token_usage, performance, metadata, or
        # additional_details in a separate follow-up.
        evaluation_results: List[EvaluationResult] = []
        seen_evaluation_result_ids: set[str] = set()
        instance_counts = _instance_counts_by_result_id(per_instance_stats_list)
        split_instance_counts = _instance_counts_by_split(stats_raw)

        for stat in stats_raw:
            # The ID helper mirrors the instance-level converter. This is the
            # key invariant: detail rows should never introduce metric IDs that
            # are absent from aggregate evaluation_results.
            metric_name = getattr(getattr(stat, 'name', None), 'name', None)
            if not is_core_metric(metric_name):
                continue
            score = _score_from_stat(stat)
            if metric_name is None or score is None:
                continue

            stat_count = getattr(stat, 'count', None)

            evaluation_result_id = _evaluation_result_id(
                metric_name,
                getattr(stat.name, 'split', None),
                getattr(stat.name, 'perturbation', None),
            )
            if evaluation_result_id is None:
                continue
            if evaluation_result_id in seen_evaluation_result_ids:
                continue
            seen_evaluation_result_ids.add(evaluation_result_id)

            metric_config = MetricConfig(
                evaluation_description=metric_name,
                metric_name=metric_name,
                **metric_config_fields(
                    metric_name,
                    harness=HELM_HARNESS_ID,
                    bounds_table=HELM_METRIC_BOUNDS,
                    lookup_name=metric_bounds_name(metric_name),
                    metric_parameters=metric_parameters(metric_name),
                ),
            )

            split = getattr(stat.name, 'split', None)
            perturbation = getattr(stat.name, 'perturbation', None)
            perturbation_label = _stat_name_part(perturbation)

            # HELM's spread is across train trials, not across samples, so it is
            # not the `standard_deviation` the schema asks for and it is 0.0 by
            # construction for the single-trial runs that make up almost every
            # published suite. It travels as itself instead.
            trial_stddev = getattr(stat, 'stddev', None)
            # The per-instance stats state the sample count exactly for this
            # split and perturbation. A worst-case-over-perturbations stat has
            # none of its own and is computed over the instances of its split,
            # which is what the unperturbed stat for the same split counts, and
            # what HELM's own `num_instances` reports for that split. The run's
            # instance count is the last resort, and covers every split at once.
            num_samples = (
                instance_counts.get(evaluation_result_id)
                or instance_counts.get(
                    _evaluation_result_id(metric_name, split)
                )
                or split_instance_counts.get(_stat_name_part(split) or '')
                or source_data.samples_number
                or None
            )

            evaluation_results.append(
                EvaluationResult(
                    evaluation_result_id=evaluation_result_id,
                    evaluation_name=source_data.dataset_name,
                    source_data=source_data,
                    evaluation_timestamp=evaluation_timestamp,
                    metric_config=metric_config,
                    score_details=ScoreDetails(
                        score=score,
                        uncertainty=(
                            Uncertainty(num_samples=num_samples)
                            if num_samples
                            else None
                        ),
                        details={
                            'num_train_trials': str(
                                stat_count if stat_count is not None else ''
                            ),
                            'stddev_across_train_trials': (
                                ''
                                if trial_stddev is None
                                else str(trial_stddev)
                            ),
                            'split': _stat_name_part(split) or '',
                            'perturbation': perturbation_label or '',
                        },
                    ),
                    generation_config=GenerationConfig(
                        generation_args=self._extract_generation_args(
                            adapter_spec=adapter_spec,
                            request_state=request_states[0],
                        ),
                        additional_details={
                            'stop_sequences': json.dumps(
                                request_states[0].request.stop_sequences
                            )
                            if request_states[0].request.stop_sequences
                            else '[]',
                            'presence_penalty': str(
                                request_states[0].request.presence_penalty
                            ),
                            'frequency_penalty': str(
                                request_states[0].request.frequency_penalty
                            ),
                            'num_completions': str(
                                request_states[0].request.num_completions
                            ),
                        },
                    ),
                )
            )

        if not evaluation_results:
            reported = sorted(
                {
                    name
                    for name in (
                        getattr(getattr(stat, 'name', None), 'name', None)
                        for stat in stats_raw
                    )
                    if name
                }
            )
            raise ValueError(
                f'HELM run {run_spec.name!r} reports no metric this converter '
                f'recognizes as a benchmark score, so there is nothing to '
                f'publish. Its stats are: {", ".join(reported) or "(none)"}. '
                f'Add the ones that are scores to CORE_METRIC_PREFIXES in '
                f'every_eval_ever/converters/helm/metrics.py.'
            )

        if request_states:
            parent_eval_output_dir = metadata_args.get('parent_eval_output_dir')
        else:
            parent_eval_output_dir = None
        if request_states and parent_eval_output_dir:
            file_uuid = require_uuid4(
                metadata_args.get('file_uuid'),
                "metadata_args['file_uuid']",
            )
            detailed_results_id = f'{file_uuid}_samples'
            evaluation_dir = datastore_output_dir(
                parent_eval_output_dir,
                source_data.dataset_name,
                model_info.id,
                model_info.developer,
            ).as_posix()

            instance_level_log_path, instance_level_rows_number = (
                HELMInstanceLevelDataAdapter(
                    evaluation_id,
                    detailed_results_id,
                    Format.jsonl.value,
                    HashAlgorithm.sha256.value,
                    evaluation_dir,
                ).convert_instance_level_logs(
                    dataset_name,
                    model_info.id,
                    request_states,
                    per_instance_stats_list,
                )
            )

            detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl,
                file_path=datastore_repo_file_path(
                    source_data.dataset_name,
                    model_info.id,
                    model_info.developer,
                    Path(instance_level_log_path).name,
                ),
                hash_algorithm=HashAlgorithm.sha256,
                checksum=sha256_file(instance_level_log_path),
                total_rows=instance_level_rows_number,
            )
        else:
            detailed_evaluation_results = None

        unknown_bounds_count = count_unknown_bounds(
            result.metric_config for result in evaluation_results
        )
        eval_log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            evaluation_timestamp=evaluation_timestamp,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=SourceMetadata(
                source_name='HELM',
                source_type=SourceType.evaluation_run,
                source_organization_name=metadata_args.get(
                    'source_organization_name'
                )
                or 'Stanford CRFM',
                source_organization_url=metadata_args.get(
                    'source_organization_url'
                ),
                source_organization_logo_url=metadata_args.get(
                    'source_organization_logo_url'
                ),
                evaluator_relationship=metadata_args.get(
                    'evaluator_relationship'
                )
                or 'third_party',
                additional_details=(
                    {'metrics_with_unknown_bounds': str(unknown_bounds_count)}
                    if unknown_bounds_count
                    else None
                ),
            ),
            eval_library=EvalLibrary(
                name=metadata_args.get('eval_library_name', 'helm'),
                version=metadata_args.get('eval_library_version', 'unknown'),
            ),
            model_info=model_info,
            evaluation_results=evaluation_results,
            detailed_evaluation_results=detailed_evaluation_results,
        )

        return eval_log

    def __init__(self, strict_validation: bool = True):
        _require_helm_dependencies()
        super().__init__(strict_validation)
