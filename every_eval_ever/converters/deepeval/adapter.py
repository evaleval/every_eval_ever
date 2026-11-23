"""DeepEval adapter: converts DeepEval test-run JSON to every_eval_ever."""

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Union

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
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
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
    evaluation_result_id,
    is_number,
    metric_key,
    parse_run_timestamp,
)

DEEPEVAL_HARNESS_ID = 'deepeval'

# Bounds are asserted only where DeepEval's own implementation fixes the
# scale: each metric below deterministically returns a 0/1 verdict or a
# ratio in [0, 1], verified in DeepEval's source. Judge metrics are left
# out deliberately -- their scores are parsed from a model's output, and
# no range was verified for them here. Anything unlisted keeps unknown
# bounds (counted in source_metadata) rather than inheriting a range it
# may not have.
DEEPEVAL_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    'exact_match': (0.0, 1.0),
    'pattern_match': (0.0, 1.0),
    'tool_correctness': (0.0, 1.0),
}


class DeepEvalAdapter(BaseEvaluationAdapter):
    """Converts DeepEval test-run JSON files to every_eval_ever format.

    One test-run file becomes one EvaluationLog with one EvaluationResult
    per metric in the run's own ``metricsScores`` summary (score = the
    mean of the per-case scores DeepEval reports). DeepEval logs do not
    record the model under test, so the operator supplies it as
    ``model_id`` in the converter metadata; nothing about the model is
    inferred from the file.
    """

    def __init__(self) -> None:
        super().__init__()
        self._source_name: str | None = None
        # Raw payloads by evaluation_id, so the CLI can build instance
        # sidecars from the same file after transformation.
        self.raw_by_evaluation_id: dict[str, dict[str, Any]] = {}

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='DeepEval to every_eval_ever Adapter',
            description=(
                'Converts DeepEval test-run JSON files to the '
                'every_eval_ever schema'
            ),
            version='1.0.0',
            supported_formats=['.json'],
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.DEEPEVAL

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a DeepEval test-run JSON file into one EvaluationLog."""
        file_path = Path(file_path)
        self._source_name = file_path.name
        raw_data = self._load_file(file_path)
        return [self._transform_single(raw_data, metadata_args)]

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        result = self.transform_from_directory_result(dir_path, metadata_args)
        result.raise_if_incomplete()
        return result.records

    def transform_from_directory_result(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> SourceConversionResult[EvaluationLog]:
        """Transform all DeepEval exports under a directory.

        Finds results-folder exports (test_run_*.json) recursively, or falls
        back to the rolling snapshot at .deepeval/.latest_run_full.json when
        no test_run_*.json files exist under the directory. Files that are not
        valid test runs are reported as failures, never skipped silently.
        """
        dir_path = Path(dir_path)
        candidates = sorted(dir_path.glob('**/test_run_*.json'))
        if not candidates:
            snapshot = dir_path / '.deepeval' / '.latest_run_full.json'
            if snapshot.is_file():
                candidates.append(snapshot)
        if not candidates:
            raise ValueError(
                f'No DeepEval test_run_*.json files found under {dir_path}'
            )

        all_logs: list[EvaluationLog] = []
        failures: list[SourceRecordFailure] = []
        for candidate in candidates:
            try:
                all_logs.extend(
                    self.transform_from_file(candidate, metadata_args)
                )
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(candidate),
                        reason=str(exc),
                        source_record={'path': str(candidate)},
                    )
                )

        return SourceConversionResult(
            source_name=f'DeepEval evaluations under {dir_path}',
            total_records=len(all_logs) + len(failures),
            records=all_logs,
            failures=failures,
        )

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        # .deepeval/.latest_test_run.json wraps the same payload.
        if isinstance(raw_data.get('testRunData'), dict):
            raw_data = raw_data['testRunData']

        model_id = metadata_args.get('model_id')
        if not model_id:
            raise ValueError(
                'model_id is required: DeepEval test-run logs do not '
                'record the model under test, so the operator must supply '
                'it (convert deepeval --model_id <developer/model>)'
            )
        model_name = metadata_args.get('model_name') or model_id
        developer = model_id.split('/')[0] if '/' in model_id else None
        model_info = ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
        )

        retrieved_timestamp = str(int(time.time()))
        evaluation_name = (
            raw_data.get('datasetAlias')
            or self._test_file_stem(raw_data.get('testFile'))
            or 'deepeval'
        )
        # Key the record on the run's own timestamp when the export file
        # name carries one, so re-converting the same export is
        # idempotent; the rolling snapshot falls back to its file stem.
        run_timestamp = parse_run_timestamp(self._source_name or '')
        if run_timestamp is not None:
            run_key = str(run_timestamp)
        elif self._source_name:
            run_key = Path(self._source_name).stem
        else:
            run_key = retrieved_timestamp
        evaluation_id = f'{evaluation_name}/{model_id}/{run_key}'

        cases = raw_data.get('testCases') or []
        conversational = raw_data.get('conversationalTestCases') or []
        per_metric = self._metric_data_by_name(cases, conversational)

        source_data = self._source_data(raw_data, evaluation_name)
        evaluation_results, without_scores = self._build_results(
            raw_data, evaluation_name, source_data, per_metric
        )
        if not evaluation_results:
            raise ValueError(
                'DeepEval test run contains no metric with a score in '
                'metricsScores; nothing convertible to an evaluation result'
            )

        evaluator_relationship = EvaluatorRelationship(
            metadata_args.get('evaluator_relationship', 'first_party')
        )
        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'deepeval'),
            version=metadata_args.get('eval_library_version', 'unknown'),
        )

        unknown_bounds_count = count_unknown_bounds(
            result.metric_config for result in evaluation_results
        )
        additional_details: dict[str, str] = {}
        if unknown_bounds_count:
            additional_details['metrics_with_unknown_bounds'] = str(
                unknown_bounds_count
            )
        if without_scores:
            additional_details['metrics_without_scores'] = str(without_scores)
        # The run's own case-level outcome counts are evaluation
        # outcomes, not operational metadata; carry them rather than
        # dropping them silently.
        if raw_data.get('testPassed') is not None:
            additional_details['deepeval_test_passed'] = str(
                raw_data['testPassed']
            )
        if raw_data.get('testFailed') is not None:
            additional_details['deepeval_test_failed'] = str(
                raw_data['testFailed']
            )
        if conversational:
            additional_details['conversational_test_cases'] = str(
                len(conversational)
            )
        source_metadata = SourceMetadata(
            source_name='deepeval',
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
            additional_details=additional_details or None,
        )

        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=(
                str(run_timestamp) if run_timestamp is not None else None
            ),
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )
        self.raw_by_evaluation_id[evaluation_id] = raw_data
        return log

    @staticmethod
    def _test_file_stem(test_file: Any) -> str | None:
        if not test_file:
            return None
        return Path(str(test_file)).stem or None

    @staticmethod
    def _source_data(
        raw_data: Dict[str, Any], evaluation_name: str
    ) -> SourceDataPrivate:
        details: dict[str, str] = {}
        if raw_data.get('testFile'):
            details['test_file'] = str(raw_data['testFile'])
        if raw_data.get('datasetId'):
            details['dataset_id'] = str(raw_data['datasetId'])
        if raw_data.get('datasetVersion'):
            details['dataset_version'] = str(raw_data['datasetVersion'])
        return SourceDataPrivate(
            dataset_name=evaluation_name,
            source_type='other',
            additional_details=details or None,
        )

    @staticmethod
    def _metric_data_by_name(
        cases: list[dict[str, Any]],
        conversational: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group per-case MetricData entries by DeepEval display name."""
        grouped: dict[str, list[dict[str, Any]]] = {}

        def collect(entries: Any) -> None:
            for entry in entries or []:
                if isinstance(entry, dict) and entry.get('name'):
                    grouped.setdefault(entry['name'], []).append(entry)

        for case in cases:
            collect(case.get('metricsData'))
        for case in conversational:
            collect(case.get('metricsData'))
            for turn in case.get('turns') or []:
                if isinstance(turn, dict):
                    collect(turn.get('metricsData'))
        return grouped

    def _build_results(
        self,
        raw_data: Dict[str, Any],
        evaluation_name: str,
        source_data: SourceDataPrivate,
        per_metric: dict[str, list[dict[str, Any]]],
    ) -> tuple[list[EvaluationResult], int]:
        results: list[EvaluationResult] = []
        without_scores = 0
        for entry in raw_data.get('metricsScores') or []:
            display_name = entry.get('metric')
            scores = [
                float(score)
                for score in entry.get('scores') or []
                if is_number(score)
            ]
            if not display_name or not scores:
                # A metric whose every case errored has no score to
                # publish; it is counted in source_metadata instead.
                without_scores += 1
                continue
            results.append(
                self._build_result(
                    display_name,
                    scores,
                    entry,
                    evaluation_name,
                    source_data,
                    per_metric.get(display_name, []),
                )
            )
        return results, without_scores

    @staticmethod
    def _build_result(
        display_name: str,
        scores: list[float],
        entry: dict[str, Any],
        evaluation_name: str,
        source_data: SourceDataPrivate,
        metric_data: list[dict[str, Any]],
    ) -> EvaluationResult:
        key = metric_key(display_name)
        fields = metric_config_fields(
            key,
            harness=DEEPEVAL_HARNESS_ID,
            bounds_table=DEEPEVAL_METRIC_BOUNDS,
        )
        additional_details = dict(fields['additional_details'] or {})
        if display_name != key:
            additional_details['deepeval_metric_name'] = display_name
        # The schema's llm_scoring block requires the judge's prompt
        # template, which DeepEval run files do not carry; rather than
        # fabricate one, the judge model the file does name is recorded
        # here (and per instance row).
        judge_models = sorted(
            {
                str(md['evaluationModel'])
                for md in metric_data
                if md.get('evaluationModel')
            }
        )
        if len(judge_models) == 1:
            additional_details['evaluation_model'] = judge_models[0]
        elif judge_models:
            additional_details['evaluation_models'] = json.dumps(judge_models)
        thresholds = {
            float(md['threshold'])
            for md in metric_data
            if is_number(md.get('threshold'))
        }
        fields['additional_details'] = additional_details or None
        if len(thresholds) == 1:
            fields['metric_parameters'] = {'threshold': next(iter(thresholds))}

        num_samples = len(scores)
        mean = statistics.fmean(scores)
        std = statistics.stdev(scores) if num_samples > 1 else None
        uncertainty = Uncertainty(
            standard_error=(
                StandardError(
                    value=std / math.sqrt(num_samples),
                    method='analytic',
                )
                if std is not None
                else None
            ),
            standard_deviation=std,
            num_samples=num_samples,
        )

        details: dict[str, str] = {}
        for field in ('passes', 'fails', 'errors'):
            if entry.get(field) is not None:
                details[field] = str(entry[field])

        metric_config = MetricConfig(
            metric_name=key,
            llm_scoring=None,
            **fields,
        )
        return EvaluationResult(
            evaluation_name=evaluation_name,
            evaluation_result_id=evaluation_result_id(display_name),
            source_data=source_data,
            metric_config=metric_config,
            score_details=ScoreDetails(
                score=mean,
                details=details,
                uncertainty=uncertainty,
            ),
        )
