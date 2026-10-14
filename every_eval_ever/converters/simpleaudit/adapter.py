"""Adapter for converting SimpleAudit audit results to every_eval_ever format.

SimpleAudit (https://github.com/SimpleAudit) is an LLM safety-audit engine: a
Target model answers frozen scenarios, an Auditor probes it, and a Judge grades
each scenario with a severity. Its `results.save()` output — the same shape
Studio's export endpoint produces — is consumed here:

    {
      "timestamp": "<iso>",
      "summary": {"total_scenarios", "score" (0-100), "passed", "failed",
                  "severity_distribution": {sev: int}, "token_usage": {...}},
      "issues": [str], "recommendations": [str],
      "results": [{scenario_name, scenario_description, conversation, severity,
                   issues_found, positive_behaviors, summary, recommendations,
                   expected_behavior, judgment, <six *_tokens ints>}, ...]
    }

A repeated-experiment file (`RepeatedExperimentResults.to_dict()`) instead has
`{version, n_repetitions, models[], judge, cancelled, aggregate{}, stability{},
runs{model -> [run]}}`, where each run is the single-run shape above; each model
label yields one aggregate log. Detection: top-level `"results"` means single,
top-level `"runs"` means repeated.

DRAFT: the two-metric shape (a continuous `safety_score` plus a `levels`
metric carrying the severity distribution in its score details) is pending
maintainer sign-off; see README.md.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
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
    Uncertainty,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
)

from .utils import (
    NON_GRADED_SEVERITIES,
    SEVERITY_LEVEL_NAMES,
    SIMPLEAUDIT_HARNESS_ID,
    SIMPLEAUDIT_METRIC_BOUNDS,
    UNIDENTIFIED_MODEL_ID,
    evaluation_result_id,
    resolve_model_id,
)


def _as_int(value: Any) -> int | None:
    """An upstream integer counter, or None when absent or not integral."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


class SimpleAuditAdapter(BaseEvaluationAdapter):
    """Converts SimpleAudit audit results to every_eval_ever format."""

    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='simpleaudit-adapter',
            version='0.1.0-draft',
            supported_library_versions=['*'],
            description=(
                'Converts SimpleAudit audit results (single-run and '
                'repeated-experiment JSON) to every_eval_ever format'
            ),
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        # SimpleAudit is neither of the three named harnesses; CUSTOM is the
        # honest label until the registry carries a slug for it.
        return SupportedLibrary.CUSTOM

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_repeated(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and isinstance(payload.get('runs'), dict)
        )

    @classmethod
    def iter_runs(cls, payload: Any) -> List[tuple[str, Dict[str, Any]]]:
        """Yield `(model_label, single_run_payload)` pairs from any input shape.

        A plain single-run file (top-level `"results"`) is the degenerate case:
        it carries no model name, so the label is `unknown`. A repeated file
        yields one entry per model in `runs`, labelled by the model key.
        """
        if cls._is_repeated(payload):
            runs = payload['runs']
            entries = []
            for model_label, model_runs in runs.items():
                if not isinstance(model_runs, list) or not model_runs:
                    continue
                # One aggregate per model label: the run the engine reports as
                # representative is the first; the rest ride along in the
                # aggregate's additional_details as repetition counts.
                entries.append((str(model_label), model_runs[0]))
            return entries
        if isinstance(payload, dict) and isinstance(
            payload.get('results'), list
        ):
            return [('unknown', payload)]
        raise ValueError(
            'unrecognized SimpleAudit payload: expected a single-run file '
            '(top-level "results") or a repeated-experiment file '
            '(top-level "runs")'
        )

    # ------------------------------------------------------------------
    # Piece builders
    # ------------------------------------------------------------------

    def _extract_model_info(
        self,
        raw_data: Dict[str, Any],
        model_label: str,
        metadata_args: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Extract model information from a SimpleAudit run.

        Single-run files carry no model name at all; only Studio's
        `studio_provenance.target_model` names one. The CLI's `--model_id`
        override wins over both, because a user converting their own run knows
        which model it was. A run with no name anywhere gets the synthetic
        `simpleaudit-unknown/model` id: the datastore publisher requires a
        known developer folder, and this keeps unidentifiable runs publishable
        without inventing an organization.
        """
        metadata_args = metadata_args or {}
        provenance = raw_data.get('studio_provenance')
        target_model = None
        if isinstance(provenance, dict):
            candidate = provenance.get('target_model')
            if isinstance(candidate, str) and candidate.strip():
                target_model = candidate.strip()

        model_id, model_source = resolve_model_id(
            target_model,
            metadata_args.get('model_id'),
            model_label if model_label != 'unknown' else None,
        )

        developer = None
        if '/' in model_id:
            developer = model_id.split('/', 1)[0]

        identified = model_id != UNIDENTIFIED_MODEL_ID
        additional = {
            'model_identified': 'true' if identified else 'false',
            'model_source': model_source,
        }

        return ModelInfo(
            name=model_id,
            id=model_id,
            developer=developer,
            additional_details=additional,
        )

    @staticmethod
    def _build_source_data(raw_data: Dict[str, Any]) -> SourceDataPrivate:
        """The audited scenarios are a private dataset owned by the auditor."""
        provenance = raw_data.get('studio_provenance')
        scenario_set = None
        if isinstance(provenance, dict):
            candidate = provenance.get('scenario_set')
            if isinstance(candidate, str) and candidate.strip():
                scenario_set = candidate.strip()
        dataset_name = scenario_set or 'simpleaudit_scenarios'
        additional = None
        if scenario_set:
            additional = {'scenario_set': scenario_set}
        return SourceDataPrivate(
            dataset_name=dataset_name,
            source_type='other',
            additional_details=additional,
        )

    def _build_safety_score_result(
        self,
        raw_data: Dict[str, Any],
        source_data: SourceDataPrivate,
        evaluation_timestamp: Optional[str],
    ) -> EvaluationResult:
        """Aggregate metric 1: the SimpleAudit safety score (0-100)."""
        summary = raw_data.get('summary') or {}
        score = summary.get('score')
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(
                f'summary.score must be numeric, got {score!r}'
            )
        score = float(score)

        total = _as_int(summary.get('total_scenarios'))
        passed = _as_int(summary.get('passed'))
        failed = _as_int(summary.get('failed'))
        non_graded = 0
        distribution = summary.get('severity_distribution') or {}
        if isinstance(distribution, dict):
            for severity, count in distribution.items():
                if severity in NON_GRADED_SEVERITIES:
                    non_graded += _as_int(count) or 0

        metric_fields = metric_config_fields(
            'safety_score',
            harness=SIMPLEAUDIT_HARNESS_ID,
            bounds_table=SIMPLEAUDIT_METRIC_BOUNDS,
        )
        # A higher SimpleAudit score is unambiguously better, which beats the
        # unknown-polarity marker the helper leaves for a name it does not know.
        metric_fields['lower_is_better'] = False
        metric_details = dict(metric_fields.pop('additional_details') or {})
        metric_details.pop('polarity', None)
        metric_fields['additional_details'] = metric_details or None

        details = {
            'score_basis': (
                'SimpleAudit safety score: 100 minus weighted penalties for '
                'graded findings; ERROR/ungraded scenarios are excluded from '
                'the denominator, so the score does not measure coverage'
            ),
        }
        if total is not None:
            details['total_scenarios'] = str(total)
        if passed is not None:
            details['passed'] = str(passed)
        if failed is not None:
            details['failed'] = str(failed)
        if non_graded:
            details['non_graded_scenarios'] = str(non_graded)
        token_usage = summary.get('token_usage')
        if isinstance(token_usage, dict) and token_usage.get('total') is not None:
            details['total_tokens'] = str(_as_int(token_usage.get('total')))

        uncertainty = None
        if total is not None and total > 0:
            uncertainty = Uncertainty(num_samples=total)

        return EvaluationResult(
            evaluation_result_id=evaluation_result_id('safety_score'),
            evaluation_name='safety_score',
            source_data=source_data,
            evaluation_timestamp=evaluation_timestamp,
            metric_config=MetricConfig(
                evaluation_description='SimpleAudit safety score (0-100)',
                metric_name='safety_score',
                **metric_fields,
            ),
            score_details=ScoreDetails(
                score=score,
                details=details,
                uncertainty=uncertainty,
            ),
        )

    def _build_severity_distribution_result(
        self,
        raw_data: Dict[str, Any],
        source_data: SourceDataPrivate,
        evaluation_timestamp: Optional[str],
    ) -> EvaluationResult:
        """Aggregate metric 2: the severity distribution as a levels metric.

        The schema's `levels` shape is ordinal — one scalar score naming a
        level — but SimpleAudit's distribution is a *count per level*, so the
        dominant-severity scalar is reported as the score and the full
        distribution goes in `score_details.details`, keeping severity explicit
        rather than collapsed to binary.
        """
        summary = raw_data.get('summary') or {}
        distribution = summary.get('severity_distribution') or {}
        if not isinstance(distribution, dict):
            raise ValueError(
                f'summary.severity_distribution must be an object, '
                f'got {distribution!r}'
            )

        # Only judged severities participate in dominance; ERROR/ungraded are
        # recorded in details but never win the dominant level.
        graded_counts = {
            severity: (_as_int(count) or 0)
            for severity, count in distribution.items()
            if severity in SEVERITY_LEVEL_NAMES
        }
        non_graded_counts = {
            severity: (_as_int(count) or 0)
            for severity, count in distribution.items()
            if severity not in SEVERITY_LEVEL_NAMES
        }

        graded_total = sum(graded_counts.values())
        if graded_total:
            # Highest severity present wins ties: reporting the worst finding
            # is the conservative choice for a safety record.
            dominant = max(
                graded_counts,
                key=lambda sev: (graded_counts[sev], SEVERITY_LEVEL_NAMES.index(sev)),
            )
            dominant_index = SEVERITY_LEVEL_NAMES.index(dominant)
        else:
            dominant = 'pass'
            dominant_index = SEVERITY_LEVEL_NAMES.index('pass')

        details = {
            'representation': (
                'score is the index into metric_config.level_names of the '
                'dominant (worst present, most common on ties) judged '
                'severity; per-level counts follow'
            ),
            'level_order': ','.join(SEVERITY_LEVEL_NAMES),
        }
        for severity in SEVERITY_LEVEL_NAMES:
            details[f'count_{severity}'] = str(graded_counts.get(severity, 0))
        for severity, count in sorted(non_graded_counts.items()):
            details[f'count_{severity}'] = str(count)
        details['non_graded_total'] = str(sum(non_graded_counts.values()))

        metric_config = MetricConfig(
            evaluation_description='Scenario severity distribution',
            metric_name='severity_distribution',
            # Not in the registry, so namespaced and marked unregistered via
            # the shared helper.
            metric_id=f'{SIMPLEAUDIT_HARNESS_ID}.severity_distribution',
            metric_unit=None,
            lower_is_better=True,  # a higher (worse) severity level is worse
            score_type='levels',
            level_names=list(SEVERITY_LEVEL_NAMES),
            level_metadata=[
                'no issue found',
                'minor issue',
                'moderate issue',
                'significant issue',
                'critical issue',
            ],
            has_unknown_level=False,
            additional_details={
                'metric_id_status': 'unregistered',
                'note': (
                    'levels shape holds the dominant judged severity; the full '
                    'per-severity counts are in score_details.details because '
                    'a levels score is a single ordinal value, not a vector'
                ),
            },
        )

        uncertainty = None
        if graded_total:
            # The distribution is computed over the judged scenarios; ERROR and
            # ungraded ones are not samples of severity, so they do not count.
            uncertainty = Uncertainty(num_samples=graded_total)

        return EvaluationResult(
            evaluation_result_id=evaluation_result_id(
                'severity_distribution'
            ),
            evaluation_name='severity_distribution',
            source_data=source_data,
            evaluation_timestamp=evaluation_timestamp,
            metric_config=metric_config,
            score_details=ScoreDetails(
                score=float(dominant_index),
                details=details,
                uncertainty=uncertainty,
            ),
        )

    # ------------------------------------------------------------------
    # Log assembly
    # ------------------------------------------------------------------

    def _transform_single(
        self,
        raw_data: Dict[str, Any],
        model_label: str,
        metadata_args: Dict[str, Any],
    ) -> EvaluationLog:
        """Transform one single-run payload into an EvaluationLog."""
        model_info = self._extract_model_info(raw_data, model_label, metadata_args)
        source_data = self._build_source_data(raw_data)

        retrieved_timestamp = get_current_unix_timestamp()
        evaluation_timestamp = raw_data.get('timestamp')
        if evaluation_timestamp is not None:
            evaluation_timestamp = str(evaluation_timestamp)

        # Deterministic id: the run's own timestamp identifies it.
        run_stamp = (
            str(raw_data['timestamp'])
            if raw_data.get('timestamp') is not None
            else retrieved_timestamp
        )
        evaluation_id = (
            f'simpleaudit/{model_info.id.replace("/", "_")}/{run_stamp}'
        )

        evaluation_results = [
            self._build_safety_score_result(
                raw_data, source_data, evaluation_timestamp
            ),
            self._build_severity_distribution_result(
                raw_data, source_data, evaluation_timestamp
            ),
        ]

        evaluator_rel_str = metadata_args.get(
            'evaluator_relationship', 'third_party'
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        library_version = str(
            raw_data.get('simpleaudit_version', '')
            or metadata_args.get('eval_library_version', 'unknown')
        )
        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'simpleaudit'),
            version=library_version,
        )

        # The safety score's denominator excludes ERROR/ungraded scenarios,
        # which the score's own details state; repeat it here so a reader of
        # the header alone cannot misread coverage.
        summary = raw_data.get('summary') or {}
        distribution = summary.get('severity_distribution') or {}
        non_graded = sum(
            _as_int(count) or 0
            for severity, count in distribution.items()
            if severity in NON_GRADED_SEVERITIES
        )

        unknown_bounds_count = count_unknown_bounds(
            result.metric_config for result in evaluation_results
        )
        additional: Dict[str, str] = {
            'score_denominator_note': (
                'safety_score covers only graded scenarios; ERROR and '
                'ungraded scenarios are excluded from its denominator and do '
                'not contribute to the score'
            )
        }
        if non_graded:
            additional['non_graded_scenario_count'] = str(non_graded)
        if unknown_bounds_count:
            additional['metrics_with_unknown_bounds'] = str(unknown_bounds_count)

        source_metadata = SourceMetadata(
            source_name='SimpleAudit',
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
            additional_details=additional or None,
        )

        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=evaluation_timestamp,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )
        # Carried for the CLI's --include-samples path only; dropped before the
        # log is serialized (EvaluationLog forbids extra fields).
        object.__setattr__(log, '_simpleaudit_scenarios', raw_data.get('results') or [])
        return log

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a SimpleAudit results file into EvaluationLogs.

        Returns one EvaluationLog per model label (one for a single-run file,
        N for a repeated-experiment file).
        """
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)
        logs = []
        for model_label, run in self.iter_runs(raw_data):
            logs.append(self._transform_single(run, model_label, metadata_args))
        return logs

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any] = None
    ) -> List[EvaluationLog]:
        result = self.transform_from_directory_result(dir_path, metadata_args)
        result.raise_if_incomplete()
        return result.records

    def transform_from_directory_result(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> SourceConversionResult[EvaluationLog]:
        """Transform every `*_results.json` under a directory, retaining failures."""
        dir_path = Path(dir_path)
        results_files = sorted(dir_path.glob('**/*results*.json'))
        if not results_files:
            raise ValueError(
                f'No SimpleAudit results files found under {dir_path}'
            )

        all_logs: list[EvaluationLog] = []
        failures: list[SourceRecordFailure] = []
        for results_file in results_files:
            try:
                all_logs.extend(
                    self.transform_from_file(results_file, metadata_args)
                )
            except Exception as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=str(results_file),
                        reason=str(exc),
                        source_record={'path': str(results_file)},
                    )
                )

        return SourceConversionResult(
            source_name=f'SimpleAudit evaluations under {dir_path}',
            total_records=len(all_logs) + len(failures),
            records=all_logs,
            failures=failures,
        )
