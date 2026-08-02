"""
Script to convert HuggingFace Open LLM Leaderboard v2 data to the EvalEval schema format.

Data source:
- HF Open LLM Leaderboard v2 API: https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted

Usage:
    uv run python -m every_eval_ever.adapters.hfopenllm_v2.adapter
"""

import time
from typing import Any, Dict, List

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    make_model_info,
    make_source_metadata,
    save_evaluation_logs,
    save_failure_report,
)

# Source URL
SOURCE_URL = 'https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted'
OUTPUT_DIR = 'data/hfopenllm_v2'

# Evaluation name mapping from API keys to display names
EVALUATION_MAPPING = {
    'ifeval': 'IFEval',
    'bbh': 'BBH',
    'math': 'MATH Level 5',
    'gpqa': 'GPQA',
    'musr': 'MUSR',
    'mmlu_pro': 'MMLU-PRO',
}


# Evaluation descriptions
EVALUATION_DESCRIPTIONS = {
    'IFEval': 'Accuracy on IFEval',
    'BBH': 'Accuracy on BBH',
    'MATH Level 5': 'Exact Match on MATH Level 5',
    'GPQA': 'Accuracy on GPQA',
    'MUSR': 'Accuracy on MUSR',
    'MMLU-PRO': 'Accuracy on MMLU-PRO',
}

# Source data mapping: eval_key -> SourceDataHf
SOURCE_DATA_MAPPING = {
    'ifeval': SourceDataHf(
        dataset_name='IFEval',
        source_type='hf_dataset',
        hf_repo='google/IFEval',
    ),
    'bbh': SourceDataHf(
        dataset_name='BBH',
        source_type='hf_dataset',
        hf_repo='SaylorTwift/bbh',
    ),
    'math': SourceDataHf(
        dataset_name='MATH Level 5',
        source_type='hf_dataset',
        hf_repo='DigitalLearningGmbH/MATH-lighteval',
    ),
    'gpqa': SourceDataHf(
        dataset_name='GPQA',
        source_type='hf_dataset',
        hf_repo='Idavidrein/gpqa',
    ),
    'musr': SourceDataHf(
        dataset_name='MUSR',
        source_type='hf_dataset',
        hf_repo='TAUR-Lab/MuSR',
    ),
    'mmlu_pro': SourceDataHf(
        dataset_name='MMLU-PRO',
        source_type='hf_dataset',
        hf_repo='TIGER-Lab/MMLU-Pro',
    ),
}


def convert_model(
    model_data: Dict[str, Any],
    retrieved_timestamp: str,
    *,
    source_ref: str | None = None,
    failures: list[SourceRecordFailure] | None = None,
) -> EvaluationLog:
    """Convert one model, optionally retaining unusable metric provenance.

    The strict public behavior is unchanged when ``failures`` is omitted:
    any unusable metric rejects the model. Batch conversion supplies a failure
    list so valid metrics from the same model can still be published.
    """
    model_id = model_data['model']['name']
    if '/' not in model_id:
        raise ValueError(f"Expected 'org/model' format, got: {model_id}")
    developer, model_name = model_id.split('/', 1)

    # Build evaluation results
    eval_results: List[EvaluationResult] = []
    for eval_key, eval_data in model_data.get('evaluations', {}).items():
        try:
            if eval_data.get('value') is None:
                raise ValueError('score is missing')
            display_name = eval_data.get(
                'name', EVALUATION_MAPPING.get(eval_key, eval_key)
            )
            description = EVALUATION_DESCRIPTIONS.get(
                display_name, f'Accuracy on {display_name}'
            )
            source_data = SOURCE_DATA_MAPPING.get(eval_key)
            if source_data is None:
                raise ValueError(
                    f"unknown evaluation key; add '{eval_key}' to "
                    'SOURCE_DATA_MAPPING'
                )

            eval_results.append(
                EvaluationResult(
                    evaluation_name=display_name,
                    source_data=source_data,
                    metric_config=MetricConfig(
                        evaluation_description=description,
                        lower_is_better=False,
                        score_type=ScoreType.continuous,
                        min_score=0.0,
                        max_score=1.0,
                    ),
                    score_details=ScoreDetails(
                        score=round(float(eval_data['value']), 4),
                    ),
                )
            )
        except Exception as exc:
            if failures is None:
                raise ValueError(
                    f"Evaluation '{eval_key}' could not be converted: {exc}"
                ) from exc
            failures.append(
                SourceRecordFailure(
                    source_ref=(
                        f'{source_ref or model_id} evaluation {eval_key!r}'
                    ),
                    reason=str(exc),
                    source_record={
                        'model': model_data.get('model'),
                        'evaluation_key': eval_key,
                        'evaluation': eval_data,
                    },
                )
            )
    if not eval_results:
        raise ValueError('model has no usable evaluation results')

    # Build additional details
    additional_details = {}
    if 'precision' in model_data['model']:
        additional_details['precision'] = str(model_data['model']['precision'])
    if 'architecture' in model_data['model']:
        additional_details['architecture'] = str(
            model_data['model']['architecture']
        )
    if 'params_billions' in model_data.get('metadata', {}):
        additional_details['params_billions'] = str(
            model_data['metadata']['params_billions']
        )

    # Build model info
    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        inference_platform='unknown',
        additional_details=additional_details if additional_details else None,
    )

    # Build evaluation ID
    evaluation_id = (
        f'hfopenllm_v2/{developer}_{model_name}/{retrieved_timestamp}'
    )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=make_source_metadata(
            source_name='HF Open LLM v2',
            organization_name='Hugging Face',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(
            name='lm-evaluation-harness',
            version='0.4.0',
            additional_details={
                'fork': 'https://github.com/huggingface/lm-evaluation-harness/tree/adding_all_changess'
            },
        ),
        model_info=model_info,
        evaluation_results=eval_results,
    )


def convert_models(
    models_data: List[Dict[str, Any]],
    retrieved_timestamp: str | None = None,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert all usable models and preserve every rejected source row."""
    timestamp = retrieved_timestamp or str(time.time())
    outputs = []
    failures: list[SourceRecordFailure] = []
    for index, model_data in enumerate(models_data):
        source_ref = f'model row {index}'
        failure_count_before = len(failures)
        try:
            model_id = model_data['model']['name']
            if '/' not in model_id:
                raise ValueError(
                    f"Expected 'org/model' format, got: {model_id}"
                )
            developer, model = model_id.split('/', 1)
            eval_log = convert_model(
                model_data,
                timestamp,
                source_ref=source_ref,
                failures=failures,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=OUTPUT_DIR,
                    developer=developer,
                    model_name=model,
                )
            )
        except Exception as exc:
            # If every metric was already recorded as unusable, add one
            # model-level entry explaining why no evaluation file was emitted.
            if len(failures) > failure_count_before:
                reason = f'no output written: {exc}'
            else:
                reason = str(exc)
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=reason,
                    source_record=model_data,
                )
            )
    return SourceConversionResult(
        source_name='HF Open LLM v2',
        total_records=len(models_data),
        records=outputs,
        failures=failures,
    )


def process_models(
    models_data: List[Dict[str, Any]], output_dir: str = OUTPUT_DIR
) -> int:
    """Save valid models, report rejected rows, and signal incompleteness."""
    result = convert_models(models_data)
    outputs = [
        EvaluationLogOutput(
            eval_log=record.eval_log,
            base_dir=output_dir,
            developer=record.developer,
            model_name=record.model_name,
        )
        for record in result.records
    ]
    paths = save_evaluation_logs(outputs)
    for path in paths:
        print(f'Saved: {path}')
    if result.failures:
        report_path = save_failure_report(
            result,
            default_failure_report_path(output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    print(f'Fetching data from {SOURCE_URL}...')
    all_models = fetch_json(SOURCE_URL)

    print(f'Processing {len(all_models)} models...')
    count = process_models(all_models)
    print(f'Done! Processed {count} models.')
