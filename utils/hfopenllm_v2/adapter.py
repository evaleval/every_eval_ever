"""
Script to convert HuggingFace Open LLM Leaderboard v2 data to the EvalEval schema format.

Data source:
- HF Open LLM Leaderboard v2 API: https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted

Usage:
    uv run python -m utils.hfopenllm_v2.adapter
"""

import time
from typing import Any, Dict, List

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    InferenceEngine,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    fetch_json,
    make_model_info,
    make_source_metadata,
    save_evaluation_log,
)
from every_eval_ever.hfopenllm_v2 import (
    DEPLOYMENT_TYPE,
    EVAL_LIBRARY_REPOSITORY,
    EVAL_LIBRARY_VERSION,
    INFERENCE_ENGINE_NAME,
    INFERENCE_ENGINE_VERSION,
    INFERENCE_PLATFORM,
    MODEL_AVAILABILITY,
    SOURCE_URL,
)

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

METRIC_MAPPING = {
    'ifeval': ('accuracy', 'Accuracy'),
    'bbh': ('accuracy', 'Accuracy'),
    'math': ('exact_match', 'Exact Match'),
    'gpqa': ('accuracy', 'Accuracy'),
    'musr': ('accuracy', 'Accuracy'),
    'mmlu_pro': ('accuracy', 'Accuracy'),
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
    model_data: Dict[str, Any], retrieved_timestamp: str
) -> EvaluationLog:
    """Convert a single model's data to EvaluationLog format."""
    model_id = model_data['model']['name']
    if '/' not in model_id:
        raise ValueError(f"Expected 'org/model' format, got: {model_id}")
    developer, model_name = model_id.split('/', 1)
    model_revision = model_data['model'].get('sha')
    if not isinstance(model_revision, str) or not model_revision.strip():
        model_revision = 'unknown'
    precision = model_data['model'].get('precision')
    if not isinstance(precision, str) or not precision.strip():
        raise ValueError(f'Missing model precision for {model_id}')

    # Build evaluation results
    eval_results: List[EvaluationResult] = []
    for eval_key, eval_data in model_data.get('evaluations', {}).items():
        display_name = eval_data.get(
            'name', EVALUATION_MAPPING.get(eval_key, eval_key)
        )
        description = EVALUATION_DESCRIPTIONS.get(
            display_name, f'Accuracy on {display_name}'
        )
        source_data = SOURCE_DATA_MAPPING.get(eval_key)
        if source_data is None:
            raise ValueError(
                f"Unknown eval_key '{eval_key}' — add it to SOURCE_DATA_MAPPING"
            )
        metric = METRIC_MAPPING.get(eval_key)
        if metric is None:
            raise ValueError(
                f"Unknown eval_key '{eval_key}' — add it to METRIC_MAPPING"
            )
        metric_id, metric_name = metric

        eval_results.append(
            EvaluationResult(
                evaluation_name=display_name,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=description,
                    metric_id=metric_id,
                    metric_name=metric_name,
                    metric_kind='accuracy',
                    metric_unit='proportion',
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(
                    score=round(eval_data.get('value', 0.0), 4),
                ),
            )
        )

    # Build additional details
    additional_details = {
        'deployment_type': DEPLOYMENT_TYPE,
        'model_availability': MODEL_AVAILABILITY,
        'model_revision': model_revision,
        'precision': precision,
    }
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
        inference_platform=INFERENCE_PLATFORM,
        additional_details=additional_details,
    )
    model_info.inference_engine = InferenceEngine(
        name=INFERENCE_ENGINE_NAME,
        version=INFERENCE_ENGINE_VERSION,
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
            version=EVAL_LIBRARY_VERSION,
            additional_details={'fork': EVAL_LIBRARY_REPOSITORY},
        ),
        model_info=model_info,
        evaluation_results=eval_results,
    )


def process_models(
    models_data: List[Dict[str, Any]], output_dir: str = OUTPUT_DIR
):
    """Process a list of model evaluation dicts and save them."""
    retrieved_timestamp = str(time.time())
    count = 0

    for model_data in models_data:
        try:
            model_id = model_data['model']['name']
            if '/' not in model_id:
                raise ValueError(
                    f"Expected 'org/model' format, got: {model_id}"
                )
            developer, model = model_id.split('/', 1)

            # Convert to EvaluationLog
            eval_log = convert_model(model_data, retrieved_timestamp)

            # Save
            filepath = save_evaluation_log(
                eval_log, output_dir, developer, model
            )
            print(f'Saved: {filepath}')
            count += 1

        except Exception as e:
            model_name = model_data.get('model', {}).get('name', 'unknown')
            print(f'Error processing {model_name}: {e}')

    return count


if __name__ == '__main__':
    print(f'Fetching data from {SOURCE_URL}...')
    all_models = fetch_json(SOURCE_URL)

    print(f'Processing {len(all_models)} models...')
    count = process_models(all_models)
    print(f'Done! Processed {count} models.')
