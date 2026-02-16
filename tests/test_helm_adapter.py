from pathlib import Path

from eval_converters.helm.adapter import HELMAdapter
from eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    SourceDataHf,
    SourceMetadata
)


def _load_eval(adapter, filepath, metadata_args):
    eval_dirpath = Path(filepath)

    converted_eval = adapter.transform_from_directory(eval_dirpath, output_path='/tmp/helm_output', metadata_args=metadata_args)

    converted_eval = converted_eval[0]
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.evaluation_results[0].source_data, SourceDataHf)

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'HELM'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval

def test_mmlu_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2', metadata_args)

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'mmlu'
    assert converted_eval.evaluation_results[0].source_data.hf_repo is None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 10

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert any('mmlu' in r.evaluation_name.lower() for r in results)
    assert all(r.metric_config is not None for r in results)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 10

def test_hellswag_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0', metadata_args)

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'hellaswag'
    assert converted_eval.evaluation_results[0].source_data.hf_repo is None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 10

    assert converted_eval.model_info.name == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.id == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.developer == 'eleutherai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].score_details.score is not None
    assert any('hellaswag' in r.evaluation_name.lower() for r in results)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 10

def test_narrativeqa_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/narrative_qa:model=openai_gpt2', metadata_args)

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'narrativeqa'
    assert converted_eval.evaluation_results[0].source_data.hf_repo is None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 5

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert any('narrativeqa' in r.evaluation_name.lower() for r in results)
    assert all(r.metric_config is not None for r in results)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 5
