from pathlib import Path

from scripts.eval_converters.helm.adapter import HELMAdapter
from eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    SourceData,
    SourceMetadata
)


def _load_eval(adapter, filepath, metadata_args):
    eval_dirpath = Path(filepath)
    converted_eval = adapter.transform_from_directory(eval_dirpath, metadata_args=metadata_args)
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.source_data, SourceData)

    assert converted_eval.source_metadata.source_name == 'helm'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval

def test_mmlu_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2', metadata_args)

    assert converted_eval.retrieved_timestamp == '1762354922'
    
    assert converted_eval.source_data.dataset_name == 'mmlu'
    assert converted_eval.source_data.hf_repo is None
    assert len(converted_eval.source_data.sample_ids) == 10

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    metric_names = ['exact_match', 'quasi_exact_match', 'prefix_exact_match', 'quasi_prefix_exact_match']
    
    for result in results:
        assert results[0].evaluation_name == 'multiple_choice_joint'
        assert results[0].metric_config.evaluation_description in metric_names
        # assert results[0].score_details.score == 1.0

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]

    assert sorted(sample_ids) == ['id105', 'id11', 'id131', 'id147', 'id222', 'id259', 'id291', 'id344', 'id59', 'id65']
    assert isinstance(results_per_sample[0].ground_truth, list)
    assert results_per_sample[0].ground_truth[0] == 'C'
    assert results_per_sample[0].response == 'D'
    assert isinstance(results_per_sample[0].choices, list)
    choices = sorted([choice for choice, resp in results_per_sample[0].choices])
    responses = sorted([resp for choice, resp in results_per_sample[0].choices])
    assert choices == ['A', 'B', 'C', 'D']
    assert responses == [
        'external meaning',
        "god's plan",
        'internalmeaning',
        'meaning in an afterlife'
    ]

def test_hellswag_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0', metadata_args)

    assert converted_eval.retrieved_timestamp == '1751729998'
    
    assert converted_eval.source_data.dataset_name == 'hellaswag'
    assert converted_eval.source_data.hf_repo is None
    assert len(converted_eval.source_data.sample_ids) == 10

    assert converted_eval.model_info.name == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.id == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.developer == 'eleutherai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    metric_names = ['exact_match', 'quasi_exact_match', 'prefix_exact_match', 'quasi_prefix_exact_match']
    
    assert results[0].score_details.score == 0.3
    for result in results:
        assert results[0].evaluation_name == 'multiple_choice_joint'
        assert results[0].metric_config.evaluation_description in metric_names

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]

    assert sorted(sample_ids) == ['id41468', 'id41992', 'id42841', 'id44284', 'id44874', 'id45277', 'id46128', 'id47299', 'id47975', 'id49438']
    assert isinstance(results_per_sample[0].ground_truth, list)
    assert results_per_sample[0].ground_truth[0] == 'C'
    assert results_per_sample[0].response == 'B'
    assert isinstance(results_per_sample[0].choices, list)
    choices = sorted([choice for choice, resp in results_per_sample[0].choices])
    responses = sorted([resp for choice, resp in results_per_sample[0].choices])
    assert choices == ['A', 'B', 'C', 'D']

    assert responses == [
        'However, you can also take your color, added color, and texture into account when deciding what to dye, and what you will use it for. [substeps] Consider adding your hair dye to your hair if you have it long or curly.', 
        "If you're not planning on dying your hair, there are other coloration measures you can take to dye your hair. [step] Photoshop hd darkers work well, but don't lack the style that can be coupled with it.", 
        'It is important to select the color that represents your hair type when you register your hair color. [substeps] Traditional semi-permanent dyes will generally not be available for hair color, like blow-dryers, curling irons, and appliances.', 
        "Pick the color that's your favorite, matches your wardrobe best, and/or is most flattering for your eye color and skin tone. Semi-permanent dyes work on all hair colors, but show up brightest on light hair."
    ]

def test_narrativeqa_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/helm/narrative_qa:model=openai_gpt2', metadata_args)

    assert converted_eval.retrieved_timestamp == '1763479296'
    
    assert converted_eval.source_data.dataset_name == 'narrativeqa'
    assert converted_eval.source_data.hf_repo is None
    assert len(converted_eval.source_data.sample_ids) == 5

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    metric_names = ['exact_match', 'quasi_exact_match', 'prefix_exact_match', 'quasi_prefix_exact_match']
    
    for result in results:
        assert results[0].evaluation_name == 'generation'
        assert results[0].metric_config.evaluation_description in metric_names
        # assert results[0].score_details.score == 1.0

    results_per_sample = converted_eval.detailed_evaluation_results_per_samples
    sample_ids = [sample.sample_id for sample in results_per_sample]

    assert sorted(sample_ids) == ['id1123', 'id1332', 'id1340', 'id1413', 'id1514']

    assert isinstance(results_per_sample[0].ground_truth, list)
    assert sorted(results_per_sample[0].ground_truth) == ['The school Mascot', 'the schools mascot']
    assert results_per_sample[0].ground_truth[0] == 'The school Mascot'
    assert results_per_sample[0].response == 'Olive.'
    assert results_per_sample[0].choices is None