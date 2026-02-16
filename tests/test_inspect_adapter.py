from pathlib import Path

from eval_converters.inspect.adapter import InspectAIAdapter
from eval_converters.inspect.utils import extract_model_info_from_model_path
from eval_types import (
    EvaluationLog, 
    EvaluatorRelationship,
    SourceDataHf,
    SourceMetadata
)

def _load_eval(adapter, filepath, metadata_args):
    eval_path = Path(filepath)
    converted_eval = adapter.transform_from_file(eval_path, metadata_args=metadata_args)
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.evaluation_results[0].source_data, SourceDataHf)

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'inspect_ai'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval


def test_pubmedqa_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_pubmedqa_gpt4o_mini.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1751553870.0'
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'pubmed_qa'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'bigbio/pubmed_qa'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 2

    assert converted_eval.model_info.name == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.id == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2


def test_arc_sonnet_eval():
    adapter = InspectAIAdapter()

    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_sonnet.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1761000045.0'
    assert converted_eval.retrieved_timestamp is not None

    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'ai2_arc'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 5

    assert converted_eval.model_info.name == 'anthropic/claude-sonnet-4-20250514'
    assert converted_eval.model_info.id == 'anthropic/claude-sonnet-4-20250514'
    assert converted_eval.model_info.developer == 'anthropic'
    assert converted_eval.model_info.inference_platform == 'anthropic'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_arc_qwen_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/data_arc_qwen.json', metadata_args)

    assert converted_eval.evaluation_timestamp == '1761001924.0'
    assert converted_eval.retrieved_timestamp is not None

    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'ai2_arc'
    assert converted_eval.evaluation_results[0].source_data.hf_repo == 'allenai/ai2_arc'
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 3

    assert converted_eval.model_info.name == 'ollama/qwen2.5:0.5b'
    assert converted_eval.model_info.id == 'ollama/qwen2.5-0.5b'
    assert converted_eval.model_info.developer == 'ollama'
    assert converted_eval.model_info.inference_platform is None
    assert converted_eval.model_info.inference_engine.name == 'ollama'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 0.3333333333333333

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0

def test_convert_model_path_to_standarized_model_ids():
    model_path_to_standarized_id_map = {
        "openai/gpt-4o-mini": "openai/gpt-4o-mini",
        "openai/azure/gpt-4o-mini": "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4-0": "anthropic/claude-sonnet-4-0",
        "anthropic/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic/claude-3-5-sonnet@20241022",
        "anthropic/vertex/claude-3-5-sonnet-v2@20241022": "anthropic/claude-3-5-sonnet@20241022",
        "google/gemini-2.5-pro": "google/gemini-2.5-pro",
        "google/vertex/gemini-2.0-flash": "google/gemini-2.0-flash",
        "mistral/mistral-large-latest": "mistral/mistral-large-latest",
        "mistral/azure/Mistral-Large-2411": "mistral/Mistral-Large-2411",
        "openai-api/deepseek/deepseek-reasoner": "deepseek/deepseek-reasoner",
        "bedrock/meta.llama2-70b-chat-v1": "meta/llama2-70b-chat",
        "azureai/Llama-3.3-70B-Instruct": "azureai/Llama-3.3-70B-Instruct",
        "together/meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "groq/llama-3.1-70b-versatile": "meta-llama/llama-3.1-70b-versatile",
        "fireworks/accounts/fireworks/models/deepseek-r1-0528": "deepseek-ai/deepseek-r1-0528",
        "sambanova/DeepSeek-V1-0324": "deepseek-ai/DeepSeek-V1-0324",
        "cf/meta/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
        "perplexity/sonar": "perplexity/sonar",
        "hf/openai-community/gpt2": "openai-community/gpt2",
        "vllm/openai-community/gpt2": "openai-community/gpt2",
        "vllm/meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "sglang/meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "ollama/llama3.1": "ollama/llama3.1",
        "llama-cpp-python/llama3": "llama-cpp-python/llama3",
        "openrouter/gryphe/mythomax-l2-13b": "gryphe/mythomax-l2-13b",
        "hf-inference-providers/openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "hf-inference-providers/openai/gpt-oss-120b:cerebras": "openai/gpt-oss-120b:cerebras",
    }

    for model_path, model_id in model_path_to_standarized_id_map.items():
        model_info = extract_model_info_from_model_path(model_path)
        assert model_info.id == model_id