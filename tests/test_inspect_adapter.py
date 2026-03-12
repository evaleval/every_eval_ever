import contextlib
from pathlib import Path
import tempfile

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
    metadata_args = dict(metadata_args)
    metadata_args.setdefault("file_uuid", "test-file-uuid")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args['parent_eval_output_dir'] = tmpdir
        converted_eval = adapter.transform_from_file(eval_path, metadata_args=metadata_args)
    
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(converted_eval.evaluation_results[0].source_data, SourceDataHf)

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'inspect_ai'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval


def _extract_file_uuid_from_detailed_results(converted_eval: EvaluationLog) -> str:
    assert converted_eval.detailed_evaluation_results is not None
    stem = Path(converted_eval.detailed_evaluation_results.file_path).stem
    assert stem.endswith("_samples")
    return stem[: -len("_samples")]


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
    assert results[0].evaluation_name == 'inspect_evals/pubmedqa - choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2


def test_transform_without_metadata_args_uses_defaults(tmp_path, caplog):
    adapter = InspectAIAdapter()
    eval_file = (
        Path(__file__).resolve().parent
        / "data/inspect/data_pubmedqa_gpt4o_mini.json"
    )
    with contextlib.chdir(tmp_path):
        converted_eval = adapter.transform_from_file(
            eval_file.as_posix(),
            metadata_args=None,
        )

    assert isinstance(converted_eval, EvaluationLog)
    assert "Missing metadata_args['file_uuid']" in caplog.text
    assert converted_eval.source_metadata.source_organization_name == 'unknown'
    assert (
        converted_eval.source_metadata.evaluator_relationship
        == EvaluatorRelationship.third_party
    )
    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2
    assert _extract_file_uuid_from_detailed_results(converted_eval) != "none"


def test_transform_directory_assigns_unique_file_uuid_per_log():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / "data/inspect"

    with tempfile.TemporaryDirectory() as tmp_logs_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            "data_pubmedqa_gpt4o_mini.json": "2026-02-01T11-00-00+00-00_pubmedqa_test1.json",
            "data_arc_qwen.json": "2026-02-01T11-05-00+00-00_arc_test2.json",
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                "source_organization_name": "TestOrg",
                "evaluator_relationship": EvaluatorRelationship.first_party,
                "parent_eval_output_dir": tmp_out_dir,
                "file_uuid": "shared-uuid",
            },
        )

    assert len(converted_logs) == 2

    uuids = {_extract_file_uuid_from_detailed_results(log) for log in converted_logs}
    assert "shared-uuid" not in uuids
    assert len(uuids) == 2


def test_transform_directory_uses_file_uuids_metadata_when_provided():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / "data/inspect"
    expected_uuids = ["explicit-uuid-1", "explicit-uuid-2"]

    with tempfile.TemporaryDirectory() as tmp_logs_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            "data_pubmedqa_gpt4o_mini.json": "2026-02-01T11-00-00+00-00_pubmedqa_test1.json",
            "data_arc_qwen.json": "2026-02-01T11-05-00+00-00_arc_test2.json",
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                "source_organization_name": "TestOrg",
                "evaluator_relationship": EvaluatorRelationship.first_party,
                "parent_eval_output_dir": tmp_out_dir,
                "file_uuids": expected_uuids,
            },
        )

    assert len(converted_logs) == 2
    uuids = {_extract_file_uuid_from_detailed_results(log) for log in converted_logs}
    assert uuids == set(expected_uuids)


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
    assert results[0].evaluation_name == 'arc_easy - choice'
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
    assert results[0].evaluation_name == 'arc_easy - choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 0.3333333333333333

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_gaia_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json', metadata_args)

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None
    
    assert converted_eval.evaluation_results[0].source_data.dataset_name == 'GAIA'
    assert converted_eval.evaluation_results[0].source_data.hf_repo is not None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) > 0

    assert converted_eval.model_info.name == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.id == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].evaluation_name == 'gaia - gaia_scorer'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score >= 0.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_humaneval_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(adapter, 'tests/data/inspect/2026-02-24T11-23-20+00-00_humaneval_ENiBTeoXr2dbbNcDtpbVvq.json', metadata_args)
    assert converted_eval.detailed_evaluation_results is not None

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
