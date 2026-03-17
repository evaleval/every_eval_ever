import pytest

pytest.importorskip(
    'inspect_ai',
    reason='inspect-ai not installed; install with: uv sync --extra inspect',
)

import contextlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
from every_eval_ever.converters.common.error import AdapterError
from every_eval_ever.converters.inspect.utils import (
    extract_model_info_from_model_path,
)
from every_eval_ever.eval_types import (
    GenerationConfig,
    EvaluationLog,
    EvaluatorRelationship,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
)


def _load_eval(adapter, filepath, metadata_args):
    eval_path = Path(filepath)
    metadata_args = dict(metadata_args)
    metadata_args.setdefault('file_uuid', 'test-file-uuid')

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args['parent_eval_output_dir'] = tmpdir
        converted_eval = adapter.transform_from_file(
            eval_path, metadata_args=metadata_args
        )

    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(
        converted_eval.evaluation_results[0].source_data, SourceDataHf
    )

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'inspect_ai'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval


def _extract_file_uuid_from_detailed_results(
    converted_eval: EvaluationLog,
) -> str:
    assert converted_eval.detailed_evaluation_results is not None
    stem = Path(converted_eval.detailed_evaluation_results.file_path).stem
    assert stem.endswith('_samples')
    return stem[: -len('_samples')]


def _make_metric(name: str, value: float):
    return SimpleNamespace(name=name, value=value)


def _make_scorer(scorer_name: str, metrics: dict[str, object]):
    return SimpleNamespace(name=scorer_name, scorer=scorer_name, params=None, metrics=metrics)


def test_pubmedqa_eval():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
        metadata_args,
    )

    assert converted_eval.evaluation_timestamp == '1751553870.0'
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name
        == 'pubmed_qa'
    )
    assert (
        converted_eval.evaluation_results[0].source_data.hf_repo
        == 'bigbio/pubmed_qa'
    )
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 2

    assert converted_eval.model_info.name == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.id == 'openai/gpt-4o-mini-2024-07-18'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'accuracy on inspect_evals/pubmedqa for scorer choice'
    assert results[0].metric_config.evaluation_description == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2


def test_transform_without_metadata_args_uses_defaults(tmp_path, caplog):
    adapter = InspectAIAdapter()
    eval_file = (
        Path(__file__).resolve().parent
        / 'data/inspect/data_pubmedqa_gpt4o_mini.json'
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
    assert _extract_file_uuid_from_detailed_results(converted_eval) != 'none'


def test_transform_directory_assigns_unique_file_uuid_per_log():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / 'data/inspect'

    with (
        tempfile.TemporaryDirectory() as tmp_logs_dir,
        tempfile.TemporaryDirectory() as tmp_out_dir,
    ):
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            'data_pubmedqa_gpt4o_mini.json': '2026-02-01T11-00-00+00-00_pubmedqa_test1.json',
            'data_arc_qwen.json': '2026-02-01T11-05-00+00-00_arc_test2.json',
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
                'parent_eval_output_dir': tmp_out_dir,
                'file_uuid': 'shared-uuid',
            },
        )

    assert len(converted_logs) == 2

    uuids = {
        _extract_file_uuid_from_detailed_results(log) for log in converted_logs
    }
    assert 'shared-uuid' not in uuids
    assert len(uuids) == 2


def test_transform_directory_uses_file_uuids_metadata_when_provided():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / 'data/inspect'
    expected_uuids = ['explicit-uuid-1', 'explicit-uuid-2']

    with (
        tempfile.TemporaryDirectory() as tmp_logs_dir,
        tempfile.TemporaryDirectory() as tmp_out_dir,
    ):
        tmp_logs_path = Path(tmp_logs_dir)
        fixture_targets = {
            'data_pubmedqa_gpt4o_mini.json': '2026-02-01T11-00-00+00-00_pubmedqa_test1.json',
            'data_arc_qwen.json': '2026-02-01T11-05-00+00-00_arc_test2.json',
        }
        for source_name, target_name in fixture_targets.items():
            source = fixture_dir / source_name
            target = tmp_logs_path / target_name
            target.write_bytes(source.read_bytes())

        converted_logs = adapter.transform_from_directory(
            tmp_logs_path,
            metadata_args={
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
                'parent_eval_output_dir': tmp_out_dir,
                'file_uuids': expected_uuids,
            },
        )

    assert len(converted_logs) == 2
    uuids = {
        _extract_file_uuid_from_detailed_results(log) for log in converted_logs
    }
    assert uuids == set(expected_uuids)


def test_arc_sonnet_eval():
    adapter = InspectAIAdapter()

    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    converted_eval = _load_eval(
        adapter, 'tests/data/inspect/data_arc_sonnet.json', metadata_args
    )

    assert converted_eval.evaluation_timestamp == '1761000045.0'
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name
        == 'ai2_arc'
    )
    assert (
        converted_eval.evaluation_results[0].source_data.hf_repo
        == 'allenai/ai2_arc'
    )
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 5

    assert (
        converted_eval.model_info.name == 'anthropic/claude-sonnet-4-20250514'
    )
    assert converted_eval.model_info.id == 'anthropic/claude-sonnet-4-20250514'
    assert converted_eval.model_info.developer == 'anthropic'
    assert converted_eval.model_info.inference_platform == 'anthropic'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'accuracy on arc_easy for scorer choice'
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

    converted_eval = _load_eval(
        adapter, 'tests/data/inspect/data_arc_qwen.json', metadata_args
    )

    assert converted_eval.evaluation_timestamp == '1761001924.0'
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name
        == 'ai2_arc'
    )
    assert (
        converted_eval.evaluation_results[0].source_data.hf_repo
        == 'allenai/ai2_arc'
    )
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) == 3

    assert converted_eval.model_info.name == 'ollama/qwen2.5:0.5b'
    assert converted_eval.model_info.id == 'ollama/qwen2.5-0.5b'
    assert converted_eval.model_info.developer == 'ollama'
    assert converted_eval.model_info.inference_platform is None
    assert converted_eval.model_info.inference_engine.name == 'ollama'

    results = converted_eval.evaluation_results
    assert results[0].evaluation_name == 'accuracy on arc_easy for scorer choice'
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

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json',
        metadata_args,
    )

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name == 'GAIA'
    )
    assert converted_eval.evaluation_results[0].source_data.hf_repo is not None
    assert len(converted_eval.evaluation_results[0].source_data.sample_ids) > 0

    assert converted_eval.model_info.name == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.id == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].evaluation_name == 'accuracy on gaia for scorer gaia_scorer'
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

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/2026-02-24T11-23-20+00-00_humaneval_ENiBTeoXr2dbbNcDtpbVvq.json',
        metadata_args,
    )
    assert converted_eval.detailed_evaluation_results is not None


def test_extract_evaluation_results_one_scorer_with_two_metrics():
    adapter = InspectAIAdapter()
    source_data = SourceDataHf(dataset_name="synthetic_ds", source_type="hf_dataset")
    generation_config = GenerationConfig()
    scores = [
        _make_scorer(
            "choice",
            {
                "accuracy": _make_metric("accuracy", 0.75),
                "f1": _make_metric("f1", 0.80),
                "stderr": _make_metric("stderr", 0.05),
            },
        )
    ]

    results = adapter._extract_evaluation_results(
        evaluation_task_name="synthetic/task",
        scores=scores,
        source_data=source_data,
        generation_config=generation_config,
        num_samples=10,
        timestamp="1234567890",
    )

    assert len(results) == 2
    assert {result.evaluation_name for result in results} == {
        "accuracy on synthetic/task for scorer choice",
        "f1 on synthetic/task for scorer choice",
    }


def test_extract_evaluation_results_two_scorers_two_metrics_each():
    adapter = InspectAIAdapter()
    source_data = SourceDataHf(dataset_name="synthetic_ds", source_type="hf_dataset")
    generation_config = GenerationConfig()
    scores = [
        _make_scorer(
            "scorer_a",
            {
                "accuracy": _make_metric("accuracy", 0.91),
                "f1": _make_metric("f1", 0.90),
            },
        ),
        _make_scorer(
            "scorer_b",
            {
                "accuracy": _make_metric("accuracy", 0.88),
                "f1": _make_metric("f1", 0.87),
            },
        ),
    ]

    results = adapter._extract_evaluation_results(
        evaluation_task_name="synthetic/task",
        scores=scores,
        source_data=source_data,
        generation_config=generation_config,
        num_samples=10,
        timestamp="1234567890",
    )

    assert len(results) == 4
    assert {result.evaluation_name for result in results} == {
        "accuracy on synthetic/task for scorer scorer_a",
        "f1 on synthetic/task for scorer scorer_a",
        "accuracy on synthetic/task for scorer scorer_b",
        "f1 on synthetic/task for scorer scorer_b",
    }


def test_convert_model_path_to_standarized_model_ids():
    model_path_to_standarized_id_map = {
        'openai/gpt-4o-mini': 'openai/gpt-4o-mini',
        'openai/azure/gpt-4o-mini': 'openai/gpt-4o-mini',
        'anthropic/claude-sonnet-4-0': 'anthropic/claude-sonnet-4-0',
        'anthropic/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0': 'anthropic/claude-3-5-sonnet@20241022',
        'anthropic/vertex/claude-3-5-sonnet-v2@20241022': 'anthropic/claude-3-5-sonnet@20241022',
        'google/gemini-2.5-pro': 'google/gemini-2.5-pro',
        'google/vertex/gemini-2.0-flash': 'google/gemini-2.0-flash',
        'mistral/mistral-large-latest': 'mistral/mistral-large-latest',
        'mistral/azure/Mistral-Large-2411': 'mistral/Mistral-Large-2411',
        'openai-api/deepseek/deepseek-reasoner': 'deepseek/deepseek-reasoner',
        'bedrock/meta.llama2-70b-chat-v1': 'meta/llama2-70b-chat',
        'azureai/Llama-3.3-70B-Instruct': 'azureai/Llama-3.3-70B-Instruct',
        'together/meta-llama/Meta-Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'groq/llama-3.1-70b-versatile': 'meta-llama/llama-3.1-70b-versatile',
        'fireworks/accounts/fireworks/models/deepseek-r1-0528': 'deepseek-ai/deepseek-r1-0528',
        'sambanova/DeepSeek-V1-0324': 'deepseek-ai/DeepSeek-V1-0324',
        'cf/meta/llama-3.1-70b-instruct': 'meta/llama-3.1-70b-instruct',
        'perplexity/sonar': 'perplexity/sonar',
        'hf/openai-community/gpt2': 'openai-community/gpt2',
        'vllm/openai-community/gpt2': 'openai-community/gpt2',
        'vllm/meta-llama/Meta-Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'sglang/meta-llama/Meta-Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'ollama/llama3.1': 'ollama/llama3.1',
        'llama-cpp-python/llama3': 'llama-cpp-python/llama3',
        'openrouter/gryphe/mythomax-l2-13b': 'gryphe/mythomax-l2-13b',
        'hf-inference-providers/openai/gpt-oss-120b': 'openai/gpt-oss-120b',
        'hf-inference-providers/openai/gpt-oss-120b:cerebras': 'openai/gpt-oss-120b:cerebras',
    }

    for model_path, model_id in model_path_to_standarized_id_map.items():
        model_info = extract_model_info_from_model_path(model_path)
        assert model_info.id == model_id


def test_supplemental_eval_details_fill_only_top_level_fields():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "model_info": {
                "additional_details": {
                    "num_parameters": 42,
                    "is_test_model": True,
                }
            },
            "source_data": {
                "additional_details": {
                    "shuffled": "should_not_overwrite",
                    "subset": {"name": "full"},
                }
            },
            "generation_config": {
                "additional_details": {
                    "runner": "inspect",
                },
            },
            "agentic_eval_config": {
                "additional_details": {
                    "agent_mode": "tool_use",
                }
            },
            "evaluation_results": [
                {
                    "evaluation_name": "accuracy on inspect_evals/pubmedqa for scorer choice",
                    "score_details": {
                        "details": {
                            "notes": ["a", "b"],
                        }
                    },
                    "metric_config": {
                        "lower_is_better": True,
                        "evaluation_description": "should_not_overwrite",
                        "score_type": ScoreType.continuous,
                        "min_score": 0.0,
                        "max_score": 1.0,
                        "additional_details": {
                            "normalization": "none",
                        },
                    },
                },
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        "tests/data/inspect/data_pubmedqa_gpt4o_mini.json",
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]

    assert converted_eval.model_info.additional_details == {
        "num_parameters": "42",
        "is_test_model": "true",
    }
    assert result.source_data.additional_details["shuffled"] == "False"
    assert result.source_data.additional_details["subset"] == '{"name": "full"}'

    assert result.generation_config is not None
    assert result.generation_config.additional_details == {"runner": "inspect"}
    assert result.generation_config.generation_args is not None
    assert result.generation_config.generation_args.agentic_eval_config is not None
    assert (
        result.generation_config.generation_args.agentic_eval_config.additional_details
        == {"agent_mode": "tool_use"}
    )

    assert result.score_details.details == {"notes": '["a", "b"]'}

    # Converter-synthetic defaults are override-eligible.
    assert result.metric_config.lower_is_better is True
    assert result.metric_config.evaluation_description == "should_not_overwrite"
    assert result.metric_config.additional_details == {"normalization": "none"}


def test_supplemental_eval_details_applies_top_level_score_details():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "evaluation_results": [
                {
                    "evaluation_name": "accuracy on inspect_evals/pubmedqa for scorer choice",
                    "score_details": {
                        "details": {
                            "matched": 1,
                        },
                    },
                }
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        "tests/data/inspect/data_pubmedqa_gpt4o_mini.json",
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]

    assert result.score_details.details == {"matched": "1"}


def test_supplemental_eval_details_does_not_overwrite_existing_generation_details():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "generation_config": {
                "additional_details": {
                    "temperature": "999",
                    "added_field": "yes",
                }
            },
        },
    }

    converted_eval = _load_eval(
        adapter,
        "tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json",
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]
    assert result.generation_config is not None
    assert result.generation_config.additional_details is not None
    # existing log value remains
    assert result.generation_config.additional_details["temperature"] == "0.5"
    # missing key gets filled
    assert result.generation_config.additional_details["added_field"] == "yes"


def test_supplemental_eval_details_does_not_apply_when_evaluation_name_does_not_match():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "evaluation_results": [
                {
                    "evaluation_name": "some_other_eval - choice",
                    "score_details": {"details": {"matched": 1}},
                }
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        "tests/data/inspect/data_pubmedqa_gpt4o_mini.json",
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]
    assert result.score_details.details is None


def test_supplemental_eval_details_fails_on_deprecated_per_result_schema():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "per_result": [
                {
                    "match": {
                        "evaluation_name": "accuracy on inspect_evals/pubmedqa for scorer choice",
                    },
                    "score_details": {"details": {"matched": 1}},
                },
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args["file_uuid"] = "test-file-uuid"
        metadata_args["parent_eval_output_dir"] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path("tests/data/inspect/data_pubmedqa_gpt4o_mini.json"),
                metadata_args=metadata_args,
            )


def test_supplemental_eval_details_fails_on_duplicate_evaluation_name():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "evaluation_results": [
                {
                    "evaluation_name": "accuracy on inspect_evals/pubmedqa for scorer choice",
                    "score_details": {"details": {"a": 1}},
                },
                {
                    "evaluation_name": "accuracy on inspect_evals/pubmedqa for scorer choice",
                    "score_details": {"details": {"b": 2}},
                },
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args["file_uuid"] = "test-file-uuid"
        metadata_args["parent_eval_output_dir"] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path("tests/data/inspect/data_pubmedqa_gpt4o_mini.json"),
                metadata_args=metadata_args,
            )


def test_supplemental_eval_details_fails_on_invalid_schema():
    adapter = InspectAIAdapter()
    metadata_args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
        "supplemental_eval_details": {
            "evaluation_results": [
                {
                    "metric_config": {
                        "unsupported_field": "x",
                    }
                }
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args["file_uuid"] = "test-file-uuid"
        metadata_args["parent_eval_output_dir"] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path("tests/data/inspect/data_pubmedqa_gpt4o_mini.json"),
                metadata_args=metadata_args,
            )