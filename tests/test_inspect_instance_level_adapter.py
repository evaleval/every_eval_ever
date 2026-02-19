import json
import tempfile
from pathlib import Path

from eval_converters.inspect.adapter import InspectAIAdapter
from eval_types import EvaluatorRelationship
from instance_level_types import InstanceLevelEvaluationLog, InteractionType


def _load_instance_level_data(adapter, filepath, metadata_args):
    eval_filepath = Path(filepath)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args['parent_eval_output_dir'] = tmpdir
        converted_eval = adapter.transform_from_file(
            eval_filepath,
            metadata_args=metadata_args
        )
    
        instance_level_path = Path(converted_eval.detailed_evaluation_results.file_path)
        instance_logs = []
        with instance_level_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instance_logs.append(InstanceLevelEvaluationLog.model_validate(data))

    return converted_eval, instance_logs


def test_pubmedqa_instance_level():
    adapter = InspectAIAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_pubmedqa'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
            metadata_args
        )
        
        assert len(instance_logs) == 2
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'openai/gpt-4o-mini-2024-07-18'
        assert log.interaction_type == InteractionType.single_turn
        
        assert log.input.raw.startswith('Context')
        
        assert log.output.raw == ['A']
        assert log.interactions is None
        
        assert log.evaluation.score == 1.0
        assert log.evaluation.is_correct is True

def test_arc_sonnet_instance_level():
    adapter = InspectAIAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_arc_sonnet'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/inspect/data_arc_sonnet.json',
            metadata_args
        )
        
        assert len(instance_logs) == 5
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'anthropic/claude-sonnet-4-20250514'
        assert log.interaction_type == InteractionType.single_turn
        
        assert len(log.input.choices) == 4
        assert 'Sunlight is the source of energy' in log.input.choices[0]
        
        assert log.output.raw == ['A']
        assert log.interactions is None
        
        assert log.evaluation.score == 1.0
        assert log.evaluation.is_correct is True
        
        assert log.token_usage.input_tokens > 0
        assert log.token_usage.output_tokens > 0
        assert log.token_usage.total_tokens > 0


def test_arc_qwen_instance_level():
    adapter = InspectAIAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_arc_qwen'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/inspect/data_arc_qwen.json',
            metadata_args
        )
        
        assert len(instance_logs) == 3
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'ollama/qwen2.5-0.5b'
        assert log.interaction_type == InteractionType.single_turn
        
        assert log.input.choices == ['Sunlight is the source of energy for nearly all ecosystems.', 'Most ecosystems are found on land instead of in water.', 'Carbon dioxide is more available than other gases.', 'The producers in all ecosystems are plants.']
        
        assert log.evaluation.score == 1.0
        
        assert log.performance.latency_ms > 0


def test_gaia_instance_level():
    adapter = InspectAIAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_gaia'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json',
            metadata_args
        )
        
        assert len(instance_logs) > 0
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'openai/gpt-4.1-mini-2025-04-14'
        
        assert log.interaction_type == InteractionType.agentic
        
        assert log.input.raw is not None or log.input.choices is not None
        
        assert log.output is None
        assert log.interactions is not None
        assert any([i.role for i in log.interactions if i.role == 'tool'])
        
        assert len(log.interactions) > 2
        assert log.interactions[0].turn_idx == 0
        assert log.interactions[0].role == 'system'
        assert log.interactions[1].role == 'user'

        assert log.evaluation.score >= 0.0
        
        assert log.token_usage is not None
        assert log.token_usage.input_tokens >= 0
        assert log.token_usage.output_tokens >= 0
