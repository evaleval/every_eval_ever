import json
import tempfile
from pathlib import Path

from eval_converters.helm.adapter import HELMAdapter
from eval_types import EvaluatorRelationship
from instance_level_types import InstanceLevelEvaluationLog, InteractionType


def _load_instance_level_data(adapter, filepath, metadata_args):
    eval_dirpath = Path(filepath)
    converted_eval_list = adapter.transform_from_directory(
        eval_dirpath,
        output_path=str(Path(metadata_args['parent_eval_output_dir']) / 'helm_output'),
        metadata_args=metadata_args
    )
    
    converted_eval = converted_eval_list[0]
    
    instance_level_path = Path(converted_eval.detailed_evaluation_results.file_path)
    instance_logs = []
    with instance_level_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instance_logs.append(InstanceLevelEvaluationLog.model_validate(data))
    
    return converted_eval, instance_logs


def test_mmlu_instance_level():
    adapter = HELMAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_mmlu'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter, 
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2', 
            metadata_args
        )
        
        assert len(instance_logs) == 10
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.evaluation_id == 'test_mmlu_samples'
        assert log.model_id == 'openai/gpt2'
        assert log.evaluation_name == 'mmlu'
        assert log.sample_id == 'id147'
        assert len(log.sample_hash) == 64
        assert log.interaction_type == InteractionType.single_turn
        
        assert log.input.raw.startswith('The')
        assert log.input.reference == ['internalmeaning']
        
        assert log.output.raw == [' D']
        
        assert log.messages is None
        
        assert len(log.answer_attribution) == 1
        assert log.answer_attribution[0].turn_idx == 0
        assert log.answer_attribution[0].source == 'output.raw'
        assert log.answer_attribution[0].extraction_method == 'exact_match'
        assert log.answer_attribution[0].is_terminal is True
        
        assert log.evaluation.score == 0.0
        assert log.evaluation.is_correct is False
        
        assert log.token_usage.input_tokens > 0
        assert log.token_usage.output_tokens > 0
        assert log.token_usage.total_tokens > 0


def test_hellaswag_instance_level():
    adapter = HELMAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_hellaswag'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter, 
            'tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0', 
            metadata_args
        )
        
        assert len(instance_logs) == 10
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'eleutherai/pythia-1b-v0'
        assert log.evaluation_name == 'hellaswag'
        assert log.interaction_type == InteractionType.single_turn
        
        assert len(log.input.choices) == 4
        
        assert log.output.raw == [' B']
        assert log.messages is None
        
        assert log.evaluation.score == 0.0
        assert log.evaluation.is_correct is False
        
        assert log.performance.generation_time_ms > 0


def test_narrativeqa_instance_level():
    adapter = HELMAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_narrativeqa'
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter, 
            'tests/data/helm/narrative_qa:model=openai_gpt2', 
            metadata_args
        )
        
        assert len(instance_logs) == 5
        log = instance_logs[0]
        
        assert log.schema_version == '0.2.0'
        assert log.model_id == 'openai/gpt2'
        assert log.evaluation_name == 'narrativeqa'
        assert log.interaction_type == InteractionType.single_turn
        
        assert log.input.reference == ['The school Mascot', 'the schools mascot']
        
        assert log.output.raw == [' Olive.']
        assert log.messages is None
        
        assert log.evaluation.score == 0.0
        assert log.evaluation.is_correct is False
        
        assert len(log.answer_attribution) == 1
        assert log.answer_attribution[0].extraction_method == 'exact_match'
