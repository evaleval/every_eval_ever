import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from every_eval_ever.converters.helm import adapter as helm_adapter_module
from every_eval_ever.converters.helm.adapter import HELMAdapter
from every_eval_ever.converters.helm.instance_level_adapter import (
    HELMInstanceLevelDataAdapter,
    _BINARY_CORRECTNESS_METRIC_NAMES,
    _evaluation_result_id,
    _is_correct_for_metric,
    _score_from_stat,
)
from every_eval_ever.eval_types import EvaluatorRelationship
from every_eval_ever.instance_level_types import (
    InstanceLevelEvaluationLog,
    InteractionType,
)


def _require_helm():
    import_error = getattr(helm_adapter_module, '_HELM_IMPORT_ERROR', None)
    if import_error is not None:
        pytest.skip(
            'HELM converter dependencies are missing: '
            f'{import_error!r}. Install with: uv sync --extra helm'
        )


def _load_instance_level_data(adapter, filepath, metadata_args):
    eval_dirpath = Path(filepath)
    converted_eval_list = adapter.transform_from_directory(
        eval_dirpath,
        output_path=str(
            Path(metadata_args['parent_eval_output_dir']) / 'helm_output'
        ),
        metadata_args=metadata_args,
    )

    converted_eval = converted_eval_list[0]

    instance_level_path = Path(
        converted_eval.detailed_evaluation_results.file_path
    )
    instance_logs = []
    with instance_level_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instance_logs.append(
                    InstanceLevelEvaluationLog.model_validate(data)
                )

    return converted_eval, instance_logs


def _by_sample_and_metric(
    instance_logs,
) -> dict[tuple[str, str | None], InstanceLevelEvaluationLog]:
    return {
        (log.sample_id, log.evaluation_result_id): log
        for log in instance_logs
    }


def _metric_name_from_result_id(result_id: str | None) -> str | None:
    if result_id is None:
        return None
    return result_id.split(':', 1)[0]


def _json_score_from_stat(stat: dict) -> float | None:
    value = stat.get('mean')
    if value is None:
        count = stat.get('count')
        total = stat.get('sum')
        if count:
            try:
                value = total / count
            except (TypeError, ValueError, ZeroDivisionError):
                return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expected_numeric_instance_stat_rows(filepath):
    per_instance_path = Path(filepath) / 'per_instance_stats.json'
    per_instance_stats = json.loads(per_instance_path.read_text())
    return sum(
        1
        for item in per_instance_stats
        for stat in item.get('stats', [])
        if _json_score_from_stat(stat) is not None
    )


def test_mmlu_instance_level():
    _require_helm()
    adapter = HELMAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_mmlu',
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )

        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 10
        em_rows = [
            log
            for log in instance_logs
            if _metric_name_from_result_id(log.evaluation_result_id)
            == 'exact_match'
        ]
        assert len(em_rows) == 10

        log = _by_sample_and_metric(instance_logs)[
            ('id147', 'exact_match:test')
        ]
        assert log.schema_version == '0.2.2'
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

        assert converted_eval.evaluation_results


def test_hellaswag_instance_level():
    _require_helm()
    adapter = HELMAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_hellaswag',
        }

        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0',
            metadata_args,
        )

        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 10

        em_rows = [
            log
            for log in instance_logs
            if _metric_name_from_result_id(log.evaluation_result_id)
            == 'exact_match'
        ]
        assert len(em_rows) == 10
        log = em_rows[0]

        assert log.schema_version == '0.2.2'
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
    _require_helm()
    adapter = HELMAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_narrativeqa',
        }

        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/narrative_qa:model=openai_gpt2',
            metadata_args,
        )

        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 5

        em_rows = [
            log
            for log in instance_logs
            if _metric_name_from_result_id(log.evaluation_result_id)
            == 'exact_match'
        ]
        assert len(em_rows) == 5
        log = em_rows[0]

        assert log.schema_version == '0.2.2'
        assert log.model_id == 'openai/gpt2'
        assert log.evaluation_name == 'narrativeqa'
        assert log.interaction_type == InteractionType.single_turn

        assert log.input.reference == [
            'The school Mascot',
            'the schools mascot',
        ]

        assert log.output.raw == [' Olive.']
        assert log.messages is None

        assert log.evaluation.score == 0.0
        assert log.evaluation.is_correct is False

        assert len(log.answer_attribution) == 1
        assert log.answer_attribution[0].extraction_method == 'exact_match'


def test_per_sample_per_metric_rows_are_emitted():
    _require_helm()
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_grain',
        }
        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )
        rows_for_id147 = [
            log for log in instance_logs if log.sample_id == 'id147'
        ]
        metric_ids = sorted(log.evaluation_result_id for log in rows_for_id147)
        assert 'exact_match:test' in metric_ids
        assert 'num_prompt_tokens:test' in metric_ids
        assert all(log.evaluation_result_id is not None for log in rows_for_id147)
        assert len(metric_ids) == len(set(metric_ids))


def test_is_correct_only_claimed_for_correctness_metrics():
    _require_helm()
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_correctness',
        }
        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )

        bookkeeping_names = {
            'num_references',
            'num_prompt_tokens',
            'num_completion_tokens',
            'num_output_tokens',
            'inference_runtime',
            'max_prob',
            'finish_reason_unknown',
            'logprob',
            'num_bytes',
            'batch_size',
        }
        bookkeeping = [
            log
            for log in instance_logs
            if _metric_name_from_result_id(log.evaluation_result_id)
            in bookkeeping_names
        ]
        assert bookkeeping, 'expected bookkeeping rows to be emitted'
        assert all(
            log.evaluation.is_correct is False for log in bookkeeping
        ), 'bookkeeping metrics must not claim correctness'

        with tempfile.TemporaryDirectory() as tmpdir2:
            metadata_args2 = dict(metadata_args)
            metadata_args2['parent_eval_output_dir'] = tmpdir2
            metadata_args2['file_uuid'] = 'test_correctness_graded'
            _, narr_logs = _load_instance_level_data(
                adapter,
                'tests/data/helm/narrative_qa:model=openai_gpt2',
                metadata_args2,
            )
            graded = [
                log
                for log in narr_logs
                if _metric_name_from_result_id(log.evaluation_result_id)
                in {'rouge_l', 'f1_score', 'bleu_1', 'bleu_4'}
            ]
            assert graded, 'expected graded score rows in narrative_qa fixture'
            assert all(
                log.evaluation.is_correct is False for log in graded
            ), (
                'graded metrics (rouge_l/f1_score/bleu_*) must not be '
                'treated as binary correctness'
            )


def test_is_correct_is_true_for_correct_exact_match_rows():
    _require_helm()
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_exact_match_true',
        }
        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )
        positive_em_rows = [
            log
            for log in instance_logs
            if _metric_name_from_result_id(log.evaluation_result_id)
            == 'exact_match'
            and log.evaluation.score > 0
        ]
        assert positive_em_rows
        for log in positive_em_rows:
            assert log.evaluation.is_correct is True


def test_is_correct_for_metric_helper():
    for name in _BINARY_CORRECTNESS_METRIC_NAMES:
        assert _is_correct_for_metric(name, 1.0) is True
        assert _is_correct_for_metric(name, 0.0) is False
    for name in (
        'num_prompt_tokens',
        'num_references',
        'inference_runtime',
        'rouge_l',
        'f1_score',
        'bleu_1',
        'logprob',
        None,
    ):
        assert _is_correct_for_metric(name, 1.0) is False
        assert _is_correct_for_metric(name, 0.0) is False


def test_score_from_stat_helper_edge_cases():
    assert _score_from_stat(SimpleNamespace(mean=0.25, sum=10, count=2)) == 0.25
    assert _score_from_stat(SimpleNamespace(mean=None, sum=3, count=2)) == 1.5
    assert _score_from_stat(SimpleNamespace(mean=None, sum=0, count=0)) is None
    assert _score_from_stat(SimpleNamespace(mean=None, sum=None, count=1)) is None
    assert _score_from_stat(SimpleNamespace(mean='bad', sum=1, count=1)) is None


def test_evaluation_result_id_helper_disambiguates_split_and_perturbation():
    assert _evaluation_result_id('exact_match') == 'exact_match'
    assert _evaluation_result_id('exact_match', 'test') == 'exact_match:test'
    assert (
        _evaluation_result_id(
            'exact_match',
            'test',
            SimpleNamespace(name='robustness'),
        )
        == 'exact_match:test:robustness'
    )


def test_total_rows_matches_numeric_per_instance_stats():
    _require_helm()
    fixture = 'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2'
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_exact_total_rows',
        }
        converted_eval, instance_logs = _load_instance_level_data(
            adapter, fixture, metadata_args
        )

        expected_rows = _expected_numeric_instance_stat_rows(fixture)
        assert converted_eval.detailed_evaluation_results.total_rows == expected_rows
        assert len(instance_logs) == expected_rows
        assert len({
            (log.sample_id, log.evaluation_result_id)
            for log in instance_logs
        }) == len(instance_logs)


def test_instance_evaluation_result_ids_join_to_aggregate_results():
    _require_helm()
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_join_keys',
        }
        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )

        aggregate_ids = {
            result.evaluation_result_id
            for result in converted_eval.evaluation_results
            if result.evaluation_result_id is not None
        }
        detail_ids = {
            log.evaluation_result_id
            for log in instance_logs
            if log.evaluation_result_id is not None
        }

        assert aggregate_ids
        assert detail_ids
        assert detail_ids <= aggregate_ids


def test_aggregate_evaluation_result_ids_are_unique_and_non_null():
    _require_helm()
    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_aggregate_ids',
        }
        converted_eval, _ = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )

        result_ids = [
            result.evaluation_result_id
            for result in converted_eval.evaluation_results
        ]
        assert all(result_ids)
        assert len(result_ids) == len(set(result_ids))
        assert 'exact_match:test' in result_ids
        assert 'num_prompt_tokens:test' in result_ids


def test_missing_inst_stats_uses_legacy_exact_match_fallback():
    _require_helm()
    completion = SimpleNamespace(text='answer')
    state = SimpleNamespace(
        instance=SimpleNamespace(
            id='sample0',
            references=[
                SimpleNamespace(
                    output=SimpleNamespace(text='answer'),
                    tags=['correct'],
                )
            ],
        ),
        result=SimpleNamespace(completions=[completion], request_time=0.1),
        request=SimpleNamespace(prompt='question'),
        output_mapping={},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = HELMInstanceLevelDataAdapter(
            'fallback_samples',
            'jsonl',
            'sha256',
            tmpdir,
        )
        path, count = adapter.convert_instance_level_logs(
            'tiny',
            'dev/model',
            [state],
            [],
        )

        assert count == 1
        data = json.loads(Path(path).read_text().strip())
        log = InstanceLevelEvaluationLog.model_validate(data)
        assert log.evaluation_result_id is None
        assert log.evaluation.score == 1.0
        assert log.evaluation.is_correct is True


def test_reasoning_traces_none_does_not_break_conversion(monkeypatch):
    _require_helm()
    from every_eval_ever.converters.helm import instance_level_adapter as mod

    monkeypatch.setattr(mod, 'extract_all_reasonings', lambda state: None)

    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_reasoning_none',
        }
        _, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
            metadata_args,
        )
        assert instance_logs
        for log in instance_logs:
            trace = log.output.reasoning_trace
            assert trace is None or trace == [] or all(
                isinstance(t, str) for t in trace
            )
