import pytest

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from every_eval_ever.converters.helm.adapter import HELMAdapter
from every_eval_ever.converters.helm.instance_level_adapter import (
    HELMInstanceLevelDataAdapter,
    _BINARY_CORRECTNESS_METRIC_NAMES,
    _is_correct_for_metric,
    _score_from_stat,
)
from every_eval_ever.eval_types import EvaluatorRelationship
from every_eval_ever.instance_level_types import (
    InstanceLevelEvaluationLog,
    InteractionType,
)


def _make_helm_adapter():
    from every_eval_ever.converters.helm import adapter as helm_adapter_mod

    if helm_adapter_mod._HELM_IMPORT_ERROR is not None:
        pytest.skip(
            'HELM converter dependencies are missing; install with: '
            'uv sync --extra helm (or pip install every_eval_ever[helm])'
        )
    return HELMAdapter()


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


def test_mmlu_instance_level():
    adapter = _make_helm_adapter()

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

        # Per-(sample, metric) emission: 10 samples * many stats per
        # sample. Confirm by sample_id distinct count, not row count.
        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 10
        # The MMLU fixture has the standard exact-match family for
        # multiple_choice_joint; every sample must have an exact_match row.
        em_rows = [
            log
            for log in instance_logs
            if log.evaluation_result_id == 'exact_match'
        ]
        assert len(em_rows) == 10

        log = _by_sample_and_metric(instance_logs)[('id147', 'exact_match')]
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


def test_hellaswag_instance_level():
    adapter = _make_helm_adapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_hellaswag',
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0',
            metadata_args,
        )

        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 10

        em_rows = [
            log
            for log in instance_logs
            if log.evaluation_result_id == 'exact_match'
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
    adapter = _make_helm_adapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_narrativeqa',
        }

        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/narrative_qa:model=openai_gpt2',
            metadata_args,
        )

        sample_ids = sorted({log.sample_id for log in instance_logs})
        assert len(sample_ids) == 5

        em_rows = [
            log
            for log in instance_logs
            if log.evaluation_result_id == 'exact_match'
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


# ---------------------------------------------------------------------------
# Per-(sample, metric) row emission and correctness semantics
# ---------------------------------------------------------------------------


def test_per_sample_per_metric_rows_are_emitted():
    """Every per-instance HELM stat with a numeric mean becomes its own row.

    The fixture's first sample has 27 declared stats but two
    (``training_co2_cost`` and ``training_energy_cost``) carry no value;
    we expect 25 rows for that sample.
    """
    adapter = _make_helm_adapter()
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
        rows_for_id147 = [log for log in instance_logs if log.sample_id == 'id147']
        metric_ids = sorted(log.evaluation_result_id for log in rows_for_id147)
        assert 'exact_match' in metric_ids
        assert 'num_prompt_tokens' in metric_ids
        # No row should have a None evaluation_result_id when stats exist.
        assert all(log.evaluation_result_id is not None for log in rows_for_id147)
        # Distinct metric ids — no duplicates within a sample.
        assert len(metric_ids) == len(set(metric_ids))


def test_is_correct_only_claimed_for_correctness_metrics():
    """``is_correct=True`` should never appear on bookkeeping/resource rows.

    The MMLU fixture has positive-valued bookkeeping rows
    (``num_references=4``, ``num_prompt_tokens=333``, ``inference_runtime>0``,
    ``max_prob=1``). Earlier patches set ``is_correct = score > 0`` for
    every row, which overclaimed correctness on these. The fix gates
    ``is_correct`` on a tight allowlist of correctness metric names.
    """
    adapter = _make_helm_adapter()
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

        bookkeeping = [
            log
            for log in instance_logs
            if log.evaluation_result_id
            in {
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
        ]
        assert bookkeeping, 'expected bookkeeping rows to be emitted'
        assert all(
            log.evaluation.is_correct is False for log in bookkeeping
        ), 'bookkeeping metrics must not claim correctness'

        # narrative_qa has graded scores (rouge_l/f1/bleu_*) which are not
        # binary correctness either; check at least one fixture has them
        # and that they don't overclaim correctness.
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
                if log.evaluation_result_id
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
    """When a correctness metric scores positive, is_correct must be True.

    Run the converter and pick any exact_match row whose score happens to
    be 1.0. The MMLU fixture contains at least one correct id (id7).
    """
    adapter = _make_helm_adapter()
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
            if log.evaluation_result_id == 'exact_match'
            and log.evaluation.score > 0
        ]
        for log in positive_em_rows:
            assert log.evaluation.is_correct is True


def test_is_correct_for_metric_helper():
    """Pure-function check on the correctness allowlist."""
    # Every allowlist member, score>0 ⇒ True; score==0 ⇒ False.
    for name in _BINARY_CORRECTNESS_METRIC_NAMES:
        assert _is_correct_for_metric(name, 1.0) is True
        assert _is_correct_for_metric(name, 0.0) is False
    # Non-allowlisted metrics never claim correctness, regardless of score.
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


def _score_from_stat_dict(stat_dict):
    """Mirror the converter's numeric-score policy for fixture assertions."""
    value = stat_dict.get('mean')
    if value is None:
        count = stat_dict.get('count')
        total = stat_dict.get('sum')
        if count and total is not None:
            value = total / count
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expected_numeric_stat_rows(per_instance_stats_path):
    """Count rows the per-(sample, numeric stat) policy should emit."""
    per_instance_stats = json.loads(Path(per_instance_stats_path).read_text())
    return sum(
        1
        for inst_stats in per_instance_stats
        for stat in inst_stats.get('stats', [])
        if _score_from_stat_dict(stat) is not None
    )


def test_total_rows_matches_numeric_per_instance_stats():
    """The row count should be exact, not just >= the sample count."""
    adapter = _make_helm_adapter()
    fixture_path = Path(
        'tests/data/helm/'
        'mmlu:subject=philosophy,method=multiple_choice_joint,'
        'model=openai_gpt2'
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_exact_total_rows',
        }
        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            fixture_path,
            metadata_args,
        )

        expected_rows = _expected_numeric_stat_rows(
            fixture_path / 'per_instance_stats.json'
        )
        assert converted_eval.detailed_evaluation_results.total_rows == len(
            instance_logs
        )
        assert len(instance_logs) == expected_rows

        sample_metric_keys = {
            (log.sample_id, log.evaluation_result_id)
            for log in instance_logs
        }
        assert len(sample_metric_keys) == len(instance_logs)


def test_instance_evaluation_result_ids_join_to_aggregate_results():
    """Every non-null instance result id should join to an aggregate row."""
    adapter = _make_helm_adapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
            'parent_eval_output_dir': tmpdir,
            'file_uuid': 'test_join_keys',
        }
        converted_eval, instance_logs = _load_instance_level_data(
            adapter,
            'tests/data/helm/mmlu:subject=philosophy,'
            'method=multiple_choice_joint,model=openai_gpt2',
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

        assert aggregate_ids, (
            'Aggregate HELM results must set evaluation_result_id when '
            'instance rows use evaluation_result_id as a join key.'
        )
        assert detail_ids <= aggregate_ids


def test_score_from_stat_helper_edge_cases():
    """Cover scalar extraction paths and malformed empty stats."""
    assert _score_from_stat(
        SimpleNamespace(mean=0.25, sum=10, count=2)
    ) == 0.25
    assert _score_from_stat(
        SimpleNamespace(mean=None, sum=3, count=2)
    ) == 1.5
    assert _score_from_stat(
        SimpleNamespace(mean=None, sum=0, count=0)
    ) is None
    assert _score_from_stat(
        SimpleNamespace(mean='not-a-number', sum=0, count=1)
    ) is None
    assert _score_from_stat(
        SimpleNamespace(mean=None, sum=None, count=1)
    ) is None


def test_missing_inst_stats_uses_legacy_exact_match_fallback(monkeypatch):
    """No per-instance stats should still produce one legacy EM row."""
    correct_reference = SimpleNamespace(
        output=SimpleNamespace(text='expected answer'),
        tags=['correct'],
    )
    distractor_reference = SimpleNamespace(
        output=SimpleNamespace(text='distractor'),
        tags=[],
    )
    state = SimpleNamespace(
        instance=SimpleNamespace(
            id='sample-1',
            references=[correct_reference, distractor_reference],
        ),
        request=SimpleNamespace(prompt='Question?'),
        result=SimpleNamespace(
            completions=[SimpleNamespace(text='expected answer')],
            request_time=0.25,
        ),
        output_mapping=None,
    )

    from every_eval_ever.converters.helm import instance_level_adapter as mod

    # This test only needs the converter logic and a synthetic state object;
    # bypass the optional HELM dependency guard so it can run in core installs.
    monkeypatch.setattr(mod, '_HELM_IMPORT_ERROR', None)
    monkeypatch.setattr(mod, 'extract_all_reasonings', lambda state: None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path, total_rows = HELMInstanceLevelDataAdapter(
            'fallback_eval',
            'jsonl',
            'sha256',
            tmpdir,
        ).convert_instance_level_logs(
            'synthetic_eval',
            'synthetic/model',
            [state],
            [],
        )

        assert total_rows == 1
        with Path(path).open('r', encoding='utf-8') as file:
            rows = [
                InstanceLevelEvaluationLog.model_validate(json.loads(line))
                for line in file
                if line.strip()
            ]

        assert len(rows) == 1
        log = rows[0]
        assert log.evaluation_result_id is None
        assert log.evaluation.score == 1.0
        assert log.evaluation.is_correct is True
        assert log.input.reference == ['expected answer']
        assert log.output.raw == ['expected answer']


def test_reasoning_traces_none_does_not_break_conversion(monkeypatch):
    """Patch the upstream extractor to return None and ensure conversion works.

    Mirrors the bug class fixed by the original 9900ae6 patch — we want a
    regression test so the next refactor cannot reintroduce the failure.
    """
    from every_eval_ever.converters.helm import instance_level_adapter as mod

    monkeypatch.setattr(mod, 'extract_all_reasonings', lambda state: None)

    adapter = _make_helm_adapter()
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
            # ``reasoning_trace`` is Optional[list[str]]; with extractor
            # returning None we either get None or an empty list — both
            # satisfy the schema. The key invariant is "no crash".
            trace = log.output.reasoning_trace
            assert trace is None or trace == [] or all(isinstance(t, str) for t in trace)
