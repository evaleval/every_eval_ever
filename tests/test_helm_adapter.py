import tempfile
from pathlib import Path

import pytest

from every_eval_ever.converters.helm import adapter as helm_adapter_module
from every_eval_ever.converters.helm.adapter import HELMAdapter
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    SourceDataHf,
    SourceMetadata,
)

TEST_UUID = '123e4567-e89b-42d3-a456-426614174000'


pytestmark = pytest.mark.skipif(
    helm_adapter_module._HELM_IMPORT_ERROR is not None,
    reason=(
        'HELM converter dependencies are missing: '
        f'{helm_adapter_module._HELM_IMPORT_ERROR!r}. '
        'Install with: uv sync --extra helm'
    ),
)


def _load_eval(adapter, filepath, metadata_args):
    """Run the HELM aggregate adapter against one fixture directory."""
    eval_dirpath = Path(filepath)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = {
            **metadata_args,
            'file_uuid': TEST_UUID,
            'parent_eval_output_dir': tmpdir,
        }
        converted_eval = adapter.transform_from_directory(
            eval_dirpath,
            metadata_args=metadata_args,
        )

    converted_eval = converted_eval[0]
    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(
        converted_eval.evaluation_results[0].source_data, SourceDataHf
    )

    assert isinstance(converted_eval.source_metadata, SourceMetadata)
    assert converted_eval.source_metadata.source_name == 'HELM'
    assert converted_eval.source_metadata.source_type.value == 'evaluation_run'

    return converted_eval


def _assert_unique_evaluation_result_ids(converted_eval):
    """Aggregate result IDs must be stable join targets for sample rows."""
    result_ids = [
        result.evaluation_result_id
        for result in converted_eval.evaluation_results
    ]
    assert all(result_ids)
    assert len(result_ids) == len(set(result_ids))


def test_mmlu_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/helm/mmlu-subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
        metadata_args,
    )

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name == 'mmlu'
    )
    assert converted_eval.evaluation_results[0].source_data.hf_repo is None
    assert (
        len(converted_eval.evaluation_results[0].source_data.sample_ids) == 10
    )

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert any('mmlu' in r.evaluation_name.lower() for r in results)
    assert all(r.metric_config is not None for r in results)
    _assert_unique_evaluation_result_ids(converted_eval)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    # Per-(sample, metric) emission: each of the 10 samples produces one
    # row per non-empty stat, so total_rows is much larger than the
    # legacy "one row per sample" count.
    assert converted_eval.detailed_evaluation_results.total_rows >= 10


def test_hellswag_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/helm/commonsense-dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0',
        metadata_args,
    )

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name
        == 'hellaswag'
    )
    assert converted_eval.evaluation_results[0].source_data.hf_repo is None
    assert (
        len(converted_eval.evaluation_results[0].source_data.sample_ids) == 10
    )

    assert converted_eval.model_info.name == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.id == 'eleutherai/pythia-1b-v0'
    assert converted_eval.model_info.developer == 'eleutherai'
    assert converted_eval.model_info.inference_platform == 'huggingface'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].score_details.score is not None
    assert any('hellaswag' in r.evaluation_name.lower() for r in results)
    _assert_unique_evaluation_result_ids(converted_eval)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    # Per-(sample, core metric): >= sample count, not equal to it.
    assert converted_eval.detailed_evaluation_results.total_rows >= 10


def test_narrativeqa_eval():
    adapter = HELMAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(
        adapter, 'tests/data/helm/narrative_qa-model=openai_gpt2', metadata_args
    )

    assert converted_eval.evaluation_timestamp is not None
    assert converted_eval.retrieved_timestamp is not None

    assert (
        converted_eval.evaluation_results[0].source_data.dataset_name
        == 'narrativeqa'
    )
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
    _assert_unique_evaluation_result_ids(converted_eval)

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    # Per-(sample, core metric): >= sample count, not equal to it.
    assert converted_eval.detailed_evaluation_results.total_rows >= 5


HELLASWAG_RUN = (
    'tests/data/helm/commonsense-dataset=hellaswag,'
    'method=multiple_choice_joint,model=eleutherai_pythia-1b-v0'
)


def test_evaluation_name_is_the_benchmark_and_the_metric_is_named():
    """Each field carries one thing: the eval, the metric, the split.

    HELM reports one stat per (metric, split, perturbation), and those three
    already identify a result through `evaluation_result_id`. `evaluation_name` is
    the benchmark, which is what the instance-level rows carry and what a registry
    lookup can resolve.
    """
    adapter = HELMAdapter()
    converted_eval = _load_eval(
        adapter,
        HELLASWAG_RUN,
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )
    results = converted_eval.evaluation_results

    assert {result.evaluation_name for result in results} == {'hellaswag'}
    assert all(
        result.metric_config.metric_name
        and result.evaluation_result_id.startswith(
            result.metric_config.metric_name
        )
        for result in results
    )
    assert {
        result.score_details.details['perturbation'] for result in results
    } == {'', 'robustness', 'fairness'}


def _run_with_rewritten_stats(tmpdir, rewrite):
    """Copy the hellaswag run into `tmpdir` with `rewrite` applied to its stats."""
    import json
    import shutil

    destination = Path(tmpdir) / 'run'
    shutil.copytree(Path(HELLASWAG_RUN), destination)
    stats_path = destination / 'stats.json'
    stats_path.write_text(
        json.dumps(rewrite(json.loads(stats_path.read_text()))),
        encoding='utf-8',
    )
    return destination


def test_num_samples_counts_instances_not_train_trials():
    """The run has 10 instances, and HELM's own `count` on each stat is 1.

    HELM aggregates by averaging one value per train trial, so `count` is the
    number of trials. Reporting it as `num_samples` would tell a reader that
    every score in every single-trial HELM run rests on one example.
    """
    adapter = HELMAdapter()
    converted_eval = _load_eval(
        adapter,
        HELLASWAG_RUN,
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )
    results = converted_eval.evaluation_results

    assert results[0].source_data.samples_number == 10
    assert {
        result.score_details.uncertainty.num_samples for result in results
    } == {10}
    assert {
        result.score_details.details['num_train_trials'] for result in results
    } == {'1'}


def test_num_samples_follows_the_split_a_score_was_computed_on():
    """This run's 10 instances are 9 `test` and 1 `valid`, and HELM scores both.

    A run-wide count would overstate every split-specific score. The worst-case
    perturbation stats carry no per-instance stats of their own, and are computed
    over the instances of their split, so they take that split's count too.
    """
    adapter = HELMAdapter()
    converted_eval = _load_eval(
        adapter,
        'tests/data/helm/mmlu:subject=philosophy,'
        'method=multiple_choice_joint,model=openai_gpt2',
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )

    by_split = {}
    for result in converted_eval.evaluation_results:
        details = result.score_details.details
        by_split.setdefault(
            (details['split'], bool(details['perturbation'])), set()
        ).add(result.score_details.uncertainty.num_samples)

    assert by_split == {
        ('test', False): {9},
        ('test', True): {9},
        ('valid', False): {1},
        ('valid', True): {1},
    }


def test_helm_spread_is_not_the_schemas_standard_deviation():
    """`standard_deviation` is the spread of the per-sample scores; HELM's is the
    spread across train trials, which is 0.0 by construction on one trial.

    It is reported as itself, next to the trial count that gives it meaning.
    """
    import json

    stats = json.loads((Path(HELLASWAG_RUN) / 'stats.json').read_text())
    assert {stat['stddev'] for stat in stats if stat.get('count')} == {0.0}

    adapter = HELMAdapter()
    converted_eval = _load_eval(
        adapter,
        HELLASWAG_RUN,
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )

    assert all(
        result.score_details.uncertainty.standard_deviation is None
        for result in converted_eval.evaluation_results
    )
    assert {
        result.score_details.details['stddev_across_train_trials']
        for result in converted_eval.evaluation_results
    } == {'0.0'}


def test_multi_trial_spread_is_still_reported_as_a_trial_spread():
    """A run over several trials has a spread that is not 0.0, and it is still a
    spread across trials rather than across samples."""
    adapter = HELMAdapter()

    def three_trials(stats):
        for stat in stats:
            if stat.get('count'):
                stat['count'] = 3
                stat['stddev'] = 0.05
        return stats

    with tempfile.TemporaryDirectory() as tmpdir:
        converted_eval = _load_eval(
            adapter,
            _run_with_rewritten_stats(tmpdir, three_trials),
            {
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
            },
        )

    for result in converted_eval.evaluation_results:
        assert result.score_details.uncertainty.standard_deviation is None
        details = result.score_details.details
        assert details['num_train_trials'] == '3'
        assert details['stddev_across_train_trials'] == '0.05'
        # The samples behind the score are unaffected by how many trials it took.
        assert result.score_details.uncertainty.num_samples == 10


def test_sample_count_survives_a_run_without_per_instance_stats():
    """HELM reports `num_instances` per split, which is the only sample count left
    when a run carries no per-instance stats.

    Falling back to the run's own instance count instead would report all 10 of
    this run's instances for a score computed on the 9 of them in `test`.
    """
    import json
    import shutil

    adapter = HELMAdapter()
    source = Path(
        'tests/data/helm/mmlu:subject=philosophy,'
        'method=multiple_choice_joint,model=openai_gpt2'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = Path(tmpdir) / 'run'
        shutil.copytree(source, run_path)
        stats_path = run_path / 'per_instance_stats.json'
        assert stats_path.exists()
        stats_path.write_text(json.dumps([]), encoding='utf-8')

        converted_eval = _load_eval(
            adapter,
            run_path,
            {
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
            },
        )

    by_split = {}
    for result in converted_eval.evaluation_results:
        by_split.setdefault(result.score_details.details['split'], set()).add(
            result.score_details.uncertainty.num_samples
        )

    assert by_split == {'test': {9}, 'valid': {1}}


def test_metric_bounds_are_claimed_only_where_they_are_known():
    """`exact_match@5` is still exact match; `bleu_2` is a scale we have not checked.

    The `@k` suffix says how many completions were considered, so it must not cost
    a metric its bounds. A metric that is on the allowlist but in no bounds table
    gets none, and the count of them is published so a reader can see it.
    """
    adapter = HELMAdapter()

    def rename_first_exact_match(stats):
        for stat in stats:
            if stat['name']['name'] == 'exact_match' and not stat['name'].get(
                'perturbation'
            ):
                stat['name']['name'] = 'bleu_2'
                break
        return stats

    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = _run_with_rewritten_stats(tmpdir, rename_first_exact_match)
        converted_eval = _load_eval(
            adapter,
            run_path,
            {
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
            },
        )

    by_metric = {
        result.metric_config.metric_name: result.metric_config
        for result in converted_eval.evaluation_results
    }

    assert by_metric['exact_match@5'].min_score == 0.0
    assert by_metric['exact_match@5'].max_score == 1.0

    assert by_metric['bleu_2'].min_score is None
    assert by_metric['bleu_2'].max_score is None
    assert by_metric['bleu_2'].additional_details['bounds_status'] == 'unknown'
    assert converted_eval.source_metadata.additional_details == {
        'metrics_with_unknown_bounds': '1'
    }


def test_run_without_a_benchmark_metric_is_reported_not_raised():
    """One unconvertible run should not take the rest of the invocation with it.

    HELM emits bookkeeping stats (token counts, runtimes) alongside scores, and a
    run that has only those has nothing to publish. The converter has to say so
    per run, so a directory of runs still converts the ones that do have scores.
    """
    adapter = HELMAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_path = _run_with_rewritten_stats(
            tmpdir,
            lambda stats: [
                stat
                for stat in stats
                if stat['name']['name'].startswith('num_')
            ],
        )
        result = adapter.transform_from_directory_result(
            run_path,
            metadata_args={
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
            },
        )

    assert result.records == []
    assert len(result.failures) == 1
    reason = result.failures[0].reason
    assert 'no metric this converter recognizes' in reason
    assert 'num_prompt_tokens' in reason
    assert 'CORE_METRIC_PREFIXES' in reason


def test_instance_rows_join_the_aggregate_results_they_belong_to():
    """A sample row the aggregate cannot be joined to is a row nobody can read."""
    import json

    adapter = HELMAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_eval = adapter.transform_from_directory(
            Path(HELLASWAG_RUN),
            metadata_args={
                'source_organization_name': 'TestOrg',
                'evaluator_relationship': EvaluatorRelationship.first_party,
                'file_uuid': TEST_UUID,
                'parent_eval_output_dir': tmpdir,
            },
        )[0]
        sidecars = list(Path(tmpdir).rglob('*_samples.jsonl'))
        assert len(sidecars) == 1
        rows = [
            json.loads(line)
            for line in sidecars[0].read_text(encoding='utf-8').splitlines()
            if line
        ]

    assert rows
    assert {row['evaluation_name'] for row in rows} == {
        result.evaluation_name for result in converted_eval.evaluation_results
    }
    assert {row['evaluation_result_id'] for row in rows} <= {
        result.evaluation_result_id
        for result in converted_eval.evaluation_results
    }


def test_missing_model_deployment_falls_back_to_model():
    """
    Copies a helm data item and explicitly removes a field to test robustness
    to model_deployment missing. Regression test for #112
    """
    import json
    import shutil
    src = Path(
        'tests/data/helm/'
        'mmlu-subject=philosophy,method=multiple_choice_joint,model=openai_gpt2'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dst = tmpdir / src.name
        shutil.copytree(src, dst)

        run_spec_fpath = dst / 'run_spec.json'
        run_spec = json.loads(run_spec_fpath.read_text())
        run_spec['adapter_spec'].pop('model_deployment', None)
        run_spec_fpath.write_text(json.dumps(run_spec))

        adapter = HELMAdapter()
        metadata_args = {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        }

        converted_eval = _load_eval(adapter, dst, metadata_args)

    assert converted_eval.model_info.name == 'openai/gpt2'
    assert converted_eval.model_info.id == 'openai/gpt2'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'unknown'
