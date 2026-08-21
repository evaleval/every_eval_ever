import pytest

pytest.importorskip(
    'inspect_ai',
    reason='inspect-ai not installed; install with: uv sync --extra inspect',
)

import contextlib
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

from every_eval_ever.converters.common.error import AdapterError
from every_eval_ever.converters.common.metrics import (
    METRIC_ID_REGISTRY_REVISION,
)
from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
from every_eval_ever.converters.inspect.utils import (
    extract_model_info_from_model_path,
)
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    GenerationConfig,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceDataUrl,
    SourceMetadata,
)

TEST_UUID = '123e4567-e89b-42d3-a456-426614174000'
OTHER_TEST_UUID = '123e4567-e89b-42d3-a456-426614174001'


def _load_eval(adapter, filepath, metadata_args):
    eval_path = Path(filepath)
    metadata_args = dict(metadata_args)
    metadata_args.setdefault('file_uuid', TEST_UUID)

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args['parent_eval_output_dir'] = tmpdir
        converted_eval = adapter.transform_from_file(
            eval_path, metadata_args=metadata_args
        )

    assert isinstance(converted_eval, EvaluationLog)
    assert isinstance(
        converted_eval.evaluation_results[0].source_data,
        SourceDataHf | SourceDataUrl | SourceDataPrivate,
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


def _make_scorer(
    scorer_name: str,
    metrics: dict[str, object],
    scored_samples: int | None = None,
):
    return SimpleNamespace(
        name=scorer_name,
        scorer=scorer_name,
        params=None,
        metrics=metrics,
        scored_samples=scored_samples,
    )


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
    assert results[0].evaluation_name == 'inspect_evals/pubmedqa'
    assert results[0].evaluation_result_id == 'choice:accuracy'
    assert results[0].metric_config.metric_name == 'accuracy'
    assert results[0].score_details.score == 1.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows == 2


def test_transform_without_output_metadata_does_not_write_samples(tmp_path):
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
    assert converted_eval.source_metadata.source_organization_name == 'unknown'
    assert (
        converted_eval.source_metadata.evaluator_relationship
        == EvaluatorRelationship.third_party
    )
    assert converted_eval.detailed_evaluation_results is None


def test_transform_directory_requires_one_uuid_per_written_log():
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

        with pytest.raises(AdapterError, match='exactly one UUID'):
            adapter.transform_from_directory(
                tmp_logs_path,
                metadata_args={
                    'source_organization_name': 'TestOrg',
                    'evaluator_relationship': (
                        EvaluatorRelationship.first_party
                    ),
                    'parent_eval_output_dir': tmp_out_dir,
                },
            )


def test_transform_directory_uses_file_uuids_metadata_when_provided():
    adapter = InspectAIAdapter()
    fixture_dir = Path(__file__).resolve().parent / 'data/inspect'
    expected_uuids = [TEST_UUID, OTHER_TEST_UUID]

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
    assert results[0].evaluation_name == 'arc_easy'
    assert results[0].evaluation_result_id == 'choice:accuracy'
    assert results[0].metric_config.metric_name == 'accuracy'
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
    assert results[0].evaluation_name == 'arc_easy'
    assert results[0].evaluation_result_id == 'choice:accuracy'
    assert results[0].metric_config.metric_name == 'accuracy'
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

    # GAIA's `dataset.location` is `inspect_evals/gaia_dataset/GAIA` —
    # a 3-segment path, not a valid HF `owner/name`. The adapter emits
    # SourceDataPrivate with the task name as `dataset_name` and
    # preserves the harness-provided `dataset.name` and `dataset.location`
    # in additional_details.
    source_data = converted_eval.evaluation_results[0].source_data
    assert source_data.__class__.__name__ == 'SourceDataPrivate'
    assert source_data.dataset_name == 'gaia'
    assert source_data.additional_details['inspect_dataset_name'] == 'GAIA'
    assert (
        source_data.additional_details['inspect_dataset_location']
        == 'inspect_evals/gaia_dataset/GAIA'
    )
    assert int(source_data.additional_details['samples_number']) > 0
    assert source_data.additional_details['sample_ids']

    assert converted_eval.model_info.name == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.id == 'openai/gpt-4.1-mini-2025-04-14'
    assert converted_eval.model_info.developer == 'openai'
    assert converted_eval.model_info.inference_platform == 'openai'
    assert converted_eval.model_info.inference_engine is None

    results = converted_eval.evaluation_results
    assert len(results) > 0
    assert results[0].evaluation_name == 'gaia'
    assert results[0].evaluation_result_id == 'gaia_scorer:accuracy'
    assert results[0].metric_config.metric_name == 'accuracy'
    assert (
        results[0].metric_config.evaluation_description
        == 'accuracy from scorer gaia_scorer'
    )
    assert results[0].score_details.score >= 0.0

    assert converted_eval.detailed_evaluation_results is not None
    assert converted_eval.detailed_evaluation_results.format is not None
    assert converted_eval.detailed_evaluation_results.total_rows > 0


def test_evaluation_name_is_the_benchmark_and_the_metric_is_named():
    """One scorer reporting three metrics: same eval, two named scores.

    The third is the scorer's `std`, which the schema carries as uncertainty on
    the scores it describes.
    """
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        metadata_args,
    )

    results = converted_eval.evaluation_results
    assert len(results) == 2
    assert {result.evaluation_name for result in results} == {
        'inspect_evals/cyse2_vulnerability_exploit'
    }
    assert {result.metric_config.metric_name for result in results} == {
        'accuracy',
        'mean',
    }
    for result in results:
        assert result.evaluation_result_id == (
            f'vul_exploit_scorer:{result.metric_config.metric_name}'
        )
        # The scorer's `std` metric, carried where the schema puts dispersion.
        assert result.score_details.uncertainty.standard_deviation == (
            0.3115628730565127
        )


def test_metric_bounds_are_claimed_only_where_they_are_known():
    """`accuracy` is a proportion; `mean` is a mean of whatever the scorer returns."""
    adapter = InspectAIAdapter()
    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )
    by_metric = {
        result.metric_config.metric_name: result.metric_config
        for result in converted_eval.evaluation_results
    }

    assert by_metric['accuracy'].min_score == 0.0
    assert by_metric['accuracy'].max_score == 1.0
    assert by_metric['accuracy'].score_type == ScoreType.continuous

    assert by_metric['mean'].min_score is None
    assert by_metric['mean'].max_score is None
    assert by_metric['mean'].additional_details['bounds_status'] == 'unknown'


def test_num_samples_comes_from_the_results_header():
    """The scores cover every completed sample, not only the samples in the log.

    A log read without its samples carries none, so counting them would report
    a handful of samples for a run over thousands.
    """
    adapter = InspectAIAdapter()
    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )

    for result in converted_eval.evaluation_results:
        assert result.score_details.uncertainty.num_samples == 2340


def test_standard_error_of_zero_is_reported():
    """A task every sample scores identically has a stderr of 0.0, not of none."""
    adapter = InspectAIAdapter()
    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
        {
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': EvaluatorRelationship.first_party,
        },
    )

    uncertainty = converted_eval.evaluation_results[0].score_details.uncertainty
    assert uncertainty.standard_error is not None
    assert uncertainty.standard_error.value == 0.0


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
    source_data = SourceDataHf(
        dataset_name='synthetic_ds', source_type='hf_dataset'
    )
    generation_config = GenerationConfig()
    scores = [
        _make_scorer(
            'choice',
            {
                'accuracy': _make_metric('accuracy', 0.75),
                'f1': _make_metric('f1', 0.80),
                'stderr': _make_metric('stderr', 0.05),
            },
        )
    ]

    results, result_ids_by_scorer = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=scores,
        source_data=source_data,
        generation_config=generation_config,
        num_samples=10,
        timestamp='1234567890',
    )

    assert len(results) == 2
    # The eval is named once; the metric is a metric field, not part of the name.
    assert {result.evaluation_name for result in results} == {'synthetic/task'}
    assert {result.metric_config.metric_name for result in results} == {
        'accuracy',
        'f1',
    }
    assert {result.evaluation_result_id for result in results} == {
        'choice:accuracy',
        'choice:f1',
    }
    assert result_ids_by_scorer == {'choice': ['choice:accuracy', 'choice:f1']}


def test_each_scorer_counts_the_samples_it_could_score():
    """Inspect computes a scorer's metrics over the samples that scorer scored, and
    says how many those were; the run-wide count includes the ones it skipped."""
    adapter = InspectAIAdapter()
    scores = [
        _make_scorer(
            'strict',
            {'accuracy': _make_metric('accuracy', 0.5)},
            scored_samples=7,
        ),
        _make_scorer('lenient', {'accuracy': _make_metric('accuracy', 0.9)}),
    ]

    results, _ = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=scores,
        source_data=SourceDataHf(
            dataset_name='synthetic_ds', source_type='hf_dataset'
        ),
        generation_config=GenerationConfig(),
        num_samples=10,
        timestamp='1234567890',
    )

    by_id = {
        result.evaluation_result_id: result.score_details.uncertainty
        for result in results
    }
    assert by_id['strict:accuracy'].num_samples == 7
    # No count of its own, so the run's stands in.
    assert by_id['lenient:accuracy'].num_samples == 10


def test_standard_error_says_how_inspect_computed_it():
    """`stderr` is the analytic standard error of the mean and `bootstrap_stderr`
    resamples, so neither should be published without saying which it was."""
    adapter = InspectAIAdapter()
    scores = [
        _make_scorer(
            'analytic',
            {
                'accuracy': _make_metric('accuracy', 0.5),
                'stderr': _make_metric('stderr', 0.05),
            },
        ),
        _make_scorer(
            'resampled',
            {
                'accuracy': _make_metric('accuracy', 0.5),
                'bootstrap_stderr': _make_metric('bootstrap_stderr', 0.06),
            },
        ),
    ]

    results, _ = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=scores,
        source_data=SourceDataHf(
            dataset_name='synthetic_ds', source_type='hf_dataset'
        ),
        generation_config=GenerationConfig(),
        num_samples=10,
        timestamp='1234567890',
    )

    # Neither dispersion metric is a score of its own.
    assert {result.evaluation_result_id for result in results} == {
        'analytic:accuracy',
        'resampled:accuracy',
    }
    by_id = {
        result.evaluation_result_id: result.score_details.uncertainty
        for result in results
    }
    assert by_id['analytic:accuracy'].standard_error.method == 'analytic'
    assert by_id['resampled:accuracy'].standard_error.value == 0.06
    assert by_id['resampled:accuracy'].standard_error.method == 'bootstrap'


def test_both_stderrs_prefer_analytic_and_keep_the_bootstrap_value():
    """A scorer that reports both the analytic stderr and a bootstrap resample of
    it should publish the analytic one as the standard error and keep the bootstrap
    value in the score details, not let dict order pick one and drop the other."""
    adapter = InspectAIAdapter()
    results, _ = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=[
            _make_scorer(
                'both',
                {
                    'accuracy': _make_metric('accuracy', 0.5),
                    # bootstrap first, so the old order-dependent pick took it.
                    'bootstrap_stderr': _make_metric('bootstrap_stderr', 0.06),
                    'stderr': _make_metric('stderr', 0.05),
                },
            )
        ],
        source_data=SourceDataHf(
            dataset_name='synthetic_ds', source_type='hf_dataset'
        ),
        generation_config=GenerationConfig(),
        num_samples=10,
        timestamp='1234567890',
    )

    [result] = results
    assert result.metric_config.metric_name == 'accuracy'
    standard_error = result.score_details.uncertainty.standard_error
    assert (standard_error.value, standard_error.method) == (0.05, 'analytic')
    # The bootstrap value is preserved, not silently dropped.
    assert result.score_details.details['bootstrap_stderr'] == '0.06'


def test_a_scorer_reporting_only_dispersion_does_not_repeat_it():
    """`std` stays a score when it is all the scorer reported, so that the run is
    not left with none; carrying it as its own uncertainty would say it twice."""
    adapter = InspectAIAdapter()

    results, _ = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=[_make_scorer('spread', {'std': _make_metric('std', 0.3)})],
        source_data=SourceDataHf(
            dataset_name='synthetic_ds', source_type='hf_dataset'
        ),
        generation_config=GenerationConfig(),
        num_samples=10,
        timestamp='1234567890',
    )

    [result] = results
    assert result.metric_config.metric_name == 'std'
    assert result.score_details.score == 0.3
    assert result.score_details.uncertainty.standard_deviation is None
    assert result.metric_config.additional_details['polarity'] == (
        'not_applicable'
    )


def test_extract_evaluation_results_two_scorers_two_metrics_each():
    adapter = InspectAIAdapter()
    source_data = SourceDataHf(
        dataset_name='synthetic_ds', source_type='hf_dataset'
    )
    generation_config = GenerationConfig()
    scores = [
        _make_scorer(
            'scorer_a',
            {
                'accuracy': _make_metric('accuracy', 0.91),
                'f1': _make_metric('f1', 0.90),
            },
        ),
        _make_scorer(
            'scorer_b',
            {
                'accuracy': _make_metric('accuracy', 0.88),
                'f1': _make_metric('f1', 0.87),
            },
        ),
    ]

    results, result_ids_by_scorer = adapter._extract_evaluation_results(
        evaluation_task_name='synthetic/task',
        scores=scores,
        source_data=source_data,
        generation_config=generation_config,
        num_samples=10,
        timestamp='1234567890',
    )

    assert len(results) == 4
    assert {result.evaluation_name for result in results} == {'synthetic/task'}
    # Two scorers reporting the same metric name must not collide in
    # `evaluation_result_id`, or sample rows cannot say which one they join.
    assert {result.evaluation_result_id for result in results} == {
        'scorer_a:accuracy',
        'scorer_a:f1',
        'scorer_b:accuracy',
        'scorer_b:f1',
    }
    assert result_ids_by_scorer == {
        'scorer_a': ['scorer_a:accuracy', 'scorer_a:f1'],
        'scorer_b': ['scorer_b:accuracy', 'scorer_b:f1'],
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
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'model_info': {
                'additional_details': {
                    'num_parameters': 42,
                    'is_test_model': True,
                }
            },
            'source_data': {
                'additional_details': {
                    'shuffled': 'should_not_overwrite',
                    'subset': {'name': 'full'},
                }
            },
            'generation_config': {
                'additional_details': {
                    'runner': 'inspect',
                },
            },
            'agentic_eval_config': {
                'additional_details': {
                    'agent_mode': 'tool_use',
                }
            },
            'evaluation_results': [
                {
                    'evaluation_result_id': 'choice:accuracy',
                    'score_details': {
                        'details': {
                            'notes': ['a', 'b'],
                        }
                    },
                    'metric_config': {
                        'lower_is_better': True,
                        'evaluation_description': 'should_not_overwrite',
                        'score_type': ScoreType.continuous,
                        'min_score': 0.0,
                        'max_score': 1.0,
                        'additional_details': {
                            'normalization': 'none',
                        },
                    },
                },
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]

    assert converted_eval.model_info.additional_details == {
        'num_parameters': '42',
        'is_test_model': 'true',
        'deployment_type': 'unknown',
        'model_availability': 'unknown',
    }
    assert result.source_data.additional_details['shuffled'] == 'False'
    assert result.source_data.additional_details['subset'] == '{"name": "full"}'

    assert result.generation_config is not None
    assert result.generation_config.additional_details == {'runner': 'inspect'}
    assert result.generation_config.generation_args is not None
    assert (
        result.generation_config.generation_args.agentic_eval_config is not None
    )
    assert (
        result.generation_config.generation_args.agentic_eval_config.additional_details
        == {'agent_mode': 'tool_use'}
    )

    assert result.score_details.details == {'notes': '["a", "b"]'}

    # Converter-synthetic defaults are override-eligible.
    assert result.metric_config.lower_is_better is True
    assert result.metric_config.evaluation_description == 'should_not_overwrite'
    # Exhaustive: supplied details merge with what the converter resolved rather
    # than replacing it, and nothing else leaks in.
    assert result.metric_config.additional_details == {
        'normalization': 'none',
        'metric_id_registry_revision': METRIC_ID_REGISTRY_REVISION,
    }


def test_supplemental_eval_details_can_correct_a_resolved_metric_identity():
    """The identity a converter resolves from a table is a caller-overrideable value.

    `metric_id`, `metric_kind`, `metric_unit` and `metric_parameters` are listed
    as synthetic so a caller who can see a wrong value may replace it. That path
    is only real if the supplement model accepts those fields; a strict model
    that forbids them rejects the supplement before the allowlist runs, leaving
    the listing dead. Overriding them here has to reach the output.
    """
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'evaluation_result_id': 'choice:accuracy',
                    'metric_config': {
                        'metric_id': 'accuracy-corrected',
                        'metric_kind': 'classification',
                        'metric_unit': 'percent',
                        'metric_parameters': {'strategy': 'greedy'},
                    },
                },
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
        metadata_args,
    )
    result = next(
        r
        for r in converted_eval.evaluation_results
        if r.evaluation_result_id == 'choice:accuracy'
    )

    # Each override differs from what the converter resolves for `accuracy`
    # (`accuracy` / `accuracy` / `proportion` / no params), so a value that
    # survived is one the override replaced, not one that happened to match.
    assert result.metric_config.metric_id == 'accuracy-corrected'
    assert result.metric_config.metric_kind == 'classification'
    assert result.metric_config.metric_unit == 'percent'
    assert result.metric_config.metric_parameters == {'strategy': 'greedy'}


def test_supplemental_eval_details_applies_top_level_score_details():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'evaluation_result_id': 'choice:accuracy',
                    'score_details': {
                        'details': {
                            'matched': 1,
                        },
                    },
                }
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]

    assert result.score_details.details == {'matched': '1'}


def test_supplemental_eval_details_does_not_overwrite_existing_generation_details():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'generation_config': {
                'additional_details': {
                    'temperature': '999',
                    'added_field': 'yes',
                }
            },
        },
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/2026-02-07T11-26-57+00-00_gaia_4V8zHbbRKpU5Yv2BMoBcjE.json',
        metadata_args,
    )
    result = converted_eval.evaluation_results[0]
    assert result.generation_config is not None
    assert result.generation_config.additional_details is not None
    # existing log value remains
    assert result.generation_config.additional_details['temperature'] == '0.5'
    # missing key gets filled
    assert result.generation_config.additional_details['added_field'] == 'yes'


def test_supplemental_eval_details_does_not_apply_when_evaluation_name_does_not_match(
    caplog,
):
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'evaluation_name': 'some_other_eval - choice',
                    'score_details': {'details': {'matched': 1}},
                }
            ],
        },
    }

    with caplog.at_level(logging.WARNING):
        converted_eval = _load_eval(
            adapter,
            'tests/data/inspect/data_pubmedqa_gpt4o_mini.json',
            metadata_args,
        )
    result = converted_eval.evaluation_results[0]
    assert result.score_details.details is None
    # A supplemental file is hand-written, so a key that selects nothing is a
    # typo the contributor needs to hear about, not a silent no-op.
    assert 'matched no evaluation result' in caplog.text
    assert "'some_other_eval - choice'" in caplog.text


def test_supplemental_eval_details_matches_all_results_of_an_evaluation():
    """`evaluation_name` now names the eval, so it selects every result."""
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'evaluation_name': 'inspect_evals/cyse2_vulnerability_exploit',
                    'score_details': {'details': {'reviewed': 'yes'}},
                },
                {
                    'evaluation_result_id': 'vul_exploit_scorer:mean',
                    'score_details': {'details': {'reviewed': 'separately'}},
                },
            ],
        },
    }

    converted_eval = _load_eval(
        adapter,
        'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        metadata_args,
    )

    details_by_result_id = {
        result.evaluation_result_id: result.score_details.details
        for result in converted_eval.evaluation_results
    }
    assert details_by_result_id['vul_exploit_scorer:accuracy'] == {
        'reviewed': 'yes'
    }
    # The specific key wins over the evaluation-wide one.
    assert details_by_result_id['vul_exploit_scorer:mean'] == {
        'reviewed': 'separately'
    }


def test_supplemental_eval_details_fails_on_deprecated_per_result_schema():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'per_result': [
                {
                    'match': {
                        'evaluation_result_id': 'choice:accuracy',
                    },
                    'score_details': {'details': {'matched': 1}},
                },
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args['file_uuid'] = TEST_UUID
        metadata_args['parent_eval_output_dir'] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path('tests/data/inspect/data_pubmedqa_gpt4o_mini.json'),
                metadata_args=metadata_args,
            )


@pytest.mark.parametrize(
    'key_field, key',
    [
        ('evaluation_result_id', 'choice:accuracy'),
        ('evaluation_name', 'inspect_evals/pubmedqa'),
    ],
)
def test_supplemental_eval_details_fails_on_duplicate_key(key_field, key):
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {key_field: key, 'score_details': {'details': {'a': 1}}},
                {key_field: key, 'score_details': {'details': {'b': 2}}},
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args['file_uuid'] = TEST_UUID
        metadata_args['parent_eval_output_dir'] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path('tests/data/inspect/data_pubmedqa_gpt4o_mini.json'),
                metadata_args=metadata_args,
            )


def test_supplemental_eval_details_fails_when_one_entry_sets_both_selectors():
    """An id selects one result; a name selects every result of the evaluation.

    An entry that sets both applies to its own result by id and, through the
    shared name, to that result's siblings as well -- never what one entry is
    meant to do. The two behaviours are still available through two entries, so
    the ambiguous single entry is rejected rather than silently fanned out.
    """
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'evaluation_result_id': 'choice:accuracy',
                    'evaluation_name': 'inspect_evals/pubmedqa',
                    'score_details': {'details': {'a': 1}},
                },
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args['file_uuid'] = TEST_UUID
        metadata_args['parent_eval_output_dir'] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path('tests/data/inspect/data_pubmedqa_gpt4o_mini.json'),
                metadata_args=metadata_args,
            )


def test_supplemental_eval_details_fails_on_invalid_schema():
    adapter = InspectAIAdapter()
    metadata_args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
        'supplemental_eval_details': {
            'evaluation_results': [
                {
                    'metric_config': {
                        'unsupported_field': 'x',
                    }
                }
            ]
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_args = dict(metadata_args)
        metadata_args['file_uuid'] = TEST_UUID
        metadata_args['parent_eval_output_dir'] = tmpdir
        with pytest.raises(AdapterError):
            adapter.transform_from_file(
                Path('tests/data/inspect/data_pubmedqa_gpt4o_mini.json'),
                metadata_args=metadata_args,
            )
