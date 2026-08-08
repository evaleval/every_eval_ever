import json
from pathlib import Path

import pytest

from every_eval_ever.converters.lighteval.adapter import LightevalAdapter
from every_eval_ever.converters.lighteval.utils import (
    find_metric_spec,
    flatten_model_config,
    is_derived_aggregate_key,
    parse_results_file_timestamp,
    split_task_key,
    stderr_method_for,
)
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    ScoreType,
    SourceDataHf,
)
from every_eval_ever.helpers.io import SourceRecordsError

DATA_DIR = Path('tests/data/lighteval')
RESULTS_FILE = (
    DATA_DIR
    / 'results/HuggingFaceTB/SmolLM2-1.7B-Instruct'
    / 'results_2026-01-21T03-44-18.458309.json'
)


def _make_metadata_args(**overrides):
    args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    args.update(overrides)
    return args


def _logs_by_task(logs):
    return {log.evaluation_results[0].evaluation_name: log for log in logs}


# ── Utility tests ──────────────────────────────────────────────────────


def test_split_task_key_keeps_subset_separator():
    assert split_task_key('mmlu:abstract_algebra|5') == (
        'mmlu:abstract_algebra',
        5,
    )
    assert split_task_key('gsm8k|0') == ('gsm8k', 0)


def test_split_task_key_without_fewshot_suffix():
    assert split_task_key('all') == ('all', None)
    assert split_task_key('weird|notanumber') == ('weird|notanumber', None)


def test_is_derived_aggregate_key():
    assert is_derived_aggregate_key('all') is True
    assert is_derived_aggregate_key('mmlu:_average|5') is True
    assert is_derived_aggregate_key('mmlu:abstract_algebra|5') is False
    assert is_derived_aggregate_key('gsm8k|0') is False


def test_parse_results_file_timestamp_restores_iso_colons():
    assert (
        parse_results_file_timestamp(
            Path('results_2026-01-21T03-44-18.458309.json')
        )
        == '2026-01-21T03:44:18.458309'
    )


def test_parse_results_file_timestamp_ignores_other_names():
    assert parse_results_file_timestamp(Path('results_latest.json')) is None


def test_stderr_method_mirrors_lighteval_choice():
    mean_spec = {'metric_name': 'acc', 'corpus_level_fn': 'mean'}
    other_spec = {'metric_name': 'mcc', 'corpus_level_fn': 'matthews_corrcoef'}
    assert stderr_method_for(mean_spec, 'acc') == 'analytic'
    assert stderr_method_for(other_spec, 'mcc') == 'bootstrap'
    assert stderr_method_for(None, 'acc') is None
    assert stderr_method_for({'metric_name': 'acc'}, 'acc') is None


def test_find_metric_spec_handles_grouped_metrics():
    task_config = {
        'metrics': [
            {
                'metric_name': ['bleu_1', 'bleu_4'],
                'corpus_level_fn': {'bleu_1': 'mean', 'bleu_4': 'corpus_bleu'},
            }
        ]
    }
    spec = find_metric_spec(task_config, 'bleu_4')
    assert spec is not None
    assert stderr_method_for(spec, 'bleu_4') == 'bootstrap'
    assert stderr_method_for(spec, 'bleu_1') == 'analytic'
    assert find_metric_spec(task_config, 'rouge1') is None


def test_flatten_model_config_drops_credentials():
    flattened, redacted = flatten_model_config(
        {
            'model_name': 'openai/gpt-4o',
            'provider': 'openai',
            'api_key': 'sk-NOT-A-REAL-KEY-FIXTURE-ONLY',
            'generation_parameters': {'temperature': 0.0},
            'timeout': None,
        }
    )
    assert redacted == ['api_key']
    assert 'api_key' not in flattened
    assert 'sk-NOT-A-REAL-KEY-FIXTURE-ONLY' not in json.dumps(flattened)
    assert flattened['provider'] == 'openai'
    assert flattened['generation_parameters'] == '{"temperature": 0.0}'
    # None means "not set by the run", which is not the same as a value.
    assert 'timeout' not in flattened


# ── Adapter: transform_from_file ───────────────────────────────────────


def test_transform_from_file_returns_only_measured_tasks():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert len(logs) == 3
    for log in logs:
        assert isinstance(log, EvaluationLog)
    assert set(_logs_by_task(logs)) == {
        'mmlu:abstract_algebra|5',
        'mmlu:anatomy|5',
        'glue:cola|0',
    }


def test_transform_from_file_model_info():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    model = logs[0].model_info

    assert model.name == 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    assert model.id == model.name
    assert model.developer == 'HuggingFaceTB'
    assert model.additional_details['dtype'] == 'bfloat16'
    assert model.additional_details['batch_size'] == '8'
    # lighteval dumps its model config with no backend discriminator.
    assert model.inference_engine is None


def test_transform_from_file_source_metadata():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    src = logs[0].source_metadata

    assert src.source_name == 'lighteval'
    assert src.source_type.value == 'evaluation_run'
    assert src.source_organization_name == 'TestOrg'
    assert (
        src.additional_details['total_evaluation_time_seconds']
        == '128.42371550000023'
    )


def test_transform_from_file_source_data():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    source = logs['mmlu:abstract_algebra|5'].evaluation_results[0].source_data
    assert isinstance(source, SourceDataHf)
    assert source.dataset_name == 'mmlu:abstract_algebra'
    assert source.hf_repo == 'lighteval/mmlu'
    assert source.hf_split == 'test'
    assert source.samples_number == 100
    assert source.additional_details['hf_subset'] == 'abstract_algebra'


def test_transform_from_file_evaluation_results():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    [result] = logs['mmlu:abstract_algebra|5'].evaluation_results
    assert result.score_details.score == 0.31
    assert result.metric_config.metric_name == 'acc'
    assert result.metric_config.lower_is_better is False
    assert result.metric_config.score_type == ScoreType.continuous
    assert result.metric_config.min_score == 0.0
    assert result.metric_config.max_score == 1.0


def test_transform_from_file_uncertainty():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    [result] = logs['mmlu:abstract_algebra|5'].evaluation_results
    uncertainty = result.score_details.uncertainty
    assert uncertainty is not None
    assert uncertainty.standard_error.value == 0.04648231987117316
    assert uncertainty.standard_error.method == 'analytic'
    assert uncertainty.num_samples == 100


def test_transform_from_file_generation_config():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    gen = (
        logs['mmlu:abstract_algebra|5'].evaluation_results[0].generation_config
    )
    assert gen is not None
    assert gen.generation_args.temperature == 0.0
    assert gen.generation_args.top_p == 0.95
    assert gen.generation_args.max_tokens == 256
    assert gen.additional_details['num_fewshots'] == '5'
    assert gen.additional_details['seed'] == '42'


def test_transform_from_file_evaluation_timestamp_comes_from_the_filename():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].evaluation_timestamp == '2026-01-21T03:44:18.458309'


def test_unknown_lighteval_sha_is_not_recorded():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].eval_library.name == 'lighteval'
    assert logs[0].eval_library.version == 'unknown'
    assert logs[0].eval_library.additional_details is None


# ── The two traps in lighteval's results mapping ───────────────────────


def test_stderr_is_never_emitted_as_its_own_metric():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    emitted = {
        result.metric_config.metric_name
        for log in logs
        for result in log.evaluation_results
    }
    assert not any(name.endswith('_stderr') for name in emitted)


def test_nan_stderr_is_omitted_rather_than_zeroed():
    """MetricsLogger writes NaN when the stderr estimate overflows. NaN means
    'no uncertainty reported', and 0.0 would claim a perfectly precise score."""
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    [result] = logs['mmlu:anatomy|5'].evaluation_results
    assert result.score_details.score == 0.4444444444444444
    uncertainty = result.score_details.uncertainty
    assert uncertainty is not None
    assert uncertainty.standard_error is None
    assert uncertainty.num_samples == 135


def test_derived_aggregate_rows_are_not_converted_but_are_recorded():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    names = {
        result.evaluation_name
        for log in logs
        for result in log.evaluation_results
    }
    assert 'all' not in names
    assert 'mmlu:_average|5' not in names
    assert (
        logs[0].source_metadata.additional_details[
            'lighteval_derived_rows_not_converted'
        ]
        == 'all,mmlu:_average|5'
    )


def test_non_finite_score_is_dropped_and_counted():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )

    log = logs['glue:cola|0']
    emitted = {
        result.metric_config.metric_name for result in log.evaluation_results
    }
    assert emitted == {'mcc', 'custom_reward'}
    assert (
        log.source_metadata.additional_details['metrics_dropped_non_finite']
        == '1'
    )


def test_bootstrap_stderr_and_unknown_bounds():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )
    results = {
        result.metric_config.metric_name: result
        for result in logs['glue:cola|0'].evaluation_results
    }

    mcc = results['mcc']
    assert mcc.metric_config.min_score == -1.0
    assert mcc.metric_config.max_score == 1.0
    assert mcc.score_details.uncertainty.standard_error.method == 'bootstrap'

    custom = results['custom_reward']
    assert custom.metric_config.score_type is None
    assert custom.metric_config.min_score is None
    assert custom.metric_config.additional_details == {
        'direction_status': 'assumed_higher_is_better',
        'bounds_status': 'unknown',
        'metric_id_source': 'namespaced_unresolved',
    }
    # Always set, because it is the cross-source join key — namespaced rather
    # than guessed when the name has no unambiguous canonical identity.
    assert custom.metric_config.metric_id == 'lighteval/custom_reward'
    assert mcc.metric_config.metric_id == 'matthews_correlation'
    assert (
        mcc.metric_config.additional_details['metric_id_source'] == 'canonical'
    )
    assert (
        logs['glue:cola|0'].source_metadata.additional_details[
            'metrics_with_unknown_bounds'
        ]
        == '1'
    )


def test_task_with_no_finite_scores_is_named_on_the_remaining_logs(tmp_path):
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    source['results']['glue:cola|0'] = {'mcc': float('nan')}
    partial = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    partial.write_text(json.dumps(source), encoding='utf-8')

    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(partial, _make_metadata_args())

    assert set(_logs_by_task(logs)) == {
        'mmlu:abstract_algebra|5',
        'mmlu:anatomy|5',
    }
    assert (
        logs[0].source_metadata.additional_details[
            'tasks_without_finite_scores'
        ]
        == 'glue:cola|0'
    )


def test_file_whose_measured_tasks_are_all_unconvertible_is_a_failure(tmp_path):
    """Total conversion loss must be distinguishable from nothing to convert.

    This file HAS a measured task; it just has no finite score. Previously the
    adapter returned an empty list and recorded no failure, so a directory made
    entirely of such files converted zero records and exited successfully —
    automation could not tell that everything had been dropped.
    """
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    source['results'] = {'glue:cola|0': {'mcc': float('nan')}}
    empty = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    empty.write_text(json.dumps(source), encoding='utf-8')

    adapter = LightevalAdapter()
    result = adapter.transform_from_directory_result(tmp_path, {})

    assert result.records == []
    assert len(result.failures) == 1
    assert 'glue:cola|0' in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_file_with_only_derived_rows_is_an_exclusion_not_a_failure(tmp_path):
    """Rows lighteval averaged itself are dropped on purpose, so they are
    excluded and accounted for rather than counted against the error budget."""
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    source['results'] = {'all': {'mcc': 0.5}, 'mmlu:_average|5': {'acc': 0.5}}
    derived_only = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    derived_only.write_text(json.dumps(source), encoding='utf-8')

    result = LightevalAdapter().transform_from_directory_result(tmp_path, {})

    assert result.records == []
    assert result.failures == []
    assert len(result.exclusions) == 1
    assert 'derived' in result.exclusions[0].reason
    result.raise_if_incomplete()


def test_total_records_counts_source_files_not_output_logs(tmp_path):
    """The coverage denominator needs one consistent unit.

    A good file yields several task-level logs while a bad file yields one
    failure, so summing the two gave a denominator that meant nothing.
    """
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    good = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    good.write_text(json.dumps(source), encoding='utf-8')

    result = LightevalAdapter().transform_from_directory_result(tmp_path, {})
    report = result.failure_report()

    assert report['total_source_records'] == 1
    assert report['converted_records'] == len(result.records)
    assert len(result.records) > 1, 'fixture should yield several task logs'


# ── Adapter: transform_from_directory ──────────────────────────────────


def test_transform_from_directory_finds_nested_results_files():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_directory(DATA_DIR, _make_metadata_args())
    assert len(logs) == 3


def test_transform_from_directory_without_results_files(tmp_path):
    adapter = LightevalAdapter()
    with pytest.raises(ValueError, match='No lighteval results_'):
        adapter.transform_from_directory_result(tmp_path, {})


def test_directory_conversion_retains_good_files_and_reports_bad_files(
    tmp_path, monkeypatch
):
    good_path = tmp_path / 'results_good.json'
    bad_path = tmp_path / 'results_bad.json'
    good_path.write_text('{}', encoding='utf-8')
    bad_path.write_text('{}', encoding='utf-8')
    adapter = LightevalAdapter()
    good_log = object()

    def fake_transform(path, _metadata):
        if Path(path).name == 'results_bad.json':
            raise ValueError('broken lighteval result')
        return [good_log]

    monkeypatch.setattr(adapter, 'transform_from_file', fake_transform)

    result = adapter.transform_from_directory_result(tmp_path, {})

    assert result.records == [good_log]
    assert result.total_records == 2
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == str(bad_path)
    with pytest.raises(SourceRecordsError, match='broken lighteval result'):
        result.raise_if_incomplete()


def test_eval_metadata_stored_after_transform():
    adapter = LightevalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    for log in logs:
        meta = adapter.get_eval_metadata(log.evaluation_id)
        assert meta['task_key']
        assert meta['parent_dir'] == str(RESULTS_FILE.parent)


# ── Metadata overrides ─────────────────────────────────────────────────


def test_inference_engine_override():
    adapter = LightevalAdapter()
    metadata = _make_metadata_args(
        inference_engine='vllm', inference_engine_version='0.6.0'
    )
    logs = adapter.transform_from_file(RESULTS_FILE, metadata)
    assert logs[0].model_info.inference_engine.name == 'vllm'
    assert logs[0].model_info.inference_engine.version == '0.6.0'


def test_provider_in_model_config_is_used_as_inference_platform():
    adapter = LightevalAdapter()
    raw_data = {
        'config_general': {
            'model_name': 'openai/gpt-4o',
            'model_config': {
                'model_name': 'openai/gpt-4o',
                'provider': 'openai',
            },
        }
    }
    model = adapter._extract_model_info(raw_data, _make_metadata_args())
    assert model.inference_platform == 'openai'


def test_missing_model_name_is_an_error():
    adapter = LightevalAdapter()
    with pytest.raises(ValueError, match='config_general.model_name'):
        adapter._extract_model_info({'config_general': {}}, {})


# ── Review findings: identity, credentials, coverage ───────────────────


def test_evaluation_id_is_identical_across_repeat_conversions():
    """Keyed on the source, so re-ingesting the same file cannot duplicate it.

    It previously carried the conversion time, which gave the same evaluation a
    new identity on every run.
    """
    adapter = LightevalAdapter()
    first = sorted(
        log.evaluation_id
        for log in adapter.transform_from_file(
            RESULTS_FILE, _make_metadata_args()
        )
    )
    second = sorted(
        log.evaluation_id
        for log in LightevalAdapter().transform_from_file(
            RESULTS_FILE, _make_metadata_args()
        )
    )

    assert first == second
    assert all(log_id for log_id in first)


def test_evaluation_id_separates_runs_that_differ_only_in_config(tmp_path):
    """Same task, same model, different settings must not collapse to one id."""
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    baseline = LightevalAdapter().transform_from_file(
        RESULTS_FILE, _make_metadata_args()
    )[0]

    source['config_general']['model_config']['temperature'] = 0.9
    variant_file = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    variant_file.write_text(json.dumps(source), encoding='utf-8')
    variant = LightevalAdapter().transform_from_file(
        variant_file, _make_metadata_args()
    )[0]

    assert baseline.evaluation_id != variant.evaluation_id


def test_nested_credentials_never_reach_additional_details(tmp_path):
    """The filter tested only top-level keys, so a nested env_vars token was
    serialized wholesale into a published field."""
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    source['config_general']['model_config'].update(
        {
            'api_key': 'sk-TOPLEVEL',
            'env_vars': {
                'OPENAI_API_KEY': 'sk-NESTED',
                'HF_TOKEN': 'hf-NESTED',
                'REGION': 'us-east-1',
            },
            'providers': [
                {'name': 'aws', 'aws_secret_access_key': 'AKIA-NESTED'}
            ],
        }
    )
    leaky = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    leaky.write_text(json.dumps(source), encoding='utf-8')

    log = LightevalAdapter().transform_from_file(leaky, _make_metadata_args())[
        0
    ]
    published = json.dumps(log.model_info.additional_details or {})

    for secret in ('sk-TOPLEVEL', 'sk-NESTED', 'hf-NESTED', 'AKIA-NESTED'):
        assert secret not in published, f'{secret} reached additional_details'
    assert 'us-east-1' in published, 'non-secret nested config should survive'
    redacted = (log.model_info.additional_details or {})[
        'redacted_model_config_keys'
    ]
    assert 'env_vars.OPENAI_API_KEY' in redacted


def test_ordinary_config_is_not_mistaken_for_a_credential(tmp_path):
    """`tokenizer` and `max_tokens` contain 'token' but are not secrets."""
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    source['config_general']['model_config'].update(
        {'tokenizer': 'gpt2', 'max_tokens': 512}
    )
    path = tmp_path / 'results_2026-01-21T03-44-18.458309.json'
    path.write_text(json.dumps(source), encoding='utf-8')

    log = LightevalAdapter().transform_from_file(path, _make_metadata_args())[0]
    published = json.dumps(log.model_info.additional_details or {})

    assert 'gpt2' in published
    assert '512' in published


def test_operator_can_declare_a_direction_the_run_omits():
    metadata = {
        **_make_metadata_args(),
        'metric_directions': {'custom_reward': False},
    }
    logs = _logs_by_task(
        LightevalAdapter().transform_from_file(RESULTS_FILE, metadata)
    )
    results = {
        result.metric_config.metric_name: result
        for result in logs['glue:cola|0'].evaluation_results
    }

    custom = results['custom_reward']
    assert custom.metric_config.lower_is_better is True
    assert (
        custom.metric_config.additional_details['direction_status']
        == 'operator_declared'
    )


def test_metrics_without_a_declared_direction_are_reported():
    adapter = LightevalAdapter()
    logs = _logs_by_task(
        adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    )
    meta = adapter.get_eval_metadata(logs['glue:cola|0'].evaluation_id)

    assert 'custom_reward' in meta['metrics_without_declared_direction']
