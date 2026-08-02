import json
import tempfile
from pathlib import Path

import pytest

from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
from every_eval_ever.converters.lm_eval.instance_level_adapter import (
    LMEvalInstanceLevelAdapter,
)
from every_eval_ever.converters.lm_eval.utils import (
    find_samples_file,
    parse_model_args,
)
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    ScoreType,
    SourceDataHf,
)
from every_eval_ever.helpers.io import SourceRecordsError

DATA_DIR = Path('tests/data/lm_eval')
RESULTS_FILE = DATA_DIR / 'results_2026-01-21T03-44-18.458309.json'
SAMPLES_FILE = (
    DATA_DIR / 'samples_math_perturbed_full_2026-01-21T03-44-18.458309.jsonl'
)


def _make_metadata_args(**overrides):
    args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    args.update(overrides)
    return args


# ── Utility tests ──────────────────────────────────────────────────────


def test_parse_model_args_basic():
    result = parse_model_args('pretrained=EleutherAI/pythia-160m,dtype=float16')
    assert result == {
        'pretrained': 'EleutherAI/pythia-160m',
        'dtype': 'float16',
    }


def test_parse_model_args_empty():
    assert parse_model_args('') == {}
    assert parse_model_args(None) == {}


def test_parse_model_args_complex():
    result = parse_model_args(
        'pretrained=RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1,trust_remote_code=True'
    )
    assert (
        result['pretrained']
        == 'RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1'
    )
    assert result['trust_remote_code'] == 'True'


def test_find_samples_file():
    found = find_samples_file(DATA_DIR, 'math_perturbed_full')
    assert found is not None
    assert found.name.startswith('samples_math_perturbed_full')


def test_find_samples_file_missing():
    assert find_samples_file(DATA_DIR, 'nonexistent_task') is None


def test_find_samples_file_does_not_cross_model_directories(tmp_path):
    nested = tmp_path / 'another-model'
    nested.mkdir()
    (nested / 'samples_shared_task_2026.jsonl').write_text(
        '{}\n', encoding='utf-8'
    )

    assert find_samples_file(tmp_path, 'shared_task') is None


# ── Adapter: transform_from_file ───────────────────────────────────────


def test_transform_from_file_returns_two_tasks():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert len(logs) == 2
    for log in logs:
        assert isinstance(log, EvaluationLog)


def test_transform_from_file_model_info():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    model = logs[0].model_info

    assert (
        model.name
        == 'RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1'
    )
    assert model.id == model.name
    assert model.developer == 'RylanSchaeffer'
    assert model.inference_engine.name == 'transformers'
    assert model.additional_details['num_parameters'] == '93069280'
    assert model.additional_details['dtype'] == 'torch.bfloat16'


def test_transform_from_file_source_metadata():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    src = logs[0].source_metadata

    assert src.source_name == 'lm-evaluation-harness'
    assert src.source_type.value == 'evaluation_run'
    assert src.source_organization_name == 'TestOrg'


def test_transform_from_file_source_data():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    # Both tasks should have HF source data
    for log in logs:
        assert isinstance(log.evaluation_results[0].source_data, SourceDataHf)

    perturbed = logs[0].evaluation_results[0].source_data
    assert perturbed.hf_repo == 'stellaathena/math_perturbed_5000'
    assert perturbed.hf_split == 'test'


def test_transform_from_file_evaluation_results():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    # First task: math_perturbed_full with exact_match = 0.0
    perturbed_results = logs[0].evaluation_results
    assert len(perturbed_results) == 1
    assert perturbed_results[0].score_details.score == 0.0
    assert (
        perturbed_results[0].metric_config.evaluation_description
        == 'exact_match'
    )
    assert perturbed_results[0].metric_config.lower_is_better is False
    assert perturbed_results[0].metric_config.min_score == 0.0
    assert perturbed_results[0].metric_config.max_score == 1.0

    # Second task: math_rephrased_full with exact_match = 0.0004
    rephrased_results = logs[1].evaluation_results
    assert rephrased_results[0].score_details.score == 0.0004


def test_transform_from_file_uncertainty():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    uncertainty = logs[1].evaluation_results[0].score_details.uncertainty
    assert uncertainty is not None
    assert uncertainty.standard_error.value == 0.0002828144211304471
    assert uncertainty.standard_error.method == 'bootstrap'
    assert uncertainty.num_samples == 5000


def test_transform_from_file_generation_config():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    gen = logs[0].evaluation_results[0].generation_config
    assert gen is not None
    assert gen.generation_args.temperature == 0.0
    assert gen.generation_args.max_tokens == 512
    assert gen.additional_details['num_fewshot'] == '0'


def test_transform_from_file_eval_timestamp():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].evaluation_timestamp == '1768964383'


# ── Adapter: transform_from_directory ──────────────────────────────────


def test_transform_from_directory():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_directory(DATA_DIR, _make_metadata_args())
    assert len(logs) == 2
    task_names = {
        r.evaluation_name for log in logs for r in log.evaluation_results
    }
    assert 'math_perturbed_full' in task_names
    assert 'math_rephrased_full' in task_names


# ── Adapter: group placeholder filtering ───────────────────────────────


def test_get_tasks_skips_group_placeholders():
    adapter = LMEvalAdapter()
    raw = {
        'results': {
            'group_task': {'alias': 'group_task', ' ': ''},
            'real_task': {'alias': 'real_task', 'acc,none': 0.5},
        }
    }
    tasks = adapter._get_tasks(raw)
    assert tasks == ['real_task']


# ── Adapter: inference engine override ─────────────────────────────────


def test_inference_engine_override():
    adapter = LMEvalAdapter()
    metadata = _make_metadata_args(
        inference_engine='vllm', inference_engine_version='0.6.0'
    )
    logs = adapter.transform_from_file(RESULTS_FILE, metadata)
    assert logs[0].model_info.inference_engine.name == 'vllm'
    assert logs[0].model_info.inference_engine.version == '0.6.0'


# ── Adapter: eval_metadata tracking ───────────────────────────────────


def test_eval_metadata_stored_after_transform():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    for log in logs:
        meta = adapter.get_eval_metadata(log.evaluation_id)
        assert 'task_name' in meta
        assert 'parent_dir' in meta


# ── Instance-level adapter ─────────────────────────────────────────────


def test_instance_level_transform_samples():
    inst_adapter = LMEvalInstanceLevelAdapter()
    logs = inst_adapter.transform_samples(
        SAMPLES_FILE,
        evaluation_id='test/eval/123',
        model_id='test-model',
        task_name='math_perturbed_full',
    )
    assert len(logs) == 10

    first = logs[0]
    assert first.sample_id == '0'
    assert first.evaluation_name == 'math_perturbed_full'
    assert first.model_id == 'test-model'
    assert first.input.reference == ['3']
    assert first.evaluation.score == 0.0
    assert first.evaluation.is_correct is False
    assert first.input.choices is None  # generation task, not MC
    assert first.sample_hash  # non-empty hash


def test_instance_level_transform_and_save():
    inst_adapter = LMEvalInstanceLevelAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = inst_adapter.transform_and_save(
            SAMPLES_FILE,
            evaluation_id='test/eval/123',
            model_id='test-model',
            task_name='math_perturbed_full',
            output_dir=tmpdir,
            file_uuid='123e4567-e89b-42d3-a456-426614174000',
            collection='test',
            developer='dev',
        )
        assert result is not None
        assert result.total_rows == 10
        assert result.format.value == 'jsonl'
        assert result.checksum  # non-empty sha256
        assert (Path(tmpdir) / Path(result.file_path).name).exists()
        assert result.file_path == (
            'data/test/dev/test-model/'
            '123e4567-e89b-42d3-a456-426614174000_samples.jsonl'
        )


def test_instance_level_transform_and_save_no_output_dir():
    inst_adapter = LMEvalInstanceLevelAdapter()
    result = inst_adapter.transform_and_save(
        SAMPLES_FILE,
        evaluation_id='test/eval/123',
        model_id='test-model',
        task_name='math_perturbed_full',
        output_dir=None,
    )
    assert result is None


def test_na_stderr_treated_as_absent():
    """lm-eval reports stderr as the string 'N/A' for non-bootstrapped metrics
    (aggregated/grouped or custom metrics, e.g. ECLeKTic). Conversion must not
    crash, and the StandardError (which requires a float) must be omitted rather
    than coerced to 0."""
    adapter = LMEvalAdapter()
    raw_data = {
        'results': {'mytask': {'acc,none': 0.5, 'acc_stderr,none': 'N/A'}},
        'n-samples': {'mytask': {'effective': 100}},
    }
    results = adapter._build_evaluation_results(raw_data, 'mytask')
    assert len(results) == 1
    uncertainty = results[0].score_details.uncertainty
    assert uncertainty is not None
    assert uncertainty.standard_error is None
    assert uncertainty.num_samples == 100


def test_unbounded_metric_uses_quoted_infinity():
    adapter = LMEvalAdapter()
    raw_data = {
        'results': {'mytask': {'word_perplexity,none': 2.0}},
    }

    [result] = adapter._build_evaluation_results(raw_data, 'mytask')

    assert result.metric_config.score_type == ScoreType.continuous
    assert result.metric_config.min_score == 1.0
    assert result.metric_config.max_score == float('inf')
    assert '"max_score":"Infinity"' in result.model_dump_json()


def test_unknown_metric_is_preserved_without_invented_bounds():
    adapter = LMEvalAdapter()
    raw_data = {
        'results': {'mytask': {'custom_metric,none': 2.0}},
    }

    [result] = adapter._build_evaluation_results(raw_data, 'mytask')

    assert result.score_details.score == 2.0
    assert result.metric_config.score_type is None
    assert result.metric_config.min_score is None
    assert result.metric_config.max_score is None
    assert result.metric_config.additional_details == {
        'bounds_status': 'unknown'
    }


def test_unknown_metric_count_is_recorded_on_the_log():
    adapter = LMEvalAdapter()
    raw_data = {
        'results': {
            'mytask': {
                'custom_metric,none': 2.0,
                'another_custom_metric,none': 3.0,
            }
        },
        'configs': {'mytask': {'task': 'mytask'}},
    }

    log = adapter._transform_single(
        raw_data,
        {
            'task_name': 'mytask',
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'first_party',
        },
    )

    assert log.source_metadata.additional_details == {
        'metrics_with_unknown_bounds': '2'
    }


def test_directory_conversion_retains_good_files_and_reports_bad_files(
    tmp_path, monkeypatch
):
    good_path = tmp_path / 'results_good.json'
    bad_path = tmp_path / 'results_bad.json'
    good_path.write_text('{}', encoding='utf-8')
    bad_path.write_text('{}', encoding='utf-8')
    adapter = LMEvalAdapter()
    good_log = object()

    def fake_transform(path, _metadata):
        if Path(path).name == 'results_bad.json':
            raise ValueError('broken lm-eval result')
        return [good_log]

    monkeypatch.setattr(adapter, 'transform_from_file', fake_transform)

    result = adapter.transform_from_directory_result(tmp_path, {})

    assert result.records == [good_log]
    assert result.total_records == 2
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == str(bad_path)
    assert result.failures[0].source_record == {'path': str(bad_path)}
    with pytest.raises(SourceRecordsError, match='broken lm-eval result'):
        result.raise_if_incomplete()


def test_directory_conversion_tracks_each_results_file_parent(tmp_path):
    adapter = LMEvalAdapter()
    source = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
    model_a = tmp_path / 'model-a'
    model_b = tmp_path / 'model-b'
    model_a.mkdir()
    model_b.mkdir()
    first = model_a / 'results_first.json'
    second = model_b / 'results_second.json'
    first.write_text(json.dumps(source), encoding='utf-8')
    second_source = json.loads(json.dumps(source))
    second_source['config']['model_args'] = 'pretrained=test/model-b'
    second.write_text(json.dumps(second_source), encoding='utf-8')

    result = adapter.transform_from_directory_result(
        tmp_path, {'parent_eval_output_dir': str(tmp_path)}
    )

    parents = {
        adapter.get_eval_metadata(log.evaluation_id)['parent_dir']
        for log in result.records
    }
    assert parents == {str(model_a), str(model_b)}
