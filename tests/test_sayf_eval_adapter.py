import json
import uuid
from pathlib import Path

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.converters.sayf_eval.adapter import SayfEvalAdapter
from every_eval_ever.eval_types import (
    EvaluationLog,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceDataUrl,
    SourceType,
)

DATA_DIR = Path('tests/data/sayf_eval')
FIXTURE = DATA_DIR / 'results_fixture.json'


def _meta(**overrides):
    args = {
        'source_organization_name': 'QCRI',
        'evaluator_relationship': 'third_party',
        'eval_library_name': 'sayf-eval',
        'collection_prefix': 'sayf-eval-',
    }
    args.update(overrides)
    return args


def _logs_by_task():
    logs = SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    return {log.evaluation_id.split('/')[0]: log for log in logs}


# ── structure ──────────────────────────────────────────────────────────


def test_one_log_per_task():
    logs = SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    tasks = sorted(log.evaluation_id.split('/')[0] for log in logs)
    assert tasks == ['ate', 'athena_vsp', 'cissp', 'mcq']


def test_every_log_validates_and_is_evaluation_run():
    for log in _logs_by_task().values():
        EvaluationLog.model_validate(log.model_dump())
        assert log.source_metadata.source_type == SourceType.evaluation_run
        assert log.source_metadata.source_organization_name == 'QCRI'
        assert log.eval_library.name == 'sayf-eval'
        # sayf-eval embeds its own version in the record; the adapter uses it.
        assert log.eval_library.version


def test_accuracy_result_is_higher_is_better_bounded_unit_interval():
    acc = _logs_by_task()['mcq'].evaluation_results[0]
    assert acc.metric_config.metric_kind == 'accuracy'
    assert acc.metric_config.lower_is_better is False
    assert acc.metric_config.score_type == ScoreType.continuous
    assert (
        acc.metric_config.min_score == 0.0
        and acc.metric_config.max_score == 1.0
    )
    # denominator counts are preserved in score_details.details
    assert acc.score_details.details['total'] == '3'


def test_vsp_has_mad_lower_is_better():
    vsp = _logs_by_task()['athena_vsp']
    kinds = {r.metric_config.metric_kind: r for r in vsp.evaluation_results}
    assert set(kinds) == {'accuracy', 'mae'}
    mad = kinds['mae']
    assert mad.metric_config.lower_is_better is True
    assert mad.metric_config.max_score == 10.0
    assert mad.score_details.score == 1.3


def test_ate_has_micro_f1():
    ate = _logs_by_task()['ate']
    kinds = {r.metric_config.metric_kind for r in ate.evaluation_results}
    assert kinds == {'accuracy', 'f1'}


# ── judge (llm_scoring) ──────────────────────────────────────────────────


def test_judge_recorded_as_llm_scoring():
    acc = _logs_by_task()['mcq'].evaluation_results[0]
    scoring = acc.metric_config.llm_scoring
    assert scoring is not None
    assert scoring.judges[0].model_info.id.startswith('anthropic/')
    assert scoring.input_prompt  # a description of the judge, not item text


# ── source_data provenance ───────────────────────────────────────────────


def test_source_data_variants_and_collection_slug():
    logs = _logs_by_task()
    hf = logs['mcq'].evaluation_results[0].source_data
    assert isinstance(hf, SourceDataHf)
    assert hf.hf_repo.startswith('RISys-Lab/')
    assert (
        hf.dataset_name == 'sayf-eval-mcq'
    )  # prefixed slug -> datastore collection
    assert hf.additional_details['subset'] == 'cti-mcq'

    url = logs['athena_vsp'].evaluation_results[0].source_data
    assert isinstance(url, SourceDataUrl)
    assert url.url and url.url[0].endswith('athena-cti-vsp.jsonl')

    other = logs['cissp'].evaluation_results[0].source_data
    assert isinstance(other, SourceDataPrivate)
    assert other.source_type == 'other'


# ── local vLLM model mapping ──────────────────────────────────────────────


def test_local_vllm_model_info():
    record = json.loads(FIXTURE.read_text())
    record['model'] = {
        'name': 'hosted_vllm/Qwen/Qwen3-8B',
        'provider': 'hosted_vllm',
        'base_url': 'http://localhost:8000/v1',
    }
    log = SayfEvalAdapter()._transform_single(
        record, {**_meta(), 'task_name': 'mcq'}
    )
    mi = log.model_info
    assert mi.id == 'Qwen/Qwen3-8B'  # transport prefix stripped
    assert mi.developer == 'Qwen'
    assert mi.inference_engine and mi.inference_engine.name == 'vllm'
    assert mi.additional_details['deployment_type'] == 'self_deployed'
    assert mi.additional_details['model_availability'] == 'open_weights'


# ── aggregate-only security posture + publish round-trip ─────────────────


def test_publish_is_aggregate_only_and_valid(tmp_path):
    out = tmp_path / 'data'
    logs = SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    paths = publish_evaluation_logs(
        logs, out, [str(uuid.uuid4()) for _ in logs]
    )
    assert len(paths) == 4
    # SECURITY: sayf-eval item text is dual-use — no instance-level samples.
    assert list(out.rglob('*_samples.jsonl')) == []
    # every published file is a schema-valid aggregate with no item text
    for p in out.rglob('*.json'):
        d = json.loads(Path(p).read_text())
        EvaluationLog.model_validate(d)
        blob = json.dumps(d).lower()
        for forbidden in ('model_response', 'extracted_answer', 'ground_truth'):
            assert forbidden not in blob


def test_directory_conversion_finds_record(tmp_path):
    # mimic a sayf-eval output dir: results/<model>/results_<ts>.json
    rec_dir = tmp_path / 'results' / 'openai__gpt-4o'
    rec_dir.mkdir(parents=True)
    (rec_dir / 'results_2026-01-01T00-00-00.json').write_text(
        FIXTURE.read_text()
    )
    result = SayfEvalAdapter().transform_from_directory_result(
        tmp_path, _meta()
    )
    assert not result.failures
    assert len(result.records) == 4
