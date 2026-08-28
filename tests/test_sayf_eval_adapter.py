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
from every_eval_ever.validate import validate_file

DATA_DIR = Path('tests/data/sayf_eval')
FIXTURE = DATA_DIR / 'results_fixture.json'


def _meta(**overrides):
    args = {
        'source_organization_name': 'QCRI',
        'evaluator_relationship': 'third_party',
        'eval_library_name': 'sayf-eval',
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
    assert tasks == ['ate', 'athena_vsp', 'mcq']


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


def test_judge_recorded_with_real_prompt_template():
    acc = _logs_by_task()['mcq'].evaluation_results[0]
    scoring = acc.metric_config.llm_scoring
    assert scoring is not None
    assert scoring.judges[0].model_info.id.startswith('anthropic/')
    # input_prompt is the actual judge template (scaffolding + placeholders),
    # not a fabricated description.
    assert scoring.input_prompt.startswith('You are a strict evaluator')
    assert '{question}' in scoring.input_prompt


def test_llm_scoring_omitted_when_no_template():
    # A record without judge_prompt_templates must not fabricate a prompt; the
    # judge model is preserved in the metric's additional_details instead.
    record = json.loads(FIXTURE.read_text())
    record.pop('judge_prompt_templates', None)
    log = SayfEvalAdapter()._transform_single(
        record, {**_meta(), 'task_name': 'mcq'}
    )
    mc = log.evaluation_results[0].metric_config
    assert mc.llm_scoring is None
    assert mc.additional_details['judge_model'].startswith('anthropic/')


# ── source_data provenance ───────────────────────────────────────────────


def test_source_data_preserves_upstream_names():
    logs = _logs_by_task()
    hf = logs['mcq'].evaluation_results[0].source_data
    assert isinstance(hf, SourceDataHf)
    assert hf.hf_repo.startswith('RISys-Lab/')
    # upstream dataset name is preserved (not overwritten by a routing slug)
    assert hf.dataset_name == 'CTI-Bench MCQ'
    assert hf.additional_details['subset'] == 'cti-mcq'

    url = logs['athena_vsp'].evaluation_results[0].source_data
    assert isinstance(url, SourceDataUrl)
    assert url.dataset_name == 'AthenaBench VSP'
    assert url.url and url.url[0].endswith('athena-cti-vsp.jsonl')


def test_other_source_maps_to_private():
    # sayf-eval no longer declares an 'other'/private source, but the adapter
    # still maps one to SourceDataPrivate for robustness.
    sd = SayfEvalAdapter()._build_source_data(
        'x', {'type': 'other', 'dataset_name': 'Private X'}
    )
    assert isinstance(sd, SourceDataPrivate)
    assert sd.source_type == 'other'


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


def test_publish_is_aggregate_only_valid_and_no_item_text(tmp_path):
    import copy

    out = tmp_path / 'data'
    logs = SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    paths = []
    for log, u in zip(logs, [str(uuid.uuid4()) for _ in logs]):
        task = log.evaluation_id.split('/', 1)[0]
        paths += publish_evaluation_logs(
            [log],
            out,
            [u],
            collection_override=f'sayf-eval-{task.replace("_", "-")}',
        )
    assert len(paths) == 3
    # one namespaced collection per task (data/sayf-eval-<task>/...)
    collections = {p.parent.parent.parent.name for p in out.rglob('*.json')}
    assert collections == {
        'sayf-eval-mcq',
        'sayf-eval-athena-vsp',
        'sayf-eval-ate',
    }
    # SECURITY: sayf-eval item text is dual-use — no instance-level samples.
    assert list(out.rglob('*_samples.jsonl')) == []
    for p in out.rglob('*.json'):
        # the repository's semantic validation gate, not just Pydantic
        report = validate_file(p)
        assert report.valid, report.errors
        # no per-sample item text anywhere except the (sample-independent) judge
        # prompt template carried in llm_scoring.input_prompt
        d = copy.deepcopy(json.loads(Path(p).read_text()))
        for r in d.get('evaluation_results', []):
            scoring = (r.get('metric_config') or {}).get('llm_scoring')
            if scoring:
                scoring.pop('input_prompt', None)
        blob = json.dumps(d).lower()
        for forbidden in ('model_response', 'extracted_answer', 'ground_truth'):
            assert forbidden not in blob


def test_evaluation_id_is_stable_across_conversions():
    # Identity is derived from the record's created_at, not wall-clock time, so
    # re-converting the same record yields the same evaluation_id each time.
    ids1 = {
        lg.evaluation_id
        for lg in SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    }
    ids2 = {
        lg.evaluation_id
        for lg in SayfEvalAdapter().transform_from_file(FIXTURE, _meta())
    }
    assert ids1 == ids2 and len(ids1) == 3
    log = SayfEvalAdapter().transform_from_file(FIXTURE, _meta())[0]
    assert log.retrieved_timestamp  # wall-clock retrieval time kept separately


def test_partial_conversion_keeps_valid_siblings(tmp_path):
    # A task that cannot convert must not discard its valid siblings; it is
    # recorded as a failure while the rest of the file still converts.
    record = json.loads(FIXTURE.read_text())
    record['results']['mcq'] = {}  # no numeric metrics -> mcq fails to convert
    rec_dir = tmp_path / 'results' / 'openai__gpt-4o'
    rec_dir.mkdir(parents=True)
    (rec_dir / 'results_2026-01-01T00-00-00.json').write_text(
        json.dumps(record)
    )
    result = SayfEvalAdapter().transform_from_directory_result(
        tmp_path, _meta()
    )
    tasks = sorted(lg.evaluation_id.split('/')[0] for lg in result.records)
    assert tasks == ['ate', 'athena_vsp']  # siblings kept, mcq dropped
    assert any('mcq' in f.source_ref for f in result.failures)


def test_cli_routes_each_task_to_its_own_collection(tmp_path):
    # End-to-end: the CLI routes each task into data/sayf-eval-<task>/... so
    # EEE's per-collection Community-Evals tool maps one collection per benchmark.
    from every_eval_ever import cli

    out = tmp_path / 'data'
    rc = cli.main(
        [
            'convert',
            'sayf_eval',
            '--log_path',
            str(FIXTURE),
            '--output_dir',
            str(out),
            '--source_organization_name',
            'QCRI',
        ]
    )
    assert rc == 0
    collections = {p.parent.parent.parent.name for p in out.rglob('*.json')}
    assert collections == {
        'sayf-eval-mcq',
        'sayf-eval-athena-vsp',
        'sayf-eval-ate',
    }


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
    assert len(result.records) == 3
