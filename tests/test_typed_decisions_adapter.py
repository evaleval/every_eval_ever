"""Tests for the typed-decisions adapter.

The fixture under ``tests/data/typed_decisions/`` is a byte-exact three-case
subset of the benchmark's English test split and a matching run's
predictions. It covers every decision shape the adapter scores: choice,
noul (with and without a criteria map), and score with both four and five
levels. Expected accuracies below were recomputed with the benchmark's
official scorer semantics; the full 400-case conversion is gated the same
way in the submission runbook.
"""

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.typed_decisions import adapter
from every_eval_ever.helpers.io import (
    SourceRecordsError,
    default_failure_report_path,
)
from every_eval_ever.validate import validate_file

FIXTURES = Path(__file__).resolve().parent / 'data' / 'typed_decisions'
ROWS = FIXTURES / 'rows_sample.jsonl'
PREDS = FIXTURES / 'preds_sample.jsonl'
COLLECTION = 'typed-decisions'
MODEL_ID = 'Phocinae/Phocinae-Largha-150M-v1'

#: Fixture expectations, recomputed with the official scorer semantics.
EXPECTED_ACCURACY = 0.8  # 12 of 15 decisions
EXPECTED_PER_TYPE = {
    'choice': 0.75,  # 3 of 4
    'noul': 0.8,  # 4 of 5
    'score': 0.8333,  # 5 of 6
}
EXPECTED_SAMPLE_IDS = {
    'agent_trace_observability_000000::action',
    'agent_trace_observability_000000::needs_review',
    'agent_trace_observability_000000::outcome',
    'agent_trace_observability_000000::risk',
    'agent_trace_observability_000000::urgency',
    'invoice_processing_000000::discrepancy_severity',
    'invoice_processing_000000::disposition',
    'invoice_processing_000000::duplicate',
    'invoice_processing_000000::matches_order',
    'invoice_processing_000000::urgency',
    'security_incidents_000000::credential_compromise',
    'security_incidents_000000::disposition',
    'security_incidents_000000::severity',
    'security_incidents_000000::true_positive',
    'security_incidents_000000::urgency',
}


def _args(output_dir: Path, *extra: str):
    return adapter.build_parser().parse_args(
        [
            '--rows',
            str(ROWS),
            '--preds',
            str(PREDS),
            '--output-dir',
            str(output_dir),
            '--evaluation-timestamp',
            '2026-10-09T02:19:10+08:00',
            '--retrieved-timestamp',
            '1791483550.3',
            *extra,
        ]
    )


def _validate(output_dir: Path, path: Path):
    root = output_dir.parent
    repo_path = path.relative_to(root).as_posix()
    available = frozenset(
        candidate.relative_to(root).as_posix()
        for candidate in path.parent.iterdir()
        if candidate.is_file()
    )

    def reader(repo_file: str) -> str:
        return (root / repo_file).read_text(encoding='utf-8')

    return validate_file(
        path,
        repo_path=repo_path,
        available_files=available,
        read_repo_file=reader,
        run_semantic_checks=True,
    )


def _run_fixture(tmp_path: Path, *extra: str):
    output_dir = tmp_path / 'data' / COLLECTION
    assert adapter.run(_args(output_dir, *extra)) == 1
    log_dir = output_dir / MODEL_ID
    aggregate = next(log_dir.glob('*.json'))
    samples = aggregate.with_name(f'{aggregate.stem}_samples.jsonl')
    return aggregate, samples


def test_fixture_conversion_matches_official_scorer():
    instances, stats, result = adapter.convert(
        _args(Path('/tmp/unused'), '--include-instances')
    )

    assert not result.failures
    assert stats['n_decisions'] == 15
    assert stats['accuracy'] == EXPECTED_ACCURACY
    assert {
        key: round(value, 4) for key, value in stats['per_type'].items()
    } == EXPECTED_PER_TYPE
    assert len(result.records) == 1
    evaluation_result = result.records[0].evaluation_results[0]
    assert evaluation_result.score_details.score == EXPECTED_ACCURACY
    assert evaluation_result.score_details.details == {
        'n_cases': '3',
        'n_decisions': '15',
        'accuracy_choice': '0.75',
        'accuracy_noul': '0.8',
        'accuracy_score': '0.8333',
        'accuracy_agent_trace_observability': '1.0',
        'accuracy_invoice_processing': '0.6',
        'accuracy_security_incidents': '0.8',
    }
    assert len(instances) == 15


def test_published_records_validate(tmp_path: Path):
    aggregate, samples = _run_fixture(tmp_path, '--include-instances')

    for path in (aggregate, samples):
        report = _validate(tmp_path / 'data', path)
        assert report.valid, report.errors
        assert not report.warnings, report.warnings


def test_published_record_identity_and_instances(tmp_path: Path):
    aggregate, samples = _run_fixture(tmp_path, '--include-instances')

    log = json.loads(aggregate.read_text(encoding='utf-8'))
    assert log['model_info']['id'] == MODEL_ID
    assert log['evaluation_id'].startswith(
        f'typed-decisions-en/{MODEL_ID}/2026-10-08T18:19:10Z'
    )
    assert log['model_info']['additional_details'] == {
        'deployment_type': 'self_deployed',
        'model_availability': 'open_weights',
    }
    evaluation_result = log['evaluation_results'][0]
    assert evaluation_result['evaluation_name'] == 'typed-decisions-en'
    assert (
        evaluation_result['evaluation_result_id']
        == f'{MODEL_ID}#typed-decisions-en#accuracy'
    )
    assert evaluation_result['source_data']['hf_repo'] == (
        'LocalLLaMA/typed-decisions'
    )
    assert evaluation_result['metric_config']['min_score'] == 0.0
    assert evaluation_result['metric_config']['max_score'] == 1.0

    sample_rows = [
        json.loads(line)
        for line in samples.read_text(encoding='utf-8').splitlines()
    ]
    assert len(sample_rows) == 15
    assert {row['sample_id'] for row in sample_rows} == EXPECTED_SAMPLE_IDS
    for row in sample_rows:
        assert row['evaluation_id'] == log['evaluation_id']
        assert row['model_id'] == MODEL_ID
        assert row['evaluation_name'] == 'typed-decisions-en'
        assert (
            row['evaluation_result_id']
            == evaluation_result['evaluation_result_id']
        )


def test_sample_hash_matches_canonical_recipe(
    tmp_path: Path,
):
    _, samples = _run_fixture(tmp_path, '--include-instances')

    for line in samples.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        assert row['sample_hash'] == adapter.sample_hash(
            row['input']['raw'], row['input']['reference']
        )
        # The one canonical recipe shared by every adapter.
        payload = json.dumps(
            {
                'raw': row['input']['raw'],
                'reference': row['input']['reference'],
            },
            sort_keys=True,
            separators=(',', ':'),
        )
        import hashlib

        assert (
            row['sample_hash']
            == hashlib.sha256(payload.encode('utf-8')).hexdigest()
        )


def test_null_criteria_noul_and_five_level_score_flow_through(
    tmp_path: Path,
):
    _, samples = _run_fixture(tmp_path, '--include-instances')

    rows = {
        json.loads(line)['sample_id']: json.loads(line)
        for line in samples.read_text(encoding='utf-8').splitlines()
    }
    duplicate = rows['invoice_processing_000000::duplicate']
    assert duplicate['metadata']['question_type'] == 'noul'
    assert duplicate['input']['choices'] == ['false', 'true']
    severity = rows['security_incidents_000000::severity']
    assert severity['metadata']['question_type'] == 'score'
    assert severity['input']['choices'] == ['0', '1', '2', '3', '4']


def _write_case_pair(tmp_path: Path, question: dict, gold: dict, pred: dict):
    rows = tmp_path / 'rows.jsonl'
    preds = tmp_path / 'preds.jsonl'
    preds.write_text(
        json.dumps(
            {
                'id': 'case_0',
                'workflow': 'unit',
                'pred': {'q': pred},
                'latency_ms': 1.0,
            }
        )
        + '\n',
        encoding='utf-8',
    )
    # Rows carry inline JSON strings, the form the raw source uses.
    rows.write_text(
        json.dumps(
            {
                'id': 'case_0',
                'workflow': 'unit',
                'state': json.dumps({'task': 'unit'}),
                'questions': json.dumps({'q': question}),
                'gold': json.dumps({'q': gold}),
            }
        )
        + '\n',
        encoding='utf-8',
    )
    return adapter.build_parser().parse_args(
        [
            '--rows',
            str(rows),
            '--preds',
            str(preds),
            '--output-dir',
            str(tmp_path / 'data' / COLLECTION),
        ]
    )


def test_noul_threshold_is_inclusive(tmp_path: Path):
    question = {'type': 'noul', 'criteria': None, 'instructions': 'x'}
    at_threshold = _write_case_pair(
        tmp_path,
        question,
        {'label': 'true', 'type': 'noul'},
        {'type': 'noul', 'noul': 0.5},
    )
    instances, stats, _ = adapter.convert(at_threshold)
    assert instances[0].answer_attribution[0].extracted_value == 'true'
    assert stats['accuracy'] == 1.0

    below = _write_case_pair(
        tmp_path,
        question,
        {'label': 'true', 'type': 'noul'},
        {'type': 'noul', 'noul': 0.4999},
    )
    instances, stats, _ = adapter.convert(below)
    assert instances[0].answer_attribution[0].extracted_value == 'false'
    assert stats['accuracy'] == 0.0


def test_score_uses_probability_argmax_not_rounded_expectation(
    tmp_path: Path,
):
    question = {
        'type': 'score',
        'criteria': ['a', 'b'],
        'instructions': 'x',
    }
    # Rounding the expectation (0.9) would answer level 1; the official
    # scorer takes the probability argmax, i.e. level 0.
    args = _write_case_pair(
        tmp_path,
        question,
        {'label': '1', 'type': 'score', 'score': 1.0},
        {
            'type': 'score',
            'score': 0.9,
            'probabilities': {'0': 0.5, '1': 0.5},
        },
    )
    instances, stats, _ = adapter.convert(args)
    assert instances[0].answer_attribution[0].extracted_value == '0'
    assert instances[0].evaluation.is_correct is False
    assert stats['accuracy'] == 0.0


def test_score_falls_back_to_rounded_score_when_probabilities_are_zero(
    tmp_path: Path,
):
    question = {
        'type': 'score',
        'criteria': ['a', 'b', 'c'],
        'instructions': 'x',
    }
    args = _write_case_pair(
        tmp_path,
        question,
        {'label': '2', 'type': 'score', 'score': 2.0},
        {'type': 'score', 'score': 1.6, 'probabilities': {}},
    )
    instances, stats, _ = adapter.convert(args)
    assert instances[0].answer_attribution[0].extracted_value == '2'
    assert stats['accuracy'] == 1.0


def test_choice_exact_match(tmp_path: Path):
    question = {
        'type': 'choice',
        'criteria': {'a': 'a', 'b': 'b'},
        'instructions': 'x',
    }
    args = _write_case_pair(
        tmp_path,
        question,
        {'label': 'a', 'type': 'choice'},
        {
            'type': 'choice',
            'choice': 'b',
            'probabilities': {'a': 0.4, 'b': 0.6},
        },
    )
    instances, stats, _ = adapter.convert(args)
    assert instances[0].answer_attribution[0].extracted_value == 'b'
    assert instances[0].evaluation.is_correct is False


def test_zh_variant_identity(tmp_path: Path):
    output_dir = tmp_path / 'data' / COLLECTION
    args = _args(output_dir, '--variant', 'zh', '--include-instances')
    instances, _, result = adapter.convert(args)
    aggregate = result.records[0]
    evaluation_result = aggregate.evaluation_results[0]
    assert evaluation_result.evaluation_name == 'typed-decisions-zh'
    assert evaluation_result.source_data.dataset_name == (
        'Phocinae/phocinae-zh400'
    )
    assert evaluation_result.evaluation_result_id == (
        f'{MODEL_ID}#typed-decisions-zh#accuracy'
    )
    assert instances[0].evaluation_name == 'typed-decisions-zh'


def test_output_dir_must_end_with_collection(tmp_path: Path):
    bad = adapter.build_parser().parse_args(
        [
            '--rows',
            str(ROWS),
            '--preds',
            str(PREDS),
            '--output-dir',
            str(tmp_path / 'data' / 'something-else'),
        ]
    )
    with pytest.raises(SystemExit):
        adapter.run(bad)


def test_missing_predictions_fail_without_publishing(tmp_path: Path):
    truncated = tmp_path / 'preds_partial.jsonl'
    lines = PREDS.read_text(encoding='utf-8').splitlines()
    truncated.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    output_dir = tmp_path / 'data' / COLLECTION
    args = adapter.build_parser().parse_args(
        [
            '--rows',
            str(ROWS),
            '--preds',
            str(truncated),
            '--output-dir',
            str(output_dir),
            '--include-instances',
        ]
    )
    with pytest.raises(SourceRecordsError):
        adapter.run(args)
    assert not list(output_dir.rglob('*.json'))
    report = default_failure_report_path(output_dir)
    assert report.is_file()
    assert not str(report).startswith(str(output_dir))


def test_limit_cases(tmp_path: Path):
    instances, stats, result = adapter.convert(
        _args(Path('/tmp/unused'), '--limit-cases', '1')
    )
    assert stats['n_cases'] == 1
    assert stats['n_decisions'] == 5
    assert (
        result.records[0]
        .evaluation_results[0]
        .source_data.model_dump()['samples_number']
        == 1
    )
