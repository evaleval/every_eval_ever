"""Offline tests for the measurement_db adapter."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

pytest.importorskip('pyarrow', reason='pyarrow not installed')

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from every_eval_ever.adapters.measurement_db import adapter  # noqa: E402
from every_eval_ever.validate import main as validate_cli  # noqa: E402
from every_eval_ever.validate import validate_file  # noqa: E402


def _make_mdb_benchmark(
    bdir: Path,
    benchmark_id: str = 'matharena',
    multi_condition: bool = False,
    include_ungraded: bool = True,
    response_type: str = 'binary',
    long_trace: bool = False,
) -> Path:
    tdir = bdir / 'formatted_tables'
    tdir.mkdir(parents=True, exist_ok=True)
    scale = (
        {'kind': 'discrete', 'values': [0, 1], 'direction': 'higher_is_better'}
        if response_type == 'binary'
        else {'kind': 'bounded', 'min': 0.0, 'max': 10.0}
    )
    benchmarks_rows = [{
        'benchmark_id': benchmark_id,
        'name': 'MathArena',
        'source_url': 'https://huggingface.co/datasets/MathArena/aime_2025',
        'response_type': response_type,
        'response_scale': json.dumps(scale),
        'paper_url': 'https://arxiv.org/abs/2505.23281',
        'release': 'public',
    }]
    subjects_rows = [
        {
            'subject_id': 'subj000000000001',
            'display_name': 'GPT-4o (low effort)',
            'normalized_name': 'openrouter/openai/gpt-4o-2024-08-06',
            'provider': 'OpenAI',
            'harness': 'matharena_runner',
            'reasoning_effort': 'low',
            'harness_version': '1.0',
            'subject_features_extra': json.dumps({
                'deployment_type': 'externally_managed',
                'model_availability': 'closed_weights',
            }),
        },
        {
            'subject_id': 'subj000000000002',
            'display_name': 'GPT-4o (high effort)',
            'normalized_name': 'openai/gpt-4o-2024-08-06',
            'provider': 'OpenAI',
            'harness': 'matharena_runner',
            'reasoning_effort': 'high',
            'harness_version': '1.0',
            'subject_features_extra': None,
        },
    ]
    items_rows = [
        {
            'item_id': f'item00000000000{idx}',
            'raw_item_id': f'aime_2025_{idx}',
            'content': f'Compute the integer value for problem {idx}: ' + ('x' * 100),
            'grading_criterion': json.dumps({'reference_answer': str(10 + idx)}),
        }
        for idx in range(3)
    ]
    responses_rows, traces_rows = [], []
    rid = 0
    for s_idx, subj in enumerate(subjects_rows):
        for i_idx, item in enumerate(items_rows):
            rid += 1
            resp_id = f'resp{rid:012d}'
            cond = ('temp=0.0' if i_idx < 2 else 'temp=0.7') if multi_condition else None
            val = None if (include_ungraded and s_idx == 1 and i_idx == 2) else (1.0 if (s_idx + i_idx) % 2 == 0 else 0.0)
            responses_rows.append({
                'subject_id': subj['subject_id'],
                'item_id': item['item_id'],
                'response_id': resp_id,
                'trial': 1,
                'test_condition': cond,
                'interactors': None,
                'response': val,
            })
            trace_body = f'Solve {i_idx}: \\boxed{{{10 + i_idx}}}' + ('T' * 12_000 if long_trace and i_idx == 0 else '')
            traces_rows.append({'response_id': resp_id, 'trace': trace_body})

    for name, rows in (
        ('benchmarks', benchmarks_rows),
        ('subjects', subjects_rows),
        ('items', items_rows),
        ('responses', responses_rows),
        ('traces', traces_rows),
    ):
        pq.write_table(pa.Table.from_pylist(rows), tdir / f'{name}.parquet')
    return bdir


def test_benchmark_selection_and_deduplication():
    candidates = ['matharena', 'hle', 'helm_bbq', 'mmlu', 'arena_140k']
    assert adapter.selected_benchmarks(candidates, None, False) == ['matharena', 'hle']
    assert adapter.selected_benchmarks(candidates, ['helm_bbq'], True) == ['helm_bbq']
    with pytest.raises(SystemExit, match='withheld'):
        adapter.selected_benchmarks(candidates, ['arena_140k'], True)


def test_aggregates_and_instances_pass_datastore_gate(tmp_path):
    bdir = _make_mdb_benchmark(tmp_path / 'matharena', multi_condition=True, long_trace=True)
    out = tmp_path / 'data' / 'measurement-db'
    argv = [
        '--benchmark-dir', str(bdir), '--output-dir', str(out),
        '--evaluation-timestamp', '1759276800.0', '--retrieved-timestamp', '1760000000.0',
        '--include-instances', '--no-registry-resolve',
    ]
    assert adapter.run(adapter.parse_args(argv)) == 2
    json_files = sorted(p for p in out.rglob('*.json') if not p.name.startswith('_'))
    assert len(json_files) == 2

    for jf in json_files:
        report = validate_file(jf)
        assert report.valid and not report.warnings

    assert validate_cli([str(p) for p in sorted(out.rglob('*.json*')) if not p.name.startswith('_')]) == 0

    logs = [json.loads(p.read_text(encoding='utf-8')) for p in json_files]
    assert {log['evaluation_id'] for log in logs} == {
        'measurement-db/matharena/subj000000000001/1759276800.0',
        'measurement-db/matharena/subj000000000002/1759276800.0',
    }

    subj1 = next(log for log in logs if 'subj000000000001' in log['evaluation_id'])
    assert subj1['model_info']['id'] == 'openai/gpt-4o-2024-08-06'
    assert subj1['model_info']['inference_platform'] == 'openrouter'
    assert subj1['evaluation_results'][0]['source_data']['hf_repo'] == 'MathArena/aime_2025'
    assert subj1['source_metadata']['additional_details']['paper_url'] == 'https://arxiv.org/abs/2505.23281'

    subj2 = next(log for log in logs if 'subj000000000002' in log['evaluation_id'])
    overall = next(r for r in subj2['evaluation_results'] if r['evaluation_name'] == 'measurement_db.matharena')
    assert overall['metric_config']['metric_id'] == 'accuracy'
    assert overall['score_details']['uncertainty']['num_samples'] == 2
    assert overall['score_details']['details']['n_ungraded'] == '1'
    assert math.isclose(overall['score_details']['score'], 0.5)

    sidecar = json_files[0].with_name(f'{json_files[0].stem}_samples.jsonl')
    first_inst = json.loads(sidecar.read_text(encoding='utf-8').splitlines()[0])
    full_prompt = 'Compute the integer value for problem 0: ' + ('x' * 100)
    assert first_inst['input']['raw'] == full_prompt
    expected_hash = hashlib.sha256(
        json.dumps(
            {'raw': first_inst['input']['raw'], 'reference': first_inst['input']['reference']},
            sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    assert first_inst['sample_hash'] == expected_hash
    assert first_inst['answer_attribution'] == []
    assert '[truncated ' in first_inst['output']['raw'][0]


def test_replace_existing_and_non_binary_metric_id(tmp_path):
    bdir = _make_mdb_benchmark(tmp_path / 'matharena', response_type='continuous')
    out = tmp_path / 'data' / 'measurement-db'
    argv = [
        '--benchmark-dir', str(bdir), '--output-dir', str(out),
        '--evaluation-timestamp', '1759276800.0', '--include-instances', '--no-registry-resolve',
    ]
    assert adapter.run(adapter.parse_args(argv)) == 2
    json_files = sorted(p for p in out.rglob('*.json') if not p.name.startswith('_'))
    log = json.loads(json_files[0].read_text(encoding='utf-8'))
    assert log['evaluation_results'][0]['metric_config']['metric_id'] == 'measurement_db.matharena.continuous'

    with pytest.raises(SystemExit, match='--replace-existing'):
        adapter.run(adapter.parse_args(argv))
    assert adapter.run(adapter.parse_args(argv + ['--replace-existing'])) == 2
    assert len([p for p in out.rglob('*.json') if not p.name.startswith('_')]) == 2


def test_extract_final_output_from_agent_json_traces():
    hle_trace = json.dumps({
        'source': 'supai',
        'events': [{'type': 'chunk', 'text': 'thinking...' * 1000}],
        'final_output': 'Final Answer: D',
    })
    assert adapter._extract_final_output(hle_trace) == 'Final Answer: D'

    taubench_trace = json.dumps({
        'logs': [{'id': '1', 'op': 'litellm'}],
        'result': {'reward': 1.0, 'taken_actions': [{'name': 'respond', 'kwargs': {'content': 'Booked!'}}]},
    })
    assert 'Booked!' in adapter._extract_final_output(taubench_trace)

    sciarena_trace = json.dumps({
        'side': 'modelB',
        'record': {'responseA': 'Wrong side', 'responseB': 'Selected model response'},
    })
    assert adapter._extract_final_output(sciarena_trace) == 'Selected model response'
