import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from every_eval_ever.adapters.hfopenllm_v2 import adapter
from every_eval_ever.adapters.hfopenllm_v2.adapter import (
    HARNESS_FORK,
    contents_parquet_revision,
    convert_model,
    convert_models,
    download_results,
    process_models,
    results_dir_revision,
)
from every_eval_ever.adapters.hfopenllm_v2.contents import (
    CONTENTS_FILE,
    CONTENTS_REVISION,
    contents_row,
)
from every_eval_ever.adapters.hfopenllm_v2.dump_index import (
    RESULTS_REVISION,
    build_dump_index,
    dump_identity,
)
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _model(evaluations):
    return {
        'model': {
            'name': 'example/model',
            'precision': 'bfloat16',
        },
        'metadata': {},
        'evaluations': evaluations,
    }


def test_single_model_conversion_remains_strict():
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )

    try:
        convert_model(source, '1234')
    except ValueError as exc:
        assert "Evaluation 'bbh' could not be converted" in str(exc)
    else:
        raise AssertionError(
            'expected strict conversion to reject missing score'
        )


def test_batch_keeps_valid_metrics_and_records_missing_metric():
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )

    result = convert_models([source], retrieved_timestamp='1234')

    assert len(result.records) == 1
    assert [
        metric.evaluation_name
        for metric in result.records[0].eval_log.evaluation_results
    ] == ['IFEval']
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == "model row 0 evaluation 'bbh'"
    assert result.failures[0].source_record == {
        'model': source['model'],
        'evaluation_key': 'bbh',
        'evaluation': source['evaluations']['bbh'],
    }


def test_process_models_writes_valid_output_and_external_failure_report(
    tmp_path,
):
    source = _model(
        {
            'ifeval': {'name': 'IFEval', 'value': 0.75},
            'bbh': {'name': 'BBH', 'value': None},
        }
    )
    output_dir = tmp_path / 'data' / 'hfopenllm_v2'

    try:
        process_models([source], str(output_dir))
    except SourceRecordsError:
        pass
    else:
        raise AssertionError('expected partial conversion to be signalled')

    outputs = list(output_dir.glob('*/*/*.json'))
    assert len(outputs) == 1
    assert validate_file(outputs[0]).valid

    report_path = tmp_path / 'adapter_reports' / 'hfopenllm_v2_failures.json'
    report = json.loads(report_path.read_text())
    assert report['converted_records'] == 1
    assert len(report['failed_records']) == 1
    assert not report_path.is_relative_to(output_dir)


RESULTS = Path(__file__).parent / 'data' / 'hfopenllm_v2' / 'results'
COGITO = 'Daemontatox/DocumentCogito'
DBRX = 'databricks/dbrx-base'


def _row(name, precision, keys=('ifeval', 'bbh', 'math', 'gpqa', 'musr')):
    names = {
        'ifeval': 'IFEval',
        'bbh': 'BBH',
        'math': 'MATH Level 5',
        'gpqa': 'GPQA',
        'musr': 'MUSR',
        'mmlu_pro': 'MMLU-PRO',
    }
    return {
        'model': {'name': name, 'precision': precision},
        'metadata': {},
        'evaluations': {
            key: {'name': names[key], 'value': 0.5}
            for key in keys + ('mmlu_pro',)
        },
    }


def _results(log):
    return {r.evaluation_name: r for r in log.evaluation_results}


def _copy_results(tmp_path, *paths):
    root = tmp_path / 'results'
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RESULTS / rel, target)
    return root


def _convert(rows, results_dir=RESULTS):
    index = build_dump_index(results_dir)
    result = convert_models(rows, retrieved_timestamp='1234', dump_index=index)
    assert not result.failures
    return [record.eval_log for record in result.records]


def test_record_matches_datastore_shape_without_dumps():
    (log,) = _convert([_row('example/model', 'bfloat16')], RESULTS / 'none')

    assert log.evaluation_id == 'hfopenllm_v2/example_model/bfloat16/1234'
    ifeval = _results(log)['IFEval']
    assert ifeval.evaluation_result_id == (
        'hfopenllm_v2/example_model/bfloat16/1234#ifeval#accuracy'
    )
    assert _results(log)['MATH Level 5'].evaluation_result_id.endswith(
        '#math_level_5#exact_match'
    )
    metric = _results(log)['MMLU-PRO'].metric_config
    assert (
        metric.metric_id,
        metric.metric_name,
        metric.metric_kind,
        metric.metric_unit,
    ) == ('accuracy', 'Accuracy', 'accuracy', 'proportion')
    assert ifeval.generation_config is None
    assert ifeval.source_data.hf_repo == 'google/IFEval'
    assert ifeval.source_data.hf_split is None
    assert ifeval.source_data.samples_number is None
    assert log.eval_library.additional_details == {
        'fork': HARNESS_FORK,
    }


def test_two_precisions_get_distinct_ids_and_their_own_dumps():
    bf16, fp16 = _convert([_row(COGITO, 'bfloat16'), _row(COGITO, 'float16')])

    assert bf16.evaluation_id != fp16.evaluation_id
    assert bf16.evaluation_id.endswith('/bfloat16/1234')
    assert fp16.evaluation_id.endswith('/float16/1234')
    assert bf16.eval_library.additional_details['transformers_version'] == (
        '4.48.0'
    )
    assert fp16.eval_library.additional_details['transformers_version'] == (
        '4.49.0'
    )
    assert bf16.eval_library.additional_details['git_hash'] == 'a781a6b'


def test_generative_groups_get_decoding_and_multiple_choice_do_not():
    (log,) = _convert([_row(COGITO, 'bfloat16')])
    results = _results(log)

    ifeval = results['IFEval'].generation_config
    assert ifeval.generation_args.temperature == 0.0
    assert ifeval.generation_args.max_tokens == 1280
    assert ifeval.additional_details == {
        'output_type': 'generate_until',
        'num_fewshot': '0',
        'do_sample': 'false',
    }
    math = results['MATH Level 5'].generation_config
    assert math.generation_args.max_tokens == 1024
    assert math.additional_details['num_fewshot'] == '4'

    bbh = results['BBH'].generation_config
    assert bbh.generation_args is None
    assert bbh.additional_details == {
        'output_type': 'multiple_choice',
        'num_fewshot': '3',
    }
    for name in ('GPQA', 'MUSR', 'MMLU-PRO'):
        config = results[name].generation_config
        assert config.generation_args is None
        assert config.additional_details['output_type'] == 'multiple_choice'


def test_dataset_split_and_counts_come_from_the_dumps():
    (log,) = _convert([_row(COGITO, 'bfloat16')])
    results = _results(log)
    for result in results.values():
        assert result.score_details.uncertainty.num_samples == (
            result.source_data.samples_number
        )

    ifeval = results['IFEval'].source_data
    assert ifeval.hf_repo == 'google/IFEval'
    assert ifeval.additional_details == {
        'harness_dataset_path': 'wis-k/instruction-following-eval'
    }
    assert ifeval.hf_split == 'train'
    assert ifeval.samples_number == 541
    math = results['MATH Level 5'].source_data
    assert math.hf_repo == 'DigitalLearningGmbH/MATH-lighteval'
    assert math.additional_details == {
        'harness_dataset_path': 'lighteval/MATH-Hard'
    }
    assert math.hf_split == 'test'
    assert math.samples_number == 1324
    assert results['BBH'].source_data.samples_number == 5761
    assert results['MMLU-PRO'].source_data.samples_number == 12032

    gpqa = results['GPQA'].source_data
    assert gpqa.samples_number == 1192
    assert gpqa.hf_split is None
    musr = results['MUSR'].source_data
    assert musr.hf_repo == 'TAUR-Lab/MuSR'
    assert musr.additional_details is None
    assert results['BBH'].source_data.additional_details is None
    assert musr.samples_number == 756
    assert musr.hf_split is None


def test_limited_run_counts_scored_items_and_reports_dataset_size():
    (log,) = _convert([_row(DBRX, 'float16')])
    results = _results(log)

    math = results['MATH Level 5']
    assert math.generation_config.generation_args.max_tokens == 512
    assert math.source_data.samples_number == 700
    assert math.source_data.additional_details == {
        'harness_dataset_path': 'lighteval/MATH-Hard',
        'dataset_size': '1324',
    }
    bbh = results['BBH'].source_data
    assert bbh.samples_number == 2400
    assert bbh.additional_details == {'dataset_size': '5761'}


def test_cross_dump_disagreement_leaves_field_null(tmp_path):
    first, second = (
        f'{COGITO}/results_2025-01-16T02-47-58.653318.json',
        f'{COGITO}/results_2025-02-13T18-27-04.338360.json',
    )
    root = _copy_results(tmp_path, first, second)
    path = root / second
    dump = json.loads(path.read_text())
    for task, config in dump['configs'].items():
        if task.startswith('leaderboard_math_'):
            config['generation_kwargs']['max_gen_toks'] = 512
    dump['n-samples']['leaderboard_ifeval']['effective'] = 100
    path.write_text(json.dumps(dump))

    (log,) = _convert([_row(COGITO, 'bfloat16')], root)
    results = _results(log)

    assert results['MATH Level 5'].generation_config is None
    assert results['MATH Level 5'].source_data.samples_number == 1324
    assert results['IFEval'].source_data.samples_number is None
    assert results['IFEval'].generation_config.generation_args.max_tokens == (
        1280
    )
    index = build_dump_index(root)
    gaps = index.lookup(COGITO, 'bfloat16').groups
    assert gaps['math'].gaps['max_tokens'] == 'cross_dump_conflict'
    assert gaps['ifeval'].gaps['samples_number'] == 'cross_dump_conflict'


def test_subtask_disagreement_within_a_dump_leaves_group_null(tmp_path):
    rel = f'{DBRX}/results_2025-02-13T18-27-04.338360.json'
    root = _copy_results(tmp_path, rel)
    dump = json.loads((root / rel).read_text())
    dump['configs']['leaderboard_math_algebra_hard']['generation_kwargs'][
        'max_gen_toks'
    ] = 1024
    (root / rel).write_text(json.dumps(dump))

    (log,) = _convert([_row(DBRX, 'float16')], root)

    math = _results(log)['MATH Level 5']
    assert math.generation_config is None
    assert math.source_data.samples_number == 700


def test_dump_contradicting_its_model_args_is_rejected(tmp_path):
    rel = f'{DBRX}/results_2025-02-13T18-27-04.338360.json'
    root = _copy_results(tmp_path, rel)
    dump = json.loads((root / rel).read_text())
    dump['config']['model_args'] = dump['config']['model_args'].replace(
        'pretrained=databricks/dbrx-base', 'pretrained=databricks/dbrx-instruct'
    )
    (root / rel).write_text(json.dumps(dump))

    index = build_dump_index(root)

    assert index.entries == {}
    assert index.rejected[0]['reason'] == 'identity_contradicts_model_args'
    (log,) = _convert([_row(DBRX, 'float16')], root)
    assert _results(log)['BBH'].generation_config is None


def test_adapter_dump_is_named_after_the_adapter():
    assert dump_identity(
        {
            'model_name': 'org/adapter',
            'config': {'model_args': 'pretrained=org/base,peft=org/adapter'},
        }
    ) == ('org/adapter', None)
    assert dump_identity(
        {
            'model_name': 'org/base',
            'config': {'model_args': 'pretrained=org/base,peft=org/adapter'},
        }
    ) == (None, 'identity_contradicts_model_args')


def test_precision_without_a_dump_converts_with_nulls():
    (log,) = _convert([_row(DBRX, 'bfloat16')])

    for result in log.evaluation_results:
        assert result.generation_config is None
        assert result.source_data.samples_number is None
        assert result.source_data.hf_split is None
    assert set(log.eval_library.additional_details) == {'fork'}


def test_process_models_output_validates_and_reports_gaps(tmp_path):
    output_dir = tmp_path / 'data' / 'hfopenllm_v2'
    rows = [
        _row(COGITO, 'bfloat16'),
        _row(COGITO, 'float16'),
        _row(DBRX, 'float16'),
        _row(DBRX, 'bfloat16'),
    ]

    count = process_models(rows, output_dir, build_dump_index(RESULTS))

    assert count == 4
    outputs = sorted(output_dir.glob('*/*/*.json'))
    assert len(outputs) == 4
    for path in outputs:
        report = validate_file(path)
        assert report.valid, report
        cli = subprocess.run(
            [sys.executable, '-m', 'every_eval_ever', 'validate', str(path)],
            capture_output=True,
            text=True,
        )
        assert cli.returncode == 0, cli.stdout + cli.stderr

    report = json.loads(
        (
            tmp_path / 'adapter_reports' / 'hfopenllm_v2_failures.json'
        ).read_text()
    )
    assert report['failed_record_count'] == 0
    counts = report['harness_evidence']['unfilled_counts']
    assert counts['gpqa.hf_split: not_stated'] == 3
    assert counts['musr.hf_split: subtask_conflict'] == 3
    assert counts['ifeval.all: no_dump'] == 1
    assert report['harness_evidence']['dumps_read'] == 4


OTHER_SHA = 'b' * 40


def _snapshot(tmp_path, name):
    root = tmp_path / 'snapshots' / name
    shutil.copytree(RESULTS, root)
    return root


def _run_main(monkeypatch, tmp_path, results_dir, *extra):
    monkeypatch.setattr(
        adapter, 'fetch_json', lambda url: [_row(DBRX, 'float16')]
    )
    output_dir = tmp_path / 'data' / 'hfopenllm_v2'
    adapter.main(
        [
            '--output-dir',
            str(output_dir),
            '--source-api',
            '--results-dir',
            str(results_dir),
            *extra,
        ]
    )
    report = tmp_path / 'adapter_reports' / 'hfopenllm_v2_failures.json'
    return json.loads(report.read_text())['harness_evidence']


def test_results_dir_at_the_pinned_revision_is_accepted(monkeypatch, tmp_path):
    root = _snapshot(tmp_path, RESULTS_REVISION)

    assert results_dir_revision(root) == (RESULTS_REVISION, False)
    evidence = _run_main(monkeypatch, tmp_path, root)
    assert evidence['results_revision'] == RESULTS_REVISION
    assert evidence['results_dir_unpinned'] is False


def test_results_dir_at_another_revision_fails_naming_both(
    monkeypatch, tmp_path
):
    root = _snapshot(tmp_path, OTHER_SHA)
    monkeypatch.setattr(
        adapter,
        'fetch_json',
        lambda url: pytest.fail('fetched before checking the results dir'),
    )

    with pytest.raises(SystemExit) as exc:
        adapter.main(['--results-dir', str(root)])
    assert OTHER_SHA in str(exc.value)
    assert RESULTS_REVISION in str(exc.value)
    with pytest.raises(ValueError, match='no commit sha'):
        results_dir_revision(RESULTS)


def test_unpinned_results_dir_is_recorded_and_reported(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(
        adapter.raw_capture,
        'record_hf_dataset',
        lambda repo, **kwargs: recorded.append(kwargs['revision']),
    )
    root = _snapshot(tmp_path, OTHER_SHA)

    evidence = _run_main(
        monkeypatch, tmp_path, root, '--allow-unpinned-results-dir'
    )

    assert recorded == [OTHER_SHA]
    assert evidence['results_revision'] == OTHER_SHA
    assert evidence['results_dir_unpinned'] is True
    assert results_dir_revision(RESULTS, allow_unpinned=True) == (
        'local',
        True,
    )


@pytest.fixture
def results_git_remote(tmp_path, monkeypatch, request):
    if shutil.which('git') is None:
        pytest.skip('Git is required for the local bulk-fetch fixture')

    def git(*args, cwd=None):
        return subprocess.run(
            ['git', *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    work = tmp_path / 'source repository'
    work.mkdir()
    git('init', '--initial-branch=fixture', cwd=work)
    git('config', 'user.name', 'Fixture', cwd=work)
    git('config', 'user.email', 'fixture@example.invalid', cwd=work)
    (work / 'README.md').write_text('First commit, without result dumps.\n')
    git('add', '.', cwd=work)
    git('commit', '-m', 'Initial fixture', cwd=work)

    cached = tmp_path / 'cached snapshot'
    shutil.copytree(RESULTS, cached)
    spaced = cached / 'extra organization' / 'model with spaces'
    spaced.mkdir(parents=True)
    shutil.copy(
        next((RESULTS / DBRX).glob('results_*.json')),
        spaced / 'results_extra.json',
    )
    for relative in (
        'single/results_shallow.json',
        'one/two/three/results_deep.json',
    ):
        target = cached / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(spaced / 'results_extra.json', target)
    if getattr(request, 'param', None) == 'lfs_pointer':
        (spaced / 'results_future.json').write_text(
            'version https://git-lfs.github.com/spec/v1\n'
            f'oid sha256:{"a" * 64}\n'
            'size 1234\n'
        )
    shutil.copytree(cached, work, dirs_exist_ok=True)
    shutil.copy(spaced / 'results_extra.json', work / 'results_root.json')
    git('add', '.', cwd=work)
    git('commit', '-m', 'Pinned fixture results', cwd=work)
    pinned = git('rev-parse', 'HEAD', cwd=work)

    dump = next((work / DBRX).glob('results_*.json'))
    dump.unlink()
    (work / 'README.md').write_text('Later commit, with a result removed.\n')
    git('add', '.', cwd=work)
    git('commit', '-m', 'Later fixture', cwd=work)
    latest = git('rev-parse', 'HEAD', cwd=work)
    remote = tmp_path / 'remote with spaces.git'
    git('clone', '--bare', str(work), str(remote))

    monkeypatch.setattr(adapter, 'RESULTS_URL', remote.as_uri())
    monkeypatch.setattr(adapter, 'RESULTS_REVISION', pinned)
    return cached, pinned, latest


def test_results_bulk_fetch_matches_cached_index_and_conversion(
    results_git_remote, tmp_path, monkeypatch
):
    cached, pinned, latest = results_git_remote
    commands = []
    actual_run = subprocess.run

    def record_command(args, **kwargs):
        commands.append(args)
        return actual_run(args, **kwargs)

    monkeypatch.setattr(adapter.subprocess, 'run', record_command)
    monkeypatch.setattr(
        adapter.raw_capture,
        'record_hf_dataset',
        lambda *args, **kwargs: pytest.fail('unexpected per-file Hub path'),
    )
    captured = []
    monkeypatch.setattr(
        adapter.raw_capture,
        'record_git_checkout',
        lambda url, checkout, **kwargs: captured.append(
            (url, checkout, kwargs['ref'])
        ),
    )

    root = download_results(tmp_path / 'download with spaces')

    assert results_dir_revision(root) == (pinned, False)
    assert pinned != latest
    assert (root / '.git' / 'shallow').read_text().splitlines() == [pinned]
    assert not (root / 'README.md').exists()
    assert not (root / 'results_root.json').exists()
    assert not list((root / '.git' / 'objects' / 'pack').glob('*.promisor'))
    fetches = [command for command in commands if command[1] == 'fetch']
    assert fetches == [
        ['git', 'fetch', '--depth', '1', adapter.RESULTS_URL, pinned]
    ]
    assert all(
        '--filter' not in argument
        for command in commands
        for argument in command
    )
    assert captured == [(adapter.RESULTS_URL, root, pinned)]
    assert build_dump_index(root) == build_dump_index(cached)
    rows = [_row(COGITO, 'bfloat16'), _row(DBRX, 'float16')]
    assert _convert(rows, root) == _convert(rows, cached)
    for expected in cached.rglob('results_*.json'):
        assert (
            root / expected.relative_to(cached)
        ).read_bytes() == expected.read_bytes()


@pytest.mark.parametrize('results_git_remote', ['lfs_pointer'], indirect=True)
def test_results_lfs_pointer_fails_loudly_after_bulk_fetch(
    results_git_remote, tmp_path, monkeypatch
):
    cached, _, _ = results_git_remote
    root = download_results(tmp_path / 'results')
    monkeypatch.setattr(
        adapter.subprocess,
        'run',
        lambda *args, **kwargs: pytest.fail('unexpected extra Git transfer'),
    )
    for results_dir in (cached, root):
        with pytest.raises(
            RuntimeError, match='results_future.json is a Git LFS pointer'
        ):
            build_dump_index(results_dir)


def test_results_git_checkout_does_not_trust_directory_name(
    results_git_remote, tmp_path, monkeypatch
):
    _, pinned, latest = results_git_remote
    root = download_results(tmp_path / latest)
    assert results_dir_revision(root) == (pinned, False)

    monkeypatch.setattr(adapter, 'RESULTS_REVISION', latest)
    with pytest.raises(ValueError, match=pinned):
        results_dir_revision(root)
    assert results_dir_revision(root, allow_unpinned=True) == (pinned, True)


def test_results_fetch_requires_git(tmp_path, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(adapter.subprocess, 'run', missing_git)
    with pytest.raises(RuntimeError, match='Git is required'):
        download_results(tmp_path / 'results')


def test_results_fetch_does_not_modify_nonempty_destination(tmp_path):
    existing = tmp_path / 'keep.txt'
    existing.write_text('caller data')
    with pytest.raises(ValueError, match='must be empty'):
        download_results(tmp_path)
    assert existing.read_text() == 'caller data'
    assert not (tmp_path / '.git').exists()


def test_results_fetch_capture_records_verified_checkout(
    results_git_remote, tmp_path, monkeypatch
):
    _, pinned, _ = results_git_remote
    sink = adapter.raw_capture.RawSink(tmp_path / 'capture')
    monkeypatch.setattr(adapter.raw_capture, 'active_sink', lambda: sink)
    monkeypatch.setattr(
        adapter.raw_capture,
        'record_hf_dataset',
        lambda *args, **kwargs: pytest.fail('unexpected Hub lookup'),
    )

    download_results(tmp_path / 'results')

    (entry,) = sink.entries()
    assert entry['kind'] == 'pointer'
    assert entry['pointer_kind'] == 'git'
    assert entry['reference'] == adapter.RESULTS_URL
    assert entry['revision'] == pinned
    assert not sink.degraded


@pytest.mark.parametrize('wrong_step', ['fetch', 'checkout'])
def test_results_fetch_rejects_wrong_commit(
    results_git_remote, tmp_path, monkeypatch, wrong_step
):
    _, _, latest = results_git_remote
    actual_git = adapter._git

    def wrong_revision(*args, **kwargs):
        result = actual_git(*args, **kwargs)
        target = (
            'FETCH_HEAD^{commit}' if wrong_step == 'fetch' else 'HEAD^{commit}'
        )
        return latest if args == ('rev-parse', target) else result

    monkeypatch.setattr(adapter, '_git', wrong_revision)
    with pytest.raises(RuntimeError, match=f'commit {latest}, expected'):
        download_results(tmp_path / 'results')


def test_results_fetch_failure_has_no_hub_fallback(tmp_path, monkeypatch):
    commands = []

    def failed_fetch(args, **kwargs):
        commands.append(args)
        if args[1] == 'fetch':
            raise subprocess.CalledProcessError(
                128, args, stderr='Pinned commit is unavailable'
            )
        return subprocess.CompletedProcess(args, 0, stdout='')

    monkeypatch.setattr(adapter.subprocess, 'run', failed_fetch)
    with pytest.raises(RuntimeError, match='Pinned commit is unavailable'):
        download_results(tmp_path / 'results')
    assert [command[1] for command in commands] == ['init', 'fetch']


@pytest.mark.parametrize('fail_indexing', [False, True])
def test_main_cleans_temporary_results_checkout(
    results_git_remote, tmp_path, monkeypatch, fail_indexing
):
    workspaces = []
    actual_download = download_results
    actual_index = build_dump_index

    def downloaded(destination):
        workspaces.append(Path(destination))
        return actual_download(destination)

    def index_results(root):
        if fail_indexing:
            raise ValueError('Cannot index results')
        return actual_index(root)

    monkeypatch.setattr(adapter, 'download_results', downloaded)
    monkeypatch.setattr(adapter, 'build_dump_index', index_results)
    monkeypatch.setattr(adapter, 'fetch_json', lambda url: [])
    monkeypatch.setattr(adapter, 'process_models', lambda *args: 0)
    argv = ['--source-api', '--output-dir', str(tmp_path / 'outputs')]
    if fail_indexing:
        with pytest.raises(ValueError, match='Cannot index results'):
            adapter.main(argv)
    else:
        assert adapter.main(argv) == 0
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_stated_model_details_are_kept_with_the_library_version():
    source = _row('example/model', 'float16')
    source['model']['architecture'] = 'LlamaForCausalLM'
    source['metadata']['params_billions'] = 4.65

    log = convert_model(source, '1234')

    details = log.model_info.additional_details
    assert details['precision'] == 'float16'
    assert details['architecture'] == 'LlamaForCausalLM'
    assert details['params_billions'] == '4.65'
    assert log.eval_library.name == 'lm-evaluation-harness'
    assert log.eval_library.version == '0.4.0'


def test_split_does_not_depend_on_the_harness_dataset_path(tmp_path):
    rel = f'{DBRX}/results_2025-02-13T18-27-04.338360.json'
    root = _copy_results(tmp_path, rel)
    dump = json.loads((root / rel).read_text())
    dump['configs']['leaderboard_math_algebra_hard']['dataset_path'] = 'x/y'
    (root / rel).write_text(json.dumps(dump))

    (log,) = _convert([_row(DBRX, 'float16')], root)

    math = _results(log)['MATH Level 5'].source_data
    assert math.hf_repo == 'DigitalLearningGmbH/MATH-lighteval'
    assert math.hf_split == 'test'
    assert math.additional_details == {'dataset_size': '1324'}


CONTENTS = (
    Path(__file__).parent / 'data' / 'hfopenllm_v2' / 'contents' / CONTENTS_FILE
)


def test_contents_row_has_the_api_row_shape():
    row = contents_row(
        {
            'fullname': 'org/model',
            'Precision': 'float16',
            'Architecture': 'LlamaForCausalLM',
            '#Params (B)': float('nan'),
            'IFEval Raw': 0.5,
            'IFEval': 50.0,
            'MATH Lvl 5 Raw': None,
        }
    )

    assert row['model'] == {
        'name': 'org/model',
        'precision': 'float16',
        'architecture': 'LlamaForCausalLM',
    }
    assert row['metadata'] == {}
    assert row['evaluations']['ifeval'] == {'name': 'IFEval', 'value': 0.5}
    assert row['evaluations']['math'] == {
        'name': 'MATH Level 5',
        'value': None,
    }
    assert set(row['evaluations']) == {
        'ifeval',
        'bbh',
        'math',
        'gpqa',
        'musr',
        'mmlu_pro',
    }


def test_contents_parquet_rows_convert():
    pytest.importorskip('pyarrow')
    from every_eval_ever.adapters.hfopenllm_v2.contents import load_contents

    rows = load_contents(CONTENTS)

    by_key = {(r['model']['name'], r['model']['precision']): r for r in rows}
    assert set(by_key) == {
        (COGITO, 'bfloat16'),
        (COGITO, 'float16'),
        (DBRX, 'float16'),
        ('gpt2', 'float16'),
    }
    dbrx = by_key[(DBRX, 'float16')]
    assert dbrx['model']['architecture'] == 'Unknown'
    assert dbrx['metadata']['params_billions'] == 0.0
    assert 0 <= dbrx['evaluations']['math']['value'] <= 1


def test_main_reads_pinned_contents_and_results(tmp_path):
    pytest.importorskip('pyarrow')
    contents = tmp_path / 'contents' / CONTENTS_REVISION / CONTENTS_FILE
    contents.parent.mkdir(parents=True)
    shutil.copy(CONTENTS, contents)
    results = _snapshot(tmp_path, RESULTS_REVISION)
    output_dir = tmp_path / 'data' / 'hfopenllm_v2'

    with pytest.raises(SourceRecordsError, match='gpt2'):
        adapter.main(
            [
                '--output-dir',
                str(output_dir),
                '--contents-parquet',
                str(contents),
                '--results-dir',
                str(results),
            ]
        )

    outputs = sorted(output_dir.glob('*/*/*.json'))
    assert len(outputs) == 3
    for path in outputs:
        assert validate_file(path).valid
    report = json.loads(
        (
            tmp_path / 'adapter_reports' / 'hfopenllm_v2_failures.json'
        ).read_text()
    )
    assert report['leaderboard_input'] == {
        'dataset': 'open-llm-leaderboard/contents',
        'revision': CONTENTS_REVISION,
        'unpinned': False,
    }
    assert report['failed_record_count'] == 1


def test_contents_parquet_pinning(tmp_path):
    other = tmp_path / OTHER_SHA / CONTENTS_FILE

    assert contents_parquet_revision(
        tmp_path / CONTENTS_REVISION / CONTENTS_FILE
    ) == (CONTENTS_REVISION, False)
    with pytest.raises(ValueError) as exc:
        contents_parquet_revision(other)
    assert OTHER_SHA in str(exc.value)
    assert CONTENTS_REVISION in str(exc.value)
    assert '--allow-unpinned-contents-parquet' in str(exc.value)
    assert contents_parquet_revision(other, allow_unpinned=True) == (
        OTHER_SHA,
        True,
    )
    assert contents_parquet_revision(CONTENTS, allow_unpinned=True) == (
        'local',
        True,
    )
    with pytest.raises(SystemExit):
        adapter.main(['--contents-parquet', str(CONTENTS)])
