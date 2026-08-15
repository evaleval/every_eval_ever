import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.bountybench import adapter
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _log(
    task='lunary',
    bounty='0',
    model='openai/gpt-4o',
    workflow='DetectWorkflow',
    success=True,
    complete=True,
    start='2026-03-26T10:00:00.000000',
    end='2026-03-26T10:04:00.000000',
    max_iterations=10,
    actions=None,
    temperature=0.5,
):
    if actions is None:
        actions = [
            {'resource_id': 'model', 'command': 'ls -la'},
            {
                'resource_id': 'kali_env',
                'command': 'ls -la',
                'message': 'total 4\nsrc\n',
            },
        ]
    phase_messages = []
    if max_iterations:
        phase_messages = [
            {
                'max_iterations': max_iterations,
                'agent_messages': [
                    {
                        'agent_id': 'system',
                        'message': f'Find the bug in {task}.',
                    },
                    {
                        'agent_id': 'model',
                        'message': 'Looking at the source.',
                        'action_messages': actions,
                    },
                ],
            }
        ]
    return {
        'start_time': start,
        'end_time': end,
        'workflow_metadata': {
            'workflow_name': workflow,
            'task': {
                'task_dir': f'bountytasks/{task}',
                'bounty_number': bounty,
            },
            'workflow_summary': {'success': success, 'complete': complete},
        },
        'workflow_usage': {
            'total_input_tokens': 1200,
            'total_output_tokens': 340,
            'total_query_time_taken_in_ms': 5100,
        },
        'resources_used': {
            'model': {
                'config': {
                    'model': model,
                    'temperature': temperature,
                    'max_output_tokens': 4096,
                    'max_input_tokens': 100000,
                    'helm': False,
                }
            }
        },
        'additional_metadata': {
            'bounty_metadata': {
                'CVE': 'CVE-2024-0001',
                'severity': 7.5,
                'CWE': 'CWE-79',
            }
        },
        'phase_messages': phase_messages,
    }


def _write(logs_dir: Path, payloads, names=None):
    logs_dir.mkdir(parents=True, exist_ok=True)
    names = names or [f'log_{index}.json' for index in range(len(payloads))]
    for name, payload in zip(names, payloads):
        (logs_dir / name).write_text(json.dumps(payload))
    return logs_dir


def _args(tmp_path: Path, logs_dir: Path, **overrides):
    argv = [
        '--logs-dir',
        str(logs_dir),
        '--output-dir',
        str(tmp_path / 'data' / 'bountybench'),
        '--source-org',
        'Test Org',
        '--retrieved-timestamp',
        '1774000000.0',
    ]
    for key, value in overrides.items():
        flag = '--' + key.replace('_', '-')
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return adapter.build_parser().parse_args(argv)


def test_published_records_validate(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(task='lunary', bounty='0', success=True),
            _log(
                task='gradio',
                bounty='0',
                success=False,
                start='2026-03-26T10:05:00.000000',
                end='2026-03-26T10:09:00.000000',
            ),
        ],
    )

    assert adapter.run(_args(tmp_path, logs_dir)) == 1

    published = sorted((tmp_path / 'data').rglob('*'))
    files = [path for path in published if path.is_file()]
    assert len(files) == 2
    for path in files:
        report = validate_file(path)
        assert report.valid, report.errors

    aggregate = json.loads(
        next(path for path in files if path.suffix == '.json').read_text()
    )
    result = aggregate['evaluation_results'][0]
    assert result['score_details']['score'] == pytest.approx(0.5)
    assert result['metric_config']['metric_id'] == 'accuracy'
    assert result['evaluation_result_id'] == 'detect'
    assert result['evaluation_name'] == 'bountybench.detect'
    assert aggregate['model_info']['additional_details'] == {
        'deployment_type': 'externally_managed',
        'model_availability': 'unknown',
    }


def test_companion_path_is_the_full_repository_path(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log()])
    adapter.run(_args(tmp_path, logs_dir))

    aggregate = json.loads(
        next((tmp_path / 'data').rglob('*.json')).read_text()
    )
    detailed = aggregate['detailed_evaluation_results']
    samples = next((tmp_path / 'data').rglob('*_samples.jsonl'))
    assert detailed['file_path'] == (
        f'data/bountybench/openai/gpt-4o/{samples.name}'
    )
    assert detailed['total_rows'] == 1
    assert detailed['hash_algorithm'] == 'sha256'


def test_workflows_of_one_model_get_distinct_evaluation_ids(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(workflow='DetectWorkflow'),
            _log(workflow='ExploitWorkflow'),
            _log(workflow='PatchWorkflow'),
        ],
    )
    logs, _kept, _attempts, _result = adapter.convert(_args(tmp_path, logs_dir))

    assert len(logs) == 3
    assert len({log.evaluation_id for log in logs}) == 3
    assert {log.evaluation_results[0].evaluation_result_id for log in logs} == {
        'detect',
        'exploit',
        'patch',
    }


def test_differing_configurations_are_not_averaged_together(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(bounty='0', temperature=0.0),
            _log(bounty='1', temperature=1.0),
        ],
    )
    logs, _kept, _attempts, _result = adapter.convert(_args(tmp_path, logs_dir))

    assert len(logs) == 2
    fingerprints = {
        log.evaluation_results[0].metric_config.additional_details[
            'config_fingerprint'
        ]
        for log in logs
    }
    assert len(fingerprints) == 2


def test_repeated_attempts_are_rejected_by_default(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs', [_log(success=True), _log(success=False)]
    )

    with pytest.raises(SystemExit, match='more than one attempt'):
        adapter.run(_args(tmp_path, logs_dir))

    assert not (tmp_path / 'data').exists()


def test_best_attempt_policy_discloses_the_selection(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs', [_log(success=False), _log(success=True)]
    )
    logs, _kept, _attempts, _result = adapter.convert(
        _args(tmp_path, logs_dir, attempt_policy='best')
    )

    assert len(logs) == 1
    result = logs[0].evaluation_results[0]
    assert result.score_details.score == pytest.approx(1.0)
    details = result.metric_config.additional_details
    assert details['attempt_selection'] == 'best'
    assert details['n_attempts_total'] == '2'
    assert details['n_bounties_with_multiple_attempts'] == '1'
    assert result.score_details.details['total'] == '1'
    assert result.generation_config.generation_args.max_attempts == 2


def test_each_execution_yields_one_paired_tool_call(tmp_path: Path):
    actions = [
        {'resource_id': 'model', 'command': 'ls -la'},
        {'resource_id': 'kali_env', 'command': 'ls -la', 'message': 'src\n'},
        {'resource_id': 'model', 'command': 'ls -la'},
        {'resource_id': 'kali_env', 'command': 'ls -la', 'message': 'src\n'},
    ]
    messages = adapter.build_messages_from_phases(
        _log(actions=actions)['phase_messages']
    )

    calls = [call for message in messages for call in message.tool_calls or []]
    results = [message for message in messages if message.role == 'tool']
    assert [call.id for call in calls] == ['call_0', 'call_1']
    assert [message.tool_call_id for message in results] == [
        ['call_0'],
        ['call_1'],
    ]
    assert all(call.name == 'bash' for call in calls)
    assert [message.turn_idx for message in messages] == list(
        range(len(messages))
    )


def test_tool_result_without_a_call_has_no_dangling_reference(tmp_path: Path):
    messages = adapter.build_messages_from_phases(
        [
            {
                'max_iterations': 5,
                'agent_messages': [
                    {
                        'agent_id': 'model',
                        'message': 'done',
                        'action_messages': [
                            {
                                'resource_id': 'kali_env',
                                'message': 'orphaned output',
                            },
                        ],
                    }
                ],
            }
        ]
    )

    result = next(message for message in messages if message.role == 'tool')
    assert result.tool_call_id is None


def test_long_tool_output_records_its_truncation(tmp_path: Path):
    actions = [
        {'resource_id': 'model', 'command': 'cat big'},
        {
            'resource_id': 'kali_env',
            'command': 'cat big',
            'message': 'x' * (adapter.TOOL_OUTPUT_CHAR_LIMIT + 25),
        },
    ]
    messages = adapter.build_messages_from_phases(
        _log(actions=actions)['phase_messages']
    )

    content = next(
        message for message in messages if message.role == 'tool'
    ).content
    assert content.endswith('[truncated 25 characters]')


def test_sample_hash_ignores_the_outcome(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(model='openai/gpt-4o', success=True),
            _log(model='anthropic/claude-opus-4', success=False),
        ],
    )
    logs, kept, attempts, _result = adapter.convert(_args(tmp_path, logs_dir))

    instances = [
        adapter.build_instance_level(group[0], log, 1)
        for log, group in zip(logs, kept)
    ]
    assert len({instance.sample_hash for instance in instances}) == 1
    assert {instance.sample_id for instance in instances} == {'lunary_0'}
    assert all(instance.input.reference == [] for instance in instances)
    assert all(instance.answer_attribution == [] for instance in instances)


def test_naive_timestamps_are_read_in_the_declared_source_zone(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log()])

    utc = adapter.convert(_args(tmp_path, logs_dir))[0][0]
    shifted = adapter.convert(
        _args(tmp_path, logs_dir, source_timezone='America/New_York')
    )[0][0]

    assert utc.evaluation_timestamp != shifted.evaluation_timestamp
    assert (
        float(shifted.evaluation_timestamp) - float(utc.evaluation_timestamp)
        == 4 * 3600
    )


def test_startup_failures_are_excluded_without_failing_the_run(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(bounty='0'),
            _log(bounty='1', max_iterations=0),
        ],
    )
    logs, _kept, _attempts, result = adapter.convert(_args(tmp_path, logs_dir))

    assert logs[0].evaluation_results[0].score_details.score == 1.0
    assert logs[0].evaluation_results[0].score_details.details['total'] == '1'
    assert len(result.exclusions) == 1
    assert 'startup failure' in result.exclusions[0].reason
    result.raise_if_incomplete()


def test_unparseable_logs_stop_the_run_before_any_write(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log()])
    (logs_dir / 'broken.json').write_text('{ not json')

    with pytest.raises(SystemExit, match='could not be parsed'):
        adapter.run(_args(tmp_path, logs_dir))

    assert not (tmp_path / 'data').exists()


def test_allow_partial_publishes_the_rest_and_reports_the_failure(
    tmp_path: Path,
):
    logs_dir = _write(tmp_path / 'logs', [_log()])
    (logs_dir / 'broken.json').write_text('{ not json')

    with pytest.raises(SourceRecordsError):
        adapter.run(_args(tmp_path, logs_dir, allow_partial=True))

    assert len(list((tmp_path / 'data').rglob('*.json'))) == 1
    report = json.loads(
        (tmp_path / 'adapter_reports' / 'bountybench_failures.json').read_text()
    )
    assert report['failed_record_count'] == 1
    assert report['converted_records'] == 1
    assert 'JSONDecodeError' in report['failed_records'][0]['reason']


def test_a_model_without_a_developer_prefix_is_not_published_as_unknown(
    tmp_path: Path,
):
    payload = _log()
    payload['resources_used']['model']['config'].pop('model')
    logs_dir = _write(
        tmp_path / 'logs', [payload], names=['claude-code_x.json']
    )

    with pytest.raises(SystemExit, match='no developer prefix'):
        adapter.run(_args(tmp_path, logs_dir))

    logs, _kept, _attempts, _result = adapter.convert(
        _args(tmp_path, logs_dir, model_developer='anthropic')
    )
    assert logs[0].model_info.id == 'anthropic/claude-code'
    assert logs[0].model_info.developer == 'anthropic'


def test_a_traversing_model_identity_cannot_escape_the_output_dir(
    tmp_path: Path,
):
    logs_dir = _write(tmp_path / 'logs', [_log(model='../../etc/passwd')])
    args = _args(tmp_path, logs_dir)

    _parsed, failures, _total = adapter.collect_logs(
        logs_dir, args.source_timezone, None
    )
    assert 'path component' in failures[0].reason

    with pytest.raises(SystemExit, match='could not be parsed'):
        adapter.run(args)
    assert not (tmp_path / 'data').exists()


def test_a_run_with_nothing_convertible_fails_loudly(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log(max_iterations=0)])

    with pytest.raises(SystemExit, match='no convertible logs'):
        adapter.run(_args(tmp_path, logs_dir))


def test_an_unknown_workflow_is_rejected_rather_than_slugged(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log(workflow='MysteryWorkflow')])
    args = _args(tmp_path, logs_dir)

    _parsed, failures, _total = adapter.collect_logs(
        logs_dir, args.source_timezone, None
    )
    assert 'unknown workflow' in failures[0].reason

    with pytest.raises(SystemExit, match='could not be parsed'):
        adapter.run(args)
    assert not (tmp_path / 'data').exists()


def test_differing_iteration_budgets_are_not_averaged_together(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(bounty='0', max_iterations=10),
            _log(bounty='1', max_iterations=40),
        ],
    )
    logs, _kept, _attempts, _result = adapter.convert(_args(tmp_path, logs_dir))

    assert len(logs) == 2
    fingerprints = {
        log.evaluation_results[0].metric_config.additional_details[
            'config_fingerprint'
        ]
        for log in logs
    }
    assert len(fingerprints) == 2
    message_limits = {
        log.evaluation_results[
            0
        ].generation_config.generation_args.eval_limits.message_limit
        for log in logs
    }
    assert message_limits == {10, 40}


def test_concurrent_model_commands_keep_distinct_paired_calls(tmp_path: Path):
    # Two model commands open before either result arrives; each result must
    # pair to its own call, with no overwritten, orphaned, or duplicated call.
    actions = [
        {'resource_id': 'model', 'command': 'cat a'},
        {'resource_id': 'model', 'command': 'cat b'},
        {'resource_id': 'kali_env', 'command': 'cat a', 'message': 'A\n'},
        {'resource_id': 'kali_env', 'command': 'cat b', 'message': 'B\n'},
    ]
    messages = adapter.build_messages_from_phases(
        _log(actions=actions)['phase_messages']
    )

    calls = [call for message in messages for call in message.tool_calls or []]
    results = [message for message in messages if message.role == 'tool']
    assert [call.id for call in calls] == ['call_0', 'call_1']
    assert [call.arguments['command'] for call in calls] == ['cat a', 'cat b']
    assert [message.tool_call_id for message in results] == [
        ['call_0'],
        ['call_1'],
    ]
    assert [message.content for message in results] == ['A\n', 'B\n']


def test_best_attempt_ties_break_toward_the_earliest_start(tmp_path: Path):
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(
                bounty='0',
                success=True,
                complete=True,
                start='2026-03-26T12:00:00.000000',
                end='2026-03-26T12:04:00.000000',
            ),
            _log(
                bounty='0',
                success=True,
                complete=True,
                start='2026-03-26T10:00:00.000000',
                end='2026-03-26T10:04:00.000000',
            ),
        ],
    )
    _logs, kept, _attempts, _result = adapter.convert(
        _args(tmp_path, logs_dir, attempt_policy='best')
    )

    assert len(kept[0]) == 1
    assert kept[0][0]['start_time'] == '2026-03-26T10:00:00.000000'


def test_all_startup_config_records_each_log_once(tmp_path: Path):
    # Two startup attempts under one configuration: each is excluded exactly
    # once, with no extra synthetic group-level exclusion doubling the count.
    logs_dir = _write(
        tmp_path / 'logs',
        [
            _log(bounty='0', max_iterations=0),
            _log(bounty='1', max_iterations=0),
        ],
    )
    logs, _kept, _attempts, result = adapter.convert(_args(tmp_path, logs_dir))

    assert logs == []
    assert len(result.exclusions) == 2
    assert all(
        'startup failure' in exclusion.reason for exclusion in result.exclusions
    )


def test_a_log_without_start_time_is_a_recorded_failure(tmp_path: Path):
    payload = _log()
    payload.pop('start_time')
    logs_dir = _write(tmp_path / 'logs', [payload])
    args = _args(tmp_path, logs_dir)

    _parsed, failures, _total = adapter.collect_logs(
        logs_dir, args.source_timezone, None
    )
    assert 'start_time' in failures[0].reason

    with pytest.raises(SystemExit, match='could not be parsed'):
        adapter.run(args)
    assert not (tmp_path / 'data').exists()


def test_output_dir_must_end_in_the_collection_name(tmp_path: Path):
    logs_dir = _write(tmp_path / 'logs', [_log()])
    args = _args(tmp_path, logs_dir)
    args.output_dir = tmp_path / 'data' / 'not-bountybench'

    with pytest.raises(SystemExit, match="must end in 'bountybench'"):
        adapter.run(args)
    assert not (tmp_path / 'data').exists()

    args.dry_run = True
    adapter.run(args)
