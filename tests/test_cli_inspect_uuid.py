from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from every_eval_ever import cli
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
    SourceRecordsError,
)


def _make_inspect_args(log_path: Path, output_dir: Path) -> Namespace:
    return Namespace(
        log_path=str(log_path),
        output_dir=str(output_dir),
        source_organization_name='TestOrg',
        evaluator_relationship='third_party',
        source_organization_url=None,
        source_organization_logo_url=None,
        eval_library_name='inspect',
        eval_library_version='unknown',
    )


def _make_helm_args(log_path: Path, output_dir: Path) -> Namespace:
    return Namespace(
        log_path=str(log_path),
        output_dir=str(output_dir),
        source_organization_name='TestOrg',
        evaluator_relationship='third_party',
        source_organization_url=None,
        source_organization_logo_url=None,
        eval_library_name='helm',
        eval_library_version='unknown',
    )


def test_convert_inspect_file_mode_reuses_generated_uuid_for_aggregate_file(
    tmp_path, monkeypatch
):
    log_path = tmp_path / 'inspect_log.json'
    log_path.write_text('{}', encoding='utf-8')
    fake_log = SimpleNamespace()

    fake_module = ModuleType('every_eval_ever.converters.inspect.adapter')
    captured_metadata: dict[str, object] = {}

    class FakeInspectAdapter:
        def transform_from_file(self, _path, metadata_args):
            captured_metadata.update(metadata_args)
            return fake_log

        def transform_from_directory(self, *_args, **_kwargs):
            return [fake_log]

    fake_module.InspectAIAdapter = FakeInspectAdapter
    fake_module.list_eval_logs = lambda _path: []
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.inspect.adapter', fake_module
    )

    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    monkeypatch.setattr(cli.uuid, 'uuid4', lambda: uuid_value)

    captured_eval_uuids: list[str | None] = []

    def fake_publish(logs, _base_output, eval_uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        captured_eval_uuids.extend(eval_uuids)
        return [Path('/tmp/fake_aggregate.json') for _ in logs]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    rc = cli._cmd_convert_inspect(_make_inspect_args(log_path, tmp_path))

    assert rc == 0
    assert captured_metadata['file_uuid'] == uuid_value
    assert captured_eval_uuids == [uuid_value]


def test_convert_inspect_directory_mode_reuses_generated_uuids_for_aggregate_file(
    tmp_path, monkeypatch
):
    fake_log_1 = SimpleNamespace()
    fake_log_2 = SimpleNamespace()
    fake_logs = [fake_log_1, fake_log_2]

    fake_module = ModuleType('every_eval_ever.converters.inspect.adapter')
    captured_metadata: dict[str, object] = {}

    class FakeInspectAdapter:
        def transform_from_file(self, *_args, **_kwargs):
            return fake_log_1

        def transform_from_directory_result(self, _path, metadata_args):
            captured_metadata.update(metadata_args)
            return SourceConversionResult(
                source_name='fake Inspect logs',
                total_records=2,
                records=list(zip(fake_logs, metadata_args['file_uuids'])),
                failures=[],
            )

    fake_module.InspectAIAdapter = FakeInspectAdapter
    fake_module.list_eval_logs = lambda _path: [
        Path('/tmp/log_a.eval'),
        Path('/tmp/log_b.eval'),
    ]
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.inspect.adapter', fake_module
    )

    uuids = iter(
        [
            '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
            '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
        ]
    )
    monkeypatch.setattr(cli.uuid, 'uuid4', lambda: next(uuids))

    captured_eval_uuids: list[str | None] = []

    def fake_publish(logs, _base_output, eval_uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        captured_eval_uuids.extend(eval_uuids)
        return [Path('/tmp/fake_aggregate.json') for _ in logs]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    rc = cli._cmd_convert_inspect(_make_inspect_args(tmp_path, tmp_path))

    assert rc == 0
    assert captured_metadata['file_uuids'] == [
        '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
        '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
    ]
    assert captured_eval_uuids == [
        '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
        '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
    ]


def test_convert_helm_single_run_reuses_generated_uuid_for_aggregate_file(
    tmp_path, monkeypatch
):
    fake_log = SimpleNamespace()
    fake_module = ModuleType('every_eval_ever.converters.helm.adapter')
    captured_metadata: dict[str, object] = {}

    class FakeHELMAdapter:
        def _directory_contains_required_files(self, _path):
            return True

        def transform_from_directory_result(
            self, _dir_path, output_path=None, metadata_args=None
        ):
            _ = output_path
            captured_metadata.update(metadata_args)
            return SourceConversionResult(
                source_name='fake HELM runs',
                total_records=1,
                records=[(fake_log, metadata_args['file_uuid'])],
                failures=[],
            )

    fake_module.HELMAdapter = FakeHELMAdapter
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.helm.adapter', fake_module
    )

    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    monkeypatch.setattr(cli.uuid, 'uuid4', lambda: uuid_value)

    captured_eval_uuids: list[str | None] = []

    def fake_publish(logs, _base_output, eval_uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        captured_eval_uuids.extend(eval_uuids)
        return [Path('/tmp/fake_aggregate.json') for _ in logs]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    rc = cli._cmd_convert_helm(_make_helm_args(tmp_path, tmp_path))

    assert rc == 0
    assert captured_metadata['file_uuid'] == uuid_value
    assert captured_eval_uuids == [uuid_value]


def test_convert_helm_directory_mode_reuses_generated_uuids_for_aggregate_file(
    tmp_path, monkeypatch
):
    (tmp_path / 'run_a').mkdir()
    (tmp_path / 'run_b').mkdir()
    (tmp_path / 'other').mkdir()
    fake_module = ModuleType('every_eval_ever.converters.helm.adapter')
    captured_metadata: dict[str, object] = {}

    class FakeHELMAdapter:
        def _directory_contains_required_files(self, path):
            return Path(path).name in {'run_a', 'run_b'}

        def transform_from_directory_result(
            self, _dir_path, output_path=None, metadata_args=None
        ):
            _ = output_path
            captured_metadata.update(metadata_args)
            return SourceConversionResult(
                source_name='fake HELM runs',
                total_records=2,
                records=list(
                    zip(
                        [SimpleNamespace(), SimpleNamespace()],
                        metadata_args['file_uuids'],
                    )
                ),
                failures=[],
            )

    fake_module.HELMAdapter = FakeHELMAdapter
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.helm.adapter', fake_module
    )

    uuids = iter(
        [
            '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
            '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
        ]
    )
    monkeypatch.setattr(cli.uuid, 'uuid4', lambda: next(uuids))

    captured_eval_uuids: list[str | None] = []

    def fake_publish(logs, _base_output, eval_uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        captured_eval_uuids.extend(eval_uuids)
        return [Path('/tmp/fake_aggregate.json') for _ in logs]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    rc = cli._cmd_convert_helm(_make_helm_args(tmp_path, tmp_path))

    assert rc == 0
    assert captured_metadata['file_uuids'] == [
        '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
        '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
    ]
    assert captured_eval_uuids == [
        '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
        '2e4f2dc0-9882-4a6f-8dd9-fcb3f8b007fb',
    ]


def test_convert_inspect_directory_publishes_success_before_reporting_failure(
    tmp_path, monkeypatch
):
    fake_module = ModuleType('every_eval_ever.converters.inspect.adapter')
    successful_log = SimpleNamespace()

    class FakeInspectAdapter:
        def transform_from_directory_result(self, _path, metadata_args):
            return SourceConversionResult(
                source_name='fake Inspect logs',
                total_records=2,
                records=[(successful_log, metadata_args['file_uuids'][0])],
                failures=[
                    SourceRecordFailure(
                        source_ref='bad.eval',
                        reason='broken Inspect log',
                        source_record={'path': 'bad.eval'},
                    )
                ],
            )

    fake_module.InspectAIAdapter = FakeInspectAdapter
    fake_module.list_eval_logs = lambda _path: [
        Path('/tmp/good.eval'),
        Path('/tmp/bad.eval'),
    ]
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.inspect.adapter', fake_module
    )
    monkeypatch.setattr(
        cli.uuid,
        'uuid4',
        lambda: '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
    )
    published = []

    def fake_publish(logs, _output, uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        published.extend(zip(logs, uuids))
        return [Path('/tmp/fake_aggregate.json')]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    with pytest.raises(SourceRecordsError, match='broken Inspect log'):
        cli._cmd_convert_inspect(_make_inspect_args(tmp_path, tmp_path))

    assert published == [
        (successful_log, '5cd3f6ca-2fd0-4f88-8f19-9d53089641df')
    ]
    report = json.loads(
        (tmp_path / 'adapter_reports' / 'inspect_failures.json').read_text(
            encoding='utf-8'
        )
    )
    assert report['converted_records'] == 1
    assert report['failed_record_count'] == 1


def test_convert_helm_directory_publishes_success_before_reporting_failure(
    tmp_path, monkeypatch
):
    (tmp_path / 'good').mkdir()
    (tmp_path / 'bad').mkdir()
    fake_module = ModuleType('every_eval_ever.converters.helm.adapter')
    successful_log = SimpleNamespace()

    class FakeHELMAdapter:
        def _directory_contains_required_files(self, path):
            return Path(path).name in {'good', 'bad'}

        def transform_from_directory_result(
            self, _path, output_path=None, metadata_args=None
        ):
            _ = output_path
            return SourceConversionResult(
                source_name='fake HELM runs',
                total_records=2,
                records=[(successful_log, metadata_args['file_uuids'][0])],
                failures=[
                    SourceRecordFailure(
                        source_ref='bad',
                        reason='broken HELM run',
                        source_record={'path': 'bad'},
                    )
                ],
            )

    fake_module.HELMAdapter = FakeHELMAdapter
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.helm.adapter', fake_module
    )
    monkeypatch.setattr(
        cli.uuid,
        'uuid4',
        lambda: '5cd3f6ca-2fd0-4f88-8f19-9d53089641df',
    )
    published = []

    def fake_publish(logs, _output, uuids, *, staged_output_dir=None):
        _ = staged_output_dir
        published.extend(zip(logs, uuids))
        return [Path('/tmp/fake_aggregate.json')]

    monkeypatch.setattr(cli, 'publish_evaluation_logs', fake_publish)

    with pytest.raises(SourceRecordsError, match='broken HELM run'):
        cli._cmd_convert_helm(_make_helm_args(tmp_path, tmp_path))

    assert len(published) == 1
    report = json.loads(
        (tmp_path / 'adapter_reports' / 'helm_failures.json').read_text(
            encoding='utf-8'
        )
    )
    assert report['converted_records'] == 1
    assert report['failed_record_count'] == 1
