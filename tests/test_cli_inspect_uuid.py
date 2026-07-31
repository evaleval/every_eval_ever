from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

from every_eval_ever import cli


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

    def fake_write_log(_log, _base_output, eval_uuid=None, compression="none"):
        captured_eval_uuids.append(eval_uuid)
        return Path('/tmp/fake_aggregate.json')

    monkeypatch.setattr(cli, '_write_log', fake_write_log)

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

        def transform_from_directory(self, _path, metadata_args):
            captured_metadata.update(metadata_args)
            return fake_logs

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

    def fake_write_log(_log, _base_output, eval_uuid=None, compression="none"):
        captured_eval_uuids.append(eval_uuid)
        return Path('/tmp/fake_aggregate.json')

    monkeypatch.setattr(cli, '_write_log', fake_write_log)

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

        def transform_from_directory(
            self, _dir_path, output_path=None, metadata_args=None
        ):
            _ = output_path
            captured_metadata.update(metadata_args)
            return [fake_log]

    fake_module.HELMAdapter = FakeHELMAdapter
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.helm.adapter', fake_module
    )

    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    monkeypatch.setattr(cli.uuid, 'uuid4', lambda: uuid_value)

    captured_eval_uuids: list[str | None] = []

    def fake_write_log(_log, _base_output, eval_uuid=None, compression="none"):
        captured_eval_uuids.append(eval_uuid)
        return Path('/tmp/fake_aggregate.json')

    monkeypatch.setattr(cli, '_write_log', fake_write_log)

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

        def transform_from_directory(
            self, _dir_path, output_path=None, metadata_args=None
        ):
            _ = output_path
            captured_metadata.update(metadata_args)
            return [SimpleNamespace(), SimpleNamespace()]

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

    def fake_write_log(_log, _base_output, eval_uuid=None, compression="none"):
        captured_eval_uuids.append(eval_uuid)
        return Path('/tmp/fake_aggregate.json')

    monkeypatch.setattr(cli, '_write_log', fake_write_log)

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
