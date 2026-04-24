from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.upgrade_schema_version import (
    AGGREGATE_SCHEMA_VERSION,
    INSTANCE_SCHEMA_VERSION,
    upgrade_aggregate_file,
    upgrade_instance_file,
)


def test_upgrade_aggregate_file_rewrites_schema_version(tmp_path: Path):
    file_path = tmp_path / 'aggregate.json'
    payload = {
        'schema_version': '0.2.1',
        'evaluation_id': 'test/model/123',
        'retrieved_timestamp': '123',
        'source_metadata': {
            'source_type': 'evaluation_run',
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'first_party',
        },
        'eval_library': {'name': 'inspect_ai', 'version': '0.3.0'},
        'model_info': {'name': 'test-model', 'id': 'org/test-model'},
        'evaluation_results': [],
    }
    file_path.write_text(json.dumps(payload), encoding='utf-8')

    report = upgrade_aggregate_file(file_path, write_changes=True)
    updated = json.loads(file_path.read_text(encoding='utf-8'))

    assert report.changed is True
    assert updated['schema_version'] == AGGREGATE_SCHEMA_VERSION


def test_upgrade_instance_file_rewrites_all_rows(tmp_path: Path):
    file_path = tmp_path / 'samples.jsonl'
    rows = [
        {'schema_version': 'instance_level_eval_0.2.1', 'sample_id': 'a'},
        {'schema_version': '0.2.2', 'sample_id': 'b'},
        {'schema_version': INSTANCE_SCHEMA_VERSION, 'sample_id': 'c'},
    ]
    file_path.write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n',
        encoding='utf-8',
    )

    report = upgrade_instance_file(file_path, write_changes=True)
    updated = [
        json.loads(line)
        for line in file_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]

    assert report.changed is True
    assert report.rows_changed == 2
    assert all(
        row['schema_version'] == INSTANCE_SCHEMA_VERSION for row in updated
    )


def test_upgrade_dry_run_does_not_write(tmp_path: Path):
    file_path = tmp_path / 'aggregate.json'
    payload = {'schema_version': '0.2.1'}
    file_path.write_text(json.dumps(payload), encoding='utf-8')

    report = upgrade_aggregate_file(file_path, write_changes=False)
    unchanged = json.loads(file_path.read_text(encoding='utf-8'))

    assert report.changed is True
    assert unchanged['schema_version'] == '0.2.1'
