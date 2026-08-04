from __future__ import annotations

import copy
import json

import pytest

from every_eval_ever.adapters.livecodebenchpro import (
    adapter as livecodebenchpro,
)
from every_eval_ever.adapters.rewardbench import migrate_to_v030 as rewardbench
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.schema import get_schema_version
from every_eval_ever.validator.validation_core import (
    check_model_deployment,
    check_score_metadata,
)


def _legacy_record(*, evaluation_id: str, score: float) -> dict:
    return {
        'schema_version': '0.2.0',
        'evaluation_id': evaluation_id,
        'retrieved_timestamp': '1234567890',
        'source_metadata': {
            'source_type': 'documentation',
            'source_organization_name': 'Test Org',
            'evaluator_relationship': 'third_party',
        },
        'source_data': {'legacy': True},
        'model_info': {
            'name': 'Test Model',
            'id': 'test-org/test-model',
            'inference_platform': 'unknown',
        },
        'eval_library': {'name': 'unknown', 'version': 'unknown'},
        'evaluation_results': [
            {
                'evaluation_name': 'Hard Problems',
                'metric_config': {'lower_is_better': False},
                'score_details': {'score': score},
            }
        ],
    }


def _assert_current_and_valid(data: dict) -> None:
    assert data['schema_version'] == get_schema_version()
    EvaluationLog.model_validate(data)
    assert check_score_metadata(data) == []
    assert check_model_deployment(data) == []
    assert 'source_data' not in data


def test_rewardbench_migration_emits_current_schema(tmp_path):
    data = _legacy_record(evaluation_id='reward-bench/model/123', score=0.75)
    path = tmp_path / 'rewardbench.json'
    path.write_text(json.dumps(data), encoding='utf-8')

    assert rewardbench.migrate_file(path)
    migrated = json.loads(path.read_text(encoding='utf-8'))

    _assert_current_and_valid(migrated)
    result = migrated['evaluation_results'][0]
    assert result['source_data']['hf_repo'] == 'allenai/reward-bench'
    assert result['metric_config']['min_score'] == 0.0
    assert result['metric_config']['max_score'] == 1.0
    assert 'inference_platform' not in migrated['model_info']
    assert rewardbench.migrate_file(path) is False


@pytest.mark.parametrize(
    ('score', 'expected_max'), [(0.42, 1.0), (42.0, 100.0)]
)
def test_livecodebench_migration_emits_current_schema(
    tmp_path, score, expected_max
):
    data = _legacy_record(
        evaluation_id='livecodebenchpro/model/123', score=score
    )
    path = tmp_path / 'livecodebenchpro.json'
    path.write_text(json.dumps(data), encoding='utf-8')

    assert livecodebenchpro.migrate_file(path)
    migrated = json.loads(path.read_text(encoding='utf-8'))

    _assert_current_and_valid(migrated)
    result = migrated['evaluation_results'][0]
    assert result['source_data']['source_type'] == 'url'
    assert result['metric_config']['min_score'] == 0.0
    assert result['metric_config']['max_score'] == expected_max
    assert livecodebenchpro.migrate_file(path) is False


def test_migrations_reject_unknown_source_versions(tmp_path):
    data = _legacy_record(evaluation_id='reward-bench/model/123', score=0.5)
    data['schema_version'] = '9.9.9'
    original = copy.deepcopy(data)
    path = tmp_path / 'unknown.json'
    path.write_text(json.dumps(data), encoding='utf-8')

    with pytest.raises(ValueError, match='expected a legacy schema version'):
        rewardbench.migrate_file(path)

    assert json.loads(path.read_text(encoding='utf-8')) == original
