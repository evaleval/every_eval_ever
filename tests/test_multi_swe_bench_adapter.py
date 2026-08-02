import json

import yaml

from every_eval_ever.adapters.multi_swe_bench import adapter
from every_eval_ever.adapters.multi_swe_bench.adapter import convert_submissions


def _submission(root, name, total_instances):
    submission = root / name
    (submission / 'results').mkdir(parents=True)
    (submission / 'metadata.yaml').write_text(
        yaml.safe_dump({'name': name}),
        encoding='utf-8',
    )
    (submission / 'results' / 'results.json').write_text(
        json.dumps(
            {
                'total_instances': total_instances,
                'resolved': ['one'],
            }
        ),
        encoding='utf-8',
    )
    return submission


def test_bad_submission_does_not_discard_valid_submission(tmp_path):
    good = _submission(tmp_path, '20260101_gpt-5_agent', 2)
    bad = _submission(tmp_path, '20260101_gpt-5_broken', 0)

    result = convert_submissions(
        [(good, 'python'), (bad, 'python')],
        retrieved_timestamp='1234',
        output_dir=str(tmp_path / 'data' / 'multi-swe-bench'),
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record['submission_dir'] == str(bad)


def test_nested_model_identity_does_not_fail_publication_batch(
    tmp_path, monkeypatch
):
    good = _submission(tmp_path, '20260101_agent_gpt-5', 2)
    nested = _submission(tmp_path, '20260101_agent_gpt-5-nested', 2)
    original_get_developer = adapter.get_developer
    original_get_model_id = adapter.get_model_id

    def get_developer(model_name):
        if model_name == 'gpt-5-nested':
            return 'org'
        return original_get_developer(model_name)

    def get_model_id(model_name, developer=None):
        if model_name == 'gpt-5-nested':
            return 'org/family/model:revision'
        return original_get_model_id(model_name, developer)

    monkeypatch.setattr(adapter, 'get_developer', get_developer)
    monkeypatch.setattr(adapter, 'get_model_id', get_model_id)

    result = adapter.convert_submissions(
        [(good, 'python'), (nested, 'python')],
        retrieved_timestamp='1234',
        output_dir=str(tmp_path / 'data' / 'multi-swe-bench'),
    )

    assert not result.failures
    assert len(result.records) == 2
    nested_output = next(
        record
        for record in result.records
        if record.eval_log.model_info.id == 'org/family/model:revision'
    )
    assert nested_output.developer == 'org'
    assert nested_output.model_name == 'family_model_revision'
