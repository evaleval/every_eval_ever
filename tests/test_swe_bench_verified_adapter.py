import json

import yaml

from every_eval_ever.adapters.swe_bench_verified.adapter import (
    convert_submissions,
)


def _submission(root, name, model):
    submission = root / name
    (submission / 'results').mkdir(parents=True)
    (submission / 'metadata.yaml').write_text(
        yaml.safe_dump(
            {
                'tags': {'model': [model]} if model is not None else {},
                'info': {},
            }
        ),
        encoding='utf-8',
    )
    (submission / 'results' / 'results.json').write_text(
        json.dumps({'resolved': ['one']}),
        encoding='utf-8',
    )
    return submission


def test_missing_model_does_not_discard_valid_submission(tmp_path):
    good = _submission(tmp_path, 'good', 'openai/gpt-5')
    bad = _submission(tmp_path, 'bad', None)

    result = convert_submissions(
        [good, bad],
        retrieved_timestamp='1234',
        total_instances=2,
        output_dir=str(tmp_path / 'data' / 'swe-bench'),
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record['submission_dir'] == str(bad)


def test_nested_model_identity_is_flattened_only_for_output_path(tmp_path):
    nested = _submission(tmp_path, 'nested', 'org/family/model:revision')

    result = convert_submissions(
        [nested],
        retrieved_timestamp='1234',
        total_instances=2,
        output_dir=str(tmp_path / 'data' / 'swe-bench'),
    )

    assert not result.failures
    assert len(result.records) == 1
    output = result.records[0]
    assert output.developer == 'org'
    assert output.model_name == 'family_model_revision'
    assert output.eval_log.model_info.id == 'org/family/model:revision'
