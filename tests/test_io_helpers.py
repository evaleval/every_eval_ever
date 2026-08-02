import pytest

from every_eval_ever.helpers.io import (
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    SourceRecordsError,
    datastore_path_components,
    datastore_repo_file_path,
    default_failure_report_path,
    generate_output_path,
    raise_for_failed_records,
    require_uuid4,
    save_failure_report,
)


def test_datastore_path_replaces_colons_for_windows():
    assert datastore_path_components(
        'benchmark::version',
        'developer/model:revision',
    ) == (
        'benchmark__version',
        'developer',
        'model_revision',
    )


def test_datastore_repo_file_path_is_canonical_and_portable():
    file_uuid = '123e4567-e89b-42d3-a456-426614174000'

    assert datastore_repo_file_path(
        'benchmark::version',
        'developer/family/model:revision',
        None,
        f'{file_uuid}_samples.jsonl',
    ) == (
        'data/benchmark__version/developer/family_model_revision/'
        f'{file_uuid}_samples.jsonl'
    )


def test_require_uuid4_rejects_non_rfc_variant():
    with pytest.raises(ValueError, match='UUIDv4'):
        require_uuid4('550e8400-e29b-41d4-0716-446655440000')


def test_basic_output_path_replaces_colons_for_windows(tmp_path):
    path = generate_output_path(
        tmp_path,
        'developer',
        'model::revision',
    )

    assert path == tmp_path / 'developer' / 'model__revision'


def test_evaluation_output_flattens_nested_model_path_and_colons():
    log = object()

    output = EvaluationLogOutput(
        eval_log=log,  # type: ignore[arg-type]
        base_dir='data/benchmark',
        developer='developer:team',
        model_name='family/model:revision',
    )

    assert output.developer == 'developer_team'
    assert output.model_name == 'family_model_revision'
    assert output.eval_log is log


def test_evaluation_output_rejects_empty_nested_model_component():
    with pytest.raises(ValueError, match='invalid model name'):
        EvaluationLogOutput(
            eval_log=object(),  # type: ignore[arg-type]
            base_dir='data/benchmark',
            developer='developer',
            model_name='family//model',
        )


def test_failed_source_records_retain_raw_provenance():
    raw_record = {'model': 'missing-developer', 'score': 0.5}
    failure = SourceRecordFailure(
        source_ref='source.json row 12',
        reason='model developer must be known',
        source_record=raw_record,
    )

    try:
        raise_for_failed_records('Example', 20, [failure])
    except SourceRecordsError as exc:
        assert exc.failures == [failure]
        assert exc.model_dump() == {
            'source_name': 'Example',
            'total_records': 20,
            'failed_records': [
                {
                    'source_ref': 'source.json row 12',
                    'reason': 'model developer must be known',
                    'source_record': raw_record,
                }
            ],
        }
    else:
        raise AssertionError('expected source record conversion to fail')


def test_failure_report_is_outside_validated_data_tree(tmp_path):
    output_dir = tmp_path / 'data' / 'benchmark'

    assert default_failure_report_path(output_dir) == (
        tmp_path / 'adapter_reports' / 'benchmark_failures.json'
    )


def test_failure_report_filename_replaces_colons_for_windows(tmp_path):
    output_dir = tmp_path / 'data' / 'benchmark:version'

    assert default_failure_report_path(output_dir) == (
        tmp_path / 'adapter_reports' / 'benchmark_version_failures.json'
    )


def test_intentional_exclusion_is_reported_but_not_a_failure():
    excluded = SourceRecordExclusion(
        source_ref='row 2',
        reason='published random baseline is not a model evaluation',
        source_record={'model': 'random'},
    )
    result = SourceConversionResult(
        source_name='Example',
        total_records=1,
        records=[],
        failures=[],
        exclusions=[excluded],
    )

    result.raise_if_incomplete()
    assert result.failure_report()['excluded_records'] == [
        excluded.model_dump()
    ]


def test_failure_report_preserves_nonfinite_source_values_as_strict_json(
    tmp_path,
):
    result = SourceConversionResult(
        source_name='Example',
        total_records=1,
        records=[],
        failures=[
            SourceRecordFailure(
                source_ref='row 1',
                reason='source has a non-finite score',
                source_record={'score': float('nan')},
            )
        ],
    )

    path = save_failure_report(result, tmp_path / 'report.json')

    assert path.read_text(encoding='utf-8').count('NaN') == 1
    assert path.read_text(encoding='utf-8').find('"score": {') >= 0
