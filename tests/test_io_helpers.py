from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    SourceRecordsError,
    datastore_path_components,
    default_failure_report_path,
    generate_output_path,
    raise_for_failed_records,
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


def test_basic_output_path_replaces_colons_for_windows(tmp_path):
    path = generate_output_path(
        tmp_path,
        'developer',
        'model::revision',
    )

    assert path == tmp_path / 'developer' / 'model__revision'


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
