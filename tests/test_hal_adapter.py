from every_eval_ever.adapters.hal.adapter import (
    BENCHMARK_BY_SLUG,
    _parse_cost,
    _parse_percent,
    build_eee_record,
    get_model_id,
    parse_table_result,
)


def test_numeric_parsers_handle_grouped_costs_and_reject_malformed_percent():
    assert _parse_cost('$1,234.56 (-9.17/+9.17)') == 1234.56
    assert _parse_percent('1.2.3%') is None


def _row(rank, model, accuracy, runs='1'):
    cells = [
        str(rank),
        'Example Agent',
        model,
        '✓',
        accuracy,
        '$1.00',
        runs,
    ]
    return '<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'


def test_missing_accuracy_is_reported_without_discarding_other_rows():
    benchmark = BENCHMARK_BY_SLUG['assistantbench']
    html = (
        '<table><tbody>'
        + _row(1, 'gpt-5', '50%')
        + _row(2, 'gpt-4.1', 'not available')
        + '</tbody></table>'
    )

    result = parse_table_result(html, benchmark)

    assert len(result.records) == 1
    assert result.records[0].model_raw == 'gpt-5'
    assert len(result.failures) == 1
    assert result.failures[0].source_record['cells'][2] == 'gpt-4.1'


def test_invalid_optional_runs_is_recorded_but_score_row_is_retained():
    benchmark = BENCHMARK_BY_SLUG['assistantbench']
    html = (
        '<table><tbody>'
        + _row(1, 'gpt-5', '50%', runs='many')
        + '</tbody></table>'
    )

    result = parse_table_result(html, benchmark)

    assert len(result.records) == 1
    assert result.records[0].runs is None
    assert len(result.failures) == 1
    assert 'invalid run count' in result.failures[0].reason


def test_model_directory_keeps_the_dots_the_model_id_uses():
    benchmark = BENCHMARK_BY_SLUG['assistantbench']
    html = '<table><tbody>' + _row(1, 'gpt-4.1', '50%') + '</tbody></table>'

    row = parse_table_result(html, benchmark).records[0]
    record, developer, model_slug = build_eee_record(benchmark, row, '0')

    assert record['model_info']['id'] == 'openai/gpt-4.1'
    assert (developer, model_slug) == ('openai', 'gpt-4.1')


def test_an_anthropic_release_is_addressed_the_way_anthropic_spells_it():
    assert get_model_id('Claude Haiku 4.5') == 'anthropic/claude-haiku-4-5'
