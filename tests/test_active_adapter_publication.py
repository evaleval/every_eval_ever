from __future__ import annotations

from pathlib import Path

from every_eval_ever.helpers import save_evaluation_logs
from every_eval_ever.validate import validate_file
from utils.bfcl import adapter as bfcl
from utils.cocoabench import adapter as cocoabench
from utils.helm import adapter as helm
from utils.sciarena import adapter as sciarena


def assert_saved_records_are_valid(records) -> list[Path]:
    paths = save_evaluation_logs(records)
    assert paths
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors
    return paths


def test_bfcl_keeps_valid_rows_when_another_row_is_malformed(
    tmp_path: Path,
):
    valid = {spec.column: '' for spec in bfcl.METRIC_SPECS}
    valid.update(
        {
            'Model': 'GPT-5',
            'Organization': 'OpenAI',
            'Rank': '1',
            'Overall Acc': '50',
        }
    )
    for spec in bfcl.METRIC_SPECS:
        if spec.use_observed_max:
            valid[spec.column] = '1'
    invalid = valid | {'Overall Acc': 'nan'}

    bounds = bfcl.compute_observed_max_scores([valid, invalid])
    result = bfcl.convert_rows(
        [valid, invalid], tmp_path / 'data', bounds, '123'
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    paths = assert_saved_records_are_valid(result.records)
    assert paths[0].is_relative_to(tmp_path / 'data' / 'bfcl')


def test_sciarena_keeps_valid_rows_when_provider_is_unmapped(
    tmp_path: Path,
):
    valid = {
        'modelId': 'GPT-5',
        'rating': '1200',
        'rank': '1',
        'num_battles': '20',
        'rating_q025': '1100',
        'rating_q975': '1300',
        'variance': '2',
    }
    invalid = valid | {'modelId': 'Unmapped Model'}

    bounds = sciarena.compute_metric_bounds([valid, invalid])
    result = sciarena.convert_rows(
        [valid, invalid], tmp_path / 'data', bounds, '123'
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    paths = assert_saved_records_are_valid(result.records)
    assert paths[0].is_relative_to(tmp_path / 'data' / 'sciarena')


def test_cocoabench_keeps_valid_rows_when_row_map_entry_is_missing(
    tmp_path: Path,
):
    valid = {
        'Agent': 'CodeX',
        'AccuracyPercent': '50',
        'Answered': '10',
        'Correct': '5',
        'Wrong': '5',
        'AvgTime_s': '1.5',
        'AvgCost_USD': '0.1',
        'TotalCost_USD': '1',
    }
    invalid = valid | {'Agent': 'Not mapped'}
    output_dir = tmp_path / 'data' / 'cocoabench'

    result = cocoabench.convert_rows(
        [valid, invalid],
        cocoabench.DEFAULT_ROW_MAP,
        output_dir=output_dir,
        bounds=cocoabench.compute_metric_bounds([valid, invalid]),
        benchmark_version='1',
        eval_library_version='unknown',
        public_source_urls=[],
        benchmark_reference_urls=cocoabench.DEFAULT_BENCHMARK_REFERENCE_URLS,
        source_metadata_details={'benchmark_version': '1'},
        retrieved_timestamp='123',
        evaluation_timestamp=None,
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    paths = assert_saved_records_are_valid(result.records)
    assert paths[0].is_relative_to(output_dir)


def test_helm_keeps_valid_models_when_identity_is_unknown(tmp_path: Path):
    leaderboard = [
        {
            'title': 'accuracy',
            'header': [
                {'value': 'Model'},
                {'value': 'MMLU - EM', 'lower_is_better': False},
            ],
            'rows': [
                [{'value': 'gpt-4o'}, {'value': 0.5}],
                [{'value': 'unmapped-model'}, {'value': 0.4}],
                [
                    {
                        'value': 'nested-model',
                        'run_spec_names': ['run:model=org_family_model'],
                    },
                    {'value': 0.3},
                ],
            ],
        }
    ]

    result = helm.convert(
        'HELM_Lite',
        leaderboard,
        source_data_url='https://example.com/results.json',
        output_dir=str(tmp_path / 'data'),
    )

    assert len(result.records) == 2
    assert len(result.failures) == 1
    nested = next(
        record
        for record in result.records
        if record.eval_log.model_info.id == 'org/family/model'
    )
    assert nested.developer == 'org'
    assert nested.model_name == 'family_model'
    paths = assert_saved_records_are_valid(result.records)
    assert all(
        path.is_relative_to(tmp_path / 'data' / 'HELM_Lite') for path in paths
    )


def test_helm_records_bad_headers_and_missing_cells_without_fake_scores(
    tmp_path: Path,
):
    leaderboard = [
        {
            'title': 'accuracy',
            'header': [
                {'value': 'Model'},
                {'value': 'MMLU - EM', 'lower_is_better': False},
                {'value': 'MalformedHeader', 'lower_is_better': False},
            ],
            'rows': [
                [
                    {'value': 'gpt-4o'},
                    {'value': 0.5},
                    {'value': 0.4},
                ],
                [{'value': 'claude-3-opus'}, {'value': 0.6}],
            ],
        }
    ]

    result = helm.convert(
        'HELM_Lite',
        leaderboard,
        source_data_url='https://example.com/results.json',
        output_dir=str(tmp_path / 'data'),
    )

    assert len(result.records) == 2
    assert len(result.failures) == 2
    assert any('metric header' in failure.reason for failure in result.failures)
    assert any(
        'missing the metric cell' in failure.reason
        for failure in result.failures
    )
    for output in result.records:
        assert all(
            result.score_details.score != -1
            for result in output.eval_log.evaluation_results
        )
