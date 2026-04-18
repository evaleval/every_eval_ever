from __future__ import annotations

import json
from pathlib import Path

from every_eval_ever.check_canonical_identity import check_paths, main


def _base_payload(evaluation_id: str, evaluation_results: list[dict]) -> dict:
    return {
        'schema_version': '0.2.2',
        'evaluation_id': evaluation_id,
        'retrieved_timestamp': '1234567890',
        'source_metadata': {
            'source_type': 'documentation',
            'source_organization_name': 'TestOrg',
            'evaluator_relationship': 'third_party',
        },
        'eval_library': {'name': 'unknown', 'version': 'unknown'},
        'model_info': {'name': 'test-model', 'id': 'org/test-model'},
        'evaluation_results': evaluation_results,
    }


def _result_with_identity(evaluation_name: str) -> dict:
    return {
        'evaluation_name': evaluation_name,
        'evaluation_result_id': f'test/model/123#{evaluation_name}#accuracy',
        'source_data': {
            'dataset_name': 'test-ds',
            'source_type': 'hf_dataset',
            'hf_repo': 'org/test-ds',
        },
        'metric_config': {
            'evaluation_description': 'accuracy',
            'metric_id': 'accuracy',
            'metric_name': 'Accuracy',
            'metric_kind': 'accuracy',
            'metric_unit': 'proportion',
            'lower_is_better': False,
            'score_type': 'continuous',
            'min_score': 0.0,
            'max_score': 1.0,
        },
        'score_details': {'score': 0.5},
    }


def test_check_paths_reports_missing_identity_fields(tmp_path: Path):
    data_dir = tmp_path / 'data' / 'demo-benchmark' / 'org' / 'model'
    data_dir.mkdir(parents=True)

    payload = _base_payload(
        'demo-benchmark/org_model/123',
        [
            {
                **_result_with_identity('slice_a'),
                'evaluation_result_id': '',
                'metric_config': {
                    'lower_is_better': False,
                    'score_type': 'continuous',
                    'min_score': 0.0,
                    'max_score': 1.0,
                },
            }
        ],
    )
    file_path = data_dir / 'run.json'
    file_path.write_text(json.dumps(payload), encoding='utf-8')

    report = check_paths([str(tmp_path)])
    as_dict = report.to_dict()

    assert report.files_scanned == 1
    assert report.results_scanned == 1
    assert as_dict['missing']['evaluation_result_id'] == 1
    assert as_dict['missing']['metric_id'] == 1
    assert as_dict['top_missing_by_benchmark']['metric_id'][0] == (
        'demo-benchmark',
        1,
    )


def test_check_paths_passes_for_complete_identity(tmp_path: Path):
    data_dir = tmp_path / 'data' / 'clean-benchmark' / 'org' / 'model'
    data_dir.mkdir(parents=True)

    payload = _base_payload(
        'clean-benchmark/org_model/123',
        [_result_with_identity('slice_a')],
    )
    file_path = data_dir / 'run.json'
    file_path.write_text(json.dumps(payload), encoding='utf-8')

    report = check_paths([str(tmp_path)])
    assert report.files_scanned == 1
    assert report.results_scanned == 1
    assert report.has_issues is False
    assert report.missing == {}
    assert report.malformed == {}


def test_main_fail_on_issues_returns_nonzero(tmp_path: Path):
    data_dir = tmp_path / 'data' / 'broken-benchmark' / 'org' / 'model'
    data_dir.mkdir(parents=True)

    payload = _base_payload(
        'broken-benchmark/org_model/123',
        [
            {
                **_result_with_identity('slice_a'),
                'evaluation_result_id': 'not_a_canonical_id',
                'metric_config': {
                    **_result_with_identity('slice_a')['metric_config'],
                    'metric_id': '',
                },
            }
        ],
    )
    file_path = data_dir / 'run.json'
    file_path.write_text(json.dumps(payload), encoding='utf-8')

    assert (
        main(
            [
                str(tmp_path),
                '--format',
                'json',
                '--fail-on-issues',
            ]
        )
        == 1
    )
