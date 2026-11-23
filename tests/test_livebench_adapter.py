import csv
import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.livebench import adapter
from every_eval_ever.helpers.eval_card_registry import Registry
from every_eval_ever.validate import validate_file

FIXTURES = Path(__file__).parent / 'data' / 'livebench'
RELEASE = '2026-01-08'


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding='utf-8')


@pytest.fixture
def converted():
    return adapter.convert_release(
        RELEASE,
        _read('table_2026_01_08.csv'),
        json.loads(_read('categories_2026_01_08.json')),
        adapter.parse_model_links(_read('modelLinks.js')),
        Registry(),
        '1234567890.0',
    )


def _scores(log):
    return {
        result.evaluation_name: result.score_details.score
        for result in log.evaluation_results
    }


def test_releases_come_from_the_site_release_picker():
    assert adapter.parse_releases(_read('App.js')) == [
        '2025-04-25',
        '2025-05-30',
        '2025-11-25',
        '2025-12-23',
        '2026-01-08',
        '2026-06-25',
    ]


def test_variants_inherit_their_base_entry():
    links = adapter.parse_model_links(_read('modelLinks.js'))

    variant = links['claude-opus-4-5-20251101-high-effort']
    assert variant['organization'] == 'Anthropic'
    assert variant['displayName'] == 'Claude 4.5 Opus High Effort'
    assert variant['url'] == 'https://www.anthropic.com/news/claude-opus-4-5'


def test_rows_become_task_category_and_overall_results(converted):
    log, developer, model = converted.records[0][0], *converted.records[0][1:]
    row = next(csv.DictReader(_read('table_2026_01_08.csv').splitlines()))
    categories = json.loads(_read('categories_2026_01_08.json'))
    scores = _scores(log)

    assert (
        log.evaluation_id
        == 'livebench/2026-01-08/qwen3-235b-a22b-instruct-2507'
    )
    # unresolved: the registry org id + the LiveBench name
    assert log.model_info.id == 'alibaba/qwen3-235b-a22b-instruct-2507'
    assert (developer, model) == ('alibaba', 'qwen3-235b-a22b-instruct-2507')
    assert log.model_info.additional_details['model_availability'] == (
        'open_weights'
    )
    # task scores are the table's cells
    assert scores['livebench/reasoning/zebra_puzzle'] == float(
        row['zebra_puzzle']
    )
    # a category is the mean of its tasks, overall the mean of categories
    category_means = [
        sum(float(row[task]) for task in tasks) / len(tasks)
        for tasks in categories.values()
    ]
    assert scores['livebench/reasoning'] == pytest.approx(
        sum(float(row[t]) for t in categories['Reasoning']) / 4
    )
    assert scores['livebench/overall'] == pytest.approx(
        sum(category_means) / len(category_means)
    )
    assert 'livebench/instruction_following' in scores
    levels = {
        result.evaluation_name: result.score_details.details[
            'aggregation_level'
        ]
        for result in log.evaluation_results
    }
    assert levels['livebench/overall'] == 'overall'
    assert levels['livebench/reasoning'] == 'category'
    assert levels['livebench/reasoning/zebra_puzzle'] == 'task'


def test_a_blank_cell_is_absent_not_zero(converted):
    log = converted.records[1][0]
    scores = _scores(log)

    assert log.model_info.additional_details['livebench_display_name'] == (
        'Claude 4.5 Opus High Effort'
    )
    assert 'livebench/language/typos' not in scores
    # the category averages only the tasks that have a score
    row = list(csv.DictReader(_read('table_2026_01_08.csv').splitlines()))[1]
    assert scores['livebench/language'] == pytest.approx(
        (float(row['connections']) + float(row['plot_unscrambling'])) / 2
    )


ZEPHYR = {
    'canonical_id': 'HuggingFaceH4/zephyr-7b-beta',
    'developer': 'Hugging Face',
    'org_id': 'huggingface',
    'open_weights': True,
    'strategy': 'exact',
    'confidence': 1.0,
    'review_status': 'reviewed',
}
QWEN = {
    'canonical_id': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'developer': 'Alibaba',
    'org_id': 'alibaba',
    'open_weights': True,
    'strategy': 'normalized',
    'confidence': 0.95,
    'review_status': 'reviewed',
}


def _convert(registry_models):
    return adapter.convert_release(
        RELEASE,
        _read('table_2026_01_08.csv'),
        json.loads(_read('categories_2026_01_08.json')),
        adapter.parse_model_links(_read('modelLinks.js')),
        Registry(),
        '1234567890.0',
        registry_models,
    )


def test_a_resolved_model_takes_its_registry_id_and_keeps_the_site_org():
    result = _convert({'qwen3-235b-a22b-instruct-2507': QWEN})

    log, developer, model = result.records[0]
    assert log.model_info.id == 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    assert (developer, model) == ('Qwen', 'Qwen3-235B-A22B-Instruct-2507')
    # the organization is still the one the site states
    assert log.model_info.developer == 'Alibaba'
    details = log.model_info.additional_details
    assert details['livebench_organization'] == 'Alibaba'
    assert details['model_registry_id'] == 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    # the record's identity stays the source's
    assert log.evaluation_id == (
        'livebench/2026-01-08/qwen3-235b-a22b-instruct-2507'
    )
    assert log.model_info.name == 'qwen3-235b-a22b-instruct-2507'


def test_a_model_the_site_leaves_unlabeled_takes_the_registry_organization(
    tmp_path,
):
    result = _convert({'zephyr-7b-beta': ZEPHYR})

    assert not result.failures
    log, developer, model = result.records[2]
    assert log.evaluation_id == 'livebench/2026-01-08/zephyr-7b-beta'
    assert log.model_info.name == 'zephyr-7b-beta'
    assert log.model_info.id == 'HuggingFaceH4/zephyr-7b-beta'
    assert log.model_info.developer == 'Hugging Face'
    details = log.model_info.additional_details
    assert details['model_registry_review_status'] == 'reviewed'
    assert details['developer_registry_id'] == 'huggingface'
    assert details['model_availability'] == 'open_weights'
    assert 'livebench_organization' not in details
    (path,) = adapter.export(
        [result.records[2]], tmp_path / 'data' / adapter.COLLECTION
    )
    report = validate_file(path)
    assert report.valid, report.errors


def test_a_model_neither_the_site_nor_the_registry_places_is_a_failure():
    result = _convert({})

    assert result.total_records == 3
    assert len(result.records) == 2
    (failure,) = result.failures
    assert 'zephyr-7b-beta' in failure.reason
    assert 'no organization' in failure.reason
    assert failure.source_record['model'] == 'zephyr-7b-beta'


def _pinned(canonical_id):
    return {**QWEN, 'canonical_id': canonical_id}


def test_a_namespace_takes_the_spelling_most_pinned_ids_use():
    casing = adapter.namespace_casing(
        {
            'a': _pinned('Qwen/Qwen2-7B-Instruct'),
            'b': _pinned('Qwen/QwQ-32B'),
            'c': _pinned('qwen/qwen3.6-plus'),
            'd': _pinned('Anthropic/claude-3-opus-20240229'),
            'e': _pinned('anthropic/claude-sonnet-3.7'),
        }
    )

    assert casing == {'qwen': 'Qwen', 'anthropic': 'anthropic'}


def test_one_publisher_files_under_one_directory():
    result = _convert(
        {
            'qwen3-235b-a22b-instruct-2507': _pinned(
                'qwen/qwen3-235b-a22b-instruct-2507'
            ),
            'qwen2-7b-instruct': _pinned('Qwen/Qwen2-7B-Instruct'),
            'qwq-32b': _pinned('Qwen/QwQ-32B'),
            'claude-3-opus-20240229': _pinned(
                'Anthropic/claude-3-opus-20240229'
            ),
        }
    )

    qwen, claude = result.records
    assert qwen[0].model_info.id == 'Qwen/qwen3-235b-a22b-instruct-2507'
    assert qwen[1:] == ('Qwen', 'qwen3-235b-a22b-instruct-2507')
    # the registry's own spelling is kept where the record names it
    details = qwen[0].model_info.additional_details
    assert details['model_registry_id'] == 'qwen/qwen3-235b-a22b-instruct-2507'
    # an id built from the site's organization follows the same spelling
    assert claude[0].model_info.id == (
        'Anthropic/claude-opus-4-5-20251101-high-effort'
    )
    assert claude[1] == 'Anthropic'


def test_the_pinned_map_ships_with_the_adapter():
    pinned = adapter.load_registry_map()

    assert pinned
    for name, entry in pinned.items():
        assert '/' in entry['canonical_id'], name
        assert entry['developer'] and entry['org_id'], name


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_refresh_pins_only_existing_namespaced_canonicals(monkeypatch):
    resolutions = [
        {'raw_value': 'kept', 'canonical_id': 'org/kept', 'strategy': 'exact'},
        # exact mode must never create; a draft it claims to have made is
        # not a resolution
        {'raw_value': 'created', 'canonical_id': 'org/c', 'created_new': True},
        {'raw_value': 'missing', 'canonical_id': None},
        # a flat id names no datastore directory
        {'raw_value': 'flat', 'canonical_id': 'flat'},
    ]
    monkeypatch.setattr(
        adapter.requests, 'post', lambda *a, **k: FakeResponse(resolutions)
    )
    monkeypatch.setattr(
        adapter.requests,
        'get',
        lambda *a, **k: FakeResponse(
            {'developer': 'Org', 'org_id': 'org', 'open_weights': False}
        ),
    )

    pinned = adapter.refresh_registry_map(
        ['kept', 'created', 'missing', 'flat']
    )

    assert list(pinned['models']) == ['kept']
    assert pinned['models']['kept']['developer'] == 'Org'
    assert pinned['_meta']['n_queried'] == 4
    assert pinned['_meta']['n_resolved'] == 1


def test_release_is_preserved_without_inventing_evaluation_dates(
    converted, tmp_path
):
    paths = adapter.export(
        converted.records, tmp_path / 'data' / adapter.COLLECTION
    )

    for path in paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        # A model can be evaluated on a benchmark released before it existed.
        # The source names the benchmark version, not the evaluation date.
        assert 'evaluation_timestamp' not in payload
        assert payload['retrieved_timestamp'] == '1234567890.0'
        assert payload['eval_library']['version'] == '2026-01-08'
        assert payload['source_metadata']['additional_details']['release'] == (
            '2026-01-08'
        )
        for result in payload['evaluation_results']:
            assert 'evaluation_timestamp' not in result
            assert result['source_data']['additional_details']['release'] == (
                '2026-01-08'
            )


def test_records_validate_at_their_datastore_path(converted, tmp_path):
    paths = adapter.export(
        converted.records, tmp_path / 'data' / adapter.COLLECTION
    )

    assert len(paths) == 2
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors
