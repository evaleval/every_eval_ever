import json
from pathlib import Path

import pytest

from every_eval_ever import check_duplicate_entries as check_module

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
SAMPLE_FILES = [
    Path(__file__).resolve().parent
    / 'data'
    / '98ea850e-7019-4728-a558-8b1819ec47c2.json',
    Path(__file__).resolve().parent
    / 'data'
    / '98ea850e-7019-4728-a558-8b1819ec47c2.json',
]


@pytest.fixture(scope='module')
def sample_payloads():
    missing = [path for path in SAMPLE_FILES if not path.exists()]
    if missing:
        pytest.skip(f'Sample data file missing: {missing[0]}')
    return [
        json.loads(path.read_text(encoding='utf-8')) for path in SAMPLE_FILES
    ]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')


def clone_payload(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def simulate_rescrape(payload: dict) -> dict:
    cloned = clone_payload(payload)
    cloned['evaluation_id'] = 'simulated-duplicate'
    cloned['retrieved_timestamp'] = '1234567890.0'
    if isinstance(cloned.get('evaluation_results'), list):
        cloned['evaluation_results'] = list(
            reversed(cloned['evaluation_results'])
        )
    return cloned


def test_normalized_hash_ignores_keys_and_list_order(sample_payloads):
    payload_a = clone_payload(sample_payloads[0])
    payload_b = simulate_rescrape(sample_payloads[0])

    assert check_module.normalized_hash(
        payload_a
    ) == check_module.normalized_hash(payload_b)


def test_normalized_hash_detects_real_changes(sample_payloads):
    payload_a = clone_payload(sample_payloads[0])
    payload_c = clone_payload(sample_payloads[0])
    payload_c['evaluation_id'] = 'eval-c'
    payload_c['retrieved_timestamp'] = '2024-01-03'
    if (
        isinstance(payload_c.get('evaluation_results'), list)
        and payload_c['evaluation_results']
    ):
        payload_c['evaluation_results'][0]['score_details']['score'] = (
            payload_c['evaluation_results'][0]['score_details']['score'] + 0.001
        )

    assert check_module.normalized_hash(
        payload_a
    ) != check_module.normalized_hash(payload_c)


def test_expand_paths_returns_json_files(tmp_path):
    top = tmp_path / 'top.json'
    nested_dir = tmp_path / 'nested'
    nested_dir.mkdir()
    nested = nested_dir / 'nested.json'
    ignored = nested_dir / 'note.txt'
    top.write_text('{}', encoding='utf-8')
    nested.write_text('{}', encoding='utf-8')
    ignored.write_text('nope', encoding='utf-8')

    expanded = check_module.expand_paths([str(tmp_path)])
    assert set(expanded) == {str(top), str(nested)}

    expanded_file = check_module.expand_paths([str(top)])
    assert expanded_file == [str(top)]

    missing = tmp_path / 'missing.json'
    with pytest.raises(Exception, match='Could not find file or directory'):
        check_module.expand_paths([str(missing)])
    
    with pytest.raises(Exception, match='Could not find file or directory'):
        check_module.expand_paths([str(ignored)])


def test_main_raises_on_invalid_json(tmp_path):
    bad = tmp_path / 'bad.json'
    bad.write_text('{not valid json', encoding='utf-8')

    with pytest.raises(json.JSONDecodeError):
        check_module.main([str(bad)])


def test_main_reports_duplicates(sample_payloads, tmp_path, capsys):
    payload = sample_payloads[0]
    file_a = tmp_path / 'a.json'
    file_b = tmp_path / 'b.json'
    write_json(file_a, payload)

    write_json(file_b, simulate_rescrape(payload))

    assert check_module.main([str(file_a), str(file_b)]) == 1
    captured = capsys.readouterr().out
    assert 'Found duplicate entries (ignoring keys: `evaluation_id`, `retrieved_timestamp`)' in captured


def test_main_reports_no_duplicates(sample_payloads, tmp_path, capsys):
    payload = sample_payloads[0]
    file_a = tmp_path / 'a.json'
    file_c = tmp_path / 'c.json'
    write_json(file_a, payload)


    payload_a = clone_payload(payload)
    payload_a['evaluation_id'] = 'eval-c'
    payload_a['retrieved_timestamp'] = '2024-01-03'
    if (
        isinstance(payload_a.get('evaluation_results'), list)
        and payload_a['evaluation_results']
    ):
        payload_a['evaluation_results'][0]['score_details']['score'] = (
            payload_a['evaluation_results'][0]['score_details']['score'] + 0.001
        )
    
    write_json(file_c, payload_a)

    assert check_module.main([str(file_a), str(file_c)]) == 0
    captured = capsys.readouterr().out
    assert 'No duplicates found.' in captured